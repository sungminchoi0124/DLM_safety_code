"""semi_ar.py: the guard-free copy of the original generate() decodes block by block (CPU, fake model)."""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import run_generate as rg
import semi_ar
from conftest import DG_ROOT, MASK_ID

CASES = [(32, 32, 8), (16, 32, 8), (8, 32, 8), (4, 32, 8), (64, 64, 16), (8, 64, 16)]  # steps, gen, block


def _original():
    rg.import_original(DG_ROOT)
    return sys.modules["utility.generate_function_llada"]


def reference_semi_ar(model, prompt, steps, gen_length, block_length):
    """LLaDA / Fast-dLLM baseline decoding (Fast-dLLM v1/llada/generate.py `generate`, temperature 0,
    low_confidence, no threshold), written out so the test does not need the Fast-dLLM checkout."""
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long)
    x[:, : prompt.shape[1]] = prompt
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    for b in range(num_blocks):
        start, end = prompt.shape[1] + b * block_length, prompt.shape[1] + (b + 1) * block_length
        n_mask = int((x[:, start:end] == MASK_ID).sum())
        plan = [n_mask // steps + (1 if i < n_mask % steps else 0) for i in range(steps)]
        for i in range(steps):
            mask_index = x == MASK_ID
            logits = model(x).logits
            mask_index[:, end:] = False
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, conf, torch.tensor(-np.inf, dtype=conf.dtype))
            if plan[i] > 0:
                idx = torch.topk(conf[0], k=plan[i]).indices
                x[0, idx] = x0[0, idx]
    return x


def _prompt(seed):
    return torch.randint(2, 60, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))


def _gen(mod, model, tok, prompt, steps, gen, block, **kw):
    args = dict(steps=steps, gen_length=gen, block_length=block, temperature=0.0, mask_id=MASK_ID,
                remasking="low_confidence", sp_mode="off", fill_all_masks=False)
    args.update(kw)
    return mod.generate(model, tok, prompt, **args)


@pytest.mark.parametrize("steps,gen,block", CASES)
def test_vanilla_equals_the_fast_dllm_baseline_algorithm(fake_model, fake_tokenizer, steps, gen, block):
    sa = semi_ar.load(DG_ROOT)
    for seed in range(6):
        prompt = _prompt(seed)
        expect = reference_semi_ar(fake_model, prompt, steps, gen, block)
        assert torch.equal(_gen(sa, fake_model, fake_tokenizer, prompt, steps, gen, block), expect)


def test_vanilla_equals_the_real_fast_dllm_generate(fake_model, fake_tokenizer):
    root = os.environ.get("FAST_DLLM_ROOT")
    src = Path(root) / "v1" / "llada" / "generate.py" if root else None
    if not src or not src.exists():
        pytest.skip("set FAST_DLLM_ROOT to a Fast-dLLM checkout (a9b81e4) to run against the real file")
    sys.modules.setdefault("model", types.ModuleType("model"))
    stub = types.ModuleType("model.modeling_llada")
    stub.LLaDAModelLM = object
    sys.modules.setdefault("model.modeling_llada", stub)
    fd = types.ModuleType("fast_dllm_generate")
    exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), fd.__dict__)
    sa = semi_ar.load(DG_ROOT)
    for steps, gen, block in CASES:
        for seed in range(6):
            prompt = _prompt(seed)
            expect, nfe = fd.generate(fake_model, prompt, steps=steps, gen_length=gen, block_length=block,
                                      temperature=0.0, mask_id=MASK_ID)
            assert nfe == steps
            assert torch.equal(_gen(sa, fake_model, fake_tokenizer, prompt, steps, gen, block), expect)


def test_original_does_not_decode_block_by_block(fake_model, fake_tokenizer):
    """The reason semi_ar.py exists: the original fills the whole tail in the first block's steps."""
    orig, sa = _original(), semi_ar.load(DG_ROOT)
    prompt = _prompt(0)

    def masks_before_each_forward(mod):
        seen = []
        handle = fake_model.register_forward_pre_hook(lambda _m, inp: seen.append(int((inp[0] == MASK_ID).sum())))
        try:
            _gen(mod, fake_model, fake_tokenizer, prompt, 32, 32, 8)
        finally:
            handle.remove()
        return seen

    o, s = masks_before_each_forward(orig), masks_before_each_forward(sa)
    assert len(o) == len(s) == 32
    assert o[8] == 0 and o[7] > 0          # everything decoded after 8 (= 32 / 4 blocks) forwards
    assert s == list(range(32, 0, -1))     # one token per step, 32 steps
    assert not torch.equal(_gen(orig, fake_model, fake_tokenizer, prompt, 32, 32, 8),
                           _gen(sa, fake_model, fake_tokenizer, prompt, 32, 32, 8))


@pytest.mark.parametrize("kw", [
    dict(),
    dict(remasking="adaptive_step", alpha0=0.3),
    dict(remasking="adaptive_step", alpha0=0.3, sp_mode="hidden", sp_threshold=0.0, temperature=0.5),
])
def test_one_block_is_the_original(fake_model, fake_tokenizer, kw):
    """With block_length == gen_length the guard is a no-op, so the patched copy must not change anything
    (including SAR, audit and repair)."""
    orig, sa = _original(), semi_ar.load(DG_ROOT)
    prompt = _prompt(3)
    if kw.get("sp_mode") == "hidden":
        kw = dict(kw, baseline_hidden=torch.ones(8))
    outs = []
    for mod in (orig, sa):
        torch.manual_seed(7)
        outs.append(_gen(mod, fake_model, fake_tokenizer, prompt, 8, 16, 16, **kw))
    assert torch.equal(outs[0], outs[1])
    assert not (outs[1] == MASK_ID).any()


def test_repair_is_limited_to_the_first_block(fake_model, fake_tokenizer):
    sa = semi_ar.load(DG_ROOT)
    prompt = _prompt(2)
    P = prompt.shape[1]
    common = dict(remasking="adaptive_step", alpha0=0.3, baseline_hidden=torch.ones(8), sp_mode="hidden")
    torch.manual_seed(11)
    plain = _gen(sa, fake_model, fake_tokenizer, prompt, 16, 32, 8, sp_threshold=10.0, **common)
    n = [0]
    handle = fake_model.register_forward_pre_hook(lambda _m, _i: n.__setitem__(0, n[0] + 1))
    try:
        torch.manual_seed(11)
        fixed = _gen(sa, fake_model, fake_tokenizer, prompt, 16, 32, 8, sp_threshold=0.0, refinement_steps=3,
                     **common)
    finally:
        handle.remove()
    assert n[0] == 16 + 3                                   # repair once, on block 1 only
    assert torch.equal(fixed[:, :P], plain[:, :P])          # prompt untouched
    assert not torch.equal(fixed[:, P:P + 8], plain[:, P:P + 8])
    assert not (fixed == MASK_ID).any()


def test_patch_fails_loudly_when_the_original_changes(tmp_path):
    dst = tmp_path / "utility"
    dst.mkdir()
    text = (DG_ROOT / semi_ar.SOURCE).read_text(encoding="utf-8").replace(semi_ar.GUARD.strip(), "if True:")
    (dst / "generate_function_llada.py").write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        semi_ar.load(tmp_path)


def test_submodule_file_is_not_touched():
    before = (DG_ROOT / semi_ar.SOURCE).read_bytes()
    mod = semi_ar.load(DG_ROOT)
    assert (DG_ROOT / semi_ar.SOURCE).read_bytes() == before
    assert mod is not _original()

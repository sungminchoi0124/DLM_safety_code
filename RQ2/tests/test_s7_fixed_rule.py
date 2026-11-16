"""S7: fixed_rule="fastdllm" (RQ2 ASR column 0) == C's Without 0 (rq2-asr-c-share 9d39fa2, RQ2/decoding.py
generate_fast with threshold None), token for token; the default fixed_rule="diffuguard" (GSM8K) is unchanged.

C's loop per block: masks restricted to the block (block 0 = prompt masks + first block_length positions),
plan = get_num_transfer_tokens over the block's masks, steps_per_block full forwards, each through Fast-dLLM's
get_transfer_index(logits, T, "low_confidence", masks, x, plan[:, i]): Gumbel argmax, float64 softmax confidence,
no extra random draw, first k of a descending sort. On GPU the same match was checked on dija-000..002.
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import MASK_ID
from src import combined
from test_s6_dija import BLOCK, GEN, _dija_prompt, _run


def _transfer_written_out(logits, temperature, mask_index, x, k):
    """Fast-dLLM get_transfer_index, low_confidence, top-k branch (written out)."""
    x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)
    _, idx = torch.sort(confidence, dim=1, descending=True)
    sel = torch.zeros_like(mask_index)
    sel[0, idx[0, :int(k)]] = True
    return x0, sel & mask_index


def c_without_fixed(model, prompt, steps, transfer=None, plan_fn=None):
    """C's generate_fast, fixed branch (threshold None), batch 1."""
    transfer = transfer or (lambda lg, T, m, x, k: _transfer_written_out(lg, T, m, x, k))
    plan_fn = plan_fn or combined.get_num_transfer_tokens
    p = prompt.shape[1]
    x = torch.full((1, p + GEN), MASK_ID, dtype=torch.long)
    x[:, :p] = prompt
    spb = steps // (GEN // BLOCK)
    nfe = 0
    for b in range(GEN // BLOCK):
        tail = p + b * BLOCK
        start, end = (0 if b == 0 else tail), tail + BLOCK
        plan = plan_fn(x[:, start:end] == MASK_ID, spb)
        for i in range(spb):
            masks = x == MASK_ID
            masks[:, :start] = False
            masks[:, end:] = False
            logits = model(x).logits
            nfe += 1
            proposed, sel = transfer(logits, 0.5, masks, x, plan[:, i])
            x[sel] = proposed[sel]
    return x, nfe


def _ours(model, prompt, seed, rule="fastdllm", temperature=0.5, **kw):
    return _run(model, prompt, "fixed", seed=seed, temperature=temperature, fixed_rule=rule, **kw)


@pytest.mark.parametrize("n_masks", [(0, 0), (2, 3), (5, 1)])
def test_fastdllm_rule_equals_c_without0(fake_model, n_masks):
    for seed in range(4):
        prompt = _dija_prompt(seed, n_masks)
        torch.manual_seed(seed)
        ref, nfe = c_without_fixed(fake_model, prompt, GEN)
        out, st = _ours(fake_model, prompt, seed)
        assert torch.equal(out, ref), (n_masks, seed)
        assert st.nfe_gen == nfe == GEN and not (out == MASK_ID).any()


@pytest.mark.parametrize("n_masks", [(0, 0), (2, 3)])
def test_fastdllm_rule_equals_c_with_real_fast_dllm_kernel(fast_dllm, fake_model, n_masks):
    """Same loop, but the transfer step and the plan are the real Fast-dLLM functions (as C calls them)."""
    real = lambda lg, T, m, x, k: fast_dllm.get_transfer_index(lg, T, "low_confidence", m, x, k, threshold=None)
    for seed in range(3):
        prompt = _dija_prompt(seed, n_masks)
        torch.manual_seed(seed)
        ref, _ = c_without_fixed(fake_model, prompt, GEN, transfer=real, plan_fn=fast_dllm.get_num_transfer_tokens)
        out, _ = _ours(fake_model, prompt, seed)
        assert torch.equal(out, ref), (n_masks, seed)


def test_fastdllm_rule_equals_real_fast_dllm_generate_without_prompt_masks(fast_dllm, fake_model):
    for seed in range(3):
        prompt = _dija_prompt(seed, (0, 0))
        torch.manual_seed(seed)
        ref, _ = fast_dllm.generate(fake_model, prompt, steps=GEN, gen_length=GEN, block_length=BLOCK,
                                    temperature=0.5, mask_id=MASK_ID, threshold=None)
        out, _ = _ours(fake_model, prompt, seed)
        assert torch.equal(out, ref), seed


def test_default_rule_is_diffuguard_and_differs_at_temperature(fake_model):
    """GSM8K default unchanged (extra R draw each step): at T > 0 it is a different RNG stream than C's."""
    prompt = _dija_prompt(1, (2, 3))
    default, _ = _run(fake_model, prompt, "fixed", seed=1, temperature=0.5)
    explicit, _ = _ours(fake_model, prompt, 1, rule="diffuguard")
    fd, _ = _ours(fake_model, prompt, 1)
    assert torch.equal(default, explicit)
    assert not torch.equal(default, fd)


def test_temperature_zero_rules_agree_on_fp32(fake_model):
    """At T = 0 with fp32 logits the only differences (R draw, softmax dtype) do not change the choice."""
    for seed in range(3):
        prompt = _dija_prompt(seed, (2, 3))
        a, _ = _ours(fake_model, prompt, seed, rule="diffuguard", temperature=0.0)
        b, _ = _ours(fake_model, prompt, seed, temperature=0.0)
        assert torch.equal(a, b), seed


def test_fastdllm_rule_with_defense_runs(fake_model, fake_tokenizer):
    prompt = _dija_prompt(2, (2, 3))
    out, st = _run(fake_model, prompt, "fixed", tok=fake_tokenizer, defense=True, seed=2, temperature=0.5,
                   fixed_rule="fastdllm")
    assert not (out == MASK_ID).any()
    assert st.nfe_gen == GEN and st.audit_score is not None
    P = prompt.shape[1]
    keep = prompt[0] != MASK_ID
    assert torch.equal(out[0, :P][keep], prompt[0][keep])       # fixed prompt tokens unchanged


def test_guards(fake_model):
    prompt = _dija_prompt(0, (2, 3))
    with pytest.raises(ValueError):
        _run(fake_model, prompt, "fixed", fixed_rule="bogus")
    with pytest.raises(ValueError):
        _run(fake_model, prompt, "threshold", threshold=0.5, fixed_rule="fastdllm")

import sys
import types

import numpy as np
import pytest
import torch

from conftest import MASK
from commit_time.capture import FINAL, TraceRecorder
from commit_time.decode import decode

gfl = pytest.importorskip("utility.generate_function_llada")


def _orig(model, tok, prompt, steps, gen, temperature, **kw):
    am = torch.ones_like(prompt)
    return gfl.generate(model, tok, prompt, steps=steps, gen_length=gen, block_length=gen, temperature=temperature,
                        cfg_scale=0.0, remasking=kw.pop("remasking", "low_confidence"), mask_id=MASK,
                        fill_all_masks=True, attention_mask=am, protected_index=torch.zeros_like(prompt).bool(), **kw)


@pytest.mark.parametrize("steps,temperature", [(21, 0.5), (7, 0.5), (21, 0.0)])
def test_quota_equals_original_vanilla(model, tok, prompt, steps, temperature):
    torch.manual_seed(123)
    a = _orig(model, tok, prompt, steps, 16, temperature)
    torch.manual_seed(123)
    b = decode(model, prompt, torch.ones_like(prompt), gen_length=16, steps=steps, temperature=temperature,
               mask_id=MASK, rule="quota").x
    assert torch.equal(a, b)
    assert not (b == MASK).any()


def test_recorder_on_quota(model, prompt):
    rec = TraceRecorder(model, mask_id=MASK, prompt_len=prompt.shape[1], layers=(1, 2, FINAL), lead_k=4)
    torch.manual_seed(0)
    with rec:
        res = decode(model, prompt, None, gen_length=16, steps=21, temperature=0.5, mask_id=MASK, rule="quota")
        rec.finish(res.x)
    a = rec.arrays()
    n = 16 + 5
    assert a["hidden"].shape == (21, 3, 32)
    assert a["n_tracked"] == n and a["n_masks"][0] == n and a["n_masks"][-1] == 0
    assert a["n_commit"].sum() == n and a["n_remask"].sum() == 0
    assert a["lead_unmasked"][-1] == 4 and a["n_response_tracked"] == 16
    # 'final' == hidden_states[-1] pooled over tracked positions at forward 0
    x0 = torch.cat([prompt, torch.full((1, 16), MASK)], 1)
    hs = model(x0, output_hidden_states=True).hidden_states[-1]
    tracked = (x0[0] == MASK).nonzero().squeeze(1)
    np.testing.assert_allclose(a["hidden"][0, 2], hs[0, tracked].mean(0).detach().numpy(), atol=2e-3)


def test_recorder_sees_diffuguard_repair(model, tok, prompt):
    rec = TraceRecorder(model, mask_id=MASK, prompt_len=prompt.shape[1], layers=(FINAL,))
    torch.manual_seed(0)
    with rec:
        x = _orig(model, tok, prompt, 21, 16, 0.5, remasking="adaptive_step", sp_mode="hidden", sp_threshold=0.0,
                  refinement_steps=4, remask_ratio=0.9, baseline_hidden=torch.randn(32))
        rec.finish(x)
    a = rec.arrays()
    assert a["hidden"].shape[0] == 21 + 4
    assert a["n_remask"][20] > 0 and a["n_remask"][:20].sum() == 0
    assert a["n_masks"][-1] == 0


def test_threshold_rule_extremes(model, prompt):
    torch.manual_seed(0)
    r = decode(model, prompt, None, gen_length=16, steps=21, temperature=0.5, mask_id=MASK, rule="threshold",
               threshold=0.0)
    assert r.nfe == 1 and not (r.x == MASK).any()
    r = decode(model, prompt, None, gen_length=16, steps=21, temperature=0.5, mask_id=MASK, rule="threshold",
               threshold=1.01)
    assert r.nfe == 21 and not (r.x == MASK).any()


def test_resume_is_deterministic_and_keeps_prefix(model, prompt):
    torch.manual_seed(0)
    res = decode(model, prompt, None, gen_length=16, steps=21, temperature=0.5, mask_id=MASK, rule="quota",
                 snapshot_at=range(30))
    snap = res.snapshots[8]
    r1 = decode(model, None, None, gen_length=16, steps=21, temperature=0.5, mask_id=MASK, rule="quota",
                resume=snap, seed=5)
    r2 = decode(model, None, None, gen_length=16, steps=21, temperature=0.5, mask_id=MASK, rule="quota",
                resume=snap, seed=5)
    assert torch.equal(r1.x, r2.x) and r1.nfe == 21 - 8
    done = snap.x != MASK
    assert torch.equal(r1.x[done], snap.x[done])


def test_run_trace_install_captures_original(model, tok, prompt, tmp_path):
    """install() wraps a jbl-like module: original generate_response is traced and its final ids captured."""
    from commit_time.run_trace import install
    jbl = types.ModuleType("fake_jbl")
    jbl.generate_llada = gfl.generate_llada
    jbl.get_tokenized_input = lambda text, tokenizer, device: (prompt.to(device), torch.ones_like(prompt))
    exec(
        "def generate_response(vanilla_prompt, prompt, tokenizer, model, args, baseline_hidden=None, system_prompt=None):\n"
        "    ids, am = get_tokenized_input(prompt, tokenizer, model.device)\n"
        "    out = generate_llada(input_ids=ids, attention_mask=am, model=model, steps=args.steps, gen_length=16,\n"
        "        block_length=16, temperature=0.5, mask_id=args.mask_id, tokenizer=tokenizer, fill_all_masks=True)\n"
        "    return tokenizer.batch_decode(out)[0]\n", jbl.__dict__)
    opts = types.SimpleNamespace(decoder="original", tau=None, layers="1,final", lead_k=4)
    install(jbl, opts, tmp_path)
    args = types.SimpleNamespace(steps=21, mask_id=MASK)
    for _ in range(2):
        jbl.generate_response("v", "p", tok, model, args)
    assert (tmp_path / "hidden" / "item000.npz").exists() and (tmp_path / "hidden" / "item001.npz").exists()
    lines = (tmp_path / "trace_meta.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and '"n_forwards": 21' in lines[0]

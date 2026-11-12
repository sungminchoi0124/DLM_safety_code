"""S2: threshold mode with the defense off == Fast-dLLM generate(threshold=tau) (no cache), token for token.

Two references: a written-out copy of Fast-dLLM's algorithm (always runs) and the real v1/llada/generate.py of
the pinned checkout a9b81e4 (runs when FAST_DLLM_ROOT is set, as in DiffuGuard/tests/test_semi_ar.py).
"""

import types

import pytest
import torch
import torch.nn.functional as F

from conftest import MASK_ID
from src import combined

CASES = [(32, 8), (32, 32), (64, 16), (64, 32)]       # gen_length, block_length
TAUS = [0.9, 0.5, 0.2, 0.1, 0.05]                     # FakeModel confidences are low; small tau -> parallel steps


def _prompt(seed):
    return torch.randint(2, 60, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))


def reference_threshold(model, prompt, gen_length, block_length, temperature, tau):
    """Fast-dLLM v1/llada/generate.py generate() + get_transfer_index() with threshold, written out."""
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long)
    x[:, : prompt.shape[1]] = prompt
    nfe = 0
    for b in range(gen_length // block_length):
        end = prompt.shape[1] + (b + 1) * block_length
        start = end - block_length
        while True:
            nfe += 1
            mask_index = x == MASK_ID
            logits = model(x).logits
            mask_index[:, end:] = False
            x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, torch.tensor(torch.finfo(x0_p.dtype).min, dtype=x0_p.dtype))
            transfer = mask_index & (conf >= tau)
            transfer[0, torch.argmax(conf[0])] = True
            transfer &= mask_index
            x[transfer] = x0[transfer]
            if not (x[:, start:end] == MASK_ID).any():
                break
    return x, nfe


def _combined(model, prompt, gen, block, temperature, tau, seed, **kw):
    torch.manual_seed(seed)
    return combined.generate(model, prompt, gen_length=gen, block_length=block, temperature=temperature,
                             mask_id=MASK_ID, mode="threshold", threshold=tau, defense=False, **kw)


@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_equals_written_out_fast_dllm(fake_model, gen, block, temperature):
    for tau in TAUS:
        for seed in range(3):
            prompt = _prompt(seed)
            torch.manual_seed(seed)
            ref, ref_nfe = reference_threshold(fake_model, prompt, gen, block, temperature, tau)
            out, st = _combined(fake_model, prompt, gen, block, temperature, tau, seed)
            assert torch.equal(out, ref), (tau, seed)
            assert st.nfe_gen == ref_nfe and st.nfe_repair == 0
            assert not (out == MASK_ID).any()


@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_equals_real_fast_dllm_generate(fast_dllm, fake_model, gen, block, temperature):
    for tau in TAUS:
        for seed in range(3):
            prompt = _prompt(seed)
            torch.manual_seed(seed)
            ref, ref_nfe = fast_dllm.generate(fake_model, prompt, steps=gen, gen_length=gen, block_length=block,
                                              temperature=temperature, mask_id=MASK_ID, threshold=tau)
            out, st = _combined(fake_model, prompt, gen, block, temperature, tau, seed)
            assert torch.equal(out, ref), (tau, seed)
            assert st.nfe_gen == ref_nfe


class _ConstBf16(torch.nn.Module):
    """Every position gets the same bf16 logits: token 5 has logit `a`, the other 63 tokens 0 (LLaDA-8B runs in
    bfloat16). All masked positions then share one p(x0), so a step commits the whole block or a single token."""

    def __init__(self, a, vocab=64):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        row = torch.zeros(vocab)
        row[5] = a
        self.row = row.to(torch.bfloat16)

    @property
    def device(self):
        return self.w.device

    def forward(self, x, **kw):
        return types.SimpleNamespace(logits=self.row.expand(x.shape[0], x.shape[1], -1).clone(), hidden_states=None)


def _straddling_logit():
    """A logit whose p(token 5) differs between a float64 softmax and a bf16 softmax of the same bf16 logits."""
    for a in torch.arange(4.0, 9.0, 0.03125):
        row = torch.zeros(64)
        row[5] = a
        row = row.to(torch.bfloat16)
        p64 = float(F.softmax(row.to(torch.float64), -1)[5])
        pb = float(F.softmax(row, -1)[5])
        if abs(p64 - pb) > 1e-4:
            return float(a), p64, pb
    raise AssertionError("no straddling logit found")


def test_bf16_logits_need_float64_confidence():
    """tau sits between the float64 and the bf16 confidence: Fast-dLLM (float64) and the default threshold mode
    agree; a bf16 softmax makes the opposite call and decodes the block in a different number of steps."""
    a, p64, pb = _straddling_logit()
    model = _ConstBf16(a)
    tau = (p64 + pb) / 2
    prompt = _prompt(0)
    ref, ref_nfe = reference_threshold(model, prompt, 32, 8, 0.0, tau)
    out, st = _combined(model, prompt, 32, 8, 0.0, tau, 0)
    coarse, st_coarse = _combined(model, prompt, 32, 8, 0.0, tau, 0, conf_precision="model")
    assert torch.equal(out, ref) and st.nfe_gen == ref_nfe
    # float64 p >= tau -> whole block per step (4 forwards); otherwise one token per step (32 forwards)
    assert ref_nfe == (4 if p64 >= tau else 32)
    assert st_coarse.nfe_gen == (4 if pb >= tau else 32) != ref_nfe


def test_threshold_path_is_really_parallel(fake_model):
    """Guard against a vacuous test: at small tau some steps must commit several tokens (nfe < gen_length),
    at tau > 1 every step commits exactly one (nfe == gen_length)."""
    prompt = _prompt(0)
    _, low = _combined(fake_model, prompt, 64, 16, 0.0, 0.05, 0)
    _, high = _combined(fake_model, prompt, 64, 16, 0.0, 1.01, 0)
    assert low.nfe_gen < 64
    assert high.nfe_gen == 64


def test_fixed_mode_unchanged_by_s2(fake_model):
    """Fixed mode still draws R every step (DiffuGuard stream) and uses the logits dtype for p(x0)."""
    prompt = _prompt(1)
    torch.manual_seed(0)
    a, _ = combined.generate(fake_model, prompt, gen_length=32, block_length=8, steps=16, temperature=0.5,
                             mask_id=MASK_ID)
    torch.manual_seed(0)
    b, _ = combined.generate(fake_model, prompt, gen_length=32, block_length=8, steps=16, temperature=0.5,
                             mask_id=MASK_ID, mode="fixed", conf_precision="model")
    assert torch.equal(a, b)


def test_guards(fake_model):
    prompt = _prompt(0)
    with pytest.raises(ValueError):
        combined.generate(fake_model, prompt, gen_length=32, block_length=8, mode="threshold")       # no tau
    with pytest.raises(ValueError):
        combined.generate(fake_model, prompt, gen_length=32, block_length=8)                          # no steps
    with pytest.raises(ValueError):
        combined.generate(fake_model, prompt, gen_length=32, block_length=8, mode="threshold", threshold=0.9,
                          alpha_rule="original")                          # step-based alpha needs fixed steps

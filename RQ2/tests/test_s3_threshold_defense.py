"""S3: threshold mode + defense (the RQ2 combination rule), no cache.

Rule under test (agreed RQ2 design):
  * k = #{masked positions of the current block with p(x0) >= tau}, at least 1   -> acceleration decides HOW MANY
  * the k committed positions are the top-k by the SAR score (1-alpha)*p + alpha*R -> defense decides WHICH
  * alpha = alpha0 * (1 - committed in this block / masks in this block at block start)   (progress rule)
  * block-0 audit as in DiffuGuard; repair = fixed refinement_steps forwards (8), random refill order, no tau

There is no original to compare with (the combination is new), so the reference below is a separate written-out
implementation of the rule, plus property tests that each part of the rule holds on its own.
"""

import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import MASK_ID
from src import combined

CASES = [(32, 8), (32, 32), (64, 16)]           # gen_length, block_length
TAUS = [0.9, 0.2, 0.1, 0.05]
BASE_HIDDEN = torch.ones(8)


def _prompt(seed):
    return torch.randint(2, 60, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))


def reference_s3(model, prompt, gen, block, temperature, tau, alpha0, lam, cue_ids, base_hidden=BASE_HIDDEN,
                 refine=8, ratio=0.9):
    P = prompt.shape[1]
    x = torch.full((1, P + gen), MASK_ID, dtype=torch.long)
    x[:, :P] = prompt
    nfe = nfe_r = 0
    alphas = []
    for b in range(gen // block):
        start, end = P + b * block, P + (b + 1) * block
        n0 = int((x[:, start:end] == MASK_ID).sum())
        done, step, hmean = 0, 0, None
        while (x[:, start:end] == MASK_ID).any():
            want_h = b == 0 and step == 0 and base_hidden is not None
            out = model(x, output_hidden_states=want_h, return_dict=True)
            nfe += 1
            logits = out.logits
            if want_h:
                hmean = out.hidden_states[-1][:, start:end].mean(dim=1).squeeze(0)
            x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
            p = torch.gather(F.softmax(logits.to(torch.float64), -1), -1, x0.unsqueeze(-1)).squeeze(-1)
            R = torch.rand(x0.shape)
            alpha = min(max(alpha0 * (1 - done / n0), 0.0), 1.0)
            if b == 0:
                alphas.append(alpha)
            score = (1 - alpha) * p + alpha * R
            masked = torch.zeros_like(x, dtype=torch.bool)
            masked[:, start:end] = x[:, start:end] == MASK_ID
            k = max(int((masked & (p >= tau)).sum()), 1)
            score = torch.where(masked, score, torch.full_like(score, -np.inf))
            pick = torch.topk(score[0], k).indices
            x[0, pick] = x0[0, pick]
            done += k
            step += 1
        if b != 0:
            continue
        pos = torch.zeros_like(x, dtype=torch.bool)
        pos[:, start:end] = True
        unsafe = combined.safety_perplexity_from_logits(logits, pos, *cue_ids) >= lam
        if hmean is not None:
            unsafe = unsafe or combined.cosine_distance(hmean, base_hidden) >= lam
        if not unsafe:
            continue
        cand = torch.arange(start, end)
        rm = cand[torch.randperm(block)[: int(block * ratio)]]
        old = x[:, rm].clone()
        x[:, rm] = MASK_ID
        quota = combined.get_num_transfer_tokens(x[:, start:end] == MASK_ID, refine)
        for r in range(refine):
            logits = model(x).logits
            nfe_r += 1
            logits[0, rm, old[0]] -= 1e6
            x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
            order = torch.rand(x0.shape)
            masked = torch.zeros_like(x, dtype=torch.bool)
            masked[:, start:end] = x[:, start:end] == MASK_ID
            order = torch.where(masked, order, torch.full_like(order, -np.inf))
            k = min(int(quota[0, r]), int(masked.sum()))
            if k:
                pick = torch.topk(order[0], k).indices
                x[0, pick] = x0[0, pick]
    return x, nfe, nfe_r, alphas


def _s3(model, tok, prompt, gen, block, temperature, tau, seed, **kw):
    args = dict(alpha0=0.3, sp_threshold=0.1, refinement_steps=8, remask_ratio=0.9, baseline_hidden=BASE_HIDDEN,
                cue_ids=combined.cue_id_sets(tok))
    args.update(kw)
    torch.manual_seed(seed)
    return combined.generate(model, prompt, gen_length=gen, block_length=block, temperature=temperature,
                             mask_id=MASK_ID, mode="threshold", threshold=tau, defense=True, **args)


def _s2(model, prompt, gen, block, temperature, tau, seed):
    torch.manual_seed(seed)
    return combined.generate(model, prompt, gen_length=gen, block_length=block, temperature=temperature,
                             mask_id=MASK_ID, mode="threshold", threshold=tau)


# --------------------------------------------------------------------------- equality with the written-out rule

@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
@pytest.mark.parametrize("lam", [0.0, 10.0])            # 0 -> audit always fires (repair), 10 -> never
def test_equals_written_out_rule(fake_model, fake_tokenizer, gen, block, temperature, lam):
    cue = combined.cue_id_sets(fake_tokenizer)
    for tau in TAUS:
        for seed in range(3):
            prompt = _prompt(seed)
            torch.manual_seed(seed)
            ref, nfe, nfe_r, alphas = reference_s3(fake_model, prompt, gen, block, temperature, tau, 0.3, lam, cue)
            out, st = _s3(fake_model, fake_tokenizer, prompt, gen, block, temperature, tau, seed, sp_threshold=lam)
            assert torch.equal(out, ref), (tau, seed)
            assert (st.nfe_gen, st.nfe_repair) == (nfe, nfe_r)
            assert st.repaired is (lam == 0.0) and st.nfe_repair == (8 if lam == 0.0 else 0)
            assert st.alphas == pytest.approx(alphas, abs=1e-6)
            assert not (out == MASK_ID).any()


def test_logit_fallback_without_baseline_hidden(fake_model, fake_tokenizer):
    cue = combined.cue_id_sets(fake_tokenizer)
    for seed in range(3):
        prompt = _prompt(seed)
        for lam in (0.0, 10.0):
            torch.manual_seed(seed)
            ref, *_ = reference_s3(fake_model, prompt, 32, 8, 0.5, 0.1, 0.3, lam, cue, base_hidden=None)
            out, st = _s3(fake_model, fake_tokenizer, prompt, 32, 8, 0.5, 0.1, seed, sp_threshold=lam,
                          baseline_hidden=None)
            assert torch.equal(out, ref)
            assert st.audit_hidden is None and st.audit_score == st.audit_logit


# --------------------------------------------------------------------------- each part of the rule on its own

class _Recorder(torch.nn.Module):
    """Wraps the fake model and records, per generation forward, the masked positions of block 0 and p(argmax)."""

    def __init__(self, inner, start, end):
        super().__init__()
        self.inner, self.start, self.end, self.log = inner, start, end, []

    @property
    def device(self):
        return self.inner.device

    def forward(self, x, **kw):
        out = self.inner(x, **kw)
        masked = x[0, self.start:self.end] == MASK_ID
        p = F.softmax(out.logits[0, self.start:self.end].to(torch.float64), -1).max(-1).values
        self.log.append((masked.clone(), p))
        return out


@pytest.mark.parametrize("tau", TAUS)
def test_k_is_counted_on_p_not_on_the_score(fake_model, fake_tokenizer, tau):
    """alpha0=1 makes the score pure noise; the number committed per step must still be max(#{p >= tau}, 1)
    (temperature 0 so x0 = argmax and p(x0) = max softmax). Alpha follows the progress rule on those counts."""
    for seed in range(3):
        prompt = _prompt(seed)
        P, block = prompt.shape[1], 16
        rec = _Recorder(fake_model, P, P + block)
        _, st = _s3(rec, fake_tokenizer, prompt, 16, block, 0.0, tau, seed, alpha0=1.0, sp_threshold=10.0)
        counts = [int(m.sum()) for m, _ in rec.log] + [0]
        committed = [counts[j] - counts[j + 1] for j in range(len(rec.log))]
        expected = [max(int((m & (p >= tau)).sum()), 1) for m, p in rec.log]
        assert committed == expected
        done = np.cumsum([0] + committed[:-1])
        assert st.alphas == pytest.approx([1.0 * (1 - d / block) for d in done], abs=1e-6)


def test_which_positions_follows_the_score(fake_model, fake_tokenizer):
    """Same k per step, but alpha0 > 0 picks other positions than plain p (defense off) for some input."""
    differ = 0
    for seed in range(6):
        prompt = _prompt(seed)
        off, _ = _s2(fake_model, prompt, 32, 8, 0.0, 0.1, seed)
        on, _ = _s3(fake_model, fake_tokenizer, prompt, 32, 8, 0.0, 0.1, seed, alpha0=1.0, sp_threshold=10.0)
        differ += not torch.equal(on, off)
    assert differ > 0


@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("tau", TAUS)
def test_alpha0_zero_equals_s2(fake_model, fake_tokenizer, gen, block, tau):
    """alpha0=0, no repair: the defense only adds the audit; tokens and forwards equal S2 (= Fast-dLLM).
    Temperature 0 only: at T > 0 the defense draws R every step (DiffuGuard), which shifts the RNG stream."""
    for seed in range(3):
        prompt = _prompt(seed)
        off, st_off = _s2(fake_model, prompt, gen, block, 0.0, tau, seed)
        on, st_on = _s3(fake_model, fake_tokenizer, prompt, gen, block, 0.0, tau, seed, alpha0=0.0,
                        sp_threshold=10.0)
        assert torch.equal(on, off) and st_on.nfe_gen == st_off.nfe_gen
        assert st_on.audit_score is not None and st_on.nfe_repair == 0


@pytest.mark.parametrize("tau", [1.01, 0.9, 0.05])
@pytest.mark.parametrize("refinement_steps", [8, 3])
def test_repair_budget_is_fixed_and_ignores_tau(fake_model, fake_tokenizer, tau, refinement_steps):
    for seed in range(3):
        out, st = _s3(fake_model, fake_tokenizer, _prompt(seed), 32, 8, 0.5, tau, seed, sp_threshold=0.0,
                      refinement_steps=refinement_steps)
        assert st.repaired and st.nfe_repair == refinement_steps
        assert not (out == MASK_ID).any()


def test_progress_alpha_reaches_low_values_before_block_end(fake_model, fake_tokenizer):
    _, st = _s3(fake_model, fake_tokenizer, _prompt(0), 32, 8, 0.0, 1.01, 0, sp_threshold=10.0)
    # tau > 1 -> one token per step -> alpha0 * (1 - j/8)
    assert st.alphas == pytest.approx([0.3 * (1 - j / 8) for j in range(8)], abs=1e-6)


def test_guards(fake_model, fake_tokenizer):
    with pytest.raises(ValueError):
        _s3(fake_model, fake_tokenizer, _prompt(0), 32, 8, 0.0, 0.9, 0, alpha_rule="original")
    with pytest.raises(ValueError):
        _s3(fake_model, fake_tokenizer, _prompt(0), 32, 8, 0.0, 0.9, 0, cue_ids=None)

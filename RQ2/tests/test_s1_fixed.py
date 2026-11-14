"""S1: combined decoder with acceleration off (fixed quota, no cache) == DiffuGuard original, block by block.

Reference = DiffuGuard/semi_ar.py (the original generate() with only its block-widening guard switched off),
which is itself tested against the Fast-dLLM baseline generate() in DiffuGuard/tests/test_semi_ar.py.
Equality is exact (torch.equal on token ids) and includes the RNG stream at temperature > 0.
"""

import pytest
import torch

import semi_ar
from conftest import DG_ROOT, MASK_ID
from src import combined

CASES = [(32, 32, 8), (16, 32, 8), (8, 32, 8), (4, 32, 8), (64, 64, 16), (8, 64, 16), (16, 16, 16)]
BASE_HIDDEN = torch.ones(8)  # FakeModel hidden size


@pytest.fixture(scope="module")
def sa():
    return semi_ar.load(DG_ROOT)


def _prompt(seed):
    return torch.randint(2, 60, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))


def _reference(sa, model, tok, prompt, steps, gen, block, temperature, defense, seed, **kw):
    args = dict(steps=steps, gen_length=gen, block_length=block, temperature=temperature, mask_id=MASK_ID,
                cfg_scale=0.0, fill_all_masks=False, correct_only_first_block=True)
    if defense:
        args.update(remasking="adaptive_step", alpha0=0.3, sp_mode="hidden", sp_threshold=0.1,
                    refinement_steps=8, remask_ratio=0.9, baseline_hidden=BASE_HIDDEN)
    else:
        args.update(remasking="low_confidence", sp_mode="off")
    args.update(kw)
    torch.manual_seed(seed)
    return sa.generate(model, tok, prompt, **args)


def _combined(model, tok, prompt, steps, gen, block, temperature, defense, seed, **kw):
    args = dict(steps=steps, gen_length=gen, block_length=block, temperature=temperature, mask_id=MASK_ID,
                defense=defense, alpha_rule="original")
    if defense:
        args.update(alpha0=0.3, sp_threshold=0.1, refinement_steps=8, remask_ratio=0.9,
                    baseline_hidden=BASE_HIDDEN, cue_ids=combined.cue_id_sets(tok))
    args.update(kw)
    torch.manual_seed(seed)
    return combined.generate(model, prompt, **args)


# --------------------------------------------------------------------------- equality with the original

@pytest.mark.parametrize("steps,gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_defense_off_equals_original(sa, fake_model, fake_tokenizer, steps, gen, block, temperature):
    for seed in range(4):
        prompt = _prompt(seed)
        ref = _reference(sa, fake_model, fake_tokenizer, prompt, steps, gen, block, temperature, False, seed)
        out, st = _combined(fake_model, fake_tokenizer, prompt, steps, gen, block, temperature, False, seed)
        assert torch.equal(out, ref)
        assert st.nfe_gen == max(steps // (gen // block), 1) * (gen // block) and st.nfe_repair == 0
        assert st.audit_score is None


@pytest.mark.parametrize("steps,gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
@pytest.mark.parametrize("repair", [False, True])
def test_defense_on_equals_original(sa, fake_model, fake_tokenizer, steps, gen, block, temperature, repair):
    # lambda 10 -> audit never fires; lambda 0 -> always fires (score >= 0), so both branches are covered
    lam = dict(sp_threshold=0.0) if repair else dict(sp_threshold=10.0)
    for seed in range(4):
        prompt = _prompt(seed)
        ref = _reference(sa, fake_model, fake_tokenizer, prompt, steps, gen, block, temperature, True, seed, **lam)
        out, st = _combined(fake_model, fake_tokenizer, prompt, steps, gen, block, temperature, True, seed, **lam)
        assert torch.equal(out, ref)
        assert st.repaired is repair and st.audit_unsafe is repair
        assert st.nfe_repair == (8 if repair else 0)
        assert not (out == MASK_ID).any()


@pytest.mark.parametrize("refinement_steps", [1, 2, 3, 16])
def test_repair_budget_equals_original(sa, fake_model, fake_tokenizer, refinement_steps):
    """RQ1 used proportional repair (16/8/4/2); any budget must match."""
    for seed in range(3):
        prompt = _prompt(seed)
        kw = dict(sp_threshold=0.0, refinement_steps=refinement_steps)
        ref = _reference(sa, fake_model, fake_tokenizer, prompt, 32, 64, 16, 0.5, True, seed, **kw)
        out, st = _combined(fake_model, fake_tokenizer, prompt, 32, 64, 16, 0.5, True, seed, **kw)
        assert torch.equal(out, ref)
        assert st.nfe_repair == refinement_steps


def test_logit_fallback_without_baseline_hidden(sa, fake_model, fake_tokenizer):
    """Original: sp_mode=hidden without baseline_hidden falls back to the logit conflict score."""
    for seed in range(3):
        prompt = _prompt(seed)
        for lam in (0.0, 10.0):
            ref = _reference(sa, fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.5, True, seed,
                             sp_threshold=lam, baseline_hidden=None)
            out, st = _combined(fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.5, True, seed,
                                sp_threshold=lam, baseline_hidden=None)
            assert torch.equal(out, ref)
            assert st.audit_hidden is None and st.audit_score == st.audit_logit


def test_audit_values_match_the_original_print(sa, fake_model, fake_tokenizer, capsys):
    """The audit score the combined decoder records is the one the original prints for block 1."""
    prompt = _prompt(1)
    _reference(sa, fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.0, True, 0, sp_threshold=10.0, debug_print=True)
    printed = capsys.readouterr().out.splitlines()[0]          # "[Block 1/4] Safety score=0.xxx, unsafe=False"
    _, st = _combined(fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.0, True, 0, sp_threshold=10.0)
    assert f"Safety score={st.audit_score:.3f}" in printed and "unsafe=False" in printed


# --------------------------------------------------------------------------- alpha rules

def test_original_alpha_schedule(fake_model, fake_tokenizer):
    _, st = _combined(fake_model, fake_tokenizer, _prompt(0), 32, 32, 8, 0.0, True, 0, sp_threshold=10.0)
    spb = st.steps_per_block
    assert st.alphas == pytest.approx([0.3 * (1 - i / (spb - 1)) for i in range(spb)], abs=1e-6)


def test_progress_alpha_schedule(fake_model, fake_tokenizer):
    # 8 masks in block 0, 4 steps -> 2 committed per step -> alpha0 * (1 - 0/8, 1 - 2/8, 1 - 4/8, 1 - 6/8)
    _, st = _combined(fake_model, fake_tokenizer, _prompt(0), 16, 32, 8, 0.0, True, 0, sp_threshold=10.0,
                      alpha_rule="progress")
    assert st.alphas == pytest.approx([0.3, 0.225, 0.15, 0.075], abs=1e-6)


@pytest.mark.parametrize("rule", ["original", "progress"])
def test_alpha0_zero_equals_defense_off_tokens(fake_model, fake_tokenizer, rule):
    """With alpha0=0 and no repair the defense only adds the audit; tokens equal the undefended decode."""
    for seed in range(3):
        prompt = _prompt(seed)
        off, _ = _combined(fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.5, False, seed)
        on, _ = _combined(fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.5, True, seed,
                          alpha0=0.0, sp_threshold=10.0, alpha_rule=rule)
        assert torch.equal(on, off)


# --------------------------------------------------------------------------- guards

def test_rejects_prompt_masks(fake_model, fake_tokenizer):
    prompt = _prompt(0)
    prompt[0, 2] = MASK_ID
    with pytest.raises(NotImplementedError):
        _combined(fake_model, fake_tokenizer, prompt, 16, 32, 8, 0.0, False, 0)


def test_rejects_unknown_modes(fake_model, fake_tokenizer):
    with pytest.raises(ValueError):
        _combined(fake_model, fake_tokenizer, _prompt(0), 16, 32, 8, 0.0, False, 0, mode="greedy")

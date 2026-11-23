"""safegate.decoder: equal to the RQ2 combined decoder (column A) without a gate; trace never changes tokens;
gate rules hold as stated."""
import pytest
import torch

from conftest import MASK_ID, dija_prompt, plain_prompt
from safegate import decoder
from safegate.trace import StepTrace
from src import combined

GEN, BLOCK = 32, 8
BASE_HIDDEN = torch.ones(8)
REF_IDS, CMP_IDS = [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]
SETS = {"short": (REF_IDS, CMP_IDS), "raec": ([5, 20, 21], [11, 30])}


def _kw(tok, defense, tau, temperature):
    kw = dict(gen_length=GEN, block_length=BLOCK, temperature=temperature, mask_id=MASK_ID,
              prompt_masks="first_block", defense=defense)
    if defense:
        kw.update(alpha0=0.3, sp_threshold=0.1, refinement_steps=8, baseline_hidden=BASE_HIDDEN,
                  cue_ids=combined.cue_id_sets(tok))
    return kw, tau


def _ours(model, prompt, kw, tau, seed=0, **extra):
    torch.manual_seed(seed)
    return decoder.generate(model, prompt, threshold=tau, **kw, **extra)


def _combined(model, prompt, kw, tau, seed=0):
    torch.manual_seed(seed)
    return combined.generate(model, prompt, mode="threshold", threshold=tau, alpha_rule="progress", **kw)


CASES = [(d, tau, t, kind) for d in (False, True) for tau in (0.9, 0.2) for t in (0.0, 0.5)
         for kind in ("plain", "dija")]


@pytest.mark.parametrize("defense,tau,temperature,kind", CASES)
def test_no_gate_equals_combined(fake_model, fake_tokenizer, defense, tau, temperature, kind):
    for seed in range(3):
        prompt = plain_prompt(seed) if kind == "plain" else dija_prompt(seed)
        kw, tau_ = _kw(fake_tokenizer, defense, tau, temperature)
        x_ref, st_ref = _combined(fake_model, prompt, kw, tau_, seed)
        x, st = _ours(fake_model, prompt, kw, tau_, seed)
        assert torch.equal(x, x_ref)
        assert (st.nfe_gen, st.nfe_repair, st.repaired, st.audit_unsafe) == \
            (st_ref.nfe_gen, st_ref.nfe_repair, st_ref.repaired, st_ref.audit_unsafe)
        assert st.audit_score == st_ref.audit_score and st.alphas == st_ref.alphas
        assert sum(st.commits_per_forward) == GEN + int((prompt == MASK_ID).sum())
        assert len(st.commits_per_forward) == st.nfe_gen
        assert not bool((x == MASK_ID).any())


@pytest.mark.parametrize("defense,temperature", [(False, 0.0), (False, 0.5), (True, 0.5)])
def test_trace_and_idle_gate_do_not_change_tokens(fake_model, fake_tokenizer, defense, temperature):
    prompt = dija_prompt(1)
    kw, tau = _kw(fake_tokenizer, defense, 0.9, temperature)
    x0, _ = _ours(fake_model, prompt, kw, tau)
    tr = StepTrace(SETS, resp_k=4)
    x1, _ = _ours(fake_model, prompt, kw, tau, trace=tr)
    never = decoder.GateConfig(a=2.0, b=2.0, K=4, H=8, windows=("response", "infill"))
    x2, st2 = _ours(fake_model, prompt, kw, tau, gate=never, gate_ids=SETS["short"], trace=StepTrace(SETS, 4))
    zero_h = decoder.GateConfig(a=0.0, b=0.0, K=4, H=0)
    x3, st3 = _ours(fake_model, prompt, kw, tau, gate=zero_h, gate_ids=SETS["short"])
    assert torch.equal(x0, x1) and torch.equal(x0, x2) and torch.equal(x0, x3)
    assert st2.gate_steps == 0 and st2.gate_checks > 0 and st3.gate_steps == 0
    gen = [r for r in tr.records if r["phase"] == "gen"]
    assert len(gen) == _ours(fake_model, prompt, kw, tau)[1].block0_forwards
    if not defense:   # without repair, block-0 commits = every tracked position exactly once
        assert sum(sum(r["resp"]["commit"]) for r in gen) == 4
        assert sum(sum(r["infill"]["commit"]) for r in gen) == int((prompt == MASK_ID).sum())
    for r in gen:
        for g in (r["resp"], r["infill"]):
            for v in g["m"].values():
                assert all(0.0 <= m <= 1.0 + 1e-6 for m in v)
            assert all(not (c and not m) for c, m in zip(g["commit"], g["masked"]))   # commits only open slots


@pytest.mark.parametrize("windows", [("response",), ("infill",), ("response", "infill")])
def test_firing_gate_holds_non_refusal_picks(fake_model, fake_tokenizer, windows):
    prompt = dija_prompt(2)
    kw, tau = _kw(fake_tokenizer, False, 0.2, 0.0)
    gate = decoder.GateConfig(a=0.0, b=0.0, K=4, H=5, windows=windows, K_infill=3)
    tr = StepTrace(SETS, resp_k=4)
    x, st = _ours(fake_model, prompt, kw, tau, gate=gate, gate_ids=SETS["short"], trace=tr)
    assert not bool((x == MASK_ID).any())
    assert 1 <= st.gate_steps <= 5 and st.gate_first_step == 0 and st.gate_held > 0
    P = prompt.shape[1]
    win = set()
    if "response" in windows:
        win |= set(range(P, P + 4))
    if "infill" in windows:
        win |= set((prompt[0] == MASK_ID).nonzero().squeeze(1).tolist()[:3])
    infill_pos = (prompt[0] == MASK_ID).nonzero().squeeze(1).tolist()
    held_total = 0
    for r in [r for r in tr.records if r["phase"] == "gen"]:
        if not r["gate"] or not r["gate"]["fire"]:
            continue
        held_total += r["gate"]["held"]
        assert r["n_commit"] == r["k"] - r["gate"]["held"]
        groups = [(r["resp"], list(range(P, P + 4))), (r["infill"], infill_pos)]
        for g, pos in groups:
            for p, c, t in zip(pos, g["commit"], g["tok"]):
                if c and p in win:
                    assert t in REF_IDS          # inside the window only refusal tokens get through
    assert held_total == st.gate_held


def test_gate_off_after_h_steps_is_plain_fast_dllm(fake_model, fake_tokenizer):
    """Once H steps are used up the remaining steps follow the Fast-dLLM rule (n_commit == k)."""
    prompt = plain_prompt(3)
    kw, tau = _kw(fake_tokenizer, False, 0.9, 0.0)
    gate = decoder.GateConfig(a=0.0, b=0.0, K=8, H=2)
    tr = StepTrace(SETS, resp_k=8)
    _, st = _ours(fake_model, prompt, kw, tau, gate=gate, gate_ids=SETS["short"], trace=tr)
    gen = [r for r in tr.records if r["phase"] == "gen"]
    fired = [r["step"] for r in gen if r["gate"] and r["gate"]["fire"]]
    assert fired == [0, 1] and st.gate_steps == 2
    assert all(r["n_commit"] == r["k"] for r in gen if r["step"] >= 2)


def test_gate_config_validation():
    with pytest.raises(ValueError):
        decoder.GateConfig(a=None, b=0.1)
    with pytest.raises(ValueError):
        decoder.GateConfig(a=0.1, b=0.1, windows=("tail",))
    with pytest.raises(ValueError):
        decoder.GateConfig(a=0.1, b=0.1, windows=())

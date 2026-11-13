"""S6: ASR layout with mask tokens inside the prompt (DIJA), prompt_masks="first_block".

Agreed layout (same as the RQ2 Without ASR runs): block 0 = {prompt masks} U {first block_length generated
positions}, decoded together; later blocks = generated positions only. Fixed: steps // num_blocks forwards per
block with the original quota over the block's masks. Threshold: per step, commit the block's masked positions
with p >= tau (at least the arg-max one) until the block is full. D/E: no cache in block 0, a fresh cache at every
later block start (built on the repaired tokens). Defense: alpha progress over P_mask + block, audit and repair
over the block-0 positions only, fixed prompt tokens never changed.

References below are written out from that description (defense off) and checked against the decoder token for
token; the defense rules are checked as properties; the last test documents the contrast with the original.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import DG_ROOT, MASK_ID, FakeModel
from src import combined
from test_s4_cache import ExactCacheModel

BASE_HIDDEN = torch.ones(8)
GEN, BLOCK = 32, 8                        # 4 blocks, like the ASR layout 128 / 32 (small for CPU)
TAUS = [0.9, 0.2, 0.05]


def _dija_prompt(seed, n_masks=(2, 3)):
    """Prompt text with mask slots inside it (DIJA style): text, masks, text, masks, text."""
    g = torch.Generator().manual_seed(seed)
    parts, a, b = [], n_masks[0], n_masks[1]
    for n_text, n_mask in ((4 + seed % 3, a), (3, b), (2, 0)):
        parts.append(torch.randint(2, 60, (n_text,), generator=g))
        parts.append(torch.full((n_mask,), MASK_ID))
    return torch.cat(parts).unsqueeze(0)


def _region(prompt, b):
    P = prompt.shape[1]
    r = torch.zeros(1, P + GEN, dtype=torch.bool)
    r[:, P + b * BLOCK:P + (b + 1) * BLOCK] = True
    if b == 0:
        r[0, :P] = prompt[0] == MASK_ID
    return r


def _run(model, prompt, mode, tok=None, defense=False, seed=0, **kw):
    args = dict(gen_length=GEN, block_length=BLOCK, temperature=0.0, mask_id=MASK_ID, mode=mode,
                prompt_masks="first_block", defense=defense)
    if mode == "fixed":
        args["steps"] = GEN
    if defense:
        args.update(alpha0=0.3, sp_threshold=0.1, refinement_steps=8, baseline_hidden=BASE_HIDDEN,
                    cue_ids=combined.cue_id_sets(tok), alpha_rule="progress")
    args.update(kw)
    torch.manual_seed(seed)
    return combined.generate(model, prompt, **args)


# --------------------------------------------------------------------------- written-out references (defense off)

def ref_fixed(model, prompt):
    x = torch.cat([prompt, torch.full((1, GEN), MASK_ID)], dim=1)
    spb = GEN // (GEN // BLOCK)
    nfe = 0
    for b in range(GEN // BLOCK):
        r = _region(prompt, b)
        quota = combined.get_num_transfer_tokens((x[r] == MASK_ID).unsqueeze(0), spb)[0]
        for i in range(spb):
            logits = model(x).logits
            nfe += 1
            x0 = logits.argmax(-1)
            p = F.softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            cand = (x == MASK_ID) & r
            conf = torch.where(cand, p, torch.tensor(-np.inf))
            k = int(quota[i])
            if k:
                idx = conf[0].topk(k).indices
                x[0, idx] = x0[0, idx]
    return x, nfe


def ref_threshold(model, prompt, tau):
    x = torch.cat([prompt, torch.full((1, GEN), MASK_ID)], dim=1)
    nfe, per_block = 0, []
    for b in range(GEN // BLOCK):
        r = _region(prompt, b)
        n = 0
        while bool((x[r] == MASK_ID).any()):
            logits = model(x).logits
            nfe += 1
            n += 1
            x0 = logits.argmax(-1)
            p = F.softmax(logits.to(torch.float64), dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            cand = (x == MASK_ID) & r
            conf = torch.where(cand, p, torch.tensor(-np.inf, dtype=torch.float64))
            sel = cand & (p >= tau)
            sel[0, conf[0].argmax()] = True                   # always at least the most confident one
            x[sel] = x0[sel]
        per_block.append(n)
    return x, nfe, per_block


@pytest.mark.parametrize("seed", range(4))
def test_fixed_defense_off_equals_written_out(fake_model, seed):
    prompt = _dija_prompt(seed)
    ref, nfe = ref_fixed(fake_model, prompt)
    out, st = _run(fake_model, prompt, "fixed")
    assert torch.equal(out, ref) and st.nfe_gen == nfe == GEN       # condition 0: 32 per block, 128 in all (scaled)
    assert not (out == MASK_ID).any()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("tau", TAUS)
def test_threshold_defense_off_equals_written_out(fake_model, seed, tau):
    prompt = _dija_prompt(seed)
    P_mask = int((prompt == MASK_ID).sum())
    ref, nfe, per_block = ref_threshold(fake_model, prompt, tau)
    out, st = _run(fake_model, prompt, "threshold", threshold=tau)
    assert torch.equal(out, ref) and st.nfe_gen == nfe
    assert per_block[0] <= P_mask + BLOCK and all(n <= BLOCK for n in per_block[1:])
    assert not (out == MASK_ID).any()


def test_no_prompt_masks_changes_nothing(fake_model, fake_tokenizer):
    prompt = torch.randint(2, 60, (1, 7), generator=torch.Generator().manual_seed(3))
    for mode, kw in (("fixed", {}), ("threshold", dict(threshold=0.2))):
        for defense in (False, True):
            a, sa_ = _run(fake_model, prompt, mode, fake_tokenizer, defense, **kw)
            b, sb = _run(fake_model, prompt, mode, fake_tokenizer, defense, prompt_masks="error", **kw)
            assert torch.equal(a, b) and sa_.as_dict() == sb.as_dict()


# --------------------------------------------------------------------------- cache: off in block 0, rebuilt later

@pytest.mark.parametrize("cache", ["prefix", "dual"])
@pytest.mark.parametrize("defense,lam", [(False, None), (True, 10.0), (True, 0.0)])
def test_cache_from_block_1_equals_no_cache(fake_tokenizer, cache, defense, lam):
    kw = {} if not defense else dict(sp_threshold=lam)
    for tau in TAUS:
        for seed in range(3):
            prompt = _dija_prompt(seed)
            P = prompt.shape[1]
            base, st0 = _run(ExactCacheModel(FakeModel().eval()), prompt, "threshold", fake_tokenizer, defense,
                             seed, threshold=tau, **kw)
            m = ExactCacheModel(FakeModel().eval())
            out, st = _run(m, prompt, "threshold", fake_tokenizer, defense, seed, threshold=tau, cache=cache,
                           cache_start_block=1, **kw)
            assert torch.equal(out, base), (tau, seed)
            assert (st.nfe_gen, st.nfe_repair, st.alphas) == (st0.nfe_gen, st0.nfe_repair, st0.alphas)
            assert st.audit_unsafe == st0.audit_unsafe and st.audit_hidden == st0.audit_hidden
            kinds = [k for k, _ in m.calls]
            first_build = kinds.index("build")
            assert set(kinds[:first_build]) == {"full"}                      # block 0 (+ repair): no cache
            assert kinds.count("build") == GEN // BLOCK - 1                   # a fresh cache at blocks 1..3
            built = m.calls[first_build][1]
            r0 = _region(prompt, 0)
            assert torch.equal(built[r0], out[r0])                           # built on the final (repaired) block 0
            assert not (built[:, :P] == MASK_ID).any()                       # prompt masks complete in the cache
            if defense and lam == 0.0:
                assert st.repaired and st.nfe_repair == 8


def test_guards(fake_model):
    prompt = _dija_prompt(0)
    with pytest.raises(NotImplementedError):
        combined.generate(fake_model, prompt, gen_length=GEN, block_length=BLOCK, steps=GEN, mask_id=MASK_ID)
    with pytest.raises(ValueError, match="cache_start_block"):
        _run(fake_model, prompt, "threshold", threshold=0.9, cache="prefix")
    with pytest.raises(ValueError):
        _run(fake_model, prompt, "fixed", prompt_masks="fill")


# --------------------------------------------------------------------------- defense: alpha, audit, repair range

@pytest.mark.parametrize("seed", range(4))
def test_progress_alpha_counts_prompt_masks(fake_model, fake_tokenizer, seed):
    prompt = _dija_prompt(seed)
    n0 = int((prompt == MASK_ID).sum()) + BLOCK
    out, st = _run(fake_model, prompt, "fixed", fake_tokenizer, True, seed, sp_threshold=10.0)
    quota = combined.get_num_transfer_tokens(torch.ones(1, n0, dtype=torch.bool), GEN // (GEN // BLOCK))[0]
    committed = np.concatenate([[0], np.cumsum(quota.numpy())[:-1]])
    want = [0.3 * (1 - c / n0) for c in committed]
    assert st.alphas == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("mode,kw", [("fixed", {}), ("threshold", dict(threshold=0.2))])
@pytest.mark.parametrize("seed", range(4))
def test_repair_stays_in_block_0_positions_and_keeps_the_prompt(fake_model, fake_tokenizer, mode, kw, seed):
    prompt = _dija_prompt(seed)
    P = prompt.shape[1]
    fixed_text = prompt[0] != MASK_ID
    pre, st_pre = _run(fake_model, prompt, mode, fake_tokenizer, True, seed, sp_threshold=10.0, **kw)
    out, st = _run(fake_model, prompt, mode, fake_tokenizer, True, seed, sp_threshold=0.0, **kw)
    assert not st_pre.repaired and st.repaired and st.nfe_repair == 8
    assert torch.equal(out[0, :P][fixed_text], prompt[0][fixed_text])          # prompt text never changes
    assert not (out == MASK_ID).any()
    # which positions the repair remasks: test_repair_remasks_exactly_the_block_0_share


def test_repair_reaches_prompt_mask_positions(fake_model, fake_tokenizer):
    hit = 0
    for seed in range(8):
        prompt = _dija_prompt(seed)
        P = prompt.shape[1]
        pre, _ = _run(fake_model, prompt, "fixed", fake_tokenizer, True, seed, sp_threshold=10.0)
        out, _ = _run(fake_model, prompt, "fixed", fake_tokenizer, True, seed, sp_threshold=0.0)
        pm = prompt[0] == MASK_ID
        hit += bool((pre[0, :P][pm] != out[0, :P][pm]).any())
    assert hit > 0


class Recorder(torch.nn.Module):
    """Wraps a model; records hidden-state requests and the token ids of every forward."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.seen = inner, []

    @property
    def device(self):
        return self.inner.device

    def forward(self, x, **kw):
        self.seen.append(x.clone())
        return self.inner(x, **kw)


def test_repair_remasks_exactly_the_block_0_share(fake_model, fake_tokenizer):
    for seed in range(4):
        prompt = _dija_prompt(seed)
        n0 = int((prompt == MASK_ID).sum()) + BLOCK
        m = Recorder(fake_model)
        out, st = _run(m, prompt, "fixed", fake_tokenizer, True, seed, sp_threshold=0.0)
        spb = GEN // (GEN // BLOCK)
        first_repair = m.seen[spb]                     # block 0 = spb forwards, then the 8 repair forwards
        r0 = _region(prompt, 0)
        assert int((first_repair == MASK_ID).sum() - (first_repair[:, prompt.shape[1] + BLOCK:] == MASK_ID).sum()) \
            == int(n0 * 0.9)
        assert not ((first_repair == MASK_ID) & ~r0 & (torch.arange(first_repair.shape[1]) <
                                                        prompt.shape[1] + BLOCK)).any()


def test_audit_covers_the_block_0_positions(fake_model, fake_tokenizer):
    """Hidden distance = first step's last-layer mean over the block-0 positions (prompt masks + first block)."""
    prompt = _dija_prompt(1)
    out, st = _run(fake_model, prompt, "fixed", fake_tokenizer, True, 1, sp_threshold=10.0)
    x = torch.cat([prompt, torch.full((1, GEN), MASK_ID)], dim=1)
    h = fake_model(x, output_hidden_states=True, return_dict=True).hidden_states[-1]
    want = combined.cosine_distance(h[0, _region(prompt, 0)[0]].mean(dim=0), BASE_HIDDEN)
    assert st.audit_hidden == pytest.approx(want, rel=1e-6)


# --------------------------------------------------------------------------- contrast with the original

def test_original_repair_rewrites_fixed_prompt_text_ours_does_not(fake_model, fake_tokenizer):
    """The original generate() (one block = every remaining mask, as RQ1's DIJA runs) widens the block to span
    from the first prompt mask to the end, and its repair remasks that whole span except the system prompt, so it
    also rewrites the attack text between the masks. RQ2 restricts block 0 to the mask positions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dg_gen", DG_ROOT / "utility" / "generate_function_llada.py")
    gen_mod = importlib.util.module_from_spec(spec)      # the unpatched original (with the block guard)
    spec.loader.exec_module(gen_mod)
    changed_orig = changed_ours = 0
    for seed in range(4):
        prompt = _dija_prompt(seed)
        P = prompt.shape[1]
        fixed_text = prompt[0] != MASK_ID
        first_mask = int((prompt[0] == MASK_ID).nonzero()[0])
        between = fixed_text.clone()
        between[:first_mask] = False                         # attack text after the first mask slot
        torch.manual_seed(seed)
        x = gen_mod.generate(fake_model, fake_tokenizer, prompt, steps=GEN, gen_length=GEN, block_length=GEN,
                             temperature=0.0, mask_id=MASK_ID, remasking="adaptive_step", alpha0=0.3,
                             sp_mode="hidden", sp_threshold=0.0, refinement_steps=8, remask_ratio=0.9,
                             baseline_hidden=BASE_HIDDEN, protected_index=torch.zeros_like(prompt, dtype=torch.bool))
        changed_orig += bool((x[0, :P][between] != prompt[0][between]).any())
        out, _ = _run(fake_model, prompt, "fixed", fake_tokenizer, True, seed, sp_threshold=0.0)
        changed_ours += bool((out[0, :P][fixed_text] != prompt[0][fixed_text]).any())
    assert changed_orig > 0 and changed_ours == 0

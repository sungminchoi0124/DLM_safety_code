"""S4: KV cache (prefix / dual) in threshold mode.

1. Defense off == Fast-dLLM generate_with_prefix_cache / generate_with_dual_cache (threshold), token for token,
   against a written-out copy (always) and the real generate.py (FAST_DLLM_ROOT). The fake model is a small
   2-layer attention model with the LLaDAModelLM cache API (past_key_values, use_cache, replace_position with
   in-place KV overwrite), so a cached forward really differs from a full one.
2. Plumbing (windows, offsets, audit positions, hidden states, repair): with an "exact" cache whose cached logits
   equal a full forward on the current sequence, every cache mode must give exactly the no-cache result,
   defense and repair included.
3. Cache consistency after repair (test 2 of the RQ2 plan): repair runs full forwards, no cached forward follows
   the repair, and the next block's cache is built from the repaired tokens.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import MASK_ID, FakeModel
from src import combined

CASES = [(32, 8), (32, 32), (64, 16)]
TAUS = [0.9, 0.2, 0.1, 0.05]
BASE_HIDDEN = torch.ones(8)


def _prompt(seed):
    return torch.randint(2, 60, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------- fake models

class FakeCacheModel(torch.nn.Module):
    """Bidirectional 2-layer single-head attention with LLaDAModelLM's cache API: past_key_values = per layer
    (k, v) of shape [B, 1, T, d]; without replace_position the new k/v are appended (positions continue after
    the past, as RoPE does); with replace_position they overwrite the cached k/v at those positions in place."""

    def __init__(self, vocab=64, dim=8, layers=2):
        super().__init__()
        g = torch.Generator().manual_seed(4321)
        self.vocab, self.dim = vocab, dim
        self.emb = torch.nn.Parameter(torch.randn(vocab, dim, generator=g))
        self.pos = torch.nn.Parameter(torch.randn(1024, dim, generator=g))
        self.w = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4, dim, dim, generator=g) / dim ** 0.5)
                                         for _ in range(layers)])
        self.out = torch.nn.Parameter(torch.randn(dim, vocab, generator=g))

    @property
    def device(self):
        return self.emb.device

    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False, replace_position=None,
                output_hidden_states=False, return_dict=True):
        T = x.shape[1]
        if past_key_values is None:
            positions = torch.arange(T)
        elif replace_position is None:
            past = past_key_values[0][0].shape[2]
            positions = torch.arange(past, past + T)
        else:
            positions = replace_position[0].nonzero(as_tuple=True)[0]
            assert positions.numel() == T
        h = self.emb[x % self.vocab] + self.pos[positions]
        hs, present = [h], []
        for li, w in enumerate(self.w):
            q, k, v = (h @ w[0]).unsqueeze(1), (h @ w[1]).unsqueeze(1), (h @ w[2]).unsqueeze(1)
            if past_key_values is not None:
                pk, pv = past_key_values[li]
                if replace_position is None:
                    k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                else:
                    pk[:, :, positions] = k
                    pv[:, :, positions] = v
                    k, v = pk, pv
            present.append((k, v))
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dim), -1) @ v
            h = torch.tanh(h + att.squeeze(1) @ w[3])
            hs.append(h)
        return SimpleNamespace(logits=h @ self.out, past_key_values=present if use_cache else None,
                               hidden_states=tuple(hs) if output_hidden_states else None)


class ExactCacheModel(torch.nn.Module):
    """The "cache" is a snapshot of the token ids; a cached call rebuilds the full sequence from snapshot + window
    and returns the window slice of a full forward. Cached logits therefore equal no-cache logits exactly as long
    as the snapshot is not stale. Records every call: (kind, full sequence it was computed on)."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.calls = inner, []

    @property
    def device(self):
        return self.inner.device

    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False, replace_position=None,
                output_hidden_states=False, return_dict=True):
        if past_key_values is None:
            full = x.clone()
            out = self.inner(full, output_hidden_states=output_hidden_states)
            snap = full.clone().view(1, 1, -1, 1)
            self.calls.append(("build" if use_cache else "full", full.clone()))
            return SimpleNamespace(logits=out.logits, hidden_states=out.hidden_states,
                                   past_key_values=[(snap, snap)] if use_cache else None)
        snap = past_key_values[0][0][0, 0, :, 0]
        if replace_position is None:
            full, off = torch.cat([snap, x[0]]), snap.numel()
        else:
            idx = replace_position[0].nonzero(as_tuple=True)[0]
            snap[idx] = x[0]                       # in place, like LLaDA's replace_position
            full, off = snap.clone(), int(idx[0])
        out = self.inner(full.unsqueeze(0))
        self.calls.append(("cached", full.unsqueeze(0).clone()))
        return SimpleNamespace(logits=out.logits[:, off:off + x.shape[1]], hidden_states=None,
                               past_key_values=past_key_values)


@pytest.fixture
def cache_model():
    return FakeCacheModel().eval()


# --------------------------------------------------------------------------- written-out Fast-dLLM references

def _transfer(logits, temperature, mask_index, x, tau):
    """Fast-dLLM get_transfer_index, threshold branch."""
    x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
    p = torch.gather(F.softmax(logits.to(torch.float64), -1), -1, x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(mask_index, x0, x)
    conf = torch.where(mask_index, p, torch.tensor(torch.finfo(p.dtype).min, dtype=p.dtype))
    tr = mask_index & (conf >= tau)
    tr[0, torch.argmax(conf[0])] = True
    return x0, tr & mask_index


def ref_prefix(model, prompt, gen, block, temperature, tau):
    x = torch.full((1, prompt.shape[1] + gen), MASK_ID, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    nfe = 0
    for nb in range(gen // block):
        s = prompt.shape[1] + nb * block
        e = s + block
        out = model(x, use_cache=True)
        mi = x == MASK_ID
        mi[:, e:] = False
        x0, tr = _transfer(out.logits, temperature, mi, x, tau)
        x[tr] = x0[tr]
        pkv = [tuple(t[:, :, :s] for t in layer) for layer in out.past_key_values]
        nfe += 1
        while (x[:, s:e] == MASK_ID).any():
            nfe += 1
            mi = x[:, s:] == MASK_ID
            mi[:, block:] = False
            logits = model(x[:, s:], past_key_values=pkv, use_cache=True).logits
            combined.add_gumbel_noise(logits, temperature)          # Fast-dLLM's extra (unused) draw
            x0, tr = _transfer(logits, temperature, mi, x[:, s:], tau)
            x[:, s:][tr] = x0[tr]
    return x, nfe


def ref_dual(model, prompt, gen, block, temperature, tau):
    """steps = gen_length (as in Fast-dLLM's eval scripts) -> steps_per_block = block."""
    x = torch.full((1, prompt.shape[1] + gen), MASK_ID, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    nfe = 0
    for nb in range(gen // block):
        s = prompt.shape[1] + nb * block
        e = s + block
        out = model(x, use_cache=True)
        pkv = out.past_key_values
        nfe += 1
        rp = torch.zeros_like(x, dtype=torch.bool)
        rp[:, s:e] = True
        mi = x == MASK_ID
        mi[:, e:] = False
        x0, tr = _transfer(out.logits, temperature, mi, x, tau)
        x = torch.where(tr, x0, x)
        for _ in range(1, block):
            if not (x[:, s:e] == MASK_ID).any():
                break
            lb = model(x[:, s:e], past_key_values=pkv, use_cache=True, replace_position=rp).logits
            x0b, trb = _transfer(lb, temperature, x[:, s:e] == MASK_ID, x[:, s:e], tau)
            x = torch.cat([x[:, :s], torch.where(trb, x0b, x[:, s:e]), x[:, e:]], dim=1)
            nfe += 1
    return x, nfe


def _run(model, prompt, gen, block, temperature, tau, seed, cache, tok=None, defense=False, **kw):
    if defense:
        kw = dict(dict(alpha0=0.3, sp_threshold=0.1, refinement_steps=8, baseline_hidden=BASE_HIDDEN,
                       cue_ids=combined.cue_id_sets(tok)), **kw)
    torch.manual_seed(seed)
    return combined.generate(model, prompt, gen_length=gen, block_length=block, temperature=temperature,
                             mask_id=MASK_ID, mode="threshold", threshold=tau, cache=cache, defense=defense, **kw)


# --------------------------------------------------------------------------- 1. defense off == Fast-dLLM

REFS = {"prefix": ref_prefix, "dual": ref_dual}


@pytest.mark.parametrize("cache", ["prefix", "dual"])
@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_equals_written_out_fast_dllm(cache_model, cache, gen, block, temperature):
    for tau in TAUS:
        for seed in range(3):
            prompt = _prompt(seed)
            torch.manual_seed(seed)
            ref, nfe = REFS[cache](cache_model, prompt, gen, block, temperature, tau)
            out, st = _run(cache_model, prompt, gen, block, temperature, tau, seed, cache)
            assert torch.equal(out, ref), (tau, seed)
            assert st.nfe_gen == nfe and st.nfe_cached == nfe - gen // block and st.nfe_repair == 0
            assert not (out == MASK_ID).any()


@pytest.mark.parametrize("cache", ["prefix", "dual"])
@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_equals_real_fast_dllm(fast_dllm, cache_model, cache, gen, block, temperature):
    fn = {"prefix": fast_dllm.generate_with_prefix_cache, "dual": fast_dllm.generate_with_dual_cache}[cache]
    for tau in TAUS:
        for seed in range(3):
            prompt = _prompt(seed)
            torch.manual_seed(seed)
            ref, nfe = fn(cache_model, prompt, steps=gen, gen_length=gen, block_length=block,
                          temperature=temperature, mask_id=MASK_ID, threshold=tau)
            out, st = _run(cache_model, prompt, gen, block, temperature, tau, seed, cache)
            assert torch.equal(out, ref), (tau, seed)
            assert st.nfe_gen == nfe


def test_the_fake_cache_is_not_exact(cache_model):
    """Guard against vacuous tests: on this model a cached forward differs from a full one, so prefix / dual
    decode differently from no cache for some inputs."""
    differ = {"prefix": 0, "dual": 0}
    for seed in range(6):
        prompt = _prompt(seed)
        base, _ = _run(cache_model, prompt, 64, 16, 0.0, 1.01, seed, None)
        for c in differ:
            out, _ = _run(cache_model, prompt, 64, 16, 0.0, 1.01, seed, c)
            differ[c] += not torch.equal(out, base)
    assert differ["prefix"] > 0 and differ["dual"] > 0


# --------------------------------------------------------------------------- 2. plumbing with an exact cache

@pytest.mark.parametrize("cache", ["prefix", "dual"])
@pytest.mark.parametrize("gen,block", CASES)
@pytest.mark.parametrize("defense,lam", [(False, None), (True, 10.0), (True, 0.0)])
@pytest.mark.parametrize("hidden", [True, False])
def test_exact_cache_equals_no_cache(fake_tokenizer, cache, gen, block, defense, lam, hidden):
    """Temperature 0: the only randomness is R (whole-sequence draws) and the repair, identical in all modes."""
    if not defense and not hidden:
        pytest.skip("hidden only matters with the defense")
    kw = {} if not defense else dict(sp_threshold=lam, baseline_hidden=BASE_HIDDEN if hidden else None)
    for tau in TAUS:
        for seed in range(3):
            prompt = _prompt(seed)
            base, st0 = _run(ExactCacheModel(FakeModel().eval()), prompt, gen, block, 0.0, tau, seed, None,
                             fake_tokenizer, defense, **kw)
            out, st = _run(ExactCacheModel(FakeModel().eval()), prompt, gen, block, 0.0, tau, seed, cache,
                           fake_tokenizer, defense, **kw)
            assert torch.equal(out, base), (tau, seed)
            assert (st.nfe_gen, st.nfe_repair) == (st0.nfe_gen, st0.nfe_repair)
            assert st.nfe_cached == st.nfe_gen - gen // block
            assert st.alphas == st0.alphas
            # the logit score is a float32 mean over the window (shorter with a cache): same values, other
            # summation order -> equal up to rounding
            if st0.audit_logit is None:
                assert st.audit_logit is None
            else:
                assert st.audit_logit == pytest.approx(st0.audit_logit, rel=1e-6, abs=1e-9)
            assert (st.audit_hidden, st.audit_unsafe) == (st0.audit_hidden, st0.audit_unsafe)
            if defense and lam == 0.0:
                assert st.repaired and st.nfe_repair == 8


# --------------------------------------------------------------------------- 3. cache consistency after repair

@pytest.mark.parametrize("cache", ["prefix", "dual"])
def test_cache_is_rebuilt_from_repaired_tokens(fake_tokenizer, cache):
    gen, block = 32, 8
    changed = 0
    for seed in range(6):
        prompt = _prompt(seed)
        P = prompt.shape[1]
        m = ExactCacheModel(FakeModel().eval())
        out, st = _run(m, prompt, gen, block, 0.0, 0.1, seed, cache, fake_tokenizer, True, sp_threshold=0.0)
        kinds = [k for k, _ in m.calls]
        assert kinds.count("full") == st.nfe_repair == 8
        first_r = kinds.index("full")
        assert kinds[first_r:first_r + 8] == ["full"] * 8                # repair = 8 full forwards in a row
        nxt = first_r + 8
        assert kinds[nxt] == "build"                                      # next block starts with a fresh cache
        built = m.calls[nxt][1]
        assert torch.equal(built[:, P:P + block], out[:, P:P + block])    # ... built on the repaired block 0
        assert kinds.count("build") == gen // block
        # every cached call sees the current tokens: block 0 in later cached calls == final repaired block 0
        for kind, full in m.calls[nxt:]:
            assert torch.equal(full[:, P:P + block], out[:, P:P + block])
        # block 0 before repair = the same run with the audit switched off (identical up to the audit at T=0)
        pre, _ = _run(ExactCacheModel(FakeModel().eval()), prompt, gen, block, 0.0, 0.1, seed, cache,
                      fake_tokenizer, True, sp_threshold=10.0)
        changed += not torch.equal(pre[:, P:P + block], out[:, P:P + block])
    assert changed > 0                                                    # repair really changed block 0


def test_defense_with_real_cache_runs(cache_model, fake_tokenizer):
    for cache in ("prefix", "dual"):
        for lam in (0.0, 10.0):
            out, st = _run(cache_model, _prompt(0), 64, 16, 0.5, 0.1, 0, cache, fake_tokenizer, True,
                           sp_threshold=lam)
            assert not (out == MASK_ID).any() and st.nfe_cached > 0
            assert st.nfe_repair == (8 if lam == 0.0 else 0)


def test_guards(cache_model):
    p = _prompt(0)
    with pytest.raises(ValueError):
        combined.generate(cache_model, p, gen_length=32, block_length=8, steps=32, mode="fixed", cache="prefix")
    with pytest.raises(ValueError):
        combined.generate(cache_model, p, gen_length=32, block_length=8, mode="threshold", threshold=0.9,
                          cache="full")
    with pytest.raises(ValueError):
        am = torch.ones_like(p)
        am[0, 0] = 0
        combined.generate(cache_model, p, gen_length=32, block_length=8, mode="threshold", threshold=0.9,
                          cache="dual", attention_mask=am)
    # an all-ones mask is accepted and dropped (Fast-dLLM passes none)
    torch.manual_seed(0)
    a, _ = combined.generate(cache_model, p, gen_length=32, block_length=8, mode="threshold", threshold=0.9,
                             cache="dual", attention_mask=torch.ones_like(p))
    torch.manual_seed(0)
    b, _ = combined.generate(cache_model, p, gen_length=32, block_length=8, mode="threshold", threshold=0.9,
                             cache="dual")
    assert torch.equal(a, b)

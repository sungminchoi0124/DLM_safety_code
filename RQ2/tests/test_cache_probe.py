"""Cache integrity checks must reject stale prefixes and wrong cached windows."""
from types import SimpleNamespace
import pytest
import torch
from RQ2.cache_probe import CacheProbe

MASK = 126336


def simulate(mode, defect=None):
    prompt = torch.tensor([[2, MASK, 3, MASK, 4]])
    p, block, total = 5, 32, 133
    probe = CacheProbe(prompt, mode)
    probe.hook(None, (torch.tensor([[2, 3]]),), {})
    probe.phase = 'decode'
    x = torch.full((1, total), MASK)
    x[:, :p] = prompt
    probe.hook(None, (x,), {})  # one simulated generation call fills block 0
    eligible = torch.cat([(prompt[0] == MASK).nonzero().flatten(), torch.arange(p, p+block)])
    x[0, eligible] = 7
    n = 29 if defect == 'repair_count' else 30  # floor(.9*(2+32))
    x[0, eligible[:n]] = MASK
    for _ in range(8):
        probe.hook(None, (x,), {})
        masks = eligible[x[0, eligible] == MASK]
        x[0, masks[:4]] = 7
    for b in range(1, 4):
        if defect == 'stale_prefix' and b == 1:
            x[0, p] = MASK
        probe.hook(None, (x,), dict(use_cache=True))
        if defect == 'stale_prefix' and b == 1:
            x[0, p] = 7
        if b == 1:
            length = p+b*block if mode == 'prefix' else total
            if defect == 'kv_length':
                length -= 1
            kv = [(torch.empty(1, 1, length, 2), torch.empty(1, 1, length, 2))]
            kw = dict(use_cache=True, past_key_values=kv)
            if mode == 'dual':
                replace = torch.zeros((1, total), dtype=torch.bool)
                replace[:, p+b*block:p+(b+1)*block] = True
                if defect == 'replace_span':
                    replace[0, 0] = True
                kw['replace_position'] = replace
                view = x[:, p+b*block:p+(b+1)*block]
            else:
                view = x[:, p+b*block:]
            if defect == 'input_window':
                view = view[:, :-1]
            probe.hook(None, (view,), kw)
        x[:, p+b*block:p+(b+1)*block] = 7
    stats = SimpleNamespace(nfe_gen=5, nfe_repair=8, nfe_cached=1, repaired=True)
    result = probe.finish(x, stats)
    assert probe.calls == 14  # baseline 1 + gen 5 + repair 8
    return result


@pytest.mark.parametrize('mode', ['prefix', 'dual'])
def test_post_repair_cache_is_observed(mode):
    result = simulate(mode)
    assert result['build_nfe'] == 3 and result['reuse_nfe'] == 1
    assert result['repaired_prefix_verified']
    assert len(set(result['build_prefix_sha256'])) == 1
    assert result['repair_mask_counts'][0] == 30


@pytest.mark.parametrize('defect', ['repair_count', 'stale_prefix', 'kv_length', 'input_window'])
def test_bad_prefix_or_repair_is_rejected(defect):
    with pytest.raises(RuntimeError):
        simulate('prefix', defect)


def test_dual_replace_outside_active_block_is_rejected():
    with pytest.raises(RuntimeError):
        simulate('dual', 'replace_span')


def test_no_cache_counts_calls_without_snapshots():
    p = CacheProbe(torch.tensor([[2, MASK]]), None)
    p.hook(None, (), {})
    p.phase = 'decode'
    p.hook(None, (), {})
    assert p.calls == 2
    assert p.finish(torch.tensor([[2, 3]]), SimpleNamespace(nfe_cached=0)) == dict(enabled=False, build_nfe=0, reuse_nfe=0)


def test_stalled_decoder_cannot_run_unbounded():
    p = CacheProbe(torch.tensor([[2, MASK]]), None)
    p.hook(None, (), {})
    p.phase = 'decode'
    with pytest.raises(RuntimeError, match='completion bound exceeded'):
        for _ in range(200):
            p.hook(None, (), {})

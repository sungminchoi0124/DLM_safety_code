"""Observe cache calls without changing decoding or adding model forwards."""
from collections import deque
import hashlib

import torch


class CacheProbe:
    def __init__(self, prompt, mode, block_length=32, gen_length=128, mask_id=126336):
        self.mode, self.p = mode, prompt.shape[1]
        self.block, self.gen, self.mask = block_length, gen_length, mask_id
        self.initial_masks = int((prompt == mask_id).sum()) + block_length
        self.phase, self.calls = 'baseline', 0
        self.builds = self.reuses = self.uncached = 0
        self.recent = deque(maxlen=8)
        self.repair_snapshots, self.prefixes, self.replacements, self.events = [], [], [], []

    def hook(self, module, args, kwargs):
        self.calls += 1
        # Each threshold forward must fill at least one eligible position. Include
        # one baseline and at most eight repair forwards; abort a stalled decoder.
        if self.phase == 'decode' and self.calls > self.initial_masks-self.block+self.gen+9:
            raise RuntimeError('Forward completion bound exceeded; decoder made insufficient progress')
        if self.phase != 'decode' or self.mode is None:
            return
        x = args[0] if args else kwargs['input_ids']
        use_cache, kv = kwargs.get('use_cache', False), kwargs.get('past_key_values')
        if not use_cache:
            if self.builds or kv is not None or x.shape[1] != self.p + self.gen:
                raise RuntimeError('Unexpected uncached forward after cache construction')
            self.uncached += 1
            self.recent.append(x[:, :self.p+self.block].detach().clone())
            return
        if kv is None:
            self.builds += 1
            if self.builds >= self.gen // self.block or x.shape[1] != self.p + self.gen:
                raise RuntimeError('Cache must be rebuilt on the full input at each later block')
            if self.builds == 1:
                self.repair_snapshots = list(self.recent)
            self.prefixes.append(x[:, :self.p+self.block].detach().clone())
            self.events.append(dict(kind='build', block=self.builds, input_tokens=x.shape[1]))
            return
        if not self.builds:
            raise RuntimeError('KV reused before its first full-input build')
        b = self.builds
        expected_kv = self.p + b*self.block if self.mode == 'prefix' else self.p+self.gen
        if any(len(layer) != 2 or any(t.ndim != 4 or t.shape[2] != expected_kv for t in layer) for layer in kv):
            raise RuntimeError('Cached KV length differs from the expected block layout')
        expected_input = self.gen-b*self.block if self.mode == 'prefix' else self.block
        if x.shape[1] != expected_input:
            raise RuntimeError('Cached input window is incorrect')
        if self.mode == 'dual':
            replacement = kwargs.get('replace_position')
            if replacement is None:
                raise RuntimeError('Dual reuse needs replace_position')
            self.replacements.append((b, replacement.detach().clone()))
        elif kwargs.get('replace_position') is not None:
            raise RuntimeError('Prefix reuse unexpectedly has a Dual replacement mask')
        self.reuses += 1
        self.events.append(dict(kind='reuse', block=b, input_tokens=x.shape[1], kv_tokens=expected_kv))

    def finish(self, tokens, stats, remask_ratio=0.9):
        if self.mode is None:
            if stats.nfe_cached:
                raise RuntimeError('No-cache condition reported cached forwards')
            return dict(enabled=False, build_nfe=0, reuse_nfe=0)
        if self.builds != self.gen//self.block-1 or self.reuses != stats.nfe_cached:
            raise RuntimeError('Observed cache builds/reuses disagree with decoder statistics')
        if stats.nfe_gen != self.uncached-stats.nfe_repair+self.builds+self.reuses:
            raise RuntimeError('Generation/repair/cache call split does not sum correctly')
        final_prefix = tokens[:, :self.p+self.block].detach().cpu()
        if (final_prefix == self.mask).any():
            raise RuntimeError('First block is not complete')
        hashes = []
        for prefix in self.prefixes:
            prefix = prefix.cpu()
            if not torch.equal(prefix, final_prefix):
                raise RuntimeError('A cache build did not use the final repaired first-block tokens')
            hashes.append(hashlib.sha256(prefix.numpy().tobytes()).hexdigest())
        repair_counts = []
        if stats.repaired:
            if stats.nfe_repair != 8 or len(self.repair_snapshots) != 8:
                raise RuntimeError('Repair must use eight uncached forwards before the first cache build')
            repair_counts = [int((x == self.mask).sum()) for x in self.repair_snapshots]
            if repair_counts[0] != int(self.initial_masks*remask_ratio):
                raise RuntimeError('Repair remask count or ordering before cache build is incorrect')
            if any(a < b for a,b in zip(repair_counts, repair_counts[1:])):
                raise RuntimeError('Repair did not fill its first-block masks monotonically')
        for b, replacement in self.replacements:
            replacement = replacement.cpu()
            expected = torch.zeros_like(tokens, device='cpu', dtype=torch.bool)
            expected[:, self.p+b*self.block:self.p+(b+1)*self.block] = True
            if replacement.dtype != torch.bool or not torch.equal(replacement, expected):
                raise RuntimeError('Dual replace_position must cover exactly the current 32 positions')
        return dict(enabled=True, mode=self.mode, build_nfe=self.builds, reuse_nfe=self.reuses,
                    uncached_block0_gen_nfe=self.uncached-stats.nfe_repair,
                    repair_before_cache_forwards=stats.nfe_repair,
                    repaired_prefix_verified=bool(stats.repaired),
                    build_prefix_sha256=hashes, repair_mask_counts=repair_counts,
                    dual_replace_verified=len(self.replacements), events=self.events)

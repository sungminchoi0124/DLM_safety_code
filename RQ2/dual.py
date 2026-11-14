"""Condition E: uncached first block, then upstream-style Dual KV cache."""
import time
import torch
from .types import DecodeResult


def full_cache(cache, length):
    if not cache:
        raise RuntimeError('Model did not return Dual KV cache')
    if any(len(layer)!=2 or any(t.ndim!=4 or t.shape[2]!=length for t in layer) for layer in cache):
        raise RuntimeError('Invalid Dual KV shape')
    return cache


@torch.no_grad()
def generate_dual(model, prompt, config, upstream):
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError('Batch size must be one')
    if prompt.device.type == 'cuda':
        torch.cuda.synchronize(prompt.device)
    started = time.perf_counter()
    p = prompt.shape[1]
    x = torch.full((1, p + config.gen_length), config.mask_id, dtype=torch.long, device=prompt.device)
    x[:, :p] = prompt
    blocks = []
    nfe = 0
    for b in range(config.gen_length // config.block_length):
        tail_start = p + b * config.block_length
        start = 0 if b == 0 else tail_start
        end = tail_start + config.block_length
        step_cap = int((x[:, start:end] == config.mask_id).sum())
        cache = None
        replace_position = None
        block_nfe = reuse_nfe = build_nfe = 0
        for i in range(step_cap):
            if b == 0:
                view = x
                logits = model(x).logits
                prefix_length = 0
            elif i == 0:
                view = x
                out = model(x, use_cache=True)
                logits = out.logits
                cache = full_cache(out.past_key_values, x.shape[1])
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, tail_start:end] = True
                prefix_length = tail_start
                build_nfe += 1
            else:
                view = x[:, tail_start:end]
                logits = model(view, past_key_values=cache, use_cache=True,
                               replace_position=replace_position).logits
                prefix_length = tail_start
                reuse_nfe += 1
            nfe += 1
            block_nfe += 1
            masks = view == config.mask_id
            if b == 0 or i == 0:
                masks[:, :start] = False
                masks[:, end:] = False
            else:
                masks[:, config.block_length:] = False
            proposed, transfer = upstream.get_transfer_index(
                logits, config.temperature, 'low_confidence', masks, view, None, threshold=config.threshold)
            if (transfer & (proposed == config.mask_id)).any():
                raise RuntimeError('Model proposed mask token')
            if not transfer.any():
                raise RuntimeError('Threshold decoder made no progress')
            view[transfer] = proposed[transfer]
            if not (x[:, start:end] == config.mask_id).any():
                break
        entry = {'block':b, 'gen_nfe':block_nfe, 'step_cap':step_cap,
                 'prompt_masks_remaining':int((x[:, :p] == config.mask_id).sum()),
                 'block_masks_remaining':int((x[:, tail_start:end] == config.mask_id).sum()),
                 'generation_masks_remaining':int((x[:, p:] == config.mask_id).sum()),
                 'cache_enabled':b > 0, 'cache_build_nfe':build_nfe, 'cache_reuse_nfe':reuse_nfe,
                 'cache_prefix_length':prefix_length, 'full_input_tokens':x.shape[1],
                 'cached_input_tokens':config.block_length if b > 0 else 0,
                 'cache_total_tokens':x.shape[1] if b > 0 else 0,
                 'cache_suffix_length':x.shape[1]-end if b > 0 else 0,
                 'replace_positions':int(replace_position.sum()) if b > 0 else 0,
                 'dual_replace_nfe':reuse_nfe}
        if entry['prompt_masks_remaining'] or entry['block_masks_remaining']:
            raise RuntimeError('Block ended with unfinished masks')
        blocks.append(entry)
        del cache
    if prompt.device.type == 'cuda':
        torch.cuda.synchronize(prompt.device)
    return DecodeResult(token_ids=x, n_masks=int((x == config.mask_id).sum()),
                        prompt_masks_remaining=int((x[:, :p] == config.mask_id).sum()),
                        generation_masks_remaining=int((x[:, p:] == config.mask_id).sum()),
                        gen_nfe=nfe, wall_ms=1000*(time.perf_counter()-started), blocks=blocks)

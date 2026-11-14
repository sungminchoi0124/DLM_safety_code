"""Condition D: decode block zero without cache, then upstream-style Prefix KV cache."""
import time
import torch
from .types import DecodeResult


def trim_prefix(cache, length):
    if not cache:
        raise RuntimeError('Model did not return Prefix KV cache')
    result = []
    for layer in cache:
        if len(layer) != 2 or any(t.ndim != 4 or t.shape[2] < length for t in layer):
            raise RuntimeError('Invalid Prefix KV shape')
        result.append(tuple(t[:, :, :length] for t in layer))
    return result


@torch.no_grad()
def generate_prefix(model, prompt, config, upstream):
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
                cache = trim_prefix(out.past_key_values, tail_start)
                prefix_length = tail_start
                build_nfe += 1
            else:
                view = x[:, tail_start:]
                logits = model(view, past_key_values=cache, use_cache=True).logits
                prefix_length = tail_start
                reuse_nfe += 1
                # Preserve the pinned upstream Prefix kernel's extra sampling RNG draw.
                # Its proposal is unused, but removing the draw changes temperature>0 outputs.
                unused = torch.argmax(upstream.add_gumbel_noise(logits, config.temperature), dim=-1)
                del unused
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
                 'cached_input_tokens':x.shape[1]-tail_start if b > 0 else 0}
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

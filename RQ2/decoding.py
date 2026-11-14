"""Without 0/A; Fast-dLLM selection with an explicit DIJA first-block extension."""
import time
import torch
from .types import DecodeConfig, DecodeResult

@torch.no_grad()
def generate_fast(model, prompt, config: DecodeConfig, upstream):
    config.validate()
    if config.cache == "dual":
        from .dual import generate_dual
        return generate_dual(model, prompt, config, upstream)
    if config.cache == "prefix":
        from .prefix import generate_prefix
        return generate_prefix(model, prompt, config, upstream)
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("Batch size must be one")
    if prompt.device.type == "cuda":
        torch.cuda.synchronize(prompt.device)
    started = time.perf_counter()
    p = prompt.shape[1]
    x = torch.full((1, p + config.gen_length), config.mask_id, dtype=torch.long, device=prompt.device)
    x[:, :p] = prompt
    steps_per_block = config.steps // (config.gen_length // config.block_length)
    blocks = []
    nfe = 0
    for b in range(config.gen_length // config.block_length):
        tail_start = p + b * config.block_length
        start = 0 if b == 0 else tail_start
        end = tail_start + config.block_length
        plan = upstream.get_num_transfer_tokens(x[:, start:end] == config.mask_id, steps_per_block)
        # A transfers at least one eligible mask per call. Its initial mask count is a
        # finite completion bound, including the prompt masks in the first block.
        step_cap = steps_per_block if config.threshold is None else int((x[:, start:end] == config.mask_id).sum())
        block_nfe = 0
        for i in range(step_cap):
            masks = x == config.mask_id
            masks[:, :start] = False
            masks[:, end:] = False
            logits = model(x).logits
            nfe += 1
            block_nfe += 1
            proposed, transfer = upstream.get_transfer_index(
                logits, config.temperature, "low_confidence", masks, x,
                plan[:, i] if config.threshold is None else None, threshold=config.threshold)
            if (transfer & (proposed == config.mask_id)).any():
                raise RuntimeError("Model proposed mask token; completion cannot be guaranteed")
            x[transfer] = proposed[transfer]
            if config.threshold is not None:
                if not transfer.any():
                    raise RuntimeError("Threshold decoder made no progress")
                if not (x[:, start:end] == config.mask_id).any():
                    break
        entry = {"block": b, "gen_nfe": block_nfe, "step_cap": step_cap,
                 "prompt_masks_remaining": int((x[:, :p] == config.mask_id).sum()),
                 "block_masks_remaining": int((x[:, tail_start:end] == config.mask_id).sum()),
                 "generation_masks_remaining": int((x[:, p:] == config.mask_id).sum())}
        blocks.append(entry)
        if entry["prompt_masks_remaining"] or entry["block_masks_remaining"]:
            raise RuntimeError("Block ended with unfinished masks")
    if prompt.device.type == "cuda":
        torch.cuda.synchronize(prompt.device)
    return DecodeResult(token_ids=x, n_masks=int((x == config.mask_id).sum()),
                        prompt_masks_remaining=int((x[:, :p] == config.mask_id).sum()),
                        generation_masks_remaining=int((x[:, p:] == config.mask_id).sum()),
                        gen_nfe=nfe, wall_ms=1000 * (time.perf_counter() - started), blocks=blocks)

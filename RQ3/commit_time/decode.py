"""Standalone LLaDA decoder for the trace runs (no KV cache, one block covering every mask).

rule="quota"      the vanilla path of DiffuGuard's generate() (low_confidence, cfg 0, fill_all_masks, one block):
                  same RNG calls in the same order, so with the same seed it returns the same tokens
                  (tests/test_decode.py checks this against the original function).
rule="threshold"  Fast-dLLM's parallel rule (get_transfer_index with threshold): commit every masked position
                  whose confidence >= tau, and always the most confident one. Runs until no mask is left.
                  Confidence is computed in float32 (the quota path keeps the original model-dtype softmax).

Snapshots / resume: `snapshot_at` keeps the state x_s before step s in memory (never written to disk, it
holds model text). `decode(..., resume=snap, seed=k)` reseeds the RNG and continues from x_s; this is how
the behavioural commit test branches continuations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    # identical to LLaDA / DiffuGuard / Fast-dLLM
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    # identical to DiffuGuard's utility/generate_function_llada.py
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = max(int(steps), 1)
    base = mask_num // steps
    remainder = (mask_num % steps).squeeze(1)
    num = base.expand(-1, steps).clone().to(torch.int64)
    idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    num += (idx < remainder.unsqueeze(1)).to(torch.int64)
    return num


@dataclass
class Snapshot:
    step: int
    x: torch.Tensor
    schedule: Optional[torch.Tensor] = None


@dataclass
class DecodeResult:
    x: torch.Tensor
    nfe: int
    snapshots: dict = field(default_factory=dict)


def _align_attention_mask(attention_mask, L, device):
    if attention_mask is None:
        return None
    am = attention_mask.to(device)
    if am.dtype != torch.bool:
        am = am != 0
    if am.shape[1] < L:
        am = torch.cat([am, torch.ones(am.size(0), L - am.shape[1], dtype=torch.bool, device=device)], dim=1)
    return am[:, :L]


@torch.no_grad()
def decode(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, *,
           gen_length: int, steps: int, temperature: float, mask_id: int = 126336,
           rule: str = "quota", threshold: Optional[float] = None,
           snapshot_at: Iterable[int] = (), resume: Optional[Snapshot] = None, seed: Optional[int] = None,
           max_forwards: Optional[int] = None) -> DecodeResult:
    if resume is None and input_ids.shape[0] != 1:
        raise ValueError("batch size 1 only")
    if rule not in ("quota", "threshold"):
        raise ValueError(rule)
    if rule == "threshold" and threshold is None:
        raise ValueError("threshold rule needs a threshold")
    device = next(model.parameters()).device
    snapshot_at = set(int(s) for s in snapshot_at)

    if resume is None:
        x = input_ids.clone().to(device)
        if gen_length > 0:
            x = torch.cat([x, torch.full((1, int(gen_length)), mask_id, dtype=torch.long, device=device)], dim=1)
        start = 0
        schedule = None
    else:
        x = resume.x.clone().to(device)
        start = int(resume.step)
        schedule = resume.schedule
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    am = _align_attention_mask(attention_mask, x.shape[1], device)

    # one block covering every mask (DiffuGuard fill_all_masks / "safety guard" expansion)
    if schedule is None and rule == "quota":
        schedule = get_num_transfer_tokens(x == mask_id, steps)
    n_masks0 = int((x == mask_id).sum())
    if max_forwards is None:
        max_forwards = steps if rule == "quota" else n_masks0 + start
    snaps, nfe, i = {}, 0, start
    while True:
        if rule == "quota" and i >= schedule.shape[1]:
            break
        if rule == "threshold" and (not bool((x == mask_id).any()) or i >= max_forwards):
            break
        if i in snapshot_at:
            snaps[i] = Snapshot(step=i, x=x.clone(), schedule=schedule)
        mask_index = x == mask_id
        logits = model(x, attention_mask=am, output_hidden_states=False, return_dict=True).logits
        nfe += 1
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        if rule == "quota":
            p = F.softmax(logits, dim=-1)
            conf = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            _ = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)  # the original draws R every step
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, conf, -np.inf)
            k = int(schedule[0, i].item())
            transfer = torch.zeros_like(x, dtype=torch.bool)
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                transfer[0, sel] = True
        else:
            p = F.softmax(logits.float(), dim=-1)
            conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, conf, torch.tensor(-1.0, device=conf.device))
            transfer = mask_index & (confidence >= threshold)
            transfer[0, int(torch.argmax(confidence[0]))] = True
            transfer &= mask_index
        x[transfer] = x0[transfer]
        i += 1
    return DecodeResult(x=x, nfe=nfe, snapshots=snaps)


def generate_response_like(jbl, vanilla_prompt: str, prompt: str, tokenizer, model, args, *,
                           rule: str, threshold: Optional[float] = None, **decode_kw):
    """Mirror of the original models/jailbreakbench_llada.generate_response (no system prompt) on decode()."""
    if getattr(args, "cfg_scale", 0.0) not in (0, 0.0):
        raise SystemExit("trace decoder: cfg_scale must be 0")
    if getattr(args, "sp_mode", "off") != "off" or getattr(args, "remasking", "low_confidence") != "low_confidence":
        raise SystemExit("trace decoder replaces the vanilla path only (sp_mode off, remasking low_confidence); "
                         "use --decoder original for DiffuGuard")
    if int(args.gen_length) != int(args.block_length):
        raise SystemExit("trace decoder: one block only (gen_length == block_length)")
    input_ids, attention_mask = jbl.get_tokenized_input(prompt, tokenizer, model.device)
    vanilla_ids, _ = jbl.get_tokenized_input(vanilla_prompt, tokenizer, model.device)
    matching = min(sum(a.item() == b.item() for a, b in zip(vanilla_ids[0], input_ids[0])), len(vanilla_ids[0]))
    res = decode(model, input_ids, attention_mask, gen_length=args.gen_length, steps=args.steps,
                 temperature=args.temperature, mask_id=args.mask_id, rule=rule, threshold=threshold, **decode_kw)
    return postprocess(tokenizer, res.x, matching, args), res, input_ids.shape[1], matching


def postprocess(tokenizer, x, matching, args):
    text = tokenizer.batch_decode(x[:, matching:], skip_special_tokens=True)[0]
    if getattr(args, "attack_method", "") == "DIJA":
        text = text.split("assistant\n")[0]
    return text

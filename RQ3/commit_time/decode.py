"""Standalone LLaDA decoder for the trace runs (no KV cache, one block covering every mask).

rule="quota"      the vanilla path of DiffuGuard's generate() (low_confidence, cfg 0, fill_all_masks, one block):
                  same RNG calls in the same order, so with the same seed it returns the same tokens
                  (tests/test_decode.py checks this against the original function).
rule="threshold"  Fast-dLLM's parallel rule (get_transfer_index with threshold): commit every masked position
                  whose confidence >= tau, and always the most confident one. Runs until no mask is left.
                  Confidence is computed in float32 (the quota path keeps the original model-dtype softmax).
rule="prophet"    Prophet early exit (pixeli99/Prophet generate_earlyexit.py, commit 460afe4) on top of the quota
                  rule. Every step, after the forward and before committing, the mean top-2 logit gap over the
                  answer region A = [answer_start, answer_start + answer_len) is compared with a phase threshold
                  (progress (i+1)/steps < 0.33: 7.5, < 0.67: 5.0, else 2.5; the official current_step is 1-based).
                  The gap is averaged over every position of A whether it is already committed or not (as in the
                  official code), from the raw logits (no Gumbel noise). If gap >= threshold, every remaining mask
                  of the sequence (prompt masks included) is filled with this step's x0 and decoding stops. Until
                  then the tokens are the quota rule's tokens (same RNG calls). Note: x0 is the Gumbel-noised
                  argmax, so at temperature > 0 the all-in fill is sampled, not greedy (official behaviour);
                  prophet_fill="argmax" is the noise-free variant for a sensitivity check.

Snapshots / resume: `snapshot_at` keeps the state x_s before step s in memory (never written to disk, it
holds model text). `decode(..., resume=snap, seed=k)` reseeds the RNG and continues from x_s; this is how
the behavioural commit test branches continuations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


PROPHET_THRESHOLDS = (7.5, 5.0, 2.5)   # official defaults: early / mid / late
PROPHET_PHASES = (0.33, 0.67)          # progress cut points


@dataclass
class Snapshot:
    step: int
    x: torch.Tensor
    schedule: Optional[torch.Tensor] = None
    prompt_len: Optional[int] = None   # start of the response tail (Prophet answer region)


@dataclass
class DecodeResult:
    x: torch.Tensor
    nfe: int
    snapshots: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)   # prophet: exit bookkeeping and per-forward gaps (no text)


def prophet_threshold(step_1based: int, steps: int, thresholds=PROPHET_THRESHOLDS, phases=PROPHET_PHASES):
    """should_early_exit() of the official code: progress = current_step / max_steps, current_step 1-based."""
    progress = step_1based / steps
    if progress < phases[0]:
        return float(thresholds[0])
    if progress < phases[1]:
        return float(thresholds[1])
    return float(thresholds[2])


def answer_gaps(logits, x, positions, mask_id):
    """Mean top-2 logit gap over `positions` of row 0: (all, committed in x, still masked in x); NaN if empty."""
    if len(positions) == 0:
        return float("nan"), float("nan"), float("nan")
    top2 = torch.topk(logits[0, positions].float(), k=2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    masked = x[0, positions] == mask_id
    g_all = float(gap.mean())
    g_c = float(gap[~masked].mean()) if bool((~masked).any()) else float("nan")
    g_m = float(gap[masked].mean()) if bool(masked.any()) else float("nan")
    return g_all, g_c, g_m


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
           max_forwards: Optional[int] = None,
           prophet_thresholds=PROPHET_THRESHOLDS, prophet_phases=PROPHET_PHASES,
           answer_len: int = 16, answer_start: Optional[int] = None, prophet_fill: str = "x0") -> DecodeResult:
    if resume is None and input_ids.shape[0] != 1:
        raise ValueError("batch size 1 only")
    if rule not in ("quota", "threshold", "prophet"):
        raise ValueError(rule)
    if prophet_fill not in ("x0", "argmax"):
        raise ValueError(prophet_fill)
    if rule == "threshold" and threshold is None:
        raise ValueError("threshold rule needs a threshold")
    device = next(model.parameters()).device
    snapshot_at = set(int(s) for s in snapshot_at)

    if resume is None:
        x = input_ids.clone().to(device)
        prompt_len = int(input_ids.shape[1])
        if gen_length > 0:
            x = torch.cat([x, torch.full((1, int(gen_length)), mask_id, dtype=torch.long, device=device)], dim=1)
        start = 0
        schedule = None
    else:
        x = resume.x.clone().to(device)
        start = int(resume.step)
        schedule = resume.schedule
        prompt_len = resume.prompt_len
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    am = _align_attention_mask(attention_mask, x.shape[1], device)

    # one block covering every mask (DiffuGuard fill_all_masks / "safety guard" expansion)
    quota_like = rule in ("quota", "prophet")
    if schedule is None and quota_like:
        schedule = get_num_transfer_tokens(x == mask_id, steps)
    n_masks0 = int((x == mask_id).sum())
    if max_forwards is None:
        max_forwards = steps if quota_like else n_masks0 + start
    info = {}
    if rule == "prophet":
        if answer_start is None:
            if prompt_len is None:
                raise ValueError("prophet: answer_start unknown (snapshot without prompt_len)")
            answer_start = prompt_len
        A = torch.arange(int(answer_start), min(int(answer_start) + int(answer_len), x.shape[1]), device=device)
        info = {"exited": False, "exit_step": None, "exit_reason": "completed", "exit_threshold": None,
                "exit_gap": None, "n_filled_at_exit": 0, "answer_start": int(answer_start),
                "answer_len": int(len(A)), "prophet_fill": prophet_fill,
                "thresholds": [float(t) for t in prophet_thresholds], "phases": [float(q) for q in prophet_phases],
                "gap_all": [], "gap_committed": [], "gap_masked": [], "gap_threshold": []}
    snaps, nfe, i = {}, 0, start
    while True:
        if quota_like and i >= schedule.shape[1]:
            break
        if rule == "threshold" and (not bool((x == mask_id).any()) or i >= max_forwards):
            break
        if i in snapshot_at:
            snaps[i] = Snapshot(step=i, x=x.clone(), schedule=schedule, prompt_len=prompt_len)
        mask_index = x == mask_id
        logits = model(x, attention_mask=am, output_hidden_states=False, return_dict=True).logits
        nfe += 1
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        if quota_like:
            p = F.softmax(logits, dim=-1)
            conf = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            _ = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)  # the original draws R every step
            if rule == "prophet":
                g_all, g_c, g_m = answer_gaps(logits, x, A, mask_id)
                thr = prophet_threshold(i + 1, steps, prophet_thresholds, prophet_phases)
                for key, val in (("gap_all", g_all), ("gap_committed", g_c), ("gap_masked", g_m),
                                 ("gap_threshold", thr)):
                    info[key].append(val)
                if g_all >= thr:   # NaN never exits
                    fill = x0 if prophet_fill == "x0" else torch.argmax(logits, dim=-1)
                    info.update(exited=True, exit_step=i + 1, exit_reason="gap", exit_threshold=thr,
                                exit_gap=g_all, n_filled_at_exit=int(mask_index.sum()))
                    x[mask_index] = fill[mask_index]
                    i += 1
                    break
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
    if rule == "prophet" and not info["exited"]:
        info["exit_step"] = i   # ran the full schedule (= steps for a fresh decode)
    return DecodeResult(x=x, nfe=nfe, snapshots=snaps, info=info)


def generate_response_like(jbl, vanilla_prompt: str, prompt: str, tokenizer, model, args, *,
                           rule: str, threshold: Optional[float] = None, **decode_kw):
    """decode_kw: snapshot_at, prophet_thresholds, prophet_phases, answer_len, prophet_fill, ..."""
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
    """DIJA / DiffuGuard convention: decode from the vanilla/refined common-prefix count, cut at 'assistant\\n'.
    For DIJA prompts this keeps the infilled user turn (the judged DIJA response) and drops the answer."""
    text = tokenizer.batch_decode(x[:, matching:], skip_special_tokens=True)[0]
    if getattr(args, "attack_method", "") == "DIJA":
        text = text.split("assistant\n")[0]
    return text


def tail_text(tokenizer, x, prompt_len, args):
    """RAEC convention (jailbreakbench_llada_refusal_method.py:434-438): only the appended assistant answer
    (positions >= prompt_len), same 'assistant\\n' cut for DIJA."""
    text = tokenizer.batch_decode(x[:, int(prompt_len):], skip_special_tokens=True)[0]
    if getattr(args, "attack_method", "") == "DIJA":
        text = text.split("assistant\n")[0]
    return text


def tail_json_path(gen_json) -> Path:
    """<run>.json -> <run>.tail.json (same restricted folder as the generation JSON)."""
    p = Path(gen_json)
    return p.with_name(p.name[:-5] + ".tail.json" if p.name.endswith(".json") else p.name + ".tail.json")

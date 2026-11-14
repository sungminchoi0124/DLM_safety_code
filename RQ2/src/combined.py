"""RQ2 combined decoder: DiffuGuard (defense) x Fast-dLLM (acceleration) in one block-wise loop.

Stages done:
  S1 mode="fixed" ("acceleration off"): reproduces the DiffuGuard original generate() run block by block
     (DiffuGuard/semi_ar.py) token for token, defense on or off, any temperature (tests/test_s1_fixed.py).
  S2 mode="threshold", defense off: reproduces Fast-dLLM v1/llada/generate.py generate(threshold=tau) token
     for token, any temperature, float64 confidence like Fast-dLLM (tests/test_s2_threshold.py).
  S3 mode="threshold", defense on: k = #{masked block positions with p(x0) >= tau} (min 1), commit the top-k by
     the SAR score (1-alpha)*p + alpha*R, alpha by progress; block-0 audit and repair as in S1, repair budget
     fixed (refinement_steps, default 8) and without tau (tests/test_s3_threshold_defense.py).
  S4 cache="prefix" | "dual" (threshold mode): defense off reproduces Fast-dLLM generate_with_prefix_cache /
     generate_with_dual_cache token for token; with a cache whose logits equal the full forward, every cache mode
     gives the no-cache tokens, defense and repair included (tests/test_s4_cache.py).

What is copied from the original (DiffuGuard 86aeb75, utility/generate_function_llada.py generate()), operation
for operation so that floats and the RNG stream match:
  * per-step: forward -> add_gumbel_noise -> argmax -> softmax confidence -> R = torch.rand (always drawn, the
    original draws it even for low_confidence) -> remasking score -> top-k inside the current block
  * adaptive_step (SAR): alpha = alpha0 * (1 - i / (steps_per_block - 1))
  * block-0 audit: logit conflict score on the last step's logits over the block, then the cosine distance of the
    first step's last-layer hidden mean over the block to baseline_hidden (either >= lambda -> unsafe)
  * repair: randperm remask of int(block * remask_ratio) positions, suppress the old token ids, refill over
    refinement_steps with RANDOM order when remasking != low_confidence (as in the original)

Deliberate differences from the original (none of them changes the output):
  * fill_all_masks=False, cfg_scale=0, no PAD/injection, protected_index empty, correct_only_first_block=True.
    Prompts with mask tokens (DIJA) need prompt_masks="first_block" (RQ2 ASR layout, see generate()); the
    original instead widens every block to span all remaining masks.
  * hidden states are requested only on the step that uses them (block 0, step 0); the original requests them on
    every step and discards them.
  * the audit is computed for block 0 only (the only block that can trigger repair); the original also computes
    and prints it for later blocks without using it.
  * returns (x, stats) instead of x. stats = forward counts split into generation / repair, audit values.

alpha_rule="progress" (RQ2 design; default and required in threshold mode, opt-in in fixed mode):
  alpha = alpha0 * (1 - tokens committed in this block / masks in this block at block start)
"""

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

MASK_ID = 126336
DEFAULT_ACCEPT_CUES = ["Sure", "Of course", "Here", "Step", "1.", "First", "Proceed", "Let's"]
DEFAULT_REFUSE_CUES = ["Sorry", "I can't", "cannot", "won't", "illegal", "harmful", "not assist", "refuse"]


# --------------------------------------------------------------------------- helpers (copies of the original)

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = max(int(steps), 1)
    base = mask_num // steps
    remainder = (mask_num % steps).squeeze(1)
    num_transfer_tokens = base.expand(-1, steps).clone().to(torch.int64)
    idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    num_transfer_tokens += (idx < remainder.unsqueeze(1)).to(torch.int64)
    return num_transfer_tokens


def cue_id_sets(tokenizer, accept_cues=None, refuse_cues=None):
    """Same token sets as the original build_cue_id_sets()."""
    def ids(phrases):
        out = set()
        for ph in phrases:
            out.update(tokenizer(ph, add_special_tokens=False).input_ids)
        return out
    return ids(accept_cues or DEFAULT_ACCEPT_CUES), ids(refuse_cues or DEFAULT_REFUSE_CUES)


def safety_perplexity_from_logits(logits, positions_mask, accept_ids, refuse_ids):
    probs = torch.softmax(logits, dim=-1)
    if len(accept_ids) == 0 or len(refuse_ids) == 0:
        return 0.0
    accept_p = probs[..., list(accept_ids)].sum(dim=-1)
    refuse_p = probs[..., list(refuse_ids)].sum(dim=-1)
    conflict = 4.0 * accept_p * refuse_p
    mask = positions_mask.to(conflict.dtype)
    return float(((conflict * mask).sum() / mask.sum().clamp_min(1.0)).item())


def cosine_distance(a, b):
    a = a.detach().to(torch.float32)
    b = b.detach().to(torch.float32)
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(1.0 - torch.dot(a, b).item())


# --------------------------------------------------------------------------- decoder

@dataclass
class Stats:
    nfe_gen: int = 0                 # forwards of the denoising loop
    nfe_repair: int = 0              # forwards of the repair loop
    nfe_audit: int = 0               # forwards of the audit: always 0 (it reuses the last step's logits and the
                                     # first step's hidden states); the baseline hidden forward is the caller's
    nfe_cached: int = 0              # of nfe_gen: forwards that ran on a cached window (cache="prefix" | "dual")
    audit_logit: Optional[float] = None    # block-0 logit conflict score
    audit_hidden: Optional[float] = None   # block-0 cosine distance to baseline_hidden (None: not available)
    audit_score: Optional[float] = None    # value the original prints (hidden distance if available, else logit)
    audit_unsafe: Optional[bool] = None
    repaired: bool = False
    steps_per_block: int = 0
    alphas: list = field(default_factory=list)  # block-0 alpha per step (SAR schedule check)

    def as_dict(self):
        return asdict(self)


def _alpha(rule, alpha0, i, steps_per_block, committed, block_masks, dtype, device):
    if rule == "original":
        frac = 1.0 - (i / (steps_per_block - 1)) if steps_per_block > 1 else 1.0
    elif rule == "progress":
        frac = 1.0 - (committed / block_masks) if block_masks > 0 else 1.0
    else:
        raise ValueError(f"alpha_rule must be 'original' or 'progress', got {rule!r}")
    return torch.clamp(torch.tensor(alpha0 * frac, device=device, dtype=dtype), 0.0, 1.0)


@torch.no_grad()
def generate(model, prompt, *, gen_length, block_length, steps=None, temperature=0.0, mask_id=MASK_ID,
             mode="fixed", threshold=None, conf_precision=None,
             defense=False, alpha_rule=None, alpha0=0.3,
             sp_threshold=0.1, refinement_steps=8, remask_ratio=0.9, suppression_value=1e6,
             baseline_hidden=None, cue_ids=None, attention_mask=None, cache=None,
             prompt_masks="error", cache_start_block=0):
    """Decode gen_length tokens after `prompt` ([1, P] token ids) block by block.

    mode="fixed"    : steps // num_blocks forwards per block, a fixed number of tokens committed per step
                      (LLaDA / DiffuGuard schedule). `steps` is required.
    mode="threshold": per step, k = number of masked block positions whose model confidence p(x0) >= threshold
                      (at least 1); the k positions with the highest score are committed; repeat until the block
                      is full (Fast-dLLM parallel decoding). `steps` is ignored.
    defense=False   : score = p(x0) (low_confidence), no audit (= DiffuGuard preset `vanilla`).
    defense=True    : adaptive_step score with alpha_rule, block-0 audit + repair (= preset `diffuguard`,
                      sp_mode hidden; falls back to the logit score when baseline_hidden is None, like the original).
                      threshold + defense (S3): k is counted on p(x0) >= tau (the acceleration decides HOW MANY),
                      the k positions are the top-k by the SAR score (the defense decides WHICH). Repair always
                      runs refinement_steps forwards with the original quota and random order; tau is not used.
    alpha_rule      : None = per mode: "original" (step-based) for fixed, "progress" for threshold (required there).
    conf_precision  : dtype of the softmax giving p(x0). None = per mode: "model" (logits dtype, DiffuGuard) for
                      fixed, "float64" (Fast-dLLM) for threshold. Matters for bf16 logits compared against tau.
    cue_ids         : (accept_ids, refuse_ids) from cue_id_sets(tokenizer); required when defense=True.
    cache           : None | "prefix" | "dual" (S4, threshold mode only = columns D, E). Like Fast-dLLM
                      generate_with_prefix_cache / generate_with_dual_cache: each block starts with a full forward
                      that builds the KV cache (and commits step 0 from its logits); later steps feed only
                      x[:, block_start:] (prefix: cache = prompt + earlier blocks) or x[:, block] with
                      replace_position (dual: cache = everything, the block's KV is overwritten in place).
                      Unlike Fast-dLLM's dual loop (range(1, steps_per_block)), the block is always decoded to the
                      end, so no `steps` is needed; with steps = gen_length the two are identical.
                      Repair never uses the cache: it runs refinement_steps full forwards, as in the other columns,
                      and the next block's first full forward rebuilds the cache from the repaired tokens.

    prompt_masks    : what to do with mask tokens inside the prompt (DIJA). "error" (default; GSM8K never has
                      any) raises. "first_block" = the RQ2 ASR layout agreed with the Without runs: block 0 is the
                      position set {prompt masks} U {first block_length generated positions} and is decoded as one
                      block (fixed: steps // num_blocks forwards with the original quota over all its masks;
                      threshold: until it is full, so at most P_mask + block_length forwards); later blocks are the
                      generated positions only. Under the defense:
                        * alpha "progress" counts block 0 as P_mask + block_length masks (committed / that number);
                          "original" stays step-based. Repair does not use alpha (random refill order).
                        * audit: logit score over the block-0 positions (last step), hidden distance of the first
                          step's last-layer mean over the same positions.
                        * repair: int(|block 0| * remask_ratio) of the block-0 positions, refill only there over
                          refinement_steps forwards, no tau, no cache. Fixed prompt tokens are never remasked
                          (the original's span would include the prompt text between masks; RQ2 keeps the attack
                          prompt as given, like the Without runs).
                      Without prompt masks "first_block" changes nothing.
    cache_start_block: first block that uses the cache (D/E). 0 = Fast-dLLM / GSM8K. The ASR layout uses 1: block 0
                      (with the prompt masks) runs full forwards every step, the cache starts at block 1. Prompt
                      masks with a cache need cache_start_block >= 1 (a prefix cache cannot hold positions that
                      are still being decoded).

    RNG stream: fixed mode draws R = torch.rand every step like the DiffuGuard original (token-identical to it
    at temperature > 0); threshold mode draws R only when the defense uses it, like Fast-dLLM (token-identical
    to Fast-dLLM generate() at temperature > 0 with the defense off). R is always drawn over the whole sequence
    and sliced to the forward's window, so a position gets the same R with or without a cache. cache="prefix"
    draws the Gumbel noise twice on cached steps, as Fast-dLLM does (no effect at temperature 0).
    """
    if mode not in ("fixed", "threshold"):
        raise ValueError(f"mode must be 'fixed' or 'threshold', got {mode!r}")
    if mode == "fixed" and steps is None:
        raise ValueError("mode='fixed' needs steps")
    if mode == "threshold" and threshold is None:
        raise ValueError("mode='threshold' needs threshold")
    alpha_rule = alpha_rule or ("original" if mode == "fixed" else "progress")
    if alpha_rule not in ("original", "progress"):
        raise ValueError(f"alpha_rule must be 'original' or 'progress', got {alpha_rule!r}")
    if mode == "threshold" and alpha_rule == "original":
        # the step-based schedule needs a step count per block, which threshold mode does not have
        raise ValueError("mode='threshold' needs alpha_rule='progress'")
    conf_precision = conf_precision or ("model" if mode == "fixed" else "float64")
    if conf_precision not in ("model", "float64"):
        raise ValueError(f"conf_precision must be 'model' or 'float64', got {conf_precision!r}")
    if prompt.shape[0] != 1:
        raise ValueError("batch size 1 only (as in every run of this project)")
    if prompt_masks not in ("error", "first_block"):
        raise ValueError(f"prompt_masks must be 'error' or 'first_block', got {prompt_masks!r}")
    has_prompt_masks = bool((prompt == mask_id).any())
    if has_prompt_masks and prompt_masks == "error":
        raise NotImplementedError("prompt contains mask tokens (DIJA): pass prompt_masks='first_block'")
    if int(cache_start_block) < 0:
        raise ValueError("cache_start_block must be >= 0")
    if has_prompt_masks and cache and int(cache_start_block) < 1:
        raise ValueError("prompt masks with a cache need cache_start_block >= 1")
    if gen_length % block_length:
        raise ValueError("gen_length must be a multiple of block_length")
    if defense and cue_ids is None:
        raise ValueError("defense=True needs cue_ids=cue_id_sets(tokenizer)")
    if cache not in (None, "prefix", "dual"):
        raise ValueError(f"cache must be None, 'prefix' or 'dual', got {cache!r}")
    if cache and mode != "threshold":
        raise ValueError("cache needs mode='threshold' (RQ2 columns D, E)")

    device = next(model.parameters()).device
    P = prompt.shape[1]
    x = torch.cat([prompt.to(device),
                   torch.full((1, gen_length), mask_id, dtype=torch.long, device=device)], dim=1)
    am = None
    if attention_mask is not None:  # same alignment as the original: pad with ones up to len(x)
        am = attention_mask.to(device)
        am = am if am.dtype == torch.bool else am != 0
        if am.shape[1] < x.shape[1]:
            am = torch.cat([am, torch.ones(1, x.shape[1] - am.shape[1], dtype=torch.bool, device=device)], dim=1)
        am = am[:, : x.shape[1]]
        if cache:
            # Fast-dLLM's cached generators pass no attention mask; batch 1 has no padding, so drop it
            if not bool(am.all()):
                raise ValueError("cache modes support only an all-ones attention_mask")
            am = None

    num_blocks = gen_length // block_length
    fixed = mode == "fixed"
    steps_per_block = max(int(steps) // num_blocks, 1) if fixed else 0
    use_hidden = defense and baseline_hidden is not None
    draw_R = fixed or defense
    stats = Stats(steps_per_block=steps_per_block)

    prompt_mask_pos = (x[0, :P] == mask_id)

    for b in range(num_blocks):
        bs, be = P + b * block_length, P + (b + 1) * block_length
        region = torch.zeros_like(x, dtype=torch.bool)        # positions decoded in this block
        region[:, bs:be] = True
        if b == 0 and has_prompt_masks:
            region[0, :P] = prompt_mask_pos
        block_mask_index = x[region].unsqueeze(0) == mask_id
        block_masks = int(block_mask_index.sum())
        quota = get_num_transfer_tokens(block_mask_index, steps_per_block) if fixed else None
        use_cache = cache if b >= int(cache_start_block) else None
        audit_here = defense and b == 0
        first_hidden_mean = None
        committed = 0

        pkv = replace_position = None
        if use_cache == "dual":
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, bs:be] = True

        i = 0
        while (i < steps_per_block) if fixed else bool((x[region] == mask_id).any()):
            want_hidden = audit_here and use_hidden and i == 0
            if use_cache is None or i == 0:
                # full forward over the whole sequence (every step without a cache; block start with a cache)
                kw = dict(use_cache=True) if use_cache else {}
                out = model(x, attention_mask=am, output_hidden_states=want_hidden, return_dict=True, **kw)
                off = 0
                if use_cache:
                    pkv = out.past_key_values
                    if use_cache == "prefix":   # keep prompt + earlier blocks only
                        pkv = [tuple(t[:, :, :bs] for t in layer) for layer in pkv]
            elif use_cache == "prefix":
                out = model(x[:, bs:], past_key_values=pkv, use_cache=True)
                off = bs
                add_gumbel_noise(out.logits, temperature)   # Fast-dLLM draws (and discards) noise once more here
            else:
                out = model(x[:, bs:be], past_key_values=pkv, use_cache=True, replace_position=replace_position)
                off = bs
            stats.nfe_gen += 1
            stats.nfe_cached += int(off > 0)
            logits = out.logits
            W = logits.shape[1]
            xw = x[:, off:off + W]          # view: writes below go into x
            rw = region[:, off:off + W]     # current block inside the window
            if want_hidden and getattr(out, "hidden_states", None):
                first_hidden_mean = out.hidden_states[-1][:, rw[0], :].mean(dim=1).squeeze(0).detach()

            mask_index = xw == mask_id
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            # p(x0) is only read inside the current block (score, tau count), so the softmax runs on those rows
            # only: same per-row values, but no [window, vocab] float64 copy (3+ GB for a 1.6k-token GSM8K
            # sequence on a full forward -> OOM on a 24 GB 3090). Rows outside the block get 0 (never used).
            rows = rw[0].nonzero(as_tuple=True)[0]
            lr = logits[0, rows]
            pr = F.softmax(lr.to(torch.float64) if conf_precision == "float64" else lr, dim=-1)
            model_confidence = torch.zeros(x0.shape, dtype=pr.dtype, device=x0.device)
            model_confidence[0, rows] = torch.gather(pr, -1, x0[0, rows].unsqueeze(-1)).squeeze(-1)
            R = torch.rand((1, x.shape[1]), device=x0.device)[:, off:off + W] if draw_R else None

            if defense:
                alpha = _alpha(alpha_rule, alpha0, i, steps_per_block, committed, block_masks,
                               model_confidence.dtype, x0.device)
                if audit_here:
                    stats.alphas.append(float(alpha))
                x0_p = (1 - alpha) * model_confidence + alpha * R
            else:
                x0_p = model_confidence

            conf = x0_p.clone().masked_fill(~rw, -np.inf)
            x0 = torch.where(mask_index, x0, xw)
            confidence = torch.where(mask_index, conf, -np.inf)

            if fixed:
                k = int(quota[0, i].item())
            else:
                # Fast-dLLM rule on the model confidence p(x0): count masked block positions with p >= tau,
                # always at least one. The k committed positions are the top-k by score; with the defense off
                # score == p, so this is exactly Fast-dLLM's set {p >= tau} U {argmax p}.
                in_block = mask_index & rw
                k = max(int((in_block & (model_confidence >= threshold)).sum()), 1)
            if k > 0:
                transfer_index = torch.zeros_like(xw, dtype=torch.bool)
                transfer_index[0, torch.topk(confidence[0], k=k).indices] = True
                xw[transfer_index] = x0[transfer_index]
                committed += k
            i += 1

        if not audit_here:
            continue

        # ---- block-0 audit (original: sp_mode="hidden")
        positions = rw                                                  # block 0 in the last step's window
        stats.audit_logit = safety_perplexity_from_logits(logits, positions, *cue_ids)
        unsafe = stats.audit_logit >= sp_threshold
        stats.audit_score = stats.audit_logit
        if use_hidden and first_hidden_mean is not None:
            stats.audit_hidden = cosine_distance(first_hidden_mean, baseline_hidden)
            stats.audit_score = stats.audit_hidden
            unsafe = unsafe or stats.audit_hidden >= sp_threshold
        stats.audit_unsafe = bool(unsafe)
        if not unsafe:
            continue

        # ---- repair (original refinement phase, block 0 only)
        cand = torch.nonzero(region[0], as_tuple=False).squeeze(1)
        n_remask = int(cand.numel() * float(remask_ratio))
        if n_remask <= 0:
            continue
        stats.repaired = True
        remask_pos = cand[torch.randperm(cand.numel(), device=device)[:n_remask]]
        old_ids = x[:, remask_pos].clone()
        x[:, remask_pos] = mask_id
        r_steps = max(int(refinement_steps), 1)
        r_quota = get_num_transfer_tokens(x[region].unsqueeze(0) == mask_id, r_steps)

        for r in range(r_steps):
            mask_index = x == mask_id
            logits = model(x, attention_mask=am).logits
            stats.nfe_repair += 1
            if np.isfinite(suppression_value):
                logits[0, remask_pos, old_ids[0]] -= suppression_value
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            # original: low_confidence -> confidence order, any other remasking -> random order. Repair only
            # runs with the defense on, whose remasking is adaptive_step, so the refill order is random.
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            conf = x0_p.clone().masked_fill(~region, -np.inf)
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, conf, -np.inf)
            k = int(min(r_quota[0, r].item(), torch.sum(confidence[0] > -np.inf).item()))
            if k > 0:
                sel = torch.zeros_like(x0, dtype=torch.bool)
                sel[0, torch.topk(confidence[0], k=k).indices] = True
                x[sel] = x0[sel]

    return x, stats

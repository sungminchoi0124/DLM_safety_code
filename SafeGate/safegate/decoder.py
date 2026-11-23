"""Column-A decoder (Fast-dLLM threshold commits, no cache) with an optional safety gate and step trace.

The loop is RQ2/src/combined.py generate() restricted to mode="threshold", cache=None (RQ2 column A, the only
setting this experiment uses), with the helpers imported from it. With gate=None it is token-for-token the
combined decoder, defense on or off, with or without DIJA prompt masks, at any temperature
(tests/test_decoder.py compares the two). Two additions, both inside block 0:

Safety gate (GateConfig, the "A + safety gate" row). Per forward, before the step's commits are written:
  window   = the gate positions that are still masks: the first K generated positions ("response") and/or the
             first K_infill prompt-mask positions ("infill", DIJA)
  m_ref    = mean over the window of sum_{refusal ids} p(v)     (same averaging as RQ3 / RAEC)
  m_cmp    = mean over the window of sum_{compliance ids} p(v)
  contention: m_ref >= a and m_cmp >= b, while fewer than H steps have been held for this item
  if contention: of the positions the Fast-dLLM rule picked, those inside the gate window whose candidate token
             is not a refusal token are not committed this step (they stay masks); every other pick is committed
             as usual. After H held steps the item decodes with the plain Fast-dLLM rule.
  p is the decoder's own float64 softmax of the block rows (no extra forward, no random draw), so a gate whose
  condition never holds gives the same tokens as no gate.
  A step may commit nothing when every pick was held; H bounds the extra forwards.

Step trace (trace.StepTrace): see trace.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "RQ2") not in sys.path:
    sys.path.insert(0, str(REPO / "RQ2"))
from src import combined  # noqa: E402  (RQ2 combined decoder: helpers and Stats)

MASK_ID = combined.MASK_ID
GATE_WINDOWS = ("response", "infill")


@dataclass
class GateConfig:
    a: float                      # refusal mass threshold
    b: float                      # compliance mass threshold
    K: int = 16                   # response positions in the window
    H: int = 8                    # at most H held steps per item
    windows: tuple = ("response",)
    K_infill: Optional[int] = 16  # first K_infill prompt masks (None = all) when "infill" is in windows

    def __post_init__(self):
        self.windows = tuple(self.windows)
        bad = [w for w in self.windows if w not in GATE_WINDOWS]
        if bad or not self.windows:
            raise ValueError(f"gate windows must be a non-empty subset of {GATE_WINDOWS}, got {self.windows}")
        if self.a is None or self.b is None:
            raise ValueError("gate needs both thresholds a and b (set them from the dev traces)")
        if int(self.K) < 1 or int(self.H) < 0:
            raise ValueError("gate needs K >= 1 and H >= 0")


@dataclass
class GateStats(combined.Stats):
    gate_steps: int = 0                    # forwards in which the gate held (contention while H not used up)
    gate_held: int = 0                     # picks not committed because of the gate (summed over steps)
    gate_first_step: Optional[int] = None  # block-0 forward index of the first held step
    gate_checks: int = 0                   # forwards in which the gate window had open positions
    block0_forwards: int = 0
    commits_per_forward: list = field(default_factory=list)   # generation forwards, all blocks


def _gate_window(prompt, P, gate: GateConfig, mask_id, L, device):
    w = torch.zeros((1, L), dtype=torch.bool, device=device)
    if "response" in gate.windows:
        w[0, P:P + int(gate.K)] = True
    if "infill" in gate.windows:
        pm = torch.nonzero(prompt[0] == mask_id, as_tuple=False).squeeze(1)
        if gate.K_infill is not None:
            pm = pm[: int(gate.K_infill)]
        w[0, pm.to(device)] = True
    return w


@torch.no_grad()
def generate(model, prompt, *, gen_length, block_length, threshold, temperature=0.0, mask_id=MASK_ID,
             defense=False, alpha0=0.3, sp_threshold=0.1, refinement_steps=8, remask_ratio=0.9,
             suppression_value=1e6, baseline_hidden=None, cue_ids=None, attention_mask=None,
             prompt_masks="error", gate: Optional[GateConfig] = None, gate_ids=None, trace=None):
    """Same arguments and result as combined.generate(mode="threshold", cache=None, alpha_rule="progress"),
    plus gate (GateConfig or None), gate_ids = (refusal_ids, compliance_ids) for the gate, trace (StepTrace or
    None). Returns (x, GateStats)."""
    if prompt.shape[0] != 1:
        raise ValueError("batch size 1 only")
    if prompt_masks not in ("error", "first_block"):
        raise ValueError(f"prompt_masks must be 'error' or 'first_block', got {prompt_masks!r}")
    has_prompt_masks = bool((prompt == mask_id).any())
    if has_prompt_masks and prompt_masks == "error":
        raise NotImplementedError("prompt contains mask tokens (DIJA): pass prompt_masks='first_block'")
    if gen_length % block_length:
        raise ValueError("gen_length must be a multiple of block_length")
    if defense and cue_ids is None:
        raise ValueError("defense=True needs cue_ids=combined.cue_id_sets(tokenizer)")
    if gate is not None:
        if gate_ids is None:
            raise ValueError("gate needs gate_ids=(refusal_ids, compliance_ids)")
        if "response" in gate.windows and int(gate.K) > block_length:
            raise ValueError("gate K must fit in block 0 (K <= block_length)")

    device = next(model.parameters()).device
    P = prompt.shape[1]
    x = torch.cat([prompt.to(device),
                   torch.full((1, gen_length), mask_id, dtype=torch.long, device=device)], dim=1)
    L = x.shape[1]
    am = None
    if attention_mask is not None:
        am = attention_mask.to(device)
        am = am if am.dtype == torch.bool else am != 0
        if am.shape[1] < L:
            am = torch.cat([am, torch.ones(1, L - am.shape[1], dtype=torch.bool, device=device)], dim=1)
        am = am[:, :L]

    num_blocks = gen_length // block_length
    use_hidden = defense and baseline_hidden is not None
    stats = GateStats(steps_per_block=0)
    prompt_mask_pos = (x[0, :P] == mask_id)
    if trace is not None:
        trace.begin(x[:, :P], mask_id)
    gwin = gref = gcmp = None
    if gate is not None:
        gwin = _gate_window(x[:, :P], P, gate, mask_id, L, device)
        gref = torch.tensor(sorted(gate_ids[0]), dtype=torch.long, device=device)
        gcmp = torch.tensor(sorted(gate_ids[1]), dtype=torch.long, device=device)

    for b in range(num_blocks):
        bs, be = P + b * block_length, P + (b + 1) * block_length
        region = torch.zeros_like(x, dtype=torch.bool)
        region[:, bs:be] = True
        if b == 0 and has_prompt_masks:
            region[0, :P] = prompt_mask_pos
        block_masks = int((x[region] == mask_id).sum())
        audit_here = defense and b == 0
        first_hidden_mean = None
        committed = 0
        rows = region[0].nonzero(as_tuple=True)[0]
        pos2row = None
        if b == 0 and (trace is not None or gate is not None):
            pos2row = torch.full((L,), -1, dtype=torch.long, device=device)
            pos2row[rows] = torch.arange(rows.numel(), device=device)

        i = 0
        while bool((x[region] == mask_id).any()):
            want_hidden = audit_here and use_hidden and i == 0
            out = model(x, attention_mask=am, output_hidden_states=want_hidden, return_dict=True)
            stats.nfe_gen += 1
            logits = out.logits
            xw, rw = x, region
            if want_hidden and getattr(out, "hidden_states", None):
                first_hidden_mean = out.hidden_states[-1][:, rw[0], :].mean(dim=1).squeeze(0).detach()

            mask_index = xw == mask_id
            x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
            lr = logits[0, rows]
            pr = F.softmax(lr.to(torch.float64), dim=-1)          # threshold mode: float64 like Fast-dLLM
            model_confidence = torch.zeros(x0.shape, dtype=pr.dtype, device=x0.device)
            model_confidence[0, rows] = torch.gather(pr, -1, x0[0, rows].unsqueeze(-1)).squeeze(-1)
            R = torch.rand((1, L), device=x0.device) if defense else None

            if defense:
                alpha = combined._alpha("progress", alpha0, i, 0, committed, block_masks,
                                        model_confidence.dtype, x0.device)
                if audit_here:
                    stats.alphas.append(float(alpha))
                x0_p = (1 - alpha) * model_confidence + alpha * R
            else:
                x0_p = model_confidence

            conf = x0_p.clone().masked_fill(~rw, -np.inf)
            x0 = torch.where(mask_index, x0, xw)
            confidence = torch.where(mask_index, conf, -np.inf)
            in_block = mask_index & rw
            k = max(int((in_block & (model_confidence >= threshold)).sum()), 1)
            transfer_index = torch.zeros_like(xw, dtype=torch.bool)
            transfer_index[0, torch.topk(confidence[0], k=k).indices] = True

            gate_rec = None
            if gate is not None and b == 0:
                open_w = gwin & mask_index
                if bool(open_w.any()):
                    stats.gate_checks += 1
                    pw = pr[pos2row[open_w[0]]]
                    m_ref = float(pw[:, gref].sum(-1).mean())
                    m_cmp = float(pw[:, gcmp].sum(-1).mean())
                    fire = m_ref >= gate.a and m_cmp >= gate.b and stats.gate_steps < int(gate.H)
                    held = 0
                    if fire:
                        cancel = transfer_index & gwin & ~torch.isin(x0, gref)
                        held = int(cancel.sum())
                        transfer_index &= ~cancel
                        stats.gate_steps += 1
                        stats.gate_held += held
                        if stats.gate_first_step is None:
                            stats.gate_first_step = i
                    gate_rec = {"m_ref": round(m_ref, 6), "m_cmp": round(m_cmp, 6), "fire": bool(fire),
                                "held": held, "n_open": int(open_w.sum())}

            if trace is not None and b == 0:
                trace.record(phase="gen", block=b, step=i, mask_index=mask_index, x0=x0, conf=model_confidence,
                             probs_for=lambda pos: pr[pos2row[pos]], transfer_index=transfer_index, k=k,
                             gate=gate_rec)
            n_commit = int(transfer_index.sum())
            if n_commit:
                xw[transfer_index] = x0[transfer_index]
                committed += n_commit
            stats.commits_per_forward.append(n_commit)
            if b == 0:
                stats.block0_forwards += 1
            i += 1

        if not audit_here:
            continue

        # ---- block-0 audit and repair: combined.generate, unchanged
        positions = rw
        stats.audit_logit = combined.safety_perplexity_from_logits(logits, positions, *cue_ids)
        unsafe = stats.audit_logit >= sp_threshold
        stats.audit_score = stats.audit_logit
        if use_hidden and first_hidden_mean is not None:
            stats.audit_hidden = combined.cosine_distance(first_hidden_mean, baseline_hidden)
            stats.audit_score = stats.audit_hidden
            unsafe = unsafe or stats.audit_hidden >= sp_threshold
        stats.audit_unsafe = bool(unsafe)
        if not unsafe:
            continue

        cand = torch.nonzero(region[0], as_tuple=False).squeeze(1)
        n_remask = int(cand.numel() * float(remask_ratio))
        if n_remask <= 0:
            continue
        stats.repaired = True
        remask_pos = cand[torch.randperm(cand.numel(), device=device)[:n_remask]]
        old_ids = x[:, remask_pos].clone()
        x[:, remask_pos] = mask_id
        r_steps = max(int(refinement_steps), 1)
        r_quota = combined.get_num_transfer_tokens(x[region].unsqueeze(0) == mask_id, r_steps)

        for r in range(r_steps):
            mask_index = x == mask_id
            logits = model(x, attention_mask=am).logits
            stats.nfe_repair += 1
            if np.isfinite(suppression_value):
                logits[0, remask_pos, old_ids[0]] -= suppression_value
            x0 = torch.argmax(combined.add_gumbel_noise(logits, temperature), dim=-1)
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            conf = x0_p.clone().masked_fill(~region, -np.inf)
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, conf, -np.inf)
            kk = int(min(r_quota[0, r].item(), torch.sum(confidence[0] > -np.inf).item()))
            if kk > 0:
                sel = torch.zeros_like(x0, dtype=torch.bool)
                sel[0, torch.topk(confidence[0], k=kk).indices] = True
                if trace is not None:
                    trace.record(phase="repair", block=0, step=r, mask_index=mask_index, x0=x0, conf=None,
                                 probs_for=None, transfer_index=sel, k=kk)
                x[sel] = x0[sel]

    return x, stats

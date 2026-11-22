"""Commit-time metrics from probe margins and trace bookkeeping (pure numpy).

All indices are forward / state indices f (see capture.py). Label y: 1 = harmful compliance, 0 = refusal.
"""
from __future__ import annotations

import numpy as np


def _first_of_true_suffix(ok: np.ndarray) -> int:
    """Smallest i such that ok[i:] is all True; len(ok) if ok[-1] is False (or ok is empty)."""
    ok = np.asarray(ok, bool)
    if ok.size == 0 or not ok[-1]:
        return int(ok.size)
    bad = np.nonzero(~ok)[0]
    return int(bad[-1] + 1) if bad.size else 0


def commit_step(margins: np.ndarray, y: int, tau: float) -> tuple[int, bool]:
    """s* = first forward after which sign(d) agrees with the final label and |d| >= tau for good.

    Returns (s_star, censored). censored=True when no such forward exists (s_star = T)."""
    d = np.asarray(margins, float)
    target = 1.0 if int(y) == 1 else -1.0
    ok = (np.sign(d) == target) & (np.abs(d) >= tau)
    s = _first_of_true_suffix(ok)
    return s, s >= d.size


def hesitation(margins: np.ndarray, tau: float) -> dict:
    d = np.asarray(margins, float)
    hes = np.abs(d) < tau
    idx = np.nonzero(hes)[0]
    return {"n_hes": int(hes.sum()),
            "first_hes": int(idx[0]) if idx.size else -1,
            "last_hes": int(idx[-1]) if idx.size else -1,
            "sign_flips": int((np.diff(np.sign(d)) != 0).sum()) if d.size > 1 else 0}


def text_commit_step(lead_unmasked: np.ndarray, lead_k: int) -> int:
    """First state index after which all lead positions stay unmasked (index into x_0..x_T)."""
    full = np.asarray(lead_unmasked) >= max(int(lead_k), 1)
    return _first_of_true_suffix(full)


def repair_start(n_remask: np.ndarray) -> int:
    """First forward of the DiffuGuard repair phase, or -1 if nothing was re-masked."""
    idx = np.nonzero(np.asarray(n_remask) > 0)[0]
    return int(idx[0] + 1) if idx.size else -1


def item_metrics(margins: np.ndarray, y: int, tau: float, tr: dict) -> dict:
    """One row of commit_metrics.csv. `tr` = arrays from capture.load_trace."""
    d = np.asarray(margins, float)
    T = d.size
    n_tracked = int(tr["n_tracked"])
    n_masks = np.asarray(tr["n_masks"])
    n_commit = np.asarray(tr["n_commit"])
    s, cens = commit_step(d, y, tau)
    r0 = repair_start(tr["n_remask"])
    main_end = r0 if r0 > 0 else T
    h = hesitation(d[:main_end], tau)
    st = text_commit_step(tr["lead_unmasked"], int(tr["lead_k"]))
    tokens_at = n_tracked - int(n_masks[min(s, T)])
    row = {
        "label": int(y),
        "n_forwards": T,
        "main_forwards": main_end,
        "n_tracked": n_tracked,
        "tau": float(tau),
        "s_star": s,
        "censored": bool(cens),
        "frac_nfe_at_s_star": s / T if T else np.nan,
        "tokens_at_s_star": tokens_at,
        "frac_tokens_at_s_star": tokens_at / n_tracked if n_tracked else np.nan,
        "max_bundle_before": int(n_commit[:s].max()) if s > 0 else 0,
        "mean_bundle_before": float(n_commit[:s].mean()) if s > 0 else 0.0,
        **h,
        "s_text": st,
        "latent_lead": st - s,
        "final_margin": float(d[-1]) if T else np.nan,
        "final_sign_ok": bool(T and np.sign(d[-1]) == (1 if y == 1 else -1)),
        "repair_start": r0,
    }
    if r0 > 0:
        pre = d[r0 - 1]
        row.update({
            "margin_before_repair": float(pre),
            "abs_margin_before_repair": float(abs(pre)),
            "repair_flipped": bool(np.sign(pre) != np.sign(d[-1])),
            "s_star_in_repair": bool(s >= r0),
        })
    return row


# --------------------------------------------------------------------------- behavioural commit

def behavioural_commit(checkpoints: list, frac_same: list, level: float = 0.9) -> tuple[int, bool]:
    """s_beh = first checkpoint from which every later checkpoint keeps >= level of continuations on the
    original label. Returns (checkpoint value, censored)."""
    order = np.argsort(checkpoints)
    cps = np.asarray(checkpoints)[order]
    fs = np.asarray(frac_same, float)[order]
    i = _first_of_true_suffix(fs >= level)
    if i >= len(cps):
        return int(cps[-1]) + 1, True
    return int(cps[i]), False


def nearest_checkpoint(s: int, checkpoints: list) -> int:
    """Smallest checkpoint >= s (the first checkpoint at which a commit at s is visible)."""
    cps = sorted(checkpoints)
    for c in cps:
        if c >= s:
            return c
    return cps[-1] + 1

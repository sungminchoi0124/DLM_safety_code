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


PROPHET_THRESHOLDS = (7.5, 5.0, 2.5)
PROPHET_PHASES = (0.33, 0.67)


def prophet_threshold_curve(n: int, steps: int, thresholds=PROPHET_THRESHOLDS, phases=PROPHET_PHASES):
    """Threshold Prophet applies at forward f = 0..n-1 (official progress = (f + 1) / steps)."""
    prog = (np.arange(n) + 1) / float(steps)
    return np.where(prog < phases[0], thresholds[0], np.where(prog < phases[1], thresholds[1], thresholds[2]))


def simulated_exit_step(gap_all: np.ndarray, steps: int, thresholds=PROPHET_THRESHOLDS,
                        phases=PROPHET_PHASES) -> int:
    """Exit step (1-based forward count) a Prophet run would take given these per-forward answer gaps, or -1.

    Exact for a vanilla quota / original trace with the same seed: Prophet decodes the same tokens until it
    exits, so its gaps up to the exit are these gaps."""
    g = np.asarray(gap_all, float)
    hit = np.nonzero(g >= prophet_threshold_curve(g.size, steps, thresholds, phases))[0]  # NaN -> False
    return int(hit[0] + 1) if hit.size else -1


def _nanmedian(a):
    a = np.asarray(a, float)
    return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


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
    if "prophet_exit_step" in tr:
        # Prophet run: the last forward (T - 1) is the exit forward; its transition holds the all-in fill.
        exited = bool(int(tr["prophet_exited"]))
        e = int(tr["prophet_exit_step"])
        frac = (n_tracked - int(n_masks[T - 1])) / n_tracked if (exited and T and n_tracked) else 1.0
        row.update({
            "exited": exited,
            "exit_step": e,
            "exit_frac_nfe": e / T if T else np.nan,
            "exit_frac_tokens": frac,               # share of tracked tokens committed before the all-in fill
            "n_filled_at_exit": int(tr["prophet_n_filled_at_exit"]),
            # within-run version of the H1 test; the cross-run one (s* of R) is in `analyze flips`
            "exit_after_s_star": bool(e > s),
        })
    if "gap_all" in tr:
        g_all = np.asarray(tr["gap_all"], float)[:main_end]
        row.update({
            "sim_exit_step": simulated_exit_step(g_all, main_end),
            "gap_committed_median": _nanmedian(np.asarray(tr["gap_committed"])[:main_end]),
            "gap_masked_median": _nanmedian(np.asarray(tr["gap_masked"])[:main_end]),
            "mass_refuse_first": float(tr["mass_refuse"][0]) if T else np.nan,
            "mass_comply_first": float(tr["mass_comply"][0]) if T else np.nan,
        })
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

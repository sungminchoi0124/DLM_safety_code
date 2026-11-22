"""Outcome probe: logistic regression on traced hidden states -> per-forward margin d_f.

Follows D^2-Monitor (linear probe, margin, hesitation tau chosen so ~50% of training trajectories hesitate)
with one change: the label is the final RESPONSE outcome (harmful compliance vs refusal), not the prompt
label, which is constant for DIJA.

fit modes
  traj_mean  one row per trajectory = mean of h_f over forwards (D^2-Monitor's base probe), applied per step
  all_steps  one row per forward, every row labelled with the final outcome, weight 1/T per trajectory
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

EVAL_FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Traj:
    run: str
    item: int          # item_index (JBB behaviour) -> CV group
    y: int
    H: np.ndarray      # [T, n_layers, d] float16
    layer_names: list
    main_end: int      # forwards before repair (T if no repair)


def _rows(trajs, li, mode):
    X, y, w, g = [], [], [], []
    for t in trajs:
        H = t.H[:, li, :].astype(np.float32)
        if mode == "traj_mean":
            X.append(H.mean(0, keepdims=True)); y.append([t.y]); w.append([1.0]); g.append([t.item])
        else:
            T = H.shape[0]
            X.append(H); y.append([t.y] * T); w.append([1.0 / T] * T); g.append([t.item] * T)
    return (np.concatenate(X), np.concatenate(y), np.concatenate(w), np.concatenate(g))


@dataclass
class Probe:
    layer: str
    layer_index: int
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float
    tau: float
    mode: str
    C: float

    def margins(self, H: np.ndarray) -> np.ndarray:
        z = (H[:, self.layer_index, :].astype(np.float32) - self.mean) / self.std
        return z @ self.w + self.b

    def save(self, path: Path, report: dict):
        path = Path(path)
        np.savez(path, mean=self.mean, std=self.std, w=self.w, b=np.float64(self.b), tau=np.float64(self.tau),
                 layer=np.array(self.layer), layer_index=np.int32(self.layer_index), mode=np.array(self.mode),
                 C=np.float64(self.C))
        path.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Probe":
        with np.load(path, allow_pickle=False) as z:
            return cls(layer=str(z["layer"]), layer_index=int(z["layer_index"]), mean=z["mean"], std=z["std"],
                       w=z["w"], b=float(z["b"]), tau=float(z["tau"]), mode=str(z["mode"]), C=float(z["C"]))


def _fit_one(trajs, li, mode, C, seed=0):
    X, y, w, _ = _rows(trajs, li, mode)
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=seed)
    clf.fit((X - mean) / std, y, sample_weight=w * len(trajs) / w.sum())
    return mean, std, clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def _margins(H, li, mean, std, w, b):
    return ((H[:, li, :].astype(np.float32) - mean) / std) @ w + b


def stepwise_auroc(trajs, margin_lists, fracs=EVAL_FRACS):
    """AUROC of the margin at fraction f of each trajectory's main phase."""
    out = {}
    ys = np.array([t.y for t in trajs])
    for f in fracs:
        vals = []
        for t, d in zip(trajs, margin_lists):
            idx = min(int(round(f * (t.main_end - 1))), len(d) - 1)
            vals.append(d[idx])
        out[f"auroc@{f:g}"] = float(roc_auc_score(ys, vals)) if len(set(ys)) == 2 else float("nan")
    return out


def choose_tau(margin_lists, main_ends) -> float:
    """tau such that half of the trajectories have at least one forward with |d| < tau."""
    mins = [np.min(np.abs(d[:m])) for d, m in zip(margin_lists, main_ends)]
    return float(np.median(mins))


def fit_probe(trajs: list, *, mode="traj_mean", Cs=(0.01, 0.1, 1.0), n_folds=5, seed=0):
    """Group-CV (by item) over layers x C; select by mean step-wise AUROC; refit on all; pick tau."""
    if len({t.y for t in trajs}) < 2:
        raise SystemExit("probe needs both labels in the training trajectories")
    layer_names = trajs[0].layer_names
    groups = np.array([t.item for t in trajs])
    n_folds = min(n_folds, len(set(groups)))
    gkf = GroupKFold(n_splits=n_folds)
    grid = []
    for li, ln in enumerate(layer_names):
        for C in Cs:
            oof = [None] * len(trajs)
            for tr_idx, te_idx in gkf.split(np.zeros(len(trajs)), groups=groups):
                tr_set = [trajs[i] for i in tr_idx]
                if len({t.y for t in tr_set}) < 2:
                    continue
                mean, std, w, b = _fit_one(tr_set, li, mode, C, seed)
                for i in te_idx:
                    oof[i] = _margins(trajs[i].H, li, mean, std, w, b)
            keep = [i for i, d in enumerate(oof) if d is not None]
            sa = stepwise_auroc([trajs[i] for i in keep], [oof[i] for i in keep])
            grid.append({"layer": ln, "layer_index": li, "C": C, **sa,
                         "score": float(np.nanmean(list(sa.values())))})
    best = max(grid, key=lambda r: r["score"])
    li, C = best["layer_index"], best["C"]
    mean, std, w, b = _fit_one(trajs, li, mode, C, seed)
    ms = [_margins(t.H, li, mean, std, w, b) for t in trajs]
    tau = choose_tau(ms, [t.main_end for t in trajs])
    probe = Probe(layer=best["layer"], layer_index=li, mean=mean, std=std, w=w, b=b, tau=tau, mode=mode, C=C)
    per_run = {}
    for run in sorted({t.run for t in trajs}):
        sub = [(t, d) for t, d in zip(trajs, ms) if t.run == run]
        per_run[run] = stepwise_auroc([s[0] for s in sub], [s[1] for s in sub])
    report = {"mode": mode, "n_traj": len(trajs), "n_harmful": int(sum(t.y for t in trajs)),
              "n_folds": n_folds, "selected": best, "tau": tau, "grid": grid,
              "in_sample_auroc_by_run": per_run,
              "note": "selection uses out-of-fold AUROC grouped by item_index; tau from in-sample margins"}
    return probe, report

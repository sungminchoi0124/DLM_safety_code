"""Small statistics helpers (numpy + torch only; no scikit-learn on the workspace)."""
from __future__ import annotations

import numpy as np
import torch


def auroc(pos, neg):
    """P(score of a random positive > random negative), ties count 1/2. None if a class is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auroc(pos, neg, n=2000, seed=0):
    """Percentile 95% CI, resampling each class separately."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    rng = np.random.default_rng(seed)
    vals = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a))
    sv = a[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def quantiles(v, qs=(10, 50, 90)):
    v = np.asarray(v, float)
    return [round(float(np.percentile(v, q)), 4) for q in qs] if len(v) else None


def fit_logreg(X, y, l2=1.0, max_iter=200):
    """L2 logistic regression on standardized features (train statistics). Returns a scoring function."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(y)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w @ w) / (2 * len(yt))
        loss.backward()
        return loss

    opt.step(closure)
    w_, b_ = w.detach().numpy(), float(b.detach())
    return lambda Z: ((np.asarray(Z, np.float64) - mu) / sd) @ w_ + b_

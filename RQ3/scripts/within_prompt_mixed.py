"""Within-prompt test on DIJA prompts whose infill outcome flips across seeds, for several training sets.
Train: logistic regression (C=0.01, standardized, trajectory-mean of layer b23) on the given runs, always leaving
out every DIJA trajectory of the held-out prompt. Test: that prompt's DIJA trajectories (infill region) at each
step fraction; concordance = share of (harmful, safe) pairs where the harmful one scores higher (0.5 = chance).
Also reports a plain out-of-prompt AUROC on all DIJA trajectories (leave-one-prompt-out, step fractions).
Numbers only."""
import csv, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
T = "/root/rq3-outputs/trace"
DIJA = ["R_s0"] + [f"R_s{k}000" for k in range(1, 9)]
BENCH = ["pad_advbench__R", "jbb_behaviors__R", "wildjailbreak__R"]
KEY = {"infill": "hidden_infill", "tail": "hidden_tail"}
FR = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

def load(run, labels, region):
    out = []
    for r in csv.DictReader(open(f"{T}/{run}/{labels}.csv")):
        if r["label"] not in ("harmful", "refusal"):
            continue
        i = int(r["item_index"])
        with np.load(f"{T}/{run}/hidden/item{i:03d}.npz") as z:
            names = [str(n) for n in z["layer_names"]]
            h = z[KEY[region]][:, names.index("b23"), :].astype(np.float32)
        out.append((run, i, int(r["label"] == "harmful"), h))
    return out

dija = [t for run in DIJA for t in load(run, "labels", "infill")]
bench_tail = [t for run in BENCH for t in load(run, "labels_tail", "tail")]
bench_infill = [t for run in BENCH for t in load(run, "labels_infill", "infill")]
items = sorted({t[1] for t in dija})
mixed = [i for i in items if len({t[2] for t in dija if t[1] == i}) == 2]
print(f"DIJA trajectories {len(dija)}, mixed prompts {len(mixed)}, bench tail {len(bench_tail)}, bench infill {len(bench_infill)}")

def frame(h, f):
    return h[min(int(round(f * (len(h) - 1))), len(h) - 1)]

CONDS = {"DIJA only (infill)": (True, []), "benchmarks only (answer)": (False, bench_tail),
         "DIJA + benchmarks (answer)": (True, bench_tail), "benchmarks only (infill labels)": (False, bench_infill)}
for name, (use_dija, extra) in CONDS.items():
    Xe = np.stack([t[3].mean(0) for t in extra]) if extra else None
    ye = np.array([t[2] for t in extra]) if extra else None
    conc = {f: [] for f in FR}
    scores = {f: [] for f in FR}; ys = []
    cache = {}
    for i in items:
        tr = [t for t in dija if t[1] != i] if use_dija else []
        X = np.stack([t[3].mean(0) for t in tr]) if tr else None
        y = np.array([t[2] for t in tr]) if tr else None
        if Xe is not None:
            X = Xe if X is None else np.concatenate([X, Xe]); y = ye if y is None else np.concatenate([y, ye])
        if not use_dija and "fit" in cache:
            sc, clf = cache["fit"]               # no DIJA in training: one fit serves every held-out prompt
        else:
            sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X), y)
            cache["fit"] = (sc, clf)
        te = [t for t in dija if t[1] == i]
        for f in FR:
            m = clf.decision_function(sc.transform(np.stack([frame(t[3], f) for t in te])))
            scores[f] += list(m)
            if i in mixed:
                pos = [v for v, t in zip(m, te) if t[2] == 1]; neg = [v for v, t in zip(m, te) if t[2] == 0]
                conc[f] += [1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg]
        ys += [t[2] for t in te]
    auc = " ".join(f"{f}:{roc_auc_score(ys, scores[f]):.2f}" for f in FR)
    cc = " ".join(f"{f}:{np.mean(conc[f]):.2f}" for f in FR)
    print(f"\n[{name}]\n  all DIJA, leave-one-prompt-out AUROC: {auc}\n  within flipping prompts (n_pairs={len(conc[0.0])}): {cc}")

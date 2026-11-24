"""Resample (behavioural) commit vs probe, on the DIJA prompts that flip across seeds. Numbers only.

For every item and checkpoint s of the reference decode (seed 0 = trace R_s0):
  p_harm(s)  = share of the continuations from state x_s judged harmful (DIJA infill convention, rs labels)
  agree(s)   = share of continuations with the reference's own final label
  margin(s)  = probe score of the reference trajectory at forward s (trace R_s0, hidden_infill, layer b23)
The probe is fit like within_prompt_mixed.py (DIJA vanilla seeds + other benchmarks' answer region), leaving out
all DIJA trajectories of the item being scored. Reports:
  * behavioural commit step per item = first checkpoint from which agree >= LEVEL at every later checkpoint
  * mean agree / p_harm by checkpoint fraction
  * within-item Spearman between margin(s) and p_harm(s), pooled over items (does the probe track the
    behavioural probability inside one prompt?)

    python resample_vs_probe.py --rs_labels /root/rq3-outputs/resample/rs_main_labels.csv --out results.json
"""
import argparse, csv, json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

T = "/root/rq3-outputs/trace"
DIJA = ["R_s0"] + [f"R_s{k}000" for k in range(1, 9)]
BENCH = ["pad_advbench__R", "jbb_behaviors__R", "wildjailbreak__R"]


def load(run, labels, key):
    out = []
    for r in csv.DictReader(open(f"{T}/{run}/{labels}.csv")):
        if r["label"] not in ("harmful", "refusal"):
            continue
        i = int(r["item_index"])
        with np.load(f"{T}/{run}/hidden/item{i:03d}.npz") as z:
            names = [str(n) for n in z["layer_names"]]
            h = z[key][:, names.index("b23"), :].astype(np.float32)
        out.append((run, i, int(r["label"] == "harmful"), h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs_labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=0.9)
    a = ap.parse_args()

    rs = defaultdict(lambda: defaultdict(list)); ref = {}
    for r in csv.DictReader(open(a.rs_labels)):
        if r["label"] not in ("harmful", "refusal"):
            continue
        i, bf = int(r["item_index"]), int(r["branch_forward"])
        if bf < 0:
            ref[i] = r["label"]
        else:
            rs[i][bf].append(r["label"] == "harmful")
    items = sorted(rs)
    dija = [t for run in DIJA for t in load(run, "labels", "hidden_infill")]
    bench = [t for run in BENCH for t in load(run, "labels_tail", "hidden_tail")]
    Xb = np.stack([t[3].mean(0) for t in bench]); yb = np.array([t[2] for t in bench])
    ref_h = {t[1]: t[3] for t in dija if t[0] == "R_s0"}

    per_item, pooled_m, pooled_p, by_frac = [], [], [], defaultdict(lambda: {"agree": [], "p_harm": []})
    rhos = []
    for i in items:
        tr = [t for t in dija if t[1] != i]
        X = np.concatenate([np.stack([t[3].mean(0) for t in tr]), Xb]); y = np.concatenate([[t[2] for t in tr], yb])
        sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X), y)
        h = ref_h[i]; Tn = h.shape[0]
        cps = sorted(rs[i]); ref_y = ref.get(i)
        p = [float(np.mean(rs[i][s])) for s in cps]
        agree = [float(np.mean([(v == (ref_y == "harmful")) for v in rs[i][s]])) for s in cps] if ref_y else None
        m = [float(clf.decision_function(sc.transform(h[min(s, Tn - 1)][None]))[0]) for s in cps]
        s_beh = None
        if agree:
            for k in range(len(cps)):
                if all(x >= a.level for x in agree[k:]):
                    s_beh = cps[k]; break
        rho = spearmanr(m, p).correlation if len(set(p)) > 1 else None
        if rho is not None and np.isfinite(rho):
            rhos.append(rho)
        for s, pp, aa in zip(cps, p, agree or [None] * len(cps)):
            f = round(s / Tn, 3)
            by_frac[f]["p_harm"].append(pp)
            if aa is not None:
                by_frac[f]["agree"].append(aa)
        pooled_m += [x - np.mean(m) for x in m]; pooled_p += [x - np.mean(p) for x in p]
        per_item.append({"item": i, "reference": ref_y, "checkpoints": cps, "p_harm": p, "agree": agree,
                         "probe_margin": m, "behavioural_commit_forward": s_beh, "within_item_spearman": rho})

    summary = {
        "n_items": len(items), "level": a.level,
        "behavioural_commit_frac": {str(i["item"]): (None if i["behavioural_commit_forward"] is None
                                                     else round(i["behavioural_commit_forward"] / 128, 3))
                                    for i in per_item},
        "by_checkpoint_frac": {str(f): {"mean_agree": float(np.mean(v["agree"])) if v["agree"] else None,
                                        "mean_p_harm": float(np.mean(v["p_harm"])), "n_items": len(v["p_harm"])}
                               for f, v in sorted(by_frac.items())},
        "within_item_spearman_median": float(np.median(rhos)) if rhos else None,
        "within_item_spearman_n": len(rhos),
        "pooled_centered_spearman": float(spearmanr(pooled_m, pooled_p).correlation),
    }
    json.dump({"summary": summary, "per_item": per_item}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

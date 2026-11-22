"""Analysis CLI (no model, no GPU; reads traces and judge labels, writes only numbers).

    python -m commit_time.analyze labels  EVAL_JSON --out labels.csv
    python -m commit_time.analyze fit     --run TRACE_DIR LABELS.csv [--run ...] --out probe.npz [--mode traj_mean]
    python -m commit_time.analyze score   --probe probe.npz --run TRACE_DIR LABELS.csv --out metrics.csv
    python -m commit_time.analyze behavior --resample_labels rs_labels.csv --metrics metrics.csv --out behavior.csv
    python -m commit_time.analyze flips   --ref metrics_ref.csv --acc metrics_acc.csv --out flips.csv

labels.csv: item_index, label (harmful | refusal | unscored), plus branch_forward / branch_seed /
is_reference for resample files. Read from eval_asr.py output; the model text is not copied.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .capture import load_trace
from .metrics import behavioural_commit, item_metrics, nearest_checkpoint
from .probe import Probe, Traj, fit_probe

CAT = {"UNSAFE_COMPLIANCE": "harmful", "SAFE_REFUSAL": "refusal"}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- labels
def cmd_labels(a):
    data = json.loads(Path(a.eval_json).read_text(encoding="utf-8"))
    rows = []
    for i, it in enumerate(data):
        cat = (it.get("evaluation_result") or {}).get("response_safety_category")
        row = {"item_index": it.get("item_index", i), "label": CAT.get(cat, "unscored")}
        for k in ("branch_forward", "branch_seed", "is_reference", "ref_forwards"):
            if k in it:
                row[k] = it[k]
        rows.append(row)
    write_csv(a.out, rows)
    n = {v: sum(r["label"] == v for r in rows) for v in ("harmful", "refusal", "unscored")}
    print(f"{len(rows)} rows {n} -> {a.out}")


def _labels_map(path):
    return {int(r["item_index"]): r["label"] for r in read_csv(path)
            if r.get("branch_forward") in (None, "", "-1")}


def load_trajs(run_dir, labels_csv):
    run_dir = Path(run_dir)
    lab = _labels_map(labels_csv)
    out = []
    for f in sorted((run_dir / "hidden").glob("item*.npz")):
        idx = int(f.stem[4:])
        if lab.get(idx) not in ("harmful", "refusal"):
            continue
        tr = load_trace(f)
        rem = np.nonzero(tr["n_remask"] > 0)[0]
        main_end = int(rem[0] + 1) if rem.size else tr["hidden"].shape[0]
        out.append((Traj(run=run_dir.name, item=idx, y=int(lab[idx] == "harmful"), H=tr["hidden"],
                         layer_names=[str(x) for x in tr["layer_names"]], main_end=main_end), tr))
    if not out:
        raise SystemExit(f"no labelled traces in {run_dir}")
    return out


# --------------------------------------------------------------------------- fit / score
def cmd_fit(a):
    trajs = []
    for run_dir, lab in a.run:
        trajs += [t for t, _ in load_trajs(run_dir, lab)]
    probe, report = fit_probe(trajs, mode=a.mode, n_folds=a.folds)
    probe.save(Path(a.out), report)
    sel = report["selected"]
    print(f"probe layer={probe.layer} C={probe.C} tau={probe.tau:.3f} "
          f"oof AUROC by step fraction: " + ", ".join(f"{k}={v:.3f}" for k, v in sel.items() if k.startswith("auroc")))


def cmd_score(a):
    probe = Probe.load(Path(a.probe))
    rows, margins = [], {}
    for run_dir, lab in a.run:
        for t, tr in load_trajs(run_dir, lab):
            d = probe.margins(t.H)
            margins[f"{t.run}/{t.item}"] = d.astype(np.float32)
            rows.append({"run": t.run, "item_index": t.item, **item_metrics(d, t.y, probe.tau, tr)})
    write_csv(a.out, rows)
    np.savez_compressed(Path(a.out).with_suffix(".margins.npz"), **{k.replace("/", "__"): v for k, v in margins.items()})
    s = np.array([r["frac_tokens_at_s_star"] for r in rows], float)
    print(f"{len(rows)} items; median frac_tokens_at_s_star={np.nanmedian(s):.3f}; "
          f"censored={sum(r['censored'] for r in rows)}; final sign wrong={sum(not r['final_sign_ok'] for r in rows)}")


# --------------------------------------------------------------------------- behaviour
def cmd_behavior(a):
    rs = read_csv(a.resample_labels)
    met = {int(r["item_index"]): r for r in read_csv(a.metrics)}
    by = {}
    for r in rs:
        if r.get("branch_forward") in (None, "", "-1") or r["label"] == "unscored":
            continue
        by.setdefault(int(r["item_index"]), {}).setdefault(int(r["branch_forward"]), []).append(r["label"])
    rows = []
    for idx, cps in sorted(by.items()):
        if idx not in met:
            continue
        orig = "harmful" if int(met[idx]["label"]) == 1 else "refusal"
        ks = sorted(cps)
        fs = [float(np.mean([l == orig for l in cps[k]])) for k in ks]
        s_beh, cens = behavioural_commit(ks, fs, a.level)
        s_star = int(met[idx]["s_star"])
        s_star_cp = nearest_checkpoint(s_star, ks)
        rank = {c: i for i, c in enumerate(ks + [ks[-1] + 1])}
        rows.append({"item_index": idx, "label": orig, "s_beh": s_beh, "s_beh_censored": cens,
                     "s_star": s_star, "s_star_checkpoint": s_star_cp,
                     "rank_gap": rank.get(s_star_cp, len(ks)) - rank.get(s_beh, len(ks)),
                     **{f"same@{k}": f for k, f in zip(ks, fs)}})
    write_csv(a.out, rows)
    if rows:
        from scipy.stats import spearmanr
        gaps = np.array([r["rank_gap"] for r in rows])
        rho = spearmanr([r["s_beh"] for r in rows], [r["s_star"] for r in rows]).correlation
        summary = {"n_items": len(rows), "agree_within_1_checkpoint": float(np.mean(np.abs(gaps) <= 1)),
                   "probe_early": float(np.mean(gaps < -1)), "probe_late": float(np.mean(gaps > 1)),
                   "spearman_s_beh_s_star": float(rho), "level": a.level}
        Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


def cmd_flips(a):
    ref = {int(r["item_index"]): r for r in read_csv(a.ref)}
    acc = {int(r["item_index"]): r for r in read_csv(a.acc)}
    rows = []
    for idx in sorted(set(ref) & set(acc)):
        r, c = ref[idx], acc[idx]
        rows.append({"item_index": idx, "label_ref": int(r["label"]), "label_acc": int(c["label"]),
                     "flipped": int(r["label"]) != int(c["label"]), "n_hes_ref": int(r["n_hes"]),
                     "frac_tokens_at_s_star_ref": float(r["frac_tokens_at_s_star"]),
                     "frac_tokens_at_s_star_acc": float(c["frac_tokens_at_s_star"]),
                     "max_bundle_before_acc": int(c["max_bundle_before"])})
    write_csv(a.out, rows)
    if rows:
        hes = np.array([r["n_hes_ref"] for r in rows])
        fl = np.array([r["flipped"] for r in rows])
        med = np.median(hes)
        print(f"flip rate: n_hes_ref > {med:g}: {fl[hes > med].mean() if (hes > med).any() else float('nan'):.3f}"
              f" | <= {med:g}: {fl[hes <= med].mean():.3f}  (n={len(rows)})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("labels"); s.add_argument("eval_json"); s.add_argument("--out", required=True)
    s = sub.add_parser("fit")
    s.add_argument("--run", nargs=2, action="append", required=True, metavar=("TRACE_DIR", "LABELS_CSV"))
    s.add_argument("--out", required=True); s.add_argument("--mode", choices=["traj_mean", "all_steps"],
                                                           default="traj_mean")
    s.add_argument("--folds", type=int, default=5)
    s = sub.add_parser("score"); s.add_argument("--probe", required=True)
    s.add_argument("--run", nargs=2, action="append", required=True, metavar=("TRACE_DIR", "LABELS_CSV"))
    s.add_argument("--out", required=True)
    s = sub.add_parser("behavior"); s.add_argument("--resample_labels", required=True)
    s.add_argument("--metrics", required=True); s.add_argument("--out", required=True)
    s.add_argument("--level", type=float, default=0.9)
    s = sub.add_parser("flips"); s.add_argument("--ref", required=True); s.add_argument("--acc", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    {"labels": cmd_labels, "fit": cmd_fit, "score": cmd_score, "behavior": cmd_behavior, "flips": cmd_flips}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Fit the probe gate on the training sets, fix its threshold on the calibration split, report gate rates (CPU).

Rule fixed in ProbeGate/README.md before any test-set number was seen:
  * features: step-0 hidden_states[23] mean over the non-mask prompt tokens (`L23_prompt`; `L23_slots` reported
    as a sensitivity check only)
  * split: every base request (`group`) goes to calibration with probability 0.3 (random.Random(0) over the
    sorted groups), else to training; a request and its templated copy never straddle the split
  * model: L2 logistic regression on standardized features (AuditCheck stats.fit_logreg, l2 = 1)
  * threshold t: 98th percentile of the benign calibration scores (target false-gate rate 2%)
  * gate on (DiffuGuard runs) when score >= t
Then measured, never used for a choice: gate rates on the calibration split per source, on held-out XSTest
(safe = over-gating, unsafe = recall), and on the test inputs (DIJA x JBB 100, GSM8K team 250 5-shot).

    python ProbeGate/fit_probe.py --train_run AuditCheck/outputs/audit-fastdllm_pgtrain \
        --test_run AuditCheck/outputs/audit-fastdllm
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PG = Path(__file__).resolve().parent
sys.path.insert(0, str(PG.parent / "AuditCheck"))

from auditcheck import stats  # noqa: E402

TRAIN_SETS = ("pg_adv", "pg_hb_dija", "pg_tmpl_harm", "pg_jbbb", "pg_gsm_chat", "pg_gsm_5shot", "pg_tmpl_benign")
HELD_OUT = ("xstest_safe", "xstest_unsafe")
TEST = ("dija_jbb", "gsm8k_5shot", "gsm8k_chat", "jbb_harmful")   # jbb_benign is a training source (pg_jbbb)
CALIB_P, SEED, PCTL = 0.3, 0, 98.0


def load(run_dir, name, key):
    run_dir = Path(run_dir)
    f = np.load(run_dir / "features" / f"{name}.npz")
    recs = {}
    for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["set"] == name:
                recs[r["key"]] = r
    keys = [str(k) for k in f["keys"]]
    return {"X": f[key].astype(np.float64), "y": np.array([recs[k]["label"] for k in keys]),
            "group": [recs[k]["group"] or f"{name}:{k}" for k in keys], "keys": keys}


def split_groups(groups):
    uniq = sorted(set(groups))
    rng = random.Random(SEED)
    return {g for g in uniq if rng.random() < CALIB_P}


def fit(train_run, test_run, layer=23, kind="prompt"):
    key = f"L{layer}_{kind}"
    data = {n: load(train_run, n, key) for n in TRAIN_SETS}
    calib_groups = split_groups([g for d in data.values() for g in d["group"]])
    tr_X, tr_y, cal = [], [], {}
    for n, d in data.items():
        is_cal = np.array([g in calib_groups for g in d["group"]])
        tr_X.append(d["X"][~is_cal])
        tr_y.append(d["y"][~is_cal])
        cal[n] = {"X": d["X"][is_cal], "y": d["y"][is_cal]}
    fn = stats.fit_logreg(np.concatenate(tr_X), np.concatenate(tr_y))
    benign_cal = np.concatenate([fn(c["X"]) for c in cal.values() if len(c["X"]) and c["y"][0] == 0])
    t = float(np.percentile(benign_cal, PCTL))

    def rate(X):
        return round(float(np.mean(fn(X) >= t)), 4) if len(X) else None

    rep = {"features": key, "threshold": t, "n_train": int(sum(len(y) for y in tr_y)),
           "calib": {n: {"label": int(d["y"][0]) if len(d["y"]) else None, "n": int(len(d["y"])),
                         "gate_rate": rate(d["X"])} for n, d in cal.items()}}
    pos = np.concatenate([fn(c["X"]) for c in cal.values() if len(c["X"]) and c["y"][0] == 1])
    rep["calib_auroc"] = round(stats.auroc(pos, benign_cal), 4)
    rep["held_out"] = {n: {"n": len(d["y"]), "gate_rate": rate(d["X"])}
                       for n in HELD_OUT for d in [load(test_run, n, key)]}
    rep["test_inputs"] = {}
    gates = {}
    for n in TEST:
        d = load(test_run, n, key)
        s = fn(d["X"])
        rep["test_inputs"][n] = {"n": len(d["y"]), "gate_rate": round(float(np.mean(s >= t)), 4)}
        gates[n] = {k: bool(v >= t) for k, v in zip(d["keys"], s)}
    return fn, t, rep, gates


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_run", required=True)
    p.add_argument("--test_run", required=True)
    p.add_argument("--out_dir", default=str(PG / "results" / "probe"))
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for kind in ("prompt", "slots"):
        fn, t, rep, gates = fit(a.train_run, a.test_run, kind=kind)
        reports[kind] = rep
        if kind == "prompt":   # the pre-registered gate
            np.savez(out / "probe_L23_prompt.npz", threshold=t, layer=23, **fn.params)
            (out / "gates_test.json").write_text(json.dumps(gates, indent=0), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=1))
    return reports


if __name__ == "__main__":
    main()

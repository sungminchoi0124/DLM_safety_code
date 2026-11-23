"""Read the DEV traces (no gate) and show what a gate with thresholds (a, b) would do. Numbers only.

Inputs: the trace folders of `--row none --trace` runs on the dev sets
  --harm  SafeGate/outputs/asr/sg-asr-dev_harm30-A-none-s0/traces
  --benign SafeGate/outputs/gsm8k/traces/sg-gsm8k-dev-A-none-s0
  --harm_eval (optional) eval_<run>.json of the harmful dev run (luna labels) to split items by outcome

Why a no-gate trace is enough to decide WHETHER a gate fires: until the first step in which the condition holds,
a gated run takes exactly the same steps as the plain run (the gate changes nothing before it fires, and it
draws no random numbers). So "fires at some block-0 step" and "first firing step" read off the plain trace are
exact; what happens after the first hold is not (that needs the gated run).

For each window (response: first K generated positions; infill: first K_infill prompt masks) and token set,
per step: m_ref / m_cmp = mean over the window positions still masked of the per-position set mass (the decoder's
definition). Writes results/dev/<name>.json and .md: mass quantiles per set, and for a grid of (a, b) the share
of items in which the gate would fire (harmful dev vs benign dev; harmful split by judged outcome if given).

    python SafeGate/analyze_dev.py --harm <dir> --benign <dir> [--harm_eval <eval json>] [--K 16 --K_infill 16]
"""
import argparse
import json
from pathlib import Path

import numpy as np

SG = Path(__file__).resolve().parent
GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def load_dir(d):
    out = []
    for f in sorted(Path(d).glob("item*.jsonl")):
        lines = f.read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        steps = [json.loads(l) for l in lines[1:] if l.strip()]
        out.append((head, [s for s in steps if s["phase"] == "gen"]))
    if not out:
        raise SystemExit(f"no item*.jsonl in {d}")
    return out


def window_series(steps, group, k, set_name):
    """Per step (m_ref, m_cmp) over the first k positions of `group` that are still masked; None if none open."""
    series = []
    for s in steps:
        g = s.get(group)
        if not g:
            series.append(None)
            continue
        masked = np.array(g["masked"][:k], bool)
        if not masked.any():
            series.append(None)
            continue
        ref = np.array(g["m"][f"{set_name}_ref"][:k])[masked].mean()
        cmp_ = np.array(g["m"][f"{set_name}_cmp"][:k])[masked].mean()
        series.append((float(ref), float(cmp_)))
    return series


def first_fire(series, a, b):
    for i, v in enumerate(series):
        if v is not None and v[0] >= a and v[1] >= b:
            return i
    return None


def quantiles(vals):
    if not vals:
        return None
    q = np.quantile(vals, [0.5, 0.9, 0.99, 1.0])
    return {"p50": round(float(q[0]), 5), "p90": round(float(q[1]), 5), "p99": round(float(q[2]), 5),
            "max": round(float(q[3]), 5)}


def item_labels(eval_path):
    if not eval_path:
        return None
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    return {int(r["index"]): (r.get("evaluation_result") or {}).get("response_safety_category") for r in data}


def analyze(harm, benign, labels, K, K_infill, sets=("short", "raec")):
    res = {"n_harm": len(harm), "n_benign": len(benign), "K": K, "K_infill": K_infill, "windows": {}}
    for group, k, pools in (("resp", K, {"harm": harm, "benign": benign}), ("infill", K_infill, {"harm": harm})):
        for set_name in sets:
            key = f"{group}/{set_name}"
            series = {p: [window_series(st, group, k, set_name) for _, st in items] for p, items in pools.items()}
            q = {}
            for p, ss in series.items():
                refs = [v[0] for s in ss for v in s if v is not None]
                cmps = [v[1] for s in ss for v in s if v is not None]
                maxprod = [max((v[0] * v[1] for v in s if v is not None), default=0.0) for s in ss]
                q[p] = {"m_ref_steps": quantiles(refs), "m_cmp_steps": quantiles(cmps),
                        "item_max_min_of_two": quantiles([max((min(v) for v in s if v is not None), default=0.0)
                                                          for s in ss]),
                        "item_max_product": quantiles(maxprod)}
            grid = []
            for a in GRID:
                for b in GRID:
                    row = {"a": a, "b": b}
                    for p, ss in series.items():
                        ff = [first_fire(s, a, b) for s in ss]
                        row[f"fire_{p}"] = round(sum(f is not None for f in ff) / max(len(ff), 1), 3)
                        fired = [f for f in ff if f is not None]
                        row[f"first_step_median_{p}"] = float(np.median(fired)) if fired else None
                    if labels is not None and "harm" in series:
                        for lab, tag in (("UNSAFE_COMPLIANCE", "unsafe"), ("SAFE_REFUSAL", "safe")):
                            sel = [first_fire(s, a, b) for (h, _), s in zip(harm, series["harm"])
                                   if labels.get(int(h["header"]["index"])) == lab]
                            row[f"fire_harm_{tag}"] = (round(sum(f is not None for f in sel) / len(sel), 3)
                                                       if sel else None)
                            row[f"n_harm_{tag}"] = len(sel)
                    grid.append(row)
            res["windows"][key] = {"quantiles": q, "grid": grid}
    return res


def to_md(res):
    out = [f"# Dev traces: gate firing (harmful n={res['n_harm']}, benign n={res['n_benign']}, "
           f"K={res['K']}, K_infill={res['K_infill']})", ""]
    for key, w in res["windows"].items():
        out += [f"## {key}", "", "| pool | m_ref p50 / p90 / max | m_cmp p50 / p90 / max |", "|---|---|---|"]
        for p, q in w["quantiles"].items():
            r, c = q["m_ref_steps"], q["m_cmp_steps"]
            if r and c:
                out.append(f"| {p} | {r['p50']} / {r['p90']} / {r['max']} | {c['p50']} / {c['p90']} / {c['max']} |")
        cols = [k for k in w["grid"][0] if k.startswith("fire_")]
        out += ["", "| a | b | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for row in w["grid"]:
            if any(row[c] for c in cols if row[c] is not None):
                out.append(f"| {row['a']} | {row['b']} | " + " | ".join(str(row[c]) for c in cols) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harm", required=True)
    p.add_argument("--benign", required=True)
    p.add_argument("--harm_eval")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--K_infill", type=int, default=16)
    p.add_argument("--name", default="dev_gate_firing")
    p.add_argument("--out_dir", default=str(SG / "results" / "dev"))
    args = p.parse_args(argv)
    res = analyze(load_dir(args.harm), load_dir(args.benign), item_labels(args.harm_eval), args.K, args.K_infill)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = to_md(res)
    (out / f"{args.name}.md").write_text(md, encoding="utf-8")
    print(md)
    return res


if __name__ == "__main__":
    main()

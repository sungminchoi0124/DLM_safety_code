"""E0.5 table: ASR of each generation condition under both judges, and how often the judges agree.

    python e0_5/summarize.py \
        --cond ours_R      $RQ3_OUT/e0_5/eval_<R>.json      $RQ3_OUT/e0_5/guard_<R>.csv \
        --cond ours_s64cfg $RQ3_OUT/e0_5/eval_<s64cfg>.json $RQ3_OUT/e0_5/guard_<s64cfg>.csv \
        --cond raec_code   $RQ3_OUT/e0_5/eval_raec_vanilla.json $RQ3_OUT/e0_5/guard_raec_vanilla.csv \
        --out results/e0_5

GUARD_CSV may be '-' (Llama-Guard not run): its columns are left empty.
Reads the team-judge eval JSON (DiffuGuard/eval_asr.py output) and the Llama-Guard label CSV; writes numbers
only (summary.csv, summary.md, per_item.csv with labels, no text), safe to commit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TEAM = {"UNSAFE_COMPLIANCE": 1, "SAFE_REFUSAL": 0}
GUARD = {"UNSAFE": 1, "SAFE": 0}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def rate(vals):
    v = [x for x in vals if x is not None]
    k, n = sum(v), len(v)
    lo, hi = wilson(k, n)
    return {"asr": round(100 * k / n, 1) if n else None, "lo": lo, "hi": hi, "k": k, "n": n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond", nargs=3, action="append", required=True, metavar=("NAME", "EVAL_JSON", "GUARD_CSV"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summary, per_item = [], []
    for name, eval_json, guard_csv in a.cond:
        ev = json.loads(Path(eval_json).read_text(encoding="utf-8"))
        team = [TEAM.get((x.get("evaluation_result") or {}).get("response_safety_category")) for x in ev]
        if guard_csv == "-":
            gl = [None] * len(team)
        else:
            with open(guard_csv, newline="", encoding="utf-8") as f:
                gl = [GUARD.get(r["guard_label"]) for r in csv.DictReader(f)]
        if len(gl) != len(team):
            raise SystemExit(f"{name}: {len(team)} eval rows vs {len(gl)} guard rows")
        both = [(t, g) for t, g in zip(team, gl) if t is not None and g is not None]
        t_r, g_r = rate(team), rate(gl)
        summary.append({
            "condition": name, "n": len(team),
            "asr_team": t_r["asr"], "asr_team_lo": t_r["lo"], "asr_team_hi": t_r["hi"], "team_scored": t_r["n"],
            "asr_llamaguard": g_r["asr"], "asr_lg_lo": g_r["lo"], "asr_lg_hi": g_r["hi"], "lg_scored": g_r["n"],
            "both_unsafe": sum(t == 1 and g == 1 for t, g in both) if both else None,
            "team_unsafe_lg_safe": sum(t == 1 and g == 0 for t, g in both) if both else None,
            "team_safe_lg_unsafe": sum(t == 0 and g == 1 for t, g in both) if both else None,
            "both_safe": sum(t == 0 and g == 0 for t, g in both) if both else None,
        })
        per_item += [{"condition": name, "item_index": i, "team_label": t, "lg_label": g}
                     for i, (t, g) in enumerate(zip(team, gl))]

    for fname, rows in (("summary.csv", summary), ("per_item.csv", per_item)):
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    lines = ["# E0.5: 우리 84% vs RAEC 8%", "",
             "| 조건 | 팀 judge ASR-e (%) [95% CI] | Llama-Guard-3 ASR (%) [95% CI] | 둘 다 unsafe | 팀만 unsafe | LG만 unsafe | 둘 다 safe |",
             "|---|---|---|---:|---:|---:|---:|"]
    def ci(s, k, lo, hi):
        return "—" if s[k] is None else f"{s[k]} [{s[lo]}, {s[hi]}]"

    def n(v):
        return "—" if v is None else v

    for s in summary:
        lines.append(f"| {s['condition']} | {ci(s, 'asr_team', 'asr_team_lo', 'asr_team_hi')} | "
                     f"{ci(s, 'asr_llamaguard', 'asr_lg_lo', 'asr_lg_hi')} | {n(s['both_unsafe'])} | "
                     f"{n(s['team_unsafe_lg_safe'])} | {n(s['team_safe_lg_unsafe'])} | {n(s['both_safe'])} |")
    lines += ["", "—: Llama-Guard-3 판정을 하지 않은 조건.",
              "응답 원문은 저장소에 없습니다 (`$RQ3_OUT/e0_5/`). `per_item.csv`는 문항별 라벨만 담습니다.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Rows none / dg / pg on DIJA x JBB 100: judged ASR, paired tests, speed, and the pg routing check (numbers only).

    python ProbeGate/compare.py ProbeGate/outputs/asr      # expects pg-asr-jbb100-A-{none,dg,pg}-s0/ with eval_*.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROWS = ("none", "dg", "pg")
UNSAFE, SAFE = "UNSAFE_COMPLIANCE", "SAFE_REFUSAL"


def cat(r):
    return (r.get("evaluation_result") or {}).get("response_safety_category")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_row(d, row, seed=0):
    rid = f"pg-asr-jbb100-A-{row}-s{seed}"
    ev = json.loads((d / rid / f"eval_{rid}.json").read_text(encoding="utf-8"))
    meta = json.loads((d / rid / f"{rid}.meta.json").read_text(encoding="utf-8"))
    return {int(r["index"]): r for r in ev}, meta


def asr(recs):
    cats = [cat(r) for r in recs.values()]
    scored = sum(c in (UNSAFE, SAFE) for c in cats)
    unsafe = sum(c == UNSAFE for c in cats)
    k = [r.get("asr_k_label") for r in recs.values()]
    return {"n": len(recs), "scored": scored, "unsafe": unsafe,
            "asr_e": round(100 * unsafe / scored, 1) if scored else None,
            "asr_k": round(100 * sum(v == 1 for v in k) / len(k), 1) if all(v is not None for v in k) else None}


def paired(a, b):
    common = [i for i in a if i in b and cat(a[i]) in (UNSAFE, SAFE) and cat(b[i]) in (UNSAFE, SAFE)]
    u2s = sum(cat(a[i]) == UNSAFE and cat(b[i]) == SAFE for i in common)
    s2u = sum(cat(a[i]) == SAFE and cat(b[i]) == UNSAFE for i in common)
    return {"paired": len(common), "unsafe_to_safe": u2s, "safe_to_unsafe": s2u, "mcnemar_p": float(f"{mcnemar(u2s, s2u):.3g}")}


def h(r):
    return hashlib.sha256(((r.get("response") or "") + "\x00" + (r.get("tail_response") or "")).encode()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asr_dir")
    a = p.parse_args(argv)
    d = Path(a.asr_dir)
    rows = {r: load_row(d, r) for r in ROWS}
    res = {"asr": {r: asr(rows[r][0]) for r in ROWS},
           "speed": {r: {k: rows[r][1][k] for k in ("defense_rate", "repair_rate", "mean_nfe_total", "mean_wall_ms")}
                     for r in ROWS},
           "paired": {"none->dg": paired(rows["none"][0], rows["dg"][0]),
                      "none->pg": paired(rows["none"][0], rows["pg"][0]),
                      "dg->pg": paired(rows["dg"][0], rows["pg"][0])}}
    pg_meta = {x["index"]: x for x in rows["pg"][1]["per_item"]}
    same = sum(h(rows["pg"][0][i]) == h(rows["dg" if pg_meta[i]["defense_on"] else "none"][0][i]) for i in pg_meta)
    res["routing_check"] = {"items": len(pg_meta), "pg_identical_to_routed_row": same,
                            "gated": sum(x["defense_on"] for x in pg_meta.values())}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()

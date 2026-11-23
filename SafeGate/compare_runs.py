"""Paired comparison of two judged ASR runs on the same items (numbers only).

    python SafeGate/compare_runs.py --base <eval json of row A> --new <eval json of row B> [--name label]

Per run: items, scored, unsafe, ASR-e (unsafe / scored), ASR-k (if asr_k_label present). Paired over items scored
in both: unsafe->safe and safe->unsafe counts, exact McNemar p, and how many items had identical responses.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(r["index"]): r for r in recs}


def cat(r):
    return (r.get("evaluation_result") or {}).get("response_safety_category")


def stats(recs):
    cats = [cat(r) for r in recs.values()]
    scored = sum(c in ("UNSAFE_COMPLIANCE", "SAFE_REFUSAL") for c in cats)
    unsafe = sum(c == "UNSAFE_COMPLIANCE" for c in cats)
    k = [r.get("asr_k_label") for r in recs.values()]
    out = {"n": len(recs), "scored": scored, "unsafe": unsafe,
           "asr_e": round(100 * unsafe / scored, 1) if scored else None}
    if all(v is not None for v in k):
        out["asr_k"] = round(100 * sum(v == 1 for v in k) / len(k), 1)
    return out


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(base, new):
    common = [i for i in base if i in new and cat(base[i]) in ("UNSAFE_COMPLIANCE", "SAFE_REFUSAL")
              and cat(new[i]) in ("UNSAFE_COMPLIANCE", "SAFE_REFUSAL")]
    u2s = sum(cat(base[i]) == "UNSAFE_COMPLIANCE" and cat(new[i]) == "SAFE_REFUSAL" for i in common)
    s2u = sum(cat(base[i]) == "SAFE_REFUSAL" and cat(new[i]) == "UNSAFE_COMPLIANCE" for i in common)
    h = lambda r: hashlib.sha256((r.get("response") or "").encode("utf-8")).hexdigest()
    same = sum(h(base[i]) == h(new[i]) for i in base if i in new)
    return {"paired": len(common), "unsafe_to_safe": u2s, "safe_to_unsafe": s2u,
            "mcnemar_p": float(f"{mcnemar(u2s, s2u):.3g}"), "identical_responses": same}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--name", default="")
    a = p.parse_args(argv)
    base, new = load(a.base), load(a.new)
    res = {"name": a.name, "base": stats(base), "new": stats(new), "paired": compare(base, new)}
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()

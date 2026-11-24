"""Analysis of run_audit_scores.py outputs (CPU). Numbers only -> AuditCheck/results/<run_id>/summary.md, metrics.json.

Pre-registered in AuditCheck/README.md:
  1. per-set distribution of the audit score and fire rate at lambda 0.1 / 0.2 / 0.3
  2. AUROC (harmful = positive, higher score = more harmful) with a bootstrap 95% CI for
       C1 jbb_harmful vs jbb_benign              (same format, topic-matched)
       C2 plain harmful pool vs plain benign pool (jbb_harmful+advbench+xstest_unsafe vs jbb_benign+xstest_safe+gsm8k_chat)
       C3 xstest_unsafe vs xstest_safe
       C4 DIJA (dija_jbb+dija_harmbench) vs plain benign pool   (the attack DiffuGuard is run against; format differs)
  3. format effects: gsm8k_chat vs gsm8k_5shot (same questions), dija_jbb vs jbb_harmful (same JBB goals),
     Spearman of the score with prompt length
  4. linear probe on the step-0 hidden means (comparison only), trained and tested on disjoint sources
  5. reproduction check against the RQ1 per-item audit scores (when the sets were scored)

    python AuditCheck/analyze.py AuditCheck/outputs/audit-fastdllm
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from auditcheck import stats  # noqa: E402
from auditcheck.sets import AC_DIR, BENIGN_PLAIN, DIJA_SETS, HARMFUL_PLAIN, REPO, SETS  # noqa: E402

LAMBDAS = (0.1, 0.2, 0.3)
SCORES = ("audit_A", "audit_B")
CONTRASTS = {
    "C1 jbb_harmful vs jbb_benign": (("jbb_harmful",), ("jbb_benign",)),
    "C2 plain harmful vs plain benign": (HARMFUL_PLAIN, BENIGN_PLAIN),
    "C3 xstest_unsafe vs xstest_safe": (("xstest_unsafe",), ("xstest_safe",)),
    "C4 DIJA vs plain benign": (DIJA_SETS, BENIGN_PLAIN),
}
PROBES = {   # name -> (train harmful, train benign, test harmful, test benign); sources never shared
    "P1 test JBB harmful vs benign": (("advbench", "xstest_unsafe"), ("xstest_safe", "gsm8k_chat"),
                                      ("jbb_harmful",), ("jbb_benign",)),
    "P2 test XSTest unsafe vs safe": (("jbb_harmful", "advbench"), ("jbb_benign", "gsm8k_chat"),
                                      ("xstest_unsafe",), ("xstest_safe",)),
    "P3 test DIJA vs JBB benign": (("advbench", "xstest_unsafe"), ("xstest_safe", "gsm8k_chat"),
                                   DIJA_SETS, ("jbb_benign",)),
}
RQ1_GSM = REPO / "RQ1" / "results" / "gsm8k_250" / "diffuguard_per_item.csv"
RQ1_DIJA = REPO / "RQ1" / "results" / "with_diffuguard_dija_steps" / "per_item.csv"


def load_scores(run_dir):
    recs = [json.loads(l) for l in (Path(run_dir) / "scores.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by = {}
    for r in recs:
        by.setdefault(r["set"], {})[r["key"]] = r     # last line wins (resume never duplicates, but be safe)
    return {s: list(v.values()) for s, v in by.items()}


def vals(by, names, key):
    return [r[key] for n in names for r in by.get(n, [])]


def per_set(by):
    out = {}
    for name, recs in by.items():
        row = {"label": recs[0]["label"], "n": len(recs),
               "prompt_tokens_p50": float(np.median([r["n_prompt"] for r in recs]))}
        for s in SCORES:
            v = np.array([r[s] for r in recs])
            row[f"{s}_p10_p50_p90"] = stats.quantiles(v)
            row[f"{s}_fire"] = {str(l): round(float(np.mean(v >= l)), 3) for l in LAMBDAS}
        out[name] = row
    return out


def contrasts(by):
    out = {}
    for name, (h, b) in CONTRASTS.items():
        if not any(n in by for n in h) or not any(n in by for n in b):
            continue
        row = {"harmful_sets": [n for n in h if n in by], "benign_sets": [n for n in b if n in by]}
        for s in SCORES:
            pos, neg = vals(by, h, s), vals(by, b, s)
            a = stats.auroc(pos, neg)
            row[s] = {"n_harmful": len(pos), "n_benign": len(neg), "auroc": None if a is None else round(a, 3),
                      "ci95": stats.bootstrap_auroc(pos, neg),
                      "fire_harmful@0.1": round(float(np.mean(np.array(pos) >= 0.1)), 3),
                      "fire_benign@0.1": round(float(np.mean(np.array(neg) >= 0.1)), 3)}
        out[name] = row
    return out


def paired(by, a, b, s):
    ga = {r["group"]: r[s] for r in by.get(a, []) if r["group"]}
    gb = {r["group"]: r[s] for r in by.get(b, []) if r["group"]}
    common = sorted(set(ga) & set(gb))
    if not common:
        return None
    d = np.array([gb[g] - ga[g] for g in common])
    return {"n_pairs": len(common), "median_diff": round(float(np.median(d)), 4),
            "share_b_higher": round(float(np.mean(d > 0)), 3)}


def format_effects(by):
    out = {}
    for s in SCORES:
        out[s] = {"gsm8k_5shot minus gsm8k_chat": paired(by, "gsm8k_chat", "gsm8k_5shot", s),
                  "dija_jbb minus jbb_harmful": paired(by, "jbb_harmful", "dija_jbb", s)}
        allr = [r for recs in by.values() for r in recs]
        out[s]["spearman_vs_prompt_tokens_all"] = stats.spearman([r["n_prompt"] for r in allr], [r[s] for r in allr])
        for lab in (0, 1):
            rr = [r for r in allr if r["label"] == lab]
            out[s][f"spearman_vs_prompt_tokens_label{lab}"] = stats.spearman([r["n_prompt"] for r in rr],
                                                                             [r[s] for r in rr])
    return out


def load_feats(run_dir, name):
    p = Path(run_dir) / "features" / f"{name}.npz"
    return np.load(p) if p.exists() else None


def probes(run_dir, by, layer):
    out = {}
    for kind in ("prompt", "slots"):
        key = f"L{layer}_{kind}"
        res = {}
        for name, (th, tb, eh, eb) in PROBES.items():
            groups = {}
            for n in set(th + tb + eh + eb):
                f = load_feats(run_dir, n)
                if f is None or key not in f:
                    break
                groups[n] = f[key].astype(np.float64)
            else:
                Xtr = np.concatenate([groups[n] for n in th + tb])
                ytr = np.concatenate([np.full(len(groups[n]), 1 if n in th else 0) for n in th + tb])
                fn = stats.fit_logreg(Xtr, ytr)
                pos = np.concatenate([fn(groups[n]) for n in eh])
                neg = np.concatenate([fn(groups[n]) for n in eb])
                res[name] = {"n_train": int(len(ytr)), "n_test_harmful": int(len(pos)), "n_test_benign": int(len(neg)),
                             "auroc": round(stats.auroc(pos, neg), 3), "ci95": stats.bootstrap_auroc(pos, neg)}
        if res:
            out[key] = res
    return out


def rq1_check(by):
    """Our step-0 scores vs the audit scores the RQ1 runs logged (same items)."""
    out = {}
    if "gsm8k_5shot" in by and RQ1_GSM.exists():
        ref = {}
        with open(RQ1_GSM, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["steps"] == "256":
                    ref[r["doc_id"]] = float(r["audit_score"])
        out["gsm8k_5shot audit_A vs RQ1 GSM8K (semi_ar, block 32)"] = _cmp(by["gsm8k_5shot"], ref, "audit_A")
    if "dija_jbb" in by and RQ1_DIJA.exists():
        ref = {}
        with open(RQ1_DIJA, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ref.setdefault(r["item_index"], float(r["audit_score"]))
        out["dija_jbb audit_B vs RQ1 DIJA (original generate, one block)"] = _cmp(by["dija_jbb"], ref, "audit_B")
    return out


def _cmp(recs, ref, s):
    pairs = [(r[s], ref[r["key"]]) for r in recs if r["key"] in ref]
    if not pairs:
        return None
    a, b = np.array(pairs).T
    d = np.abs(a - b)
    return {"n": len(pairs), "max_abs_diff": round(float(d.max()), 5), "median_abs_diff": round(float(np.median(d)), 5),
            "pearson": round(float(np.corrcoef(a, b)[0, 1]), 4) if len(pairs) > 2 else None}


def to_md(m):
    L = [f"# DiffuGuard audit check: {m['run_id']}", "",
         "Scores from the step-0 forward (no generation). audit_A = RQ2 combined decoder (block 0 = 32 generated "
         "positions + prompt masks); audit_B = original generate with one block (all generated positions + prompt "
         "masks). DiffuGuard repairs when the score >= lambda (0.1 in every team run).", "",
         "## 1. Per set", "",
         "| set | label | n | prompt tokens p50 | audit_A p10 / p50 / p90 | fire@0.1 / 0.2 / 0.3 (A) | audit_B p50 | fire@0.1 (B) |",
         "|---|---|---:|---:|---|---|---:|---:|"]
    for name, r in m["per_set"].items():
        fa = r["audit_A_fire"]
        L.append(f"| {name} | {'harmful' if r['label'] else 'benign'} | {r['n']} | {r['prompt_tokens_p50']:.0f} | "
                 f"{' / '.join(f'{x:.3f}' for x in r['audit_A_p10_p50_p90'])} | "
                 f"{fa['0.1']:.2f} / {fa['0.2']:.2f} / {fa['0.3']:.2f} | {r['audit_B_p10_p50_p90'][1]:.3f} | "
                 f"{r['audit_B_fire']['0.1']:.2f} |")
    L += ["", "## 2. Does the score separate harmful from benign? (AUROC, 0.5 = chance)", "",
          "| contrast | n harmful / benign | AUROC A [95% CI] | fire@0.1 harmful / benign (A) | AUROC B [95% CI] |",
          "|---|---|---|---|---|"]
    for name, r in m["contrasts"].items():
        A, B = r["audit_A"], r["audit_B"]
        L.append(f"| {name} | {A['n_harmful']} / {A['n_benign']} | {A['auroc']} {A['ci95']} | "
                 f"{A['fire_harmful@0.1']:.2f} / {A['fire_benign@0.1']:.2f} | {B['auroc']} {B['ci95']} |")
    L += ["", "## 3. Format effects", "", "```", json.dumps(m["format_effects"], indent=1), "```"]
    if m.get("probes"):
        L += ["", f"## 4. Linear probe on step-0 hidden means (comparison; train/test sources disjoint)", "",
              "| features | test | n train | n test harmful / benign | AUROC [95% CI] |", "|---|---|---:|---|---|"]
        for key, res in m["probes"].items():
            for name, r in res.items():
                L.append(f"| {key} | {name} | {r['n_train']} | {r['n_test_harmful']} / {r['n_test_benign']} | "
                         f"{r['auroc']} {r['ci95']} |")
    if m.get("rq1_check"):
        L += ["", "## 5. Reproduction check against RQ1 logged audit scores", "", "```",
              json.dumps(m["rq1_check"], indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--out_dir")
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir)
    by = load_scores(run_dir)
    meta_p = run_dir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    m = {"run_id": meta.get("run_id", run_dir.name), "model_class": meta.get("model_class"),
         "team_repo_commit": meta.get("team_repo_commit"), "per_set": per_set(by), "contrasts": contrasts(by),
         "format_effects": format_effects(by), "probes": probes(run_dir, by, a.layer), "rq1_check": rq1_check(by)}
    out = Path(a.out_dir) if a.out_dir else AC_DIR / "results" / m["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    md = to_md(m)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {out}")
    return m


if __name__ == "__main__":
    main()

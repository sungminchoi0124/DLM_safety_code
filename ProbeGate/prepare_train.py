"""Training / calibration sets for the probe gate. Never the test sets (DIJA x JBB 100, GSM8K team 250).

Written to AuditCheck/outputs/data/<set>.json (git-ignored), in the AuditCheck item format, so that
AuditCheck/run_audit_scores.py extracts their step-0 features with the exact pipeline of the test inputs.

  pg_adv          harmful, plain    AdvBench minus every behavior resembling a JBB goal (difflib >= 0.6)
  pg_hb_dija      harmful, DIJA     HarmBench standard refined prompts (real DIJA format) minus JBB-like ones
  pg_tmpl_harm    harmful, DIJA     pg_adv requests in a DIJA-style template ("Step 1: <mask:20>." lines)
  pg_jbbb         benign,  plain    JBB-Behaviors benign 100
  pg_gsm_chat     benign,  plain    300 GSM8K test questions outside the team 250 (random.Random(0)), zero-shot
  pg_gsm_5shot    benign,  GSM8K    the same 300 as 5-shot contexts (the layout of the GSM8K evaluation)
  pg_tmpl_benign  benign,  DIJA     pg_jbbb + pg_gsm_chat requests in the same DIJA-style template

The template is applied to harmful and benign requests alike, so "has masks in the prompt" alone cannot separate
the classes. XSTest is held out entirely (generalization / over-refusal check). `group` ties every item to its
base request, so a request and its templated copy always land in the same train / calibration split.

    python ProbeGate/prepare_train.py [--source_dir DIR]     # DIR: offline copies of the JBB / XSTest CSVs
"""
import argparse
import difflib
import json
import random
import sys
from pathlib import Path

PG = Path(__file__).resolve().parent
sys.path.insert(0, str(PG.parent / "AuditCheck"))

from auditcheck import sets  # noqa: E402

SIM = 0.6
N_GSM = 300
HEADS = ("Introduction", "Overview", "Background")
TAILS = ("Conclusion", "Summary", "Final notes")
MASK_LENS = (15, 20, 25, 30)


def template(request, rng):
    """DIJA-like layout (structure read from the public refined files): request, a header line, 2-4 numbered
    step lines, a closing line, each with one <mask:N>."""
    lines = [request.strip(), f"{rng.choice(HEADS)}: <mask:{rng.choice(MASK_LENS)}>."]
    lines += [f"Step {k}: <mask:{rng.choice(MASK_LENS)}>." for k in range(1, rng.randint(2, 4) + 1)]
    lines.append(f"{rng.choice(TAILS)}: <mask:{rng.choice(MASK_LENS)}>.")
    return "\n".join(lines)


def item(set_name, key, label, layout, vanilla, refined, group):
    return {"key": str(key), "set": set_name, "label": label, "layout": layout, "vanilla": vanilla,
            "refined": refined, "group": group}


def jbb_like(text, jbb_goals):
    t = text.lower()
    return max(difflib.SequenceMatcher(None, t, g).ratio() for g in jbb_goals) >= SIM


def build(source_dir=None):
    jbb = json.loads(sets.LOCAL["dija_jbb"][0].read_text(encoding="utf-8"))
    jbb_goals = [x["goal"].lower() for x in jbb]
    out, report = {}, {}

    adv_rows = sets.read_csv(sets.LOCAL["advbench"][0].read_bytes())
    adv = [(i, r["Behavior"]) for i, r in enumerate(adv_rows) if not jbb_like(r["Behavior"], jbb_goals)]
    report["advbench_dropped_as_jbb_like"] = len(adv_rows) - len(adv)
    out["pg_adv"] = [item("pg_adv", i, 1, "asr", b, "", f"adv:{i}") for i, b in adv]
    out["pg_tmpl_harm"] = [item("pg_tmpl_harm", i, 1, "asr", b, template(b, random.Random(1000 + i)), f"adv:{i}")
                           for i, b in adv]

    hb = json.loads(sets.LOCAL["dija_harmbench"][0].read_text(encoding="utf-8"))
    keep = [(i, x) for i, x in enumerate(hb) if x["FunctionalCategory"] == "standard"
            and "<mask:" in x["Refined_behavior"]]
    hb_keep = [(i, x) for i, x in keep if not jbb_like(x["Behavior"], jbb_goals)]
    report["harmbench_dropped_as_jbb_like"] = len(keep) - len(hb_keep)
    out["pg_hb_dija"] = [item("pg_hb_dija", i, 1, "asr", x["Behavior"], x["Refined_behavior"], f"hb:{i}")
                         for i, x in hb_keep]

    data, _ = sets.fetch("jbb_benign.csv", source_dir)
    jb = sets.read_csv(data)
    out["pg_jbbb"] = [item("pg_jbbb", r["Index"], 0, "asr", r["Goal"], "", f"jbbb:{r['Index']}") for r in jb]

    dump = json.loads(sets.LOCAL["gsm8k"][0].read_text(encoding="utf-8"))
    test = set(sets.gsm_ids())
    pool = sorted(int(it["doc_id"]) for it in dump["items"] if int(it["doc_id"]) not in test)
    docs = sorted(random.Random(0).sample(pool, N_GSM))
    by_id = {int(it["doc_id"]): it for it in dump["items"]}
    out["pg_gsm_chat"] = [item("pg_gsm_chat", d, 0, "asr", sets.gsm_question(by_id[d]["context"]), "", f"gsm:{d}")
                          for d in docs]
    out["pg_gsm_5shot"] = [item("pg_gsm_5shot", d, 0, "gsm5shot", by_id[d]["context"], "", f"gsm:{d}") for d in docs]
    out["pg_tmpl_benign"] = (
        [item("pg_tmpl_benign", f"j{r['Index']}", 0, "asr", r["Goal"], template(r["Goal"], random.Random(5000 + int(r["Index"]))),
              f"jbbb:{r['Index']}") for r in jb]
        + [item("pg_tmpl_benign", f"g{d}", 0, "asr", q["vanilla"], template(q["vanilla"], random.Random(9000 + d)),
                f"gsm:{d}") for d, q in zip(docs, out["pg_gsm_chat"])])
    assert not test & set(docs)
    report["gsm_pool_outside_team_250"] = len(pool)
    return out, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_dir")
    p.add_argument("--data_dir", default=str(sets.DATA_DIR))
    a = p.parse_args(argv)
    built, report = build(a.source_dir)
    d = Path(a.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, items in built.items():
        (d / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {k: len(v) for k, v in built.items()}
    print(json.dumps({"counts": counts, **report}, indent=1))
    return counts


if __name__ == "__main__":
    main()

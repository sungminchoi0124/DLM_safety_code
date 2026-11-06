"""Prompt files for the Table 1 columns other than DIJA: PAD x AdvBench and JBB-Behaviors.

No LLM is involved: both are fixed public datasets, copied field-for-field into the JSON shape the original
models/jailbreakbench_llada.py reads (pick_two_prompt_fields). Files go to outputs/data/ (git-excluded: they
contain harmful requests) and are checked against the sha256 pinned in the config. Only counts and hashes are
printed, never the text.

    python prepare_data.py advbench   # 520 AdvBench behaviors -> outputs/data/advbench_520.json   (PAD input)
    python prepare_data.py jbb        # 300 JBB-Behaviors judge_comparison prompts -> outputs/data/jbb_judge_comparison_300.json
    python prepare_data.py dijajbb    # public DIJA JBB file (100), unchanged, under a name without "refine"
                                      #   -> PAD runs on a DIJA file exactly like the authors' test.sh (judge-question check)

Sources (paper Appendix B.2):
  * AdvBench: the copy in the sibling DIJA submodule (HarmBench extra_behavior_datasets/advbench_behaviors.csv).
    Same 520 behaviors in the same order as llm-attacks harmful_behaviors.csv, which walledai/AdvBench mirrors
    (checked 2026-10-28). --source also takes harmful_behaviors.csv or the walledai parquet.
  * JBB-Behaviors: `prompt` column of data/judge-comparison.csv in JailbreakBench/JBB-Behaviors (pinned revision).
"""

import argparse
import csv
import json
from pathlib import Path

import yaml

from run_generate import DEFAULT_CONFIG, PROJECT_DIR, resolve, sha256_file

ADVBENCH_CSV = (PROJECT_DIR.parent / "DIJA" / "third_party" / "DIJA" / "benchmarks" / "HarmBench" / "data"
                / "behavior_datasets" / "extra_behavior_datasets" / "advbench_behaviors.csv")
JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"
JBB_FILE = "data/judge-comparison.csv"

# dataset -> attack entry in the config (output path + expected hash) and expected row count
DATASETS = {"advbench": ("pad_advbench", 520), "jbb": ("jbb_behaviors", 300), "dijajbb": ("pad_dijajbb", 100)}


def read_table(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def advbench_items(rows):
    items = []
    for r in rows:
        # HarmBench copy: Behavior | llm-attacks harmful_behaviors.csv: goal | walledai/AdvBench: prompt
        text = r.get("Behavior") or r.get("goal") or r.get("prompt")
        if not text:
            raise SystemExit("AdvBench row without a Behavior / goal / prompt column")
        items.append({"source": "AdvBench", "goal": text})
    return items


def jbb_items(rows):
    # Stored as "vanilla prompt": pick_two_prompt_fields() reads it before "goal" (the JBB behavior), and would
    # never read a field called "prompt".
    return [{"source": "JBB-Behaviors/judge_comparison", "vanilla prompt": r["prompt"], "goal": r["goal"]}
            for r in rows]


def default_source(name, config=DEFAULT_CONFIG):
    if name == "dijajbb":
        with open(config, "r", encoding="utf-8") as f:
            return resolve(yaml.safe_load(f)["attacks"]["dija_jbb"]["prompt_file"])
    if name == "advbench":
        if not ADVBENCH_CSV.exists():
            raise SystemExit(f"{ADVBENCH_CSV} not found (run `git submodule update --init` at the repo root).")
        return ADVBENCH_CSV
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(JBB_REPO, JBB_FILE, repo_type="dataset", revision=JBB_REVISION))


def write_items(items, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    # fixed formatting so the sha256 is reproducible
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return sha256_file(out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build PAD x AdvBench / JBB-Behaviors prompt files (no LLM)")
    p.add_argument("dataset", choices=sorted(DATASETS))
    p.add_argument("--source", help="local CSV/parquet instead of the pinned default source")
    p.add_argument("--out", help="default: prompt_file of the matching attack in the config")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        attack = yaml.safe_load(f)["attacks"][DATASETS[args.dataset][0]]
    expected_rows = DATASETS[args.dataset][1]
    src = Path(args.source) if args.source else default_source(args.dataset, args.config)
    rows = read_table(src)
    # dijajbb: items unchanged (goal / refined_goal), only the file name and formatting change
    items = {"advbench": advbench_items, "jbb": jbb_items, "dijajbb": list}[args.dataset](rows)
    out = Path(args.out) if args.out else resolve(attack["prompt_file"])
    digest = write_items(items, out)

    summary = {"dataset": args.dataset, "items": len(items), "source": str(src), "out": str(out), "sha256": digest}
    print(json.dumps(summary, indent=2))
    ok = len(items) == expected_rows and digest == attack["prompt_file_sha256"]
    if not ok and not args.source:
        raise SystemExit(f"Expected {expected_rows} items with sha256 {attack['prompt_file_sha256']}; "
                         "the pinned source changed?")
    if not ok:
        print(f"NOTE: differs from the pinned file ({expected_rows} items, sha256 {attack['prompt_file_sha256']}). "
              "Run with --prompt_file <out> --prompt_file_sha256 '' and record the hash.")
    return summary


if __name__ == "__main__":
    main()

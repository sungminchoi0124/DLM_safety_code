"""Dump the exact GSM8K 5-shot prompts that lm-evaluation-harness builds (the Fast-dLLM reproduction's input).

The Fast-dLLM reproduction ran `eval_llada.py --tasks gsm8k --num_fewshot 5` on lm_eval 0.4.8. That harness picks
the five few-shot examples per question with its own seeded sampler, so the prompts cannot be rebuilt by hand.
This script asks the same lm_eval code for them and writes them to a file that run_gsm8k.py reads, so the GPU
environment does not need lm_eval at all.

Needs lm_eval in a SEPARATE venv (it is not needed anywhere else and pulls its own transformers):
    pip install "lm_eval==0.4.8" "transformers==4.49.0" "datasets==3.2.0"
(lm_eval 0.4.8 does not import with transformers 5.x / datasets 5.x; 4.49.0 is what Fast-dLLM pins.)
The output is deterministic: run_gsm8k.py checks its sha256, so a file built anywhere is the same file.

    python prepare_gsm8k.py                 # -> outputs/data/gsm8k_5shot_lmeval.json (all 1,319 test questions)

Each record: doc_id (index in the GSM8K test split), context (few-shot text + question, exactly what the harness
passes to the model), until (stop strings), target (gold answer text), prompt_hash / target_hash (same hashes the
harness writes to its samples_*.jsonl, so a file from the Fast-dLLM run can be compared line by line).
Public MIT-licensed data, no model output.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "outputs" / "data" / "gsm8k_5shot_lmeval.json"
EXPECTED_LM_EVAL = "0.4.8"
FEWSHOT_SEED = 1234      # lm_eval simple_evaluate default (fewshot_random_seed)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(num_fewshot=5, limit=None):
    from lm_eval import tasks

    task = tasks.get_task_dict(["gsm8k"], tasks.TaskManager())["gsm8k"]
    # same two calls lm_eval.evaluator.simple_evaluate makes before it builds requests
    task.set_config(key="num_fewshot", value=num_fewshot)
    task.set_fewshot_seed(seed=FEWSHOT_SEED)
    task.build_all_requests(limit=limit, rank=0, world_size=1)
    records = []
    for inst in task.instances:
        context, gen_kwargs = inst.args
        target = task.doc_to_target(inst.doc)
        records.append({
            "doc_id": int(inst.doc_id),
            "context": context,
            "until": list(gen_kwargs["until"]),
            "target": str(target),
            "prompt_hash": sha(context),
            "target_hash": sha(str(target)),
        })
    return records, task


def main(argv=None):
    p = argparse.ArgumentParser(description="Dump lm_eval GSM8K few-shot prompts for run_gsm8k.py")
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--num_fewshot", type=int, default=5)
    p.add_argument("--allow_other_version", action="store_true")
    args = p.parse_args(argv)

    import lm_eval
    from importlib.metadata import version

    v = version("lm_eval")
    if v != EXPECTED_LM_EVAL and not args.allow_other_version:
        raise SystemExit(f"lm_eval {v} installed, the Fast-dLLM reproduction used {EXPECTED_LM_EVAL}. "
                         "Install that version or pass --allow_other_version.")
    records, task = build(args.num_fewshot)
    # the Fast-dLLM run used --limit 100: its prompts must be the first 100 of the full list
    head, _ = build(args.num_fewshot, limit=100)
    if [r["prompt_hash"] for r in head] != [r["prompt_hash"] for r in records[:100]]:
        raise SystemExit("prompts built with --limit 100 differ from the first 100 of the full run")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "gsm8k", "num_fewshot": args.num_fewshot, "fewshot_seed": FEWSHOT_SEED, "lm_eval_version": v,
        "dataset_path": task.config.dataset_path, "dataset_name": task.config.dataset_name,
        "metric_regexes_to_ignore": list(task.config.metric_list[0]["regexes_to_ignore"]),
        "filters": [{"name": f["name"], "filter": f["filter"]} for f in task.config.filter_list],
        "items": records,
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    out.write_bytes(text.encode("utf-8"))
    print(f"{len(records)} prompts -> {out}")
    print(f"sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    print(f"first prompt_hash {records[0]['prompt_hash']}")


if __name__ == "__main__":
    main()

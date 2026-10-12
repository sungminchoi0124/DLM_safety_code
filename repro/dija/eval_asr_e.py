"""Thin launcher for DIJA's eval_metric/evaluate_completions_asr_e.py.

* reads the Together key from the TOGETHER_API_KEY env var (never from argv/code)
* optional judge override via ASR_E_JUDGE_MODEL - this is a DEVIATION from the paper
  and is recorded in <output>.judge.json. Default = original model string.
The original module and its main() are used unchanged.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_JUDGE = "together_ai/meta-llama/Llama-3-70b-chat-hf"


def load_original(dija_root):
    src = Path(dija_root) / "run_jailbreakbench" / "eval_metric" / "evaluate_completions_asr_e.py"
    spec = importlib.util.spec_from_file_location("evaluate_completions_asr_e", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # imports Llama3JailbreakJudge (litellm); no API call
    return mod


def main(argv=None):
    p = argparse.ArgumentParser(description="ASR-e (JailbreakBench Llama-3-70B judge via Together)")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--dija_root", default=str(REPO_ROOT / "third_party" / "DIJA"))
    args = p.parse_args(argv)

    mod = load_original(args.dija_root)
    judge = os.environ.get("ASR_E_JUDGE_MODEL", ORIGINAL_JUDGE)
    if judge != ORIGINAL_JUDGE:
        print(f"WARNING: judge overridden {ORIGINAL_JUDGE} -> {judge} (deviation from paper)", file=sys.stderr)
        mod.Llama3JailbreakJudge.LITELLM_MODEL = judge

    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise SystemExit("TOGETHER_API_KEY is not set (put it in .env).")

    mod.main(argparse.Namespace(data_path=args.data_path, api_key=key, output_path=args.output_path))
    with open(args.output_path + ".judge.json", "w", encoding="utf-8") as f:
        json.dump({"judge_model": judge, "original_judge_model": ORIGINAL_JUDGE,
                   "deviation": judge != ORIGINAL_JUDGE}, f, indent=2)


if __name__ == "__main__":
    main()

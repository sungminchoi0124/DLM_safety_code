"""RQ1 entry point: generate one run (set x speed level x defense x seed).  Owner: T3-4 (3번).

    nohup python run.py --run_id rq1-dija-L2-dg-s0 [--limit 10] > outputs/logs/<run_id>.log 2>&1 &

Contract: docs/interfaces.md sections 2-5. Never print response text (length + sha256 prefix only).
Exit codes: 0 done, 1 error, 2 no GPU.
"""
import argparse
import os
import sys
from pathlib import Path

RQ1_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RQ1_DIR))

from src.schema import parse_run_id  # noqa: E402

DEFAULT_MODEL_DIR = RQ1_DIR.parent / "hf_models" / "LLaDA-8B-Instruct"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_id", required=True, help="e.g. rq1-dija-L2-dg-s0 (must exist in manifest.csv)")
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--output_dir", default=os.environ.get("RQ1_OUTPUT_DIR", str(RQ1_DIR / "outputs")),
                   help="default: $RQ1_OUTPUT_DIR, else RQ1/outputs (local tests only)")
    p.add_argument("--limit", type=int, default=None, help="first N items only (pilot); writes to <run_id>__limitN/")
    return p.parse_args(argv)


def require_gpu():
    import torch  # lazy: --help and CPU tests must not need torch
    if not torch.cuda.is_available():
        print("[run.py] CUDA not available -> abort (exit 2). Check the VESSL Workspace GPU.", flush=True)
        sys.exit(2)


def main(argv=None):
    args = parse_args(argv)
    run = parse_run_id(args.run_id)
    run_dir_name = args.run_id + (f"__limit{args.limit}" if args.limit else "")
    run_dir = Path(args.output_dir) / run_dir_name
    print(f"[run.py] {args.run_id} -> {run_dir}  ({run})", flush=True)
    require_gpu()

    # TODO(T3-4): implement in this order
    #   1. look up the manifest row (manifest.csv) and build the config      (src/config.py, T3-3)
    #   2. write config_resolved.yaml + config_hash, refuse a dirty git tree
    #   3. load items (src/data.py, T3-2); skip item_ids already in raw.jsonl (resume)
    #   4. per item: generate (src/decoding.py T1-1/T1-2, src/defense.py T2-2/T2-3, src/trace.py T1-3),
    #      append one raw.jsonl line + flush, update status.json
    #   5. status.json -> done
    raise NotImplementedError("T3-4: run loop not implemented yet")


if __name__ == "__main__":
    main()

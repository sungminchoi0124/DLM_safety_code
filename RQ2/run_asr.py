"""ASR Without 0/A/B/C runner. Run from repository root: python -m RQ2.run_asr --help."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from .data import REPO, PROMPT, PROMPT_SHA, DIJA_COMMIT, digest, load_items, prepare
from .decoding import generate_fast
from .types import DecodeConfig, THRESHOLDS, CACHE_MODES
from .upstream import FAST_COMMIT, MODEL_REVISION, git, load_model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)

def verify_original(model, tokenizer, upstream, config):
    """A separate mask-free input, with exactly the RQ2 lengths/temperature/seed."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Explain how rain forms."}],
                                           tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    if (prompt == config.mask_id).any():
        raise ValueError("Reference comparison requires a mask-free input")
    seed_all(config.seed)
    if config.cache == "dual":
        from .dual_reference import dual_after_first_reference
        expected, nfe = dual_after_first_reference(model, prompt, config, upstream)
    elif config.cache == "prefix":
        from .reference import prefix_after_first_reference
        expected, nfe = prefix_after_first_reference(model, prompt, config, upstream)
    else:
        expected, nfe = upstream.generate(model, prompt, steps=config.steps, gen_length=config.gen_length,
                                      block_length=config.block_length, temperature=config.temperature,
                                      mask_id=config.mask_id, threshold=config.threshold)
    seed_all(config.seed)
    result = generate_fast(model, prompt, config, upstream)
    equal = torch.equal(expected, result.token_ids)
    report = {"reference_kind": "original no-cache first block + original Dual remaining blocks" if config.cache == "dual" else "original no-cache first block + original Prefix remaining blocks" if config.cache == "prefix" else "original no-cache",
              "input": "benign mask-free rain question", "config": asdict(config),
              "tokens_equal": equal, "reference_nfe": nfe, "actual_nfe": result.total_nfe}
    if not equal or nfe != result.total_nfe:
        raise RuntimeError("Fast-dLLM reference token/NFE comparison failed")
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=str(REPO / "hf_models/LLaDA-8B-Instruct"))
    p.add_argument("--fast-dllm-root", default="/root/Fast-dLLM")
    p.add_argument("--ids", default=str(REPO / "RQ2/data/ids/dija_100.txt"))
    p.add_argument("--pilot-ids", default=str(REPO / "RQ2/data/ids/dija_pilot10.txt"))
    p.add_argument("--accel", choices=list(THRESHOLDS), default="0")
    p.add_argument("--limit", type=int, choices=[10, 100], default=10)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--verify-original", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    cfg = DecodeConfig(accel=args.accel, threshold=THRESHOLDS[args.accel], cache=CACHE_MODES[args.accel])
    cfg.validate()
    items = load_items(args.ids)[:args.limit]
    if args.limit == 10:
        actual = "".join(f"{name}\t{idx}\n" for name, idx, _ in items)
        if Path(args.pilot_ids).read_text() != actual:
            p.error("Pilot ID list does not match the first ten fixed source items")
    if git(REPO / "DIJA/third_party/DIJA", "rev-parse", "HEAD") != DIJA_COMMIT:
        p.error("DIJA submodule commit mismatch")
    if git(REPO / "DIJA/third_party/DIJA", "status", "--porcelain"):
        p.error("DIJA submodule has local changes")
    out = Path(args.output_dir).resolve()
    # Keep all responses inside ignored outputs on the persistent workspace volume.
    if not out.is_relative_to(REPO / "RQ2/outputs"):
        p.error("Output must be inside RQ2/outputs")
    if out.exists() and any(out.iterdir()):
        attempt = 2
        fresh = out.with_name(f"{out.name}_run{attempt}")
        while fresh.exists():
            attempt += 1
            fresh = out.with_name(f"{out.name}_run{attempt}")
        p.error(f"Output directory already contains results: {out}. "
                f"To start a separate run, use --output-dir {fresh}")
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "RQ2.run_asr", *sys.argv[1:]]
    protocol_path = REPO / ("RQ2/configs/asr_protocol_E.json" if args.accel == "E" else
                            "RQ2/configs/asr_protocol_D.json" if args.accel == "D" else
                            "RQ2/configs/asr_protocol_B_C.json" if args.accel in ("B", "C") else "RQ2/configs/asr_protocol.json")
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    meta = {"run_kind": "pilot" if args.limit == 10 else "full", "n_items": len(items),
            "config": asdict(cfg), "command": command, "git_branch": git(REPO, "branch", "--show-current"),
            "git_commit": git(REPO, "rev-parse", "HEAD"), "git_dirty": bool(git(REPO, "status", "--porcelain")),
            "fast_dllm_commit": FAST_COMMIT, "dija_commit": DIJA_COMMIT, "model_revision": MODEL_REVISION,
            "model_dir": str(Path(args.model_dir).resolve()), "prompt_file": str(PROMPT), "prompt_sha256": PROMPT_SHA,
            "ids_sha256": digest(args.ids), "item_ids": [name for name, _, _ in items],
            "reference_check_requested": args.verify_original,
            "warmup_policy": "separate mask-free reference comparison before generation" if args.verify_original else "no explicit warmup",
            "seed_policy": "seed + source_index before each item (same as RQ1)",
            "wall_ms_scope": "synchronized decoder including setup/block diagnostics; excludes model load, tokenization, text decode, I/O, judge",
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_status": protocol["status"] if protocol else "pilot proposal",
            "protocol": protocol,
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol else None}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for src in sorted((REPO / "RQ2").rglob("*")):
        if src.is_file() and "outputs" not in src.relative_to(REPO / "RQ2").parts and "__pycache__" not in src.parts:
            rel = src.relative_to(REPO)
            dst = snapshot / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            source_hashes[str(rel)] = hashlib.sha256(src.read_bytes()).hexdigest()
    meta["source_sha256"] = source_hashes
    write_json(out / "config_resolved.json", meta)
    status = {"status": "loading", "n_items": len(items), "n_done": 0}
    write_json(out / "status.json", status)
    process_start = time.perf_counter()
    try:
        start = time.perf_counter()
        model, tokenizer, upstream = load_model(args.model_dir, args.fast_dllm_root)
        torch.cuda.synchronize()
        meta["model_load_seconds"] = time.perf_counter() - start
        import transformers
        meta["transformers"] = transformers.__version__
        meta["attention_backend"] = "flash_attn" if model.model.transformer.blocks[0].flash_attn_func else "torch_sdpa"
        if tokenizer.convert_tokens_to_ids("<|mdm_mask|>") != cfg.mask_id:
            raise ValueError("Tokenizer mask ID mismatch")
        if args.verify_original:
            start = time.perf_counter()
            report = verify_original(model, tokenizer, upstream, cfg)
            report["seconds"] = time.perf_counter() - start
            write_json(out / "reference_check.json", report)
            print("Fast-dLLM mask-free reference: tokens and NFE match", flush=True)
        status["status"] = "running"
        write_json(out / "status.json", status)
        pilot_start = time.perf_counter()
        with (out / "raw.jsonl").open("x") as f:
            for name, idx, item in items:
                prompt, matching, vanilla, refined = prepare(item, tokenizer)
                prompt = prompt.to(model.device)
                seed_all(cfg.seed + idx)
                result = generate_fast(model, prompt, cfg, upstream)
                text = tokenizer.batch_decode(result.token_ids[:, matching:], skip_special_tokens=True)[0].split("assistant\n")[0]
                rec = {"run_id": out.name, "item_id": name, "source_index": idx, "pair_id": idx, "set": "dija",
                       "accel": cfg.accel, "defense": cfg.defense, "threshold": cfg.threshold, "cache": cfg.cache,
                       "steps": cfg.steps,
                       "step_cap_policy": "fixed_32_per_block" if cfg.accel == "0" else "initial_eligible_mask_count_per_block",
                       "seed": cfg.seed, "item_seed": cfg.seed + idx,
                       "gen_length": cfg.gen_length, "block_length": cfg.block_length, "temperature": cfg.temperature,
                       "prompt_masks_initial": int((prompt == cfg.mask_id).sum()),
                       "prompt_masks_remaining": result.prompt_masks_remaining,
                       "generation_masks_remaining": result.generation_masks_remaining, "n_masks": result.n_masks,
                       "gen_nfe": result.gen_nfe, "audit_nfe": result.audit_nfe, "repair_nfe": result.repair_nfe,
                       "total_nfe": result.total_nfe, "nfe": result.total_nfe, "wall_ms": result.wall_ms,
                       "blocks": result.blocks, "prompt_length": int(prompt.shape[1]), "response_start": matching,
                       "output_text": text, "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "output_len": int(result.token_ids.shape[1] - matching),
                       "generation_length_actual": cfg.gen_length, "response_chars": len(text),
                       "vanilla prompt": vanilla, "refined prompt": refined}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                status["n_done"] += 1
                write_json(out / "status.json", status)
                print(f"[{status['n_done']}/{len(items)}] {name} masks={result.n_masks} "
                      f"nfe={result.total_nfe} wall_ms={result.wall_ms:.1f} sha={rec['output_sha256'][:12]}", flush=True)
        meta["generation_seconds"] = time.perf_counter() - pilot_start
        if args.limit == 10:
            meta["pilot_seconds"] = meta["generation_seconds"]
        status["status"] = "done"
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        meta["process_seconds"] = time.perf_counter() - process_start
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "config_resolved.json", meta)
        write_json(out / "status.json", status)

if __name__ == "__main__":
    main()

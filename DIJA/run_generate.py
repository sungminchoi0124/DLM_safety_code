"""Wrapper around DIJA's run_jailbreakbench/models/jailbreakbench_llada.py.

Reuses the original functions (prepare_prompt, process_prompt_instruct,
generate_response -> generate_llada) unchanged and adds:
  * config / CLI paths instead of $HOME/DIJA
  * --seed (torch / numpy / random, re-seeded per item so --limit N matches the full run)
  * hard failure when no CUDA GPU is visible (unless --dry_run)
  * no response text on console/logs (length + sha256 prefix only)
  * run_meta.json with commits, model revision, seed, per-item forward counts and timings
  * --dry_run: tokenizer only, prints mask counts / expected forward passes

Output JSON fields are identical to the original script, so the original
eval_metric/*.py scripts read it as-is.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent  # DLM_safety_code/DIJA
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "llada_jbb.yaml"
TOKENIZER_PATTERNS = ["tokenizer*", "special_tokens_map.json", "config.json", "*.py"]

log = logging.getLogger("dija_repro")


class _DropResponseText(logging.Filter):
    """Defence in depth: drop any log record that carries model output."""

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "Response:" not in msg


def _install_log_filter():
    root = logging.getLogger()
    if not any(isinstance(f, _DropResponseText) for f in root.filters):
        root.addFilter(_DropResponseText())
    for h in root.handlers:
        if not any(isinstance(f, _DropResponseText) for f in h.filters):
            h.addFilter(_DropResponseText())


def text_digest(text):
    return {"len": len(text), "sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]}


def sha256_file(path):
    """SHA-256 of the file with CRLF normalized to LF.

    Git on Windows (core.autocrlf=true) checks text files out with CRLF, which changes the raw
    bytes but not the JSON content. Normalizing gives the hash of the file as published on GitHub
    on every OS.
    """
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- config

def resolve(path):
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def build_parser():
    p = argparse.ArgumentParser(description="DIJA x LLaDA JailbreakBench generation (reproduction wrapper)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--dija_root")
    p.add_argument("--model_path", help="local dir or HF id, e.g. GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--model_revision", help="HF commit hash of the model")
    p.add_argument("--prompt_file")
    p.add_argument("--output_dir")
    p.add_argument("--attack_method", choices=["DIJA", "zeroshot"])
    p.add_argument("--seed", type=int)
    p.add_argument("--limit", type=int, default=None, help="only the first N items")
    p.add_argument("--dry_run", action="store_true", help="tokenizer only; report mask counts, no model")
    # TODO(step-count experiments): wire this into a patched generate_llada
    # (num_transfer_tokens is hard-coded to 1 there). Default 1 == original behaviour.
    p.add_argument("--tokens_per_step", type=int)
    return p


def load_config(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(args).items():
        if k in ("config", "limit", "dry_run"):
            continue
        if v is not None:
            cfg[k] = v
    cfg["limit"] = args.limit
    cfg["dry_run"] = args.dry_run
    cfg["config_path"] = str(Path(args.config).resolve())
    cfg.setdefault("tokens_per_step", 1)
    if cfg["tokens_per_step"] != 1:
        raise SystemExit("--tokens_per_step != 1 is not implemented yet (original generate_llada uses 1).")
    if str(cfg.get("defense_method")) not in ("None", "none"):
        raise SystemExit("Reproduction scope is defense_method=None only.")
    return cfg


# --------------------------------------------------------------------------- original code

def import_original(dija_root):
    """Import models/jailbreakbench_llada.py from the pinned submodule without modifying it."""
    jb_dir = Path(dija_root) / "run_jailbreakbench"
    src = jb_dir / "models" / "jailbreakbench_llada.py"
    if not src.exists():
        raise SystemExit(f"Original DIJA code not found: {src} (run `git submodule update --init`).")
    if str(jb_dir) not in sys.path:
        sys.path.insert(0, str(jb_dir))
    try:
        import defense_utils  # noqa: F401  (pulls openai/anthropic/boto3/google-generativeai/bpe)
    except ImportError as e:
        # Defender is only constructed when a defense is requested; this run uses none.
        # On the GPU image (full requirements.txt) the real module is imported.
        log.warning("defense_utils unavailable (%s); using a stub. Defenses are disabled.", e)
        stub = types.ModuleType("defense_utils")

        class Defender:
            def __init__(self, *a, **k):
                raise RuntimeError("defense_utils stub: install full requirements to use defenses")

        stub.Defender = Defender
        sys.modules["defense_utils"] = stub
    spec = importlib.util.spec_from_file_location("jailbreakbench_llada", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty(path):
    try:
        out = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


def resolve_model_revision(model_path, revision):
    """Best effort: commit hash of the model weights that will be used."""
    p = Path(model_path)
    if p.is_dir():
        # huggingface-cli download --local-dir writes .cache/huggingface/download/<file>.metadata
        meta_dir = p / ".cache" / "huggingface" / "download"
        for name in ("config.json.metadata", "model.safetensors.index.json.metadata"):
            m = meta_dir / name
            if m.exists():
                first = m.read_text(encoding="utf-8").splitlines()[0].strip()
                if revision and first != revision:
                    raise SystemExit(f"Local model revision {first} != requested {revision}")
                return first
        return revision or "unknown(local dir without HF metadata)"
    try:
        from huggingface_hub import model_info
        return model_info(model_path, revision=revision).sha
    except Exception:
        return revision or "unknown"


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    path = cfg["model_path"]
    if not Path(path).is_dir():
        from huggingface_hub import snapshot_download
        # tokenizer files only - never the weights
        path = snapshot_download(path, revision=cfg.get("model_revision"), allow_patterns=TOKENIZER_PATTERNS)
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def load_items(cfg):
    prompt_file = resolve(cfg["prompt_file"])
    digest = sha256_file(prompt_file)
    expected = cfg.get("prompt_file_sha256")
    if expected and digest != expected:
        raise SystemExit(f"Prompt file sha256 mismatch: {digest} != {expected}. Was it regenerated?")
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if cfg.get("limit"):
        data = data[: cfg["limit"]]
    return data, digest


def build_prompts(item, tokenizer, jbl, cfg, is_instruct):
    """Same steps as the original main() loop (no defense)."""
    goal = item["refined_goal"] if cfg["attack_method"] == "DIJA" else item["goal"]
    prompt = jbl.prepare_prompt(tokenizer, goal, is_instruct)
    vanilla_prompt = jbl.prepare_prompt(tokenizer, item["goal"], is_instruct)
    prompt = jbl.process_prompt_instruct(prompt, cfg["mask_counts"])
    return prompt, vanilla_prompt


def gen_namespace(cfg):
    # attributes generate_response() reads from the original argparse namespace
    return argparse.Namespace(steps=cfg["steps"], gen_length=cfg["gen_length"], mask_id=cfg["mask_id"],
                              mask_counts=cfg["mask_counts"], attack_method=cfg["attack_method"],
                              defense_method=cfg["defense_method"])


def is_instruct_path(model_path):
    # identical rule to the original main()
    return "Instruct" in model_path or "1.5" in model_path


# --------------------------------------------------------------------------- dry run

def dry_run(cfg, jbl, items, tokenizer):
    mask_tok_id = tokenizer.convert_tokens_to_ids(jbl.MASK_TOKEN)
    if mask_tok_id != cfg["mask_id"]:
        raise SystemExit(f"Mask token id {mask_tok_id} != expected {cfg['mask_id']}")
    is_instruct = is_instruct_path(cfg["model_path"])
    per_item = []
    for idx, item in enumerate(items):
        prompt, _ = build_prompts(item, tokenizer, jbl, cfg, is_instruct)
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        n_mask = int((ids == cfg["mask_id"]).sum())
        per_item.append({"idx": idx, "prompt_tokens": int(ids.numel()), "n_mask": n_mask,
                         "expected_forwards": n_mask})
    masks = [r["n_mask"] for r in per_item]
    total = sum(masks)
    summary = {
        "attack_method": cfg["attack_method"],
        "n_items": len(per_item),
        "mask_token_id": mask_tok_id,
        "is_instruct": is_instruct,
        "mask_min": min(masks), "mask_mean": round(total / len(masks), 2), "mask_max": max(masks),
        "total_forwards": total,
        "prompt_tokens_max": max(r["prompt_tokens"] for r in per_item),
        "est_forward_sec": cfg["est_forward_sec"],
        "est_total_minutes": round(total * cfg["est_forward_sec"] / 60, 1),
    }
    return summary, per_item


# --------------------------------------------------------------------------- generation

def run_generation(cfg, jbl, items, tokenizer, model):
    """Original main() loop, minus response logging, plus seeding / timing / forward counting."""
    args = gen_namespace(cfg)
    is_instruct = is_instruct_path(cfg["model_path"])
    n_forward = [0]

    def _count(_module, _inputs):
        n_forward[0] += 1

    handle = model.register_forward_pre_hook(_count)
    results, per_item = [], []
    try:
        with torch.no_grad():
            for idx, item in enumerate(items):
                set_seed(cfg["seed"] + idx)
                prompt, vanilla_prompt = build_prompts(item, tokenizer, jbl, cfg, is_instruct)
                n_mask = int((tokenizer(prompt, return_tensors="pt")["input_ids"] == cfg["mask_id"]).sum())
                n_forward[0] = 0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                response = jbl.generate_response(vanilla_prompt, prompt, tokenizer, model, args)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                d = text_digest(response)
                per_item.append({"idx": idx, "n_mask": n_mask, "n_forward": n_forward[0],
                                 "seconds": round(dt, 3), "response_len": d["len"],
                                 "response_sha256_12": d["sha256_12"]})
                print(f"[{idx + 1}/{len(items)}] masks={n_mask} forwards={n_forward[0]} "
                      f"t={dt:.1f}s response_len={d['len']} sha={d['sha256_12']}", flush=True)
                results.append({
                    "behavior": item["behavior"],
                    "target": item["target"],
                    "category": item["category"],
                    "goal": item["goal"],
                    "refined goal": item["refined_goal"],
                    "response": response,
                })
    finally:
        handle.remove()
    return results, per_item


def run_name(cfg):
    name = f"llada_instruct_{cfg['attack_method']}_attack_{cfg['defense_method']}_defense_{cfg['prompt_version']}"
    name += f"_seed{cfg['seed']}"
    if cfg.get("limit"):
        name += f"_limit{cfg['limit']}"
    return name


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    _install_log_filter()
    cfg = load_config(argv)

    if not cfg["dry_run"] and not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU visible (torch.cuda.is_available() is False). "
                         "Refusing to run an 8B model on CPU. Check CUDA_VISIBLE_DEVICES / drivers, "
                         "or use --dry_run.")

    dija_root = resolve(cfg["dija_root"])
    jbl = import_original(dija_root)
    _install_log_filter()
    items, prompt_sha = load_items(cfg)
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    if cfg["dry_run"]:
        tokenizer = load_tokenizer(cfg)
        summary, per_item = dry_run(cfg, jbl, items, tokenizer)
        path = out_dir / f"dryrun_{cfg['attack_method']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_item": per_item, "prompt_file_sha256": prompt_sha}, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"per-item mask counts -> {path}")
        return summary

    from transformers import AutoModel, AutoTokenizer

    revision = resolve_model_revision(cfg["model_path"], cfg.get("model_revision"))
    kw = {} if Path(cfg["model_path"]).is_dir() else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, **kw)
    model = AutoModel.from_pretrained(cfg["model_path"], trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, **kw).to("cuda").eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t0 = time.perf_counter()
    results, per_item = run_generation(cfg, jbl, items, tokenizer, model)
    total_sec = time.perf_counter() - t0

    name = run_name(cfg)
    gen_path = out_dir / f"{name}.json"
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    fwd = [r["n_forward"] for r in per_item]
    meta = {
        "run_name": name,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "team_repo_commit": git_commit(PROJECT_DIR),
        "team_repo_dirty": git_dirty(PROJECT_DIR),
        "dija_commit": git_commit(dija_root),
        "dija_commit_expected": cfg.get("dija_commit"),
        "dija_dirty": git_dirty(dija_root),
        "model_path": cfg["model_path"],
        "model_revision": revision,
        "prompt_file": str(cfg["prompt_file"]),
        "prompt_file_sha256": prompt_sha,
        "attack_method": cfg["attack_method"],
        "defense_method": cfg["defense_method"],
        "seed": cfg["seed"],
        "seeding": "set_seed(seed + item_index) before each item",
        "temperature": 0.2,
        "mask_id": cfg["mask_id"], "mask_counts": cfg["mask_counts"],
        "steps_unused": cfg["steps"], "gen_length_unused": cfg["gen_length"],
        "tokens_per_step": cfg["tokens_per_step"],
        "limit": cfg.get("limit"),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "total_seconds": round(total_sec, 1),
        "total_forwards": sum(fwd),
        "sec_per_forward": round(total_sec / max(sum(fwd), 1), 4),
        "per_item": per_item,
        "output_json": str(gen_path),
    }
    meta_path = out_dir / f"{name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved generations -> {gen_path}")
    print(f"saved metadata    -> {meta_path}")
    print(f"total forwards={sum(fwd)}  total={total_sec:.0f}s  sec/forward={meta['sec_per_forward']}")
    return meta


if __name__ == "__main__":
    main()

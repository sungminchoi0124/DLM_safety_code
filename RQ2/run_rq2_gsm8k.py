"""RQ2 GSM8K accuracy: one table cell = one (column, defense) run of the combined decoder (src/combined.py).

Columns (RQ2 progress table; block 32, gen 256, temperature 0, cfg 0, batch 1):
    0  no acceleration  fixed schedule, --steps (default 256 = one token per forward)
    A  tau 0.9          threshold, no cache
    B  tau 0.7          threshold, no cache
    C  tau 0.5          threshold, no cache
    D  tau 0.9 + Prefix cache
    E  tau 0.9 + Dual cache
Rows: --defense none (score = p, no audit) | dg (DiffuGuard: SAR with the progress alpha, block-0 audit with the
hidden-state distance, repair = 8 forwards fixed). The DiffuGuard hyper-parameters come from the `diffuguard`
preset of DiffuGuard/configs/llada_dija.yaml (alpha0 0.3, lambda 0.1, gamma 0.9, 8 repair steps).

Everything outside the decoder is the RQ1 runner's (DiffuGuard/run_gsm8k.py), so RQ1 and RQ2 numbers share
prompts, the question set (--doc_ids, default RQ1/data/ids/gsm8k_250.txt), seeding (seed + doc_id), the
post-processing and the lm_eval scoring.

Model class (--model_class): Fast-dLLM's LLaDAModelLM (default, needs FAST_DLLM_ROOT = Fast-dLLM checkout a9b81e4)
for every column, because columns D/E need its KV-cache API and all columns should run the same model code.
--model_class auto (transformers AutoModel, as RQ1) is allowed for columns 0-C only.

Outputs (RQ2/outputs/gsm8k/, git-ignored): <run_id>.json (answers), <run_id>.meta.json (settings, summary,
per-item stats without text), <run_id>.partial.jsonl while running (resumed on restart). The console shows
counts only, never response text.

    FAST_DLLM_ROOT=/root/Fast-dLLM python run_rq2_gsm8k.py --column A --defense dg
    python run_rq2_gsm8k.py --column D --defense none --limit 10        # pilot / timing
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

RQ2_DIR = Path(__file__).resolve().parent
REPO = RQ2_DIR.parent
DG_DIR = REPO / "DiffuGuard"
for p in (RQ2_DIR, DG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_generate as rg          # noqa: E402  (DiffuGuard/run_generate.py: seeding, original-code import, git)
import run_gsm8k as rq1            # noqa: E402  (DiffuGuard/run_gsm8k.py: prompts, scoring, post-processing)
from src import combined           # noqa: E402

COLUMNS = {
    "0": dict(mode="fixed", threshold=None, cache=None),
    "A": dict(mode="threshold", threshold=0.9, cache=None),
    "B": dict(mode="threshold", threshold=0.7, cache=None),
    "C": dict(mode="threshold", threshold=0.5, cache=None),
    "D": dict(mode="threshold", threshold=0.9, cache="prefix"),
    "E": dict(mode="threshold", threshold=0.9, cache="dual"),
}
TASK = dict(gen_length=256, block_length=32, temperature=0.0, steps=256)
DEFAULT_IDS = REPO / "RQ1" / "data" / "ids" / "gsm8k_250.txt"
DEFAULT_OUT = RQ2_DIR / "outputs" / "gsm8k"
DG_CONFIG = DG_DIR / "configs" / "llada_dija.yaml"


# --------------------------------------------------------------------------- settings

def build_parser():
    p = argparse.ArgumentParser(description="RQ2 GSM8K: combined decoder, one (column, defense) cell per run")
    p.add_argument("--column", required=True, choices=sorted(COLUMNS))
    p.add_argument("--defense", required=True, choices=["none", "dg"])
    p.add_argument("--doc_ids", default=str(DEFAULT_IDS), help="RQ1 id list; '' = all 1,319 (or --limit)")
    p.add_argument("--limit", type=int, help="first N questions of the id list (pilot)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_class", choices=["fastdllm", "auto"], default="fastdllm")
    p.add_argument("--fast_dllm_root", default=os.environ.get("FAST_DLLM_ROOT"))
    p.add_argument("--model_path")
    p.add_argument("--model_revision")
    p.add_argument("--prompt_file", default=str(rq1.DEFAULT_PROMPTS))
    p.add_argument("--prompt_file_sha256", default=rq1.PROMPTS_SHA256)
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--tag")
    # overrides (sweeps / RQ1-compatible checks); the defaults are the agreed RQ2 design
    p.add_argument("--steps", type=int, help="column 0 only (default 256)")
    p.add_argument("--threshold", type=float)
    p.add_argument("--block_length", type=int)
    p.add_argument("--refinement_steps", type=int)
    p.add_argument("--alpha_rule", choices=["progress", "original"], help="default progress (original: column 0)")
    p.add_argument("--alpha0", type=float)
    p.add_argument("--sp_threshold", type=float)
    p.add_argument("--dry_run", action="store_true", help="print the settings and exit (no model, no data)")
    return p


def settings(args):
    """Decoder keyword arguments for this cell (everything combined.generate needs except model / prompt)."""
    col = COLUMNS[args.column]
    s = dict(TASK, **col)
    if args.block_length:
        s["block_length"] = args.block_length
    if args.steps is not None:
        if col["mode"] != "fixed":
            raise SystemExit("--steps applies to column 0 only")
        s["steps"] = args.steps
    if col["mode"] != "fixed":
        s.pop("steps")
    if args.threshold is not None:
        if col["mode"] != "threshold":
            raise SystemExit("--threshold applies to columns A-E only")
        s["threshold"] = args.threshold
    s["defense"] = args.defense == "dg"
    if s["defense"]:
        with open(DG_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pre = cfg["presets"]["diffuguard"]
        s.update(alpha0=float(pre["alpha0"]), sp_threshold=float(pre["sp_threshold"]),
                 remask_ratio=float(pre["remask_ratio"]), refinement_steps=int(pre["refinement_steps"]),
                 suppression_value=float(cfg["suppression_value"]), alpha_rule="progress")
        for k in ("refinement_steps", "alpha_rule", "alpha0", "sp_threshold"):
            if getattr(args, k) is not None:
                s[k] = getattr(args, k)
        if s["alpha_rule"] == "original" and col["mode"] != "fixed":
            raise SystemExit("--alpha_rule original needs a fixed step count (column 0)")
    else:
        for k in ("refinement_steps", "alpha_rule", "alpha0", "sp_threshold"):
            if getattr(args, k) is not None:
                raise SystemExit(f"--{k} needs --defense dg")
    if s["gen_length"] % s["block_length"]:
        raise SystemExit("gen_length must be a multiple of block_length")
    return s


def run_id(args):
    rid = f"rq2-gsm8k-{args.column}-{args.defense}-s{args.seed}"
    if args.doc_ids and Path(args.doc_ids).resolve() != DEFAULT_IDS.resolve():
        rid += f"_ids{Path(args.doc_ids).stem}"
    if not args.doc_ids:
        rid += "_all"
    if args.limit:
        rid += f"_limit{args.limit}"
    if args.tag:
        rid += f"_{args.tag}"
    return rid


# --------------------------------------------------------------------------- generation (model-agnostic, CPU-testable)

def run_items(items, spec, tokenizer, model, s, jbl, *, seed=0, is_instruct=True, partial_path=None, echo=print):
    """One record per item, resumable from partial_path. jbl = original models/jailbreakbench_llada.py (prompt
    building, tokenization, baseline hidden state), as in the RQ1 runner."""
    device = next(model.parameters()).device
    cue_ids = combined.cue_id_sets(tokenizer) if s["defense"] else None
    gen_kw = {k: v for k, v in s.items() if k not in ("temperature",)}
    done = {}
    if partial_path is not None and Path(partial_path).exists():
        for line in Path(partial_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["doc_id"]] = r
        echo(f"resuming: {len(done)} items already in {Path(partial_path).name}")
    records = []
    for pos, item in enumerate(items):
        if item["doc_id"] in done:
            records.append(done[item["doc_id"]])
            continue
        rg.set_seed(seed + item["doc_id"])          # by question, as RQ1: subsets match the full run
        prompt = jbl.build_chat_prompt(tokenizer, item["context"], is_instruct, None)
        input_ids, attention_mask = jbl.get_tokenized_input(prompt, tokenizer, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        baseline_hidden = None
        if s["defense"]:
            baseline_hidden = jbl.compute_baseline_hidden(vanilla_text=item["context"], tokenizer=tokenizer,
                                                          model=model, is_instruct=is_instruct, system_prompt=None,
                                                          debug_print=False)
            if baseline_hidden is None:
                raise SystemExit(f"doc {item['doc_id']}: baseline hidden state failed")
        out, st = combined.generate(model, input_ids, attention_mask=attention_mask, temperature=s["temperature"],
                                    baseline_hidden=baseline_hidden, cue_ids=cue_ids, **gen_kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tail = out[0][input_ids.shape[1]:]
        if bool((tail == combined.MASK_ID).any()):
            raise SystemExit(f"doc {item['doc_id']}: masks left after generation")
        response, n_tokens = rq1.postprocess(tokenizer, tail, item["until"])
        rec = {"doc_id": item["doc_id"], "response": response, "target": item["target"],
               **rq1.score(response, item["target"], spec),
               "prompt_tokens": int(input_ids.shape[1]), "n_tokens": n_tokens,
               "nfe_gen": st.nfe_gen, "nfe_cached": st.nfe_cached, "nfe_repair": st.nfe_repair,
               "nfe_baseline": int(s["defense"]), "wall_ms": round(1000 * dt, 1),
               "audit_score": st.audit_score, "audit_logit": st.audit_logit, "audit_hidden": st.audit_hidden,
               "audit_unsafe": st.audit_unsafe, "repaired": st.repaired}
        records.append(rec)
        if partial_path is not None:
            with open(partial_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        acc = sum(r["correct_flexible"] for r in records) / len(records)
        sd = "-" if st.audit_score is None else f"{st.audit_score:.3f}"
        echo(f"[{pos + 1}/{len(items)}] doc={item['doc_id']} nfe={st.nfe_gen}+{st.nfe_repair} "
             f"t={dt:.1f}s sd={sd} repaired={st.repaired} correct={rec['correct_flexible']} acc={100 * acc:.1f}")
    return records


def summarize(records):
    n = len(records)
    out = {"n": n}
    for key in ("flexible", "strict"):
        k = sum(bool(r[f"correct_{key}"]) for r in records)
        out[f"accuracy_{key}"] = round(100 * k / n, 2) if n else None
        out[f"accuracy_{key}_ci95"] = rq1.wilson(k, n)
    inv = sum(r["pred_flexible"] == "[invalid]" for r in records)
    out["invalid_flexible"] = inv
    out["invalid_rate"] = round(inv / max(n, 1), 4)          # non-answer rate
    out["repair_rate"] = round(sum(bool(r["repaired"]) for r in records) / max(n, 1), 4)
    for k in ("nfe_gen", "nfe_cached", "nfe_repair", "nfe_baseline", "wall_ms"):
        tot = sum(r[k] for r in records)
        out[f"total_{k}"] = round(tot, 1)
        out[f"mean_{k}"] = round(tot / max(n, 1), 2)
    secs = out["total_wall_ms"] / 1000
    out["tokens_per_second"] = round(sum(r["n_tokens"] for r in records) / max(secs, 1e-9), 3)
    return out


# --------------------------------------------------------------------------- model

def load_model(args, model_path, revision):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    kw = {} if Path(model_path).is_dir() else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **kw)
    if args.model_class == "auto":
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **kw)
        source = "transformers.AutoModel (trust_remote_code)"
    else:
        root = Path(args.fast_dllm_root or "")
        llada = root / "v1" / "llada"
        if not (llada / "model" / "modeling_llada.py").exists():
            raise SystemExit("--model_class fastdllm needs --fast_dllm_root / FAST_DLLM_ROOT (Fast-dLLM a9b81e4)")
        sys.path.insert(0, str(llada))
        from model.modeling_llada import LLaDAModelLM      # same loading as Fast-dLLM eval_llada.py
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, **kw)
        model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             config=config, **kw)
        source = f"Fast-dLLM LLaDAModelLM ({rg.git_commit(root)})"
    return tokenizer, model.to("cuda").eval(), source


def main(argv=None):
    args = build_parser().parse_args(argv)
    s = settings(args)
    if args.model_class == "auto" and s.get("cache"):
        raise SystemExit("columns D/E need --model_class fastdllm (KV-cache API)")
    rid = run_id(args)
    if args.dry_run:
        info = {"run_id": rid, "settings": s, "model_class": args.model_class, "doc_ids": args.doc_ids,
                "limit": args.limit}
        print(json.dumps(info, indent=2))
        return info

    with open(DG_CONFIG, "r", encoding="utf-8") as f:
        dg_cfg = yaml.safe_load(f)
    model_path = args.model_path or dg_cfg["model_path"]
    items, spec, prompt_sha = rq1.load_items({
        "prompt_file": args.prompt_file, "prompt_file_sha256": args.prompt_file_sha256,
        "doc_ids": str(Path(args.doc_ids).resolve()) if args.doc_ids else None, "limit": None})
    if args.limit:
        items = items[: args.limit]
    dg_root = rg.resolve(dg_cfg["diffuguard_root"])
    jbl = rg.import_original(dg_root)

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU visible. Refusing to run an 8B model on CPU (use --dry_run).")
    revision = rg.resolve_model_revision(model_path, args.model_revision or dg_cfg.get("model_revision"))
    tokenizer, model, model_source = load_model(args, model_path, revision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{rid}.partial.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    records = run_items(items, spec, tokenizer, model, s, jbl, seed=args.seed,
                        is_instruct=rg.is_instruct_path(model_path), partial_path=partial,
                        echo=lambda m: print(m, flush=True))
    wall = time.perf_counter() - t0

    ans_path = out_dir / f"{rid}.json"
    keep = ("doc_id", "response", "target", "pred_flexible", "pred_strict", "correct_flexible", "correct_strict")
    with open(ans_path, "w", encoding="utf-8") as f:
        json.dump([{k: r[k] for k in keep} for r in records], f, indent=1, ensure_ascii=False)
    meta = {
        "run_id": rid, "task": "gsm8k", "column": args.column, "defense": args.defense,
        "accel": COLUMNS[args.column]["mode"], "threshold": s.get("threshold"), "cache": s.get("cache"),
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        **summarize(records), "wall_seconds_this_invocation": round(wall, 1),
        "settings": s, "seed": args.seed, "seeding": "set_seed(seed + doc_id) before each item",
        "doc_ids": args.doc_ids, "limit": args.limit, "prompt_file_sha256": prompt_sha,
        "model_class": model_source, "model_path": model_path, "model_revision": revision,
        "team_repo_commit": rg.git_commit(REPO), "team_repo_dirty": rg.git_dirty(REPO),
        "diffuguard_commit": rg.git_commit(dg_root),
        "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "per_item": [{k: v for k, v in r.items() if k not in ("response", "target")} for r in records],
        "output_json": str(ans_path),
    }
    meta_path = out_dir / f"{rid}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    partial.unlink(missing_ok=True)
    print(f"saved -> {ans_path.name}, {meta_path.name}")
    print(f"n={meta['n']} flexible={meta['accuracy_flexible']}% strict={meta['accuracy_strict']}% "
          f"invalid={meta['invalid_rate']} repair_rate={meta['repair_rate']} "
          f"nfe/item={meta['mean_nfe_gen']}+{meta['mean_nfe_repair']} tok/s={meta['tokens_per_second']}")
    return meta


if __name__ == "__main__":
    main()

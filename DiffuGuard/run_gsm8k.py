"""GSM8K accuracy through the DiffuGuard generation code (RQ1 "성능" columns, with / without DiffuGuard).

Matches the Fast-dLLM reproduction (Fast-dLLM/README.md, eval_llada.py on lm_eval 0.4.8) wherever the DiffuGuard
code allows it:
  * prompts: the exact 5-shot contexts lm_eval builds (prepare_gsm8k.py dumps them), wrapped as one user turn with
    the chat template + generation prompt, like eval_llada.py does for Instruct models
  * decoding: gen_length 256, block_length 32, temperature 0, cfg 0, batch 1; --steps 256 / 128 / 64 / 32 = L1..L8
  * post-processing: cut at the task's stop strings, re-tokenize, decode without special tokens (eval_llada.py)
  * scoring: lm_eval's gsm8k filters and exact_match (flexible-extract = team metric, strict-match also kept)

Decoder (--decoder):
  semi_ar  (default) the original DiffuGuard generate() with its block-widening guard switched off (semi_ar.py),
           i.e. real block-by-block decoding. With --defense vanilla this is token-for-token the Fast-dLLM
           baseline algorithm (tests/test_semi_ar.py). DiffuGuard's audit + repair then act on the first
           32-token block (correct_only_first_block, as in the paper); SAR runs in every block.
  original the unmodified function. With block_length < gen_length it decodes the whole tail during the first
           block's steps (steps / 8) and wastes the rest; only meaningful with --block_length 256.

Differences from the jailbreak runs (run_generate.py), on purpose: fill_all_masks off and no 36-mask user-turn tail
(mask_counts 0), because the Fast-dLLM baseline has neither. The audit baseline h_origin is the same prompt
without the mask tail, exactly what the original script does for prompts that are not attacks (zeroshot).

Known remaining difference from the Fast-dLLM run: it loads the model through Fast-dLLM's own model class and
ranks confidences in float64; the DiffuGuard code uses AutoModel and the model's dtype. Compare with/without
DiffuGuard only between runs of THIS script; use the Fast-dLLM 82% (first 100) as a sanity check for
`--defense vanilla --steps 256 --limit 100`.

    python run_gsm8k.py --defense vanilla    --steps 256 --limit 100
    python run_gsm8k.py --defense diffuguard --steps 32  --limit 250
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

import run_generate as rg
import semi_ar

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = PROJECT_DIR / "outputs" / "data" / "gsm8k_5shot_lmeval.json"
# sha256 of the file prepare_gsm8k.py writes with lm_eval 0.4.8 (1,319 prompts)
PROMPTS_SHA256 = "028420a0b305a1f47ea6f229592bc82b4ae7417677bf2ce9fef166f8bb9578c6"
# Fast-dLLM reproduction settings (Fast-dLLM/README.md); the DiffuGuard preset is applied on top
TASK_GEN = {"steps": 256, "gen_length": 256, "block_length": 32, "temperature": 0.0, "cfg_scale": 0.0,
            "fill_all_masks": False, "mask_counts": 0}
OVERRIDE_KEYS = {"steps": int, "gen_length": int, "block_length": int, "alpha0": float, "sp_threshold": float,
                 "remask_ratio": float, "refinement_steps": int}
EOS_ID = 126081   # eval_llada.py counts generated tokens as ids != 126081


# --------------------------------------------------------------------------- scoring (lm_eval gsm8k)

def regex_filter(resp, pattern, group_select=0, fallback="[invalid]"):
    """lm_eval.filters.extraction.RegexFilter for one response."""
    match = re.compile(pattern).findall(resp)
    if not match:
        return fallback
    match = match[group_select]
    if isinstance(match, tuple):
        match = [m for m in match if m]
        match = match[0] if match else fallback
    return match.strip()


def exact_match(pred, target, regexes_to_ignore):
    """lm_eval exact_match with regexes_to_ignore + ignore_case (the gsm8k task settings)."""
    for s in regexes_to_ignore:
        pred, target = re.sub(s, "", pred), re.sub(s, "", target)
    return pred.lower() == target.lower()


def score(response, target, spec):
    out = {}
    for f in spec["filters"]:
        rx = next(x for x in f["filter"] if x["function"] == "regex")
        if [x["function"] for x in f["filter"]] != ["regex", "take_first"]:
            raise SystemExit(f"unsupported filter chain in prompt file: {f}")
        pred = regex_filter(response, rx["regex_pattern"], rx.get("group_select", 0))
        key = f["name"].split("-")[0]          # strict-match -> strict, flexible-extract -> flexible
        out[f"pred_{key}"] = pred
        out[f"correct_{key}"] = exact_match(pred, target, spec["metric_regexes_to_ignore"])
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def postprocess(tokenizer, tail_ids, until):
    """eval_llada.py generate_until, non-HumanEval branch."""
    text = tokenizer.decode(tail_ids, skip_special_tokens=False)
    for stop in until:
        if stop in text:
            text = text.split(stop)[0]
    ids = tokenizer(text)["input_ids"]
    n_tokens = sum(1 for t in ids if int(t) != EOS_ID)
    return tokenizer.decode(ids, skip_special_tokens=True), n_tokens


# --------------------------------------------------------------------------- config

def build_parser():
    p = argparse.ArgumentParser(description="GSM8K x LLaDA x {vanilla | DiffuGuard} through the DiffuGuard code")
    p.add_argument("--config", default=str(rg.DEFAULT_CONFIG), help="model path / presets (same file as run_generate)")
    p.add_argument("--defense", required=True, help="preset: vanilla | diffuguard | no_sar | no_audit")
    p.add_argument("--decoder", choices=["semi_ar", "original"], default="semi_ar")
    p.add_argument("--prompt_file", default=str(DEFAULT_PROMPTS))
    p.add_argument("--prompt_file_sha256", default=PROMPTS_SHA256, help="'' disables the check")
    p.add_argument("--limit", type=int, help="first N test questions (the Fast-dLLM run used 100)")
    p.add_argument("--doc_ids", help="RQ1 id list (tab separated, header item_id/source_key; source_key = test index)")
    p.add_argument("--diffuguard_root")
    p.add_argument("--model_path")
    p.add_argument("--model_revision")
    p.add_argument("--output_dir", default="outputs/gsm8k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    for k, t in OVERRIDE_KEYS.items():
        p.add_argument(f"--{k}", type=t)
    p.add_argument("--dry_run", action="store_true", help="tokenizer only: prompt lengths and forward counts")
    return p


def load_config(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("defense", "decoder", "prompt_file", "prompt_file_sha256", "limit", "doc_ids", "diffuguard_root",
              "model_path", "model_revision", "output_dir", "seed", "tag", "dry_run"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    cfg.setdefault("limit", None), cfg.setdefault("doc_ids", None), cfg.setdefault("tag", None)
    presets = cfg.get("presets", {})
    if cfg["defense"] not in presets:
        raise SystemExit(f"Unknown defense preset {cfg['defense']!r}; choose from {sorted(presets)}")
    if cfg["limit"] and cfg["doc_ids"]:
        raise SystemExit("use --limit or --doc_ids, not both")

    gen = dict(rg.ORIGINAL_DEFAULTS)
    gen.update({k: cfg[k] for k in rg.GEN_KEYS if k in cfg})
    gen.update(TASK_GEN)
    gen.update(presets[cfg["defense"]])
    overrides = {k: getattr(args, k) for k in OVERRIDE_KEYS if getattr(args, k) is not None}
    gen.update(overrides)
    for k, v in rg.ORIGINAL_DEFAULTS.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and gen[k] is not None:
            gen[k] = type(v)(gen[k])
    if gen["sp_mode"] is False:
        gen["sp_mode"] = "off"
    if gen["gen_length"] % gen["block_length"]:
        raise SystemExit("gen_length must be a multiple of block_length")
    num_blocks = gen["gen_length"] // gen["block_length"]
    if gen["steps"] % num_blocks:
        raise SystemExit(f"steps must be a multiple of the number of blocks ({num_blocks}), like Fast-dLLM asserts")
    if cfg["decoder"] == "original" and num_blocks > 1:
        rg.log.warning("decoder=original with %d blocks: the original code decodes everything in the first "
                       "block's %d steps (see semi_ar.py)", num_blocks, gen["steps"] // num_blocks)
    cfg["gen"], cfg["overrides"] = gen, overrides
    cfg["config_path"] = str(Path(args.config).resolve())
    return cfg


def load_items(cfg):
    path = rg.resolve(cfg["prompt_file"])
    if not path.exists():
        raise SystemExit(f"Prompt file not found: {path}\nBuild it with prepare_gsm8k.py (needs lm_eval 0.4.8) "
                         "or copy it from a machine that has it.")
    digest = rg.sha256_file(path)
    expected = cfg.get("prompt_file_sha256")
    if expected and digest != expected:
        raise SystemExit(f"Prompt file sha256 mismatch: {digest} != {expected}")
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    items = spec.pop("items")
    if cfg["doc_ids"]:
        lines = Path(cfg["doc_ids"]).read_text(encoding="utf-8").split("\n")[1:]
        wanted = [int(line.split("\t")[1]) for line in lines if line.strip()]
        by_id = {it["doc_id"]: it for it in items}
        items = [by_id[i] for i in wanted]
    elif cfg["limit"]:
        items = items[: cfg["limit"]]
    return items, spec, digest


def run_name(cfg):
    g = cfg["gen"]
    name = f"llada_instruct_gsm8k_{cfg['defense']}_{cfg['decoder']}_steps{g['steps']}_seed{cfg['seed']}"
    if cfg["tag"]:
        name += f"_{cfg['tag']}"
    if cfg["limit"]:
        name += f"_limit{cfg['limit']}"
    if cfg["doc_ids"]:
        name += f"_ids{Path(cfg['doc_ids']).stem}"
    return name


def expected_forwards(g, repaired):
    num_blocks = g["gen_length"] // g["block_length"]
    return (g["steps"] // num_blocks) * num_blocks + (max(int(g["refinement_steps"]), 1) if repaired else 0)


# --------------------------------------------------------------------------- generation

def run(cfg, items, spec, tokenizer, model, jbl, gfl, partial_path=None):
    """One record per item. Resumes from partial_path (jsonl) when it exists."""
    g = cfg["gen"]
    is_instruct = rg.is_instruct_path(cfg["model_path"])
    steps_per_block = g["steps"] // (g["gen_length"] // g["block_length"])
    accept_ids, refuse_ids = gfl.build_cue_id_sets(tokenizer)
    state = {"n": 0, "gen_start": None, "prompt_len": None, "logit": None, "sd": None}
    original_cosine_distance = gfl.cosine_distance

    def _record_sd(a, b):
        dist = original_cosine_distance(a, b)
        if state["sd"] is None:        # first block = the one the audit can repair
            state["sd"] = dist
        return dist

    def _count(_module, _inputs):
        state["n"] += 1

    def _logit_score(_module, _inputs, out):
        # logits of the last denoising step of block 1: what the original's logit audit path reads
        if state["gen_start"] is None or state["n"] - state["gen_start"] != steps_per_block:
            return
        end = g["gen_length"] if cfg["decoder"] == "original" else g["block_length"]
        positions = torch.zeros(out.logits.shape[:2], dtype=torch.bool, device=out.logits.device)
        positions[:, state["prompt_len"]: state["prompt_len"] + end] = True
        state["logit"] = gfl.safety_perplexity_from_logits(out.logits, positions, accept_ids, refuse_ids)

    done = {}
    if partial_path is not None and Path(partial_path).exists():
        for line in Path(partial_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["doc_id"]] = r
        print(f"resuming: {len(done)} items already in {partial_path}", flush=True)

    gfl.cosine_distance = _record_sd
    h1 = model.register_forward_pre_hook(_count)
    h2 = model.register_forward_hook(_logit_score) if g["sp_mode"] != "off" else None
    records = []
    try:
        with torch.no_grad():
            for pos, item in enumerate(items):
                if item["doc_id"] in done:
                    records.append(done[item["doc_id"]])
                    continue
                rg.set_seed(cfg["seed"] + item["doc_id"])       # by question, so subsets match the full run
                prompt = jbl.build_chat_prompt(tokenizer, item["context"], is_instruct, None)
                input_ids, attention_mask = jbl.get_tokenized_input(prompt, tokenizer, model.device)
                if bool((input_ids == g["mask_id"]).any()):
                    raise SystemExit(f"doc {item['doc_id']}: mask token inside the prompt")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                state.update(n=0, gen_start=None, prompt_len=input_ids.shape[1], logit=None, sd=None)
                debug = io.StringIO()
                with contextlib.redirect_stdout(debug):
                    baseline_hidden = None
                    if g["sp_mode"] == "hidden":
                        baseline_hidden = jbl.compute_baseline_hidden(
                            vanilla_text=item["context"], tokenizer=tokenizer, model=model,
                            is_instruct=is_instruct, system_prompt=None, debug_print=False)
                        if baseline_hidden is None:
                            raise SystemExit(f"doc {item['doc_id']}: baseline hidden state failed")
                    state["gen_start"] = state["n"]
                    out = gfl.generate_llada(
                        input_ids=input_ids, attention_mask=attention_mask, model=model,
                        steps=g["steps"], gen_length=g["gen_length"], block_length=g["block_length"],
                        temperature=g["temperature"], mask_id=g["mask_id"], tokenizer=tokenizer,
                        cfg_scale=g["cfg_scale"], remasking=g["remasking"], random_rate=g["random_rate"],
                        injection_step=g["injection_step"], alpha0=g["alpha0"], sp_mode=g["sp_mode"],
                        sp_threshold=g["sp_threshold"], refinement_steps=g["refinement_steps"],
                        remask_ratio=g["remask_ratio"], suppression_value=g["suppression_value"],
                        correct_only_first_block=g["correct_only_first_block"],
                        fill_all_masks=g["fill_all_masks"], debug_print=True, baseline_hidden=baseline_hidden,
                        attack_method="none", protected_index=torch.zeros_like(input_ids, dtype=torch.bool))
                    gen_forwards = state["n"] - state["gen_start"]
                    state["gen_start"] = None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                tail = out[0][input_ids.shape[1]:]
                if bool((tail == g["mask_id"]).any()):
                    raise SystemExit(f"doc {item['doc_id']}: masks left after generation")
                response, n_tokens = postprocess(tokenizer, tail, item["until"])
                audit = rg.parse_audit(debug.getvalue())
                if state["sd"] is not None:
                    audit["audit_score"] = state["sd"]
                if gen_forwards != expected_forwards(g, audit["repaired"]):
                    rg.log.warning("doc %d: %d generation forwards, expected %d", item["doc_id"], gen_forwards,
                                   expected_forwards(g, audit["repaired"]))
                rec = {"doc_id": item["doc_id"], "response": response, "target": item["target"],
                       **score(response, item["target"], spec),
                       "prompt_tokens": int(input_ids.shape[1]), "n_tokens": n_tokens,
                       "n_forward": state["n"], "gen_forwards": gen_forwards, "seconds": round(dt, 3),
                       **audit, "logit_score": state["logit"]}
                records.append(rec)
                if partial_path is not None:
                    with open(partial_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                acc = sum(r["correct_flexible"] for r in records) / len(records)
                sd = "-" if audit["audit_score"] is None else f"{audit['audit_score']:.3f}"
                print(f"[{pos + 1}/{len(items)}] doc={item['doc_id']} forwards={gen_forwards} t={dt:.1f}s sd={sd} "
                      f"repaired={audit['repaired']} correct={rec['correct_flexible']} acc={100 * acc:.1f}",
                      flush=True)
    finally:
        gfl.cosine_distance = original_cosine_distance
        h1.remove()
        if h2 is not None:
            h2.remove()
    return records


def summarize(records):
    n = len(records)
    out = {"n": n}
    for key in ("flexible", "strict"):
        k = sum(bool(r[f"correct_{key}"]) for r in records)
        out[f"accuracy_{key}"] = round(100 * k / n, 2) if n else None
        out[f"accuracy_{key}_ci95"] = wilson(k, n)
    out["invalid_flexible"] = sum(r["pred_flexible"] == "[invalid]" for r in records)
    out["repair_rate"] = round(sum(bool(r["repaired"]) for r in records) / max(n, 1), 4)
    out["total_forwards"] = sum(r["n_forward"] for r in records)
    out["total_gen_forwards"] = sum(r["gen_forwards"] for r in records)
    out["total_item_seconds"] = round(sum(r["seconds"] for r in records), 1)
    out["tokens_per_second"] = round(sum(r["n_tokens"] for r in records) / max(out["total_item_seconds"], 1e-9), 3)
    return out


def main(argv=None):
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    cfg = load_config(argv)
    g = cfg["gen"]
    items, spec, prompt_sha = load_items(cfg)
    dg_root = rg.resolve(cfg["diffuguard_root"])
    jbl = rg.import_original(dg_root)
    gfl = semi_ar.load(dg_root) if cfg["decoder"] == "semi_ar" else sys.modules["utility.generate_function_llada"]
    out_dir = rg.resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(cfg)

    if cfg["dry_run"]:
        tokenizer = rg.load_tokenizer(cfg)
        is_instruct = rg.is_instruct_path(cfg["model_path"])
        lens = [len(tokenizer(jbl.build_chat_prompt(tokenizer, it["context"], is_instruct, None))["input_ids"])
                for it in items]
        info = {"run_name": name, "n_items": len(items), "prompt_tokens_min": min(lens),
                "prompt_tokens_max": max(lens), "gen": g,
                "forwards_per_item": [expected_forwards(g, False) + (g["sp_mode"] == "hidden"),
                                      expected_forwards(g, g["sp_mode"] != "off") + (g["sp_mode"] == "hidden")]}
        print(json.dumps(info, indent=2))
        return info

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU visible. Refusing to run an 8B model on CPU (use --dry_run).")
    from transformers import AutoModel, AutoTokenizer

    revision = rg.resolve_model_revision(cfg["model_path"], cfg.get("model_revision"))
    kw = {} if Path(cfg["model_path"]).is_dir() else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, **kw)
    model = AutoModel.from_pretrained(cfg["model_path"], trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, **kw).to("cuda").eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    started = datetime.now(timezone.utc).isoformat()
    partial = out_dir / f"{name}.partial.jsonl"
    t0 = time.perf_counter()
    records = run(cfg, items, spec, tokenizer, model, jbl, gfl, partial_path=partial)
    wall = time.perf_counter() - t0

    gen_path = out_dir / f"{name}.json"
    keep = ("doc_id", "response", "target", "pred_flexible", "pred_strict", "correct_flexible", "correct_strict")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump([{k: r[k] for k in keep} for r in records], f, indent=1, ensure_ascii=False)
    meta = {
        "run_name": name, "task": "gsm8k", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        **summarize(records),
        "wall_seconds_this_invocation": round(wall, 1),
        "defense": cfg["defense"], "decoder": cfg["decoder"],
        "decoder_patch": getattr(gfl, "SEMI_AR_PATCH", None),
        "gen": g, "cli_overrides": cfg["overrides"], "seed": cfg["seed"],
        "seeding": "set_seed(seed + doc_id) before each item",
        "limit": cfg["limit"], "doc_ids": cfg["doc_ids"],
        "prompt_file_sha256": prompt_sha,
        "prompt_source": {k: spec.get(k) for k in ("task", "num_fewshot", "fewshot_seed", "lm_eval_version")},
        "team_repo_commit": rg.git_commit(PROJECT_DIR), "team_repo_dirty": rg.git_dirty(PROJECT_DIR),
        "diffuguard_commit": rg.git_commit(dg_root), "diffuguard_dirty": rg.git_dirty(dg_root),
        "model_path": cfg["model_path"], "model_revision": revision,
        "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "per_item": [{k: v for k, v in r.items() if k not in ("response", "target")} for r in records],
        "output_json": str(gen_path),
    }
    meta_path = out_dir / f"{name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    partial.unlink(missing_ok=True)
    print(f"saved answers  -> {gen_path}")
    print(f"saved metadata -> {meta_path}")
    print(f"n={meta['n']}  flexible={meta['accuracy_flexible']}%  strict={meta['accuracy_strict']}%  "
          f"repair_rate={meta['repair_rate']}  forwards={meta['total_forwards']}  "
          f"tok/s={meta['tokens_per_second']}")
    return meta


if __name__ == "__main__":
    main()

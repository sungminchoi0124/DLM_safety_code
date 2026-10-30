"""Wrapper around DiffuGuard's models/jailbreakbench_llada.py (LLaDA x {DIJA | PAD | zeroshot} prompts).

Reuses the original functions unchanged (pick_two_prompt_fields, should_use_refined, process_user_text,
build_chat_prompt, build_detection_prompt, first_step_tail_mean_hidden, compute_baseline_hidden,
generate_response -> utility/generate_function_llada.generate_llada) in the same order as the original
main() loop, and adds:
  * config / presets (vanilla | diffuguard | no_sar | no_audit) and attacks (dija_jbb | pad_advbench |
    jbb_behaviors) instead of long CLI lines
  * --seed (torch / numpy / random, re-seeded per item so --limit N matches the full run)
  * hard failure when no CUDA GPU is visible (unless --dry_run); no nvidia-smi auto-pick
  * no response text on console/logs (length + sha256 prefix only)
  * per-item audit record parsed from the original debug output: SD score, unsafe flag, repair triggered,
    plus the logit-based score the original also checks (overwritten by SD in its debug print), recomputed
    with the original safety_perplexity_from_logits on the last denoising step -> repair set for any lambda
  * run_meta.json with commits, model revision, resolved hyper-parameters, forward counts and timings
  * --dry_run: tokenizer only, prints prompt lengths / mask counts / expected forward passes

Output JSON records have the same fields as the original script, so the original evaluation.py reads them.
Non-DIJA records also get "prompt" (the plain request), which evaluation.py judges when "refined prompt" is empty.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent  # DLM_safety_code/DiffuGuard
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "llada_dija.yaml"
TOKENIZER_PATTERNS = ["tokenizer*", "special_tokens_map.json", "config.json", "*.py"]

# argparse defaults of the original models/jailbreakbench_llada.py; config and presets override them
ORIGINAL_DEFAULTS = {
    "remasking": "low_confidence", "random_rate": 0.0, "injection_step": None, "alpha0": 0.3,
    "sp_mode": "off", "sp_threshold": 0.35, "refinement_steps": 8, "remask_ratio": 0.9,
    "suppression_value": 1e6, "fill_all_masks": False, "correct_only_first_block": True,
    "cfg_scale": 0.1, "temperature": 0.5, "steps": 64, "gen_length": 128, "block_length": 128,
    "mask_id": 126336, "mask_counts": 36, "ref_tail_len": 128,
}
GEN_KEYS = list(ORIGINAL_DEFAULTS)
# CLI overrides applied on top of the preset (for later step-count / threshold sweeps)
OVERRIDE_KEYS = {"steps": int, "gen_length": int, "block_length": int, "temperature": float,
                 "cfg_scale": float, "alpha0": float, "sp_threshold": float, "remask_ratio": float,
                 "refinement_steps": int}
# --attack_method values of the original script this wrapper runs ("other" is not implemented there either)
SUPPORTED_ATTACKS = ("DIJA", "PAD", "zeroshot")

# printed by the original generate() when debug_print=True; no model text in these lines
BLOCK_LINE = re.compile(r"\[Block (\d+)/(\d+)\] Safety score=(\S+), unsafe=(True|False)")
REPAIR_LINE = "--> Refinement Phase"

log = logging.getLogger("diffuguard_repro")


class _DropResponseText(logging.Filter):
    """Defence in depth: drop any log record that carries model output or full prompts."""

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(k in msg for k in ("Response:", "PROMPT_PREVIEW=", "TAIL_VAN=", "TAIL_REF="))


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
    """SHA-256 with CRLF normalized to LF (same as DIJA/run_generate.py, so hashes match on Windows checkouts)."""
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
    p = argparse.ArgumentParser(description="DiffuGuard x LLaDA x {DIJA | PAD | zeroshot} (reproduction wrapper)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--defense", help="preset name from the config: vanilla | diffuguard | no_sar | no_audit")
    p.add_argument("--attack", help="attack entry from the config: dija_jbb | pad_advbench | jbb_behaviors")
    p.add_argument("--diffuguard_root")
    p.add_argument("--model_path", help="local dir or HF id, e.g. GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--model_revision", help="HF commit hash of the model")
    p.add_argument("--prompt_file")
    p.add_argument("--prompt_file_sha256", help="expected hash of --prompt_file ('' disables the check)")
    p.add_argument("--prompt_set", help="short name used in output file names, e.g. advbench")
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--tag", help="extra suffix for the output file name (use it when sweeping overrides)")
    for k, t in OVERRIDE_KEYS.items():
        p.add_argument(f"--{k}", type=t)
    p.add_argument("--fill_all_masks", dest="fill_all_masks", action="store_true", default=None)
    p.add_argument("--no_fill_all_masks", dest="fill_all_masks", action="store_false")
    p.add_argument("--limit", type=int, default=None, help="only the first N items")
    p.add_argument("--dry_run", action="store_true", help="tokenizer only; report prompt sizes, no model")
    return p


def load_config(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    attacks = cfg.get("attacks", {})
    cfg["attack"] = args.attack or cfg.get("attack")
    if cfg["attack"] not in attacks:
        raise SystemExit(f"Unknown attack {cfg['attack']!r}; choose from {sorted(attacks)}")
    cfg.update(attacks[cfg["attack"]])  # then explicit --prompt_file etc. win
    for k in ("defense", "diffuguard_root", "model_path", "model_revision", "prompt_file",
              "prompt_file_sha256", "prompt_set", "output_dir", "seed"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    presets = cfg.get("presets", {})
    if cfg["defense"] not in presets:
        raise SystemExit(f"Unknown defense preset {cfg['defense']!r}; choose from {sorted(presets)}")

    gen = dict(ORIGINAL_DEFAULTS)
    gen.update({k: cfg[k] for k in GEN_KEYS if k in cfg})
    gen.update(presets[cfg["defense"]])
    overrides = {k: getattr(args, k) for k in list(OVERRIDE_KEYS) + ["fill_all_masks"]
                 if getattr(args, k) is not None}
    gen.update(overrides)
    for k, v in ORIGINAL_DEFAULTS.items():
        # YAML quirks: `1.0e6` loads as str, so coerce every numeric knob to the original type
        if isinstance(v, (int, float)) and not isinstance(v, bool) and gen[k] is not None:
            gen[k] = type(v)(gen[k])
    if gen["sp_mode"] is False:  # unquoted `off` in YAML
        gen["sp_mode"] = "off"
    if gen["sp_mode"] not in ("off", "logit", "hidden"):
        raise SystemExit(f"bad sp_mode {gen['sp_mode']!r}")
    if gen["gen_length"] % gen["block_length"]:
        raise SystemExit("gen_length must be a multiple of block_length (original assert)")

    cfg["gen"] = gen
    cfg["overrides"] = overrides
    cfg["tag"] = args.tag
    cfg["limit"] = args.limit
    cfg["dry_run"] = args.dry_run
    cfg["config_path"] = str(Path(args.config).resolve())
    if cfg.get("attack_method") not in SUPPORTED_ATTACKS:
        raise SystemExit(f"attack_method must be one of {SUPPORTED_ATTACKS}, got {cfg.get('attack_method')!r}")
    return cfg


# --------------------------------------------------------------------------- original code

def import_original(diffuguard_root):
    """Import models/jailbreakbench_llada.py from the pinned submodule without modifying it."""
    root = Path(diffuguard_root).resolve()
    src = root / "models" / "jailbreakbench_llada.py"
    if not src.exists():
        raise SystemExit(f"Original DiffuGuard code not found: {src} (run `git submodule update --init`).")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import defense_utils  # noqa: F401  (pulls bpe/openai/anthropic/boto3/google-generativeai)
    except ImportError as e:
        # Defender is only constructed for --defense_method self-reminder/ppl/...; this run uses none.
        log.warning("defense_utils unavailable (%s); using a stub. Prompt-level defenses are disabled.", e)
        stub = types.ModuleType("defense_utils")

        class Defender:
            def __init__(self, *a, **k):
                raise RuntimeError("defense_utils stub: install full requirements to use prompt defenses")

        stub.Defender = Defender
        sys.modules["defense_utils"] = stub
    spec = importlib.util.spec_from_file_location("diffuguard_jailbreakbench_llada", src)
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
    if not prompt_file.exists():
        raise SystemExit(f"Prompt file not found: {prompt_file}")
    digest = sha256_file(prompt_file)
    expected = cfg.get("prompt_file_sha256")
    if expected and digest != expected:
        raise SystemExit(f"Prompt file sha256 mismatch: {digest} != {expected}. Was it regenerated?")
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if cfg.get("limit"):
        data = data[: cfg["limit"]]
    return data, digest


def gen_namespace(cfg):
    """Attributes the original generate_response() reads from its argparse namespace."""
    g = cfg["gen"]
    return argparse.Namespace(attack_method=cfg["attack_method"], debug_print=True,
                              **{k: g[k] for k in GEN_KEYS})


def is_instruct_path(model_path):
    # identical rule to the original main()
    return ("instruct" in model_path.lower()) or ("1.5" in model_path)


def build_prompts(item, tokenizer, jbl, cfg, is_instruct):
    """Same steps as the original main() loop with defense_method=None (no system prompt)."""
    g = cfg["gen"]
    vanilla_text, refined_text = jbl.pick_two_prompt_fields(item)
    use_refined = jbl.should_use_refined(gen_namespace(cfg), str(cfg["prompt_file"]), refined_text) \
        and bool(refined_text.strip())
    if cfg["attack_method"] == "DIJA" and not use_refined:
        raise SystemExit("DIJA item without a refined prompt; check the prompt file fields.")
    if cfg["attack_method"] != "DIJA" and use_refined:
        # the original silently switches to the refined (DIJA) text when the file name contains "refine"
        raise SystemExit(f"{cfg['attack_method']} run would use refined (DIJA) prompts because the prompt file "
                         "name contains 'refine'; use a file without refined prompts or rename it.")
    if not vanilla_text.strip():
        raise SystemExit("item without a prompt (expected 'vanilla prompt', 'goal' or 'Behavior').")
    chosen_text = refined_text if use_refined else vanilla_text
    chosen_user = jbl.process_user_text(chosen_text, g["mask_counts"])
    vanilla_user = jbl.process_user_text(vanilla_text, g["mask_counts"])
    return {
        "vanilla_text": vanilla_text,
        "refined_text": refined_text,
        "chosen_text": chosen_text,
        "used_prompt_type": "refined" if use_refined else "vanilla",
        "prompt": jbl.build_chat_prompt(tokenizer, chosen_user, is_instruct, None),
        "vanilla_prompt": jbl.build_chat_prompt(tokenizer, vanilla_user, is_instruct, None),
        "det_vanilla_prompt": jbl.build_detection_prompt(tokenizer, vanilla_text, is_instruct, None,
                                                         g["ref_tail_len"]),
        "det_refined_prompt": jbl.build_detection_prompt(tokenizer, chosen_text, is_instruct, None,
                                                         g["ref_tail_len"]),
    }


def parse_audit(debug_out):
    """SD / unsafe flag / repair from the original debug prints (first block)."""
    m = BLOCK_LINE.search(debug_out)
    score = None
    if m:
        try:
            score = float(m.group(3))
        except ValueError:
            score = None
    return {"audit_score": score, "audit_unsafe": (m.group(4) == "True") if m else None,
            "repaired": REPAIR_LINE in debug_out}


def expected_forwards(g, repaired):
    num_blocks = max(g["gen_length"] // g["block_length"], 1)
    n = max(g["steps"] // num_blocks, 1) * num_blocks
    if repaired:
        n += max(int(g["refinement_steps"]), 1)
    return n


# --------------------------------------------------------------------------- dry run

def dry_run(cfg, jbl, items, tokenizer):
    g = cfg["gen"]
    mask_tok_id = tokenizer.convert_tokens_to_ids(jbl.MASK_TOKEN)
    if mask_tok_id != g["mask_id"]:
        raise SystemExit(f"Mask token id {mask_tok_id} != expected {g['mask_id']}")
    is_instruct = is_instruct_path(cfg["model_path"])
    per_item = []
    for idx, item in enumerate(items):
        p = build_prompts(item, tokenizer, jbl, cfg, is_instruct)
        ids = tokenizer(p["prompt"], return_tensors="pt")["input_ids"][0]
        det = tokenizer(p["det_refined_prompt"], return_tensors="pt")["input_ids"][0]
        n_mask = int((ids == g["mask_id"]).sum())
        per_item.append({"idx": idx, "prompt_tokens": int(ids.numel()), "prompt_masks": n_mask,
                         "seq_len": int(ids.numel()) + g["gen_length"],
                         "masks_to_fill": n_mask + g["gen_length"],
                         "det_tokens": int(det.numel())})
    extra = 2 + (1 if g["sp_mode"] == "hidden" else 0)  # tail-detection pair + baseline pass
    lo = len(items) * (expected_forwards(g, False) + extra)
    hi = len(items) * (expected_forwards(g, g["sp_mode"] != "off") + extra)
    summary = {
        "attack": cfg["attack"],
        "attack_method": cfg["attack_method"],
        "defense": cfg["defense"],
        "gen": g,
        "n_items": len(per_item),
        "mask_token_id": mask_tok_id,
        "is_instruct": is_instruct,
        "prompt_masks_min": min(r["prompt_masks"] for r in per_item),
        "prompt_masks_max": max(r["prompt_masks"] for r in per_item),
        "seq_len_max": max(r["seq_len"] for r in per_item),
        "total_forwards_min": lo,
        "total_forwards_max": hi,
        "est_forward_sec": cfg["est_forward_sec"],
        "est_total_minutes": [round(lo * cfg["est_forward_sec"] / 60, 1),
                              round(hi * cfg["est_forward_sec"] / 60, 1)],
    }
    return summary, per_item


# --------------------------------------------------------------------------- generation

def run_generation(cfg, jbl, items, tokenizer, model):
    """Original main() loop, minus response logging, plus seeding / timing / audit bookkeeping."""
    args = gen_namespace(cfg)
    g = cfg["gen"]
    is_instruct = is_instruct_path(cfg["model_path"])
    n_forward = [0]
    gfl = sys.modules["utility.generate_function_llada"]
    accept_ids, refuse_ids = gfl.build_cue_id_sets(tokenizer)
    # the original audit reads the logits of the last denoising step over the block; with fill_all_masks and a
    # single block that block is the whole sequence
    score_logits = g["fill_all_masks"] and g["gen_length"] // g["block_length"] == 1
    logit_state = {"gen_start": None, "score": None}

    def _count(_module, _inputs):
        n_forward[0] += 1

    def _logit_score(_module, _inputs, out):
        if logit_state["gen_start"] is None or n_forward[0] - logit_state["gen_start"] != g["steps"]:
            return
        logits = out.logits
        if g["cfg_scale"] > 0.0:  # same guidance as the original before it audits
            cond, uncond = torch.chunk(logits, 2, dim=0)
            logits = uncond + (g["cfg_scale"] + 1) * (cond - uncond)
        positions = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        logit_state["score"] = gfl.safety_perplexity_from_logits(logits, positions, accept_ids, refuse_ids)

    handle = model.register_forward_pre_hook(_count)
    score_handle = model.register_forward_hook(_logit_score) if score_logits else None
    results, per_item = [], []
    try:
        with torch.no_grad():
            for idx, item in enumerate(items):
                set_seed(cfg["seed"] + idx)
                p = build_prompts(item, tokenizer, jbl, cfg, is_instruct)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                # tail-only detection metric (recorded by the original script, not used for repair)
                sp_hid_tail, template_attack = None, None
                try:
                    van_mu = jbl.first_step_tail_mean_hidden(model, tokenizer, p["det_vanilla_prompt"],
                                                             g["mask_id"], g["ref_tail_len"])
                    ref_mu = jbl.first_step_tail_mean_hidden(model, tokenizer, p["det_refined_prompt"],
                                                             g["mask_id"], g["ref_tail_len"])
                    sp_hid_tail = jbl.cosine_distance_vec(van_mu, ref_mu)
                    template_attack = sp_hid_tail >= g["sp_threshold"]
                except Exception as e:
                    log.info("[TailDetection] skipped due to: %s", type(e).__name__)

                debug = io.StringIO()
                with contextlib.redirect_stdout(debug):
                    # h_origin of Eq. 9: mean last-layer hidden state over the vanilla prompt
                    baseline_hidden = None
                    if g["sp_mode"] == "hidden":
                        baseline_hidden = jbl.compute_baseline_hidden(
                            vanilla_text=p["vanilla_text"], tokenizer=tokenizer, model=model,
                            is_instruct=is_instruct, system_prompt=None, debug_print=False)
                    n_before = n_forward[0]
                    logit_state.update(gen_start=n_before, score=None)
                    response = jbl.generate_response(p["vanilla_prompt"], p["prompt"], tokenizer, model, args,
                                                     baseline_hidden=baseline_hidden, system_prompt=None)
                    logit_state["gen_start"] = None
                gen_forwards = n_forward[0] - n_before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                audit = parse_audit(debug.getvalue())
                d = text_digest(response)
                rec_meta = {"idx": idx, "n_forward": n_forward[0], "gen_forwards": gen_forwards,
                            "seconds": round(dt, 3), "baseline_hidden_ok": baseline_hidden is not None,
                            **audit, "logit_score": logit_state["score"],
                            "response_len": d["len"], "response_sha256_12": d["sha256_12"]}
                n_forward[0] = 0
                if gen_forwards != expected_forwards(g, audit["repaired"]):
                    log.warning("item %d: %d generation forwards, expected %d", idx, gen_forwards,
                                expected_forwards(g, audit["repaired"]))
                per_item.append(rec_meta)
                score = "-" if audit["audit_score"] is None else f"{audit['audit_score']:.3f}"
                lscore = "-" if logit_state["score"] is None else f"{logit_state['score']:.3f}"
                print(f"[{idx + 1}/{len(items)}] forwards={rec_meta['n_forward']} t={dt:.1f}s "
                      f"sd={score} logit={lscore} repaired={audit['repaired']} response_len={d['len']} "
                      f"sha={d['sha256_12']}", flush=True)

                # same record as the original main()
                rec = {
                    "vanilla prompt": p["vanilla_text"],
                    "refined prompt": p["refined_text"],
                    "used_prompt_type": p["used_prompt_type"],
                    "final_prompt": p["prompt"],
                    "response": response,
                    "sp_hid_tail": sp_hid_tail,
                    "template_attack": template_attack,
                    "ref_tail_len": g["ref_tail_len"],
                }
                for k in ("source", "category", "goal", "refined_goal", "Behavior", "Refined_behavior", "target"):
                    if k in item:
                        rec[k] = item[k]
                if p["used_prompt_type"] == "vanilla":
                    # evaluation.py judges item["refined prompt"] or item["prompt"]; the original record has
                    # neither for PAD / zeroshot, so give the judge the request the model was given
                    rec["prompt"] = p["vanilla_text"]
                results.append(rec)
    finally:
        handle.remove()
        if score_handle is not None:
            score_handle.remove()
    return results, per_item


def run_name(cfg):
    g = cfg["gen"]
    name = (f"llada_instruct_{cfg['attack_method']}_{cfg['prompt_set']}_{cfg['defense']}"
            f"_steps{g['steps']}_seed{cfg['seed']}")
    if cfg.get("tag"):
        name += f"_{cfg['tag']}"
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

    dg_root = resolve(cfg["diffuguard_root"])
    jbl = import_original(dg_root)
    _install_log_filter()
    items, prompt_sha = load_items(cfg)
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    if cfg["dry_run"]:
        tokenizer = load_tokenizer(cfg)
        summary, per_item = dry_run(cfg, jbl, items, tokenizer)
        path = out_dir / f"dryrun_{cfg['prompt_set']}_{cfg['defense']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_item": per_item, "prompt_file_sha256": prompt_sha}, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"per-item sizes -> {path}")
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
    repaired = [r["repaired"] for r in per_item]
    meta = {
        "run_name": name,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "team_repo_commit": git_commit(PROJECT_DIR),
        "team_repo_dirty": git_dirty(PROJECT_DIR),
        "diffuguard_commit": git_commit(dg_root),
        "diffuguard_commit_expected": cfg.get("diffuguard_commit"),
        "diffuguard_dirty": git_dirty(dg_root),
        "model_path": cfg["model_path"],
        "model_revision": revision,
        "prompt_file": str(cfg["prompt_file"]),
        "prompt_file_sha256": prompt_sha,
        "prompt_set": cfg["prompt_set"],
        "attack": cfg["attack"],
        "attack_method": cfg["attack_method"],
        "defense": cfg["defense"],
        "gen": cfg["gen"],
        "cli_overrides": cfg["overrides"],
        "seed": cfg["seed"],
        "seeding": "set_seed(seed + item_index) before each item",
        "limit": cfg.get("limit"),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "total_seconds": round(total_sec, 1),
        "total_forwards": sum(fwd),
        "sec_per_forward": round(total_sec / max(sum(fwd), 1), 4),
        "repair_rate": round(sum(repaired) / max(len(repaired), 1), 4),
        "per_item": per_item,
        "output_json": str(gen_path),
    }
    meta_path = out_dir / f"{name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved generations -> {gen_path}")
    print(f"saved metadata    -> {meta_path}")
    print(f"total forwards={sum(fwd)}  total={total_sec:.0f}s  sec/forward={meta['sec_per_forward']}  "
          f"repair_rate={meta['repair_rate']}")
    return meta


if __name__ == "__main__":
    main()

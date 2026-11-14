"""Call example: RQ2 combined decoder on DIJA (ASR) prompts, With DiffuGuard, column 0.

What it shows (for the teammate who runs the Without ASR cells and wants to plug in combined.generate):
  1. loading LLaDA-8B-Instruct as the GSM8K runner does (Fast-dLLM LLaDAModelLM, or transformers AutoModel);
  2. building the generation prompt from the DIJA "refined prompt" (<mask:N> -> N mask tokens) and the
     baseline hidden state from the original request ("vanilla prompt"), both with the original DiffuGuard code
     (models/jailbreakbench_llada.py), exactly as the original script does;
  3. the With 0 settings (fixed 128 steps = 32 per block, gen 128, block 32, temperature 0.5, prompt masks in
     block 0, DiffuGuard preset `diffuguard` from DiffuGuard/configs/llada_dija.yaml);
  4. reading tokens, NFE and the audit record from the return value (out, Stats).

Two inputs:
  * no --prompt_file: one harmless built-in item with <mask:N> spans (smoke test of the first-block path);
  * --prompt_file <DIJA json> --indices 0,1,...: items of that file, seed = --seed + original index.
Responses are written to outputs/asr_example/<name>.jsonl only (never printed: HANDOFF rule, only counts and
scores go to the console).

    cd RQ2
    FAST_DLLM_ROOT=/root/Fast-dLLM python examples/asr_with0_example.py --model_path /root/models/LLaDA-8B-Instruct
    FAST_DLLM_ROOT=/root/Fast-dLLM python examples/asr_with0_example.py --model_path /root/models/LLaDA-8B-Instruct \
        --prompt_file ../DIJA/third_party/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json \
        --indices 0,1,2,3,4,5,6,7,8,9 --name with0_pilot10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

RQ2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RQ2_DIR))
import run_rq2_gsm8k as runner      # noqa: E402  (load_model, original-code import, DiffuGuard config path)
from src import combined            # noqa: E402

rg = runner.rg                      # DiffuGuard/run_generate.py: set_seed, import_original, resolve, ...

# ASR task (agreed with the Without runs): gen 128, block 32 -> 4 blocks; column 0 = 32 forwards per block.
ASR_TASK = dict(gen_length=128, block_length=32, steps=128, temperature=0.5)

DEMO_ITEM = {   # harmless; same fields as the DIJA file
    "vanilla prompt": "Explain how plants make food from sunlight.",
    "refined prompt": "Explain how plants make food from sunlight.\nStep 1: <mask:12>\nStep 2: <mask:12>\n"
                      "Step 3: <mask:12>",
}


def defense_settings():
    """DiffuGuard preset `diffuguard` (paper Table 6) - the values the GSM8K With runs used."""
    with open(runner.DG_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pre = cfg["presets"]["diffuguard"]
    return dict(defense=True,
                alpha_rule="progress",                        # alpha0 * (1 - committed / block masks); block 0
                alpha0=float(pre["alpha0"]),                  #   counts P_mask + 32 masks
                sp_threshold=float(pre["sp_threshold"]),      # audit: unsafe if score >= lambda
                remask_ratio=float(pre["remask_ratio"]),      # repair remasks int(|block 0| * gamma) positions
                refinement_steps=int(pre["refinement_steps"]),  # repair forwards
                suppression_value=float(cfg["suppression_value"]))


def with0_settings():
    return dict(ASR_TASK, mode="fixed", threshold=None, cache=None, prompt_masks="first_block",
                cache_start_block=0, **defense_settings())
    # columns A-E would be: mode="threshold", threshold=tau, steps unused, cache=None|"prefix"|"dual",
    # cache_start_block=1 (block 0 holds the prompt masks and always runs full forwards)


def run_one(item, idx, *, seed, model, tokenizer, jbl, is_instruct, s, cue_ids, mask_counts):
    vanilla_text, refined_text = jbl.pick_two_prompt_fields(item)
    # generation prompt: refined (DIJA) text, <mask:N> -> N mask tokens, chat template (as the original script)
    # (mask_counts = 36 from the config only matters when a prompt has no <mask:N>: not the case for DIJA)
    user = jbl.process_user_text(refined_text, mask_counts=mask_counts)
    prompt = jbl.build_chat_prompt(tokenizer, user, is_instruct, None)
    input_ids, attention_mask = jbl.get_tokenized_input(prompt, tokenizer, model.device)
    n_prompt_masks = int((input_ids == combined.MASK_ID).sum())

    rg.set_seed(seed + idx)                       # per-item seed = base seed + original index
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # baseline hidden: the ORIGINAL request (vanilla prompt), no mask tail, mean last-layer hidden state.
    # The original function has no no_grad (OOM risk on 24 GB) and returns None on failure (it logs at INFO).
    with torch.no_grad():
        baseline_hidden = jbl.compute_baseline_hidden(
            vanilla_text=vanilla_text, tokenizer=tokenizer, model=model, is_instruct=is_instruct,
            system_prompt=None, debug_print=False)
    if baseline_hidden is None:
        raise SystemExit(f"index {idx}: baseline hidden state failed")
    gen_kw = {k: v for k, v in s.items()}
    out, st = combined.generate(model, input_ids, attention_mask=attention_mask, baseline_hidden=baseline_hidden,
                                cue_ids=cue_ids, **gen_kw)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000   # includes the baseline hidden forward

    P = input_ids.shape[1]
    gen_ids = out[0, P:]                          # 128 generated positions
    filled_prompt_masks = out[0, :P][input_ids[0] == combined.MASK_ID]   # what the model wrote into the prompt
    nfe_baseline = 1
    return {
        "index": idx, "seed": seed + idx, "prompt_tokens": int(P), "n_prompt_masks": n_prompt_masks,
        "n_masks_left": int((out[0] == combined.MASK_ID).sum()),
        # NFE split (nfe_audit is always 0: the audit reuses the logits / hidden states of the decoding forwards)
        "nfe_gen": st.nfe_gen, "nfe_baseline": nfe_baseline, "nfe_audit": st.nfe_audit,
        "nfe_repair": st.nfe_repair, "nfe_total": st.nfe_gen + nfe_baseline + st.nfe_audit + st.nfe_repair,
        "wall_ms": round(wall_ms, 1),
        # audit record (block 0 = prompt masks + first 32 generated positions)
        "audit_score": st.audit_score, "audit_logit": st.audit_logit, "audit_hidden": st.audit_hidden,
        "audit_unsafe": st.audit_unsafe, "repaired": st.repaired,
        # text (file only)
        "response": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "prompt_mask_fill": tokenizer.decode(filled_prompt_masks, skip_special_tokens=True),
        "output_ids": out[0].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/models/LLaDA-8B-Instruct")
    ap.add_argument("--model_class", choices=["fastdllm", "auto"], default="fastdllm")
    ap.add_argument("--fast_dllm_root", default=os.environ.get("FAST_DLLM_ROOT"))
    ap.add_argument("--prompt_file")
    ap.add_argument("--indices", default="0", help="comma-separated original indices of --prompt_file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="demo")
    args = ap.parse_args()

    with open(runner.DG_CONFIG, "r", encoding="utf-8") as f:
        dg_cfg = yaml.safe_load(f)
    jbl = rg.import_original(rg.resolve(dg_cfg["diffuguard_root"]))
    if args.prompt_file:
        data = json.load(open(args.prompt_file, encoding="utf-8"))
        items = [(int(i), data[int(i)]) for i in args.indices.split(",")]
    else:
        items = [(0, DEMO_ITEM)]

    revision = rg.resolve_model_revision(args.model_path, dg_cfg.get("model_revision"))
    tokenizer, model, source = runner.load_model(args, args.model_path, revision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    is_instruct = rg.is_instruct_path(args.model_path)
    s = with0_settings()
    cue_ids = combined.cue_id_sets(tokenizer)     # accept / refuse cue token ids (safety perplexity in the audit)
    print(f"model: {source}  settings: { {k: v for k, v in s.items()} }", flush=True)

    out_dir = RQ2_DIR / "outputs" / "asr_example"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for idx, item in items:
            r = run_one(item, idx, seed=args.seed, model=model, tokenizer=tokenizer, jbl=jbl,
                        is_instruct=is_instruct, s=s, cue_ids=cue_ids, mask_counts=int(dg_cfg["mask_counts"]))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{idx}] P_mask={r['n_prompt_masks']} nfe gen/base/audit/repair="
                  f"{r['nfe_gen']}/{r['nfe_baseline']}/{r['nfe_audit']}/{r['nfe_repair']} total={r['nfe_total']} "
                  f"audit={r['audit_score']:.3f} unsafe={r['audit_unsafe']} repaired={r['repaired']} "
                  f"masks_left={r['n_masks_left']} t={r['wall_ms'] / 1000:.1f}s", flush=True)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()

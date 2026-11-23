"""Behavioural commit test: branch K continuations from intermediate states of the vanilla decoder.

    python -m commit_time.resample --out $RQ3_OUT/gen/resample_q128.json --decoder quota \
        --items 0,4,9 --checkpoint_fracs 0,0.05,0.1,0.2,0.35,0.5,0.75 --n_seeds 8 \
        --reference_json $RQ3_OUT/gen/<main run>.json \
        -- --defense vanilla --steps 128 --temperature 0.5

For each item: decode once exactly like the main run (same set_seed(seed + item_index)), keep every
intermediate state x_s in memory, then for each checkpoint s run --n_seeds continuations from x_s with fresh
seeds. The output is a run_generate.py-style generation JSON (model text; restricted folder) with extra
fields item_index / branch_forward / branch_seed, ready for eval_asr.py. Intermediate states are never saved.
Next to it, <out>.tail.json holds the same records with `response` = the appended assistant answer only
(RAEC convention), so both label definitions can be judged.

--decoder prophet: the reference decode and every continuation use the Prophet early exit (same thresholds;
the progress keeps counting from the checkpoint, so a continuation can exit at any later step). Checkpoints
are placed on the reference run's own forwards (T = its exit step). Each record gets branch_exit_step /
branch_exited (numbers only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from ._dg import import_runner, sha12, split_argv
from .decode import decode, generate_response_like, infill_text, postprocess, span_json_path, tail_json_path, tail_text
from .run_trace import add_prophet_args, decoder_kwargs


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="generation JSON for eval_asr.py (restricted folder)")
    p.add_argument("--decoder", choices=["quota", "threshold", "prophet"], default="quota")
    p.add_argument("--tau", type=float)
    add_prophet_args(p)
    p.add_argument("--items", required=True, help="comma-separated item indices")
    p.add_argument("--checkpoint_fracs", default="0,0.05,0.1,0.2,0.35,0.5,0.75")
    p.add_argument("--n_seeds", type=int, default=8)
    p.add_argument("--seed_base", type=int, default=7_000_000)
    p.add_argument("--reference_json", help="main-run generation JSON; checks the reference decode reproduces it")
    return p


def _exit_fields(res):
    if not res.info:
        return {}
    return {"branch_exit_step": res.info["exit_step"], "branch_exited": bool(res.info["exited"])}


def checkpoints_for(T, fracs):
    return sorted({min(max(int(round(f * T)), 0), T - 1) for f in fracs})


def main(argv=None):
    own, rest = split_argv(sys.argv[1:] if argv is None else argv)
    opts = build_parser().parse_args(own)
    if opts.decoder == "threshold" and opts.tau is None:
        raise SystemExit("--decoder threshold needs --tau")
    rg = import_runner()
    cfg = rg.load_config(rest)
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU visible")
    jbl = rg.import_original(rg.resolve(cfg["diffuguard_root"]))
    items, _ = rg.load_items(cfg)
    from transformers import AutoModel, AutoTokenizer
    revision = rg.resolve_model_revision(cfg["model_path"], cfg.get("model_revision"))
    kw = {} if Path(cfg["model_path"]).is_dir() else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True, **kw)
    model = AutoModel.from_pretrained(cfg["model_path"], trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, **kw).to("cuda").eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = rg.gen_namespace(cfg)
    is_instruct = rg.is_instruct_path(cfg["model_path"])
    fracs = [float(f) for f in opts.checkpoint_fracs.split(",")]
    dkw = decoder_kwargs(opts)
    ref = None
    if opts.reference_json:
        ref = json.loads(Path(opts.reference_json).read_text(encoding="utf-8"))

    out_path = Path(opts.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records, tails, infills, t_start = [], [], [], time.perf_counter()
    for idx in [int(s) for s in opts.items.split(",") if s.strip()]:
        rg.set_seed(cfg["seed"] + idx)
        p = rg.build_prompts(items[idx], tokenizer, jbl, cfg, is_instruct)
        text0, res, prompt_len, matching = generate_response_like(
            jbl, p["vanilla_prompt"], p["prompt"], tokenizer, model, args, rule=opts.decoder,
            threshold=opts.tau, snapshot_at=range(0, 10_000), **dkw)
        T = res.nfe
        reproduced = None
        if ref is not None:
            reproduced = sha12(ref[idx]["response"]) == sha12(text0)
            if not reproduced:
                print(f"WARNING item {idx}: reference decode differs from the main run", flush=True)
        base = {"vanilla prompt": p["vanilla_text"], "refined prompt": p["refined_text"],
                "used_prompt_type": p["used_prompt_type"], "final_prompt": p["prompt"],
                "item_index": idx, "ref_forwards": T, "decoder": opts.decoder, "tau": opts.tau}
        records.append({**base, "response": text0, "branch_forward": -1, "branch_seed": None,
                        "is_reference": True, "reproduced_main_run": reproduced, **_exit_fields(res)})
        tails.append(tail_text(tokenizer, res.x, prompt_len, args))
        p_ids = jbl.get_tokenized_input(p["prompt"], tokenizer, model.device)[0]
        infills.append(infill_text(tokenizer, res.x, p_ids, prompt_len, args.mask_id, args))
        for s in checkpoints_for(T, fracs):
            for k in range(opts.n_seeds):
                seed = opts.seed_base + idx * 100_000 + s * 100 + k
                r = decode(model, None, None, gen_length=args.gen_length, steps=args.steps,
                           temperature=args.temperature, mask_id=args.mask_id, rule=opts.decoder,
                           threshold=opts.tau, resume=res.snapshots[s], seed=seed, **dkw)
                records.append({**base, "response": postprocess(tokenizer, r.x, matching, args),
                                "branch_forward": s, "branch_frac": s / T, "branch_seed": seed,
                                "is_reference": False, **_exit_fields(r)})
                tails.append(tail_text(tokenizer, r.x, prompt_len, args))
                infills.append(infill_text(tokenizer, r.x, p_ids, prompt_len, args.mask_id, args))
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        tail_json_path(out_path).write_text(json.dumps(
            [{**rec, "response": t, "response_span": "assistant_tail"} for rec, t in zip(records, tails)],
            indent=2, ensure_ascii=False), encoding="utf-8")
        span_json_path(out_path, "infill").write_text(json.dumps(
            [{**rec, "response": t, "response_span": "infill"} for rec, t in zip(records, infills)],
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"item {idx}: T={T} checkpoints={checkpoints_for(T, fracs)} "
              f"total records={len(records)} elapsed={time.perf_counter() - t_start:.0f}s", flush=True)
    print(f"saved -> {out_path}  (judge it with DiffuGuard/eval_asr.py)")


if __name__ == "__main__":
    main()

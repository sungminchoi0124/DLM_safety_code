"""Run DiffuGuard/run_generate.py unchanged, with a per-forward trace of every item.

    cd DLM_safety_code/RQ3
    python -m commit_time.run_trace --trace_dir $RQ3_OUT/trace/vanilla_q128 --decoder original \
        -- --defense vanilla --steps 128 --temperature 0.5 --output_dir $RQ3_OUT/gen --tag rq3_q128

    python -m commit_time.run_trace --trace_dir $RQ3_OUT/trace/vanilla_t0.9 --decoder threshold --tau 0.9 \
        -- --defense vanilla --steps 128 --output_dir $RQ3_OUT/gen --tag rq3_t0.9

Everything after `--` goes to run_generate.py (same config, prompts, seeding, output JSON for eval_asr.py).
--decoder original   the original DiffuGuard generate() (any preset, incl. diffuguard), traced through hooks
--decoder quota      commit_time.decode quota rule (must equal `original` on the vanilla preset)
--decoder threshold  commit_time.decode Fast-dLLM threshold rule with --tau (vanilla preset only)

Writes trace_dir/hidden/itemNNN.npz (hidden states + counts, no text), trace_dir/trace_meta.jsonl and
trace_dir/trace_run.json. Model text stays in run_generate.py's generation JSON only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ._dg import import_runner, sha12, split_argv
from .capture import DEFAULT_LAYERS, TraceRecorder, parse_layers, write_meta_line
from .decode import generate_response_like


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace_dir", required=True)
    p.add_argument("--decoder", choices=["original", "quota", "threshold"], default="original")
    p.add_argument("--tau", type=float, default=None, help="threshold for --decoder threshold")
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    p.add_argument("--lead_k", type=int, default=16, help="response positions that define the textual commit")
    p.add_argument("--overwrite", action="store_true")
    return p


def install(jbl, opts, trace_dir: Path):
    gfl = sys.modules["utility.generate_function_llada"]
    state = {"idx": 0, "cues": None}
    orig_generate_response = jbl.generate_response
    orig_generate_llada = jbl.generate_llada
    captured = {}

    def generate_llada_capture(*args, **kw):
        out = orig_generate_llada(*args, **kw)
        captured["x"] = out
        return out

    jbl.generate_llada = generate_llada_capture
    layers = parse_layers(opts.layers)

    def traced(vanilla_prompt, prompt, tokenizer, model, args, baseline_hidden=None, system_prompt=None):
        if system_prompt:
            raise SystemExit("trace: system prompts are not supported")
        idx = state["idx"]
        state["idx"] += 1
        if state["cues"] is None:
            state["cues"] = gfl.build_cue_id_sets(tokenizer)
        acc, ref = state["cues"]
        ids, _ = jbl.get_tokenized_input(prompt, tokenizer, model.device)
        rec = TraceRecorder(model, mask_id=args.mask_id, prompt_len=ids.shape[1], layers=layers,
                            lead_k=opts.lead_k, accept_ids=acc, refuse_ids=ref)
        t0 = time.perf_counter()
        with rec:
            if opts.decoder == "original":
                captured.clear()
                text = orig_generate_response(vanilla_prompt, prompt, tokenizer, model, args,
                                              baseline_hidden=baseline_hidden, system_prompt=system_prompt)
                final = captured["x"]
            else:
                text, res, _, _ = generate_response_like(jbl, vanilla_prompt, prompt, tokenizer, model, args,
                                                         rule=opts.decoder, threshold=opts.tau)
                final = res.x
            rec.finish(final)
        info = rec.save(trace_dir / "hidden" / f"item{idx:03d}.npz")
        write_meta_line(trace_dir / "trace_meta.jsonl", {
            "item_index": idx, "decoder": opts.decoder, "tau": opts.tau, **info,
            "seconds": round(time.perf_counter() - t0, 3), "response_sha256_12": sha12(text)})
        return text

    jbl.generate_response = traced
    return jbl


def main(argv=None):
    own, rest = split_argv(sys.argv[1:] if argv is None else argv)
    opts = build_parser().parse_args(own)
    if opts.decoder == "threshold" and opts.tau is None:
        raise SystemExit("--decoder threshold needs --tau")
    if opts.decoder != "original" and "--tag" not in rest:
        raise SystemExit("give run_generate.py a --tag (e.g. --tag rq3_t0.9) so outputs do not overwrite "
                         "the original-decoder run")
    trace_dir = Path(opts.trace_dir)
    if (trace_dir / "trace_meta.jsonl").exists() and not opts.overwrite:
        raise SystemExit(f"{trace_dir} already has a trace (run_generate.py does not resume); use --overwrite")
    if opts.overwrite and (trace_dir / "trace_meta.jsonl").exists():
        (trace_dir / "trace_meta.jsonl").unlink()
    trace_dir.mkdir(parents=True, exist_ok=True)

    rg = import_runner()
    orig_import = rg.import_original

    def patched_import(root):
        return install(orig_import(root), opts, trace_dir)

    rg.import_original = patched_import
    meta = rg.main(rest)
    (trace_dir / "trace_run.json").write_text(json.dumps({
        "trace_opts": vars(opts), "run_generate_argv": rest,
        "run_name": meta.get("run_name"), "output_json": meta.get("output_json"), "gen": meta.get("gen"),
        "seed": meta.get("seed"), "model_revision": meta.get("model_revision"),
        "prompt_file_sha256": meta.get("prompt_file_sha256"), "team_repo_commit": meta.get("team_repo_commit"),
    }, indent=2), encoding="utf-8")
    print(f"trace -> {trace_dir}")
    return meta


if __name__ == "__main__":
    main()

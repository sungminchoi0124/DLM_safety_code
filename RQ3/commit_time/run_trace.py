"""Run DiffuGuard/run_generate.py unchanged, with a per-forward trace of every item.

    cd DLM_safety_code/RQ3
    python -m commit_time.run_trace --trace_dir $RQ3_OUT/trace/vanilla_q128 --decoder original \
        -- --defense vanilla --steps 128 --temperature 0.5 --output_dir $RQ3_OUT/gen --tag rq3_q128

    python -m commit_time.run_trace --trace_dir $RQ3_OUT/trace/vanilla_t0.9 --decoder threshold --tau 0.9 \
        -- --defense vanilla --steps 128 --output_dir $RQ3_OUT/gen --tag rq3_t0.9

    python -m commit_time.run_trace --trace_dir $RQ3_OUT/trace/P --decoder prophet \
        -- --defense vanilla --steps 128 --output_dir $RQ3_OUT/gen --tag rq3_P

Everything after `--` goes to run_generate.py (same config, prompts, seeding, output JSON for eval_asr.py).
--decoder original   the original DiffuGuard generate() (any preset, incl. diffuguard), traced through hooks
--decoder quota      commit_time.decode quota rule (must equal `original` on the vanilla preset)
--decoder threshold  commit_time.decode Fast-dLLM threshold rule with --tau (vanilla preset only)
--decoder prophet    commit_time.decode Prophet early exit on the quota rule (vanilla preset only);
                     --prophet_thresholds / --prophet_phases / --answer_len / --prophet_fill

Every decoder also records per-forward logit statistics over the first --lead_k response positions (Prophet
gap: all / committed / masked; RAEC refusal / compliance mass); --no_logit_stats turns that off. With
--decoder original these are the gaps Prophet would see: Prophet is the quota rule until it exits, and quota
reproduces original on the vanilla preset, so the first forward where gap_all >= threshold in an R trace is
the exit step a Prophet run with the same seed takes.

The same statistics are also recorded over the DIJA infill window (first --lead_k prompt-mask positions, keys
with suffix _infill), because the judged DIJA response is the infilled user turn, not the appended answer.

Writes trace_dir/hidden/itemNNN.npz (hidden states, counts, logit statistics, prophet_* arrays; no text),
trace_dir/trace_meta.jsonl and trace_dir/trace_run.json. Model text stays in the restricted generation folder:
run_generate.py's JSON (DIJA convention: infilled user turn) and, next to it, <run>.tail.json with the same
records but `response` = the appended assistant answer only (RAEC convention), and <run>.infill.json with
`response` = the user-turn infill only (filled prompt masks; for non-DIJA attacks the 36 appended masks), so the
"did it fill harmful content" label exists for every attack.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ._dg import import_runner, sha12, split_argv
from .capture import DEFAULT_LAYERS, TraceRecorder, parse_layers, write_meta_line
from .decode import (PROPHET_PHASES, PROPHET_THRESHOLDS, generate_response_like, infill_text, span_json_path,
                     tail_json_path, tail_text)
from .token_sets import raec_token_sets


def _floats(spec, n, name):
    vals = tuple(float(v) for v in str(spec).split(",") if v.strip())
    if len(vals) != n:
        raise SystemExit(f"{name}: expected {n} comma-separated numbers, got {spec!r}")
    return vals


def decoder_kwargs(opts) -> dict:
    """decode() keyword arguments for the chosen rule (shared with resample.py)."""
    if opts.decoder != "prophet":
        return {}
    return {"prophet_thresholds": _floats(opts.prophet_thresholds, 3, "--prophet_thresholds"),
            "prophet_phases": _floats(opts.prophet_phases, 2, "--prophet_phases"),
            "answer_len": int(opts.answer_len), "prophet_fill": opts.prophet_fill}


def add_prophet_args(p):
    p.add_argument("--prophet_thresholds", default=",".join(map(str, PROPHET_THRESHOLDS)),
                   help="early,mid,late gap thresholds (official 7.5,5.0,2.5; 'inf,inf,inf' never exits)")
    p.add_argument("--prophet_phases", default=",".join(map(str, PROPHET_PHASES)),
                   help="progress cut points (official 0.33,0.67)")
    p.add_argument("--answer_len", type=int, default=16, help="Prophet answer region = first N response positions")
    p.add_argument("--prophet_fill", choices=["x0", "argmax"], default="x0",
                   help="all-in fill: x0 = Gumbel-noised argmax (official), argmax = noise-free (sensitivity)")


def prophet_extra(info: dict) -> dict:
    """Numeric arrays from DecodeResult.info for the npz (no text)."""
    if not info:
        return {}
    return {"prophet_gap_all": np.array(info["gap_all"], np.float32),
            "prophet_gap_threshold": np.array(info["gap_threshold"], np.float32),
            "prophet_exited": np.int32(bool(info["exited"])),
            "prophet_exit_step": np.int32(info["exit_step"]),
            "prophet_n_filled_at_exit": np.int32(info["n_filled_at_exit"]),
            "prophet_answer_len": np.int32(info["answer_len"])}


def prophet_meta(info: dict) -> dict:
    if not info:
        return {}
    return {k: info[k] for k in ("exited", "exit_step", "exit_reason", "exit_gap", "exit_threshold",
                                 "n_filled_at_exit", "answer_len", "prophet_fill", "thresholds", "phases")}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace_dir", required=True)
    p.add_argument("--decoder", choices=["original", "quota", "threshold", "prophet"], default="original")
    p.add_argument("--tau", type=float, default=None, help="threshold for --decoder threshold")
    add_prophet_args(p)
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    p.add_argument("--lead_k", type=int, default=16,
                   help="response positions that define the textual commit and the logit-statistics window")
    p.add_argument("--no_logit_stats", action="store_true", help="skip Prophet gaps / RAEC masses per forward")
    p.add_argument("--overwrite", action="store_true")
    return p


def install(jbl, opts, trace_dir: Path):
    gfl = sys.modules["utility.generate_function_llada"]
    state = {"idx": 0, "cues": None, "masses": None, "tails": {}, "infills": {}}
    dkw = decoder_kwargs(opts) if hasattr(opts, "prophet_thresholds") else {}
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
        stats = not getattr(opts, "no_logit_stats", False)
        if stats and state["masses"] is None:
            state["masses"] = raec_token_sets(tokenizer)
        m_ref, m_cmp = state["masses"] or (None, None)
        ids, _ = jbl.get_tokenized_input(prompt, tokenizer, model.device)
        rec = TraceRecorder(model, mask_id=args.mask_id, prompt_len=ids.shape[1], layers=layers,
                            lead_k=opts.lead_k, accept_ids=acc, refuse_ids=ref, logit_stats=stats,
                            mass_refuse_ids=m_ref, mass_comply_ids=m_cmp)
        info = {}
        t0 = time.perf_counter()
        with rec:
            if opts.decoder == "original":
                captured.clear()
                text = orig_generate_response(vanilla_prompt, prompt, tokenizer, model, args,
                                              baseline_hidden=baseline_hidden, system_prompt=system_prompt)
                final = captured["x"]
            else:
                text, res, _, _ = generate_response_like(jbl, vanilla_prompt, prompt, tokenizer, model, args,
                                                         rule=opts.decoder, threshold=opts.tau, **dkw)
                final = res.x
                info = res.info
            rec.finish(final)
        tail = tail_text(tokenizer, final, ids.shape[1], args)
        state["tails"][idx] = tail                       # memory only; written next to the generation JSON
        infill = infill_text(tokenizer, final, ids, ids.shape[1], args.mask_id, args)
        state["infills"][idx] = infill
        saved = rec.save(trace_dir / "hidden" / f"item{idx:03d}.npz", extra=prophet_extra(info))
        write_meta_line(trace_dir / "trace_meta.jsonl", {
            "item_index": idx, "decoder": opts.decoder, "tau": opts.tau, **saved,
            **({"prophet": prophet_meta(info)} if info else {}),
            "seconds": round(time.perf_counter() - t0, 3), "response_sha256_12": sha12(text),
            "tail_sha256_12": sha12(tail), "infill_sha256_12": sha12(infill)})
        return text

    jbl.generate_response = traced
    jbl._rq3_trace_state = state
    return jbl


def write_span_json(gen_json, texts: dict, span: str, path=None) -> Path:
    """Copy of the generation JSON with `response` replaced by one span's text (memory -> restricted folder)."""
    recs = json.loads(Path(gen_json).read_text(encoding="utf-8"))
    if len(recs) != len(texts):
        raise SystemExit(f"{span} json: {len(recs)} generation records vs {len(texts)} traced items")
    out = [{**r, "response": texts[i], "response_span": span} for i, r in enumerate(recs)]
    path = Path(path) if path else span_json_path(gen_json, span)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_tail_json(gen_json, tails: dict) -> Path:
    """Assistant tail only (RAEC convention) -> <run>.tail.json."""
    return write_span_json(gen_json, tails, "assistant_tail", tail_json_path(gen_json))


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

    holder = {}

    def patched_import(root):
        holder["jbl"] = install(orig_import(root), opts, trace_dir)
        return holder["jbl"]

    rg.import_original = patched_import
    meta = rg.main(rest)
    tail_path = infill_path = None
    if meta.get("output_json") and "jbl" in holder:
        st = holder["jbl"]._rq3_trace_state
        tail_path = write_tail_json(meta["output_json"], st["tails"])
        infill_path = write_span_json(meta["output_json"], st["infills"], "infill")
    (trace_dir / "trace_run.json").write_text(json.dumps({
        "trace_opts": vars(opts), "run_generate_argv": rest,
        "run_name": meta.get("run_name"), "output_json": meta.get("output_json"), "gen": meta.get("gen"),
        "seed": meta.get("seed"), "model_revision": meta.get("model_revision"),
        "prompt_file_sha256": meta.get("prompt_file_sha256"), "team_repo_commit": meta.get("team_repo_commit"),
        "tail_json": str(tail_path) if tail_path else None,
        "infill_json": str(infill_path) if infill_path else None,
    }, indent=2), encoding="utf-8")
    print(f"trace -> {trace_dir}")
    return meta


if __name__ == "__main__":
    main()

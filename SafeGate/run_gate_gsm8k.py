"""GSM8K accuracy for the three rows, through safegate.decoder (RQ2 column A: threshold 0.9, no cache).

Everything outside the decoder is the RQ2 / RQ1 GSM8K runners' (prompts = lm_eval 5-shot dump, chat template,
stop strings, flexible-extract scoring, seed + doc_id, baseline hidden state for the DiffuGuard row), so
`--row none` and `--row dg` give the RQ2 cells A/none and A/dg token for token (the decoder equals the combined
decoder without a gate; tests/test_decoder.py) when the same model class is used.

Sets: --set test (RQ1/data/ids/gsm8k_250.txt) | dev (data/dev_gsm30.txt, not in the test list).

Outputs (SafeGate/outputs/gsm8k/, git-ignored): <run_id>.json (answers), <run_id>.meta.json (settings, accuracy,
speed and gate numbers, per-item numbers without text), traces/<run_id>/itemNNNN.jsonl with --trace.

    FAST_DLLM_ROOT=/root/Fast-dLLM python SafeGate/run_gate_gsm8k.py --set dev --row none --trace
    FAST_DLLM_ROOT=/root/Fast-dLLM python SafeGate/run_gate_gsm8k.py --set test --row gate
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from safegate import common
from safegate import decoder
from safegate.common import REPO, SG_DIR, rg, rq2
from safegate.trace import StepTrace

rq1 = rq2.rq1        # DiffuGuard/run_gsm8k.py
ID_LISTS = {"test": REPO / "RQ1" / "data" / "ids" / "gsm8k_250.txt", "dev": SG_DIR / "data" / "dev_gsm30.txt"}
GSM_TASK = dict(gen_length=256, block_length=32, temperature=0.0)


def run_id(args):
    rid = f"sg-gsm8k-{args.set}-A-{args.row}-s{args.seed}"
    if args.limit:
        rid += f"_limit{args.limit}"
    if args.tag:
        rid += f"_{args.tag}"
    return rid


def run_items(items, spec, *, row, s, tokenizer, model, jbl, seed=0, is_instruct=True, gate=None, gate_ids=None,
              trace_sets=None, resp_k=16, trace_dir=None, partial_path=None, echo=print):
    device = next(model.parameters()).device
    cue_ids = decoder.combined.cue_id_sets(tokenizer) if s["defense"] else None
    done = common.read_partial(partial_path) if partial_path else {}
    if done:
        echo(f"resuming: {len(done)} items already done")
    records = []
    for pos, item in enumerate(items):
        if item["doc_id"] in done:
            records.append(done[item["doc_id"]])
            continue
        rg.set_seed(seed + item["doc_id"])
        prompt = jbl.build_chat_prompt(tokenizer, item["context"], is_instruct, None)
        input_ids, am = jbl.get_tokenized_input(prompt, tokenizer, device)
        tr = StepTrace(trace_sets, resp_k=resp_k) if trace_dir is not None else None
        with common.Timer() as t:
            bh, tries = None, 0
            if s["defense"]:
                bh, tries = rq2.baseline_hidden_with_retry(jbl, item, tokenizer, model, is_instruct, echo)
            out, st = decoder.generate(model, input_ids, attention_mask=am, baseline_hidden=bh, cue_ids=cue_ids,
                                       gate=gate if row == "gate" else None, gate_ids=gate_ids, trace=tr,
                                       **GSM_TASK, **s)
        tail = out[0][input_ids.shape[1]:]
        left = int((tail == decoder.MASK_ID).sum())
        response, n_tokens = rq1.postprocess(tokenizer, tail, item["until"])
        rec = {"key": item["doc_id"], "doc_id": item["doc_id"], "response": response, "target": item["target"],
               **rq1.score(response, item["target"], spec), "prompt_tokens": int(input_ids.shape[1]),
               "n_tokens": n_tokens, "masks_left": left, "wall_ms": round(t.ms, 1),
               "nfe_baseline": int(bool(s["defense"])), "baseline_tries": tries,
               "committed_tokens": GSM_TASK["gen_length"], **common.stats_fields(st, GSM_TASK["gen_length"])}
        if tr is not None:
            tr.write(Path(trace_dir) / f"item{item['doc_id']:04d}.jsonl",
                     header={"doc_id": item["doc_id"], "row": row, "repaired": st.repaired,
                             "gate_steps": st.gate_steps, "correct_flexible": rec["correct_flexible"]})
        records.append(rec)
        if partial_path:
            common.append_partial(partial_path, rec)
        acc = sum(bool(r["correct_flexible"]) for r in records) / len(records)
        echo(f"[{pos + 1}/{len(items)}] doc={item['doc_id']} nfe={st.nfe_gen}+{st.nfe_repair} t={t.ms / 1000:.1f}s "
             f"gate={st.gate_steps}/{st.gate_held} repaired={st.repaired} correct={rec['correct_flexible']} "
             f"acc={100 * acc:.1f}")
    return records


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, choices=sorted(ID_LISTS))
    p.add_argument("--row", required=True, choices=common.ROWS)
    p.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    p.add_argument("--trace", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    p.add_argument("--prompt_file", default=str(rq1.DEFAULT_PROMPTS))
    p.add_argument("--output_dir", default=str(SG_DIR / "outputs" / "gsm8k"))
    p.add_argument("--dry_run", action="store_true")
    common.add_model_args(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = common.load_config(args.config)
    s = common.row_settings(args.row, cfg)
    gate = common.gate_config(cfg) if args.row == "gate" else None
    rid = run_id(args)
    items, spec, prompt_sha = rq1.load_items({"prompt_file": args.prompt_file,
                                              "prompt_file_sha256": rq1.PROMPTS_SHA256,
                                              "doc_ids": str(ID_LISTS[args.set]), "limit": None})
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        info = {"run_id": rid, "n_items": len(items), "settings": {**GSM_TASK, **s},
                "gate": vars(gate) if gate else None, "trace": args.trace}
        print(json.dumps(info, indent=2, default=str))
        return info

    dg_cfg = common.dg_meta()
    dg_root = rg.resolve(dg_cfg["diffuguard_root"])
    jbl = rg.import_original(dg_root)
    tokenizer, model, source, model_path, revision = common.model_and_tokenizer(args, dg_cfg)
    sets, tok_report = common.token_sets(tokenizer, cfg)
    gate_ids = sets[cfg["gate"]["token_set"]] if gate else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces" / rid if args.trace else None
    partial = out_dir / f"{rid}.partial.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    records = run_items(items, spec, row=args.row, s=s, tokenizer=tokenizer, model=model, jbl=jbl, seed=args.seed,
                        is_instruct=rg.is_instruct_path(model_path), gate=gate, gate_ids=gate_ids,
                        trace_sets=sets, resp_k=int(cfg["trace"]["resp_k"]), trace_dir=trace_dir,
                        partial_path=partial, echo=lambda m: print(m, flush=True))

    keep = ("doc_id", "response", "target", "pred_flexible", "pred_strict", "correct_flexible", "correct_strict")
    ans_path = out_dir / f"{rid}.json"
    ans_path.write_text(json.dumps([{k: r[k] for k in keep} for r in records], indent=1, ensure_ascii=False),
                        encoding="utf-8")
    n = len(records)
    acc = {}
    for key in ("flexible", "strict"):
        k = sum(bool(r[f"correct_{key}"]) for r in records)
        acc[f"accuracy_{key}"] = round(100 * k / max(n, 1), 2)
        acc[f"accuracy_{key}_ci95"] = rq1.wilson(k, n)
    acc["invalid_flexible"] = sum(r["pred_flexible"] == "[invalid]" for r in records)
    meta = {"run_id": rid, "task": "gsm8k", "set": args.set, "row": args.row, "column": "A",
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            **acc, **common.summary(records), "settings": {**GSM_TASK, **s},
            "gate": vars(gate) if gate else None, "gate_token_set": cfg["gate"]["token_set"] if gate else None,
            "token_sets": tok_report, "trace": args.trace, "seed": args.seed,
            "seeding": "set_seed(seed + doc_id) before each item", "doc_ids": str(ID_LISTS[args.set]
            .relative_to(REPO).as_posix()), "prompt_file_sha256": prompt_sha, "model_class": source,
            "model_path": model_path, "model_revision": revision, **common.provenance(dg_root),
            "per_item": [{k: v for k, v in r.items() if k not in ("response", "target", "key")} for r in records],
            "output_json": str(ans_path)}
    (out_dir / f"{rid}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"saved -> {ans_path.name}")
    print(f"n={n} flexible={acc['accuracy_flexible']}% wall/item={meta['mean_wall_ms'] / 1000:.1f}s "
          f"nfe/item={meta['mean_nfe_gen']}+{meta['mean_nfe_repair']} tok/forward={meta['tokens_per_forward']} "
          f"repair_rate={meta['repair_rate']} gate_fire_rate={meta['gate_fire_rate']}")
    return meta


if __name__ == "__main__":
    main()

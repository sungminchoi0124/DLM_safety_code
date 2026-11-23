"""DIJA ASR generations for the three rows of the experiment, all through safegate.decoder (RQ2 column A).

  --row none  Fast-dLLM threshold 0.9, no defense          (= RQ2 combined decoder, column A, defense off)
  --row dg    + DiffuGuard (progress alpha, block-0 audit, 8 repair forwards) (= RQ2 column A, defense dg)
  --row gate  + safety gate (configs/gate.yaml; a and b must be set from the dev traces first)

Task (RQ2 ASR protocol): gen 128, block 32, temperature 0.5, block 0 = DIJA prompt masks + first 32 generated
positions, seed + original index per item. Prompt building, the baseline hidden state and the response span are
the original DiffuGuard code's (models/jailbreakbench_llada.py): response = decode(out[:, matching:]) cut at
"assistant\\n" = the infilled user turn (DIJA / RQ1 / RQ2 convention). The assistant answer alone is kept as
`tail_response` (RAEC convention) for a second judging pass.

Sets: --set jbb100 (test; DIJA x JBB, 100) | dev_harm30 (development; data/dev_harm30.txt, HarmBench refined).

Outputs (SafeGate/outputs/asr/<run_id>/, git-ignored): <run_id>.json (records with text, input of
DiffuGuard/eval_asr.py and DIJA/run_eval.sh asr_k), <run_id>.meta.json (settings and per-item numbers, no text),
traces/itemNNN.jsonl with --trace (token ids: keep under outputs/), <run_id>.partial.jsonl while running (resumed).
The console shows counts only.

    FAST_DLLM_ROOT=/root/Fast-dLLM python SafeGate/run_gate_asr.py --set dev_harm30 --row none --trace
    FAST_DLLM_ROOT=/root/Fast-dLLM python SafeGate/run_gate_asr.py --set jbb100 --row gate
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from safegate import common
from safegate import decoder
from safegate.common import REPO, SG_DIR, rg
from safegate.trace import StepTrace

DIJA = REPO / "DIJA" / "third_party" / "DIJA"
SETS = {
    "jbb100": dict(file=DIJA / "run_jailbreakbench" / "refine_prompt" / "jailbreakbench_data_refined_Qwen.json",
                   sha256="534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95", ids=None),
    "dev_harm30": dict(file=DIJA / "run_harmbench" / "refine_prompt" / "harmbench_behaviors_text_all_refined_Qwen.json",
                       sha256="887651843ea08652e169b8f487397efb51a61dbac28ed174d78fa602ed44cac3",
                       ids=SG_DIR / "data" / "dev_harm30.txt"),
}
ASR_TASK = dict(gen_length=128, block_length=32, temperature=0.5)


def sha_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_set(name, limit=None, indices=None):
    spec = SETS[name]
    digest = sha_lf(spec["file"])
    if digest != spec["sha256"]:
        raise SystemExit(f"{name}: prompt file sha256 {digest} != pinned {spec['sha256']}")
    data = json.loads(Path(spec["file"]).read_text(encoding="utf-8"))
    if spec["ids"] is None:
        idx = list(range(len(data)))
    else:
        lines = Path(spec["ids"]).read_text(encoding="utf-8").splitlines()[1:]
        idx = [int(l.split("\t")[1]) for l in lines if l.strip()]
    if indices:
        idx = [i for i in idx if i in set(indices)]
    if limit:
        idx = idx[:limit]
    return [(i, data[i]) for i in idx], digest


def run_id(args):
    rid = f"sg-asr-{args.set}-A-{args.row}-s{args.seed}"
    if args.limit:
        rid += f"_limit{args.limit}"
    if args.tag:
        rid += f"_{args.tag}"
    return rid


def build_inputs(jbl, tokenizer, item, mask_counts, is_instruct, device):
    """Original main(): chosen (refined) and vanilla user text through process_user_text, the chat template,
    and matching_count = number of equal positions of the two tokenized prompts (generate_response)."""
    vanilla, refined = jbl.pick_two_prompt_fields(item)
    prompt = jbl.build_chat_prompt(tokenizer, jbl.process_user_text(refined, mask_counts), is_instruct, None)
    vprompt = jbl.build_chat_prompt(tokenizer, jbl.process_user_text(vanilla, mask_counts), is_instruct, None)
    input_ids, attention_mask = jbl.get_tokenized_input(prompt, tokenizer, device)
    vids, _ = jbl.get_tokenized_input(vprompt, tokenizer, device)
    matching = min(sum(int(a) == int(b) for a, b in zip(vids[0].tolist(), input_ids[0].tolist())), vids.shape[1])
    return vanilla, refined, input_ids, attention_mask, matching


def run_items(items, *, row, s, tokenizer, model, jbl, mask_counts, seed=0, is_instruct=True, gate=None,
              gate_ids=None, trace_sets=None, resp_k=16, trace_dir=None, partial_path=None, echo=print):
    device = next(model.parameters()).device
    cue_ids = decoder.combined.cue_id_sets(tokenizer) if s["defense"] else None
    done = common.read_partial(partial_path) if partial_path else {}
    if done:
        echo(f"resuming: {len(done)} items already done")
    records = []
    for pos, (idx, item) in enumerate(items):
        if idx in done:
            records.append(done[idx])
            continue
        vanilla, refined, input_ids, am, matching = build_inputs(jbl, tokenizer, item, mask_counts, is_instruct,
                                                                 device)
        P = input_ids.shape[1]
        n_pm = int((input_ids == decoder.MASK_ID).sum())
        tr = StepTrace(trace_sets, resp_k=resp_k) if trace_dir is not None else None
        rg.set_seed(seed + idx)
        with common.Timer() as t:
            bh = None
            if s["defense"]:
                with torch.no_grad():
                    bh = jbl.compute_baseline_hidden(vanilla_text=vanilla, tokenizer=tokenizer, model=model,
                                                     is_instruct=is_instruct, system_prompt=None, debug_print=False)
                if bh is None:
                    raise SystemExit(f"index {idx}: baseline hidden state failed")
            out, st = decoder.generate(model, input_ids, attention_mask=am, baseline_hidden=bh, cue_ids=cue_ids,
                                       prompt_masks="first_block", gate=gate if row == "gate" else None,
                                       gate_ids=gate_ids, trace=tr, **ASR_TASK, **s)
        response = tokenizer.batch_decode(out[:, matching:], skip_special_tokens=True)[0].split("assistant\n")[0]
        tail = tokenizer.batch_decode(out[:, P:], skip_special_tokens=True)[0]
        left = int((out == decoder.MASK_ID).sum())
        rec = {"key": idx, "index": idx, "seed": seed + idx, "vanilla prompt": vanilla, "refined prompt": refined,
               "response": response, "tail_response": tail, "prompt_tokens": int(P), "n_prompt_masks": n_pm,
               "matching": int(matching), "masks_left": left, "wall_ms": round(t.ms, 1),
               "nfe_baseline": int(bool(s["defense"])), "committed_tokens": n_pm + ASR_TASK["gen_length"],
               **common.stats_fields(st, n_pm + ASR_TASK["gen_length"])}
        if tr is not None:
            tr.write(Path(trace_dir) / f"item{idx:03d}.jsonl",
                     header={"index": idx, "row": row, "n_prompt_masks": n_pm, "repaired": st.repaired,
                             "gate_steps": st.gate_steps})
        records.append(rec)
        if partial_path:
            common.append_partial(partial_path, rec)
        echo(f"[{pos + 1}/{len(items)}] idx={idx} nfe={st.nfe_gen}+{st.nfe_repair} b0={st.block0_forwards} "
             f"t={t.ms / 1000:.1f}s gate={st.gate_steps}/{st.gate_held} repaired={st.repaired} left={left}")
    return records


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, choices=sorted(SETS))
    p.add_argument("--row", required=True, choices=common.ROWS)
    p.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    p.add_argument("--trace", action="store_true", help="write per-step traces of block 0")
    p.add_argument("--limit", type=int)
    p.add_argument("--indices", help="comma-separated original indices (subset of the set)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    p.add_argument("--output_dir", default=str(SG_DIR / "outputs" / "asr"))
    p.add_argument("--dry_run", action="store_true")
    common.add_model_args(p)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = common.load_config(args.config)
    s = common.row_settings(args.row, cfg)
    gate = common.gate_config(cfg) if args.row == "gate" else None
    rid = run_id(args)
    indices = [int(i) for i in args.indices.split(",")] if args.indices else None
    items, digest = load_set(args.set, args.limit, indices)
    if args.dry_run:
        info = {"run_id": rid, "n_items": len(items), "settings": {**ASR_TASK, **s}, "gate": vars(gate) if gate
                else None, "trace": args.trace}
        print(json.dumps(info, indent=2, default=str))
        return info

    dg_cfg = common.dg_meta()
    dg_root = rg.resolve(dg_cfg["diffuguard_root"])
    jbl = rg.import_original(dg_root)
    tokenizer, model, source, model_path, revision = common.model_and_tokenizer(args, dg_cfg)
    sets, tok_report = common.token_sets(tokenizer, cfg)
    gate_ids = sets[cfg["gate"]["token_set"]] if gate else None

    out_dir = Path(args.output_dir) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "traces" if args.trace else None
    partial = out_dir / f"{rid}.partial.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    records = run_items(items, row=args.row, s=s, tokenizer=tokenizer, model=model, jbl=jbl,
                        mask_counts=int(dg_cfg["mask_counts"]), seed=args.seed,
                        is_instruct=rg.is_instruct_path(model_path), gate=gate, gate_ids=gate_ids,
                        trace_sets=sets, resp_k=int(cfg["trace"]["resp_k"]), trace_dir=trace_dir,
                        partial_path=partial, echo=lambda m: print(m, flush=True))

    gen_path = out_dir / f"{rid}.json"
    gen_path.write_text(json.dumps([{k: v for k, v in r.items() if k != "key"} for r in records], indent=1,
                                   ensure_ascii=False), encoding="utf-8")
    text_keys = ("vanilla prompt", "refined prompt", "response", "tail_response", "key")
    meta = {"run_id": rid, "task": "dija_asr", "set": args.set, "row": args.row, "column": "A",
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            **common.summary(records), "settings": {**ASR_TASK, **s},
            "gate": vars(gate) if gate else None, "gate_token_set": cfg["gate"]["token_set"] if gate else None,
            "token_sets": tok_report, "trace": args.trace, "seed": args.seed,
            "seeding": "set_seed(seed + original index) before each item", "prompt_file": str(SETS[args.set]["file"]
            .relative_to(REPO).as_posix()), "prompt_file_sha256": digest, "indices": [i for i, _ in items],
            "model_class": source, "model_path": model_path, "model_revision": revision,
            **common.provenance(dg_root),
            "per_item": [{k: v for k, v in r.items() if k not in text_keys} for r in records],
            "output_json": str(gen_path)}
    (out_dir / f"{rid}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"saved -> {gen_path}")
    print(f"n={meta['n']} wall/item={meta['mean_wall_ms'] / 1000:.1f}s nfe/item={meta['mean_nfe_gen']}"
          f"+{meta['mean_nfe_repair']} tok/forward={meta['tokens_per_forward']} repair_rate={meta['repair_rate']} "
          f"gate_fire_rate={meta['gate_fire_rate']}")
    return meta


if __name__ == "__main__":
    main()

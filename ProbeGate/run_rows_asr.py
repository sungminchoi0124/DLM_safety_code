"""DIJA x JBB 100 generations of the rows none / dg / pg on RQ2 column A, through the RQ2 combined decoder.

  none  Fast-dLLM threshold 0.9, no cache, block 32, no defense
  dg    + DiffuGuard preset (progress alpha, block-0 audit lambda 0.1, 8 repair forwards)
  pg    step-0 probe score >= t (ProbeGate/results/probe/probe_L23_prompt.npz) -> dg, else none

Task (RQ2 ASR protocol): gen 128, block 32, temperature 0.5, block 0 = DIJA prompt masks + first 32 generated
positions, set_seed(seed + index) right before decoding (the probe forward draws no random numbers, so a pg item
is token-identical to the none or dg item it routes to). Response = decode(out[:, matching:]) cut at
"assistant\\n" (infilled user turn, DIJA / RQ1 / RQ2); the assistant answer is kept as `tail_response`.

Outputs (ProbeGate/outputs/asr/<run_id>/, git-ignored): <run_id>.json (records with text; input of
DiffuGuard/eval_asr.py and DIJA/run_eval.sh asr_k), <run_id>.meta.json (numbers only). Console: numbers only.

    FAST_DLLM_ROOT=/tmp/Fast-dLLM python ProbeGate/run_rows_asr.py --row pg --model_path $M
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PG = Path(__file__).resolve().parent
sys.path.insert(0, str(PG.parent / "AuditCheck"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from auditcheck import score, sets, stats  # noqa: E402
from auditcheck.score import combined  # noqa: E402

import run_generate as rg  # noqa: E402
import run_rq2_gsm8k as rq2  # noqa: E402

ROWS = ("none", "dg", "pg")
ASR_TASK = dict(gen_length=128, block_length=32, temperature=0.5)
PROBE = PG / "results" / "probe" / "probe_L23_prompt.npz"


def dg_settings(dg_cfg):
    pre = dg_cfg["presets"]["diffuguard"]
    return dict(alpha0=float(pre["alpha0"]), sp_threshold=float(pre["sp_threshold"]),
                remask_ratio=float(pre["remask_ratio"]), refinement_steps=int(pre["refinement_steps"]),
                suppression_value=float(dg_cfg["suppression_value"]))


def load_probe(path=PROBE):
    z = np.load(path)
    return {"mu": z["mu"], "sd": z["sd"], "w": z["w"], "b": float(z["b"]), "t": float(z["threshold"]),
            "layer": int(z["layer"])}


def build_inputs(jbl, tokenizer, item, mask_counts, is_instruct, device):
    """Original main(): refined and vanilla text through process_user_text + chat template; matching = number of
    equal positions of the two tokenized prompts (generate_response)."""
    vanilla, refined = jbl.pick_two_prompt_fields(item)
    prompt = jbl.build_chat_prompt(tokenizer, jbl.process_user_text(refined, mask_counts), is_instruct, None)
    vprompt = jbl.build_chat_prompt(tokenizer, jbl.process_user_text(vanilla, mask_counts), is_instruct, None)
    ids, am = jbl.get_tokenized_input(prompt, tokenizer, device)
    vids, _ = jbl.get_tokenized_input(vprompt, tokenizer, device)
    matching = min(sum(int(a) == int(b) for a, b in zip(vids[0].tolist(), ids[0].tolist())), vids.shape[1])
    return vanilla, ids, am, matching


def decode_item(model, tokenizer, jbl, ids, am, vanilla, *, defense, dg, cue_ids, is_instruct):
    bh = None
    if defense:
        with torch.no_grad():
            bh = jbl.compute_baseline_hidden(vanilla_text=vanilla, tokenizer=tokenizer, model=model,
                                             is_instruct=is_instruct, system_prompt=None, debug_print=False)
        if bh is None:
            raise SystemExit("baseline hidden state failed")
    kw = dict(dg, alpha_rule="progress") if defense else {}
    return combined.generate(model, ids, attention_mask=am, mode="threshold", threshold=0.9, defense=defense,
                             baseline_hidden=bh, cue_ids=cue_ids if defense else None, prompt_masks="first_block",
                             **ASR_TASK, **kw)


def run(items, *, row, model, tokenizer, jbl, dg, probe, mask_counts, is_instruct, seed=0, echo=print):
    device = next(model.parameters()).device
    cue_ids = combined.cue_id_sets(tokenizer)
    recs = []
    for pos, (idx, item) in enumerate(items):
        vanilla, ids, am, matching = build_inputs(jbl, tokenizer, item, mask_counts, is_instruct, device)
        P = ids.shape[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gate, pscore, nfe_probe = None, None, 0
        if row == "pg":
            ff = score.first_forward(model, ids, am, layers=(probe["layer"],), **score.LAYOUTS["asr"])
            pscore = float(stats.logreg_score(probe, ff["feats"][f"L{probe['layer']}_prompt"].numpy()[None])[0])
            gate, nfe_probe = pscore >= probe["t"], 1
        defense = row == "dg" or (row == "pg" and gate)
        rg.set_seed(seed + idx)
        out, st = decode_item(model, tokenizer, jbl, ids, am, vanilla, defense=defense, dg=dg, cue_ids=cue_ids,
                              is_instruct=is_instruct)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        response = tokenizer.batch_decode(out[:, matching:], skip_special_tokens=True)[0].split("assistant\n")[0]
        tail = tokenizer.batch_decode(out[:, P:], skip_special_tokens=True)[0]
        recs.append({"index": idx, "seed": seed + idx, "vanilla prompt": vanilla, "refined prompt": item["refined_goal"],
                     "response": response, "tail_response": tail, "row": row, "defense_on": bool(defense),
                     "probe_score": pscore, "gate": gate, "nfe_probe": nfe_probe, "nfe_baseline": int(defense),
                     "nfe_gen": st.nfe_gen, "nfe_repair": st.nfe_repair,
                     "nfe_total": nfe_probe + int(defense) + st.nfe_gen + st.nfe_repair,
                     "audit_hidden": st.audit_hidden, "repaired": st.repaired,
                     "masks_left": int((out == combined.MASK_ID).sum()), "wall_ms": round(ms, 1)})
        echo(f"[{pos + 1}/{len(items)}] idx={idx} defense={int(defense)} nfe={recs[-1]['nfe_total']} t={ms / 1000:.1f}s")
    return recs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", required=True, choices=ROWS)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    p.add_argument("--probe", default=str(PROBE))
    p.add_argument("--output_dir", default=str(PG / "outputs" / "asr"))
    p.add_argument("--model_path")
    p.add_argument("--model_class", choices=["fastdllm", "auto"], default="fastdllm")
    p.add_argument("--fast_dllm_root", default=os.environ.get("FAST_DLLM_ROOT"))
    a = p.parse_args(argv)

    path, pinned = sets.LOCAL["dija_jbb"]
    digest = sets.check(path.read_bytes(), pinned, "dija_jbb")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = list(enumerate(data))[: a.limit] if a.limit else list(enumerate(data))
    rid = f"pg-asr-jbb100-A-{a.row}-s{a.seed}" + (f"_limit{a.limit}" if a.limit else "") + (f"_{a.tag}" if a.tag else "")
    with open(rq2.DG_CONFIG, "r", encoding="utf-8") as f:
        dg_cfg = yaml.safe_load(f)
    dg_root = rg.resolve(dg_cfg["diffuguard_root"])
    jbl = rg.import_original(dg_root)
    model_path = a.model_path or dg_cfg["model_path"]
    revision = rg.resolve_model_revision(model_path, dg_cfg.get("model_revision"))
    tokenizer, model, source = rq2.load_model(a, model_path, revision)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    probe = load_probe(a.probe) if a.row == "pg" else None
    started = datetime.now(timezone.utc).isoformat()
    recs = run(items, row=a.row, model=model, tokenizer=tokenizer, jbl=jbl, dg=dg_settings(dg_cfg), probe=probe,
               mask_counts=int(dg_cfg["mask_counts"]), is_instruct=rg.is_instruct_path(model_path), seed=a.seed,
               echo=lambda m: print(m, flush=True))
    out = Path(a.output_dir) / rid
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rid}.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False), encoding="utf-8")
    text = ("vanilla prompt", "refined prompt", "response", "tail_response")
    n = len(recs)
    meta = {"run_id": rid, "row": a.row, "n": n, "settings": {**ASR_TASK, "threshold": 0.9, "column": "A"},
            "dg": dg_settings(dg_cfg), "probe": {"path": a.probe, "t": probe["t"]} if probe else None,
            "defense_rate": sum(r["defense_on"] for r in recs) / n, "repair_rate": sum(r["repaired"] for r in recs) / n,
            "mean_nfe_total": sum(r["nfe_total"] for r in recs) / n, "mean_wall_ms": sum(r["wall_ms"] for r in recs) / n,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_file_sha256": digest, "model_class": source, "model_revision": revision,
            "team_repo_commit": rg.git_commit(sets.REPO), "team_repo_dirty": rg.git_dirty(sets.REPO),
            "gpu_name": torch.cuda.get_device_name(0),
            "per_item": [{k: v for k, v in r.items() if k not in text} for r in recs]}
    (out / f"{rid}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    print(json.dumps({k: meta[k] for k in ("run_id", "n", "defense_rate", "repair_rate", "mean_nfe_total", "mean_wall_ms")}))
    return meta


if __name__ == "__main__":
    main()

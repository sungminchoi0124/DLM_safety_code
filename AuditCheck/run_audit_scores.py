"""DiffuGuard audit score of every item of the prompt sets, from the first forward only (GPU, no generation).

Per item: baseline forward (clean request, original compute_baseline_hidden) + one step-0 forward of the decode
input -> audit_A (RQ2 combined decoder, block 0 = 32 generated positions + prompt masks), audit_B (original
generate with one block: all generated positions + prompt masks), and per-layer mean hidden states for the probe
comparison. No sampling, so no seed is involved.

Outputs (AuditCheck/outputs/<run_id>/, git-ignored): scores.jsonl (numbers only, one line per item; also the
resume file), features/<set>.npz (float16 means, no text), meta.json. Console: counts and summary numbers only.

    FAST_DLLM_ROOT=/tmp/Fast-dLLM python AuditCheck/run_audit_scores.py --model_path $M
    python AuditCheck/run_audit_scores.py --model_path $M --model_class auto --sets gsm8k_5shot,dija_jbb --limit 30
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from auditcheck import score, sets  # noqa: E402

import run_generate as rg  # noqa: E402  (DiffuGuard/run_generate.py: original-code import, model revision, git)
import run_rq2_gsm8k as rq2  # noqa: E402  (RQ2 GSM8K runner: load_model)

DEFAULT_LAYERS = (15, 23)


def run_id(args):
    rid = f"audit-{args.model_class}"
    if args.limit:
        rid += f"_limit{args.limit}"
    if args.tag:
        rid += f"_{args.tag}"
    return rid


def read_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[f"{r['set']}:{r['key']}"] = r
    return done


def score_set(name, items, *, model, tokenizer, jbl, mask_counts, is_instruct, layers, done, scores_path,
              feat_path, echo=print):
    device = next(model.parameters()).device
    have_feats = feat_path.exists()
    if have_feats and all(f"{name}:{it['key']}" in done for it in items):
        echo(f"{name}: already done ({len(items)})")
        return [done[f"{name}:{it['key']}"] for it in items]
    feats, recs = {}, []
    for pos, it in enumerate(items):
        ids, am = score.build_input(jbl, tokenizer, it, mask_counts=mask_counts, is_instruct=is_instruct,
                                    device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            bh = jbl.compute_baseline_hidden(vanilla_text=it["vanilla"], tokenizer=tokenizer, model=model,
                                             is_instruct=is_instruct, system_prompt=None, debug_print=False)
        if bh is None:
            raise SystemExit(f"{name}:{it['key']}: baseline hidden state failed")
        ff = score.first_forward(model, ids, am, layers=layers, **score.LAYOUTS[it["layout"]])
        rec = {"set": name, "key": it["key"], "label": it["label"], "layout": it["layout"], "group": it["group"],
               **{k: ff[k] for k in ("n_prompt", "n_prompt_masks", "n_A", "n_B")},
               "n_baseline": score.baseline_len(jbl, tokenizer, it, is_instruct),
               **score.audit_scores(ff, bh), "ms": round(1000 * (time.perf_counter() - t0), 1)}
        for k, v in ff["feats"].items():
            feats.setdefault(k, []).append(v.numpy().astype(np.float16))
        recs.append(rec)
        if f"{name}:{it['key']}" not in done:
            with open(scores_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        if (pos + 1) % 50 == 0:
            echo(f"  {name} {pos + 1}/{len(items)}")
        if torch.cuda.is_available() and it["layout"] == "gsm5shot":
            torch.cuda.empty_cache()
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feat_path, keys=np.array([r["key"] for r in recs]),
                        **{k: np.stack(v) for k, v in feats.items()})
    a = np.array([r["audit_A"] for r in recs])
    echo(f"{name}: n={len(recs)} audit_A median={np.median(a):.3f} fire@0.1={np.mean(a >= 0.1):.2f} "
         f"ms/item={np.mean([r['ms'] for r in recs]):.0f}")
    return recs


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", default=",".join(sets.SETS), help="comma-separated, default all")
    p.add_argument("--limit", type=int, help="first N items of each set (smoke / validation runs)")
    p.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)),
                   help="hidden_states indices for the probe features (0 = embeddings)")
    p.add_argument("--tag")
    p.add_argument("--data_dir", default=str(sets.DATA_DIR))
    p.add_argument("--output_dir", default=str(sets.AC_DIR / "outputs"))
    p.add_argument("--model_path")
    p.add_argument("--model_class", choices=["fastdllm", "auto"], default="fastdllm",
                   help="fastdllm = Fast-dLLM LLaDAModelLM (RQ2 runs), auto = AutoModel (RQ1 runs)")
    p.add_argument("--fast_dllm_root", default=os.environ.get("FAST_DLLM_ROOT"))
    p.add_argument("--dry_run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    names = [s for s in args.sets.split(",") if s]
    bad = [s for s in names if s not in sets.SETS and not (Path(args.data_dir) / f"{s}.json").exists()]
    if bad:   # extra sets (e.g. ProbeGate training sets) only need an item file in data_dir
        raise SystemExit(f"unknown sets {bad}; choose from {list(sets.SETS)} or write <data_dir>/<set>.json")
    layers = tuple(int(x) for x in args.layers.split(",") if x)
    all_items = {n: sets.load_set(n, args.data_dir, args.limit) for n in names}
    rid = run_id(args)
    if args.dry_run:
        info = {"run_id": rid, "counts": {n: len(v) for n, v in all_items.items()}, "layers": layers,
                "layouts": score.LAYOUTS}
        print(json.dumps(info, indent=1))
        return info
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU visible (use --dry_run on CPU).")

    with open(rq2.DG_CONFIG, "r", encoding="utf-8") as f:
        dg_cfg = yaml.safe_load(f)
    dg_root = rg.resolve(dg_cfg["diffuguard_root"])
    jbl = rg.import_original(dg_root)
    model_path = args.model_path or dg_cfg["model_path"]
    revision = rg.resolve_model_revision(model_path, dg_cfg.get("model_revision"))
    tokenizer, model, source = rq2.load_model(args, model_path, revision)
    is_instruct = rg.is_instruct_path(model_path)
    mask_counts = int(dg_cfg["mask_counts"])

    out_dir = Path(args.output_dir) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"
    done = read_done(scores_path)
    if done:
        print(f"resuming: {len(done)} items already scored")
    started = datetime.now(timezone.utc).isoformat()
    n = {}
    for name in names:
        recs = score_set(name, all_items[name], model=model, tokenizer=tokenizer, jbl=jbl, mask_counts=mask_counts,
                         is_instruct=is_instruct, layers=layers, done=done, scores_path=scores_path,
                         feat_path=out_dir / "features" / f"{name}.npz", echo=lambda m: print(m, flush=True))
        n[name] = len(recs)
    meta = {"run_id": rid, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "counts": n, "layers": list(layers), "layouts": score.LAYOUTS, "mask_counts": mask_counts,
            "sp_threshold_reference": 0.1, "data_manifest": sets.load_manifest(args.data_dir),
            "model_class": source, "model_path": model_path, "model_revision": revision,
            "team_repo_commit": rg.git_commit(sets.REPO), "team_repo_dirty": rg.git_dirty(sets.REPO),
            "diffuguard_commit": rg.git_commit(dg_root), "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir}")
    return meta


if __name__ == "__main__":
    main()

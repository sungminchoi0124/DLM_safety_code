"""RQ2 GPU checks on the real model (HANDOFF 5절 2번). Console and JSON carry counts/scores only, never text.

  fastdllm  (1-2) defense off, combined decoder vs the original Fast-dLLM generate / generate_with_prefix_cache /
                  generate_with_dual_cache (steps=256, threshold 0.9) on columns A / D / E: token identity + NFE.
  loader          Fast-dLLM LLaDAModelLM vs transformers AutoModel: baseline hidden, block-0 logits / hidden of
                  the first full forward, and a column-A defense run (tokens, audit_logit, audit_hidden).
  cache     (3)   columns D / E with the defense: in samples that were repaired, the first forward of block 1
                  is a full forward over the repaired tokens, every later cached forward reuses that cache, and
                  a cached forward's logits agree with a no-cache forward on the same sequence.
  rq1       (2)   per-item comparison of a run's .meta.json with RQ1 per_item.csv (pred_flexible / correct).

Usage (from RQ2/):
  FAST_DLLM_ROOT=/root/Fast-dLLM python scripts/verify_gpu.py fastdllm --limit 10 \
      --model_path /root/models/LLaDA-8B-Instruct --prompt_file /root/data/gsm8k_5shot_lmeval.json
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

RQ2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RQ2_DIR))
import run_rq2_gsm8k as runner                     # noqa: E402  (also puts DiffuGuard/ on sys.path)
from run_rq2_gsm8k import rg, rq1, combined        # noqa: E402

GEN = dict(gen_length=256, block_length=32)
OUT_DIR = RQ2_DIR / "outputs" / "verify"


def log(msg):
    print(msg, flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def common_args(p):
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model_path", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--fast_dllm_root", default=None)
    p.add_argument("--output_dir", default=str(OUT_DIR))


def setup(args, model_class="fastdllm"):
    """Items, original helper module, and a model loaded exactly like the runner."""
    import os
    root = args.fast_dllm_root or os.environ.get("FAST_DLLM_ROOT")
    items, spec, _ = rq1.load_items({"prompt_file": args.prompt_file, "prompt_file_sha256": rq1.PROMPTS_SHA256,
                                     "doc_ids": str(runner.DEFAULT_IDS), "limit": None})
    items = items[: args.limit]
    import yaml
    dg_cfg = yaml.safe_load(open(runner.DG_CONFIG, encoding="utf-8"))
    jbl = rg.import_original(rg.resolve(dg_cfg["diffuguard_root"]))
    revision = rg.resolve_model_revision(args.model_path, None)
    ns = argparse.Namespace(model_class=model_class, fast_dllm_root=root)
    tok, model, source = runner.load_model(ns, args.model_path, revision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return items, jbl, tok, model, source, revision, root


def encode(jbl, tok, model, item, model_path):
    prompt = jbl.build_chat_prompt(tok, item["context"], rg.is_instruct_path(model_path), None)
    return jbl.get_tokenized_input(prompt, tok, next(model.parameters()).device)


def import_fast_dllm(root):
    llada = Path(root) / "v1" / "llada"
    if str(llada) not in sys.path:
        sys.path.insert(0, str(llada))
    import generate as fd                          # Fast-dLLM v1/llada/generate.py (a9b81e4)
    return fd


def diff_info(a, b, P):
    a, b = a[0, P:].cpu(), b[0, P:].cpu()
    d = (a != b).nonzero().flatten()
    return {"equal": bool(d.numel() == 0), "n_diff": int(d.numel()),
            "first_diff": int(d[0]) if d.numel() else None}


def save(args, name, payload):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"saved -> {path}")


def env_info(source, revision):
    return {"model_class": source, "model_revision": revision, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "team_repo_commit": rg.git_commit(runner.REPO), "team_repo_dirty": rg.git_dirty(runner.REPO)}


# --------------------------------------------------------------------------- (1-2) vs original Fast-dLLM

def cmd_fastdllm(args):
    items, jbl, tok, model, source, revision, root = setup(args)
    fd = import_fast_dllm(root)
    fns = {"A": (fd.generate, None), "D": (fd.generate_with_prefix_cache, "prefix"),
           "E": (fd.generate_with_dual_cache, "dual")}
    rows = []
    for col in args.columns:
        fn, cache = fns[col]
        tau = runner.COLUMNS[col]["threshold"]
        for item in items:
            ids, am = encode(jbl, tok, model, item, args.model_path)
            P = ids.shape[1]
            rg.set_seed(item["doc_id"])
            sync(); t0 = time.perf_counter()
            ours, st = combined.generate(model, ids, attention_mask=am, temperature=0.0, mode="threshold",
                                         threshold=tau, cache=cache, defense=False, **GEN)
            sync(); t1 = time.perf_counter()
            rg.set_seed(item["doc_id"])
            ref, ref_nfe = fn(model, ids, steps=GEN["gen_length"], temperature=0.0, threshold=tau, **GEN)
            sync(); t2 = time.perf_counter()
            r = {"column": col, "doc_id": item["doc_id"], "prompt_tokens": P, **diff_info(ours, ref, P),
                 "nfe_ours": st.nfe_gen, "nfe_fastdllm": int(ref_nfe),
                 "masks_left_ours": int((ours[0, P:] == combined.MASK_ID).sum()),
                 "masks_left_fastdllm": int((ref[0, P:] == combined.MASK_ID).sum()),
                 "sec_ours": round(t1 - t0, 2), "sec_fastdllm": round(t2 - t1, 2)}
            rows.append(r)
            log(f"[{col}] doc={r['doc_id']} equal={r['equal']} n_diff={r['n_diff']} first={r['first_diff']} "
                f"nfe={r['nfe_ours']}/{r['nfe_fastdllm']} t={r['sec_ours']}/{r['sec_fastdllm']}s")
    summary = {c: {"n": sum(r["column"] == c for r in rows),
                   "equal": sum(r["equal"] for r in rows if r["column"] == c),
                   "nfe_equal": sum(r["nfe_ours"] == r["nfe_fastdllm"] for r in rows if r["column"] == c)}
               for c in args.columns}
    log(f"summary {json.dumps(summary)}")
    save(args, f"check1-2_fastdllm_limit{args.limit}.json",
         {"check": "1-2", "env": env_info(source, revision), "fast_dllm_commit": rg.git_commit(root),
          "summary": summary, "rows": rows})


# --------------------------------------------------------------------------- LLaDAModelLM vs AutoModel

def loader_pass(args, model_class):
    items, jbl, tok, model, source, revision, _ = setup(args, model_class)
    cue = combined.cue_id_sets(tok)
    s = runner.settings(argparse.Namespace(column="A", defense="dg", block_length=None, steps=None,
                                           threshold=None, refinement_steps=None, alpha_rule=None,
                                           alpha0=None, sp_threshold=None))
    kw = {k: v for k, v in s.items() if k != "temperature"}
    res = []
    for item in items:
        ids, am = encode(jbl, tok, model, item, args.model_path)
        P = ids.shape[1]
        with torch.no_grad():
            bh = jbl.compute_baseline_hidden(vanilla_text=item["context"], tokenizer=tok, model=model,
                                             is_instruct=rg.is_instruct_path(args.model_path), system_prompt=None,
                                             debug_print=False)
            x = torch.cat([ids, torch.full((1, GEN["gen_length"]), combined.MASK_ID, device=ids.device)], 1)
            amx = torch.ones_like(x, dtype=torch.bool)
            out = model(x, attention_mask=amx, output_hidden_states=True, return_dict=True)
            blk = slice(P, P + GEN["block_length"])
            logits0 = out.logits[0, blk].float().cpu()
            hid0 = out.hidden_states[-1][0, blk].float().mean(0).cpu()
            del out
        rg.set_seed(item["doc_id"])
        with torch.no_grad():
            toks, st = combined.generate(model, ids, attention_mask=am, temperature=0.0, baseline_hidden=bh,
                                         cue_ids=cue, **kw)
        res.append({"doc_id": item["doc_id"], "P": P, "baseline_hidden": None if bh is None else bh.float().cpu(),
                    "logits0": logits0, "hidden0": hid0, "tokens": toks[0, P:].cpu(),
                    "audit_logit": st.audit_logit, "audit_hidden": st.audit_hidden,
                    "audit_unsafe": st.audit_unsafe, "repaired": st.repaired, "nfe": st.nfe_gen + st.nfe_repair})
        log(f"[{model_class}] doc={item['doc_id']} audit_logit={st.audit_logit:.5f} "
            f"audit_hidden={st.audit_hidden if st.audit_hidden is None else round(st.audit_hidden, 5)} "
            f"repaired={st.repaired}")
    info = env_info(source, revision)
    del model
    torch.cuda.empty_cache()
    return res, info


def cmd_loader(args):
    a, info_a = loader_pass(args, "fastdllm")
    b, info_b = loader_pass(args, "auto")
    rows = []
    for ra, rb in zip(a, b):
        cos = lambda u, v: float(torch.nn.functional.cosine_similarity(u, v, dim=0))
        r = {"doc_id": ra["doc_id"],
             "baseline_hidden_equal": bool(torch.equal(ra["baseline_hidden"], rb["baseline_hidden"])),
             "baseline_hidden_maxabs": float((ra["baseline_hidden"] - rb["baseline_hidden"]).abs().max()),
             "baseline_hidden_cos": cos(ra["baseline_hidden"], rb["baseline_hidden"]),
             "block0_logits_equal": bool(torch.equal(ra["logits0"], rb["logits0"])),
             "block0_logits_maxabs": float((ra["logits0"] - rb["logits0"]).abs().max()),
             "block0_argmax_agree": int((ra["logits0"].argmax(-1) == rb["logits0"].argmax(-1)).sum()),
             "block0_hidden_maxabs": float((ra["hidden0"] - rb["hidden0"]).abs().max()),
             "tokens_equal": bool(torch.equal(ra["tokens"], rb["tokens"])),
             "tokens_n_diff": int((ra["tokens"] != rb["tokens"]).sum()),
             "audit_logit": [ra["audit_logit"], rb["audit_logit"]],
             "audit_hidden": [ra["audit_hidden"], rb["audit_hidden"]],
             "audit_logit_equal": ra["audit_logit"] == rb["audit_logit"],
             "audit_hidden_equal": ra["audit_hidden"] == rb["audit_hidden"],
             "repaired": [ra["repaired"], rb["repaired"]], "nfe": [ra["nfe"], rb["nfe"]]}
        rows.append(r)
        log(f"doc={r['doc_id']} bh_eq={r['baseline_hidden_equal']} logits_eq={r['block0_logits_equal']} "
            f"(maxabs {r['block0_logits_maxabs']:.3g}) tok_eq={r['tokens_equal']} "
            f"audit_eq={r['audit_logit_equal']}/{r['audit_hidden_equal']}")
    keys = ("baseline_hidden_equal", "block0_logits_equal", "tokens_equal", "audit_logit_equal",
            "audit_hidden_equal")
    summary = {k: sum(bool(r[k]) for r in rows) for k in keys}
    summary["n"] = len(rows)
    log(f"summary {json.dumps(summary)}")
    save(args, f"check_loader_limit{args.limit}.json",
         {"check": "loader", "env": {"fastdllm": info_a, "auto": info_b}, "decode": "column A, defense dg",
          "summary": summary, "rows": rows})


# --------------------------------------------------------------------------- (3) cache rebuild after repair

class Spy:
    """Delegates to the model; records every forward (kind, window, input ids, cache object identities)."""

    def __init__(self, model, P):
        self.m, self.P, self.calls, self.probe = model, P, [], None

    def parameters(self):
        return self.m.parameters()

    def __call__(self, x, **kw):
        pkv_in = kw.get("past_key_values")
        out = self.m(x, **kw)
        if "use_cache" not in kw:
            kind = "repair"
        elif pkv_in is None:
            kind = "full"
        else:
            kind = "cached"
        pkv_out = getattr(out, "past_key_values", None) if kind == "full" else None
        rec = {"kind": kind, "len": int(x.shape[1]), "ids": x[0].detach().cpu().clone(),
               "argmax": out.logits[0].argmax(-1).cpu() if kind != "repair" else None,
               "pkv_in_ptr": None if pkv_in is None else int(pkv_in[0][0].data_ptr()),
               "pkv_out_ptr": None if pkv_out is None else int(pkv_out[0][0].data_ptr())}
        if self.probe is not None and kind == "cached":
            self.probe(rec, x, kw, out)
        self.calls.append(rec)
        return out


def cmd_cache(args):
    items, jbl, tok, model, source, revision, _ = setup(args)
    cue = combined.cue_id_sets(tok)
    rows = []
    for col in args.columns:
        ns = argparse.Namespace(column=col, defense="dg", block_length=None, steps=None, threshold=None,
                                refinement_steps=None, alpha_rule=None, alpha0=None,
                                sp_threshold=args.sp_threshold)
        s = runner.settings(ns)
        kw = {k: v for k, v in s.items() if k != "temperature"}
        for item in items:
            ids, am = encode(jbl, tok, model, item, args.model_path)
            P = ids.shape[1]
            with torch.no_grad():
                bh = jbl.compute_baseline_hidden(vanilla_text=item["context"], tokenizer=tok, model=model,
                                                 is_instruct=rg.is_instruct_path(args.model_path),
                                                 system_prompt=None, debug_print=False)
            spy = Spy(model, P)
            probe_res = {}
            bs1 = P + GEN["block_length"]

            def probe(rec, x, kw_, out, spy=spy, probe_res=probe_res, bs1=bs1, P=P, cache=s["cache"]):
                # First cached forward of block 1 after a repair. Replay it with a cache rebuilt from
                # (a) the sequence of block 1's first full forward (repaired block 0): must equal what ran;
                # (b) the same sequence with block 0 set back to the pre-repair tokens: must differ.
                calls = spy.calls
                if probe_res or not any(c["kind"] == "repair" for c in calls):
                    return
                start = next(c for c in reversed(calls) if c["kind"] == "full")["ids"].to(x.device)
                first_rep = next(i for i, c in enumerate(calls) if c["kind"] == "repair")
                pre = pre_repair_block0(calls, first_rep, P).to(x.device)
                got = out.logits.float()
                for name, blk0 in (("fresh", None), ("stale", pre)):
                    xs = start.clone().unsqueeze(0)
                    if blk0 is not None:
                        xs[0, P:P + GEN["block_length"]] = blk0
                    with torch.no_grad():
                        pkv = spy.m(xs, use_cache=True).past_key_values
                        if cache == "prefix":
                            pkv = [tuple(t[:, :, :bs1] for t in layer) for layer in pkv]
                            ref = spy.m(x, past_key_values=pkv, use_cache=True).logits.float()
                        else:
                            ref = spy.m(x, past_key_values=pkv, use_cache=True,
                                        replace_position=kw_["replace_position"]).logits.float()
                    probe_res[name] = {"logits_equal": bool(torch.equal(ref, got)),
                                       "maxabs": float((ref - got).abs().max()),
                                       "argmax_agree": int((ref.argmax(-1) == got.argmax(-1)).sum()),
                                       "positions": int(x.shape[1])}
                probe_res["pre_vs_repaired_block0_n_diff"] = int((pre != start[P:P + GEN["block_length"]]).sum())

            spy.probe = probe
            rg.set_seed(item["doc_id"])
            with torch.no_grad():
                out, st = combined.generate(spy, ids, attention_mask=am, temperature=0.0, baseline_hidden=bh,
                                            cue_ids=cue, **kw)
            r = {"column": col, "doc_id": item["doc_id"], "repaired": st.repaired,
                 "audit_score": st.audit_score, "nfe_gen": st.nfe_gen, "nfe_repair": st.nfe_repair}
            if st.repaired:
                r.update(check_rebuild(spy.calls, out[0].cpu(), P, s["cache"]))
                r["probe_cached_vs_full"] = probe_res or None
            rows.append(r)
            log(f"[{col}] doc={item['doc_id']} repaired={st.repaired} "
                + (f"rebuild_ok={r.get('ok')} probe={probe_res}" if st.repaired else ""))
    summary = {c: {"n": sum(r["column"] == c for r in rows),
                   "repaired": sum(r["repaired"] for r in rows if r["column"] == c),
                   "rebuild_ok": sum(bool(r.get("ok")) for r in rows if r["column"] == c)} for c in args.columns}
    log(f"summary {json.dumps(summary)}")
    save(args, f"check3_cache_limit{args.limit}_sp{args.sp_threshold}.json",
         {"check": "3", "env": env_info(source, revision), "sp_threshold": args.sp_threshold,
          "forced_repair": args.sp_threshold is not None and args.sp_threshold <= 0,
          "summary": summary, "rows": rows})


def pre_repair_block0(calls, first_rep, P):
    """Block 0 right before the repair: the last generation forward's input with its still-masked block
    positions filled by that forward's argmax (temperature 0: the last threshold step commits all of them)."""
    B = GEN["block_length"]
    last = calls[first_rep - 1]
    off = 0 if last["kind"] == "full" else P
    blk = last["ids"][P - off:P - off + B].clone()
    m = blk == combined.MASK_ID
    blk[m] = last["argmax"][P - off:P - off + B][m]
    return blk


def check_rebuild(calls, final, P, cache):
    B = GEN["block_length"]
    kinds = [c["kind"] for c in calls]
    rep = [i for i, k in enumerate(kinds) if k == "repair"]
    first_rep, last_rep = rep[0], rep[-1]
    nxt = last_rep + 1
    c1 = calls[nxt] if nxt < len(calls) else None
    pre_block0 = calls[first_rep]["ids"][P:P + B]          # 90% masked at repair start
    res = {"n_repair_calls": len(rep), "repair_calls_contiguous": rep == list(range(first_rep, last_rep + 1)),
           "repair_calls_full_length": all(calls[i]["len"] == final.numel() for i in rep),
           "repair_start_masked": int((pre_block0 == combined.MASK_ID).sum())}
    if c1 is None:
        res["ok"] = False
        return res
    blk0_in = c1["ids"][P:P + B]
    res.update({
        "after_repair_is_full": c1["kind"] == "full" and c1["len"] == final.numel(),
        "after_repair_block0_eq_final": bool(torch.equal(blk0_in, final[P:P + B])),
        "after_repair_block0_no_mask": bool((blk0_in != combined.MASK_ID).all()),
    })
    # cached forwards of block 1 (until the next full forward) must use the cache built by c1
    ptr = c1["pkv_out_ptr"]
    b1_cached = []
    for c in calls[nxt + 1:]:
        if c["kind"] != "cached":
            break
        b1_cached.append(c)
    stale = {c["pkv_out_ptr"] for c in calls[:first_rep] if c["kind"] == "full"}
    res.update({"block1_cached_calls": len(b1_cached),
                "block1_cache_from_rebuild": all(c["pkv_in_ptr"] == ptr for c in b1_cached),
                "block1_cache_stale": any(c["pkv_in_ptr"] in stale for c in b1_cached)})
    res["ok"] = all([res["repair_calls_contiguous"], res["repair_calls_full_length"],
                     res["after_repair_is_full"], res["after_repair_block0_eq_final"],
                     res["after_repair_block0_no_mask"], res["block1_cache_from_rebuild"],
                     not res["block1_cache_stale"]])
    return res


# --------------------------------------------------------------------------- (2) vs RQ1 per_item.csv

def cmd_rq1(args):
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    ref = {}
    with open(args.rq1_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["defense"] == args.rq1_defense and int(row["steps"]) == args.rq1_steps:
                ref[int(row["doc_id"])] = row
    rows = []
    for r in meta["per_item"]:
        q = ref.get(r["doc_id"])
        if q is None:
            continue
        rows.append({"doc_id": r["doc_id"], "pred_equal": str(r["pred_flexible"]) == q["pred_flexible"],
                     "correct": [bool(r["correct_flexible"]), q["correct_flexible"] == "True"],
                     "n_tokens": [r["n_tokens"], int(q["n_tokens"])],
                     "repaired": [r["repaired"], q["repaired"] == "True"],
                     "audit_score": [r["audit_score"], float(q["audit_score"]) if q["audit_score"] else None]})
    summary = {"n": len(rows), "pred_equal": sum(r["pred_equal"] for r in rows),
               "correct_equal": sum(r["correct"][0] == r["correct"][1] for r in rows),
               "n_tokens_equal": sum(r["n_tokens"][0] == r["n_tokens"][1] for r in rows),
               "acc_rq2": sum(r["correct"][0] for r in rows), "acc_rq1": sum(r["correct"][1] for r in rows)}
    for r in rows:
        log(f"doc={r['doc_id']} pred_eq={r['pred_equal']} correct={r['correct']} n_tok={r['n_tokens']}")
    log(f"summary {json.dumps(summary)}")
    save(args, f"check2_rq1_{Path(args.meta).stem}_vs_{args.rq1_defense}{args.rq1_steps}.json",
         {"check": "2", "meta": str(args.meta), "rq1_csv": str(args.rq1_csv), "summary": summary, "rows": rows})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fastdllm"); common_args(a); a.add_argument("--columns", nargs="+", default=["A", "D", "E"])
    b = sub.add_parser("loader"); common_args(b)
    c = sub.add_parser("cache"); common_args(c); c.add_argument("--columns", nargs="+", default=["D", "E"])
    c.add_argument("--sp_threshold", type=float, default=None, help="<= 0 forces a repair on every sample")
    d = sub.add_parser("rq1"); d.add_argument("--meta", required=True)
    d.add_argument("--rq1_csv", default=str(runner.REPO / "RQ1" / "results" / "gsm8k_250" / "per_item.csv"))
    d.add_argument("--rq1_defense", default="vanilla"); d.add_argument("--rq1_steps", type=int, default=256)
    d.add_argument("--output_dir", default=str(OUT_DIR))
    args = p.parse_args()
    if args.cmd != "rq1" and not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU visible")
    {"fastdllm": cmd_fastdllm, "loader": cmd_loader, "cache": cmd_cache, "rq1": cmd_rq1}[args.cmd](args)


if __name__ == "__main__":
    main()

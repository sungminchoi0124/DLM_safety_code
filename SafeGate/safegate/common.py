"""Pieces shared by run_gate_asr.py and run_gate_gsm8k.py: config, rows, model, resume, summaries."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import yaml

SG_DIR = Path(__file__).resolve().parents[1]
REPO = SG_DIR.parent
for p in (REPO / "RQ2", REPO / "DiffuGuard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_rq2_gsm8k as rq2     # noqa: E402  (RQ2 GSM8K runner: load_model, DiffuGuard preset, summaries)
import run_generate as rg       # noqa: E402  (DiffuGuard/run_generate.py: seeding, original-code import, git)

from .decoder import GateConfig  # noqa: E402
from .tokens import build_token_sets  # noqa: E402

DEFAULT_CONFIG = SG_DIR / "configs" / "gate.yaml"
ROWS = ("none", "dg", "gate")


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gate_config(cfg) -> GateConfig:
    g = cfg["gate"]
    if g.get("a") is None or g.get("b") is None:
        raise SystemExit("configs: gate.a / gate.b are null. Choose them from the dev traces (analyze_dev.py) "
                         "and write them into the config before running --row gate.")
    return GateConfig(a=float(g["a"]), b=float(g["b"]), K=int(g["K"]), H=int(g["H"]),
                      windows=tuple(g["windows"]), K_infill=g.get("K_infill"))


def dg_settings():
    """DiffuGuard preset `diffuguard` (alpha0 0.3, lambda 0.1, gamma 0.9, 8 repair steps), as the RQ2 runs."""
    with open(rq2.DG_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pre = cfg["presets"]["diffuguard"]
    return dict(defense=True, alpha0=float(pre["alpha0"]), sp_threshold=float(pre["sp_threshold"]),
                remask_ratio=float(pre["remask_ratio"]), refinement_steps=int(pre["refinement_steps"]),
                suppression_value=float(cfg["suppression_value"]))


def row_settings(row, cfg):
    """Decoder keyword arguments of a row (on top of the task's gen_length / block_length / temperature)."""
    if row not in ROWS:
        raise SystemExit(f"--row must be one of {ROWS}")
    s = {"threshold": float(cfg["accel"]["threshold"]), "defense": False}
    if row == "dg":
        s.update(dg_settings())
    return s


def token_sets(tokenizer, cfg):
    t = cfg.get("tokens") or {}
    return build_token_sets(tokenizer, t.get("refusal"), t.get("compliance"))


def dg_meta():
    with open(rq2.DG_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_and_tokenizer(args, dg_cfg):
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU visible (use --dry_run on CPU).")
    model_path = args.model_path or dg_cfg["model_path"]
    revision = rg.resolve_model_revision(model_path, dg_cfg.get("model_revision"))
    tokenizer, model, source = rq2.load_model(args, model_path, revision)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return tokenizer, model, source, model_path, revision


def add_model_args(p):
    import os
    p.add_argument("--model_path")
    p.add_argument("--model_class", choices=["fastdllm", "auto"], default="fastdllm",
                   help="fastdllm = Fast-dLLM LLaDAModelLM (RQ2 runs; needs FAST_DLLM_ROOT), auto = AutoModel")
    p.add_argument("--fast_dllm_root", default=os.environ.get("FAST_DLLM_ROOT"))


def read_partial(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def append_partial(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.ms = 1000 * (time.perf_counter() - self.t0)
        return False


def stats_fields(st, gen_tokens):
    """Per-item numbers from GateStats (no text)."""
    return {"nfe_gen": st.nfe_gen, "nfe_repair": st.nfe_repair, "block0_forwards": st.block0_forwards,
            "commits_per_forward": round(gen_tokens / max(st.nfe_gen, 1), 4),
            "audit_score": st.audit_score, "audit_logit": st.audit_logit, "audit_hidden": st.audit_hidden,
            "audit_unsafe": st.audit_unsafe, "repaired": st.repaired,
            "gate_checks": st.gate_checks, "gate_steps": st.gate_steps, "gate_held": st.gate_held,
            "gate_first_step": st.gate_first_step, "gate_fired": st.gate_steps > 0}


def summary(records):
    """Speed / defense / gate numbers over items (accuracy or ASR are added by the runner / judge)."""
    n = max(len(records), 1)
    mean = lambda k: round(sum(float(r[k]) for r in records) / n, 3)
    nfe_total = [r["nfe_gen"] + r["nfe_repair"] + r["nfe_baseline"] for r in records]
    return {"n": len(records), "mean_wall_ms": mean("wall_ms"), "mean_nfe_gen": mean("nfe_gen"),
            "mean_nfe_repair": mean("nfe_repair"), "mean_nfe_total": round(sum(nfe_total) / n, 3),
            "tokens_per_forward": round(sum(r["committed_tokens"] for r in records)
                                        / max(sum(r["nfe_gen"] for r in records), 1), 4),
            "repair_rate": round(sum(bool(r["repaired"]) for r in records) / n, 4),
            "gate_fire_rate": round(sum(bool(r["gate_fired"]) for r in records) / n, 4),
            "mean_gate_steps": mean("gate_steps"), "mean_gate_held": mean("gate_held")}


def provenance(dg_root):
    return {"team_repo_commit": rg.git_commit(REPO), "team_repo_dirty": rg.git_dirty(REPO),
            "diffuguard_commit": rg.git_commit(dg_root), "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None, "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__}

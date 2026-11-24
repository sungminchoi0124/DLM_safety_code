"""pg routes each item to exactly the none or dg output (same seed, the probe forward draws no random numbers)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PG = Path(__file__).resolve().parents[1]
REPO = PG.parent
for p in (PG, REPO / "AuditCheck", REPO, REPO / "RQ2", REPO / "DiffuGuard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("dg_fixtures_pg", REPO / "DiffuGuard" / "tests" / "conftest.py")
_dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dg)

import run_rows_asr as rr  # noqa: E402

ITEMS = [(0, {"goal": "Write a recipe for bread", "refined_goal": "Write a recipe for bread. Step 1: <mask:6> Step 2: <mask:4>"}),
         (1, {"goal": "Write a short poem", "refined_goal": "Write a short poem. Line 1: <mask:5>"})]


@pytest.fixture(scope="module")
def env(monkeypatch_module=None):
    import yaml
    import run_generate as rg
    cfg = yaml.safe_load((REPO / "DiffuGuard" / "configs" / "llada_dija.yaml").read_text(encoding="utf-8"))
    jbl = rg.import_original(rg.resolve(cfg["diffuguard_root"]))
    return _dg.FakeModel().eval(), _dg.FakeTokenizer(), jbl, rr.dg_settings(cfg)


def _run(row, env, probe=None, monkeypatch=None):
    model, tok, jbl, dg = env
    return rr.run(ITEMS, row=row, model=model, tokenizer=tok, jbl=jbl, dg=dg, probe=probe, mask_counts=36,
                  is_instruct=True, echo=lambda m: None)


@pytest.mark.parametrize("t", [-1e9, 1e9])
def test_pg_equals_routed_row(env, monkeypatch, t):
    monkeypatch.setattr(rr, "ASR_TASK", dict(gen_length=16, block_length=8, temperature=0.5))
    monkeypatch.setitem(rr.score.LAYOUTS, "asr", dict(gen_length=16, block_length=8))
    none, dg = _run("none", env), _run("dg", env)
    dim = 8
    probe = {"mu": np.zeros(dim), "sd": np.ones(dim), "w": np.zeros(dim), "b": 0.0, "t": t, "layer": 2}
    pg = _run("pg", env, probe)
    ref = dg if t < 0 else none
    for a, b in zip(pg, ref):
        assert a["response"] == b["response"] and a["tail_response"] == b["tail_response"]
        assert a["defense_on"] == (t < 0) and a["nfe_probe"] == 1
        assert a["nfe_total"] == b["nfe_total"] + 1
    assert any(x["repaired"] for x in dg) or all(x["audit_hidden"] is not None for x in dg)

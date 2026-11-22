"""Prophet early-exit decoder, logit statistics in the recorder, and the Prophet metrics / analysis."""
import csv
import json
import math

import numpy as np
import pytest
import torch

from conftest import MASK
from commit_time import analyze, decode as dec, metrics
from commit_time.capture import FINAL, TraceRecorder
from commit_time.decode import decode

STEPS, GEN, LEAD = 21, 16, 4
INF3 = (math.inf, math.inf, math.inf)


def _run(model, prompt, rule="quota", seed=0, stats=True, **kw):
    rec = TraceRecorder(model, mask_id=MASK, prompt_len=prompt.shape[1], layers=(1, FINAL), lead_k=LEAD,
                        logit_stats=stats, mass_refuse_ids={1, 2, 3}, mass_comply_ids={4, 5})
    torch.manual_seed(seed)
    with rec:
        res = decode(model, prompt, None, gen_length=GEN, steps=STEPS, temperature=0.5, mask_id=MASK, rule=rule,
                     answer_len=LEAD, **kw)
        rec.finish(res.x)
    return res, rec


def test_infinite_threshold_equals_quota(model, prompt):
    q, _ = _run(model, prompt, "quota")
    p, _ = _run(model, prompt, "prophet", prophet_thresholds=INF3)
    assert torch.equal(q.x, p.x) and p.nfe == q.nfe == STEPS
    info = p.info
    assert not info["exited"] and info["exit_step"] == STEPS and info["exit_reason"] == "completed"
    assert len(info["gap_all"]) == len(info["gap_masked"]) == len(info["gap_threshold"]) == STEPS
    assert math.isnan(info["gap_committed"][0]) and not math.isnan(info["gap_masked"][0])


def test_zero_threshold_exits_at_first_forward(model, prompt):
    p, rec = _run(model, prompt, "prophet", prophet_thresholds=(0.0, 0.0, 0.0))
    assert p.nfe == 1 and p.info["exited"] and p.info["exit_step"] == 1
    assert not (p.x == MASK).any()
    a = rec.arrays()
    n = GEN + 5
    assert p.info["n_filled_at_exit"] == n
    # the all-in fill is the transition of the last (= only) forward
    assert a["hidden"].shape[0] == 1 and a["n_commit"].tolist() == [n] and a["n_masks"].tolist() == [n, 0]


def test_exit_matches_quota_before_exit_and_records_fill(model, prompt):
    q, qrec = _run(model, prompt, "quota")
    g = qrec.arrays()["gap_all"].astype(float)
    late = 0.95 * float(np.min(g[15:]))            # guarantees an exit in the late phase
    p, prec = _run(model, prompt, "prophet", prophet_thresholds=(math.inf, math.inf, late))
    e = p.info["exit_step"]
    assert p.info["exited"] and 14 <= e <= STEPS
    # the quota-path tokens up to the exit are the quota tokens
    torch.manual_seed(0)
    q_snap = decode(model, prompt, None, gen_length=GEN, steps=STEPS, temperature=0.5, mask_id=MASK,
                    rule="quota", snapshot_at=[e - 1]).snapshots[e - 1].x
    done = q_snap != MASK
    assert torch.equal(p.x[done], q_snap[done])
    # decoder gaps == recorder gaps (same logits), and the recorder sees the all-in fill
    pa = prec.arrays()
    np.testing.assert_allclose(np.array(p.info["gap_all"]), pa["gap_all"], rtol=1e-5)
    assert pa["hidden"].shape[0] == e and pa["n_commit"][-1] == p.info["n_filled_at_exit"]
    assert int(pa["n_commit"].sum()) == int(pa["n_tracked"]) and pa["n_masks"][-1] == 0
    # exit step simulated from the quota trace == the real Prophet exit (same seed)
    sim = metrics.simulated_exit_step(g, STEPS, thresholds=(math.inf, math.inf, late))
    assert sim == e


def test_argmax_fill_is_noise_free(model, prompt):
    p1, _ = _run(model, prompt, "prophet", seed=0, prophet_thresholds=(0.0, 0.0, 0.0), prophet_fill="argmax")
    p2, _ = _run(model, prompt, "prophet", seed=9, prophet_thresholds=(0.0, 0.0, 0.0), prophet_fill="argmax")
    assert torch.equal(p1.x, p2.x)                 # first forward is the same; no Gumbel noise in the fill


def test_resume_keeps_answer_region(model, prompt):
    torch.manual_seed(0)
    ref = decode(model, prompt, None, gen_length=GEN, steps=STEPS, temperature=0.5, mask_id=MASK, rule="prophet",
                 prophet_thresholds=INF3, answer_len=LEAD, snapshot_at=range(30))
    snap = ref.snapshots[6]
    assert snap.prompt_len == prompt.shape[1]
    kw = dict(gen_length=GEN, steps=STEPS, temperature=0.5, mask_id=MASK, rule="prophet", answer_len=LEAD,
              prophet_thresholds=INF3, resume=snap, seed=3)
    r1, r2 = decode(model, None, None, **kw), decode(model, None, None, **kw)
    assert torch.equal(r1.x, r2.x) and r1.nfe == STEPS - 6
    assert r1.info["answer_start"] == prompt.shape[1] and r1.info["exit_step"] == STEPS
    assert len(r1.info["gap_all"]) == STEPS - 6


def test_threshold_curve_matches_decoder():
    assert metrics.PROPHET_THRESHOLDS == dec.PROPHET_THRESHOLDS and metrics.PROPHET_PHASES == dec.PROPHET_PHASES
    curve = metrics.prophet_threshold_curve(128, 128)
    assert all(curve[f] == dec.prophet_threshold(f + 1, 128) for f in range(128))
    assert curve[0] == 7.5 and curve[41] == 7.5 and curve[42] == 5.0 and curve[127] == 2.5


def test_logit_stats_in_recorder(model, prompt):
    _, rec = _run(model, prompt, "quota")
    a = rec.arrays()
    T = a["hidden"].shape[0]
    for k in ("gap_all", "gap_committed", "gap_masked", "mass_refuse", "mass_comply"):
        assert a[k].shape == (T,)
    assert np.all(a["gap_all"] >= 0)
    assert np.isnan(a["gap_committed"][0])
    m = a["mass_refuse"][np.isfinite(a["mass_refuse"])]
    assert m.size and np.all((m >= 0) & (m <= 1))
    # once every lead position is committed there is no masked lead position left -> NaN
    full = np.nonzero(a["lead_unmasked"][:T] == LEAD)[0]
    if full.size:
        assert np.isnan(a["gap_masked"][full[0]]) and np.isnan(a["mass_refuse"][full[0]])
    _, rec2 = _run(model, prompt, "quota", stats=False)
    assert "gap_all" not in rec2.arrays()


def test_item_metrics_prophet_columns():
    tr = {"n_tracked": 10, "n_masks": np.array([10, 8, 6, 0]), "n_commit": np.array([2, 2, 6]),
          "n_remask": np.zeros(3, int), "lead_unmasked": np.array([0, 2, 3, 4]), "lead_k": 4,
          "prophet_exited": 1, "prophet_exit_step": 3, "prophet_n_filled_at_exit": 6,
          "gap_all": np.array([1.0, 2.0, 9.0]), "gap_committed": np.array([np.nan, 4.0, 9.5]),
          "gap_masked": np.array([1.0, 1.5, 2.0]), "mass_refuse": np.array([0.2, 0.1, 0.0]),
          "mass_comply": np.array([0.3, 0.2, 0.1])}
    row = metrics.item_metrics(np.array([-0.1, -0.8, -1.0]), 0, 0.5, tr)
    assert row["exited"] and row["exit_step"] == 3 and row["n_filled_at_exit"] == 6
    assert row["exit_frac_tokens"] == pytest.approx(0.4)          # 4 of 10 committed before the fill
    assert row["s_star"] == 1 and row["exit_after_s_star"]
    assert row["sim_exit_step"] == 3 and row["gap_committed_median"] == pytest.approx(6.75)
    assert row["mass_refuse_first"] == pytest.approx(0.2)


def test_gaps_cli_and_flips_with_exit(model, prompt, tmp_path):
    for i in range(3):
        _, rec = _run(model, prompt, "quota", seed=i)
        rec.save(tmp_path / "R" / "hidden" / f"item{i:03d}.npz")
    out = tmp_path / "gaps.csv"
    analyze.main(["gaps", "--run", str(tmp_path / "R"), "--out", str(out), "--bins", "4"])
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3 and {"sim_exit_step", "s_text", "gap_committed_median"} <= set(rows[0])
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert len(summ["bins"]) == 4 and summ["n_items"] == 3

    base = {"n_hes": 2, "frac_tokens_at_s_star": 0.5, "max_bundle_before": 1, "censored": False}
    ref = [{"item_index": i, "label": 1, "s_star": 5, **base} for i in range(2)]
    acc = [{"item_index": 0, "label": 1, "exit_step": 9, "exited": True, **base},
           {"item_index": 1, "label": 0, "exit_step": 3, "exited": True, **base}]
    analyze.write_csv(tmp_path / "ref.csv", ref)
    analyze.write_csv(tmp_path / "acc.csv", acc)
    analyze.main(["flips", "--ref", str(tmp_path / "ref.csv"), "--acc", str(tmp_path / "acc.csv"),
                  "--out", str(tmp_path / "flips.csv")])
    fl = list(csv.DictReader(open(tmp_path / "flips.csv")))
    assert [r["exit_after_s_star_ref"] for r in fl] == ["True", "False"]
    assert [r["flipped"] for r in fl] == ["False", "True"]


def test_run_trace_prophet_install(model, tok, prompt, tmp_path):
    """--decoder prophet through install(): npz carries the prophet arrays, the meta line the exit info."""
    import types
    from commit_time.run_trace import build_parser, install
    jbl = types.ModuleType("fake_jbl")
    jbl.get_tokenized_input = lambda text, tokenizer, device: (prompt.to(device), torch.ones_like(prompt))
    jbl.generate_llada = lambda *a, **k: None
    jbl.generate_response = lambda *a, **k: None
    opts = build_parser().parse_args(["--trace_dir", str(tmp_path), "--decoder", "prophet", "--layers", "1,final",
                                      "--lead_k", "4", "--answer_len", "4", "--prophet_thresholds", "0,0,0"])
    install(jbl, opts, tmp_path)
    args = types.SimpleNamespace(steps=STEPS, mask_id=MASK, gen_length=GEN, block_length=GEN, temperature=0.5,
                                 cfg_scale=0.0, sp_mode="off", remasking="low_confidence", attack_method="DIJA")
    torch.manual_seed(0)
    jbl.generate_response("v", "p", tok, model, args)
    z = np.load(tmp_path / "hidden" / "item000.npz")
    assert int(z["prophet_exited"]) == 1 and int(z["prophet_exit_step"]) == 1 and z["hidden"].shape[0] == 1
    assert "gap_all" in z.files and "mass_refuse" in z.files
    meta = json.loads((tmp_path / "trace_meta.jsonl").read_text().strip())
    assert meta["prophet"]["exited"] and meta["prophet"]["exit_step"] == 1 and "response_sha256_12" in meta
    assert not any(isinstance(v, str) and len(v) > 40 for v in meta.values())   # no text in the meta line

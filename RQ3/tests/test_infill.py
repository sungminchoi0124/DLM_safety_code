"""DIJA infill window (first prompt masks) in the recorder / metrics, and the assistant-tail JSON (RAEC span)."""
import json
import types

import numpy as np
import pytest
import torch

from conftest import MASK
from commit_time import decode as dec, metrics
from commit_time.capture import FINAL, LOGIT_KEYS, TraceRecorder
from commit_time.decode import decode

STEPS, GEN, LEAD = 21, 16, 4


def _run(model, prompt, stats=True):
    rec = TraceRecorder(model, mask_id=MASK, prompt_len=prompt.shape[1], layers=(1, FINAL), lead_k=LEAD,
                        logit_stats=stats, mass_refuse_ids={1, 2, 3}, mass_comply_ids={4, 5})
    torch.manual_seed(0)
    with rec:
        res = decode(model, prompt, None, gen_length=GEN, steps=STEPS, temperature=0.5, mask_id=MASK, rule="quota")
        rec.finish(res.x)
    return res, rec


def test_infill_window_positions_and_counts(model, prompt):
    res, rec = _run(model, prompt)
    a = rec.arrays()
    T = a["hidden"].shape[0]
    assert rec.infill_pos.tolist() == [5, 6, 7, 12]                 # first LEAD prompt masks
    assert int(a["infill_k"]) == LEAD and int(a["n_infill_tracked"]) == 5
    iu = a["infill_unmasked"]
    assert iu.shape == (T + 1,) and iu[0] == 0 and iu[-1] == LEAD and np.all(np.diff(iu) >= 0)
    final = res.x[0]
    assert int((final[rec.infill_pos] != MASK).sum()) == iu[-1]
    for k in LOGIT_KEYS:
        assert a[k + "_infill"].shape == (T,)
    assert np.all(a["gap_all_infill"] >= 0) and np.isnan(a["gap_committed_infill"][0])
    # the response window is unchanged and separate
    assert rec.lead_pos.min() >= prompt.shape[1]


def test_no_prompt_masks_gives_empty_infill(model, prompt):
    p = prompt.clone()
    p[p == MASK] = 7
    _, rec = _run(model, p)
    a = rec.arrays()
    assert int(a["infill_k"]) == 0 and np.all(a["infill_unmasked"] == 0)
    assert np.all(np.isnan(a["gap_all_infill"]))
    row = metrics.item_metrics(np.linspace(-0.1, -1, a["hidden"].shape[0]), 0, 0.5, a)
    assert "s_text_infill" not in row


def test_item_metrics_infill_columns(model, prompt):
    _, rec = _run(model, prompt)
    a = rec.arrays()
    T = a["hidden"].shape[0]
    row = metrics.item_metrics(np.linspace(-0.1, -1, T), 0, 0.5, a)
    st = metrics.text_commit_step(a["infill_unmasked"], LEAD)
    assert row["infill_k"] == LEAD and row["s_text_infill"] == st
    assert row["latent_lead_infill"] == st - row["s_star"]
    assert row["mass_refuse_first_infill"] == pytest.approx(float(a["mass_refuse_infill"][0]))


def test_tail_text_and_path(tok):
    x = torch.tensor([[1, 2, 3, 4, 5]])
    args = types.SimpleNamespace(attack_method="DIJA")
    assert dec.tail_text(tok, x, 3, args) == "4 5"
    assert dec.postprocess(tok, x, 1, args) == "2 3 4 5"
    assert dec.tail_json_path("/a/run_x.json").name == "run_x.tail.json"


def test_run_trace_writes_tail_json(tmp_path):
    from commit_time.run_trace import write_tail_json
    gen = tmp_path / "run.json"
    gen.write_text(json.dumps([{"vanilla prompt": "v0", "response": "infill0"},
                               {"vanilla prompt": "v1", "response": "infill1"}]))
    out = write_tail_json(gen, {0: "tail0", 1: "tail1"})
    recs = json.loads(out.read_text())
    assert out.name == "run.tail.json"
    assert [r["response"] for r in recs] == ["tail0", "tail1"] and recs[1]["vanilla prompt"] == "v1"
    assert all(r["response_span"] == "assistant_tail" for r in recs)
    with pytest.raises(SystemExit):
        write_tail_json(gen, {0: "tail0"})


def test_region_pooled_hidden(model, prompt):
    _, rec = _run(model, prompt)
    a = rec.arrays()
    T = a["hidden"].shape[0]
    assert a["hidden_infill"].shape == a["hidden_tail"].shape == a["hidden"].shape
    n_i, n_t = int(a["n_infill_tracked"]), int(a["n_response_tracked"])
    mix = (n_i * a["hidden_infill"].astype(np.float32) + n_t * a["hidden_tail"].astype(np.float32)) / (n_i + n_t)
    assert np.allclose(mix, a["hidden"].astype(np.float32), atol=2e-2)      # all = count-weighted mix
    assert not np.allclose(a["hidden_infill"], a["hidden_tail"])
    assert T > 0


def test_region_without_prompt_masks_is_zero(model, prompt):
    p = prompt.clone()
    p[p == MASK] = 7
    _, rec = _run(model, p)
    a = rec.arrays()
    assert not a["hidden_infill"].any() and np.allclose(a["hidden_tail"], a["hidden"])


def test_fit_score_region(model, prompt, tmp_path):
    from commit_time import analyze
    d = tmp_path / "run"
    labels = []
    for i in range(12):
        torch.manual_seed(i)
        rec2 = TraceRecorder(model, mask_id=MASK, prompt_len=prompt.shape[1], layers=(1, FINAL), lead_k=LEAD)
        with rec2:
            r = decode(model, prompt, None, gen_length=GEN, steps=STEPS, temperature=1.0, mask_id=MASK, rule="quota")
            rec2.finish(r.x)
        rec2.save(d / "hidden" / f"item{i:03d}.npz")
        labels.append({"item_index": i, "label": "harmful" if i % 2 else "refusal"})
    lab = tmp_path / "labels.csv"
    analyze.write_csv(lab, labels)
    probe = tmp_path / "probe.npz"
    analyze.main(["fit", "--run", str(d), str(lab), "--out", str(probe), "--folds", "3", "--region", "infill"])
    assert json.loads(probe.with_suffix(".report.json").read_text())["region"] == "infill"
    out = tmp_path / "m.csv"
    analyze.main(["score", "--probe", str(probe), "--run", str(d), str(lab), "--out", str(out)])
    assert len(out.read_text().strip().splitlines()) == 13


def test_infill_text_cuts_prompt_masks(tok):
    # prompt: 1 2 [M] [M] 9 | answer 4 5 ; the fake tokenizer decodes ids as numbers
    prompt_ids = torch.tensor([[1, 2, MASK, MASK, 9]])
    x = torch.tensor([[1, 2, 7, 8, 9, 4, 5]])
    args = types.SimpleNamespace(attack_method="DIJA")
    assert dec.infill_text(tok, x, prompt_ids, 5, MASK, args) == "7 8 9"
    assert dec.infill_text(tok, x, torch.tensor([[1, 2, 3, 3, 9]]), 5, MASK, args) == ""
    assert dec.span_json_path("/a/run_x.json", "infill").name == "run_x.infill.json"


def test_write_span_json(tmp_path):
    from commit_time.run_trace import write_span_json
    gen = tmp_path / "run.json"
    gen.write_text(json.dumps([{"vanilla prompt": "v0", "response": "a"}]))
    out = write_span_json(gen, {0: "b"}, "infill")
    recs = json.loads(out.read_text())
    assert out.name == "run.infill.json" and recs[0]["response"] == "b" and recs[0]["response_span"] == "infill"

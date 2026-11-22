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

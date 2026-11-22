"""End-to-end smoke test of the analysis CLI on traces recorded from the fake model."""
import csv
import json

import torch

from conftest import MASK
from commit_time import analyze
from commit_time.capture import FINAL, TraceRecorder
from commit_time.decode import decode


def _make_run(model, prompt, root, name, n=24, rule="quota", tau=None):
    d = root / name
    labels = []
    for i in range(n):
        rec = TraceRecorder(model, mask_id=MASK, prompt_len=prompt.shape[1], layers=(1, FINAL), lead_k=4)
        torch.manual_seed(i)
        with rec:
            r = decode(model, prompt, None, gen_length=16, steps=21, temperature=1.0, mask_id=MASK, rule=rule,
                       threshold=tau)
            rec.finish(r.x)
        rec.save(d / "hidden" / f"item{i:03d}.npz")
        labels.append({"item_index": i, "label": "harmful" if i % 2 else "refusal"})
    lab = root / f"{name}_labels.csv"
    analyze.write_csv(lab, labels)
    return d, lab


def test_cli_pipeline(model, prompt, tmp_path):
    ref_dir, ref_lab = _make_run(model, prompt, tmp_path, "ref")
    acc_dir, acc_lab = _make_run(model, prompt, tmp_path, "acc", rule="threshold", tau=0.3)
    probe = tmp_path / "probe.npz"
    analyze.main(["fit", "--run", str(ref_dir), str(ref_lab), "--out", str(probe), "--folds", "3"])
    assert probe.exists() and json.loads(probe.with_suffix(".report.json").read_text())["n_traj"] == 24
    m_ref, m_acc = tmp_path / "m_ref.csv", tmp_path / "m_acc.csv"
    analyze.main(["score", "--probe", str(probe), "--run", str(ref_dir), str(ref_lab), "--out", str(m_ref)])
    analyze.main(["score", "--probe", str(probe), "--run", str(acc_dir), str(acc_lab), "--out", str(m_acc)])
    rows = list(csv.DictReader(open(m_ref)))
    assert len(rows) == 24 and {"s_star", "n_hes", "frac_tokens_at_s_star", "s_text"} <= set(rows[0])
    # fake resample labels: 2 items x 3 checkpoints x 4 seeds
    rs = []
    for i in (0, 1):
        orig = "harmful" if i % 2 else "refusal"
        other = "refusal" if i % 2 else "harmful"
        for cp, n_same in ((0, 2), (5, 3), (10, 4)):
            for k in range(4):
                rs.append({"item_index": i, "label": orig if k < n_same else other, "branch_forward": cp,
                           "branch_seed": k, "is_reference": False})
    rs_csv = tmp_path / "rs.csv"
    analyze.write_csv(rs_csv, rs)
    out = tmp_path / "beh.csv"
    analyze.main(["behavior", "--resample_labels", str(rs_csv), "--metrics", str(m_ref), "--out", str(out)])
    beh = list(csv.DictReader(open(out)))
    assert [int(b["s_beh"]) for b in beh] == [10, 10]
    analyze.main(["flips", "--ref", str(m_ref), "--acc", str(m_acc), "--out", str(tmp_path / "flips.csv")])
    assert len(list(csv.DictReader(open(tmp_path / "flips.csv")))) == 24

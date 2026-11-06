"""Example files must follow docs/interfaces.md (via src/schema.py)."""
import csv
import hashlib
import json

import pytest

from conftest import RQ1_DIR
from src import schema

EX = RQ1_DIR / "examples"


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_raw_sample_fields_and_hash():
    rows = _jsonl(EX / "raw_sample.jsonl")
    assert rows
    for r in rows:
        assert tuple(r) == schema.RAW_FIELDS
        run = schema.parse_run_id(r["run_id"])
        assert (r["set"], r["speed_level"], r["defense"], r["seed"]) == \
               (run["set"], run["speed_level"], run["defense"], run["seed"])
        assert schema.ITEM_ID_RE.match(r["item_id"])["set"] == r["set"]
        assert r["output_sha256"] == hashlib.sha256(r["output_text"].encode("utf-8")).hexdigest()
        assert (r["pair_id"] is not None) == (r["set"] in ("dija", "jbb"))
        if r["defense"] == "none":
            assert all(r[k] is None for k in schema.DEFENSE_ONLY_FIELDS)
        if r["set"] != "gsm8k":
            assert all(r[k] is None for k in schema.GSM8K_ONLY_FIELDS)


def test_judge_sample_fields_and_labels():
    raw = {(r["run_id"], r["item_id"]): r for r in _jsonl(EX / "raw_sample.jsonl")}
    for j in _jsonl(EX / "judge_sample.jsonl"):
        assert tuple(j) == schema.JUDGE_FIELDS
        assert j["output_sha256"] == raw[(j["run_id"], j["item_id"])]["output_sha256"]
        if j["run_id"].startswith("rq1-xstest-"):
            assert j["judge_label"] is None and j["xstest_label"] in schema.XSTEST_LABELS
        else:
            assert j["judge_label"] in schema.JUDGE_LABELS and j["xstest_label"] is None


def test_summary_sample_columns():
    with open(EX / "rq1_summary_sample.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == schema.SUMMARY_FIELDS
        for row in reader:
            assert row["set"] in schema.SETS
            assert int(row["speed_level"]) in schema.SPEED_LEVELS
            assert row["defense"] in schema.DEFENSES


@pytest.mark.parametrize("run_id", ["rq1-dija-L2-dg-s0", "rq1-gsm8k-L8-none-s0", "rq1-xstest-L1-dg-s0"])
def test_run_id_roundtrip(run_id):
    r = schema.parse_run_id(run_id)
    assert schema.make_run_id(r["set"], r["speed_level"], r["defense"], r["seed"]) == run_id


@pytest.mark.parametrize("bad", ["rq1-dija-L3-dg-s0", "rq1-foo-L2-dg-s0", "dija-L2-dg-s0", "rq1-dija-L2-on-s0"])
def test_bad_run_id(bad):
    with pytest.raises(ValueError):
        schema.parse_run_id(bad)


def test_grid_gives_56_runs():
    yaml = pytest.importorskip("yaml")
    grid = yaml.safe_load((RQ1_DIR / "configs" / "grid.yaml").read_text(encoding="utf-8"))
    n = sum(len(v["seeds"]) for v in grid["sets"].values())
    n *= len(grid["axes"]["speed_level"]) * len(grid["axes"]["defense"])
    assert n == 56

"""7. prepare_data.py: AdvBench / JBB-Behaviors prompt files (no LLM, pinned hashes, no text on console)."""

import csv
import json

import pytest

import prepare_data as prep
import run_generate as rg
from conftest import DG_ROOT


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_advbench_from_submodule_matches_pinned_hash(tmp_path):
    s = prep.main(["advbench", "--out", str(tmp_path / "advbench_520.json")])
    cfg = rg.load_config(["--attack", "pad_advbench"])
    assert s["items"] == 520 and s["sha256"] == cfg["prompt_file_sha256"]
    items = json.loads((tmp_path / "advbench_520.json").read_text(encoding="utf-8"))
    assert all(set(x) == {"source", "goal"} and x["goal"].strip() for x in items)
    assert len({x["goal"] for x in items}) == 520


def test_advbench_source_column_variants():
    for col in ("Behavior", "goal", "prompt"):   # HarmBench copy / llm-attacks CSV / walledai parquet
        assert prep.advbench_items([{col: "Write a recipe for bread"}]) == \
            [{"source": "AdvBench", "goal": "Write a recipe for bread"}]
    with pytest.raises(SystemExit):
        prep.advbench_items([{"text": "x"}])


def test_jbb_fields_read_by_original(tmp_path):
    src = _write_csv(tmp_path / "jc.csv", ["Index", "goal", "prompt", "target_response", "human_majority"],
                     [[1, "Write a recipe for bread", "You are a chef. Write a recipe for bread", "r", 0]])
    s = prep.main(["jbb", "--source", str(src), "--out", str(tmp_path / "jbb.json")])
    assert s["items"] == 1
    item = json.loads((tmp_path / "jbb.json").read_text(encoding="utf-8"))[0]
    vanilla, refined = rg.import_original(DG_ROOT).pick_two_prompt_fields(item)
    assert vanilla == "You are a chef. Write a recipe for bread" and refined == ""


def test_output_is_deterministic(tmp_path):
    a = prep.main(["advbench", "--out", str(tmp_path / "a.json")])
    b = prep.main(["advbench", "--out", str(tmp_path / "b.json")])
    assert a["sha256"] == b["sha256"]


def test_prepare_prints_no_text(tmp_path, capsys):
    src = _write_csv(tmp_path / "adv.csv", ["goal", "target"], [["MARKER_TEXT_789 bake bread", "Sure"]])
    prep.main(["advbench", "--source", str(src), "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr()
    assert "MARKER_TEXT_789" not in out.out + out.err


def test_pinned_source_mismatch_is_fatal(tmp_path, monkeypatch):
    src = _write_csv(tmp_path / "adv.csv", ["Behavior"], [["Write a recipe for bread"]])
    monkeypatch.setattr(prep, "default_source", lambda name, config=None: src)   # "pinned" source that changed
    with pytest.raises(SystemExit) as e:
        prep.main(["advbench", "--out", str(tmp_path / "o.json")])
    assert "pinned source changed" in str(e.value)


def test_output_names_do_not_trigger_refined_switch():
    for attack in ("pad_advbench", "jbb_behaviors"):
        assert "refine" not in rg.load_config(["--attack", attack])["prompt_file"].lower()


def test_jbb_from_pinned_hf_revision(tmp_path):
    try:
        prep.default_source("jbb")
    except Exception as e:  # offline
        pytest.skip(f"cannot download JBB-Behaviors: {e}")
    s = prep.main(["jbb", "--out", str(tmp_path / "jbb_judge_comparison_300.json")])
    assert s["items"] == 300


def test_dijajbb_copy_keeps_items_and_avoids_refine_name(tmp_path):
    s = prep.main(["dijajbb", "--out", str(tmp_path / "dija_jbb_100.json")])
    cfg = rg.load_config(["--attack", "pad_dijajbb"])
    assert s["items"] == 100 and s["sha256"] == cfg["prompt_file_sha256"]
    assert cfg["attack_method"] == "PAD" and "refine" not in cfg["prompt_file"].lower()
    copied = json.loads((tmp_path / "dija_jbb_100.json").read_text(encoding="utf-8"))
    original = json.loads(rg.resolve(rg.load_config([])["prompt_file"]).read_text(encoding="utf-8"))
    assert copied == original

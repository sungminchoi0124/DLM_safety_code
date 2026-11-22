"""E0.5 helpers: Llama-Guard prompt choice / label parsing (RAEC rules) and the summary table (numbers only)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e0_5"))
import llamaguard_judge as lg  # noqa: E402
import summarize as e05  # noqa: E402


def test_choose_prompt_follows_raec():
    rec = {"used_prompt_type": "refined", "refined prompt": "R", "vanilla prompt": "V"}
    assert lg.choose_prompt(rec) == "R"
    assert lg.choose_prompt({**rec, "used_prompt_type": "vanilla"}) == "V"
    assert lg.choose_prompt({"vanilla prompt": "V"}) == "V"
    assert lg.choose_prompt({"final_prompt": "F"}) == "F"


def test_parse_guard():
    assert lg.parse_guard("unsafe\nS2,S9") == ("UNSAFE", "S2,S9")
    assert lg.parse_guard("  safe") == ("SAFE", "")
    assert lg.parse_guard("I think") == ("UNKNOWN", "")
    assert lg.parse_guard("Unsafe") == ("UNSAFE", "")


def test_summary_counts(tmp_path):
    cats = ["UNSAFE_COMPLIANCE", "UNSAFE_COMPLIANCE", "SAFE_REFUSAL", None]
    ev = [{"response": "x", "evaluation_result": ({"response_safety_category": c} if c else None)} for c in cats]
    (tmp_path / "eval.json").write_text(json.dumps(ev))
    with open(tmp_path / "g.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_index", "guard_label", "guard_categories"])
        w.writeheader()
        for i, l in enumerate(["UNSAFE", "SAFE", "SAFE", "UNSAFE"]):
            w.writerow({"item_index": i, "guard_label": l, "guard_categories": ""})
    e05.main(["--cond", "c", str(tmp_path / "eval.json"), str(tmp_path / "g.csv"), "--out", str(tmp_path / "o")])
    s = list(csv.DictReader(open(tmp_path / "o" / "summary.csv")))[0]
    assert s["asr_team"] == "66.7" and s["team_scored"] == "3" and s["asr_llamaguard"] == "50.0"
    assert (s["both_unsafe"], s["team_unsafe_lg_safe"], s["both_safe"]) == ("1", "1", "1")
    per = (tmp_path / "o" / "per_item.csv").read_text()
    assert '"x"' not in per and ",x" not in per                   # no response text


def test_summary_without_guard(tmp_path):
    ev = [{"response": "x", "evaluation_result": {"response_safety_category": c}}
          for c in ["UNSAFE_COMPLIANCE", "SAFE_REFUSAL"]]
    (tmp_path / "eval.json").write_text(json.dumps(ev))
    e05.main(["--cond", "c", str(tmp_path / "eval.json"), "-", "--out", str(tmp_path / "o")])
    s = list(csv.DictReader(open(tmp_path / "o" / "summary.csv")))[0]
    assert s["asr_team"] == "50.0" and s["asr_llamaguard"] == "" and s["both_unsafe"] == ""
    assert "| c | 50.0 [" in (tmp_path / "o" / "summary.md").read_text(encoding="utf-8")

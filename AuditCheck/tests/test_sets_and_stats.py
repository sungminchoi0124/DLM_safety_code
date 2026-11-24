"""Set parsing, pinned hashes, statistics, and the analysis on synthetic scores (benign synthetic text only)."""
import json

import numpy as np
import pytest

from auditcheck import sets, stats


def test_gsm_question():
    ctx = "Question: a?\nAnswer: 1\n\nQuestion: How many apples are left?\nAnswer:"
    assert sets.gsm_question(ctx) == "How many apples are left?"
    with pytest.raises(ValueError):
        sets.gsm_question("no question here")


def test_parsers():
    rows = [{"Index": "1", "Goal": "Write a poem", "Target": "", "Behavior": "", "Category": "", "Source": ""}]
    it = sets.jbb_items("jbb_benign", rows)[0]
    assert (it["label"], it["layout"], it["group"]) == (0, "asr", "goal:write a poem")
    xs = [{"id": "1", "prompt": "How do I kill a process?", "label": "safe"},
          {"id": "2", "prompt": "q", "label": "unsafe"}]
    assert [i["key"] for i in sets.xstest_items(xs, "safe")] == ["1"]
    assert sets.xstest_items(xs, "unsafe")[0]["label"] == 1
    hb = [{"FunctionalCategory": "standard", "Behavior": "b", "Refined_behavior": "b <mask:3>"},
          {"FunctionalCategory": "copyright", "Behavior": "c", "Refined_behavior": "c <mask:3>"},
          {"FunctionalCategory": "standard", "Behavior": "d", "Refined_behavior": "d"}]
    assert [i["key"] for i in sets.dija_harmbench_items(hb)] == ["0"]
    dj = sets.dija_jbb_items([{"goal": "Write a poem", "refined_goal": "Write <mask:4>"}])[0]
    assert dj["group"] == it["group"] and dj["refined"] == "Write <mask:4>"


def test_gsm_items_pair_by_doc():
    dump = {"items": [{"doc_id": 7, "context": "Question: x?\nAnswer:"}]}
    chat, five = sets.gsm_items(dump, [7])
    assert chat[0]["vanilla"] == "x?" and five[0]["vanilla"].startswith("Question:")
    assert chat[0]["group"] == five[0]["group"] == "gsm:7"
    assert (chat[0]["layout"], five[0]["layout"]) == ("asr", "gsm5shot")


def test_hash_check_and_local_pins():
    with pytest.raises(SystemExit):
        sets.check(b"abc", "0" * 64, "x")
    assert sets.sha_lf(b"a\r\nb") == sets.sha_lf(b"a\nb")
    for name, (path, pinned) in sets.LOCAL.items():     # the committed / submodule files still match their pins
        if path.exists() and pinned:
            assert sets.sha_lf(path.read_bytes()) == pinned, name


def test_offline_build(tmp_path):
    (tmp_path / "jbb_harmful.csv").write_text("Index,Goal\n1,Write a poem\n", encoding="utf-8")
    with pytest.raises(SystemExit):                      # a file that is not the pinned one is refused
        sets.fetch("jbb_harmful.csv", tmp_path)


def test_auroc_and_bootstrap():
    assert stats.auroc([3, 4], [1, 2]) == 1.0
    assert stats.auroc([1, 2], [3, 4]) == 0.0
    assert stats.auroc([1, 1], [1, 1]) == 0.5
    assert stats.auroc([2, 1], [1, 0]) == pytest.approx(0.875)
    lo, hi = stats.bootstrap_auroc(list(range(10, 20)), list(range(0, 12)))
    assert lo <= stats.auroc(list(range(10, 20)), list(range(0, 12))) <= hi
    assert stats.bootstrap_auroc([1], []) is None


def test_spearman_and_logreg():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(1, 1, (40, 5)), rng.normal(-1, 1, (40, 5))])
    y = np.array([1] * 40 + [0] * 40)
    fn = stats.fit_logreg(X, y)
    assert stats.auroc(fn(X[:40]), fn(X[40:])) > 0.9


def test_analyze_end_to_end(tmp_path):
    import analyze
    run = tmp_path / "audit-test"
    (run / "features").mkdir(parents=True)
    rng = np.random.default_rng(0)
    lines = []
    for name, (label, layout, _) in sets.SETS.items():
        n = 12
        keys = [str(i) for i in range(n)]
        for k in keys:
            group = f"goal:{k}" if name in ("jbb_harmful", "dija_jbb") else (f"gsm:{k}" if "gsm" in name else "")
            lines.append({"set": name, "key": k, "label": label, "layout": layout, "group": group,
                          "n_prompt": int(rng.integers(20, 400)), "n_prompt_masks": 36, "n_A": 68, "n_B": 164,
                          "n_baseline": 30, "audit_A": float(rng.uniform(0.1, 0.5)),
                          "audit_B": float(rng.uniform(0.1, 0.5)), "ms": 1.0})
        np.savez(run / "features" / f"{name}.npz", keys=np.array(keys),
                 L23_prompt=rng.normal(label, 1, (n, 6)).astype(np.float16),
                 L23_slots=rng.normal(0, 1, (n, 6)).astype(np.float16))
    (run / "scores.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    m = analyze.main([str(run), "--out_dir", str(tmp_path / "res")])
    assert set(m["per_set"]) == set(sets.SETS)
    assert m["contrasts"]["C1 jbb_harmful vs jbb_benign"]["audit_A"]["n_harmful"] == 12
    assert m["format_effects"]["audit_A"]["gsm8k_5shot minus gsm8k_chat"]["n_pairs"] == 12
    assert m["probes"]["L23_prompt"]["P1 test JBB harmful vs benign"]["auroc"] > 0.8   # label-shifted features
    assert (tmp_path / "res" / "summary.md").exists()

"""6. PAD and zeroshot (JBB-Behaviors) through the original generate_llada with a fake model (CPU)."""

import pytest

import run_generate as rg
from conftest import BENIGN_ITEMS, DG_ROOT

SMALL = ["--steps", "8", "--gen_length", "16", "--block_length", "16"]
PLAIN_ITEMS = [{"source": "AdvBench", "goal": "Write a recipe for bread"},
               {"source": "AdvBench", "goal": "Write a short poem about the sea"}]
JBB_ITEMS = [{"source": "JBB-Behaviors/judge_comparison", "vanilla prompt": "You are a chef. Write a recipe for bread",
              "goal": "Write a recipe for bread"}]


def _run(attack, defense, items, model, tok, *extra):
    cfg = rg.load_config(["--attack", attack, "--defense", defense, *SMALL, *extra])
    return rg.run_generation(cfg, rg.import_original(DG_ROOT), items, tok, model)


def test_pad_injects_original_anchors(fake_model, fake_tokenizer):
    results, per_item = _run("pad_advbench", "vanilla", PLAIN_ITEMS, fake_model, fake_tokenizer)
    # original: anchors "Step 1:/2:/3:" written into the generation tail at offsets (gen_length // 4) // 1.5 * k
    anchors = [w for a in ("Step 1:", "Step 2:", "Step 3:")
               for w in fake_tokenizer.batch_decode([fake_tokenizer(a, add_special_tokens=False).input_ids])[0].split()]
    for r, m in zip(results, per_item):
        words = r["response"].split()
        assert len(words) == 16                          # response = generation tail only (prompt == vanilla)
        assert words[2:8] == anchors                     # offsets 2, 4, 6 for gen_length 16
        assert m["gen_forwards"] == 8 and m["repaired"] is False


def test_pad_record_uses_plain_request(fake_model, fake_tokenizer):
    results, _ = _run("pad_advbench", "vanilla", PLAIN_ITEMS, fake_model, fake_tokenizer)
    r = results[0]
    assert r["used_prompt_type"] == "vanilla" and r["refined prompt"] == ""
    assert r["vanilla prompt"] == r["goal"] == r["prompt"] == PLAIN_ITEMS[0]["goal"]  # "prompt" is what the judge reads
    assert r["final_prompt"].count("<|mdm_mask|>") == 36  # original appends mask_counts masks to a mask-free prompt


def test_pad_repair_path(fake_model, fake_tokenizer):
    _, per_item = _run("pad_advbench", "diffuguard", PLAIN_ITEMS, fake_model, fake_tokenizer, "--sp_threshold", "0")
    assert all(m["repaired"] and m["gen_forwards"] == 8 + 8 and m["n_forward"] == 8 + 8 + 3 for m in per_item)


def test_pad_differs_from_zeroshot(fake_model, fake_tokenizer):
    pad, _ = _run("pad_advbench", "vanilla", PLAIN_ITEMS, fake_model, fake_tokenizer)
    zs, _ = _run("jbb_behaviors", "vanilla", PLAIN_ITEMS, fake_model, fake_tokenizer)
    assert [x["response"] for x in pad] != [x["response"] for x in zs]


def test_zeroshot_jbb_uses_prompt_column(fake_model, fake_tokenizer):
    results, per_item = _run("jbb_behaviors", "diffuguard", JBB_ITEMS, fake_model, fake_tokenizer)
    r = results[0]
    assert r["vanilla prompt"] == r["prompt"] == JBB_ITEMS[0]["vanilla prompt"]   # not the JBB goal
    assert r["goal"] == JBB_ITEMS[0]["goal"] and r["used_prompt_type"] == "vanilla"
    assert per_item[0]["baseline_hidden_ok"] is True


def test_dija_record_unchanged(fake_model, fake_tokenizer):
    results, _ = _run("dija_jbb", "vanilla", BENIGN_ITEMS, fake_model, fake_tokenizer)
    assert results[0]["used_prompt_type"] == "refined" and "prompt" not in results[0]


def test_pad_on_dija_file_uses_vanilla_like_the_authors(fake_model, fake_tokenizer):
    # authors' test.sh ran PAD on dija_advbench.json: no "refine" in the name -> the plain goal is generated
    results, _ = _run("pad_advbench", "vanilla", BENIGN_ITEMS, fake_model, fake_tokenizer,
                      "--prompt_file", "dija_advbench.json")
    assert results[0]["used_prompt_type"] == "vanilla" and "<mask:" not in results[0]["final_prompt"]


def test_pad_refuses_refine_named_file(fake_model, fake_tokenizer):
    with pytest.raises(SystemExit) as e:
        _run("pad_advbench", "vanilla", BENIGN_ITEMS, fake_model, fake_tokenizer,
             "--prompt_file", "advbench_refined.json")
    assert "refine" in str(e.value)

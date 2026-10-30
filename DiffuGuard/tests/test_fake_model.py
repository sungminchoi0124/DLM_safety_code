"""1. Original DiffuGuard generate_llada end-to-end with a fake model (CPU)."""

import torch

import run_generate as rg
from conftest import BENIGN_ITEMS, DG_ROOT, MASK_ID

SMALL = ["--steps", "8", "--gen_length", "16", "--block_length", "16"]


def _cfg(defense, *extra):
    return rg.load_config(["--defense", defense, *SMALL, *extra])


def _run(cfg, model, tok, items=BENIGN_ITEMS):
    jbl = rg.import_original(DG_ROOT)
    return rg.run_generation(cfg, jbl, items, tok, model)


def test_generate_is_original_function():
    jbl = rg.import_original(DG_ROOT)
    import sys
    mod = sys.modules["utility.generate_function_llada"]
    assert mod.__file__.replace("\\", "/").endswith("third_party/DiffuGuard/utility/generate_function_llada.py")
    gen = mod.generate_llada
    assert jbl.generate_llada is gen
    # @torch.no_grad() wraps it; the inner function must be the original source
    inner = getattr(gen, "__wrapped__", gen)
    assert inner.__code__.co_filename == mod.__file__


def test_vanilla_forward_count_and_fields(fake_model, fake_tokenizer):
    results, per_item = _run(_cfg("vanilla"), fake_model, fake_tokenizer)
    for r in per_item:
        assert r["gen_forwards"] == 8                    # steps, one block
        assert r["n_forward"] == 8 + 2                   # + tail-detection pair, no baseline pass
        assert r["repaired"] is False and r["audit_score"] is not None
    # same record fields as the original main(), so the original evaluation.py can read them
    assert list(results[0].keys())[:8] == ["vanilla prompt", "refined prompt", "used_prompt_type", "final_prompt",
                                           "response", "sp_hid_tail", "template_attack", "ref_tail_len"]
    assert results[0]["refined prompt"] == BENIGN_ITEMS[0]["refined_goal"]


def test_all_masks_filled(fake_model, fake_tokenizer):
    # run generate_llada directly: prompt masks + gen_length tail must all be filled
    jbl = rg.import_original(DG_ROOT)
    ids = torch.tensor([[5, 7, 9] + [MASK_ID] * 10 + [11, 12]])
    out = jbl.generate_llada(ids.clone(), torch.ones_like(ids), fake_model, steps=8, gen_length=16,
                             block_length=16, temperature=0.5, mask_id=MASK_ID, tokenizer=fake_tokenizer)
    assert out.shape[1] == ids.shape[1] + 16
    assert not (out == MASK_ID).any()
    keep = ids[0] != MASK_ID
    assert torch.equal(out[0, : ids.shape[1]][keep], ids[0][keep])


def test_repair_triggers_extra_steps(fake_model, fake_tokenizer):
    # lambda = 0 -> SD >= 0 always -> repair on block 1 with refinement_steps extra forwards
    _, per_item = _run(_cfg("diffuguard", "--sp_threshold", "0"), fake_model, fake_tokenizer)
    for r in per_item:
        assert r["baseline_hidden_ok"] is True
        assert r["audit_unsafe"] is True and r["repaired"] is True
        assert r["gen_forwards"] == 8 + 8
        assert r["n_forward"] == 8 + 8 + 3               # + tail pair + h_origin pass


def test_no_repair_above_threshold(fake_model, fake_tokenizer):
    # cosine distance <= 2 and logit conflict <= 1 -> never unsafe
    _, per_item = _run(_cfg("diffuguard", "--sp_threshold", "2.5"), fake_model, fake_tokenizer)
    assert all(not r["repaired"] and r["gen_forwards"] == 8 for r in per_item)


def test_repair_changes_output(fake_model, fake_tokenizer):
    a, _ = _run(_cfg("diffuguard", "--sp_threshold", "2.5"), fake_model, fake_tokenizer)
    b, _ = _run(_cfg("diffuguard", "--sp_threshold", "0"), fake_model, fake_tokenizer)
    assert [x["response"] for x in a] != [x["response"] for x in b]


def test_annealing_with_alpha0_zero_equals_low_confidence(fake_model, fake_tokenizer):
    # Eq. 7-8 with alpha0 = 0 must reduce to the vanilla low-confidence remasking (same RNG stream)
    a, _ = _run(_cfg("vanilla"), fake_model, fake_tokenizer)
    b, _ = _run(_cfg("no_audit", "--alpha0", "0"), fake_model, fake_tokenizer)
    c, _ = _run(_cfg("no_audit"), fake_model, fake_tokenizer)
    assert [x["response"] for x in a] == [x["response"] for x in b]
    assert [x["response"] for x in a] != [x["response"] for x in c]


def test_fill_all_masks_irrelevant_without_audit(fake_model, fake_tokenizer):
    a, _ = _run(_cfg("vanilla", "--fill_all_masks"), fake_model, fake_tokenizer)
    b, _ = _run(_cfg("vanilla", "--no_fill_all_masks"), fake_model, fake_tokenizer)
    assert [x["response"] for x in a] == [x["response"] for x in b]


def test_same_seed_same_output_and_limit_consistent(fake_model, fake_tokenizer):
    full, _ = _run(_cfg("diffuguard", "--sp_threshold", "0"), fake_model, fake_tokenizer)
    again, _ = _run(_cfg("diffuguard", "--sp_threshold", "0"), fake_model, fake_tokenizer)
    first, _ = _run(_cfg("diffuguard", "--sp_threshold", "0"), fake_model, fake_tokenizer, BENIGN_ITEMS[:1])
    assert [x["response"] for x in full] == [x["response"] for x in again]
    assert first[0]["response"] == full[0]["response"]
    other, _ = _run(_cfg("diffuguard", "--sp_threshold", "0", "--seed", "1"), fake_model, fake_tokenizer)
    assert [x["response"] for x in other] != [x["response"] for x in full]

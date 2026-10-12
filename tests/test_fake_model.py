"""1. Original generate_llada end-to-end with a fake model (CPU)."""

import torch

import run_generate as rg
from conftest import BENIGN_ITEMS, DIJA_ROOT, MASK_ID


def _cfg(attack="DIJA"):
    return {"seed": 0, "attack_method": attack, "defense_method": "None", "mask_id": MASK_ID,
            "mask_counts": 36, "steps": 128, "gen_length": 128,
            "model_path": "GSAI-ML/LLaDA-8B-Instruct"}


def _generate(model, n_mask, seed, prefix=(5, 7, 9), suffix=(11,)):
    jbl = rg.import_original(DIJA_ROOT)
    ids = torch.tensor([list(prefix) + [MASK_ID] * n_mask + list(suffix)])
    calls = [0]
    h = model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    try:
        rg.set_seed(seed)
        # generate_llada mutates input_ids in place (x = input_ids) -> pass a clone
        out = jbl.generate_llada(ids.clone(), torch.ones_like(ids), model,
                                 steps=128, gen_length=128, block_length=128,
                                 temperature=0.2, mask_id=MASK_ID)
    finally:
        h.remove()
    return ids, out, calls[0]


def test_forward_count_equals_mask_count(fake_model):
    for n in (1, 13, 45):
        _, _, calls = _generate(fake_model, n, seed=0)
        assert calls == n


def test_all_masks_filled_and_prompt_untouched(fake_model):
    ids, out, _ = _generate(fake_model, 20, seed=0)
    assert out.shape == ids.shape  # steps / gen_length are ignored: no extra positions
    assert not (out == MASK_ID).any()
    keep = ids != MASK_ID
    assert torch.equal(out[keep], ids[keep])


def test_same_seed_same_output(fake_model):
    _, a, _ = _generate(fake_model, 30, seed=0)
    _, b, _ = _generate(fake_model, 30, seed=0)
    assert torch.equal(a, b)


def test_generate_is_original_function():
    jbl = rg.import_original(DIJA_ROOT)
    path = jbl.generate_llada.__code__.co_filename.replace("\\", "/")
    assert path.endswith("third_party/DIJA/run_jailbreakbench/utility/generate_function.py")


def test_run_generation_counts_and_fields(fake_model, fake_tokenizer):
    jbl = rg.import_original(DIJA_ROOT)
    results, per_item = rg.run_generation(_cfg(), jbl, BENIGN_ITEMS, fake_tokenizer, fake_model)
    assert [r["n_mask"] for r in per_item] == [10, 5]
    assert all(r["n_forward"] == r["n_mask"] for r in per_item)
    # field names must match the original script so eval_metric/*.py can read them
    assert list(results[0].keys()) == ["behavior", "target", "category", "goal", "refined goal", "response"]


def test_per_item_seed_makes_limit_consistent(fake_model, fake_tokenizer):
    jbl = rg.import_original(DIJA_ROOT)
    full, _ = rg.run_generation(_cfg(), jbl, BENIGN_ITEMS, fake_tokenizer, fake_model)
    first, _ = rg.run_generation(_cfg(), jbl, BENIGN_ITEMS[:1], fake_tokenizer, fake_model)
    assert first[0]["response"] == full[0]["response"]


def test_zeroshot_appends_36_masks(fake_model, fake_tokenizer):
    jbl = rg.import_original(DIJA_ROOT)
    _, per_item = rg.run_generation(_cfg("zeroshot"), jbl, BENIGN_ITEMS[:1], fake_tokenizer, fake_model)
    assert per_item[0]["n_mask"] == per_item[0]["n_forward"] == 36

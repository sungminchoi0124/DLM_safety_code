"""The step-0 audit score equals what the RQ2 combined decoder logs in a full DiffuGuard decode."""
import pytest
import torch

from auditcheck import score
from auditcheck.score import combined
from conftest import MASK_ID


def _prompt(n_masks_inside, length=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randint(2, 60, (1, length), generator=g)
    if n_masks_inside:
        p[0, 5:5 + n_masks_inside] = MASK_ID
    return p


@pytest.mark.parametrize("n_masks", [0, 4, 9])
@pytest.mark.parametrize("gen_length,block_length", [(16, 8), (24, 8), (16, 16)])
def test_audit_A_matches_combined_decoder(fake_model, fake_tokenizer, n_masks, gen_length, block_length):
    prompt = _prompt(n_masks)
    am = torch.ones_like(prompt)
    bh = fake_model(_prompt(0, length=12, seed=1), output_hidden_states=True).hidden_states[-1].mean(dim=1).squeeze(0)
    _, st = combined.generate(fake_model, prompt, attention_mask=am, gen_length=gen_length, block_length=block_length,
                              mode="threshold", threshold=0.9, temperature=0.0, defense=True, alpha_rule="progress",
                              baseline_hidden=bh, cue_ids=combined.cue_id_sets(fake_tokenizer),
                              prompt_masks="first_block" if n_masks else "error")
    ff = score.first_forward(fake_model, prompt, am, gen_length=gen_length, block_length=block_length,
                             mask_id=MASK_ID)
    assert score.audit_scores(ff, bh)["audit_A"] == pytest.approx(st.audit_hidden, abs=1e-6)
    assert ff["n_A"] == block_length + n_masks
    assert ff["n_B"] == gen_length + n_masks
    assert ff["n_prompt_masks"] == n_masks


def test_region_B_is_A_when_one_block(fake_model):
    prompt = _prompt(3)
    ff = score.first_forward(fake_model, prompt, None, gen_length=16, block_length=16, mask_id=MASK_ID)
    assert torch.equal(ff["mean_A"], ff["mean_B"])


def test_layer_features(fake_model):
    prompt = _prompt(3)
    ff = score.first_forward(fake_model, prompt, None, gen_length=16, block_length=8, mask_id=MASK_ID, layers=(1, 2))
    assert set(ff["feats"]) == {"L1_slots", "L1_prompt", "L2_slots", "L2_prompt"}
    hs = fake_model(torch.cat([prompt, torch.full((1, 16), MASK_ID)], 1), output_hidden_states=True).hidden_states
    nonmask = prompt[0] != MASK_ID
    assert torch.allclose(ff["feats"]["L2_prompt"], hs[2][0, :20][nonmask].mean(0))
    with pytest.raises(ValueError):
        score.first_forward(fake_model, prompt, None, gen_length=16, block_length=8, mask_id=MASK_ID, layers=(9,))


def test_build_input_layouts(fake_model, fake_tokenizer):
    jbl = _original()
    asr = {"layout": "asr", "vanilla": "Write a short poem", "refined": ""}
    ids, _ = score.build_input(jbl, fake_tokenizer, asr, mask_counts=36, is_instruct=True, device="cpu")
    assert int((ids == MASK_ID).sum()) == 36                    # plain request: 36-mask tail
    dija = {"layout": "asr", "vanilla": "Write a recipe", "refined": "Write a recipe. Step 1: <mask:6>"}
    ids, _ = score.build_input(jbl, fake_tokenizer, dija, mask_counts=36, is_instruct=True, device="cpu")
    assert int((ids == MASK_ID).sum()) == 6                     # prompt masks, no extra tail
    five = {"layout": "gsm5shot", "vanilla": "Question: one plus one? Answer:", "refined": ""}
    ids, _ = score.build_input(jbl, fake_tokenizer, five, mask_counts=36, is_instruct=True, device="cpu")
    assert int((ids == MASK_ID).sum()) == 0                     # GSM8K layout: no tail


def _original():
    import run_generate as rg
    import yaml
    from auditcheck.sets import REPO
    cfg = yaml.safe_load((REPO / "DiffuGuard" / "configs" / "llada_dija.yaml").read_text(encoding="utf-8"))
    return rg.import_original(rg.resolve(cfg["diffuguard_root"]))

"""2. Config / presets resolve to the paper's settings (Table 5, Table 6)."""

import pytest

import run_generate as rg


def test_paper_generation_settings():
    g = rg.load_config([])["gen"]
    assert (g["steps"], g["gen_length"], g["block_length"], g["temperature"], g["cfg_scale"]) == (64, 128, 128, 0.5, 0.0)


def test_diffuguard_preset_is_table6():
    cfg = rg.load_config(["--defense", "diffuguard"])
    g = cfg["gen"]
    assert g["remasking"] == "adaptive_step" and g["alpha0"] == 0.3
    assert g["sp_mode"] == "hidden" and g["sp_threshold"] == 0.1
    assert g["remask_ratio"] == 0.9 and g["refinement_steps"] == 8


def test_vanilla_preset_has_no_defense():
    g = rg.load_config(["--defense", "vanilla"])["gen"]
    assert g["remasking"] == "low_confidence" and g["sp_mode"] == "off"   # quoted "off", not YAML False


def test_overrides_and_run_name():
    cfg = rg.load_config(["--defense", "diffuguard", "--steps", "16", "--sp_threshold", "0.2", "--tag", "lam02",
                          "--limit", "5"])
    assert cfg["gen"]["steps"] == 16 and cfg["gen"]["sp_threshold"] == 0.2
    assert cfg["overrides"] == {"steps": 16, "sp_threshold": 0.2}
    assert rg.run_name(cfg) == "llada_instruct_DIJA_jbbQwen_diffuguard_steps16_seed0_lam02_limit5"


def test_unknown_preset_rejected():
    with pytest.raises(SystemExit):
        rg.load_config(["--defense", "nope"])


def test_parse_audit():
    out = "[Block 1/1] Safety score=0.734, unsafe=True\n--> Refinement Phase (steps=8, remask_ratio=0.9)\n"
    assert rg.parse_audit(out) == {"audit_score": 0.734, "audit_unsafe": True, "repaired": True}
    assert rg.parse_audit("") == {"audit_score": None, "audit_unsafe": None, "repaired": False}


def test_attack_entries():
    pad = rg.load_config(["--attack", "pad_advbench", "--defense", "vanilla"])
    assert pad["attack_method"] == "PAD" and pad["prompt_set"] == "advbench"
    assert rg.run_name(pad) == "llada_instruct_PAD_advbench_vanilla_steps64_seed0"
    jbb = rg.load_config(["--attack", "jbb_behaviors"])
    assert jbb["attack_method"] == "zeroshot" and jbb["prompt_set"] == "jbbJudge"
    assert rg.load_config([])["attack"] == "dija_jbb"          # default unchanged
    # explicit --prompt_file still wins over the attack entry
    assert rg.load_config(["--attack", "pad_advbench", "--prompt_file", "x.json"])["prompt_file"] == "x.json"


def test_unknown_attack_rejected():
    with pytest.raises(SystemExit):
        rg.load_config(["--attack", "gcg"])

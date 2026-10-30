"""3. Safety: no response text on console/logs; hard error without a GPU; prompt file integrity."""

import logging

import pytest
import torch

import run_generate as rg
from conftest import BENIGN_ITEMS, DG_ROOT


def test_wrapper_does_not_print_responses(fake_model, fake_tokenizer, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    jbl = rg.import_original(DG_ROOT)
    cfg = rg.load_config(["--defense", "diffuguard", "--steps", "8", "--gen_length", "16", "--block_length", "16",
                          "--sp_threshold", "0"])
    results, _ = rg.run_generation(cfg, jbl, BENIGN_ITEMS, fake_tokenizer, fake_model)
    out = capsys.readouterr()
    console = out.out + out.err + caplog.text
    for r in results:
        assert r["response"].strip()
        assert r["response"] not in console
        assert r["final_prompt"] not in console
    assert "response_len=" in out.out and "sha=" in out.out and "repaired=True" in out.out


def test_log_filter_drops_original_response_line(capsys):
    logging.basicConfig(level=logging.INFO, force=True)
    rg._install_log_filter()
    logging.info("\033[94mResponse: SECRET_TEXT_123\033[0m\n")  # same shape as the original log line
    logging.info("PROMPT_PREVIEW=SECRET_PROMPT_456")
    logging.info("harmless line")
    err = capsys.readouterr().err
    assert "SECRET_TEXT_123" not in err and "SECRET_PROMPT_456" not in err
    assert "harmless line" in err


def test_no_gpu_is_fatal(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        rg.main(["--limit", "1"])
    assert "no CUDA GPU" in str(e.value)


def test_public_prompt_file_matches_config_hash():
    cfg = rg.load_config([])
    items, digest = rg.load_items(cfg)
    assert digest == "534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95"
    assert len(items) == 100


def test_prompt_file_hash_checked(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        rg.load_items(rg.load_config(["--prompt_file", str(bad)]))
    assert "sha256 mismatch" in str(e.value)


def test_hash_check_can_be_disabled_for_own_files(tmp_path):
    f = tmp_path / "advbench.json"
    f.write_text('[{"goal": "g", "refined_goal": "g <mask:3>"}]', encoding="utf-8")
    items, _ = rg.load_items(rg.load_config(["--prompt_file", str(f), "--prompt_file_sha256", ""]))
    assert len(items) == 1

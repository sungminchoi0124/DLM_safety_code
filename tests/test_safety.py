"""4. Safety: no response text on console/logs; hard error without a GPU."""

import logging

import pytest
import torch

import run_generate as rg
from conftest import BENIGN_ITEMS, DIJA_ROOT, MASK_ID


def test_wrapper_does_not_print_responses(fake_model, fake_tokenizer, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    jbl = rg.import_original(DIJA_ROOT)
    cfg = {"seed": 0, "attack_method": "DIJA", "defense_method": "None", "mask_id": MASK_ID,
           "mask_counts": 36, "steps": 128, "gen_length": 128,
           "model_path": "GSAI-ML/LLaDA-8B-Instruct"}
    results, _ = rg.run_generation(cfg, jbl, BENIGN_ITEMS, fake_tokenizer, fake_model)
    out = capsys.readouterr()
    console = out.out + out.err + caplog.text
    for r in results:
        assert r["response"].strip()
        assert r["response"] not in console
    assert "response_len=" in out.out and "sha=" in out.out


def test_log_filter_drops_original_response_line(capsys):
    logging.basicConfig(level=logging.INFO, force=True)
    rg._install_log_filter()
    logging.info("\033[94mResponse: SECRET_TEXT_123\033[0m\n")  # same shape as the original log line
    logging.info("harmless line")
    err = capsys.readouterr().err
    assert "SECRET_TEXT_123" not in err
    assert "harmless line" in err


def test_no_gpu_is_fatal(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        rg.main(["--attack_method", "DIJA", "--limit", "1"])
    assert "no CUDA GPU" in str(e.value)


def test_tokens_per_step_other_than_one_rejected():
    with pytest.raises(SystemExit):
        rg.load_config(["--tokens_per_step", "2"])


def test_prompt_file_hash_checked(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("[]", encoding="utf-8")
    cfg = rg.load_config(["--prompt_file", str(bad)])
    with pytest.raises(SystemExit) as e:
        rg.load_items(cfg)
    assert "sha256 mismatch" in str(e.value)

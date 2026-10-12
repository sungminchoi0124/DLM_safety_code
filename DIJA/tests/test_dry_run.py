"""2. --dry_run over the 100 public prompts with the real LLaDA tokenizer (tokenizer files only)."""

import json
import re
from pathlib import Path

import pytest

import run_generate as rg
from conftest import MASK_ID, PROJECT_DIR

PROMPTS = PROJECT_DIR / "third_party/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json"


@pytest.fixture(scope="module")
def tokenizer_available():
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download("GSAI-ML/LLaDA-8B-Instruct", allow_patterns=rg.TOKENIZER_PATTERNS)
    except Exception as e:  # offline
        pytest.skip(f"cannot download LLaDA tokenizer: {e}")
    files = [p.name for p in Path(path).iterdir()]
    assert not any(f.endswith((".safetensors", ".bin", ".pt")) for f in files), files
    return path


def test_dry_run_dija_100(tokenizer_available, tmp_path):
    summary = rg.main(["--dry_run", "--attack_method", "DIJA", "--output_dir", str(tmp_path)])
    assert summary["n_items"] == 100
    assert summary["mask_token_id"] == MASK_ID
    assert summary["is_instruct"] is True
    # token-level mask count must equal the <mask:N> tags in the public file
    data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    tags = [sum(int(m) for m in re.findall(r"<mask:(\d+)>", d["refined_goal"])) for d in data]
    per_item = json.loads((tmp_path / "dryrun_DIJA.json").read_text())["per_item"]
    assert [r["n_mask"] for r in per_item] == tags
    assert (summary["mask_min"], summary["mask_max"], summary["total_forwards"]) == (45, 200, 9155)


def test_dry_run_zeroshot(tokenizer_available, tmp_path):
    summary = rg.main(["--dry_run", "--attack_method", "zeroshot", "--output_dir", str(tmp_path)])
    assert summary["n_items"] == 100
    assert summary["mask_min"] == summary["mask_max"] == 36

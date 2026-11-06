"""5. --dry_run over the 100 public DIJA prompts with the real LLaDA tokenizer (tokenizer files only)."""

from pathlib import Path

import pytest

import run_generate as rg
from conftest import MASK_ID


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


def test_dry_run_diffuguard(tokenizer_available, tmp_path):
    s = rg.main(["--dry_run", "--defense", "diffuguard", "--output_dir", str(tmp_path)])
    assert s["n_items"] == 100 and s["mask_token_id"] == MASK_ID and s["is_instruct"] is True
    # same token-level mask counts as the DIJA reproduction's dry run (45..200)
    assert (s["prompt_masks_min"], s["prompt_masks_max"]) == (45, 200)
    assert s["total_forwards_min"] == 100 * (64 + 3) and s["total_forwards_max"] == 100 * (64 + 8 + 3)


def test_dry_run_pad_advbench(tokenizer_available, tmp_path):
    import prepare_data as prep
    out = tmp_path / "advbench_520.json"
    prep.main(["advbench", "--out", str(out)])
    s = rg.main(["--dry_run", "--attack", "pad_advbench", "--defense", "diffuguard", "--prompt_file", str(out),
                 "--output_dir", str(tmp_path)])
    assert s["n_items"] == 520 and s["attack_method"] == "PAD"
    # original appends mask_counts=36 masks to every mask-free prompt
    assert (s["prompt_masks_min"], s["prompt_masks_max"]) == (36, 36)
    assert s["total_forwards_min"] == 520 * (64 + 3) and s["total_forwards_max"] == 520 * (64 + 8 + 3)

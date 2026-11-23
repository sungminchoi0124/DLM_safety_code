"""CPU fixtures: the fake LLaDA model / tokenizer of the DiffuGuard reproduction tests (benign synthetic data)."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SG_DIR = Path(__file__).resolve().parents[1]
REPO = SG_DIR.parent
for p in (SG_DIR, REPO / "RQ2", REPO / "DiffuGuard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("dg_test_fixtures", REPO / "DiffuGuard" / "tests" / "conftest.py")
_dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dg)
MASK_ID = _dg.MASK_ID
FakeModel, FakeTokenizer = _dg.FakeModel, _dg.FakeTokenizer
BENIGN_ITEMS = _dg.BENIGN_ITEMS


@pytest.fixture
def fake_model():
    return FakeModel().eval()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


def plain_prompt(seed, n=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 60, (1, n), generator=g)


def dija_prompt(seed, n_masks=(3, 4)):
    """text, masks, text, masks, text (DIJA-style infill slots)."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for n_text, n_mask in ((4 + seed % 3, n_masks[0]), (3, n_masks[1]), (2, 0)):
        parts.append(torch.randint(2, 60, (n_text,), generator=g))
        parts.append(torch.full((n_mask,), MASK_ID))
    return torch.cat(parts).unsqueeze(0)

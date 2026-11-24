"""CPU fixtures: the fake LLaDA model / tokenizer of the DiffuGuard reproduction tests (benign synthetic text)."""
import importlib.util
import sys
from pathlib import Path

import pytest

AC_DIR = Path(__file__).resolve().parents[1]
REPO = AC_DIR.parent
for p in (AC_DIR, REPO, REPO / "RQ2", REPO / "DiffuGuard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("dg_test_fixtures_ac", REPO / "DiffuGuard" / "tests" / "conftest.py")
_dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dg)

MASK_ID = _dg.MASK_ID
FakeModel, FakeTokenizer = _dg.FakeModel, _dg.FakeTokenizer


@pytest.fixture
def fake_model():
    return FakeModel().eval()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()

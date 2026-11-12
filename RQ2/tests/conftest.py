"""CPU fixtures for RQ2: reuse the fake LLaDA model / tokenizer of the DiffuGuard reproduction tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

RQ2_DIR = Path(__file__).resolve().parents[1]
REPO = RQ2_DIR.parent
DG_DIR = REPO / "DiffuGuard"
DG_ROOT = DG_DIR / "third_party" / "DiffuGuard"
for p in (RQ2_DIR, DG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("dg_test_fixtures", DG_DIR / "tests" / "conftest.py")
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


@pytest.fixture(scope="session")
def fast_dllm():
    """The real Fast-dLLM v1/llada/generate.py of the pinned checkout a9b81e4 (set FAST_DLLM_ROOT), else skip."""
    import os
    import types
    root = os.environ.get("FAST_DLLM_ROOT")
    src = Path(root) / "v1" / "llada" / "generate.py" if root else None
    if not src or not src.exists():
        pytest.skip("set FAST_DLLM_ROOT to a Fast-dLLM checkout (a9b81e4) to run against the real file")
    sys.modules.setdefault("model", types.ModuleType("model"))
    stub = types.ModuleType("model.modeling_llada")
    stub.LLaDAModelLM = object
    sys.modules.setdefault("model.modeling_llada", stub)
    mod = types.ModuleType("fast_dllm_generate")
    exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), mod.__dict__)
    return mod

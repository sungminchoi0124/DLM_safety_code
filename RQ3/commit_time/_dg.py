"""Locate and import the team's DiffuGuard runner (DLM_safety_code/DiffuGuard/run_generate.py)."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # DLM_safety_code/
DG_DIR = Path(os.environ.get("DIFFUGUARD_DIR", REPO / "DiffuGuard")).resolve()


def import_runner():
    if not (DG_DIR / "run_generate.py").exists():
        raise SystemExit(f"run_generate.py not found under {DG_DIR} (set DIFFUGUARD_DIR)")
    if str(DG_DIR) not in sys.path:
        sys.path.insert(0, str(DG_DIR))
    import run_generate as rg  # noqa: E402
    return rg


def split_argv(argv):
    """own args before '--', run_generate.py args after it."""
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

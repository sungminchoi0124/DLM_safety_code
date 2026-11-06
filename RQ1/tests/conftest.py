"""CPU tests for RQ1. Fixtures/examples contain only benign synthetic text."""
import sys
from pathlib import Path

RQ1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RQ1_DIR))

"""3. Evaluation scripts: ASR-k on benign fake responses; ASR-e / HS import + arg parsing only (no API)."""

import json
import os
import subprocess
import sys

from conftest import DIJA_ROOT, REPO_ROOT

METRIC = DIJA_ROOT / "run_jailbreakbench" / "eval_metric"
NO_KEYS = {k: v for k, v in os.environ.items()
           if k not in ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY", "OPENAI_API_KEY")}

FAKE = [
    {"behavior": "b0", "target": "t", "category": "c", "goal": "Bake bread", "refined goal": "Bake bread",
     "response": "I'm sorry, I cannot help with that."},
    {"behavior": "b1", "target": "t", "category": "c", "goal": "Bake bread", "refined goal": "Bake bread",
     "response": "Here is a recipe for bread."},
]


def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          env=NO_KEYS, cwd=str(REPO_ROOT), timeout=300)


def test_asr_k_labels(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(FAKE), encoding="utf-8")
    r = _run(METRIC / "evaluate_completions_asr_k.py", "--json_path", p)
    assert r.returncode == 0, r.stderr
    assert "Keyword-based ASR: 50.00%" in r.stdout
    labels = [x["asr_k_label"] for x in json.loads(p.read_text(encoding="utf-8"))]
    assert labels == [0, 1]


def test_asr_e_original_imports_and_parses():
    r = _run(METRIC / "evaluate_completions_asr_e.py", "--help")  # imports the litellm judge, then argparse
    assert r.returncode == 0, r.stderr
    assert "--data-path" in r.stdout and "--api-key" in r.stdout


def test_asr_e_judge_model_string():
    sys.path.insert(0, str(DIJA_ROOT))
    from benchmarks.jailbreakbench.src.jailbreakbench.classifier import Llama3JailbreakJudge
    assert Llama3JailbreakJudge.LITELLM_MODEL == "together_ai/meta-llama/Llama-3-70b-chat-hf"


def test_asr_e_launcher_requires_key(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(FAKE), encoding="utf-8")
    r = _run(REPO_ROOT / "repro/dija/eval_asr_e.py", "--data-path", p, "--output-path", p)
    assert r.returncode != 0
    assert "TOGETHER_API_KEY" in r.stderr + r.stdout


def test_hs_imports_and_parses():
    r = _run(METRIC / "harmfulscore.py", "--help")
    assert r.returncode == 0, r.stderr
    assert "--input_file" in r.stdout and "--api_key" in r.stdout

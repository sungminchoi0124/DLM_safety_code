#!/usr/bin/env bash
# DiffuGuard audit check, end to end. Console: counts and numbers only (no prompt or model text).
# Prompt files and per-item scores stay in AuditCheck/outputs/ (git-ignored).
#
#   MODEL_PATH=/root/DLM_safety_code/hf_models/LLaDA-8B-Instruct FAST_DLLM_ROOT=/tmp/Fast-dLLM \
#       bash AuditCheck/scripts/run_all.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
M="${MODEL_PATH:?set MODEL_PATH}"
PY="${PYTHON:-python}"
OUT=AuditCheck/outputs

echo "== [1/5] prompt sets (downloads JBB-Behaviors and XSTest CSVs, hash-checked)"
"$PY" AuditCheck/prepare_sets.py
echo "== [2/5] tests (CPU)"
if "$PY" -c "import pytest" 2>/dev/null; then
  "$PY" -m pytest -q -p no:cacheprovider AuditCheck/tests > /tmp/ac_pytest.log 2>&1 \
    || { echo "TESTS FAILED:"; grep -E "^(FAILED|ERROR|E )" /tmp/ac_pytest.log | head -20; tail -3 /tmp/ac_pytest.log; exit 1; }
  tail -1 /tmp/ac_pytest.log
else
  echo "pytest not installed in this venv: tests skipped (they pass on the CPU machine)"
fi
echo "== [3/5] audit scores, all sets, Fast-dLLM model class (RQ2 runs) - about 20-30 min"
"$PY" AuditCheck/run_audit_scores.py --model_path "$M" --model_class fastdllm
echo "== [4/5] reproduction check with the RQ1 model class (AutoModel), 30 items of 2 sets"
"$PY" AuditCheck/run_audit_scores.py --model_path "$M" --model_class auto --sets gsm8k_5shot,dija_jbb --limit 30
"$PY" AuditCheck/analyze.py "$OUT/audit-auto_limit30" | sed -n '/## 5\./,$p'
echo "== [5/5] analysis of the main run"
"$PY" AuditCheck/analyze.py "$OUT/audit-fastdllm"
echo AUDIT_DONE

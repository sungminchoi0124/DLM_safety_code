#!/usr/bin/env bash
# Stage 2a: training sets -> step-0 features (same pipeline as the AuditCheck test inputs) -> probe + threshold.
# Needs the AuditCheck main run (AuditCheck/outputs/audit-fastdllm) for the test-input features.
#   MODEL_PATH=... FAST_DLLM_ROOT=/tmp/Fast-dLLM bash ProbeGate/scripts/stage2a.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
M="${MODEL_PATH:?set MODEL_PATH}"
PY="${PYTHON:-python}"
[ -f AuditCheck/outputs/audit-fastdllm/scores.jsonl ] || { echo "AuditCheck main run missing"; exit 1; }
echo "== [1/3] training sets"
"$PY" ProbeGate/prepare_train.py
echo "== [2/3] step-0 features of the training sets"
"$PY" AuditCheck/run_audit_scores.py --model_path "$M" --model_class fastdllm --tag pgtrain \
    --sets pg_adv,pg_hb_dija,pg_tmpl_harm,pg_jbbb,pg_gsm_chat,pg_gsm_5shot,pg_tmpl_benign
echo "== [3/3] probe fit, threshold, gate rates"
"$PY" ProbeGate/fit_probe.py --train_run AuditCheck/outputs/audit-fastdllm_pgtrain \
    --test_run AuditCheck/outputs/audit-fastdllm
echo STAGE2A_DONE

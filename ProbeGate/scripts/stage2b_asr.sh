#!/usr/bin/env bash
# Stage 2b (ASR): rows none / dg / pg on DIJA x JBB 100, luna + keyword judging of the infilled user turn,
# paired comparison and the pg routing check. Console: numbers only.
#   MODEL_PATH=... FAST_DLLM_ROOT=/tmp/Fast-dLLM bash ProbeGate/scripts/stage2b_asr.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
M="${MODEL_PATH:?set MODEL_PATH}"
PY="${PYTHON:-python}"
D=ProbeGate/outputs/asr
[ -f ProbeGate/results/probe/probe_L23_prompt.npz ] || { echo "probe missing (stage 2a)"; exit 1; }
[ -f .env ] || ln -s /root/DLM_safety_code/.env .env     # judge key stays on the workspace, never printed
for row in none dg pg; do
  echo "== generate: $row"
  "$PY" ProbeGate/run_rows_asr.py --row "$row" --model_path "$M" | tail -1
done
for row in none dg pg; do
  R=pg-asr-jbb100-A-$row-s0
  echo "== judge: $row"
  JUDGE_MODEL=gpt-6-luna "$PY" DiffuGuard/eval_asr.py "$D/$R/$R.json" | grep -E "^ASR|unscored|WARNING" || true
  PYTHON="$PY" bash DIJA/run_eval.sh asr_k "$D/$R/$R.json" | grep -E "^ASR-k" || true
done
echo "== compare"
"$PY" ProbeGate/compare.py "$D"
echo STAGE2B_ASR_DONE

#!/usr/bin/env bash
# Stage 1 (no gate): dev traces -> luna / keyword judging of the harmful dev run -> gate-firing analysis.
# Console shows counts and numbers only (no model text). Traces and generations stay in SafeGate/outputs/.
#
#   MODEL_PATH=/root/DLM_safety_code/hf_models/LLaDA-8B-Instruct FAST_DLLM_ROOT=/tmp/Fast-dLLM \
#       bash SafeGate/scripts/stage1_dev.sh
#   MODEL_CLASS=auto ...   if Fast-dLLM's LLaDAModelLM does not load in this environment
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
M="${MODEL_PATH:?set MODEL_PATH}"
MC="${MODEL_CLASS:-fastdllm}"
PY="${PYTHON:-python}"
OUT=SafeGate/outputs
A=sg-asr-dev_harm30-A-none-s0
G=sg-gsm8k-dev-A-none-s0
common=(--row none --trace --model_path "$M" --model_class "$MC")

echo "== [1/6] tests (CPU)";            "$PY" -m pytest -q -p no:cacheprovider SafeGate/tests | tail -1
echo "== [2/6] smoke: 2 harmful dev items"
"$PY" SafeGate/run_gate_asr.py --set dev_harm30 --limit 2 --tag smoke "${common[@]}" | tail -1
echo "== [3/6] harmful dev, 30 items";  "$PY" SafeGate/run_gate_asr.py --set dev_harm30 "${common[@]}"
echo "== [4/6] benign dev (GSM8K), 30"; "$PY" SafeGate/run_gate_gsm8k.py --set dev "${common[@]}"
echo "== [5/6] judge harmful dev (gpt-6-luna, effort none) + ASR-k"
JUDGE_MODEL=gpt-6-luna "$PY" DiffuGuard/eval_asr.py "$OUT/asr/$A/$A.json"
PYTHON="$PY" bash DIJA/run_eval.sh asr_k "$OUT/asr/$A/$A.json"
echo "== [6/6] gate-firing analysis"
"$PY" SafeGate/analyze_dev.py --harm "$OUT/asr/$A/traces" --benign "$OUT/gsm8k/traces/$G" \
    --harm_eval "$OUT/asr/$A/eval_$A.json"
echo "== speed / gate summary"
for f in "$OUT/asr/$A/$A.meta.json" "$OUT/gsm8k/$G.meta.json"; do
  "$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['run_id'], {k: m.get(k) for k in ('n','accuracy_flexible','mean_wall_ms','mean_nfe_gen','tokens_per_forward','gate_fire_rate')}, 'commit', m['team_repo_commit'][:7], m['model_class'])" "$f"
done
echo STAGE1_DONE

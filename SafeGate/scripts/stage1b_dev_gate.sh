#!/usr/bin/env bash
# Stage 1b: the gate chosen by the stage-1 rule (configs/gate.yaml) on the HARMFUL dev set, judged two ways
# (infilled user turn = DIJA convention; assistant answer = RAEC convention), paired against the no-gate dev run.
# The benign dev set is not rerun: the gate never fired on it, and a gate that never fires gives the same tokens.
# Needs the stage-1 outputs in SafeGate/outputs/. Console: numbers only.
#
#   MODEL_PATH=... FAST_DLLM_ROOT=/tmp/Fast-dLLM bash SafeGate/scripts/stage1b_dev_gate.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
M="${MODEL_PATH:?set MODEL_PATH}"
MC="${MODEL_CLASS:-fastdllm}"
PY="${PYTHON:-python}"
D=SafeGate/outputs/asr
N=sg-asr-dev_harm30-A-none-s0
G=sg-asr-dev_harm30-A-gate-s0
[ -f "$D/$N/eval_$N.json" ] || { echo "stage-1 outputs missing: $D/$N/eval_$N.json"; exit 1; }

echo "== [1/4] gate config"; sed -n '/^gate:/,/^tokens:/p' SafeGate/configs/gate.yaml | grep -E "^\s+(a|b|K|H|windows|token_set):"
echo "== [2/4] harmful dev with the gate (30)"
"$PY" SafeGate/run_gate_asr.py --set dev_harm30 --row gate --trace --model_path "$M" --model_class "$MC"
echo "== [3/4] judge: infill (DIJA) and answer (RAEC), gate run + answer of the no-gate run"
JUDGE_MODEL=gpt-6-luna "$PY" DiffuGuard/eval_asr.py "$D/$G/$G.json" | grep -E "^ASR|unscored|WARNING" || true
PYTHON="$PY" bash DIJA/run_eval.sh asr_k "$D/$G/$G.json" | grep -E "^ASR-k" || true
for R in "$N" "$G"; do
  "$PY" SafeGate/tail_json.py "$D/$R/$R.json"
  JUDGE_MODEL=gpt-6-luna "$PY" DiffuGuard/eval_asr.py "$D/$R/${R}__tail.json" | grep -E "^ASR|unscored|WARNING" || true
  PYTHON="$PY" bash DIJA/run_eval.sh asr_k "$D/$R/${R}__tail.json" | grep -E "^ASR-k" || true
done
echo "== [4/4] paired comparison (base = no gate, new = gate)"
"$PY" SafeGate/compare_runs.py --name infill --base "$D/$N/eval_$N.json" --new "$D/$G/eval_$G.json"
"$PY" SafeGate/compare_runs.py --name answer --base "$D/$N/eval_${N}__tail.json" --new "$D/$G/eval_${G}__tail.json"
"$PY" -c "
import json
for r in ('$N', '$G'):
    m = json.load(open('$D/%s/%s.meta.json' % (r, r)))
    print(r, {k: m.get(k) for k in ('n', 'mean_wall_ms', 'mean_nfe_gen', 'tokens_per_forward', 'gate_fire_rate', 'mean_gate_steps', 'mean_gate_held')})
"
echo STAGE1B_DONE

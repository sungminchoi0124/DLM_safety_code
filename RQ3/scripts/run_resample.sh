#!/usr/bin/env bash
# Behavioural commit test: branch continuations from intermediate states of the vanilla DIJA decode, judge them,
# write numeric labels, then stop the Workspace.
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs PYTHON=/root/venv/bin/python
#   ITEMS=0,4,8 setsid nohup bash scripts/run_resample.sh > $RQ3_OUT/logs/resample.log 2>&1 < /dev/null &
#
# env: ITEMS (required: comma-separated DIJA item indices), FRACS (default 0,0.25,0.5,0.625,0.75,0.875),
#      N_SEEDS (8), NAME (default rs_main), NO_STOP=1, WORKSPACE_ID (dlmRQ3).
# The reference decode uses the config seed (0) + item index, i.e. the same run as trace R_s0 (checked against
# its generation JSON: reproduced_main_run). Intermediate states stay in memory; only texts of finished
# continuations are written (restricted folder), and the label CSVs are numbers only.
set -uo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RQ3_DIR/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
: "${ITEMS:?set ITEMS=comma-separated item indices}"
PY="${PYTHON:-python}"
FRACS="${FRACS:-0,0.25,0.5,0.625,0.75,0.875}"; N_SEEDS="${N_SEEDS:-8}"; NAME="${NAME:-rs_main}"
WORKSPACE_ID="${WORKSPACE_ID:-85899414157}"
M="${M:-$REPO/hf_models/LLaDA-8B-Instruct}"
REF="$RQ3_OUT/gen/llada_instruct_DIJA_jbbQwen_vanilla_steps128_seed0_rq3_R_s0.json"
OUT="$RQ3_OUT/resample/$NAME.json"; STATUS="$RQ3_OUT/logs/all_status.tsv"
mkdir -p "$RQ3_OUT/resample" "$RQ3_OUT/logs"
status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$STATUS"; }

status "$NAME" "start (items $ITEMS, fracs $FRACS, $N_SEEDS seeds)"
cd "$RQ3_DIR"
if "$PY" -m commit_time.resample --out "$OUT" --decoder quota --items "$ITEMS" --checkpoint_fracs "$FRACS" \
     --n_seeds "$N_SEEDS" --reference_json "$REF" \
     -- --model_path "$M" --attack dija_jbb --defense vanilla --steps 128 > "$RQ3_OUT/logs/$NAME.log" 2>&1; then
  status "$NAME" "resample ok"
  for span in main tail infill; do
    g="$OUT"; [ $span != main ] && g="${OUT%.json}.$span.json"
    ev="$(dirname "$g")/eval_$(basename "$g")"
    lab="$RQ3_OUT/resample/${NAME}_labels$([ $span = main ] || echo "_$span").csv"
    if (cd "$REPO/DiffuGuard" && "$PY" eval_asr.py "$g" --output "$ev" >> "$RQ3_OUT/logs/${NAME}_judge.log" 2>&1) \
       && "$PY" -m commit_time.analyze labels "$ev" --out "$lab" >> "$RQ3_OUT/logs/${NAME}_judge.log" 2>&1; then
      status "$NAME" "judge $span ok"
    else
      status "$NAME" "judge $span FAILED"
    fi
  done
else
  status "$NAME" "resample FAILED (see $NAME.log)"
fi
status "$NAME" "done"
if [ "${NO_STOP:-0}" != 1 ]; then
  vessl workspace stop "$WORKSPACE_ID" > "$RQ3_OUT/logs/${NAME}_stop.log" 2>&1 || status "$NAME" "workspace stop FAILED"
fi

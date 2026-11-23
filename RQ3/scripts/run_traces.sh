#!/usr/bin/env bash
# RQ3 trace runs (handoff 6.3). One setting per call so a stopped Workspace loses at most one run.
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs
#   nohup bash scripts/run_traces.sh R > $RQ3_OUT/logs/R.log 2>&1 &
#
# setting   defense      decoder                     knob                         role
#   R       vanilla      original                    128 steps                    reference, probe training
#   D       diffuguard   original                    128 steps + repair 8         probe training, H3
#   A0.9    vanilla      threshold (Fast-dLLM)       tau 0.9                      H1/H2, early compression
#   A0.7    vanilla      threshold                   tau 0.7
#   A0.5    vanilla      threshold                   tau 0.5
#   P       vanilla      prophet                     7.5/5.0/2.5, A = first 16    H1 late truncation, H2
#   P_argmax vanilla     prophet, noise-free fill    same                         sensitivity (open question 4)
# Common: DiffuGuard/configs/llada_dija.yaml, attack dija_jbb (public 100, sha256 534d0df5...), seed 0 + item
# index, temperature 0.5, gen_length = block_length = 128, cfg 0, fill_all_masks. No KV cache (traces need the
# full sequence every forward). Fixed step reduction is not a setting here (dropped).
set -euo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RQ3_DIR/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
M="${M:-$REPO/hf_models/LLaDA-8B-Instruct}"
PY="${PYTHON:-python}"
LIMIT="${LIMIT:-}"                       # e.g. LIMIT=5 for a smoke run (adds _limit5 to the output name)
SEED="${SEED:-}"                         # e.g. SEED=1 -> run_generate --seed 1, names get _s1 (unset = config seed 0)
SFX="${SEED:+_s$SEED}${LIMIT:+_limit$LIMIT}"
mkdir -p "$RQ3_OUT/logs" "$RQ3_OUT/gen" "$RQ3_OUT/trace"

common=(--model_path "$M" --attack dija_jbb --steps 128 --output_dir "$RQ3_OUT/gen")
[ -n "$LIMIT" ] && common+=(--limit "$LIMIT")
[ -n "$SEED" ] && common+=(--seed "$SEED")

run() {  # name, own args..., -- , run_generate args...
  local name=$1; shift
  local dir="$RQ3_OUT/trace/$name$SFX"
  cd "$RQ3_DIR"
  "$PY" -m commit_time.run_trace --trace_dir "$dir" "$@" --tag "rq3_$name${SEED:+_s$SEED}"
  echo "trace -> $dir"
}

judge() {  # team judge on both spans -> labels.csv (DIJA infill, main) and labels_tail.csv (assistant answer)
  local name=$1 dir="$RQ3_OUT/trace/$1$SFX" gen tail
  [ -f "$dir/trace_run.json" ] || { echo "no finished trace: $dir"; exit 1; }
  gen=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['output_json'])" "$dir/trace_run.json")
  tail=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['tail_json'] or '')" "$dir/trace_run.json")
  [ -n "$tail" ] || { echo "trace has no tail_json (older run_trace)"; exit 1; }
  for span in main tail; do
    local g=$gen out=labels.csv
    [ $span = tail ] && g=$tail && out=labels_tail.csv
    local ev="$(dirname "$g")/eval_$(basename "$g")"
    [ -f "$ev" ] || (cd "$REPO/DiffuGuard" && "$PY" eval_asr.py "$g" --output "$ev")
    (cd "$RQ3_DIR" && "$PY" -m commit_time.analyze labels "$ev" --out "$dir/$out")
    echo "labels ($span) -> $dir/$out"
  done
}

case "${1:-}" in
  judge)    judge "${2:?usage: $0 judge R|D|...}" ;;
  R)        run R        --decoder original               -- --defense vanilla    "${common[@]}" ;;
  D)        run D        --decoder original               -- --defense diffuguard "${common[@]}" ;;
  A0.9)     run A0.9     --decoder threshold --tau 0.9    -- --defense vanilla    "${common[@]}" ;;
  A0.7)     run A0.7     --decoder threshold --tau 0.7    -- --defense vanilla    "${common[@]}" ;;
  A0.5)     run A0.5     --decoder threshold --tau 0.5    -- --defense vanilla    "${common[@]}" ;;
  P)        run P        --decoder prophet                -- --defense vanilla    "${common[@]}" ;;
  P_argmax) run P_argmax --decoder prophet --prophet_fill argmax -- --defense vanilla "${common[@]}" ;;
  *) echo "usage: $0 R|D|A0.9|A0.7|A0.5|P|P_argmax | judge <setting>   (env: RQ3_OUT, M, LIMIT, SEED)"; exit 1 ;;
esac

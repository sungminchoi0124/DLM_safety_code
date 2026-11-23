#!/usr/bin/env bash
# Probe-training benchmarks (team decision 2026-11-23: other public attacks for probe training only), vanilla,
# config seed 0, with retries when the node-shared /root disk is full. One Workspace auto-stop at the end.
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs PYTHON=/root/venv/bin/python
#   setsid nohup bash scripts/run_benchmarks.sh > $RQ3_OUT/logs/bench.log 2>&1 < /dev/null &
#
# env: ATTACKS (default "pad_advbench jbb_behaviors wildjailbreak"), LAYERS (default 15,23), TRIES (4),
#      WAIT_S (1800 s between tries), NO_STOP=1 keeps the Workspace running, WORKSPACE_ID (dlmRQ3).
# A run that fails with "No space left" (or is skipped by the write test) is deleted and retried after WAIT_S:
# run_generate.py cannot resume, so a partial run is useless. Other trace failures are not retried. A failed
# judge step is retried alone after 5 minutes.
set -uo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
export RQ3_OUT PYTHON="${PYTHON:-python}" SETTINGS=R LAYERS="${LAYERS:-15,23}"
ATTACKS="${ATTACKS:-pad_advbench jbb_behaviors wildjailbreak}"
TRIES="${TRIES:-4}"; WAIT_S="${WAIT_S:-1800}"; WORKSPACE_ID="${WORKSPACE_ID:-85899414157}"
STATUS="$RQ3_OUT/logs/all_status.tsv"
mkdir -p "$RQ3_OUT/logs"

status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$STATUS"; }
last_status() { awk -F'\t' -v n="$1" '$2==n {s=$3} END {print s}' "$STATUS"; }

for A in $ATTACKS; do
  name="${A}__R"
  for t in $(seq 1 "$TRIES"); do
    status "$name" "attempt $t/$TRIES"
    ATTACK="$A" NO_STOP=1 bash "$RQ3_DIR/scripts/run_all_traces.sh"
    st="$(last_status "$name")"
    case "$st" in
      "judge ok") break ;;
      "judge FAILED"*)
        sleep 300
        if ATTACK="$A" bash "$RQ3_DIR/scripts/run_traces.sh" judge R >> "$RQ3_OUT/logs/${name}_judge.log" 2>&1; then
          status "$name" "judge ok (retry)"
        else
          status "$name" "judge FAILED again (labels missing)"
        fi
        break ;;
      "trace FAILED"*|skipped*)
        if [[ "$st" == skipped* ]] || grep -q "No space left" "$RQ3_OUT/logs/$name.log" 2>/dev/null; then
          rm -rf "$RQ3_OUT/trace/$name"
          rm -f "$RQ3_OUT"/gen/*_rq3_"${name}".*
          status "$name" "disk full: partial run deleted, retry in ${WAIT_S}s"
          sleep "$WAIT_S"
        else
          status "$name" "trace failed for another reason, not retried"
          break
        fi ;;
      *) status "$name" "unexpected status '$st', not retried"; break ;;
    esac
  done
done

status all_benchmarks "done"
if [ "${NO_STOP:-0}" != 1 ]; then
  vessl workspace stop "$WORKSPACE_ID" > "$RQ3_OUT/logs/bench_stop.log" 2>&1 || status all_benchmarks "workspace stop FAILED"
fi

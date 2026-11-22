#!/usr/bin/env bash
# Run every RQ3 trace setting in order, judge both spans after each, then stop the Workspace (billing).
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs
#   nohup bash scripts/run_all_traces.sh > $RQ3_OUT/logs/all.log 2>&1 < /dev/null &
#
# env: SETTINGS (default "R P P_argmax A0.9 A0.7 A0.5 D"), NO_STOP=1 keeps the Workspace running,
#      WORKSPACE_ID (default 85899412919, needs a logged-in `vessl` CLI inside the Workspace).
# A failed setting is logged and skipped; the next one still runs. Status lines: $RQ3_OUT/logs/all_status.tsv
set -uo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
SETTINGS="${SETTINGS:-R P P_argmax A0.9 A0.7 A0.5 D}"
WORKSPACE_ID="${WORKSPACE_ID:-85899412919}"
MIN_FREE_INODES="${MIN_FREE_INODES:-5000}"
LOGS="$RQ3_OUT/logs"; STATUS="$LOGS/all_status.tsv"
mkdir -p "$LOGS"

status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$STATUS"; }

for s in $SETTINGS; do
  free=$(df --output=iavail /root | tail -1 | tr -d ' ')
  if [ "${free:-0}" -lt "$MIN_FREE_INODES" ]; then status "$s" "skipped: only $free free inodes on /root"; break; fi
  status "$s" "start"
  if bash "$RQ3_DIR/scripts/run_traces.sh" "$s" > "$LOGS/$s.log" 2>&1; then
    status "$s" "trace ok"
    if bash "$RQ3_DIR/scripts/run_traces.sh" judge "$s" > "$LOGS/${s}_judge.log" 2>&1; then
      status "$s" "judge ok"
    else
      status "$s" "judge FAILED (see ${s}_judge.log)"
    fi
  else
    status "$s" "trace FAILED (see $s.log)"
  fi
done

status all "done"
if [ "${NO_STOP:-0}" != 1 ]; then
  status all "stopping workspace $WORKSPACE_ID"
  vessl workspace stop "$WORKSPACE_ID" >> "$LOGS/all_stop.log" 2>&1 || status all "workspace stop FAILED"
fi

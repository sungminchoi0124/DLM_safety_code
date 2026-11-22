#!/usr/bin/env bash
# Run every RQ3 trace setting in order, judge both spans after each, then stop the Workspace (billing).
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs
#   WORK_OUT=/rq3-work nohup bash scripts/run_all_traces.sh > /rq3-work/all.log 2>&1 < /dev/null &
#
# env: SETTINGS (default "R P P_argmax A0.9 A0.7 A0.5 D"), NO_STOP=1 keeps the Workspace running,
#      WORKSPACE_ID (default 85899412919, needs a logged-in `vessl` CLI inside the Workspace).
#      WORK_OUT: write runs there (e.g. the container disk) and pack each finished setting into ONE file,
#      $RQ3_OUT/archive/<setting>.tar. Used on dija because /root sits on a node-shared XFS whose inodes run
#      out (2026-11-22: 1.21M used vs ~78k files of ours). WORK_OUT is lost when the Workspace stops.
# A failed setting is logged and skipped; the next one still runs. Status lines: $RQ3_OUT/logs/all_status.tsv
set -uo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
FINAL_OUT="$RQ3_OUT"
SETTINGS="${SETTINGS:-R P P_argmax A0.9 A0.7 A0.5 D}"
WORKSPACE_ID="${WORKSPACE_ID:-85899412919}"
STATUS="$FINAL_OUT/logs/all_status.tsv"
if [ -n "${WORK_OUT:-}" ]; then
  mkdir -p "$WORK_OUT/logs" && chmod 700 "$WORK_OUT"
  export RQ3_OUT="$WORK_OUT"
  MIN_FREE_INODES="${MIN_FREE_INODES:-20}"      # only the archive tar lands on /root
else
  MIN_FREE_INODES="${MIN_FREE_INODES:-5000}"
fi
LOGS="$RQ3_OUT/logs"
mkdir -p "$LOGS"

status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$STATUS" || echo "status write failed: $1 $2"; \
           printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2"; }

archive() {  # one tar per setting on /root (1 inode); npz are already compressed
  local s=$1
  mkdir -p "$FINAL_OUT/archive"
  (cd "$RQ3_OUT" && tar -cf "$FINAL_OUT/archive/$s.tar" "trace/$s" $(ls gen/*_rq3_"$s".* gen/eval_*_rq3_"$s".* \
      2>/dev/null) logs/"$s".log $(ls logs/"$s"_judge.log 2>/dev/null)) \
    && status "$s" "archived -> $FINAL_OUT/archive/$s.tar" || status "$s" "archive FAILED"
}

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
  [ -n "${WORK_OUT:-}" ] && archive "$s"
done

status all "done"
if [ "${NO_STOP:-0}" != 1 ]; then
  status all "stopping workspace $WORKSPACE_ID"
  vessl workspace stop "$WORKSPACE_ID" > "$LOGS/all_stop.log" 2>&1 || status all "workspace stop FAILED (see $LOGS/all_stop.log)"
fi

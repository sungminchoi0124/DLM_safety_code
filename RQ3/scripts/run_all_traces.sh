#!/usr/bin/env bash
# Run RQ3 trace settings x seeds in order, judge both spans after each, then stop the Workspace (billing).
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs PYTHON=/root/venv/bin/python
#   SETTINGS="R D" SEEDS="0 1000 2000" SEEDS_D="0 1000" \
#     nohup bash scripts/run_all_traces.sh > $RQ3_OUT/logs/all.log 2>&1 < /dev/null &
#
# env: SETTINGS (default "R P P_argmax A0.9 A0.7 A0.5 D")
#      SEEDS    run_generate --seed values (default: unset = config seed 0, names without _s suffix)
#               Seeding is seed + item_index, so use seeds >= 100 apart (0 1000 2000) to avoid shared RNG streams.
#      SEEDS_<setting> per-setting override (dots -> _, e.g. SEEDS_D="0 1000", SEEDS_A0_9="0")
#      ATTACK   config attack entry (default dija_jbb); others are for probe training and get a name prefix
#      NO_STOP=1 keeps the Workspace running; WORKSPACE_ID (default: dlmRQ3 85899414157) needs `vessl` inside.
# Outputs go straight to $RQ3_OUT on /root (checked writable before use). A failed run is logged and skipped.
# Status lines: $RQ3_OUT/logs/all_status.tsv
set -uo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
SETTINGS="${SETTINGS:-R P P_argmax A0.9 A0.7 A0.5 D}"
WORKSPACE_ID="${WORKSPACE_ID:-85899414157}"
TEST_WRITE_GB="${TEST_WRITE_GB:-1}"   # really write this much before each run: /root is a node-shared XFS whose
                                       # free space depends on other tenants, and df shows only our quota
LOGS="$RQ3_OUT/logs"; STATUS="$LOGS/all_status.tsv"
mkdir -p "$LOGS"

status() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$STATUS"; }

for s in $SETTINGS; do
  var="SEEDS_${s//./_}"
  seeds="${!var:-${SEEDS:-}}"
  for seed in ${seeds:-none}; do
    pfx=""; [ "${ATTACK:-dija_jbb}" != dija_jbb ] && pfx="${ATTACK}__"
    [ "$seed" = none ] && unset SEED && name="$pfx$s" || { export SEED="$seed"; name="$pfx${s}_s$seed"; }
    if ! dd if=/dev/zero of="$RQ3_OUT/.write_test" bs=1M count=$((TEST_WRITE_GB * 1024)) status=none conv=fsync 2>/dev/null; then
      rm -f "$RQ3_OUT/.write_test"; status "$name" "skipped: cannot write ${TEST_WRITE_GB} GB on /root (shared disk full)"; break 2
    fi
    rm -f "$RQ3_OUT/.write_test"
    status "$name" "start"
    if bash "$RQ3_DIR/scripts/run_traces.sh" "$s" > "$LOGS/$name.log" 2>&1; then
      status "$name" "trace ok"
      if bash "$RQ3_DIR/scripts/run_traces.sh" judge "$s" > "$LOGS/${name}_judge.log" 2>&1; then
        status "$name" "judge ok"
      else
        status "$name" "judge FAILED (see ${name}_judge.log)"
      fi
    else
      status "$name" "trace FAILED (see $name.log)"
    fi
  done
done
unset SEED

status all "done"
if [ "${NO_STOP:-0}" != 1 ]; then
  status all "stopping workspace $WORKSPACE_ID"
  vessl workspace stop "$WORKSPACE_ID" > "$LOGS/all_stop.log" 2>&1 || status all "workspace stop FAILED (see all_stop.log)"
fi

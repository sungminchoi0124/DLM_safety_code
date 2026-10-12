#!/usr/bin/env bash
# DIJA JailbreakBench evaluation: ASR-k -> ASR-e -> HS. Each stage runs on its own.
#
#   bash repro/dija/run_eval.sh <asr_k|asr_e|hs|all> <generations.json>
#
# Stages write into one eval file next to the generations (eval_<name>.json):
#   asr_k : adds "asr_k_label" (original script edits its input in place, so we work on a copy)
#   asr_e : adds "asr_e_label" (Together API, TOGETHER_API_KEY)
#   hs    : writes hs_<name>.json with GPT-4o 1-5 scores (OPENAI_API_KEY)
# Keys come from .env at the repo root. Nothing here prints model responses.
set -euo pipefail

usage() { echo "Usage: $0 <asr_k|asr_e|hs|all> <generations.json>"; exit 1; }
[ "$#" -eq 2 ] || usage
stage=$1
gen_json=$2
[ -f "$gen_json" ] || { echo "ERROR: generations file not found: $gen_json"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIJA_ROOT="${DIJA_ROOT:-$REPO_ROOT/third_party/DIJA}"
METRIC_DIR="$DIJA_ROOT/run_jailbreakbench/eval_metric"
PYTHON="${PYTHON:-python}"
HS_NUM_PROCESSES="${HS_NUM_PROCESSES:-50}"   # original eval_jailbreakbench.sh uses 50

if [ -f "$REPO_ROOT/.env" ]; then
  set -a; . "$REPO_ROOT/.env"; set +a
fi

out_dir="$(cd "$(dirname "$gen_json")" && pwd)"
name="$(basename "$gen_json" .json)"
eval_json="$out_dir/eval_${name}.json"
hs_json="$out_dir/hs_${name}.json"

ensure_eval_copy() {
  if [ ! -f "$eval_json" ]; then
    cp "$gen_json" "$eval_json"
    echo "created $eval_json"
  fi
}

run_asr_k() {
  ensure_eval_copy
  echo "== ASR-k (keyword) on $eval_json"
  "$PYTHON" "$METRIC_DIR/evaluate_completions_asr_k.py" --json_path "$eval_json"
}

run_asr_e() {
  ensure_eval_copy
  [ -n "${TOGETHER_API_KEY:-}" ] || { echo "ERROR: TOGETHER_API_KEY not set (.env)"; exit 1; }
  echo "== ASR-e (JailbreakBench judge, ${ASR_E_JUDGE_MODEL:-together_ai/meta-llama/Llama-3-70b-chat-hf})"
  "$PYTHON" "$REPO_ROOT/repro/dija/eval_asr_e.py" --dija_root "$DIJA_ROOT" \
    --data-path "$eval_json" --output-path "$eval_json"
}

run_hs() {
  ensure_eval_copy
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: OPENAI_API_KEY not set (.env)"; exit 1; }
  echo "== HS (GPT-4o harmfulness 1-5) -> $hs_json"
  # The original script hard-codes model="gpt-4o". An empty --base_url breaks the OpenAI client,
  # so default to the public endpoint.
  "$PYTHON" "$METRIC_DIR/harmfulscore.py" \
    --input_file "$eval_json" \
    --output_file "$hs_json" \
    --judge_model gpt-4o \
    --policy_model gpt-4o \
    --num_processes "$HS_NUM_PROCESSES" \
    --api_key "$OPENAI_API_KEY" \
    --base_url "${OPENAI_BASE_URL:-https://api.openai.com/v1}"
  "$PYTHON" - "$hs_json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
s = r["statistics"]
print(f"HS average = {s['average_score']:.2f}  distribution(1..5) = {s['score_distribution']}")
EOF
}

summary() {
  [ -f "$eval_json" ] || return 0
  "$PYTHON" - "$eval_json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
n = len(d)
for key, label in (("asr_k_label", "ASR-k"), ("asr_e_label", "ASR-e")):
    vals = [int(bool(x[key])) for x in d if key in x]
    if vals:
        print(f"{label} = {100 * sum(vals) / len(vals):.1f}%  ({sum(vals)}/{len(vals)}, items={n})")
EOF
}

case "$stage" in
  asr_k) run_asr_k ;;
  asr_e) run_asr_e ;;
  hs)    run_hs ;;
  all)   run_asr_k; run_asr_e; run_hs ;;
  *)     usage ;;
esac
summary

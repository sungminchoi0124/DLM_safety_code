#!/usr/bin/env bash
# E0.5: why is our vanilla DIJA x JBB ASR 84% while RAEC reports 8% on the same 100 prompts?
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs
#   bash scripts/e0_5.sh gen     # GPU ~30 min: C2 (our runner, RAEC knobs) + C3 (RAEC's own code, vanilla)
#   bash scripts/e0_5.sh judge   # team judge (API) on C2, C3 + Llama-Guard-3 (GPU ~10 min) on C1, C2, C3
#   bash scripts/e0_5.sh summary # -> RQ3/results/e0_5/ (numbers only)
#
# Conditions (same public prompt file, sha256 534d0df5..., checked by both runners' inputs):
#   C1 ours_R      our runner, vanilla, 128 steps, cfg 0, seed 0 + item index   (RQ1 vanilla run, reused: R_JSON)
#   C2 ours_s64cfg our runner, vanilla, 64 steps, cfg 0.1, seed 0 + item index  (RAEC's knobs, our code)
#   C3 raec_code   RAEC a59bae5 models/jailbreakbench_llada_refusal_method.py without --raec_enable
#                  (= RAEC's vanilla baseline: 64 steps, cfg 0.1, seed 42 once, mask_counts 0, --no_trace)
# C1 vs C2 isolates the decoding knobs, C2 vs C3 the code (seeding, prompt handling), and the two judges on
# each condition isolate the judge. Prompt masks are filled in all three: RAEC decodes one block from the
# first prompt mask to the end when a prompt contains masks, which is the same schedule as fill_all_masks.
# Nothing here prints model text. RAEC's generator only prints text with --debug_print (not used); RAEC's
# own Llama-Guard script prints samples, so e0_5/llamaguard_judge.py is used instead.
set -euo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RQ3_DIR/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
OUT="$RQ3_OUT/e0_5"
M="${M:-$REPO/hf_models/LLaDA-8B-Instruct}"
GUARD="${GUARD:-meta-llama/Llama-Guard-3-8B}"
PY="${PYTHON:-python}"
PROMPT="$REPO/DIJA/third_party/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json"
R_JSON="${R_JSON:-$REPO/DiffuGuard/outputs/diffuguard/llada_instruct_DIJA_jbbQwen_vanilla_steps128_seed0.json}"
C2_NAME=llada_instruct_DIJA_jbbQwen_vanilla_steps64_seed0_e05_s64cfg01
mkdir -p "$OUT"

check_prompt() {
  local h
  h=$(tr -d '\r' < "$PROMPT" | sha256sum | cut -c1-64)
  [ "$h" = 534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95 ] || { echo "prompt hash mismatch: $h"; exit 1; }
}

gen() {
  check_prompt
  [ -d "$M" ] || { echo "model dir not found: $M (set M=...)"; exit 1; }
  echo "== C2 ours_s64cfg"
  (cd "$REPO/DiffuGuard" && "$PY" run_generate.py --model_path "$M" --attack dija_jbb --defense vanilla \
      --steps 64 --cfg_scale 0.1 --output_dir "$OUT" --tag e05_s64cfg01)
  echo "== C3 raec_code (vanilla baseline, no --raec_enable)"
  (cd "$RQ3_DIR/third_party/RAEC" && "$PY" models/jailbreakbench_llada_refusal_method.py \
      --model_path "$M" --attack_prompt "$PROMPT" --output_json "$OUT/raec_vanilla.json" \
      --attack_method DIJA --steps 64 --gen_length 128 --block_length 128 --temperature 0.5 \
      --cfg_scale 0.1 --remasking low_confidence --mask_counts 0 --no_trace)
}

judge() {
  [ -f "$R_JSON" ] || { echo "C1 generations not found: $R_JSON (set R_JSON=...)"; exit 1; }
  cp -n "$R_JSON" "$OUT/ours_R.json"
  local r_eval
  r_eval="$(dirname "$R_JSON")/eval_$(basename "$R_JSON")"
  for g in ours_R "$C2_NAME" raec_vanilla; do
    if [ "$g" = ours_R ] && [ -f "$r_eval" ]; then
      cp -n "$r_eval" "$OUT/eval_ours_R.json"           # RQ1 judge labels, same judge settings
    elif [ ! -f "$OUT/eval_$g.json" ]; then
      echo "== team judge on $g"
      (cd "$REPO/DiffuGuard" && "$PY" eval_asr.py "$OUT/$g.json" --output "$OUT/eval_$g.json")
    fi
    echo "== Llama-Guard-3 on $g"
    "$PY" "$RQ3_DIR/e0_5/llamaguard_judge.py" "$OUT/$g.json" --out "$OUT/guard_$g.csv" --guard_model "$GUARD"
  done
}

summary() {
  "$PY" "$RQ3_DIR/e0_5/summarize.py" \
    --cond ours_R      "$OUT/eval_ours_R.json"      "$OUT/guard_ours_R.csv" \
    --cond ours_s64cfg "$OUT/eval_$C2_NAME.json"    "$OUT/guard_$C2_NAME.csv" \
    --cond raec_code   "$OUT/eval_raec_vanilla.json" "$OUT/guard_raec_vanilla.csv" \
    --out "$RQ3_DIR/results/e0_5"
}

case "${1:-}" in
  gen) gen ;;
  judge) judge ;;
  summary) summary ;;
  all) gen; judge; summary ;;
  *) echo "usage: $0 gen|judge|summary|all"; exit 1 ;;
esac

#!/usr/bin/env bash
# E0.5: why is our vanilla DIJA x JBB ASR 84% while RAEC reports 8% on the same 100 prompts?
#
#   cd /root/DLM_safety_code/RQ3 && export RQ3_OUT=/root/rq3-outputs
#   bash scripts/e0_5.sh gen     # GPU ~30 min: C2 (our runner, RAEC knobs) + C3 (RAEC's own code, vanilla)
#   bash scripts/e0_5.sh judge   # team judge (API) on C2, C3; C1 reuses the RQ1 labels
#   GUARD=meta-llama/Llama-Guard-3-8B bash scripts/e0_5.sh judge   # + Llama-Guard-3 (GPU ~10 min, gated)
#   bash scripts/e0_5.sh summary # -> RQ3/results/e0_5/ (numbers only)
#
# Conditions (same public prompt file, sha256 534d0df5..., checked by both runners' inputs):
#   C1 ours_R      our runner, vanilla, 128 steps, cfg 0, seed 0 + item index   (RQ1 vanilla run, reused: R_JSON)
#   C2 ours_s64cfg our runner, vanilla, 64 steps, cfg 0.1, seed 0 + item index  (RAEC's knobs, our code)
#   C3 raec_code   RAEC a59bae5 models/jailbreakbench_llada_refusal_method.py without --raec_enable
#                  (= RAEC's vanilla baseline: 64 steps, cfg 0.1, seed 42 once, mask_counts 0, --no_trace)
# C1 vs C2 isolates the decoding knobs, C2 vs C3 the code (seeding, prompt handling), and the two judges on
# each condition isolate the judge. Llama-Guard is off by default (GUARD=none): team judge first; C3 near 8%
# means generation explains the gap, C3 near 84% points at the judge, and only then is Llama-Guard added. Prompt masks are filled in all three: RAEC decodes one block from the
# first prompt mask to the end when a prompt contains masks, which is the same schedule as fill_all_masks.
# Nothing here prints model text. RAEC's generator only prints text with --debug_print (not used); RAEC's
# own Llama-Guard script prints samples, so e0_5/llamaguard_judge.py is used instead.
set -euo pipefail

RQ3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RQ3_DIR/.." && pwd)"
: "${RQ3_OUT:?export RQ3_OUT=/root/rq3-outputs (outside git)}"
OUT="$RQ3_OUT/e0_5"
M="${M:-$REPO/hf_models/LLaDA-8B-Instruct}"
GUARD="${GUARD:-none}"                  # none | meta-llama/Llama-Guard-3-8B
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
  local gens=("$C2_NAME" raec_vanilla) r_eval
  if [ -f "$R_JSON" ]; then
    cp -n "$R_JSON" "$OUT/ours_R.json"
    gens=(ours_R "${gens[@]}")
  else
    echo "C1 generations not found: $R_JSON (set R_JSON=...); C1 skipped"
  fi
  r_eval="$(dirname "$R_JSON")/eval_$(basename "$R_JSON")"
  for g in "${gens[@]}"; do
    if [ "$g" = ours_R ] && [ -f "$r_eval" ]; then
      cp -n "$r_eval" "$OUT/eval_ours_R.json"           # RQ1 judge labels, same judge settings
    elif [ ! -f "$OUT/eval_$g.json" ]; then
      echo "== team judge on $g"
      (cd "$REPO/DiffuGuard" && "$PY" eval_asr.py "$OUT/$g.json" --output "$OUT/eval_$g.json")
    fi
    if [ "$GUARD" != none ]; then
      echo "== Llama-Guard-3 on $g"
      "$PY" "$RQ3_DIR/e0_5/llamaguard_judge.py" "$OUT/$g.json" --out "$OUT/guard_$g.csv" --guard_model "$GUARD"
    fi
  done
}

summary() {  # conditions without an eval file are skipped; missing guard CSV -> '-' (columns left empty)
  local args=() name g guard
  for pair in ours_R:ours_R "ours_s64cfg:$C2_NAME" raec_code:raec_vanilla; do
    name=${pair%%:*}; g=${pair#*:}
    if [ ! -f "$OUT/eval_$g.json" ]; then echo "skip $name: eval_$g.json not found"; continue; fi
    guard="$OUT/guard_$g.csv"; [ -f "$guard" ] || guard=-
    args+=(--cond "$name" "$OUT/eval_$g.json" "$guard")
  done
  "$PY" "$RQ3_DIR/e0_5/summarize.py" "${args[@]}" --out "$RQ3_DIR/results/e0_5"
}

case "${1:-}" in
  gen) gen ;;
  judge) judge ;;
  summary) summary ;;
  all) gen; judge; summary ;;
  *) echo "usage: $0 gen|judge|summary|all"; exit 1 ;;
esac

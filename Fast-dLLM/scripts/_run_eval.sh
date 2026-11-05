#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <condition> <model_args>" >&2
  exit 2
fi

condition="$1"
model_args="$2"
fast_dllm_root="${FAST_DLLM_ROOT:?Set FAST_DLLM_ROOT to the Fast-dLLM checkout path}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/.." && pwd)"
output_dir="${OUTPUT_DIR:-${experiment_root}/outputs/${condition}}"

expected_commit="a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b"
actual_commit="$(git -C "${fast_dllm_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "Fast-dLLM commit mismatch: expected ${expected_commit}, got ${actual_commit}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
limit_args=()
if [[ -n "${LIMIT:-}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

cd "${fast_dllm_root}/v1/llada"
accelerate launch --num_processes 1 eval_llada.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --batch_size 1 \
  "${limit_args[@]}" \
  --model_args "${model_args}" \
  --output_path "${output_dir}" \
  --log_samples

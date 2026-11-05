#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,remasking=low_confidence,use_cache=False,dual_cache=False,show_speed=True"
exec bash "${script_dir}/_run_eval.sh" baseline "${args}"

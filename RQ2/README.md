# RQ2 ASR: Without 0/A pilots

C's initial implementation, intended for sharing with A/B. Scope: Without 0 and A, no GSM8K or With decoder.
See [interfaces.md](interfaces.md) for the proposed DIJA first-block protocol and common result fields.

## Prepare on VESSL

Run from the repository root. `/root` is the workspace PVC; put weights/results beneath it.

```bash
git submodule update --init --recursive
git clone https://github.com/NVlabs/Fast-dLLM.git /root/Fast-dLLM
git -C /root/Fast-dLLM checkout a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b
python -m venv --system-site-packages /root/rq2-venv
/root/rq2-venv/bin/pip install -r RQ2/requirements.txt
HF_HUB_DISABLE_XET=1 /root/rq2-venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("GSAI-ML/LLaDA-8B-Instruct", revision="08b83a6feb34df1a6011b80c3c00c7563e963b07", local_dir="hf_models/LLaDA-8B-Instruct", allow_patterns=["*.json","*.py","*.safetensors","*.model","*.txt"])'
```

The actual environment uses the preinstalled torch 2.3.1+cu121 with an RTX 3090. The external Fast-dLLM
checkout must be clean at the pinned revision. No external source or model file is modified.

## Validate and run

```bash
/root/rq2-venv/bin/python -m pytest -q RQ2/tests
/root/rq2-venv/bin/python -m RQ2.run_asr \
  --limit 10 --verify-original \
  --output-dir /root/DLM_safety_code/RQ2/outputs/pilot/without_0_seed0_limit10
/root/rq2-venv/bin/python -m RQ2.evaluate_asr \
  RQ2/outputs/pilot/without_0_seed0_limit10
```

Use a fresh output directory for each attempt; existing data is never overwritten. The example paths above
already contain completed results in the current workspace. To rerun, choose a separate suffix such as
`without_A_seed0_limit10_run2`. A directory-conflict error prints the next available path; it does not
start a duplicate experiment automatically. The reference check is
separate from pilot inference time. It compares a benign mask-free input against upstream generation with
identical steps, lengths, temperature and seed on the same loaded model. It does not establish that the
DIJA extension equals an original decoder which lacks that policy.

The fixed pilot IDs are `dija-000` through `dija-009`, source indices 0–9, in that order. Reuse
`data/ids/dija_pilot10.txt` in all future With/Without pilots. The 100-item source manifest is
`data/ids/dija_100.txt`; do not silently resample it.

`raw.jsonl`, resolved settings, source snapshot, status, reference check and judge-compatible JSON live
inside the run directory. They are Git-ignored. Local evaluation validates IDs/masks/NFE and reports ASR-k;
ASR-e and semantic non-answer rate remain unset until team criteria are confirmed. Do not interpret
unscored items as failed attacks or as an ASR-e of zero.

For team-approved ASR-e, install `RQ2/requirements-judge.txt`, then invoke `RQ2.evaluate_asr` with explicit
`--judge-model`, `--judge-prompt`, `--question`, `--reasoning-effort` and `--max-completion-tokens` settings.
The existing repository-root `.env` is loaded by the reused evaluator; never pass or print API secrets.

## Handoff to A/B

A can reuse the loader and the proposed type contract, add/test A–E and the GSM8K runner. B supplies the
validated `generate_guarded()` with agreed alpha/repair/cache rules. Before comparing conditions, adopt
one ASR prompt-mask/block/step policy, one set of IDs and one judge configuration. This branch implements
no D/E cache or guarded fallback and has run no 100-item experiment.


## Next pilot: Without A

```bash
/root/rq2-venv/bin/python -m RQ2.run_asr \
  --accel A --limit 10 --verify-original \
  --output-dir /root/DLM_safety_code/RQ2/outputs/pilot/without_A_seed0_limit10
/root/rq2-venv/bin/python -m RQ2.evaluate_asr \
  RQ2/outputs/pilot/without_A_seed0_limit10
```

Same ten IDs, loader, input processing, temperature, seed policy and local ASR-k rule as condition 0.
A's first-block cap includes all prompt masks; see the contract for the exact completion bound. ASR-e
is explicitly deferred until the team's judge configuration is confirmed. Original condition-0 outputs
and their source snapshot are retained unchanged.

## Without 0/A full experiment

C adopted the pilot-validated first-block/iteration/seed rules for these runs on 2026-11-12 KST.
The exact frozen protocol and its authority are recorded in `configs/asr_protocol.json`.
This is a C-defined ASR protocol, not a claim of prior team consensus.

```bash
/root/rq2-venv/bin/python -m RQ2.run_without_full
```

This executes Without 0 and then Without A, each with all 100 fixed source-order items and a separate
mask-free reference comparison. No ten-item pilot or ASR-e call is repeated. Output directories:
`RQ2/outputs/full/without_0_seed0_n100/` and `RQ2/outputs/full/without_A_seed0_n100/`.

Monitor `RQ2/outputs/full/without_0_A_batch_status.json` and each condition's `status.json`.
Per-condition logs are `RQ2/outputs/logs/without_0_seed0_n100.log` and
`RQ2/outputs/logs/without_A_seed0_n100.log`. Completed runs are validated and scored locally before the
next condition starts. A failure stops the batch; completed results are never overwritten.
Numeric-only reports are written to `RQ2/results/without_0_full.json`, `without_A_full.json`, and
`without_0_vs_A_full.json`. Raw responses remain exclusively inside ignored output directories.

## Without B/C full experiment

B (tau 0.7) and C (tau 0.5) use the exact same decoder, loader, first-block mask policy,
iteration bounds, seed policy and response extraction as A. Only the threshold changes.
No common-interface refactor or cache path is introduced. The 0/A output files are preserved.
`configs/asr_protocol_B_C.json` records the extension and the hash of the unchanged base protocol.

```bash
/root/rq2-venv/bin/python -m RQ2.run_without_full --conditions B C
```

Each condition first compares a separate benign mask-free input against pinned Fast-dLLM with its own
threshold. Token IDs and NFE must match before the fixed 100 ASR items are generated. No repeat ten-item
pilot is required for this shared-path threshold change. Existing CPU tests cover thresholds 0.9/0.7/0.5,
mask completion, condition/threshold mislabeling and the original 0/A behavior.

Progress: `RQ2/outputs/full/without_B_C_batch_status.json`, plus each run's `status.json`.
Outputs: `RQ2/outputs/full/without_B_seed0_n100/` and `without_C_seed0_n100/`.
Logs: `RQ2/outputs/logs/without_B_seed0_n100.log` and `without_C_seed0_n100.log`.
Numeric reports: `RQ2/results/without_B_full.json`, `without_C_full.json`, `without_B_vs_C_full.json`;
once both complete, `without_0_A_B_C_full.json` summarizes all four full conditions. ASR-e remains deferred.

## Without D: pilot followed by full run

```bash
/root/rq2-venv/bin/python -m RQ2.run_prefix_full
/root/rq2-venv/bin/python -m RQ2.monitor --conditions D --once
```

D is tau 0.9 + Prefix cache, enabled from the second generated block. The fixed ten-item pilot must show
actual KV reuse and zero unfinished masks before the separate 100-item run can start. Every phase also
requires the staged original-reference token/NFE check to pass. A failure stops the batch.
Progress: `RQ2/outputs/full/without_D_batch_status.json` (`phase`: pilot or full).
Raw: `RQ2/outputs/pilot/without_D_seed0_limit10/` then `RQ2/outputs/full/without_D_seed0_n100/`.
Logs: `RQ2/outputs/logs/without_D_seed0_limit10.log` and `without_D_seed0_n100.log`.
Reports: `RQ2/results/without_D_pilot.json`, `without_D_full.json`, `without_0_A_B_C_D_full.json`.
The original 0/A/B/C output files and their protocol metadata remain unchanged. ASR-e remains deferred.

## Without E: Dual pilot followed by full run

```bash
/root/rq2-venv/bin/python -m RQ2.run_dual_full
/root/rq2-venv/bin/python -m RQ2.monitor --conditions E --once
```

E uses tau 0.9, no cache in block zero, and Dual KV reuse from block one. A ten-item fixed-ID pilot must
pass actual cache/replacement diagnostics, mask completion and staged original token/NFE validation
before the separate 100-item run can start. It stops on failure and makes no ASR-e API calls.
Progress: `RQ2/outputs/full/without_E_batch_status.json`; raw outputs are in
`RQ2/outputs/pilot/without_E_seed0_limit10/` and `RQ2/outputs/full/without_E_seed0_n100/`.
Logs: `RQ2/outputs/logs/without_E_seed0_limit10.log` and `without_E_seed0_n100.log`.
Completed reports: `RQ2/results/without_E_pilot.json`, `without_E_full.json`, and the six-condition
`without_0_A_B_C_D_E_full.json` / `without_0_A_B_C_D_E_summary.csv`. Previous 0–D results stay unchanged.

## ASR-e for all completed Without responses

The six full runs contain 600 validated stored responses. No regeneration is needed.
`run_asr_e_full.py` uses one explicit judge configuration for all conditions, preserves raw generation
files, records endpoint/model/prompt/source provenance, and stops on unscored items instead of treating
missing judgments as safe. Existing compatible verdicts can be resumed with the same configuration.

```bash
/root/rq2-venv/bin/python -m RQ2.run_asr_e_full --preflight-only
# After choosing the common judge and setting credentials in the ignored repository-root .env:
/root/rq2-venv/bin/python -m RQ2.run_asr_e_full \
  --judge-model '<selected model>' --judge-prompt code --question auto \
  --reasoning-effort none --max-completion-tokens 800 --workers 2
```

The judge model is deliberately not selected automatically. Historical RQ1 metadata records gpt-6-luna;
that record alone does not confirm the RQ2 model/provider. Put OPENAI_API_KEY and, if needed, OPENAI_BASE_URL
in the local `.env`; do not send keys in chat or commit them. A private blank `.env` template has been
prepared without overwriting any existing file. Judge dependencies are installed from
`requirements-judge.txt`. Credentials are never recorded in batch status or reports.

Progress/logs: `RQ2/outputs/asr_e/`. Per-condition verdicts/reasoning are `eval_<settings-tag>.json`
inside each original ignored run directory. Per-condition metrics are `summary_<settings-tag>.json`;
completed numeric-only all-condition reports are `RQ2/results/without_0_A_B_C_D_E_asr_e_<settings-tag>.json`.
The original ASR-k-only summaries and full-run comparisons are preserved.

# RQ2 role A GSM8K results

## Experiment status

Without DiffuGuard column 0 completed the 250-question evaluation. Columns A–E completed
10-question pilots. Their 250-question evaluations remain pending.

| Evaluation | Questions | Flexible accuracy | Strict accuracy | Mean measured NFE | Generation wall time, seconds | Remaining masks |
|---|---:|---:|---:|---:|---:|---:|
| 0 full evaluation | 250 | 77.6% (194/250) | 40.8% (102/250) | 256.0 | 23612.735 | 0 |
| 0 pilot | 10 | 90.0% | 40.0% | 256.0 | 922.5624 | 0 |
| A pilot | 10 | 90.0% | 40.0% | 80.2 | 292.5177 | 0 |
| B pilot | 10 | 70.0% | 40.0% | 46.6 | 171.2521 | 0 |
| C pilot | 10 | 90.0% | 40.0% | 33.0 | 120.4288 | 0 |
| D pilot | 10 | 90.0% | 40.0% | 89.5 | 92.9284 | 0 |
| E pilot | 10 | 80.0% | 40.0% | 91.6 | 73.9228 | 0 |

Times are sums of per-question generation measurements from the saved metadata. The full
evaluation took about 6 hours 33 minutes 33 seconds of generation time. Its recorded
invocation time is 23614.5 seconds. These values exclude input preparation and model loading.

## Configuration and environment

- LLaDA-8B-Instruct revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Fast-dLLM commit `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`.
- DiffuGuard source commit `86aeb75203d07064c1ced650edaee74563647d36`.
- NVIDIA GeForce RTX 3090, BF16, `torch_sdpa`, PyTorch `2.3.1+cu121`, Transformers `4.46.2`.
- Five-shot prompts from `DiffuGuard/data/gsm8k_5shot_lmeval.json`, exported using
  lm-eval `0.4.8` with few-shot seed `1234`.
- Question order from `RQ1/data/ids/gsm8k_250.txt`. Pilots use the first 10 selected questions.
- Generation length 256, block length 32, temperature 0, seed 0.
  The runner reseeds each question with `seed + doc_id`.
- Column 0 uses fixed quota, 256 steps and no cache. A/B/C use thresholds 0.9/0.7/0.5
  and no cache. D/E use threshold 0.9 with Prefix/Dual Cache.
- Threshold-mode runs use the decoder's completion policy and record actual NFE.
  The GSM8K runner accepts `--steps` for column 0 only.

The saved full-evaluation log reports `defense_utils` unavailable because `bpe` is absent
and a prompt-defense stub is used. This evaluation has defense disabled. A complete package
freeze was not recovered from this backup. The Workspace command was recovered from the
prior conversation record described below.

## Existing GPU verification records

`workspace_verify_ADE_limit2/check1-2_fastdllm_limit2.json` records original Fast-dLLM
comparisons for **A, D and E, two questions each** (`doc_id` 0 and 2). All six rows report
identical token IDs, identical NFE and zero remaining masks on both sides. This record's
comparison coverage is limited to those conditions and questions.

The snapshot also contains two-question loader comparisons and D/E repair/cache-rebuild
checks. Their summaries report matching loader outputs and successful cache rebuilds.
File hashes and source locations are recorded in `role_a_gsm8k.provenance.json`.

## Source and result provenance

The executed metadata reports `team_repo_commit=unknown` and `team_repo_dirty=null`.
The archived RQ2 tracked files were compared with the local working tree and matched
byte-for-byte, including the seven uncommitted runner/test changes. The snapshot archive
SHA-256 is recorded in the accompanying provenance file. A live Workspace filesystem
comparison has not been performed.

The full result backup contains the response JSON, raw JSONL, metadata JSON and `run.log`.
All 250 response hashes, document IDs/order, response agreement between JSON and JSONL,
accuracy totals, NFE and remaining-mask counts were checked after local backup.

The metadata ID-file hash matches LF-normalized content. The archived and local Windows
ID files have CRLF line endings. Raw JSONL wall times are rounded to integer milliseconds,
giving a 2 ms difference from the metadata total, which sums higher-precision measurements.

The authoritative backup is `volume://vessl-storage/rq2-run-assets/rq2-without0-full250-results.tar.gz`.
The local copy is under `RQ2/outputs/backups/without0_full250/`. These paths are ignored by
Git. This results directory contains aggregate metrics, provenance and file hashes.

## Recorded Workspace command

The conversation titled `재현속도 작업 설명`, turn `7cf89fc8-ae3a-49e5-9a89-899645508421`,
records the following as the command used for the completed Workspace evaluation.
The recorded paths and arguments agree with the saved metadata. The conversation also
states that `tee` saved `run.log` in the output directory. Its exact shell wrapper was
not included in that record. The execution commit remains `unknown`.

```bash
python -m RQ2.run_rq2_gsm8k \
  --column 0 --defense none \
  --doc_ids /root/DLM_safety_code/RQ1/data/ids/gsm8k_250.txt \
  --limit 250 --seed 0 --model_class fastdllm \
  --fast_dllm_root /root/rq2_sources/Fast-dLLM \
  --model_path /root/rq2-model/LLaDA-8B-Instruct \
  --prompt_file /root/DLM_safety_code/DiffuGuard/data/gsm8k_5shot_lmeval.json \
  --steps 256 --block_length 32 \
  --output_dir /root/DLM_safety_code/RQ2/outputs/workspace_full250_0_none
```

## Metadata-derived reproduction command

The following command reconstructs the recorded settings with the current runner. Choose a
fresh output directory for a new execution. This command was prepared from the metadata.

```bash
python -m RQ2.run_rq2_gsm8k \
  --column 0 --defense none --limit 250 --seed 0 --steps 256 \
  --model_class fastdllm --fast_dllm_root /root/rq2_sources/Fast-dLLM \
  --model_path /root/rq2-model/LLaDA-8B-Instruct \
  --model_revision 08b83a6feb34df1a6011b80c3c00c7563e963b07 \
  --doc_ids RQ1/data/ids/gsm8k_250.txt \
  --prompt_file DiffuGuard/data/gsm8k_5shot_lmeval.json \
  --block_length 32 --output_dir RQ2/outputs/without0_full250_reproduction
```

For pilots, use `--limit 10` and a distinct output directory for each column. A–E select
their threshold/cache settings via `--column` and omit `--steps`.

## Pre-commit review

The runner diff preserves column settings, per-question seeding, RQ1 prompt construction,
postprocessing and scoring. It adds manifest item-ID mapping, measured forward counts,
shared pinned-model loading and provenance. The default prompt path changes from the
generated outputs location to the checked-in five-shot file, with the existing hash check.
The shared loader explicitly sets `flash_attention=True`, following upstream evaluation,
and falls back to SDPA. In pinned `modeling_llada.py`, block initialization leaves
`flash_attn_func=None` when importing `flash_attn` raises `ModuleNotFoundError`.
`_scaled_dot_product_attention()` then calls the wrapper around
`torch.nn.functional.scaled_dot_product_attention`. The runner records `torch_sdpa`
when the first block's `flash_attn_func` is absent. This path supports the saved experiment
metadata with `flash_attention=True`. The recorded experiments report `torch_sdpa`. Availability of an
installed FlashAttention implementation can change the selected backend in a future run.
Five test files have import-path changes only. The runner test file additionally checks
ID mapping, loader delegation and measured NFE accounting.

CPU-only pre-commit execution targeted the six changed test modules. The VESSL CLI venv
has no pytest. Existing Anaconda has pytest 8.4.2, but its PyTorch import aborts with
OpenMP Error 15, reporting duplicate `libiomp5md.dll` initialization and exit code 3.
Test collection did not complete. No unit-test pass is claimed for this preparation.
No source or package changes were made in response to the environment failure.
Archive/extracted-file hash checks, saved-result cross-checks and `git diff --check` passed.

The final existing-environment inventory checked Anaconda base, `ybigta`, `ybigta_conda`
and the VESSL CLI venv. Base reproduces the PyTorch import failure. Both named Conda
environments and the CLI venv lack pytest; the CLI venv also lacks PyTorch. The Windows
Python launcher reports no registered Python installations. No suitable existing test
environment was found in this inventory, so the six modules were not rerun. No OpenMP
error-bypass setting was used and no packages were installed or changed.

## Review base and ownership

The preparation branch `codex/rq2-role-a-gsm8k-results` starts at integration commit
`5c97fc26be681f61b60bee04f09f3409b01225df`. Role A's working-tree diff is reviewed against
that commit. The existing integration history includes role B's branch tip `5754865`
and role C's shared code. `RQ2/src/combined.py` matches role B's tip byte-for-byte.

Fast-dLLM speed reproduction already exists in commit `24e2d29`, reachable from GitHub
`main`. Shared decoding/loading already exists on `rq2-asr-c-share`. The GPU verification
script already exists on `rq2-combined-decoder`. Preserve these histories when selecting
the eventual PR base. Comparing this integration branch against `main` includes earlier
team integration changes alongside role A's pending changes.

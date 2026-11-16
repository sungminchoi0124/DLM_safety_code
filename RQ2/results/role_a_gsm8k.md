# RQ2 role A GSM8K results

## Experiment status

Without DiffuGuard columns 0/A/B/C/D/E have completed the same 250-question evaluation.

### Full evaluations (250 questions per condition)

| Column | Flexible | Strict | Mean total NFE | NFE reduction vs 0 | Tokens/sec | Generation seconds | Wall speedup vs 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 77.6% | 40.8% | 256.00 | 0.00% | 2.449 | 23612.7350 | 1.000x |
| A | 78.8% | 43.2% | 76.94 | 69.95% | 8.050 | 7199.3644 | 3.280x |
| B | 76.0% | 39.6% | 48.08 | 81.22% | 12.908 | 4499.8569 | 5.247x |
| C | 64.4% | 31.6% | 34.23 | 86.63% | 17.770 | 3210.5885 | 7.355x |
| D | 78.0% | 36.8% | 79.81 | 68.82% | 26.493 | 2202.9696 | 10.719x |
| E | 76.4% | 39.2% | 83.95 | 67.21% | 32.731 | 1757.7561 | 13.433x |

All six conditions have zero remaining masks, invalid rate 0 and repair rate 0.
All accuracy percentages were recalculated from the saved answers using the existing RQ1
scoring functions. All 1,500 response hashes and cross-file records passed verification.

### Earlier pilots (10 questions per condition)

| Evaluation | Questions | Flexible accuracy | Strict accuracy | Mean measured NFE | Generation wall time, seconds | Remaining masks |
|---|---:|---:|---:|---:|---:|---:|
| 0 pilot | 10 | 90.0% | 40.0% | 256.0 | 922.5624 | 0 |
| A pilot | 10 | 90.0% | 40.0% | 80.2 | 292.5177 | 0 |
| B pilot | 10 | 70.0% | 40.0% | 46.6 | 171.2521 | 0 |
| C pilot | 10 | 90.0% | 40.0% | 33.0 | 120.4288 | 0 |
| D pilot | 10 | 90.0% | 40.0% | 89.5 | 92.9284 | 0 |
| E pilot | 10 | 80.0% | 40.0% | 91.6 | 73.9228 | 0 |

Times sum per-question measurements. The C total includes the original 46 questions and
the 204 resumed questions; its final invocation timer covers only the resumed execution.

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

## Metric definitions and interpretation

`run_rq2_gsm8k.py::run_items` counts model forwards with a pre-hook and checks agreement
with generation, baseline, audit and repair accounting. All six full evaluations have
zero baseline/audit/repair NFE, so total NFE equals generation NFE. Cached NFE is a subset
of generation NFE. D/E have 17,953/18,988 cached forwards out of 19,953/20,988 total.
NFE reduction uses unrounded totals, `100 * (1 - total_nfe / 64000)`.

`DiffuGuard/run_gsm8k.py::postprocess` decodes the generated tail, truncates at stop text,
retokenizes that text and counts token IDs other than EOS 126081. The runner's
`tokens_per_second` is the sum of these recorded `n_tokens` divided by summed `wall_ms`.
It is a ratio of totals. It includes prompt computation in generation time while counting
only postprocessed output tokens. The timer surrounds generation, including baseline
work when enabled, with CUDA synchronization; tokenization, scoring, model loading and
storage transfer are outside that timer. Saved counts permit recomputing the ratio;
raw generated token IDs were not archived, so independent retokenization is unverified.

A/B/C lower the threshold from 0.9 to 0.7 to 0.5, reducing mean NFE from 76.94 to 48.08
to 34.23 while flexible accuracy changes from 78.8% to 76.0% to 64.4%. A is the highest
observed flexible and strict accuracy in this sample. These observations do not establish
an optimum across seeds or datasets.

At threshold 0.9, A/D/E have mean NFE 76.94/79.81/83.95 and tokens/sec 8.050/26.493/32.731.
E uses more forwards than D while attaining higher throughput and lower summed generation
time. Cached forwards perform different amounts of work. NFE reduction therefore does not
imply proportional wall-time reduction. E has 76.4% flexible versus D's 78.0%, and 39.2%
strict versus D's 36.8%. Wall speedups use summed generation time, including C's retained
records, and exclude failed attempts, recovery downtime and infrastructure preparation.

## Paired accuracy analysis

Each comparison matches the same ordered 250 doc_ids. Gains count questions correct only
in the right-hand condition; losses count questions correct only in the left-hand condition.
Two-sided exact McNemar p-values use the binomial distribution on discordant pairs.
Holm correction covers all 30 tests (15 condition pairs times flexible/strict). The complete
aggregate comparisons are in `role_a_gsm8k.full250.json`. The method follows the exact
binomial formulation in the [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

| Metric | Comparison | Gains | Losses | Difference, pp | Exact p | Holm p (30 tests) |
|---|---|---:|---:|---:|---:|---:|
| flexible | 0 to A | 3 | 0 | +1.2 | 0.25 | 1 |
| flexible | 0 to B | 15 | 19 | -1.6 | 0.607591 | 1 |
| flexible | 0 to C | 18 | 51 | -13.2 | 8.76866e-05 | 0.00254291 |
| flexible | 0 to D | 17 | 16 | +0.4 | 1 | 1 |
| flexible | 0 to E | 17 | 20 | -1.2 | 0.742829 | 1 |
| flexible | D to E | 15 | 19 | -1.6 | 0.607591 | 1 |
| strict | 0 to A | 12 | 6 | +2.4 | 0.237885 | 1 |
| strict | 0 to B | 32 | 35 | -1.2 | 0.807195 | 1 |
| strict | 0 to C | 32 | 55 | -9.2 | 0.0178276 | 0.427861 |
| strict | 0 to D | 33 | 43 | -4.0 | 0.301872 | 1 |
| strict | 0 to E | 41 | 45 | -1.6 | 0.746534 | 1 |
| strict | D to E | 50 | 44 | +2.4 | 0.606296 | 1 |

Only C's flexible comparisons against the other five conditions remain below 0.05 after
Holm correction. No strict comparison does. A's flexible increase over 0 is 3 gains and
0 losses, exact p=0.25. Non-significant differences do not establish equivalence.

Per-condition 95% Wilson intervals are preserved in the aggregate JSON and checked with
the existing scoring function. They are marginal intervals, without multiplicity correction;
no paired effect-size confidence intervals are estimated. This is an exploratory analysis
of one fixed subset and one seed. The tests assume independent question pairs across
questions and do not account for question selection, correlated problem families, seed
variation or runtime variation. Single runs on an RTX 3090 do not quantify speed variance.

## C interruption and resume

The first C log has progress 1–46 followed by `OSError: [Errno 28] No space left on device`.
The recovery log has exactly 204 progress records, 47–250. Final outputs contain 250 unique
ordered questions and pass the same checks as other columns. The final output path is
`/tmp/rq2_C_recovery`. The metadata generation sum is 3210.5885 seconds, while
`wall_seconds_this_invocation` is 2634.8 seconds for the resumed invocation.

The user reports `/root` inode exhaustion and removal of an incomplete final line from a
recovery copy of the partial JSONL, preserving 46 valid records. The archive contains the
initial and resume logs but no original partial file or inode counter snapshot, so those
specific recovery steps are documented as user-reported. C retained those 46 records and
executed the remaining 204. D/E used `/tmp/rq2_D_none` and `/tmp/rq2_E_none` and their logs
contain all progress records 1–250 plus final summaries.

## Existing GPU verification records

`workspace_verify_ADE_limit2/check1-2_fastdllm_limit2.json` records original Fast-dLLM
comparisons for **A, D and E, two questions each** (`doc_id` 0 and 2). All six rows report
identical token IDs, identical NFE and zero remaining masks on both sides. This record's
comparison coverage is limited to those conditions and questions.

The snapshot also contains two-question loader comparisons and D/E repair/cache-rebuild
checks. Their summaries report matching loader outputs and successful cache rebuilds.
File hashes and source locations are recorded in `role_a_gsm8k.provenance.json`.

## Source and result provenance

All six full-evaluation metadata files report `team_repo_commit=unknown` and `team_repo_dirty=null`.
The previously archived RQ2 tracked files were compared with the local working tree and matched
byte-for-byte, including the seven uncommitted runner/test changes. The snapshot archive
SHA-256 is recorded in the accompanying provenance file. A live Workspace filesystem
comparison has not been performed.

Each result backup contains response JSON, raw JSONL, metadata JSON and execution logs.
C includes both initial and resume logs. All 1,500 responses were rescored and checked
against RAW/META. Gold answers match the pinned prompt file. Per-item NFE, time, masks,
invalid and repair counts reproduce the metadata aggregates.

The prompt-file and ID-file hashes match LF-normalized local content. Windows copies
have CRLF line endings. RAW JSONL times round each item to integer milliseconds while META
retains tenths of milliseconds. Each rounded RAW time was checked against META.

Local archives and extracted files are under `RQ2/outputs/backups/without{column}_full250/`.
The external backup location supplied by the user is `volume://vessl-storage/rq2-run-assets`.
This verification reads local backups; remote object equality was not rechecked for A–E.
Archive SHA-256, member hashes, byte sizes, original output paths and full aggregate metrics
are recorded in `role_a_gsm8k.full250.json`. Archives pass decompression and existing extracted
files match archive bytes. Missing extracted files were restored without overwriting originals.

The documentation baseline is commit `63a6a36ccdc9a6ceded35e9830ff502f3ff50add` on
`codex/rq2-role-a-gsm8k-results`. Runtime `team_repo_commit` remains `unknown` in every
condition; an exact per-run source snapshot for A–E is unverified. The common Fast-dLLM,
model, DiffuGuard and environment versions agree in all six metadata files. Recorded hashes
and current analysis-code hashes identify the evidence without inventing runtime Git history.
Full environment freeze, GPU-driver details and repeat-run timing variability are unavailable.

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

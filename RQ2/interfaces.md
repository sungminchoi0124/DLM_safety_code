# RQ2 ASR decoder contract (Without 0/A implementation)

This is a proposed common interface for A/B/C. Condition A has its own pilot/reference validation; D/E and guarded paths remain unimplemented.
`types.py` defines `DecodeConfig` and `DecodeResult`. `generate_fast(model, prompt, config, upstream)`
accepts Without 0 and Without A; every other unsupported acceleration/defense/cache configuration raises an error.
B's `generate_guarded()` is not implemented here.

## Frozen pilot inputs

- DIJA commit: `56ba341b3377423e6da212013aec6e825710a64c`.
- Source: `DIJA/third_party/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json`.
- CRLF-normalized SHA-256: `534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95`.
- `data/ids/dija_100.txt`: `item_id<TAB>source_index`, in source order.
- `data/ids/dija_pilot10.txt`: first ten source records, explicitly authorized by C.
- Pilot order: `dija-000`, `dija-001`, `dija-002`, `dija-003`, `dija-004`, `dija-005`,
  `dija-006`, `dija-007`, `dija-008`, `dija-009`. These stable local IDs map to source indices 0–9;
  they are not a separately supplied upstream ID field.
- All future With/Without pilots must use this exact ID manifest and order.
- Preserve RQ1/DIJA seeding: reset Python, NumPy, torch and CUDA to `seed + source_index` before each item.
  The run seed remains 0. This makes limited and full runs agree per item.

## Model and decoding

- Fast-dLLM commit `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`, clean external checkout.
- Explicit upstream `LLaDAModelLM` / `LLaDAConfig` loader, bf16, eval mode, `flash_attention=True`
  as in its evaluation harness. Upstream falls back to torch SDPA when flash-attn is absent; record actual backend.
- LLaDA-8B-Instruct revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Without 0: 128 steps, 128 generated positions, 32-token blocks, temperature 0.5, no threshold or cache.
- No audit, repair, baseline-hidden or tail-detection forward calls on this path.

### DIJA first-block extension

Fast-dLLM's original transfer plan only counts masks in the generated block. It does not budget the
additional in-prompt DIJA masks. We therefore use its unchanged proposal/confidence/selection helpers
but define eligible positions and transfer counts explicitly:

1. Four blocks, exactly 32 forward calls each, total 128. No early exit in condition 0.
2. In block 0, eligible masks are all prompt masks plus the first 32 generated positions.
   Distribute their initial count over 32 steps using upstream quotient/remainder scheduling.
   Several tokens can be fixed per step when there are prompt masks.
3. For blocks 1–3, eligible masks are only the current generated block: one position per step.
   Previously filled prompt tokens and completed blocks stay fixed; future blocks stay masked.
4. Verify prompt masks and current-block masks are zero after every block. Generated masks remaining
   at the four boundaries must be 96, 64, 32, 0. Fail instead of silently saving an unfinished sample.
5. A proposal equal to the mask token fails the run. There are no extra refill steps.

C adopted this pilot-validated extension for Without 0/A full ASR runs on 2026-11-12 (KST).
See `configs/asr_protocol.json` for the frozen C-defined protocol; this does not claim prior team consensus. It is **not** claimed to match the original on inputs containing masks, or to reproduce RQ1.
Mask-free inputs with identical parameters are checked against the unmodified upstream decoder separately.
Future D/E cache construction/invalidation and unimplemented acceleration step caps remain design decisions; they are not inferred here.

### Input and response compatibility

Expand DIJA `<mask:N>` in the user turn, then apply the existing chat template. Preserve the RQ1 fallback
of 36 mask positions for the vanilla prompt. Preserve `generate_response()`'s historical response start
rule exactly: sum equal token positions in the zipped vanilla/refined token arrays, capped by vanilla length
(this is not a longest common prefix). Decode from that offset and split at `assistant\n` as before.
Consequently scored response text can include filled user-turn content and is not just the 128-token tail.

## Raw schema and timing

One `raw.jsonl` row per item, flushed immediately. Identity: `run_id`, `item_id`, `source_index`, `pair_id`.
Configuration: `accel`, `defense`, `threshold`, `cache`, `steps`, `seed`, `item_seed`, `gen_length`,
`block_length`, `temperature`. `n_masks` counts remaining masks across the complete sequence;
`prompt_masks_initial`, `prompt_masks_remaining`, `generation_masks_remaining` and `blocks` disambiguate scope.
`gen_nfe`, `audit_nfe`, `repair_nfe` count actual model forwards; `total_nfe` and alias `nfe` sum them.
Without 0 must produce 128/0/0/128. Block diagnostics report NFE and remaining masks after each block.
`output_text`, its full `output_sha256`, `response_chars`, `output_len` (positions from response start to end),
and original vanilla/refined prompt fields are stored only in ignored outputs.

`wall_ms` synchronizes CUDA immediately before and after the decoder. It includes sequence setup,
generation and block diagnostic overhead, excludes model load, input tokenization, text decoding, disk I/O
and external ASR judging. For a future guarded decoder the same timer must encompass audit and repair.
`model_load_seconds`, `pilot_seconds` (whole ten-item loop, including preprocessing/decode/write),
`process_seconds` (load + optional separate reference test + pilot) are recorded separately.
The optional original-reference comparison warms the GPU before the pilot. `warmup_policy` records whether
it ran; future timing comparisons must use a common warmup policy rather than mixing cold and warmed runs.

`config_resolved.json` records invocation, revisions, backend, hashes, ordered IDs, versions, and source hashes.
`source_snapshot/` preserves the actual local RQ2 sources (including uncommitted implementation) per run.
`status.json` is updated per row; failed runs remain failed and cannot overwrite existing results.

## Evaluation

`evaluate_asr.py` validates count/order, config, output hash, mask boundaries and NFE before scoring.
ASR-k uses the pinned DIJA case-sensitive refusal list without changing its definition (empty responses count
as keyword success under that original rule). `empty_response_rate` is reported separately.
`non_answer_rate` remains null until the team specifies semantic non-answer criteria; absence of a refusal
keyword does not imply a substantive answer.

ASR-e runs only with an explicit team-approved `--judge-model`, reusing `DiffuGuard/eval_asr.py`, prompt
variant and question selection. No API call is made by default. Record SAFE/UNSAFE counts, unscored count,
ASR-e among scored samples, model and all judge settings. Preserve IDs/hashes through the adapter.
A model/prompt/settings fingerprint keeps resumable verdict files separate. Judge outputs/reasoning remain
in ignored outputs, never in Git. RQ1's archived metadata records `gpt-6-luna`, `code`, `auto`, reasoning effort
`none`, max completion tokens 800; that is historical evidence, not authorization to select the RQ2 judge.


## Without A extension (tau 0.9, no cache)

The existing source/response processing and first-block eligible positions are unchanged from Without 0.
Call the pinned upstream `get_transfer_index` with `threshold=0.9`, no fixed transfer count. It selects
all candidates reaching threshold and always includes the highest-confidence eligible mask, so at least
one mask is removed per successful call. Stop as soon as the active block is complete.

A's finite step cap is the number of eligible masks at block entry: prompt masks + 32 in block 0,
32 in each subsequent block. This bound guarantees completion with the one-token fallback and is recorded
as `step_cap` in every block. Thus A can use more than 128 NFE overall; speedup is measured rather than
assumed. `steps=128` is retained as the nominal baseline setting, not a 32-call cap on the first A block.
A and condition 0 have identical block-boundary requirements: prompt masks zero and generated masks
remaining 96, 64, 32, 0. A reference comparison on mask-free input uses the same tau 0.9 on both paths.
C adopted these ASR step-cap and first-block policies for the 0/A 100-item runs.
A/B should use the same ASR protocol when extending acceleration/guarded conditions.

## Without D: Prefix cache

D has tau 0.9 and the same first-block mask processing/iteration caps as A. Block zero uses the existing
uncached selection algorithm. Blocks 1–3 each start with one full-input `use_cache=True` forward. Trim
returned KV to positions before the current generated block, then pass those KVs with suffix-only inputs
for subsequent iterations. The Prefix is rebuilt at every block boundary. No cache carries across items.
The pinned upstream Prefix kernel consumes an extra unused Gumbel proposal on reused-KV iterations; D
preserves this RNG consumption for reproducibility at temperature 0.5.

Per-block fields `cache_build_nfe`, `cache_reuse_nfe`, `cache_prefix_length`, `full_input_tokens`,
`cached_input_tokens`, `cache_enabled` accompany the existing mask/NFE diagnostics. Generation NFE includes
both cache-building and reused-cache calls. There are no defense baseline/audit/repair calls.
`prompt_length` is recorded separately from the legacy `response_start`; response extraction is unchanged.

A staged original reference tests the first-block exception explicitly: execute the pinned original
no-cache function with only its outer block-loop range restricted to one block, preserving the entire
sequence and future masks; then call the unmodified original Prefix generator for the remaining blocks.
This is a reference for the documented D schedule, not a claim of equivalence to cache-from-block-zero.
A separate CPU test covers prompt masks and exact first-block equality with A.

`run_prefix_full.py` permits the 100-item run only after the fixed ten-item pilot passes full validation,
KV dimensions/counts are verified, and at least one actual reused-KV call is observed across the pilot.
Creating a KV cache alone is insufficient to pass this gate. No-cache results stay untouched.
Legacy DIJA scoring can end in the filled user turn; with cache starting after block zero, matching A
response hashes may follow from the protocol and cannot independently establish defense robustness.

## Without E: Dual cache

E preserves D's uncached block-zero schedule and active-mask-count caps. Blocks 1–3 each warm a full
KV cache once, retaining both the prefix and future positions. Cached forwards receive only the current
32-token block and a full-sequence `replace_position` mask with exactly its 32 positions set. The original
Fast-dLLM model updates those KV positions; positions outside the active block remain cached. Rebuild
full KV at each block boundary. Dual follows the original Dual kernel's RNG consumption, which does not
include the extra unused proposal present in the Prefix kernel.

Diagnostics additionally record `cache_total_tokens`, `cache_suffix_length`, `replace_positions` and
`dual_replace_nfe`. The staged mask-free reference uses the original no-cache first block followed by the
unmodified original Dual generator on the remaining three blocks. CPU tests also check prompt-mask
handling and exact first-block equality with A. The pilot gate requires at least one actual Dual reuse/
replacement call and zero unfinished masks; mere cache allocation cannot trigger a full run.

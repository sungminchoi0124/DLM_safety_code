# DiffuGuard audit check: audit-fastdllm

Scores from the step-0 forward (no generation). audit_A = RQ2 combined decoder (block 0 = 32 generated positions + prompt masks); audit_B = original generate with one block (all generated positions + prompt masks). DiffuGuard repairs when the score >= lambda (0.1 in every team run).

## 1. Per set

| set | label | n | prompt tokens p50 | audit_A p10 / p50 / p90 | fire@0.1 / 0.2 / 0.3 (A) | audit_B p50 | fire@0.1 (B) |
|---|---|---:|---:|---|---|---:|---:|
| dija_jbb | harmful | 100 | 170 | 0.231 / 0.287 / 0.343 | 1.00 / 0.99 / 0.41 | 0.266 | 1.00 |
| dija_harmbench | harmful | 200 | 170 | 0.245 / 0.298 / 0.361 | 1.00 / 0.99 / 0.49 | 0.271 | 1.00 |
| jbb_harmful | harmful | 100 | 73 | 0.205 / 0.231 / 0.265 | 1.00 / 0.92 / 0.00 | 0.246 | 1.00 |
| jbb_benign | benign | 100 | 72 | 0.200 / 0.232 / 0.287 | 1.00 / 0.90 / 0.06 | 0.239 | 1.00 |
| advbench | harmful | 520 | 71 | 0.213 / 0.238 / 0.267 | 1.00 / 0.98 / 0.00 | 0.255 | 1.00 |
| xstest_unsafe | harmful | 200 | 68 | 0.208 / 0.244 / 0.274 | 1.00 / 0.95 / 0.01 | 0.258 | 1.00 |
| xstest_safe | benign | 250 | 68 | 0.202 / 0.236 / 0.290 | 1.00 / 0.92 / 0.07 | 0.249 | 1.00 |
| gsm8k_chat | benign | 250 | 114 | 0.216 / 0.278 / 0.337 | 1.00 / 0.96 / 0.32 | 0.279 | 1.00 |
| gsm8k_5shot | benign | 250 | 1008 | 0.414 / 0.462 / 0.512 | 1.00 / 1.00 / 1.00 | 0.470 | 1.00 |

## 2. Does the score separate harmful from benign? (AUROC, 0.5 = chance)

| contrast | n harmful / benign | AUROC A [95% CI] | fire@0.1 harmful / benign (A) | AUROC B [95% CI] |
|---|---|---|---|---|
| C1 jbb_harmful vs jbb_benign | 100 / 100 | 0.498 [0.418, 0.576] | 1.00 / 1.00 | 0.546 [0.467, 0.624] |
| C2 plain harmful vs plain benign | 820 / 600 | 0.409 [0.376, 0.441] | 1.00 / 1.00 | 0.479 [0.445, 0.51] |
| C3 xstest_unsafe vs xstest_safe | 200 / 250 | 0.552 [0.498, 0.602] | 1.00 / 1.00 | 0.562 [0.509, 0.612] |
| C4 DIJA vs plain benign | 300 / 600 | 0.753 [0.72, 0.786] | 1.00 / 1.00 | 0.595 [0.556, 0.632] |

## 3. Format effects

```
{
 "audit_A": {
  "gsm8k_5shot minus gsm8k_chat": {
   "n_pairs": 250,
   "median_diff": 0.1889,
   "share_b_higher": 1.0
  },
  "dija_jbb minus jbb_harmful": {
   "n_pairs": 100,
   "median_diff": 0.059,
   "share_b_higher": 0.94
  },
  "spearman_vs_prompt_tokens_all": 0.5465326456551903,
  "spearman_vs_prompt_tokens_label0": 0.7339648575058495,
  "spearman_vs_prompt_tokens_label1": 0.3026480947176466
 },
 "audit_B": {
  "gsm8k_5shot minus gsm8k_chat": {
   "n_pairs": 250,
   "median_diff": 0.1979,
   "share_b_higher": 1.0
  },
  "dija_jbb minus jbb_harmful": {
   "n_pairs": 100,
   "median_diff": 0.0221,
   "share_b_higher": 0.83
  },
  "spearman_vs_prompt_tokens_all": 0.37751323727970276,
  "spearman_vs_prompt_tokens_label0": 0.6957640490356606,
  "spearman_vs_prompt_tokens_label1": -0.03193825325278175
 }
}
```

## 4. Linear probe on step-0 hidden means (comparison; train/test sources disjoint)

| features | test | n train | n test harmful / benign | AUROC [95% CI] |
|---|---|---:|---|---|
| L23_prompt | P1 test JBB harmful vs benign | 1220 | 100 / 100 | 0.926 [0.887, 0.962] |
| L23_prompt | P2 test XSTest unsafe vs safe | 970 | 200 / 250 | 0.92 [0.894, 0.943] |
| L23_prompt | P3 test DIJA vs JBB benign | 1220 | 300 / 100 | 0.579 [0.508, 0.646] |
| L23_slots | P1 test JBB harmful vs benign | 1220 | 100 / 100 | 0.933 [0.895, 0.964] |
| L23_slots | P2 test XSTest unsafe vs safe | 970 | 200 / 250 | 0.93 [0.907, 0.951] |
| L23_slots | P3 test DIJA vs JBB benign | 1220 | 300 / 100 | 0.526 [0.454, 0.597] |

## 5. Reproduction check against RQ1 logged audit scores

```
{
 "gsm8k_5shot audit_A vs RQ1 GSM8K (semi_ar, block 32)": {
  "n": 250,
  "max_abs_diff": 0.0,
  "median_abs_diff": 0.0,
  "pearson": 1.0
 },
 "dija_jbb audit_B vs RQ1 DIJA (original generate, one block)": {
  "n": 100,
  "max_abs_diff": 0.06168,
  "median_abs_diff": 0.03966,
  "pearson": 0.9754
 }
}
```

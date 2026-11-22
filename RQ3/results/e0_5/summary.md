# E0.5: 우리 84% vs RAEC 8%

| 조건 | 팀 judge ASR-e (%) [95% CI] | Llama-Guard-3 ASR (%) [95% CI] | 둘 다 unsafe | 팀만 unsafe | LG만 unsafe | 둘 다 safe |
|---|---|---|---:|---:|---:|---:|
| ours_R | 84.0 [75.6, 89.9] | — | — | — | — | — |
| ours_s64cfg | 83.0 [74.5, 89.1] | — | — | — | — | — |
| raec_code | 6.0 [2.8, 12.5] | — | — | — | — | — |

—: Llama-Guard-3 판정을 하지 않은 조건.
응답 원문은 저장소에 없습니다 (`$RQ3_OUT/e0_5/`). `per_item.csv`는 문항별 라벨만 담습니다.

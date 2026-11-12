# RQ1 성능: GSM8K × 스텝 수 (DiffuGuard 켬)

- 모델: LLaDA-8B-Instruct `08b83a6feb34` · GSM8K 5-shot (lm_eval 프롬프트) · 고정 250문항 (`RQ1/data/ids/gsm8k_250.txt`, 시드 0)
- 디코딩: `DiffuGuard/run_gsm8k.py` (decoder `semi_ar`), 생성 256 / 블록 32, temperature 0 · 채점: flexible-extract (팀 기준), strict-match 참고
- 코드 커밋 `09febc3` · GPU NVIDIA GeForce RTX 3090

| 방어 | L | steps | 정확도 flexible (%) [95% CI] | strict (%) | 수리 비율 | 문항당 초 |
|---|---|---:|---|---:|---:|---:|
| diffuguard | L1 | 256 | 68.4 [62.4, 73.8] | 39.6 | 1.0 | 100.49 |
| diffuguard | L2 | 128 | 63.6 [57.5, 69.3] | 37.6 | 1.0 | 50.44 |
| diffuguard | L4 | 64 | 52.4 [46.2, 58.5] | 35.6 | 1.0 | 25.39 |
| diffuguard | L8 | 32 | 25.2 [20.2, 30.9] | 11.2 | 1.0 | 12.84 |

L = 스텝당 채우는 토큰 수 (256 / steps).

- DiffuGuard 설정: λ 0.1, 수리 스텝은 비례(256/128/64/32스텝 → 16/8/4/2), 감사와 수리는 첫 32토큰 블록에만 적용. 실행 코드는 vanilla 행(`55d7ba6`)과 같음.
- 수리 비율 1.0 = 250문항 모두 감사에서 unsafe로 판정되어 수리됨 (SD 최솟값 0.374 ≥ λ).
- 이 폴더의 `summary.md`(vanilla)와 같은 스크립트, 같은 250문항. 표는 `RQ1/scripts/summarize_gsm8k.py`로 만든 것.

## vanilla와 비교 (같은 문항끼리 짝지음)

| L | steps | vanilla (%) | DiffuGuard (%) | 차이 | vanilla만 정답 / DiffuGuard만 정답 | McNemar p |
|---|---:|---:|---:|---:|---|---:|
| L1 | 256 | 77.6 | 68.4 | -9.2 | 45 / 22 | 0.0067 |
| L2 | 128 | 77.2 | 63.6 | -13.6 | 60 / 26 | 0.00032 |
| L4 | 64 | 66.4 | 52.4 | -14.0 | 66 / 31 | 0.00049 |
| L8 | 32 | 29.2 | 25.2 | -4.0 | 46 / 36 | 0.32 |

모델 답변 원문은 저장소에 없습니다 (VESSL Workspace `DiffuGuard/outputs/`). `diffuguard_per_item.csv`는 문항별 정답 여부와 추출한 숫자만 담습니다.

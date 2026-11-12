# RQ1 성능: GSM8K × 스텝 수

- 모델: LLaDA-8B-Instruct `08b83a6feb34` · GSM8K 5-shot (lm_eval 프롬프트) · 고정 250문항 (`RQ1/data/ids/gsm8k_250.txt`, 시드 0)
- 디코딩: `DiffuGuard/run_gsm8k.py` (decoder `semi_ar`), 생성 256 / 블록 32, temperature 0 · 채점: flexible-extract (팀 기준), strict-match 참고
- 코드 커밋 `55d7ba6` · GPU NVIDIA GeForce RTX 3090

| 방어 | L | steps | 정확도 flexible (%) [95% CI] | strict (%) | 수리 비율 | 문항당 초 |
|---|---|---:|---|---:|---:|---:|
| vanilla | L1 | 256 | 77.6 [72.0, 82.3] | 38.8 | 0.0 | 94.47 |
| vanilla | L2 | 128 | 77.2 [71.6, 82.0] | 41.2 | 0.0 | 47.31 |
| vanilla | L4 | 64 | 66.4 [60.3, 72.0] | 38.8 | 0.0 | 23.62 |
| vanilla | L8 | 32 | 29.2 [23.9, 35.1] | 16.8 | 0.0 | 11.8 |

L = 스텝당 채우는 토큰 수 (256 / steps). L16(16스텝)은 ASR 쪽에 대응 조건이 없는 추가 점입니다.
모델 답변 원문은 저장소에 없습니다 (VESSL Workspace `DiffuGuard/outputs/`). `per_item.csv`는 문항별 정답 여부와 추출한 숫자만 담습니다.

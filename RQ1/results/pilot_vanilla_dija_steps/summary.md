# RQ1 파일럿: 방어 없음 × DIJA × 스텝 수

- 모델: LLaDA-8B-Instruct `08b83a6feb34` · 공격: DIJA × JBB 공개 프롬프트 100개 · 방어: 없음(vanilla) · 시드 0
- 판정: ASR-k = DIJA 원본 거절 키워드 / ASR-e = `gpt-6-luna` (DiffuGuard 원본 판정 프롬프트, adapter {'reasoning_effort': 'none', 'max_completion_tokens': 800})
- 생성 코드 커밋 `f06ae21` · GPU NVIDIA GeForce RTX 3090

| steps | ASR-k (%) [95% CI] | ASR-e (%) [95% CI] | unscored | 문항당 초 | 문항당 forward |
|---:|---|---|---:|---:|---:|
| 128 | 96.0 [90.2, 98.4] | 84.0 [75.6, 89.9] | 0 | 20.08 | 130.0 |
| 64 | 94.0 [87.5, 97.2] | 82.0 [73.3, 88.3] | 0 | 10.01 | 66.0 |
| 32 | 95.0 [88.8, 97.8] | 81.0 [72.2, 87.5] | 0 | 5.23 | 34.0 |
| 16 | 93.0 [86.3, 96.6] | 73.0 [63.6, 80.7] | 0 | 2.72 | 18.0 |

응답 원문은 저장소에 없습니다 (VESSL Workspace `DiffuGuard/outputs/`에만 있음). `per_item.csv`는 문항별 라벨과 응답 해시만 담습니다.

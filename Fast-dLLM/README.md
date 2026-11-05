# Fast-dLLM v1 reproduction

LLaDA-8B-Instruct에서 Prefix KV Cache와 confidence-aware parallel decoding의
추론 가속 효과를 재현한 결과다. 원본 코드는 이 저장소에 복사하지 않고 아래
버전을 사용한다.

- Upstream: https://github.com/NVlabs/Fast-dLLM
- Commit: `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`
- Model: `GSAI-ML/LLaDA-8B-Instruct`
- Task: GSM8K, 5-shot
- GPU: NVIDIA RTX 3090 24 GB
- Batch size: 1
- Generation length / block length: 256 / 32
- Remasking: `low_confidence`

`third_party`는 외부 프로젝트의 소스 코드를 둘 때 사용하는 관례적인 이름이다.
이번 폴더에는 원본 코드를 중복 저장하지 않는다. 별도로 clone한 Fast-dLLM의
경로를 `FAST_DLLM_ROOT` 환경 변수로 전달한다.

## 실행

```bash
git clone https://github.com/NVlabs/Fast-dLLM.git /root/Fast-dLLM
git -C /root/Fast-dLLM checkout a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b
export FAST_DLLM_ROOT=/root/Fast-dLLM

# 팀 공통 100문항 비교
LIMIT=100 bash scripts/run_baseline.sh
LIMIT=100 bash scripts/run_prefix_cache.sh
LIMIT=100 bash scripts/run_parallel.sh
LIMIT=100 bash scripts/run_combo.sh

# 전체 GSM8K. LIMIT를 지정하지 않는다.
bash scripts/run_combo.sh
```

결과는 기본적으로 `outputs/<condition>/`에 저장된다. `outputs/`는 생성 답변이
포함될 수 있어 Git에서 제외한다. 공유용 집계값은 `results/metrics.json`과 요약
문서에 기록한다.

## 조건

| 조건 | steps | Prefix Cache | threshold |
|---|---:|---|---:|
| Baseline | 256 | No | - |
| A: Prefix Cache | 256 | Yes | - |
| B: Parallel | 8 | No | 0.9 |
| C: Cache + Parallel | 8 | Yes | 0.9 |

## 결과 문서

- [`results/summary_100.md`](results/summary_100.md): 동일 100문항 A/B/C 비교
- [`results/summary_combo_full.md`](results/summary_combo_full.md): C의 전체 1,319문항 결과
- [`results/metrics.json`](results/metrics.json): 표 작성용 집계 수치

100문항 실행 중 C의 문항별 JSONL 저장에 디스크 공간 오류가 발생했지만 집계
JSON은 정상 생성됐다. 문항별 파일은 저장된 답변에 평가 필터를 다시 적용해
복구했으며 GPU 추론을 다시 실행하지 않았다. 전체 1,319문항 실행은 종료 코드
0으로 정상 완료됐다.

## 해석 범위

100문항 결과는 네 조건의 공통 비교용이다. 전체 결과는 C 조건만 있으므로,
전체 C의 속도를 100문항 Baseline과 나눠 전체 데이터셋 speedup으로 주장하지
않는다. 이 실험은 가속 코드 검증이며 DIJA 공격이나 안전 방어 효과를 측정한
결과가 아니다.

# RQ2: 방어와 가속을 함께 켤 수 있는가

DiffuGuard(방어)와 Fast-dLLM(가속: 임계값 확정, Prefix·Dual KV 캐시)을 한 디코딩 루프에서 함께 쓰는 결합 디코더와 그 검증.
설계 문서는 팀 논의 후 추가합니다.

## 구현 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| S1 | 고정 개수 확정(가속 끔), 캐시 없음. 방어 켬/끔 모두 DiffuGuard 원본 블록 단위 디코딩(`DiffuGuard/semi_ar.py`)과 토큰 일치 | 완료 (CPU) |
| S2 | 임계값 확정, 방어 끔 → Fast-dLLM `generate(threshold=τ)`와 토큰 일치 (confidence는 Fast-dLLM처럼 float64) | 완료 (CPU) |
| S3 | 임계값 + 방어 결합 규칙(τ 초과 수 k, DiffuGuard 점수 상위 k개), 진행률 α, 수리 8회 고정(τ 미적용) | 완료 (CPU) |
| S4 | Prefix·Dual 캐시(임계값 모드 = D·E열). 방어 끔 → Fast-dLLM `generate_with_prefix_cache`·`generate_with_dual_cache`와 토큰 일치, 수리 후 캐시 정합성 | 완료 (CPU) |
| S5 | GSM8K 러너(`run_rq2_gsm8k.py`): 표의 칸(열 0·A–E × 방어 none·dg) 하나 = 실행 하나, NFE(생성/캐시/수리/기준선)·시간·감사 기록 | 완료 (CPU) |

## 파일

```
RQ2/
├── run_rq2_gsm8k.py     # GSM8K 러너 (프롬프트·채점·문항·시드는 RQ1 러너 DiffuGuard/run_gsm8k.py 재사용)
├── src/combined.py      # 결합 디코더 (원본과 다른 점은 파일 맨 위 docstring)
└── tests/
    ├── conftest.py      # DiffuGuard/tests의 가짜 모델·토크나이저 재사용
    ├── test_s1_fixed.py     # S1: DiffuGuard 원본(semi_ar)과 동등성
    ├── test_s2_threshold.py # S2: Fast-dLLM 임계값 디코딩과 동등성
    ├── test_s3_threshold_defense.py # S3: 결합 규칙 (별도 참조 구현 + 규칙별 성질 검사)
    ├── test_s4_cache.py     # S4: 캐시 (Fast-dLLM 캐시 함수와 동등성, 정확한 캐시로 배관 검사, 수리 후 캐시 재구성)
    └── test_s5_runner.py    # S5: 칸→설정 매핑, 러너 = 디코더 직접 호출, 재개, 콘솔에 응답 미출력
```

## GSM8K 실행 (GPU)

```bash
cd RQ2
export FAST_DLLM_ROOT=/root/Fast-dLLM          # a9b81e4 체크아웃 (모델 클래스 LLaDAModelLM)
python run_rq2_gsm8k.py --column A --defense dg --limit 10      # 파일럿: 시간 측정
python run_rq2_gsm8k.py --column A --defense dg                 # 250문항 (RQ1/data/ids/gsm8k_250.txt)
python run_rq2_gsm8k.py --column E --defense none --dry_run     # 설정만 출력
```

| 열 | 설정 | | 행 | 설정 |
|---|---|---|---|---|
| 0 | 고정 스텝 256 (가속 없음) | | none | 방어 없음 (p 순서) |
| A / B / C | τ 0.9 / 0.7 / 0.5, 캐시 없음 | | dg | DiffuGuard: 진행률 α (α0 0.3), 감사 λ 0.1, 수리 8회 |
| D / E | τ 0.9 + Prefix / Dual 캐시 | | | |

공통: gen 256, block 32, temperature 0. run_id = `rq2-gsm8k-{열}-{none|dg}-s0`.
결과는 `RQ2/outputs/gsm8k/`(git 제외): `<run_id>.json`(답), `<run_id>.meta.json`(정확도·95% CI·무응답률·수리율·NFE·tok/s, 문항별 기록).
중간에 끊기면 같은 명령으로 다시 실행하면 이어서 합니다.

GPU에서 먼저 확인할 것 (실험 전 한 번):
1. 실제 모델에서 S2·S4 동등성: 열 A/D/E `--defense none` 10문항의 토큰이 Fast-dLLM `generate`·`generate_with_prefix_cache`·`generate_with_dual_cache`(steps=256)와 같은지
2. `LLaDAModelLM`에서 `output_hidden_states`·기준선 hidden이 AutoModel(RQ1)과 같은 값인지 (감사 점수 비교)
3. `--limit 10` 파일럿으로 칸당 시간 추정

## 테스트 (CPU)

```bash
cd RQ2 && python -m pytest tests -q
# 실제 Fast-dLLM 코드(a9b81e4)와도 비교하려면 체크아웃 경로를 지정
FAST_DLLM_ROOT=/root/Fast-dLLM python -m pytest tests -q
```

캐시 모드에서 정한 것:
- 수리는 캐시를 쓰지 않고 전체 forward 8회로 돈다(모든 열에서 수리 비용이 같음). 다음 블록 시작의 전체 forward가 수리된 토큰으로 캐시를 새로 만든다.
- Dual은 블록이 다 찰 때까지 돈다. Fast-dLLM 원본은 `steps = gen_length`일 때만 블록을 다 채우므로, 그 설정에서 둘이 같다.
- 방어의 난수 R은 항상 전체 길이로 뽑는다. 그래서 캐시를 켜도 위치별 R이 같다.
- attention_mask는 전부 1일 때만 받고 버린다(Fast-dLLM 캐시 함수는 마스크를 넘기지 않음).

현재 범위는 GSM8K형 입력(프롬프트 안에 마스크 없음)입니다. DIJA처럼 프롬프트에 마스크가 있는 입력은 ASR 블록 구성이 팀에서 정해질 때까지 `NotImplementedError`를 냅니다.

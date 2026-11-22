# RQ3: 거절 확정 시점 측정 (`commit_time`)

디노이징 forward마다 은닉 상태를 저장하고, 응답 결과(거절 / 해로운 순응)를 맞히는 선형 프로브(D²-Monitor 방식, 라벨만 응답 결과로 바꿈)로
"결정이 언제 굳는가"(확정 시점 s\*)와 "얼마나 망설이는가"(n_τ)를 잰다. 가속은 필드 표준 두 계열을 쓴다.

- **Fast-dLLM 임계값 병렬 디코딩:** 초반을 압축한다. `threshold` 규칙, τ ∈ {0.9, 0.7, 0.5}.
- **Prophet 조기 종료:** 후반을 잘라낸다. `prophet` 규칙, 임계값 7.5 / 5.0 / 2.5, 답 영역은 응답 앞 16개.

고정 스텝 축소는 쓰지 않는다. 실험 절차(E0–E5)와 성공 기준은 명세서 문서를 따른다.

## 설치와 테스트

```bash
cd DLM_safety_code
git submodule update --init DiffuGuard/third_party/DiffuGuard RQ3/third_party/RAEC
cd RQ3
pip install -r requirements.txt          # DiffuGuard 환경(torch 2.6.0, transformers 4.52.3) 위에
python -m pytest -q tests                # CPU, 가짜 LLaDA, 27개
```

핵심 테스트는 아래 네 가지다.

- `test_decode_capture.py::test_quota_equals_original_vanilla`: quota 규칙이 DiffuGuard 원본 `generate()`의 vanilla 경로와 같은 시드에서 같은 토큰을 내는지 본다.
- `test_prophet.py::test_infinite_threshold_equals_quota`: 임계값이 무한대인 Prophet이 quota와 같은 출력을 내는지 본다.
- `test_prophet.py::test_exit_matches_quota_before_exit_and_records_fill`: 세 가지를 확인한다.
  - 종료 전까지 quota와 같은 토큰을 낸다.
  - all-in 채움이 마지막 `n_commit`에 기록된다.
  - R trace의 격차로 시뮬레이션한 종료 스텝이 실제 Prophet 종료 스텝과 같다.
- `test_prophet.py::test_zero_threshold_exits_at_first_forward`: 임계값 0이면 첫 forward에서 종료하고 마스크가 남지 않는지 본다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `commit_time/capture.py` | 모델 hook으로 forward마다 아래 값을 기록한다. 디코더는 고치지 않는다. KV 캐시처럼 부분 입력이 들어오는 forward는 거부한다.<br>• 은닉 상태(추적 위치 평균)<br>• 커밋·재마스킹 수<br>• logit 통계: 응답 앞 16개 위치의 Prophet 격차(전체·커밋된 위치·마스크 위치), RAEC식 거절·순응 질량 |
| `commit_time/decode.py` | KV 캐시 없는 단일 블록 디코더. 중간 상태에서 재개할 수 있다.<br>• `quota`: DiffuGuard vanilla와 같다.<br>• `threshold`: Fast-dLLM 규칙.<br>• `prophet`: quota에 Prophet 조기 종료를 더한 것. |
| `commit_time/token_sets.py` | RAEC 거절·순응 토큰 집합. RAEC `a59bae5`의 `build_cue_id_sets` 기본값을 그대로 옮겼다. |
| `commit_time/run_trace.py` | `DiffuGuard/run_generate.py`를 그대로 실행하면서 trace를 남긴다. `--decoder original\|quota\|threshold\|prophet` |
| `commit_time/resample.py` | 중간 상태에서 시드만 바꿔 이어 생성한다(행동 확정 시점 검증). prophet도 지원한다. |
| `commit_time/probe.py` | 그룹 CV(문항 단위)로 층과 C를 고르고 τ를 정한다. |
| `commit_time/metrics.py` | s\*, n_τ, s_text, 결정 전 커밋 비율, 수리 전후 마진.<br>Prophet 열: `exit_step`, `exit_frac_tokens`, `exit_after_s_star`, `sim_exit_step` |
| `commit_time/analyze.py` | CLI: `labels` · `fit` · `score` · `behavior` · `flips` · `gaps`.<br>• `flips`에 R의 s\* 대비 종료 순서(H1)를 추가했다.<br>• `gaps`는 Prophet 복사 경향 분석이다. |
| `scripts/run_traces.sh` | 6.3 설정표의 trace 실행: R, D, A0.9, A0.7, A0.5, P, P_argmax |
| `scripts/e0_5.sh`, `e0_5/` | E0.5: 우리 84% vs RAEC 8%. 생성 조건 3개 × judge 2개 |
| `third_party/RAEC` | RAEC 공개 코드(submodule, `a59bae5`, 수정 금지). E0.5의 C3에서만 쓴다. |

## 실험 설정 (인수인계 6.3)

| 설정 | 방어 | 디코더 | 손잡이 | 역할 |
|---|---|---|---|---|
| R | vanilla | original | 128스텝 | 기준, 프로브 학습 |
| D | diffuguard | original | 128스텝 + 수리 8 | 프로브 학습, H3 |
| A0.9 / A0.7 / A0.5 | vanilla | threshold | τ 0.9 / 0.7 / 0.5 | H1·H2 (초반 압축) |
| P | vanilla | prophet | 7.5 / 5.0 / 2.5, 답 영역 = 응답 앞 16 | H1 (후반 절삭), H2 |
| P_argmax | vanilla | prophet, 노이즈 없는 채움 | 같음 | 민감도 (열린 질문 4) |
| D_τ, D_P | diffuguard + 가속 | RQ2 결합 디코더 | — | H3 (RQ2 코드가 나온 뒤) |

공통 설정은 RQ1과 같다.
- 모델: LLaDA-8B-Instruct `08b83a6`
- 프롬프트: DIJA×JBB 공개 100문항 (`534d0df5…`)
- 디코딩: 시드 0 + 문항 번호, temperature 0.5, gen = block = 128, cfg 0, fill_all_masks
- trace는 KV 캐시 없이 돌린다.

### Prophet 구현 세부 (공식 `generate_earlyexit.py`, commit `460afe4` 기준)
- **판정 시점:** 매 스텝 forward 직후, 커밋하기 전에 판정한다.
- **격차:** 노이즈 없는 원래 logit의 top-2 격차를 답 영역 16개 위치에서 평균한다. 이미 커밋된 위치도 포함한다(공식 구현 그대로).
- **진행률:** p = (i+1)/steps. 공식 코드는 `global_step`을 먼저 1 올린 뒤 계산하므로, 인수인계 문서의 step/T와 한 칸 다르다. 공식 쪽을 따랐다.
- **종료:** ḡ ≥ τ(p)이면 남은 마스크 전부(DIJA 프롬프트 마스크 포함)를 그 스텝의 `x0`로 채우고 끝낸다.
- **temperature 0.5에서의 채움:** `x0`는 Gumbel 노이즈가 섞인 argmax라서 all-in 채움도 샘플링이다(공식 동작). `--prophet_fill argmax`는 노이즈 없는 변형이다.
- **종료 전 토큰:** quota와 같은 RNG 호출 순서를 지키므로, 종료 전까지는 quota·original과 같은 토큰이 나온다.
  - 그래서 R trace에서 `gap_all ≥ τ(p)`가 처음 되는 forward가 같은 시드 P 실행의 종료 스텝이다(`sim_exit_step`).
  - H1 비교(종료 스텝 > s\*(R))는 R의 시계로 한다(`analyze flips`).

## 실행

```bash
cd /root/DLM_safety_code/RQ3 && source /root/venv/bin/activate
export RQ3_OUT=/root/rq3-outputs            # 팀 공용 볼륨 (git 밖)
mkdir -p $RQ3_OUT/logs

# E0.5 (먼저): 84% vs 8% 원인
bash scripts/e0_5.sh gen && bash scripts/e0_5.sh judge && bash scripts/e0_5.sh summary   # -> results/e0_5/

# E1 / E4: trace (설정 하나씩, 백그라운드). LIMIT=5로 먼저 연기 테스트
LIMIT=5 bash scripts/run_traces.sh P
nohup bash scripts/run_traces.sh R > $RQ3_OUT/logs/R.log 2>&1 &

# 판정 -> 라벨 (텍스트는 복사하지 않음)
python ../DiffuGuard/eval_asr.py $RQ3_OUT/gen/<생성 파일>.json
python -m commit_time.analyze labels $RQ3_OUT/gen/eval_<생성 파일>.json --out $RQ3_OUT/labels/R.csv

# Prophet 복사 경향 (열린 질문 1): R만 있으면 된다
python -m commit_time.analyze gaps --run $RQ3_OUT/trace/R --out $RQ3_OUT/metrics/gaps_R.csv

# 프로브 학습 -> 점수 -> 뒤집힘
python -m commit_time.analyze fit --run $RQ3_OUT/trace/R $RQ3_OUT/labels/R.csv \
  --run $RQ3_OUT/trace/D $RQ3_OUT/labels/D.csv --out $RQ3_OUT/probe/probe.npz
python -m commit_time.analyze score --probe $RQ3_OUT/probe/probe.npz \
  --run $RQ3_OUT/trace/P $RQ3_OUT/labels/P.csv --out $RQ3_OUT/metrics/P.csv
python -m commit_time.analyze flips --ref $RQ3_OUT/metrics/R.csv --acc $RQ3_OUT/metrics/P.csv \
  --out $RQ3_OUT/metrics/flips_P.csv
```

### E0.5 조건

같은 공개 프롬프트 파일을 쓴다. RAEC 저장소의 `DIJA_JBB_refined_Qwen.json`도 SHA-256이 같다.

| 조건 | 코드 | 스텝 | cfg | 시드 | 무엇을 가르나 |
|---|---|---:|---:|---|---|
| C1 `ours_R` | 팀 러너 (RQ1 vanilla 출력 재사용) | 128 | 0 | 0 + 문항 번호 | 기준 |
| C2 `ours_s64cfg` | 팀 러너 | 64 | 0.1 | 0 + 문항 번호 | C1 대비 디코딩 손잡이 효과 |
| C3 `raec_code` | RAEC `a59bae5`, `--raec_enable` 없음 (RAEC README의 vanilla baseline) | 64 | 0.1 | 42, 한 번만 | C2 대비 코드 차이(시드, 프롬프트 처리) |

- **판정:** 각 조건을 팀 judge(`eval_asr.py`)와 Llama-Guard-3-8B로 둘 다 판정한다.
- **Llama-Guard 실행:** `e0_5/llamaguard_judge.py`를 쓴다. 프롬프트 선택과 라벨 파싱은 RAEC와 같고, vLLM 대신 transformers를 쓰며, 원문을 출력하지 않는다. RAEC의 판정 스크립트는 앞 3개 문항 원문을 출력하므로 쓰지 않는다.
- **접근 권한:** Llama-Guard-3-8B는 gated 모델이라 HF 계정에서 라이선스를 먼저 승인받아야 한다.
- **fill_all_masks는 원인이 아닐 가능성이 크다.** 코드로 확인한 바로는 RAEC도 프롬프트에 마스크가 있으면 첫 마스크부터 끝까지를 블록 하나로 디코딩한다. 이는 fill_all_masks와 같은 스케줄이다. 남은 후보는 cfg 0.1, 64스텝, judge, 시드다.

## 보안

- **텍스트 없는 파일:** `hidden/*.npz`, `trace_meta.jsonl`, `labels.csv`, `metrics*.csv`, `gaps*.csv`, `guard_*.csv`, `results/`에는 벡터, 개수, 라벨, 해시 앞 12자만 들어 있다. logit 통계도 위치 평균 float만 저장하고, 토큰 id나 토큰별 확률은 저장하지 않는다.
- **모델 텍스트:** 생성 JSON에만 있다. 이 파일은 `run_generate.py`, `resample.py`, RAEC 생성기가 쓴다. `$RQ3_OUT`은 git 밖에 둔다.
- **중간 상태 x_s(부분 텍스트):** 메모리에서만 쓰고 저장하지 않는다.
- **개입:** 순응 prefill 같은 공격성 개입은 없다. RAEC 개입(`--raec_enable`)도 E0.5에서는 쓰지 않는다.
- **KV 캐시 디코더:** 지원하지 않는다. `TraceRecorder`가 부분 입력 forward를 거부한다.

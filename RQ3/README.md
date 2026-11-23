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

### 두 응답 구간: 인필과 assistant 답변 (E0.5 결과로 추가, 2026-11-23)
DIJA 프롬프트는 사용자 턴 안에 빈칸(마스크)이 있다. 그래서 생성 결과는 두 구간으로 나뉜다.

| 구간 | 무엇인가 | 판정 정의 | E0.5 ASR-e |
|---|---|---|---:|
| 인필 (주 분석) | 모델이 사용자 턴의 빈칸을 채운 부분 | DIJA·DiffuGuard·RQ1과 같다. 생성 JSON의 `response` | 84% |
| assistant 답변 | 프롬프트 뒤에 붙은 128칸 답변 | RAEC와 같다. `<run>.tail.json`의 `response` | 6% |

- **기록:** trace는 두 구간을 모두 기록한다.
  - 응답 앞 16칸은 기존 그대로다: `lead_unmasked`, `gap_*`, `mass_*`.
  - 인필 앞 16칸(첫 프롬프트 마스크 16개)은 `infill_unmasked`와 `*_infill`로 기록한다.
  - metrics에는 `s_text_infill`, `latent_lead_infill`, `*_infill` 열이 추가된다.
- **라벨:** `run_traces.sh judge <설정>`이 두 구간을 모두 판정한다.
  - 인필 기준은 `labels.csv`, 답변 기준은 `labels_tail.csv`로 trace 폴더에 저장한다.
  - 프로브와 s\*의 주 분석은 인필 라벨로 한다. 답변 라벨은 보조로 쓴다(해로운 사례가 적다).
- **Prophet:** 답 영역은 공식대로 assistant 답변 앞 16칸이다. 인필 쪽 격차는 비교용으로만 기록한다.
- **은닉 상태:** 세 가지를 저장한다(2026-11-23부터).
  - `hidden`: 추적 위치 전체(인필 + 답변) 평균
  - `hidden_infill`: 인필 위치만의 평균
  - `hidden_tail`: 답변 위치만의 평균
  - 프로브는 `analyze fit --region all|infill|tail`로 영역을 고른다. `score`는 fit 때 고른 영역을 그대로 쓴다.
- **시드:** `SEED=1000 bash scripts/run_traces.sh R`처럼 돌리면 이름에 `_s1000`이 붙는다.
  - 문항마다 `seed + 문항 번호`로 시드를 정하므로, 시드끼리는 100 이상 떨어뜨린다.
  - 전체 실행: `SETTINGS="R D" SEEDS="0 1000 2000" SEEDS_D="0 1000" bash scripts/run_all_traces.sh`

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

# 판정 -> 라벨 (텍스트는 복사하지 않음). 인필 -> trace/R/labels.csv, 답변 -> trace/R/labels_tail.csv
bash scripts/run_traces.sh judge R

# Prophet 복사 경향 (열린 질문 1): R만 있으면 된다
python -m commit_time.analyze gaps --run $RQ3_OUT/trace/R --out $RQ3_OUT/metrics/gaps_R.csv

# 프로브 학습 -> 점수 -> 뒤집힘
python -m commit_time.analyze fit --run $RQ3_OUT/trace/R $RQ3_OUT/trace/R/labels.csv \
  --run $RQ3_OUT/trace/D $RQ3_OUT/trace/D/labels.csv --out $RQ3_OUT/probe/probe.npz
python -m commit_time.analyze score --probe $RQ3_OUT/probe/probe.npz \
  --run $RQ3_OUT/trace/P $RQ3_OUT/trace/P/labels.csv --out $RQ3_OUT/metrics/P.csv
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

- **판정:** 팀 judge(`eval_asr.py`)를 먼저 쓴다. C1은 RQ1 판정 결과를 그대로 쓴다.
  - C3이 8% 근처면 생성 쪽 차이다.
  - C3이 84% 근처면 judge 차이가 의심된다. 이때만 Llama-Guard-3-8B를 추가한다(`GUARD=meta-llama/Llama-Guard-3-8B bash scripts/e0_5.sh judge`).
  - 기본값은 `GUARD=none`이다. Llama-Guard를 돌리지 않은 조건은 요약표에서 `—`로 표시한다.
- **Llama-Guard 실행:** `e0_5/llamaguard_judge.py`를 쓴다. 프롬프트 선택과 라벨 파싱은 RAEC와 같고, vLLM 대신 transformers를 쓰며, 원문을 출력하지 않는다. RAEC의 판정 스크립트는 앞 3개 문항 원문을 출력하므로 쓰지 않는다.
- **접근 권한:** Llama-Guard-3-8B는 gated 모델이라 HF 계정에서 라이선스를 먼저 승인받아야 한다.
- **결과 (2026-11-22, 팀 judge):** C1 84.0, C2 83.0, C3 6.0. 원인은 judge도 디코딩 설정도 아니다. **judge에 넘기는 응답 구간이 다르다.**
  - DIJA 원본, DiffuGuard, 우리 러너는 사용자 턴의 인필(채운 프롬프트 마스크)을 판정한다.
  - RAEC는 프롬프트 뒤 assistant 답변만 판정한다.
  - 자세한 내용은 [`results/e0_5/diagnostics.md`](results/e0_5/diagnostics.md).

## 보안

- **텍스트 없는 파일:** `hidden/*.npz`, `trace_meta.jsonl`, `labels.csv`, `metrics*.csv`, `gaps*.csv`, `guard_*.csv`, `results/`에는 벡터, 개수, 라벨, 해시 앞 12자만 들어 있다. logit 통계도 위치 평균 float만 저장하고, 토큰 id나 토큰별 확률은 저장하지 않는다.
- **모델 텍스트:** 생성 JSON에만 있다. 이 파일은 `run_generate.py`, `resample.py`, RAEC 생성기가 쓴다. `$RQ3_OUT`은 git 밖에 둔다.
- **중간 상태 x_s(부분 텍스트):** 메모리에서만 쓰고 저장하지 않는다.
- **개입:** 순응 prefill 같은 공격성 개입은 없다. RAEC 개입(`--raec_enable`)도 E0.5에서는 쓰지 않는다.
- **KV 캐시 디코더:** 지원하지 않는다. `TraceRecorder`가 부분 입력 forward를 거부한다.

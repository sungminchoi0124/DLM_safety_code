# SafeGate: 안전 게이트 commit 규칙 (최소 시제품)

> **상태: 종료(2026-11-24), 음성 결과.** 개발셋에서 멈췄고 테스트셋(JBB 100, GSM8K 250)은 돌리지 않았다.
> - 1단계([results/dev/stage1_summary.md](results/dev/stage1_summary.md)): 거부·순응 경합(b > 0)은 유해 개발셋에서 거의 생기지 않았다(최대 4/30). 미리 정한 규칙은 거부 질량만 보는 a = 0.005, b = 0을 골랐다.
> - 1b단계([results/dev/stage1b_summary.md](results/dev/stage1b_summary.md)): 게이트가 23/30에서 발동했지만 효과가 없었다. 인필 ASR-e 27/29 → 27/30, 답변 ASR-e 3/30 → 3/30. forward는 +4.5%.
> - 이유: DIJA의 해로움은 거부 표현이 나오지 않는 인필(사용자 턴)에 있고, 답변은 이미 대부분 거절이다. 응답 앞쪽 질량을 보는 게이트는 DIJA에 구조적으로 맞지 않는다.
> - 시험하지 않은 것: 답변이 판정 대상인 공격(PAD, JBB 원문, WildJailbreak)에서의 효과. 인필 위치의 hidden 위험 점수(신호 2).
>
> 아래는 원래 설계 기록이다.

목표: DiffuGuard 수준으로 ASR을 낮추면서, GSM8K 정확도와 속도는 Fast-dLLM(RQ2 A열)에 가깝게 유지한다.
RQ2에서 DiffuGuard는 A열 GSM8K를 78.8% → 64.8%로 떨어뜨렸다.

## 비교할 행 (모두 같은 디코더 `safegate/decoder.py`)

| 행 | 설정 |
|---|---|
| `none` | RQ2 A열: Fast-dLLM threshold 0.9, 캐시 없음, 블록 32 |
| `dg` | A + DiffuGuard (진행률 α, 첫 블록 감사 λ 0.1, 수리 8회) = RQ2 A열 dg |
| `gate` | A + 안전 게이트 (`configs/gate.yaml`) |

`gate` 없이 돌리면 이 디코더는 RQ2 결합 디코더(`RQ2/src/combined.py`, A열)와 토큰 단위로 같다(방어 켬/끔, DIJA 프롬프트 마스크, temperature 0/0.5 모두). GSM8K `none`/`dg` 행은 RQ2 러너와 답·forward 수가 같다. CPU 테스트로 확인했다.

공통 조건: LLaDA-8B-Instruct `08b83a6f`, 시드 0(문항마다 seed + 번호), Fast-dLLM `LLaDAModelLM`(a9b81e4).
ASR = DIJA × JBB 100, gen 128, temperature 0.5, 첫 블록 = 프롬프트 마스크 + 생성 앞 32자리(RQ2 ASR 방식). 응답 = 채워진 사용자 턴(DIJA·RQ1·RQ2 기준), assistant 답변은 `tail_response`에 따로 저장.
GSM8K = `RQ1/data/ids/gsm8k_250.txt`, gen 256, temperature 0.
판정 = gpt-6-luna(effort none, `DiffuGuard/eval_asr.py`) ASR-e, DIJA 키워드 ASR-k.

## 게이트 규칙

첫 블록의 매 forward에서, Fast-dLLM 규칙이 이번 스텝에 확정할 자리를 고른 뒤:

1. 창 = 응답 앞 K자리(기본 16) 중 아직 마스크인 자리 (설정으로 DIJA 프롬프트 마스크 앞 `K_infill`자리 추가 가능)
2. m_ref = 창 자리별 거부 토큰 확률 합의 평균, m_cmp = 순응 토큰 확률 합의 평균 (디코더가 이미 계산한 float64 softmax, 추가 forward 없음)
3. m_ref ≥ a 이고 m_cmp ≥ b 이면 "경합": 이번 스텝에 창 안에서 고른 자리 중 후보 토큰이 거부 토큰이 아닌 것은 확정하지 않고 마스크로 둔다. 나머지 자리는 그대로 확정한다.
4. 보류한 스텝이 H(기본 8)에 이르면 그 문항은 이후 원래 Fast-dLLM 규칙대로 간다.

토큰 집합(`short`): 거부 Sorry, I'm, cannot, can't, unable / 순응 Sure, Here, Certainly, Step. 앞 공백 있는/없는 변형 중 **한 토큰인 것만** 쓴다. LLaDA에서 "I'm"(= I + 'm), "can't"(= can + 't)는 두 토큰이라 빠진다. 조각("I", "can")은 흔한 단어라 넣으면 무해한 글에서도 질량이 커진다. 비교용으로 RAEC 집합(RQ3와 같음)도 trace에 기록한다.

게이트는 난수를 쓰지 않는다. 그래서 첫 발동 전까지는 게이트 없는 실행과 같은 궤적을 따른다. **게이트 없는 trace만으로 "발동 여부"와 "첫 발동 스텝"을 정확히 알 수 있다**(테스트로 확인). a, b는 이 성질을 이용해 개발셋 trace에서 고른다.

## 단계

### 1단계: 개발셋 trace (게이트 없음)
개발셋은 테스트셋과 겹치지 않는다(`prepare_dev.py`).
- `data/dev_harm30.txt`: DIJA HarmBench refined 파일의 standard 행동 중 JBB 목표와 비슷한 것(difflib ≥ 0.6, 34개)을 뺀 166개에서 30개
- `data/dev_gsm30.txt`: GSM8K 테스트셋 중 250문항 목록에 없는 1,069개에서 30개

```bash
python SafeGate/run_gate_asr.py   --set dev_harm30 --row none --trace --model_path $M
python SafeGate/run_gate_gsm8k.py --set dev        --row none --trace --model_path $M
JUDGE_MODEL=gpt-6-luna python DiffuGuard/eval_asr.py SafeGate/outputs/asr/sg-asr-dev_harm30-A-none-s0/sg-asr-dev_harm30-A-none-s0.json
python SafeGate/analyze_dev.py \
  --harm   SafeGate/outputs/asr/sg-asr-dev_harm30-A-none-s0/traces \
  --benign SafeGate/outputs/gsm8k/traces/sg-gsm8k-dev-A-none-s0 \
  --harm_eval SafeGate/outputs/asr/sg-asr-dev_harm30-A-none-s0/eval_sg-asr-dev_harm30-A-none-s0.json
```
trace 형식은 `safegate/trace.py` 맨 위 설명 참고. 토큰 번호가 들어 있으므로 `outputs/`(git 제외) 밖으로 꺼내지 않는다. `analyze_dev.py` 결과(`results/dev/`)는 숫자만 담는다.

### a, b 선택 규칙 (데이터를 보기 전에 고정)
- 후보: `analyze_dev.py`의 격자(a, b ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}), 창은 `response` K=16, 토큰 집합 `short`
- 조건: 무해 개발셋(GSM8K 30) 발동률 ≤ 1/30
- 그중 유해 개발셋 발동률이 가장 높은 (a, b). 같으면 a, b가 큰 쪽(보수적)
- 중단 조건: 이 조건에서 유해 개발셋 발동률이 30%에 못 미치면 신호 1(확률 경합)만으로는 게이트가 성립하지 않는 것으로 보고, 테스트셋을 돌리지 않고 신호 2(hidden 위험 점수)로 넘어간다
- H = 8, K = 16은 고정(개발셋에서 바꾸지 않음)

### 2단계: 테스트셋 3행
```bash
for row in none dg gate; do python SafeGate/run_gate_asr.py --set jbb100 --row $row --model_path $M; done
for row in none dg gate; do python SafeGate/run_gate_gsm8k.py --set test --row $row --model_path $M; done
```
속도 측정 실행에는 `--trace`를 붙이지 않는다(기록이 GPU 동기화를 추가함).

## 측정 (meta.json, 문항별 숫자 포함)
ASR-e / ASR-k(판정 후), GSM8K 정확도, 문항당 실제 소요 시간(`wall_ms`, DiffuGuard 기준선 forward 포함), forward 수(생성·수리·기준선), forward당 확정 토큰 수(`tokens_per_forward`), 게이트 발동률(`gate_fire_rate`, GSM8K vs DIJA)·보류 스텝 수·보류 토큰 수.

## 파일
```
SafeGate/
├── configs/gate.yaml        # a, b(개발셋에서 정하기 전에는 null), K, H, 창, 토큰 집합
├── data/                    # 개발셋 id 목록 (공개 데이터 번호만)
├── prepare_dev.py           # 개발셋 목록 생성
├── run_gate_asr.py          # DIJA ASR 생성 (none / dg / gate)
├── run_gate_gsm8k.py        # GSM8K 정확도 (none / dg / gate)
├── analyze_dev.py           # 개발셋 trace → (a, b)별 발동률
├── safegate/decoder.py      # A열 디코더 + 게이트 + trace 호출
├── safegate/trace.py        # 스텝별 기록 (공용 모듈)
├── safegate/tokens.py       # 거부/순응 토큰 집합
└── tests/                   # CPU 테스트 (python -m pytest SafeGate/tests)
```

# ProbeGate: 프로브로 DiffuGuard를 켜고 끄기

## 왜 하나

AuditCheck(1번 실험) 결과:
- DiffuGuard 감사는 λ 0.1에서 모든 입력(1,970/1,970)에 발동하고, 같은 형식의 유해/무해를 가르지 못한다(JBB AUROC 0.50).
- 그래서 DiffuGuard는 사실상 항상 수리하고, GSM8K 손실(RQ2 A열 78.8 → 64.8)은 고정 비용이다.
- 같은 첫 forward의 hidden으로 학습한 선형 프로브는 일반 요청을 가른다(JBB 0.93, XSTest 0.92). 다만 일반 요청으로만 학습하면 DIJA 형식에는 옮겨지지 않는다(0.53~0.58).

이 실험은 **첫 forward에서 프로브가 위험하다고 본 입력에만 DiffuGuard를 켜는** 방식을 시험한다. 이기는 조건 A("DiffuGuard와 안전은 비슷, GSM8K는 덜 깎음")를 겨냥한다.

## 방법

| 행 | 설정 |
|---|---|
| none | RQ2 A열: Fast-dLLM threshold 0.9, 캐시 없음, 블록 32, 방어 없음 |
| dg | A + DiffuGuard (진행률 α, 블록-0 감사 λ 0.1, 수리 8회) |
| pg | 첫 forward에서 프로브 점수 ≥ t이면 dg, 아니면 none |

- 프로브: 첫 스텝(프롬프트 + 전부 마스크) `hidden_states[23]`의 프롬프트 토큰 평균 → 표준화 → L2 로지스틱 회귀. forward 1회가 추가된다.
- pg의 출력은 문항마다 none 또는 dg 출력과 토큰까지 같다. 같은 시드(seed + 문항 번호)이고, 프로브 forward는 난수를 쓰지 않기 때문이다. 그래서 pg의 정확도와 ASR은 none·dg 실행 결과와 문항별 게이트 결정으로 계산한다. 일부 문항은 실제 pg 실행으로 토큰 일치를 확인한다.

## 학습 데이터 (테스트셋 제외)

`prepare_train.py`. 테스트셋(DIJA×JBB 100, GSM8K 팀 250)과 겹치는 것은 넣지 않는다.

| 세트 | 라벨 | 형식 | 내용 |
|---|---|---|---|
| pg_adv | 유해 | 일반 | AdvBench − JBB와 비슷한 행동(difflib ≥ 0.6) |
| pg_hb_dija | 유해 | DIJA(실제) | HarmBench standard DIJA 프롬프트 − JBB와 비슷한 것 |
| pg_tmpl_harm | 유해 | DIJA(템플릿) | pg_adv 요청을 DIJA식 틀("Step 1: <mask:20>.")에 넣은 것 |
| pg_jbbb | 무해 | 일반 | JBB-Behaviors benign 100 |
| pg_gsm_chat | 무해 | 일반 | GSM8K 팀 250 밖의 300문항, zero-shot |
| pg_gsm_5shot | 무해 | GSM8K 5-shot | 같은 300문항, GSM8K 평가 형식 |
| pg_tmpl_benign | 무해 | DIJA(템플릿) | pg_jbbb·pg_gsm_chat 요청을 같은 틀에 넣은 것 |

- 템플릿은 유해·무해에 똑같이 적용한다. 그래서 "프롬프트에 마스크가 있다"는 것만으로는 두 클래스를 가를 수 없다.
- XSTest는 학습에서 완전히 뺀다. 과잉 게이트(safe)와 재현율(unsafe)을 처음 보는 데이터로 확인하는 용도다.

## 미리 정한 규칙 (테스트 결과를 보기 전에 고정)

- 특징: `L23_prompt`. `L23_slots`는 민감도 확인용으로만 보고한다.
- 분할: 기본 요청(`group`) 단위로 30%를 보정용(random.Random(0)), 나머지를 학습용으로 쓴다.
- 임계값 t: 무해 보정 점수의 98번째 백분위수(목표 오게이트율 2%).
- 2b(생성) 진행 조건:
  - 보정 세트의 DIJA 형식 유해(pg_hb_dija) 게이트율 ≥ 0.8
  - 테스트 입력 GSM8K 5-shot 게이트율 ≤ 0.1
  - 하나라도 못 넘으면 2b는 하지 않고 결과만 기록한다.
- 테스트 입력의 게이트율은 측정만 하고 t를 고르는 데 쓰지 않는다.

## 평가 (2b)

- ASR: DIJA×JBB 100, gen 128, block 32, temperature 0.5. 판정은 gpt-6-luna(effort none, DiffuGuard 원본 프롬프트)의 인필 기준 ASR-e와 ASR-k.
- GSM8K: 팀 250, gen 256, block 32, temperature 0, flexible-extract.
- 속도: 문항당 forward(프로브 forward 포함), 시간.
- 행: none / dg / pg. 짝지은 비교는 McNemar로 한다.

## 실행

```bash
MODEL_PATH=/root/DLM_safety_code/hf_models/LLaDA-8B-Instruct FAST_DLLM_ROOT=/tmp/Fast-dLLM \
    bash ProbeGate/scripts/stage2a.sh     # 약 10분, 생성 없음
```

# RQ1 ASR-e와 ASR-k의 문항별 판정 불일치

## 1. 분석 목적

기존 RQ1 DIJA-JBB 결과에서 ASR-e와 ASR-k의 판정 불일치가 Step 및 DiffuGuard 적용 여부에 따라
어떻게 달라지는지 분석한다. 새로운 Safety Judge나 방어 방법을 만들지 않으며 두 판정 중 어느 것이
정답인지를 확정하지 않는다. 핵심 수치는 자동 생성된 [summary.md](summary.md)에 있다.

## 2. 사용한 데이터와 평가 기준

기준 main은 `91c8406eb8cb591721c199a1b64c08c0eb74df33`이다. 입력은 아래 두 폴더의
`per_item.csv`, `runs.json`, `summary.csv`, `summary.md`다.

- `RQ1/results/pilot_vanilla_dija_steps`: Vanilla 128/64/32/16 Step, 각 100문항.
- `RQ1/results/with_diffuguard_dija_steps`: DiffuGuard 7조건, 각 100문항.

주 분석의 DiffuGuard는 `(Step, Repair)=(128,16),(64,8),(32,4),(16,2)`다.
추가 고정 Repair 8회 `(128,8),(32,8),(16,8)`는 보조 분석이다. `(64,8)`은 두 정책이 동일한
단일 실행이므로 주 분석에만 한 번 포함한다. 총 11조건·1100행이며 독립 문항은 100개다.
Vanilla의 설정상 `refinement_steps=8`도 기록하되 감사 OFF이므로 실제 수리 조건으로 해석하지 않는다.

### 검증한 메타데이터

- 모델 revision: `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- DiffuGuard 원본 commit: `86aeb75203d07064c1ced650edaee74563647d36`.
- 공격: `dija_jbb`, 프롬프트 파일 SHA-256:
  `534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95`.
- seed 0, 생성 128, 블록 128, temperature 0.5, cfg 0, mask_counts 36,
  fill_all_masks=true, correct_only_first_block=true.
- Vanilla: low_confidence, 감사 OFF. DiffuGuard: adaptive_step, hidden 감사,
  λ=0.1, α0=0.3, remask_ratio=0.9 및 조건별 비례/고정 수리.
- Judge: `gpt-6-luna`, `judge_prompt=code`, `question=auto`,
  reasoning_effort=none, max_completion_tokens=800.
- 기록된 생성 commit: Vanilla `f06ae214db8238f2bd680b4b774d6eae49b4e6e3`,
  DiffuGuard `716e7d0b7dc59fc87c24948e777551e7f03bbf91`. 둘 다 `team_repo_dirty=true`다.

11조건 모두 위 공통 값과 조건별 설정을 검사한다. 조건별 seed·문항 인덱스·프롬프트 해시는 일치한다.
각 파일의 실제 열 이름은 `provenance.json`에 기록한다. steps/refinement_steps/item_index는 정수,
라벨은 엄격한 0/1 및 두 Category, 응답 길이는 음이 아닌 정수, 응답 해시는 12자리 hex로 검사한다.
DiffuGuard의 repaired는 True/False, audit/logit_score는 유한한 실수여야 한다.
방어별 `(steps, refinement_steps, item_index)` 중복·누락과 0–99 인덱스 집합을 검사한다.
두 CSV를 합칠 때는 defense도 키에 포함하여 서로 다른 실행을 충돌로 취급하지 않는다.

기존 `summary.csv` 및 `summary.md`의 11개 ASR-e/ASR-k 값과 전부 일치한다.
사용자가 제시한 128 Step 검산값도 일치했다. 원자료 값을 수정하거나 누락 라벨을 보정하지 않는다.

### 같은 응답과 평가 대상 구간의 근거

1. `DiffuGuard/run_generate.py::run`은 원본 `jailbreakbench_llada.generate_response()`의 반환값을
   동일 레코드의 `response`에 저장한다.
2. 고정 DiffuGuard 원본 `models/jailbreakbench_llada.py::generate_response`는
   `output_ids[:, matching_count:]`를 decode하고 DIJA에서 `assistant\n` 앞을 반환한다.
   따라서 이 경로의 `response`는 DIJA 사용자 턴 인필 구간이며 별도의 Assistant Tail 평가가 아니다.
   matching_count는 코드가 계산한 경계이므로 별도의 이상적인 토큰 span 검증으로 과장하지 않는다.
3. `DIJA/run_eval.sh::run_asr_k`는 생성 JSON의 평가 복사본에 원본
   `run_jailbreakbench/eval_metric/evaluate_completions_asr_k.py`를 적용한다.
   저장소의 `DIJA/tests/test_eval_scripts.py::test_asr_k_labels`도 거절 표현 유무에 따른 0/1 계약을 명시한다.
4. RQ1의 ASR-e는 `DIJA/eval_asr_e.py`의 Together 판정이 아니라 `DiffuGuard/eval_asr.py`다.
   `judge_pair(question=auto)`는 원본 `evaluation.py::_extract_prompt_and_response`를 이용해
   `refined prompt`(없으면 `prompt`)와 동일 `response`를 전달한다. 앞뒤 공백은 strip한다.
5. `summarize_pilot.py::main`과 `summarize_with_diffuguard.py::summarize_asr`는 같은 평가 JSON
   레코드에서 `asr_k_label`, `evaluation_result.response_safety_category`, `response` 해시를 함께 추출한다.

이는 저장 코드와 레코드의 동일 응답 사용 근거다. 원본 평가 JSON을 현재 로컬에서 확보하지 못했으므로
두 판정의 실제 입력을 원문 단위로 재대조한 것은 아니다. 12자리 해시만으로 입력이 같았음을
독립적으로 증명할 수 없고, 현재 DIJA submodule이 비어 있어 정확한 키워드 목록 전체도 새로 감사하지 않았다.
ASR-k는 요청대로 CSV 라벨을 그대로 사용하며 재계산하지 않는다.
읽은 DiffuGuard 원본은 기존 로컬 `RQ2/outputs/vessl_assets/sources/DiffuGuard`의 clean 고정 checkout이다.

`item_index`는 두 집계 스크립트의 `enumerate(data)`로 만든 배열 순번이다. 원래 JBB 고유 ID가 아니다.
같은 파일 해시·전체 100문항·동일 순서 실행 코드가 짝지음의 근거지만, 문항별 프롬프트 해시나 원문 없이
조건 사이의 순서 대응을 완전히 독립 검증할 수는 없다. 이 제한을 paired 분석에 적용한다.

## 3. ASR-e와 ASR-k의 차이 및 계산

- E=1: UNSAFE_COMPLIANCE, E=0: SAFE_REFUSAL. 다른 Category는 실패 처리한다.
- K는 CSV의 asr_k_label 0/1이다. 1은 거절 키워드 부재 기반 공격 성공이며 의미적 유해성의 완전한 판정이 아니다.
- 네 셀 E1/K1, E1/K0, E0/K1, E0/K0의 합은 100이다.
- 불일치율은 `(E1/K0 + E0/K1)/100`, ASR 차이 K−E는 `(E0/K1 − E1/K0)/100`이다.
- 모두 유효 판정이므로 ASR-e·ASR-k의 분모는 각 조건 100이다. 미판정은 0이며 조용히 제외하지 않는다.
- 방어 전후 전환은 행=Vanilla, 열=DiffuGuard의 4×4 교차표로 기록한다. 비교 대상 응답 자체는 방어에 따라 달라진다.

불확실성은 100개 item_index를 복원추출하는 paired bootstrap 5000회, 분석용 seed 0으로 계산한다.
매 회 같은 표집 인덱스를 모든 11조건에 적용한다. 조건별 불일치율과 동일 Step의
DiffuGuard 비례 − Vanilla 불일치율 차이에 2.5/97.5 percentile 선형 보간 구간을 제시한다.
1000개/800개/400개 독립 관측으로 합쳐서 추정하지 않는다. 문항 간 독립성은 가정하며, 시드·Judge 재호출
변동과 표본 선택 불확실성까지 포함하지 않는다. 다중비교 보정 없는 탐색적 구간이며 유의성 검정은 추가하지 않는다.
Kappa는 계산하지 않았다. 이 분석에서는 방향별 교차표와 불일치율이 질문에 직접 답하며,
클래스 불균형에 민감한 단일 지표를 더하지 않아도 된다.

## 4. 실험 결과

[summary.md](summary.md)에 주·보조 조건, 일치율, 방향별 불일치, paired 차이와 구간을 분리했다.
[metrics.json](metrics.json)에는 전체 집계와 방어 전후 4×4 전환 건수를,
[crosstab.csv](crosstab.csv)에는 조건별 건수·비율을 저장했다. 응답·프롬프트·Judge 이유는 저장하지 않는다.

128 Step에서 Vanilla E1/K0=4, E0/K1=16으로 20% 불일치다.
DiffuGuard Repair 16은 각각 11/52로 63%다. 방어 전후 일치→불일치 49건,
불일치→일치 6건으로 차이는 +43%p다. 비례 조건 전체의 불일치는 63/67/60/73%로 단조롭지 않다.
보조 고정 Repair 8의 128/32/16 Step 불일치는 59/62/66%다.

## 5. 우리 연구에서의 의미

ASR 주변 비율 차이는 서로 반대 방향의 불일치를 상쇄한다. 같은 생성 결과에 대해서도
거절 키워드 기반 기준과 의미 기반 기준이 다른 정보를 제공하므로 지표 이름과 평가 구간을 함께 명시해야 한다.
이 분석은 두 지표를 대체하거나 판정 정확도 순위를 정하지 않는다. 인필/Tail 차이는
RQ3가 이미 다룬 전제이며 이번 결과의 새로운 발견으로 주장하지 않는다.

## 6. 해석상의 한계

- 단일 seed·DIJA-JBB 100문항에 한정된다. 다른 모델·공격·표본으로 일반화할 수 없다.
- 원문이 없어 키워드가 무관한 맥락에 있었는지, 불완전 출력인지, Judge가 오판했는지 확정할 수 없다.
- ASR-k가 틀렸거나 LLM Judge가 더 정확하다는 결론은 도출하지 않는다.
- 방어 전후 생성 텍스트가 달라지므로 불일치 변화는 SAR·Audit·Repair 어느 하나의 인과 효과가 아니다.
- 실행 메타데이터가 dirty commit을 가리킨다. source commit과 당시 미커밋 코드의 완전한 대응은 미검증이다.
- 현재 `eval_asr.py`는 기록된 초기 실행 커밋 대비 checkpoint 저장 시점/원자적 파일 저장이 변경됐다.
  확인한 diff에서는 Judge 입력 추출·Category 집계의 변경은 없다. 다만 dirty 실행까지 확정할 수는 없다.
- 연구팀 연결의 [RQ1 실험 개요](https://app.notion.com/p/40fd926b6fb08007936ffb63665af4d7)는
  GPT-4o-mini·복수 seed·사람 검수 계획을 포함하지만 실제 runs.json은 gpt-6-luna·seed 0이다.
  계획을 실행 결과로 취급하지 않고 실제 메타데이터를 분석 기준으로 삼았다. Notion은 수정하지 않았다.

## 7. 추가 검증이 필요한 사항과 재현

불일치의 구체적 원인 분류에는 승인된 원문 저장소의 접근 가능 여부, 문항 대응, 판정 이유와
독립 검수 기준을 확인해야 한다. 현재 `DiffuGuard/outputs/`에는 원문이 없고 기존 요약은 VESSL 보관을
가리킨다. 이번에는 다운로드·새 Judge 호출·GPU 실행·원문 재채점을 하지 않았다.

```bash
python -m unittest RQ1.analysis.test_asr_label_disagreement
python -m RQ1.analysis.asr_label_disagreement --bootstrap 5000 --seed 0
```

표준 라이브러리만 필요하다. 원본은 읽기 전용이며 위 명령은 분석 전용 출력 네 파일을 재생성한다.
provenance.json에 입력/코드의 SHA-256, 실행 명령, Python, 기준 main과 분석 checkout commit,
11조건의 메타데이터를 기록한다. Git 줄바꿈 변환에 따라 바이트 해시가 달라질 수 있으므로 이를 구분한다.
원래 실험 seed와 분석용 bootstrap seed는 별개다.

검증 과정에서 설명문에 잘못 입력된 Step별 불일치 숫자를 발견해 집계 결과로부터 자동 생성하도록 수정했고,
설명문 검산 테스트를 추가했다. 원자료나 라벨을 바꾼 것은 아니다. 최소 테스트 8개는 라벨 매핑,
교차표·비율, 중복·누락, Repair 구분, 공동 재표집, 전환표, 기존 요약·메타데이터·초기 검산,
설명문과 수치의 일치를 검사한다.

# RQ1 Audit Score와 GSM8K Utility의 탐색적 분석

## 범위와 입력

main `abd562d717550caca9417ded3a1a7538963638b1`의 기존 수치 결과만 분석한다.
`RQ1/results/gsm8k_250/per_item.csv`와 `diffuguard_per_item.csv`를 `(steps, doc_id)`로 짝짓는다.
각 CSV는 250문항 × 4 Step이며, `RQ1/data/ids/gsm8k_250.txt`의 고정 250개 ID와 완전 일치해야 한다.
중복 키, 결측 조건, 잘못된 불리언, 비유한 Audit Score, 주요 메타데이터 불일치는 오류로 처리한다.
추출 답안 열은 분석하거나 출력하지 않는다. 파생 결과에는 집계 수치와 재현성 정보만 저장한다.

- 모델: LLaDA-8B-Instruct `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- 생성 256, 블록 32, temperature 0, seed 0, GSM8K 5-shot.
- Vanilla 실행 커밋: `55d7ba6ea5db9024d9600ceb12808d16618d6ea9`.
- DiffuGuard 실행 커밋: `09febc36a0ad497a8a9f01760942ae03096e8706`.
- 두 실행 커밋 및 분석 기준 main 사이에서 `DiffuGuard/run_gsm8k.py`, `DiffuGuard/semi_ar.py`의 diff는 없다.
- 저장된 실행 환경: PyTorch 2.6.0+cu124, Transformers 4.52.3, RTX 3090.
- DiffuGuard 수리 예산은 256/128/64/32 Step에 각각 16/8/4/2회다. RQ2의 고정 수리 8회와 혼동하지 않는다.

`runs.json`, `diffuguard_runs.json`에서 실행 설정을 확인한다. 응답 원문은 사용하지 않았으므로
저장된 정답 불리언의 원문 기반 재채점은 미검증이다. SHA-256은 입력 파일의 동일성을 확인할 뿐,
원래 실행의 무결성 전체나 원격 환경을 독립적으로 입증하지 않는다.

## Audit Score가 Step에 따라 변하지 않는 이유

실제 코드 경로는 다음과 같다.

1. `DiffuGuard/run_gsm8k.py::run`은 문항별로 seed를 초기화하고, `compute_baseline_hidden()`에
   동일한 문항 context를 전달한다. `sp_mode=hidden`이며 기준 은닉 상태가 없으면 중단한다.
2. 고정 DiffuGuard `86aeb75203d07064c1ced650edaee74563647d36`의
   `models/jailbreakbench_llada.py::compute_baseline_hidden`은 마스크 꼬리를 붙이지 않은
   채팅 프롬프트를 한 번 forward하고 마지막 층 전체 시퀀스의 평균을 구한다(175–205행).
3. 같은 원본의 `utility/generate_function_llada.py::generate`는 블록의 `i == 0`에서
   마지막 층의 현재 블록 위치 은닉 상태 평균을 저장한다(294–297행). 첫 블록에서는
   아직 토큰이 확정되지 않았고 입력은 동일한 프롬프트 + 256개 마스크다.
4. 블록 디노이징이 끝난 뒤, 저장한 **첫 forward**의 벡터와 기준 벡터의 cosine distance를
   계산한다(352–356행). `cosine_distance`는 float32 정규화 후 `1 - dot(a,b)`를 반환한다(45–50행).
5. `run_gsm8k.py`의 `_record_sd`는 첫 번째 거리만 `state['sd']`에 저장하고,
   최종 `audit_score`를 그 값으로 덮어쓴다(230–235, 306–307행).
   `RQ1/scripts/summarize_gsm8k.py`는 해당 문항 필드를 CSV로 옮긴다.

따라서 이 점수는 **첫 블록 생성 결과의 완성도나 Step별 최종 응답의 위험도**가 아니다.
모델·프롬프트·초기 마스크 배치가 같고 eval 경로가 결정적이면, 총 Step 수가 바뀌어도
첫 forward의 입력과 두 은닉 벡터는 같다. 총 Step 수는 이후 확정 스케줄과 수리 예산에 영향을 준다.
실제로 250문항 모두 네 Step의 점수가 정확히 같고 문항 내 범위는 0이다.
동일한 점수를 CSV에 잘못 복제했다는 근거는 발견하지 못했으며 코드의 측정 시점과 일치한다.
GPU에서 은닉 벡터를 다시 계산한 것은 아니므로 수치적 재현까지 새로 검증한 것은 아니다.

`semi_ar.py::load`는 원본의 첫 블록 확대 조건만 메모리에서 비활성화한다.
감사·수리 및 은닉 측정 시점은 바꾸지 않는다. 외부 원본은 수정하지 않았다.
이번 코드 확인은 로컬에 보관된 clean 원본 checkout
`RQ2/outputs/vessl_assets/sources/DiffuGuard`에서 위 커밋을 확인한 뒤 수행했다.
원본 파일은 이 분석에 복사하거나 새로 업로드하지 않는다.

추가 주의점: 원본은 logit 점수가 기준을 넘으면 `unsafe_flag=True`로 설정하고,
hidden 거리도 기준을 넘으면 True로 설정한다. 따라서 기록된 hidden 점수만으로 일반적인
감사 판정 전체를 재구성할 수 없다. 이 자료에서는 hidden 최솟값 자체가 0.1보다 크고
1000행 모두 수리되어 이 구분으로 인한 미수리 표본은 없다. Vanilla의 점수 0은 감사 OFF의
기록값으로, DiffuGuard의 실제 hidden 점수와 같은 측정값으로 해석하지 않는다.

## 분석 방법

- 주 지표는 `correct_flexible`이다. `correct_strict`는 별도 표에 둔다.
- 각 Step에서 정답 유지 / 정답→오답 / 오답→정답 / 오답 유지의 수와 전체 250문항 대비 비율을 계산한다.
- 정확도 차이는 DiffuGuard − Vanilla(pp), 조건부 손실률은 정답→오답 / Vanilla 정답 수다.
- Audit AUC는 **해당 Step에서 Vanilla 정답인 문항만** 대상으로 한다. 양성은 DiffuGuard 오답이다.
  무작위 손실 문항의 점수가 정답 유지 문항보다 높을 확률에 동률의 절반을 더한다.
  임계값 선택이나 예측 모델 학습은 없다. 0.5보다 작으면 높은 점수가 손실 방향을 구분하지 못한다.
- 전체 요약 AUC는 Step별 AUC의 동일 가중 평균이다. 서로 다른 Step의 정답 모집단이 달라질 수 있으므로
  한 모집단의 AUC나 1000개 독립 관측의 AUC로 해석하지 않는다.
- 표본 단위는 문항이다. 250개 doc_id를 복원추출하고 동일 인덱스로 모든 Step을 함께 재표집한다.
  bootstrap 2000회, 분석용 seed 0, 2.5/97.5 percentile 보간 구간을 사용한다.
  분석용 난수는 원래 모델 생성 seed를 변경하지 않는다. 한 클래스가 없는 재표집은 AUC 계산에서 제외하며
  유효 횟수를 JSON에 기록한다. 구간은 다중비교 보정 없는 탐색적 구간이다.

## 확인된 사실과 해석의 한계

수치는 자동 생성된 [summary.md](summary.md), [transitions.csv](transitions.csv), [metrics.json](metrics.json)에 있다.
기존 DiffuGuard 요약의 flexible 정확도 차이와 손실/이득 문항 수가 일치한다.
250문항의 Audit Score는 Step 전체에서 불변이지만, 손실률과 정확도는 Step별로 다르다.
flexible 평균 AUC의 95% 구간은 0.5를 포함한다. 이 점수의 큰 값이 utility 손실을
일관되게 구분한다는 증거가 부족하다. 무관함이나 역인과를 입증한 것은 아니다.

이 분석은 한 seed와 고정 부분집합에 대한 관찰 비교다. 문항 간 독립성을 가정하는 bootstrap은
문제 유형 간 상관, 표본 선택, seed·환경 변동을 포괄하지 않는다. Vanilla 정답 조건부 분석은
Step별로 선택된 표본이며 이를 전체 GSM8K나 해로운 요청으로 일반화할 수 없다.
전 문항 수리, SAR 사용, 생성 Step와 비례 수리 예산의 동시 변화 때문에 감사·수리·SAR 각각의
인과 효과를 분리할 수 없다. 감사값 자체도 마스크 입력과 기준 프롬프트의 표현 차이에 영향을 받을 수 있다.
strict의 형식 민감도와 적은 유지 표본도 함께 고려해야 한다.
strict 32 Step에서는 정답 유지가 4문항뿐이라 2000회 중 29회가 단일 클래스 재표집이었다.
해당 Step 및 strict 평균 AUC 구간은 유효한 1971회에 조건부로 계산했으며 불안정성이 크다.
초기 결과 점검에서 모든 AUC 재표집이 유효할 것이라는 검사는 실패했다. 원인을 위 희소 표본으로
확인했고, 계산 규칙이나 원자료를 바꾸지 않고 유효 횟수와 제외 사실을 명시했다.

## 다른 연구와의 범위 구분

- 확인한 `rq3-prophet` 커밋 `5cf3c156b158d89a8485997d43f1d80bb6dc0feb`의
  `RQ3/README.md`: 은닉 궤적, 거절 확정 시점, 프로브, Prophet 조기 종료 및 재표집 실험.
- 확인한 `safe-gate` 커밋 `475f655c1e55df7d37068dd580f52e46c620bd8d`의
  `SafeGate/README.md`: 토큰 질량 기반 안전 게이트와 별도 개발셋 기준값 선택.
- 이번 분석은 RQ1의 이미 저장된 Audit Score와 정답 전환의 관계만 집계한다.
  새 trace·프로브·게이트·임계값 튜닝이나 RQ2/RQ3 방법 변경을 하지 않는다.

## 후속 분석 제안

기존 기록에 문항별 길이·유형이나 첫 forward의 벡터 요약이 있다면 문항 난도·길이의 교란 가능성을
별도 분석할 수 있다. 추가 seed나 수리/SAR 분리 비교는 별도의 설계와 실행 승인이 있어야 한다.
점수 기준값을 이 250문항에 맞춰 선택하는 것은 본 탐색 결과로 정당화하지 않는다.

## 재현

저장소 루트에서 표준 라이브러리만 사용한다. PyTorch, GPU, 모델, API 접속은 필요 없다.

```bash
python -m unittest RQ1.analysis.test_audit_utility
python -m RQ1.analysis.audit_utility --bootstrap 2000 --seed 0
```

`summary.md`, `metrics.json`, `transitions.csv`, `provenance.json`은 분석 전용 출력이며 위 명령으로 갱신된다.
원본 CSV와 메타데이터는 읽기만 한다. 입력 파일·분석 코드 SHA-256, Python 버전,
실행 명령, 분석 당시 checkout 및 원래 실험 커밋은 `provenance.json`에 기록한다.
커밋 이후 재실행하면 checkout SHA는 바뀌지만 수치 결과는 동일해야 한다.
Windows CRLF와 Linux LF의 바이트 해시는 달라질 수 있으므로 동일 내용 여부는 줄바꿈도 확인한다.

최소 테스트 5개는 AUC 방향·동률·단일 클래스, 4종 전환 계산, 키 누락/중복 거부,
CSV 중복 거부, 동일 문항의 Step 간 bootstrap 결합을 검증한다. GPU 및 기존 전체 테스트는 실행하지 않는다.

# RQ3: 거절 확정 시점 측정 (`commit_time`)

디노이징 forward마다 은닉 상태를 저장하고, 응답 결과(거절 / 해로운 순응)를 맞히는 선형 프로브로
"결정이 언제 굳는가"(확정 시점 s\*)와 "얼마나 망설이는가"(n_τ)를 잰다. 실험 절차와 성공 기준은 명세서 문서를 따른다.

## 설치와 테스트

```bash
cd DLM_safety_code/RQ3
pip install -r requirements.txt          # DiffuGuard 환경(torch, transformers) 위에
python -m pytest -q tests                # CPU, 가짜 LLaDA로 2초
```

`tests/test_decode_capture.py::test_quota_equals_original_vanilla`는 `decode.py`의 quota 규칙이
DiffuGuard 원본 `generate()`의 vanilla 경로와 같은 시드에서 같은 토큰을 내는지 확인한다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `commit_time/capture.py` | 모델 hook으로 forward마다 은닉 상태(추적 위치 평균)와 커밋·재마스킹 수를 기록. 디코더를 고치지 않음 |
| `commit_time/decode.py` | KV 캐시 없는 단일 블록 디코더. `quota` = DiffuGuard vanilla와 동일, `threshold` = Fast-dLLM 규칙. 중간 상태에서 재개 가능 |
| `commit_time/run_trace.py` | `DiffuGuard/run_generate.py`를 그대로 실행하면서 trace를 남김 |
| `commit_time/resample.py` | 중간 상태에서 시드만 바꿔 이어 생성 (행동 확정 시점 검증) |
| `commit_time/probe.py` | 그룹 CV(문항 단위)로 층·C 선택, τ 결정 |
| `commit_time/metrics.py` | s\*, n_τ, s_text, 결정 전 커밋 비율, 수리 전후 마진 |
| `commit_time/analyze.py` | `labels` · `fit` · `score` · `behavior` · `flips` CLI |

## 실행 예

```bash
export RQ3_OUT=/root/rq3-outputs            # 팀 공용 볼륨 (git 밖)
# 1) 기준 trace (원본 디코더, vanilla, 128스텝)
python -m commit_time.run_trace --trace_dir $RQ3_OUT/trace/R --decoder original \
  -- --defense vanilla --steps 128 --output_dir $RQ3_OUT/gen --tag rq3_R
# 2) 판정 -> 라벨 (텍스트는 복사하지 않음)
python ../DiffuGuard/eval_asr.py $RQ3_OUT/gen/<R 생성 파일>.json
python -m commit_time.analyze labels $RQ3_OUT/gen/eval_<R 생성 파일>.json --out $RQ3_OUT/labels/R.csv
# 3) 프로브 학습 -> 점수
python -m commit_time.analyze fit --run $RQ3_OUT/trace/R $RQ3_OUT/labels/R.csv \
  --run $RQ3_OUT/trace/D $RQ3_OUT/labels/D.csv --out $RQ3_OUT/probe/probe.npz
python -m commit_time.analyze score --probe $RQ3_OUT/probe/probe.npz \
  --run $RQ3_OUT/trace/A0.9 $RQ3_OUT/labels/A0.9.csv --out $RQ3_OUT/metrics/A0.9.csv
```

## 보안

- `hidden/*.npz`, `trace_meta.jsonl`, `labels.csv`, `metrics*.csv`에는 텍스트가 없다 (벡터, 개수, 해시 앞 12자).
- 모델 텍스트는 `run_generate.py`와 `resample.py`가 쓰는 생성 JSON에만 있다. `$RQ3_OUT`은 git 밖에 둔다.
- 중간 상태 x_s(부분 텍스트)는 메모리에서만 쓰고 저장하지 않는다.
- KV 캐시 디코더는 지원하지 않는다 (`TraceRecorder`가 부분 입력 forward를 거부).

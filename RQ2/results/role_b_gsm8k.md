# RQ2 담당 B: GSM8K With DiffuGuard 결과

담당 B(결합 디코더)가 GSM8K With DiffuGuard 0·A–E를 250문항씩 돌린 결과와 GPU 검증 기록이다.
수치만 담았고, 모델 응답 원문은 `RQ2/outputs/`(git 제외)에만 있다.

## 1. 상태

- With DiffuGuard 0/A/B/C/D/E 여섯 조건 모두 250문항 실행을 마쳤다(2026-11-13 UTC, RTX 3090).
- 여섯 조건 모두 남은 마스크 0, 무효 답 0, baseline 재시도 0이다.
- 수리 비율은 모든 조건에서 1.0이다(250/250). 아래 5절 참고.
- RQ1 With 재현이 어긋났던 원인은 실행 환경 차이로 확인했다(4절).

## 2. 본실행 결과 (250문항, With DiffuGuard)

| 열 | flexible (95% CI) | strict (95% CI) | 생성 NFE | 캐시 NFE | 수리 NFE | baseline | 합계 NFE | 시간 합계(초) | tok/s | With 0 대비 속도 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 68.4% (62.4–73.8) | 44.8% (38.8–51.0) | 256.00 | 0.00 | 8 | 1 | 265.00 | 24576.0 | 2.113 | 1.00x |
| A | 64.8% (58.7–70.5) | 42.0% (36.0–48.2) | 96.25 | 0.00 | 8 | 1 | 105.25 | 9842.0 | 5.281 | 2.50x |
| B | 70.4% (64.5–75.7) | 47.6% (41.5–53.8) | 65.79 | 0.00 | 8 | 1 | 74.79 | 6981.9 | 7.567 | 3.52x |
| C | 58.8% (52.6–64.7) | 33.6% (28.0–39.7) | 45.59 | 0.00 | 8 | 1 | 54.59 | 5103.6 | 10.441 | 4.82x |
| D | 71.6% (65.7–76.8) | 45.6% (39.5–51.8) | 98.96 | 90.96 | 8 | 1 | 107.96 | 3461.9 | 14.920 | 7.10x |
| E | 69.2% (63.2–74.6) | 35.6% (29.9–41.7) | 106.51 | 98.51 | 8 | 1 | 115.51 | 2909.4 | 18.003 | 8.45x |

- NFE는 문항당 평균이다. 캐시 NFE는 생성 NFE의 일부다. 감사는 마지막 스텝 logits와 hidden을 다시 써서 별도 forward가 없다(감사 NFE 0).
- 시간은 문항별 `wall_ms`(CUDA 동기화) 합계다. 생성, baseline hidden, 감사, 수리를 포함하고 모델 로드, 토큰화, 채점, 저장은 제외한다.
- CI는 러너가 기록한 95% Wilson 구간이다. 다중 비교 보정은 하지 않았다.

## 3. Without(담당 A)과 나란히 보기

Without 수치는 `role_a_gsm8k.md`의 250문항 결과다.

| 열 | flexible Without → With | strict Without → With | 생성 NFE Without → With | 시간(초) Without → With | 시간 비율 With/Without |
|---|---|---|---|---|---:|
| 0 | 77.6 → 68.4 (−9.2) | 40.8 → 44.8 (+4.0) | 256.00 → 256.00 | 23612.7 → 24576.0 | 1.04 |
| A | 78.8 → 64.8 (−14.0) | 43.2 → 42.0 (−1.2) | 76.94 → 96.25 | 7199.4 → 9842.0 | 1.37 |
| B | 76.0 → 70.4 (−5.6) | 39.6 → 47.6 (+8.0) | 48.08 → 65.79 | 4499.9 → 6981.9 | 1.55 |
| C | 64.4 → 58.8 (−5.6) | 31.6 → 33.6 (+2.0) | 34.23 → 45.59 | 3210.6 → 5103.6 | 1.59 |
| D | 78.0 → 71.6 (−6.4) | 36.8 → 45.6 (+8.8) | 79.81 → 98.96 | 2203.0 → 3461.9 | 1.57 |
| E | 76.4 → 69.2 (−7.2) | 39.2 → 35.6 (−3.6) | 83.95 → 106.51 | 1757.8 → 2909.4 | 1.66 |

**이 비교는 참고용이다.** 두 행은 다음 조건이 다르다.

- **Transformers 버전:** Without은 4.46.2, With는 4.49.0으로 돌렸다. GSM8K에서 버전에 따라 생성 토큰이 달라지는 것을 확인했으므로(4절), 정확도와 시간 차이에 방어 효과와 환경 차이가 섞여 있다.
- **문항별 대응 비교를 하지 않았다:** A의 문항별 결과 파일이 이 저장소에 없어 McNemar 같은 짝지은 검정을 하지 않았다.

방향만 보면 다음과 같다.

- 방어를 켜면 flexible 정확도는 여섯 조건 모두 내려갔다(−5.6 ~ −14.0%p). strict는 조건마다 방향이 다르다.
- 임계값 열(A–E)에서는 방어를 켜면 생성 NFE도 늘었다(19~27%). DiffuGuard 점수에 난수 항 `α·R`이 들어가 확정 순서가 바뀌고, 그만큼 τ를 넘는 토큰이 늦게 나오는 것으로 추정한다.
- 그래도 With 행 안에서는 가속이 유지된다. With 0 대비 시간은 A 2.5배, C 4.8배, E 8.5배 빨랐다.

## 4. GPU 검증 기록

모두 `scripts/verify_gpu.py`로 돌렸고 결과 파일은 `RQ2/outputs/verify/`에 있다.

| 검증 | 범위 | 결과 | 파일 |
|---|---|---|---|
| ①-2 방어 끔 = Fast-dLLM 원본 | A·D·E, 10문항씩 | 토큰·NFE 30/30 일치 | `check1-2_fastdllm_limit10.json` |
| 로더 비교 (LLaDAModelLM vs AutoModel) | 10문항 | hidden·감사 점수·토큰 비트 단위 일치 | `logs/check_loader.log` |
| ③ 수리 후 캐시 재구성 | D·E dg, 10문항씩 | 20/20 통과, 수리 전 캐시 재사용 0건 | `check3_cache_limit10_spNone.json` |
| ② RQ1 Without 재현 | 열 0 none, 10문항 | 답·정답 여부·토큰 수 10/10 일치 | `check2_rq1_rq2-gsm8k-0-none-s0_limit10.meta_vs_vanilla256.json` |
| ② RQ1 With 재현 (환경 다름) | 열 0 dg, 원본 α, 수리 16회 | NFE·수리·감사 판정 10/10, 정답 여부 9/10, 답 7/10 | `check2_rq1_rq2-gsm8k-0-dg-s0_limit10_rq1with.meta_vs_diffuguard256.json` |
| ② RQ1 With 재현 (같은 환경) | 위와 같은 설정, RQ1 원본 스크립트를 같은 환경에서 실행 | 응답·답·정답 여부·토큰 수 10/10 일치 | `check2_envcheck_rq1original_vs_rq2.json` |

마지막 줄로 RQ1 With 불일치의 원인이 결합 디코더가 아니라 실행 환경 차이였음을 확인했다.

## 5. 수리 비율 1.0

- 감사 점수(블록 0 첫 스텝 hidden과 baseline hidden의 코사인 거리)는 0.373–0.597(중앙값 0.463)로, 250문항 모두 임계값 λ = 0.1을 넘었다.
- 감사는 블록 0의 첫 스텝에서 하므로 여섯 조건의 감사 점수가 문항마다 같다.
- 5-shot 템플릿 프롬프트는 무해해도 위험으로 판정된다. 원본 DiffuGuard와 같은 동작이다. RQ1 With GSM8K의 수리 비율과 비교해 볼 필요가 있다.

## 6. 설정과 환경

- 모델: LLaDA-8B-Instruct `08b83a6feb34df1a6011b80c3c00c7563e963b07`, Fast-dLLM `LLaDAModelLM` (`a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`)
- DiffuGuard `86aeb75203d07064c1ced650edaee74563647d36`
- RTX 3090 24GB, PyTorch `2.3.1+cu121`, Transformers `4.49.0`
- 문항: `RQ1/data/ids/gsm8k_250.txt`, 프롬프트 파일 sha256 `028420a0b305a1f47ea6f229592bc82b4ae7417677bf2ce9fef166f8bb9578c6`
- 생성 길이 256, 블록 32, temperature 0, 시드 `seed + doc_id`(seed 0)
- 열 0은 고정 256스텝, A/B/C는 τ 0.9/0.7/0.5, D/E는 τ 0.9 + Prefix/Dual 캐시
- 방어: 진행률 α(α0 = 0.3), 감사 λ = 0.1, 수리 시 90% 재마스크 후 8스텝, suppression 1e6, 수리에는 τ와 캐시를 쓰지 않음

## 7. 출처와 재현

- 코드: 여섯 조건 모두 커밋 `9a79118`에서 실행했다. 메타의 `team_repo_dirty`는 True다. 당시 어떤 파일이 커밋되지 않은 상태였는지는 기록되지 않았다.
- **E 이어 돌리기:** E는 첫 시도(커밋 `7ee047a`)가 doc 202에서 GPU 메모리 부족으로 멈췄다. `9a79118`(softmax를 현재 블록 행에만 계산)로 고친 뒤 앞서 끝난 43문항을 이어받아 나머지를 돌렸다. 이 수정은 행별 값을 바꾸지 않으며, GPU에서 파일럿 응답과 같은 것을 확인했다(E dg, D none, C dg 10/10, 0 dg 2/2).
- 실행 스크립트: `RQ2/outputs/scripts/run_with_main.sh`(E→D→C→B→A→0 순서, 다시 실행하면 이어서 함). 조건 하나는 다음 명령과 같다.

```bash
cd RQ2
FAST_DLLM_ROOT=/root/Fast-dLLM CUDA_VISIBLE_DEVICES=0 \
python run_rq2_gsm8k.py --column A --defense dg \
  --model_path /root/models/LLaDA-8B-Instruct --prompt_file /root/data/gsm8k_5shot_lmeval.json
```

- 결과 파일 (`RQ2/outputs/gsm8k/`, sha256)

| 파일 | sha256 |
|---|---|
| `rq2-gsm8k-0-dg-s0.meta.json` | `1b3fbe6ca5ba4f171d04b4bce1dd2c2b30588c18bb686f4e969a72d39fa3c321` |
| `rq2-gsm8k-0-dg-s0.raw.jsonl` | `34cc209eef8420b0bda00cb4df71e227126b75d130f8f32b0759fec2bd6d070a` |
| `rq2-gsm8k-A-dg-s0.meta.json` | `91bda9ecadc08f3ac7aab06bb41ff29a2ff854c1ec51703f02f949be028156ae` |
| `rq2-gsm8k-A-dg-s0.raw.jsonl` | `20b6312256a6c5810c8ebd2e3de2e5f3a408e7fa931b484fbc31bf34f369ac1f` |
| `rq2-gsm8k-B-dg-s0.meta.json` | `dce3e20132609b263a2cef3f6b86a8aa2a3c5a792f4aae41c46a24708bad1b1d` |
| `rq2-gsm8k-B-dg-s0.raw.jsonl` | `f856cc8479d5a8eba365c49de8c00da96fba63215863f513e6b9c94820c16b0d` |
| `rq2-gsm8k-C-dg-s0.meta.json` | `c8b03a6017ebac3bb60a27e7bf1cecd94b6eae1f8caa2833e7ca2dd578e17869` |
| `rq2-gsm8k-C-dg-s0.raw.jsonl` | `d27523d63b2b1f745208bfdc07e0edc4279725b0b4ea62debda94968c581b873` |
| `rq2-gsm8k-D-dg-s0.meta.json` | `a00d894116eb031ac3b5a3c028f457a2afcdaf6342bb8cf6eda526e730fc2fa7` |
| `rq2-gsm8k-D-dg-s0.raw.jsonl` | `563ff9e031c207a1afa4bf67256fb2d705875daadae2863e19e69251c2f8d040` |
| `rq2-gsm8k-E-dg-s0.meta.json` | `fa07540d6e0fb88e2bc557c2398e1fcc005ed37b01d1d4f027a01f6dc0dc5a21` |
| `rq2-gsm8k-E-dg-s0.raw.jsonl` | `f08aa3ba1ca97d265a987c92fcf414a7b7dc23731283ad935b9fb5e2b63f1f60` |

## 8. 남은 일

1. With와 Without을 같은 Transformers 버전으로 맞출지 팀에서 결정한다. 맞춘다면 한쪽 행을 다시 돌리고, 문항별 짝지은 비교를 추가한다.
2. RQ1 With GSM8K의 수리 비율과 비교한다(5절).
3. ASR With 경로: `d321365`를 main에 올리고, `generate_guarded()`와 `types.py` 수정을 거쳐 C의 ASR 러너에 연결한다.

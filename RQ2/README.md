# RQ2 ASR — Without / With DiffuGuard

담당 C의 ASR 실행·판정·집계 코드와 숫자 결과다. GSM8K는 이 문서의 범위에 포함하지 않는다.
최종 확인: 2026-11-24 UTC. 이 공유본은 이미 수행한 실험의 코드·결과 정리이며, 새 실험을 자동으로 시작하지 않는다.

## 현재 완료 상태

| 구분 | 생성 | ASR-k | Luna ASR-e | GPT-4o mini ASR-e |
|---|---:|---|---:|---:|
| Without 0/A~E | 600/600 | 완료 | 600/600 | 600/600 |
| With 0/A~E | 600/600 | 완료 | 599/600 | 597/600 |

With의 미판정 4건은 모두 `dija-018` 관련이다. Luna E는 API의 `bio_policy` 거부로 미판정이며,
동일한 판정 입력을 갖는 Mini A/D/E의 각 1건은 재전송하지 않았다. SAFE/UNSAFE를 임의로 채우지 않았다.
`partial` / `done_with_unscored`는 이 미판정을 나타내며, 현재 실행 중인 생성·판정 작업은 없다.

가장 먼저 볼 파일은 [최종 12조건 요약 CSV](results/rq2_asr_summary.csv)다.
응답 원문, 개별 판정 reasoning, 모델 가중치와 API 키는 공유하지 않는다.

## 실험 설정과 방법의 범위

- 모델: `GSAI-ML/LLaDA-8B-Instruct`, revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- Fast-dLLM: `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`.
- With 결합 디코더: 별도 `rq2-combined-decoder` 브랜치의 `d321365a1f2dad28fdda36e1b540b04c0a56b624`.
- 실제 C 실행 환경: RTX 3090 24GB, PyTorch `2.3.1+cu121`, Transformers `4.46.2`, bf16, PyTorch SDPA.
- DIJA × JBB 고정 100문항: `DIJA/third_party/DIJA/run_jailbreakbench/refine_prompt/jailbreakbench_data_refined_Qwen.json`.
- 입력 SHA-256: `534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95`.
- ID/순서: [전체](data/ids/dija_100.txt)의 `dija-000`~`dija-099`; [파일럿](data/ids/dija_pilot10.txt)은 앞 10개.
- 문항별 시드: 기본 시드 0 + **원본 인덱스**. 부분집합을 새로 재번호하지 않는다.
- 생성 길이 128, 블록 길이 32, temperature 0.5.

| 조건 | 토큰 확정 | τ | 캐시 |
|---|---|---:|---|
| 0 | 고정 128회, 블록당 32회 | 없음 | 없음 |
| A | 임계값 | 0.9 | 없음 |
| B | 임계값 | 0.7 | 없음 |
| C | 임계값 | 0.5 | 없음 |
| D | 임계값 | 0.9 | Prefix |
| E | 임계값 | 0.9 | Dual |

첫 블록은 원래 프롬프트의 마스크 P개와 생성 첫 32칸을 함께 채운다. 이후에는 생성 구간을 32칸씩 채운다.
임계값 모드는 채워지면 종료하며, 첫 블록의 완료 상한은 P+32회, 이후 블록은 각 32회다.
D/E는 첫 블록에서 캐시를 사용하지 않고, 블록 1부터 매 블록 시작 시 전체 입력으로 새 캐시를 만든다.

With는 `alpha_rule="progress"`, α0=0.3, 감사 임계값 0.1, 수리 비율 0.9, 수리 8회를 사용한다.
수리에는 τ·캐시를 적용하지 않는다. baseline hidden은 원래 요청 문장으로 한 번 계산한다.
조건 0은 `fixed_rule="fastdllm"`을 명시한다. 기본 `"diffuguard"`는 B의 GSM8K/RQ1 경로용이므로 혼용하지 않는다.

**본 결과의 With는 고정 공격 문장을 보호하고, 원래 프롬프트 마스크 위치와 생성 첫 32칸만 감사·수리한다.**
원본 DiffuGuard와 수리 범위가 다른 RQ2 변형이다. 사용자 지시에 따라 파일럿과 같은 설정을 본실험에 채택했으며,
팀 전체 합의나 원본 방어와의 동등성이 별도로 입증됐다는 뜻은 아니다.
채택 기록은 [본실험 프로토콜](configs/asr_with_full_protocol.json)에 있다.

C의 초기 공유 커밋은 `9d39fa2`였다. 후속 With 러너는 실험 당시 미커밋 상태였으므로, 정확한 실행 코드는
각 비공개 출력 디렉터리의 `source_snapshot`·`config_resolved.json`·소스 해시로 추적한다.
이번 정리 커밋을 과거 생성 실행 커밋과 동일하다고 표기하지 않는다.

## 최종 ASR 결과

각 칸은 **ASR-e(Luna) / ASR-k**, 단위는 %이며 낮을수록 좋다.

| 방어 | 0 | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| Without | 84 / 97 | 81 / 94 | 84 / 93 | 81 / 94 | 82 / 94 | 83 / 94 |
| With — RQ2 결합 설정 | 85 / 97 | 86 / 97 | 79 / 98 | 86 / 97 | 86 / 97 | 82.83* / 97 |

`*` With E의 Luna ASR-e는 99건 판정 기준(UNSAFE 82, SAFE 17, 미판정 1)이다.
ASR-k는 모든 조건에서 100문항 기준이다. 82.83%를 100문항의 확정 점수로 쓰지 않는다.
E의 같은 99개 ID로 비교하면 Without/With가 모두 82.83%다.

Mini 보조 판정은 다음과 같다. 각 값은 판정된 문항 기준 ASR-e다.

| 방어 | 0 | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| Without | 95 | 95 | 92 | 95 | 95 | 94 |
| With | 92 | 95.96* | 94 | 97 | 95.96* | 95.96* |

Mini의 `*`는 각 99건 판정(UNSAFE 95, SAFE 4, 미판정 1)이다.
ASR-e 정의는 `UNSAFE / (SAFE + UNSAFE)`이며 미판정 수를 별도로 기록한다.
두 판정 모델 결과를 섞거나 결과에 따라 유리한 모델만 선택하지 않는다.

- [Without/With 원래 판정 분모 비교](results/with_d321365_seed0_n100_asr_e.csv)
- [모든 조건에서 공통으로 판정된 99개 ID의 대응 비교 — 보조 분석](results/with_d321365_seed0_n100_asr_e_common_scored_ids.csv)
- [고정 100문항 기준 가능한 ASR-e 범위](results/with_d321365_seed0_n100_asr_e_bounds.csv)
- [미판정 ID 및 독립 검수 기록](results/with_d321365_seed0_n100_asr_e_verification.json)
- [Without 두 판정 모델 비교](results/without_0_A_B_C_D_E_judge_comparison.csv)

입력·생성 100문항을 99문항으로 바꾼 것은 아니다. 공통 99개 비교는 원본 기록을 보존한 보조 분석이다.
A/D/E의 With 채점 대상 응답은 100개 모두 동일했다. 독립 판정에서 생긴 점수 차이를 캐시가 만든 응답 차이로 해석하지 않는다.
현재 결과에서 일관된 방어 효과 유지를 주장하지 않으며, 원본과 다른 수리 범위·판정 모델 간 차이를 함께 논의해야 한다.

## 추론 시간과 NFE

100문항 합계이며 시간 단위는 초다. NFE는 실제 모델 forward 수다.

| 조건 | Without 시간 | With 시간 | Without 총 NFE | With 총 NFE |
|---|---:|---:|---:|---:|
| 0 | 2072.9 | 1880.8 | 12,800 | 13,700 |
| A | 1584.1 | 1479.9 | 9,536 | 10,561 |
| B | 1264.7 | 1209.3 | 7,651 | 8,641 |
| C | 927.4 | 964.4 | 5,754 | 6,949 |
| D | 1495.6 | 1364.7 | 9,503 | 10,659 |
| E | 1484.1 | 1367.7 | 9,577 | 10,542 |

With의 `wall_ms`는 baseline 준비/forward, 생성, 감사, 수리를 포함하고 모델 로드·생성 입력 토큰화·응답 디코딩·파일 I/O·API 채점을 제외한다.
총 NFE는 생성 + baseline + 감사 + 수리다. 감사는 생성 중 logits/hidden을 재사용하므로 추가 forward는 0이다.
모든 With 조건에서 100문항 모두 수리 8회가 발생했다. D/E의 계측은 작은 토큰/마스크 snapshot을 추가하지만 모델 호출은 추가하지 않는다.
Without/With 시간은 서로 다른 실행 세션에서 측정했으므로 작은 차이를 방어가 만든 속도 향상으로 단정하지 않는다.
가속비는 같은 방어 행의 0을 기준으로 계산하고, 실측 wall time과 NFE를 함께 제시한다.

빈 응답률은 전 조건 0%였다. 의미적 무응답률(non-answer rate)은 별도 기준으로 판정하지 않았으므로 0%로 기재하지 않는다.

## 코드 구성

| 파일 | 역할 |
|---|---|
| `run_asr.py`, `decoding.py`, `prefix.py`, `dual.py` | Without 0/A~E |
| `data.py`, `data/ids/`, `upstream.py` | 고정 입력·시드 기준, 모델 로더 |
| `run_asr_with0_pilot.py` | 이름은 초기 파일럿 때 그대로이며 현재 With 0/A~E, `--limit 10/100` 지원 |
| `cache_probe.py` | 실제 호출 수, 수리 후 캐시 생성, Prefix/Dual 입력·KV·교체 범위 검수 |
| `run_asr_with_full.py` | 0 → A → D → E → B → C 자동 순차 실행·검수·판정 |
| `monitor_with_full.py` | 생성 600개와 모델별 판정 진행 표시 |
| `evaluate_asr.py`, `run_asr_e_full.py` | Without ASR-k·ASR-e |
| `run_asr_e_with_pilots.py` | With 파일럿의 두 모델 판정 |
| `judge_with_full.py` | With 본실험의 두 모델 판정 |
| `resume_with_full_judging.py` | 알려진 차단 입력을 재전송하지 않고 허용된 나머지 판정만 재개 |

[interfaces.md](interfaces.md)는 기존 Without 계약과 캐시 정책의 상세 기록이다.
With는 B의 `(tokens, Stats)`를 직접 호출하고 C의 입력·응답 추출·출력 스키마에 연결한다.
기존 `DecodeConfig`/`DecodeResult`가 B의 공통 타입으로 완전히 통합됐다고 가정하지 않는다.

## 새 Workspace에서 코드 준비

명령은 레포 루트(`/root/DLM_safety_code`) 기준이다. 실제 키·가중치·비공개 실험 출력은 Git으로 전달되지 않는다.
기존 Workspace에서는 준비된 디렉터리를 다시 만들거나 완료된 실험을 재실행할 필요가 없다.

```bash
git submodule update --init --recursive
# /root/Fast-dLLM이 없을 때만 clone한다.
git clone https://github.com/NVlabs/Fast-dLLM.git /root/Fast-dLLM
git -C /root/Fast-dLLM checkout a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b

git fetch origin rq2-combined-decoder
# 같은 worktree가 이미 있으면 아래 add는 생략한다.
git worktree add --detach /root/rq2-b-d321365 d321365a1f2dad28fdda36e1b540b04c0a56b624

python -m venv --system-site-packages /root/rq2-venv
/root/rq2-venv/bin/pip install -r RQ2/requirements.txt -r RQ2/requirements-judge.txt
HF_HUB_DISABLE_XET=1 /root/rq2-venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("GSAI-ML/LLaDA-8B-Instruct", revision="08b83a6feb34df1a6011b80c3c00c7563e963b07", local_dir="hf_models/LLaDA-8B-Instruct", allow_patterns=["*.json","*.py","*.safetensors","*.model","*.txt"])'
```

`run_asr_with_full.py`의 결합 디코더 기본 경로는 `/root/rq2-b-d321365/RQ2/src/combined.py`다.
다른 경로를 사용하려면 해당 설정을 확인하고, 새 실행의 소스 해시·환경을 기록한다.

### Without 재현 예시

완료된 원래 출력 대신 새 경로를 지정한다.

```bash
/root/rq2-venv/bin/python -m RQ2.run_asr \
  --accel A --limit 100 --verify-original \
  --output-dir RQ2/outputs/reproduce/without_A_n100
/root/rq2-venv/bin/python -m RQ2.evaluate_asr RQ2/outputs/reproduce/without_A_n100
```

### With 파일럿과 본실험 재현

새 Workspace의 자동 본실험 사전 점검은 로컬 파일럿 6조건의 검수 기록을 요구한다.
Git에 있는 숫자 요약만으로 비공개 파일럿 출력이 복원되지는 않는다.

```bash
for condition in 0 A D E B C; do
  /root/rq2-venv/bin/python -m RQ2.run_asr_with0_pilot \
    --accel "$condition" --limit 10 \
    --combined-path /root/rq2-b-d321365/RQ2/src/combined.py \
    --output-dir "RQ2/outputs/pilot/with_${condition}_seed0_n10_d321365" || break
done
```

현재 완료된 Workspace에서는 위 파일럿을 반복하지 않는다. 재현 시 같은 경로가 이미 있으면 출력 덮어쓰기를 거부한다.
새 경로의 생성-only 본실험은 다음과 같다. 생성 전용 명령은 API 요청을 보내지 않는다.

```bash
mkdir -p RQ2/outputs/logs
nohup /root/rq2-venv/bin/python -u -m RQ2.run_asr_with_full \
  --resume --generation-only \
  --output-root RQ2/outputs/full/with_d321365_seed0_n100_reproduce \
  >> RQ2/outputs/logs/with_full_reproduce.log 2>&1 < /dev/null &
```

순서는 0 → A → D → E → B → C이며 각 100문항이다. 실패하면 뒤 조건을 실행하지 않는다.
동일 설정의 `--resume`은 검수된 완료 조건을 건너뛰며, 중간 실패 조건은 기존 출력을 `failed_attempts/`로 보존한 뒤 그 조건 100문항을 새로 실행한다.
문항 단위 이어받기는 아니다. 재개 시 소스·환경·설정이 달라졌으면 혼합 실행을 거부한다.
SSH 종료와 별개로 VESSL Workspace 실행 상태는 유지해야 한다.

기본 실행기는 생성 후 두 모델 판정을 연결할 수 있으나, API 모드 사전 점검에는 로컬의 기존 Without 원문·판정과 자격 증명도 필요하다.
현재 알려진 차단 입력을 재전송하려고 새 경로·모델·문구 변경을 사용하지 않는다.

```bash
/root/rq2-venv/bin/python -m RQ2.monitor_with_full --once
# 재현 경로를 모니터링할 때:
/root/rq2-venv/bin/python -m RQ2.monitor_with_full \
  --output-root RQ2/outputs/full/with_d321365_seed0_n100_reproduce --once
```

## 판정 설정과 미판정 복구

공통 설정은 `judge_prompt=code`, `question=auto`, 최대 800토큰, workers=2다.
Luna는 `gpt-6-luna`, `reasoning_effort=none`이고, Mini는 `gpt-4o-mini-2024-07-18`이며 reasoning 인자를 보내지 않는다.
판정 temperature·seed는 원본 평가 코드와 같이 명시하지 않았으므로 결정적 판정이라고 가정하지 않는다.
설정 파일은 [Luna](configs/asr_e_judge.json), [Mini](configs/asr_e_judge_gpt4omini.json)다.

`OPENAI_API_KEY`와 필요한 경우 `OPENAI_BASE_URL`은 로컬 `.env`에만 설정한다. 키를 명령행·문서·Git에 넣지 않는다.
기존 `.env`를 예제 파일로 덮어쓰지 않는다.

본실험의 원래 배치는 Luna E의 내용 차단에서 중단됐다. 이후 원문과 완료된 Luna 599건을 보존하고,
알려진 동일 입력을 건너뛴 Mini 597건을 채점했다. 복구 과정은 원본 판정 프롬프트·모델·파싱을 유지했고,
전송/함수 재시도는 비활성화했다. 차단된 건을 다른 판정 모델의 라벨로 채우지 않았다.

로컬 비공개 기록이 있는 동일 실행의 허용된 판정만 재개할 때 사용하는 명령은 다음과 같다.
현재 실행에서는 이미 판정 가능한 건을 모두 완료했으므로 다시 실행할 필요가 없다.

```bash
/root/rq2-venv/bin/python -m RQ2.resume_with_full_judging \
  RQ2/outputs/full/with_d321365_seed0_n100
```

`api_policy_blocks.json`이 있는 실행에 일반 재시작·일반 판정을 적용하면 거부한다.
차단 문항을 SAFE/UNSAFE로 추정하거나, 내용을 바꿔 다시 보내거나, 미판정을 0점으로 처리하지 않는다.
공통 판정 ID의 보조 비교와 100문항 점수 범위를 함께 보존한다. 의미적 무응답과 API 미판정은 서로 다른 지표다.

## 검증 범위

- With 0/A 방어 OFF는 실제 C 코드와 고정 10문항 전체 토큰·응답·NFE가 일치했다.
- With B/C/D/E 사전 대조는 각 3문항(원본 인덱스 0/3/9)이다. D/E는 실제 KV 재사용 문항을 포함했다.
- With 6조건 파일럿 및 본실험 모두 ID·시드·응답 해시·NFE 합계·잔여 마스크·고정 프롬프트 유지 검수를 수행했다.
- D/E는 수리 후 토큰으로 캐시 생성, 캐시 입력·KV 길이, Dual 32칸 교체 범위를 모델 hook으로 검수했다.
- B가 공유한 GSM8K RQ1 With 동일 환경 재현 통과 결과는 B의 검증이다. 이를 C가 별도로 실행했다고 기재하지 않는다.

GPU/API를 실행하지 않는 회귀 테스트 예시:

```bash
OMP_NUM_THREADS=1 /root/rq2-venv/bin/python -m pytest -q \
  RQ2/tests/test_with_full_batch.py RQ2/tests/test_cache_probe.py
```

이 테스트들은 순차 실행, 실패 시 중단, 완료 조건 건너뛰기, 실패 출력 보존, 100문항 검증,
수리 후 캐시, 잘못된 KV/교체 범위 거부, 반복 상한을 다룬다. GPU 원본 동등성 검증을 대신하지 않는다.

## GitHub 공유 범위와 결과 파일 읽기

- 공유: 실행·평가 코드, 고정 ID·설정, README, 테스트, 숫자/ID/해시만 담긴 `results/` 요약.
- 제외: `outputs/`의 원문·reasoning·토큰 ID, `.env`·`.env.*`, 로그, 모델 가중치·캐시.
- `handoff/combined_d321365_preflight*.json`은 현재 디코더 대조 기록이다.
  이전 `combined575_asr_compat.patch`는 채택 전 수정 제안이므로 현재 d321365에 적용하지 않는다.

`results/`의 이전 `*_full.json` 중 `asr_e=not_run`이 있는 것은 생성 당시의 보존된 보고서다.
최종 판정 완료 여부는 위 요약 CSV와 `*_asr_e*.csv`를 기준으로 읽는다.
파일럿 결과는 이름에 `pilot`이 붙으며 100문항 본실험과 혼합하지 않는다.
원문과 과거 판정의 보존 여부는 비공개 기록의 해시 및 [최종 독립 검수 요약](results/with_d321365_seed0_n100_asr_e_verification.json)으로 확인했다.

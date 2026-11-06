# RQ1: 디퓨전 스텝 수를 줄이면 DiffuGuard가 버티는가

> **RQ1.** 디퓨전 스텝 수를 줄이면(가속하면) DiffuGuard를 켠 모델과 끈 모델의 ASR은 각각 어떻게 변하고, 성능(GSM8K)은 얼마나 떨어지는가?

- 설계 전체: 노션 "RQ1 실험 개요"
- 할 일과 담당: 노션 "RQ1 작업 카드"
- 파일 형식 약속: [`docs/interfaces.md`](docs/interfaces.md) ← **코드를 쓰기 전에 먼저 읽기**

이 폴더는 **뼈대만 있는 상태**입니다. 각 파일 맨 위에 담당 카드 ID가 적혀 있고, 아직 구현하지 않은 함수는 `NotImplementedError("T3-4 …")`를 냅니다.

## 실험 조건 요약

| 항목 | 값 |
|---|---|
| 모델 | LLaDA-8B-Instruct, 리비전 `08b83a6f…` (DIJA·DiffuGuard 재현과 같은 가중치, `../hf_models/` 공용) |
| 방어 | DiffuGuard 켬 / 끔. 하이퍼파라미터는 `DiffuGuard/` 재현 config와 같음 (α0 0.3, λ 0.1, γ 0.9, extra_steps 8, cfg_scale 0) |
| 속도 단계 L | 1 / 2 / 4 / 8 → 안전 세트 스텝 128 / 64 / 32 / 16, GSM8K 스텝 256 / 128 / 64 / 32 |
| 세트 | DIJA 100(시드 3개), JBB 원문 100, WildJailbreak 100, XSTest 250, GSM8K 250 |
| 실행 수 | 56개 (`run_id` 하나 = 세트 × 속도 단계 × 방어 × 시드) |

## 폴더 구조

```
RQ1/
├── README.md
├── docs/interfaces.md        # 팀 사이 파일 형식 계약서 (변경은 팀장 승인)
├── configs/
│   ├── base.yaml             # 공통값                         T3-3
│   ├── grid.yaml             # 실험 축 → manifest.csv          T3-3, T3-5
│   └── sets/*.yaml           # 세트별 덮어쓰기                 T3-3
├── data/ids/                 # 세트별 문항 ID 목록 (형식: data/ids/README.md)  T3-2
├── examples/                 # 가짜 예시 파일. 4·6번이 개발용으로 씀
├── run.py                    # 실행 진입점                     T3-4
├── src/
│   ├── config.py             # 설정 합치기                     T3-3
│   ├── data.py               # 세트 로더                       T3-2
│   ├── decoding.py           # 속도 단계, NFE·시간 기록        T1-1, T1-2
│   ├── defense.py            # 진행률 스케줄, 방어 끔 경로     T2-2, T2-3
│   └── trace.py              # trace 기록기                   T1-3
├── scripts/
│   ├── make_manifest.py      # grid.yaml → manifest.csv        T3-5
│   └── sync_manifest.py      # status.json → manifest.csv      T3-5
├── judge/
│   ├── judge.py              # 키워드·GPT-4o-mini·XSTest 판정  T4-1
│   ├── merge.py              # → items.parquet                T4-2
│   └── prompts/              # 판정 프롬프트 (버전 붙여 커밋)
├── audit/sample.py           # 사람 검수 표본 추출             T5-1, T5-2
├── analysis/
│   ├── summarize.py          # → rq1_summary.csv              T6-2
│   └── plots.py              # 그림 3장                        T6-1
├── tests/                    # CPU pytest
└── outputs/                  # (git 제외) 로컬 테스트용. 본실행 출력은 공용 볼륨
```

## GPU 환경 (VESSL Workspace)

루트 README의 "GPU 환경 (VESSL)"을 그대로 따릅니다. RQ1에서 추가로 지킬 것은 아래와 같습니다.

- **Workspace 1개 = RTX 3090 1장 = 동시에 실행 1개.** LLaDA-8B bf16이 약 16GB라 한 장에 두 개가 들어가지 않습니다. 더 빨리 끝내려면 Workspace를 하나 더 띄웁니다.
- **본실행 출력은 팀 공용 볼륨 `rq1-outputs`에 씁니다.** Workspace를 만들 때 볼륨을 마운트하고, 그 경로를 환경변수로 지정합니다.
  ```bash
  export RQ1_OUTPUT_DIR=/root/rq1-outputs   # 마운트한 경로. 지정하지 않으면 RQ1/outputs/ (로컬 테스트용)
  ```
  그래야 4번(판정)과 5번(검수)이 자기 Workspace에서 모든 실행 결과를 읽을 수 있습니다. 원문은 이 볼륨 밖으로 내보내지 않습니다.
- Workspace를 Stop하면 실행도 끊깁니다. `run.py`는 끊긴 지점(`n_done`)부터 이어서 돌립니다.

## 실행 (설정팀이 T3-6을 마친 뒤)

```bash
cd /root/DLM_safety_code && source /root/venv/bin/activate && git pull
cd RQ1
mkdir -p outputs/logs
nohup python run.py --run_id rq1-dija-L2-dg-s0 > outputs/logs/rq1-dija-L2-dg-s0.log 2>&1 &
python scripts/sync_manifest.py      # 진행 상황을 manifest.csv에 반영 → 커밋
```

로그와 콘솔에는 원문을 찍지 않습니다(문항 수, 길이, 해시만).

## 테스트 (CPU)

```bash
cd RQ1 && python -m pytest tests -q
```

지금은 예시 파일이 `docs/interfaces.md` 형식에 맞는지만 검사합니다. 각자 모듈을 구현하면서 테스트를 추가합니다.

## 규칙

- 브랜치: 각자 브랜치 → PR → `main`. **`docs/interfaces.md`를 바꾸는 PR만 팀장 승인.**
- 실행팀은 생성 코드(`run.py`, `src/`)를 고치지 않습니다. 버그는 설정팀에 넘깁니다.
- 새 공격 프롬프트를 만들지 않습니다. 공개 세트만 씁니다 (`refine_prompt/run_refine.sh` 실행 금지).
- 원본 DiffuGuard 코드(`DiffuGuard/third_party/DiffuGuard`)는 고치지 않습니다. 진행률 스케줄처럼 바꿔야 하는 부분은 `src/`에 새로 구현하고, 원본과 다른 점을 파일 맨 위에 적습니다.

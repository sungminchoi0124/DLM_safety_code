# DLM_safety_code

디퓨전 언어모델(dLLM) 안전성 연구 코드.
연구 질문: **추론 가속(스텝 수 감소·병렬 디코딩·KV 캐시)이 안전 방어를 약하게 만드는가?**
첫 단계로 논문 3편의 공개 코드를 재현한다. 논문마다 최상단 폴더 하나를 쓴다.

| 폴더 | 논문 | 담당 | 상태 |
|---|---|---|---|
| [`DIJA/`](DIJA/README.md) | DIJA (ICLR 2026, arXiv 2507.11097) — JailbreakBench × LLaDA-8B-Instruct | | 실행 준비 완료, GPU 실행 중 |
| `<논문2>/` | | | |
| `<논문3>/` | | | |

## 구조 규칙

```
DLM_safety_code/
├── README.md            # 이 파일 (논문 목록)
├── .env.example         # API 키 템플릿. 복사한 .env는 세 논문이 같이 씀 (git 제외)
├── .gitignore           # 모든 폴더의 outputs/, .env, hf_models/ 제외
├── hf_models/           # (git 제외) 모델 가중치. 논문 간 공용
└── <논문>/              # 논문 하나 = 폴더 하나, 안에서 완결
    ├── third_party/<원본>/   # 원본 공개 코드 git submodule, 커밋 고정, 수정 금지
    ├── README.md             # 실행 순서, 목표 수치, 결과 표, 미확인 목록
    ├── configs/              # 경로·시드·리비전
    ├── env/                  # Dockerfile, CPU 테스트용 requirements
    ├── tests/                # CPU에서 도는 pytest
    └── outputs/              # (git 제외) 생성 결과. 유해 텍스트 포함 가능
```

- 명령은 각 논문 폴더 안에서 실행한다 (`cd DIJA`). 경로는 그 폴더 기준이다.
- 원본 코드는 고치지 않는다. 꼭 바꿔야 하면 `<논문>/patches/`에 diff와 이유를 남긴다.
- 유해 출력은 `outputs/`에만 저장한다. git·메신저·OneDrive 등에 올리지 않는다.
- API 키는 `.env`에만 둔다. 코드나 명령 기록에 남기지 않는다.

## 새 논문 추가하기

```bash
mkdir -p <논문>/third_party <논문>/outputs && touch <논문>/outputs/.gitkeep
git submodule add <원본 저장소 URL> <논문>/third_party/<원본>
git -C <논문>/third_party/<원본> checkout <고정할 커밋>
git add .gitmodules <논문> && git commit -m "Add <논문> reproduction scaffold"
```

## GPU (VESSL)

- 논문(사람)마다 Workspace를 따로 만든다. GPU 1장 = Workspace 1개.
- 클론 위치는 `/root/DLM_safety_code`, 모델은 `/root/DLM_safety_code/hf_models/`에 둔다 (`/root`만 재시작 후 유지).
- 다 쓰면 Workspace를 Stop한다 (켜져 있는 동안 과금).

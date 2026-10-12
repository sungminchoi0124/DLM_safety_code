# DLM_safety_code

디퓨전 언어모델(dLLM) 안전성 연구 코드.
연구 질문: **추론 가속(스텝 수 감소·병렬 디코딩·KV 캐시)이 안전 방어를 약하게 만드는가?**
첫 단계로 논문 3편의 공개 코드를 재현한다. 논문마다 최상단 폴더 하나를 쓴다.

| 폴더 | 논문 | 담당 | 상태 |
|---|---|---|---|
| [`DIJA/`](DIJA/README.md) | DIJA (ICLR 2026, arXiv 2507.11097) — JailbreakBench × LLaDA-8B-Instruct | | 5문항 확인 완료 (RTX 3090), 100문항 실행 대기 |
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

## GPU 환경 (VESSL) — DIJA 셋업 기준, 팀원 참고용

### Workspace 설정
| 항목 | 값 | 비고 |
|---|---|---|
| 리소스 | `gpu-1` 계열 | 클러스터의 GPU는 전부 **RTX 3090 (24GB)**. 1장이면 8B 모델(bf16 약 16GB) 충분. `gpu-2` 이상은 원본 코드가 1장만 써서 낭비 |
| 이미지 | 관리형 `PyTorch 2.3.1 (CUDA 12.1)` | requirements 설치 시 torch 2.6.0+cu124로 바뀜. 드라이버 550.107.02라 문제없음 |
| 디스크 | **50~60 GiB** | 기본 15GiB는 부족 (모델 16GB + 패키지 약 10GB). `/root`에 붙음 |
| 포트 | 8888 jupyter, 22 ssh | 기본값 그대로 |
| 노드 | 아무거나 하나 | 논문(사람)마다 Workspace 따로. 한 재현 안에서는 가능하면 같은 노드 |

확인된 사양: Python 3.10.12, RAM 503GB, CUDA 12.4 드라이버.

### 접속
Workspace가 Running이 되면 **Jupyter → Terminal**에서 명령을 실행한다 (가장 쉬움). SSH를 쓰려면 VESSL 계정에 공개키를 등록한다.

### 처음 한 번
```bash
cd /root
git clone --recurse-submodules https://github.com/sungminchoi0124/DLM_safety_code
cd DLM_safety_code
python -m venv /root/venv && source /root/venv/bin/activate   # 패키지도 /root 안에 둬야 재시작 후 남음
pip install --upgrade pip
# 이후 의존성 설치·모델 다운로드는 논문 폴더 README 참고 (예: DIJA/README.md)
cp .env.example .env   # API 키 입력 (git 제외)
```
- 모델은 `/root/DLM_safety_code/hf_models/`에 받는다 (논문 간 공용, git 제외). 16GB급은 `nohup ... &`로 백그라운드 다운로드.
- `pip install`은 `--no-cache-dir`을 붙여 디스크를 아낀다.

### 일시정지 / 재개
- **Stop 전:** `jobs`, `nvidia-smi`로 돌고 있는 작업이 없는지 확인 (Stop하면 프로세스는 끊김).
- **Stop:** VESSL 웹 → Workspace → Stop. `/root` 아래(코드, venv, 모델, outputs)는 남고, `/root` 밖에 설치한 것은 사라짐.
- **재개:** Start → Terminal에서
  ```bash
  cd /root/DLM_safety_code && source /root/venv/bin/activate && git pull
  ```
- 켜져 있는 동안 과금되므로 다 쓰면 Stop.

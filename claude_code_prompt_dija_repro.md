# Claude Code 작업 지시: DIJA 재현 준비 (GPU 없이 "실행 직전"까지)

## 0. 배경 (너는 이 프로젝트를 처음 본다고 가정)

학부 연구팀이 디퓨전 언어모델(dLLM)의 안전성을 연구한다.
연구 질문: **추론 가속(스텝 수 감소·병렬 디코딩·KV 캐시)이 안전 방어를 약하게 만드는가?**
그 첫 단계로, 공격 논문 **DIJA**(ICLR 2026, arXiv 2507.11097)의 공개 코드로 원 논문 수치를 재현한다.

- 공개 코드: https://github.com/ZichenWen1/DIJA (고정 커밋 `56ba341b3377423e6da212013aec6e825710a64c`)
- 팀 저장소: https://github.com/sungminchoi0124/DLM_safety_code (현재 작업 폴더)
- 재현 범위: **JailbreakBench 100문항 × LLaDA-8B-Instruct × DIJA × 방어 없음**. 이것 하나만.
- **지금은 GPU가 없다.** GPU를 받는 순간 명령어 한 줄로 돌릴 수 있는 상태까지만 만드는 것이 목표다.

### 재현 목표 수치 (DIJA 논문 Table 2, JailbreakBench, LLaDA-Instruct)

| 행 | ASR-k | ASR-e | HS |
|---|---|---|---|
| Zeroshot (공격 없음, 정상 확인용) | 0.0 | 0.0 | 1.0 |
| DIJA | 95.0 | 81.0 | 4.6 |
| DIJA* | 99.0 | 81.0 | 4.8 |

- ASR-k: 거절 문구 키워드가 없으면 성공. API·GPU 불필요
- ASR-e: JailbreakBench 판정기(Together API의 Llama-3-70B)
- HS: GPT-4o가 매긴 1~5점 유해도
- 통과 기준: ±5%p. 저장소의 `_Qwen` 프롬프트 파일이 DIJA와 DIJA* 중 어느 행에 해당하는지는 논문에서 확인할 것. 모르면 모른다고 보고서에 적어라.

## 1. 내가 미리 코드를 읽고 찾은 사실 (반드시 확인하고 반영할 것)

1. **`steps`, `gen_length`, `block_length` 인자가 실제로는 쓰이지 않는다.**
   - `run_jailbreakbench/utility/generate_function.py`의 `generate_llada`는 `x = input_ids`로 시작한다. 그래서 프롬프트 안에 박힌 마스크만 채운다.
   - `num_transfer_tokens = 1`이라서 forward 한 번에 한 토큰씩, 마스크가 없어질 때까지 반복한다.
   - 100개 프롬프트의 마스크 수: 최소 45, 평균 91.6, 최대 200. 따라서 문항당 forward 횟수 = 그 문항의 마스크 수다.
   - 재현에서는 이 동작을 **절대 바꾸지 말 것.** 나중에 스텝 수 실험을 위해 `--tokens_per_step` 옵션을 붙일 자리만 TODO 주석으로 표시하라. 기본값 1이면 원본과 완전히 같아야 한다.
2. **시드가 고정되어 있지 않다.** `temperature=0.2`이고 Gumbel noise를 쓴다. 래퍼에서 `torch`, `numpy`, `random` 시드를 고정하는 `--seed` 옵션(기본 0)을 추가하라. 원본 파일을 고치지 말고 래퍼에서 처리하라.
3. **경로가 `$HOME/DIJA`로 하드코딩되어 있다.** `eval_jailbreakbench.sh`, `run_refine.sh` 등이 그렇다.
4. **GPU 자동 선택이 위험하다.** `nvidia-smi` 결과에서 사용 메모리가 0인 GPU만 고른다. 클라우드(VESSL)에서는 값이 비어 조용히 CPU로 돌 수 있다. GPU가 없으면 **에러로 멈추게** 하라.
5. **응답 원문을 콘솔에 출력한다.** `logging.info(f"Response: ...")`가 그렇다. 유해 텍스트가 로그에 남지 않게 래퍼에서 막아라. 길이와 해시만 출력.
6. **`refine_prompt/run_refine.sh`는 실행하지 말 것.**
   - 이 스크립트는 LLM으로 새 공격 프롬프트를 만든다. 팀 규칙상 금지다.
   - `version=Qwen`으로 돌리면 공개된 `jailbreakbench_data_refined_Qwen.json`을 **덮어쓴다.**
   - 공개 파일만 쓰고, 그 파일의 SHA-256을 기록해 둬라.
7. **ASR-e 판정기 모델이 아직 있는지 모른다.** `benchmarks/jailbreakbench/src/jailbreakbench/classifier.py`의 `Llama3JailbreakJudge`는 litellm으로 `together_ai/meta-llama/Llama-3-70b-chat-hf`를 부른다. 이 모델이 Together에서 아직 서비스되는지 확인하는 방법을 문서에 적어라. 너는 API 키가 없으니 실제 호출은 하지 마라.

## 2. 할 일

### 2-1. 저장소 구성
```
DLM_safety_code/
├── third_party/DIJA/          # git submodule, 위 커밋에 고정. 원본 수정 금지
├── repro/dija/
│   ├── run_generate.py        # 원본 jailbreakbench_llada.py를 import해서 감싸는 래퍼
│   ├── run_eval.sh            # ASR-k → ASR-e → HS 순서, 각 단계 따로 실행 가능
│   ├── configs/llada_jbb.yaml # 모델 경로·리비전, 시드, 프롬프트 파일, 출력 폴더
│   └── README.md              # GPU 받은 날 그대로 따라 할 실행 순서
├── env/
│   ├── Dockerfile             # CUDA 12.4 + Python 3.10 + 원본 requirements.txt
│   └── requirements-cpu.txt   # 로컬 CPU 테스트용 최소 의존성
├── tests/                     # CPU에서 도는 테스트
├── outputs/                   # .gitignore 처리. 생성 결과는 여기만
├── .env.example               # TOGETHER_API_KEY=, OPENAI_API_KEY=, HF_TOKEN= (빈 값)
└── .gitignore                 # outputs/, *.jsonl, .env, hf_models/, __pycache__/
```
- 원본을 고쳐야만 하는 곳이 생기면 `repro/dija/patches/`에 diff로 남기고, 이유를 README에 적어라.

### 2-2. 래퍼 `run_generate.py`
- 모든 경로는 config나 인자로 받는다.
- `--model_path`는 로컬 경로와 HF 이름(`GSAI-ML/LLaDA-8B-Instruct`)을 모두 받는다. 모델 리비전(커밋 해시)도 기록한다.
- `--attack_method {DIJA,zeroshot}`, `--seed`, `--limit N` (처음 N문항만), `--dry_run` (모델 없이 입력 처리만 하고 문항별 마스크 수와 예상 forward 횟수를 출력)을 받는다.
- 출력 JSON의 필드 이름은 원본과 같게 한다. 원본 평가 스크립트가 그대로 읽을 수 있어야 한다.
- 실행 메타데이터를 별도 파일로 저장한다: 커밋, 모델 리비전, 시드, 문항별 forward 횟수와 소요 시간, GPU 이름.

### 2-3. CPU 테스트 (`pytest`로 전부 통과해야 함)
1. **가짜 모델로 끝까지 실행**
   - `generate_llada`의 인터페이스를 흉내 내는 작은 가짜 모델을 만든다. `model(x, attention_mask=...)` 호출에 `.logits`가 있는 출력을 돌려주면 된다.
   - 원본 `generate_llada` 함수를 그대로 써서 다음을 확인한다: 마스크 수만큼 반복하는가, 마스크가 다 사라지는가, 같은 시드면 같은 결과가 나오는가.
2. **`--dry_run`으로 100문항 입력 처리**
   - `process_prompt_instruct`와 chat template 적용이 에러 없이 끝나는지 확인한다.
   - 토크나이저는 `GSAI-ML/LLaDA-8B-Instruct`에서 **토크나이저 파일만** 받는다(가중치 받지 말 것).
   - 마스크 토큰 ID가 126336인지 확인한다.
3. **평가 스크립트 확인**
   - 무해한 가짜 응답("I'm sorry, I cannot help with that." / "Here is a recipe for bread.")으로 ASR-k 스크립트가 0과 1을 맞게 내는지 본다.
   - ASR-e와 HS는 API를 부르지 않는 선에서 import와 인자 파싱까지만 확인한다.
4. **안전 확인**
   - 래퍼가 콘솔에 응답 원문을 찍지 않는지 확인한다.
   - GPU가 없을 때 `--dry_run`이 아니면 에러로 멈추는지 확인한다.

### 2-4. `repro/dija/README.md` (GPU 받은 날 쓰는 문서)
- 사전 준비: 모델 다운로드 명령(약 16GB), API 키 설정, Together 판정기 모델 확인
- 실행 순서
  - ① zeroshot 5문항
  - ② DIJA 5문항
  - ③ 100문항 전체
  - ④ ASR-k
  - ⑤ ASR-e
  - ⑥ HS
- 단계마다 "정상이면 이렇게 보인다"를 적는다. 예: zeroshot은 거의 전부 거절
- 예상 소요 시간
  - 문항당 forward 수 × forward 1회 시간으로 계산한다.
  - forward 1회 시간은 A100 기준 추정치임을 밝히고, ①에서 실측해 고치라고 적는다.
- 결과 기록 표: 목표 수치, 우리 수치, 차이, 원인 추정

## 3. 규칙
- **새 공격 프롬프트를 만들지 않는다.** 공개된 `_Qwen` JSON만 쓴다.
- 유해 출력은 `outputs/`에만 저장한다. git에 올리지 않는다. 콘솔·로그·테스트 픽스처에 실제 유해 텍스트를 넣지 않는다.
- 모델 가중치(수 GB)를 받지 않는다. 토크나이저 파일만 허용.
- API 키를 코드에 쓰지 않는다. `.env`에서 읽고, `.env`는 git에서 제외.
- 원본 `third_party/DIJA`의 동작을 바꾸지 않는다. 바꿔야 하면 patch 파일과 이유를 남긴다.
- 확인하지 못한 것은 추측으로 채우지 말고 README의 "미확인" 목록에 적는다.

## 4. 끝났을 때 보고할 것
1. 만든 파일 목록과 각 파일의 역할 한 줄씩
2. `pytest` 결과
3. `--dry_run` 결과 요약: 문항 수, 마스크 수 분포, 총 forward 횟수
4. "미확인" 목록. 예: Together 판정기 모델 서비스 여부, `_Qwen`이 DIJA/DIJA* 중 어느 쪽인지
5. GPU를 받으면 실행할 명령어 (복사해서 바로 쓸 수 있게)

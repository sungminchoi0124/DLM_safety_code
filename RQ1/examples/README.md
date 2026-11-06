# 예시 파일 (전부 가짜 값)

`docs/interfaces.md` 형식을 따르는 개발용 예시입니다. 실제 모델 출력이 아닙니다.

| 파일 | 형식 | 쓰는 사람 |
|---|---|---|
| `raw_sample.jsonl` | interfaces 5절 | 4번 (판정 코드 개발) |
| `judge_sample.jsonl` | interfaces 6절 | 4번 (병합), 6번 |
| `rq1_summary_sample.csv` | interfaces 10절 | 6번 (그림 코드 개발) |

형식을 바꾸면 `tests/test_examples.py`가 실패합니다. 계약서와 예시 파일을 같이 고칩니다.

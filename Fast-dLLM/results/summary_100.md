# GSM8K 100문항 공통 비교

모든 조건은 LLaDA-8B-Instruct, GSM8K 5-shot, batch size 1,
generation length 256, block length 32 조건이다. 팀 공통 비교에는
`flexible-extract` 정확도를 사용한다.

| 조건 | Flexible | Strict | tok/s | Baseline 대비 | 총 NFE | 평균 NFE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 82% | 37% | 2.248 | 1.00x | 25,600 | 256.00 |
| A: Prefix Cache | 81% | 41% | 10.722 | 4.77x | 25,600 | 256.00 |
| B: Parallel | 82% | 39% | 7.142 | 3.18x | 7,973 | 79.73 |
| C: Cache + Parallel | 81% | 42% | 24.64 | 10.96x | 8,228 | 82.28 |

C는 23,253개 토큰 생성에 943.76초가 걸렸다. 네 조건의 문항,
프롬프트, 정답 해시가 일치함을 확인했다. 다만 각 조건은 서로 다른 실행에서
측정했으므로 GPU 부하와 클럭을 완전히 통제한 반복 측정 결과는 아니다.

한 문항이 1%p에 해당하므로 이 표만으로 정확도 보존을 확정하지 않는다.

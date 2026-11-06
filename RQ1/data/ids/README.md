# 문항 ID 목록 (T3-2, 3번)

세트마다 파일 하나: `dija_100.txt`, `jbb_100.txt`, `wildjailbreak_100.txt`, `xstest_250.txt`, `gsm8k_250.txt`.

한 줄에 한 문항, 탭으로 구분, 첫 줄은 헤더:

```
item_id	source_key
dija-000	0
dija-001	1
```

- `item_id` = `{set}-{번호 3자리}` (이 파일의 순서, 000부터)
- `source_key` = 원본 데이터에서 그 문항을 찾는 키 (JBB 행동 번호, WildJailbreak 행 번호, XSTest `id`, GSM8K 테스트셋 인덱스)
- 문항 텍스트는 넣지 않습니다. ID 목록만 git에 올립니다.

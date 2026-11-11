"""Fixed GSM8K question list for RQ1: 250 of the 1,319 test questions, sampled with seed 0.

    python RQ1/scripts/make_gsm8k_ids.py            -> RQ1/data/ids/gsm8k_250.txt

Format (RQ1/data/ids/README.md): header line, then "item_id<TAB>source_key" per line, where
source_key = GSM8K test index = doc_id in DiffuGuard/data/gsm8k_5shot_lmeval.json.
Read by: python DiffuGuard/run_gsm8k.py --doc_ids ../RQ1/data/ids/gsm8k_250.txt
Deterministic: the same file (same sha256, printed at the end) on every machine.
"""
import hashlib
import random
from pathlib import Path

N_TEST, N_PICK, SEED = 1319, 250, 0
OUT = Path(__file__).resolve().parents[1] / "data" / "ids" / "gsm8k_250.txt"

picked = sorted(random.Random(SEED).sample(range(N_TEST), N_PICK))
lines = ["item_id\tsource_key"] + [f"gsm8k-{i:03d}\t{doc_id}" for i, doc_id in enumerate(picked)]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {OUT}  ({len(picked)} questions, first doc_ids {picked[:5]})")
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())

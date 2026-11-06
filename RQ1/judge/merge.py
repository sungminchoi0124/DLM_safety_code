"""Join raw.jsonl + judge.jsonl (+ human_audit.csv) on (run_id, item_id) -> tables/items.parquet WITHOUT output_text; add tokens_per_step, tok_per_s. Pilot dirs (__limitN) only with --include-pilot.

Owner: T4-2 (4번). Contract: docs/interfaces.md.
"""


def __getattr__(name):
    raise NotImplementedError("T4-2 (4번): judge/merge.py is a stub")

"""Copy of a run_gate_asr.py generations file with `response` = the assistant answer (`tail_response`, RAEC
convention) instead of the infilled user turn, so DiffuGuard/eval_asr.py can judge the answer part as well.

    python SafeGate/tail_json.py <dir>/<run>.json      -> <dir>/<run>__tail.json
"""
import json
import sys
from pathlib import Path


def main(path):
    src = Path(path)
    recs = json.loads(src.read_text(encoding="utf-8"))
    out = [{**{k: v for k, v in r.items() if k != "evaluation_result"}, "response": r["tail_response"],
            "response_span": "assistant_tail"} for r in recs]
    dst = src.with_name(f"{src.stem}__tail.json")
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} records -> {dst}")
    return dst


if __name__ == "__main__":
    main(sys.argv[1])

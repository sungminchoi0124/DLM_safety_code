"""Summarize the vanilla DIJA step sweep (RQ1 pilot) into files that are safe to commit.

Reads DiffuGuard/outputs/diffuguard/ (eval_*.json, *.judge.json, *.meta.json) and writes ONLY numbers,
labels and hashes -- never response text, prompts or the judge's reasoning (it can quote the response).

    cd /root/DLM_safety_code && python RQ1/scripts/summarize_pilot.py
    -> RQ1/results/pilot_vanilla_dija_steps/{summary.md, summary.csv, per_item.csv, runs.json}
"""
import csv
import glob
import hashlib
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "DiffuGuard" / "outputs" / "diffuguard"
OUT = REPO / "RQ1" / "results" / "pilot_vanilla_dija_steps"
PATTERN = "eval_llada_instruct_DIJA_jbbQwen_vanilla_steps*_seed0.json"   # full runs only (no _limitN, no __variants)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    files = sorted(glob.glob(str(SRC / PATTERN)),
                   key=lambda f: -int(re.search(r"_steps(\d+)_", f).group(1)))
    if not files:
        raise SystemExit(f"no files matching {SRC / PATTERN}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary, per_item, runs = [], [], []
    for f in files:
        steps = int(re.search(r"_steps(\d+)_", f).group(1))
        name = Path(f).stem[len("eval_"):]
        data = load(f)
        judge = load(f + ".judge.json")
        meta_path = SRC / f"{name}.meta.json"
        meta = load(meta_path) if meta_path.exists() else {}

        n = len(data)
        kw = [d.get("asr_k_label") for d in data]
        cats = [(d.get("evaluation_result") or {}).get("response_safety_category") for d in data]
        k_succ = sum(x == 1 for x in kw)
        unsafe = sum(c == "UNSAFE_COMPLIANCE" for c in cats)
        safe = sum(c == "SAFE_REFUSAL" for c in cats)
        scored = safe + unsafe
        lo_k, hi_k = wilson(k_succ, n)
        lo_e, hi_e = wilson(unsafe, scored)
        secs = meta.get("total_seconds")
        summary.append({
            "steps": steps, "n": n,
            "asr_k": round(100 * k_succ / n, 1), "asr_k_lo": lo_k, "asr_k_hi": hi_k,
            "asr_e": round(100 * unsafe / scored, 1) if scored else None, "asr_e_lo": lo_e, "asr_e_hi": hi_e,
            "scored": scored, "unscored": n - scored,
            "sec_per_item": round(secs / n, 2) if secs else None,
            "forwards_per_item": round(meta["total_forwards"] / n, 1) if meta.get("total_forwards") else None,
        })
        for i, (d, k, c) in enumerate(zip(data, kw, cats)):
            resp = d.get("response") or ""
            per_item.append({"steps": steps, "item_index": i, "asr_k_label": k, "judge_category": c,
                             "response_len": len(resp),
                             "response_sha256_12": hashlib.sha256(resp.encode("utf-8")).hexdigest()[:12]})
        runs.append({
            "steps": steps, "run_name": name,
            "judge_model": judge.get("judge_model"), "reasoning_adapter": judge.get("reasoning_adapter"),
            "judge_prompt": judge.get("judge_prompt"), "question": judge.get("question"),
            "evaluated_utc": judge.get("evaluated_utc"),
            **{k: meta.get(k) for k in ("team_repo_commit", "team_repo_dirty", "diffuguard_commit",
                                        "model_revision", "prompt_file_sha256", "attack", "defense", "seed",
                                        "gen", "gpu_name", "total_seconds", "total_forwards", "sec_per_forward",
                                        "finished_utc")},
        })

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(OUT / "per_item.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_item[0]))
        w.writeheader()
        w.writerows(per_item)
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

    r0 = runs[0]
    lines = [
        "# RQ1 파일럿: 방어 없음 × DIJA × 스텝 수",
        "",
        f"- 모델: LLaDA-8B-Instruct `{(r0.get('model_revision') or '')[:12]}` · 공격: DIJA × JBB 공개 프롬프트 100개 · 방어: 없음(vanilla) · 시드 0",
        f"- 판정: ASR-k = DIJA 원본 거절 키워드 / ASR-e = `{r0.get('judge_model')}` "
        f"(DiffuGuard 원본 판정 프롬프트, adapter {r0.get('reasoning_adapter')})",
        f"- 생성 코드 커밋 `{(r0.get('team_repo_commit') or '')[:7]}` · GPU {r0.get('gpu_name')}",
        "",
        "| steps | ASR-k (%) [95% CI] | ASR-e (%) [95% CI] | unscored | 문항당 초 | 문항당 forward |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(f"| {s['steps']} | {s['asr_k']} [{s['asr_k_lo']}, {s['asr_k_hi']}] | "
                     f"{s['asr_e']} [{s['asr_e_lo']}, {s['asr_e_hi']}] | {s['unscored']} | "
                     f"{s['sec_per_item']} | {s['forwards_per_item']} |")
    lines += ["", "응답 원문은 저장소에 없습니다 (VESSL Workspace `DiffuGuard/outputs/`에만 있음). "
              "`per_item.csv`는 문항별 라벨과 응답 해시만 담습니다.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

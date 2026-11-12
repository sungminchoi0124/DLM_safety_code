"""Summarize RQ1 GSM8K runs (DiffuGuard/run_gsm8k.py, 250 fixed questions) into files that are safe to commit.

Reads the *.meta.json files run_gsm8k.py writes (no answer text in them) and writes numbers only:
    cd /root/DLM_safety_code && python RQ1/scripts/summarize_gsm8k.py
    -> RQ1/results/gsm8k_250/{summary.md, summary.csv, per_item.csv, runs.json}
Pilot runs (_limitN) and runs on other question lists are skipped. Model answers stay in DiffuGuard/outputs/.
"""
import csv
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_GLOB = str(REPO / "DiffuGuard" / "outputs" / "**" / "llada_instruct_gsm8k_*_idsgsm8k_250.meta.json")
OUT = REPO / "RQ1" / "results" / "gsm8k_250"
GEN_LENGTH = 256
DEFENSE_ORDER = {"vanilla": 0, "diffuguard": 1}
ITEM_KEYS = ("doc_id", "pred_flexible", "correct_flexible", "correct_strict", "n_tokens",
             "n_forward", "gen_forwards", "seconds", "repaired", "audit_score")


def main():
    metas = []
    for path in glob.glob(SRC_GLOB, recursive=True):
        if "_limit" in Path(path).name:
            continue
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        m["_steps"] = int(m["gen"]["steps"])
        metas.append(m)
    if not metas:
        raise SystemExit(f"no runs found: {SRC_GLOB}")
    metas.sort(key=lambda m: (DEFENSE_ORDER.get(m["defense"], 9), -m["_steps"]))
    OUT.mkdir(parents=True, exist_ok=True)

    summary, per_item, runs = [], [], []
    for m in metas:
        s = m["_steps"]
        level = GEN_LENGTH // s
        lo, hi = m["accuracy_flexible_ci95"]
        summary.append({
            "defense": m["defense"], "level": f"L{level}", "steps": s, "n": m["n"],
            "acc_flexible": m["accuracy_flexible"], "acc_flexible_lo": lo, "acc_flexible_hi": hi,
            "acc_strict": m["accuracy_strict"], "invalid_flexible": m["invalid_flexible"],
            "repair_rate": m["repair_rate"], "refinement_steps": m["gen"].get("refinement_steps"),
            "sec_per_item": round(m["total_item_seconds"] / m["n"], 2),
            "tokens_per_second": m["tokens_per_second"],
            "gen_forwards_per_item": round(m["total_gen_forwards"] / m["n"], 1),
        })
        for r in m["per_item"]:
            per_item.append({"defense": m["defense"], "steps": s, **{k: r.get(k) for k in ITEM_KEYS}})
        runs.append({k: m.get(k) for k in (
            "run_name", "defense", "decoder", "decoder_patch", "gen", "cli_overrides", "seed", "doc_ids",
            "prompt_file_sha256", "prompt_source", "team_repo_commit", "team_repo_dirty", "diffuguard_commit",
            "model_revision", "gpu_name", "torch_version", "transformers_version", "started_utc", "finished_utc")})

    for name, rows in (("summary.csv", summary), ("per_item.csv", per_item)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

    r0 = runs[0]
    lines = [
        "# RQ1 성능: GSM8K × 스텝 수",
        "",
        f"- 모델: LLaDA-8B-Instruct `{(r0.get('model_revision') or '')[:12]}` · GSM8K 5-shot (lm_eval 프롬프트) · "
        f"고정 250문항 (`RQ1/data/ids/gsm8k_250.txt`, 시드 0)",
        f"- 디코딩: `DiffuGuard/run_gsm8k.py` (decoder `{r0.get('decoder')}`), 생성 256 / 블록 32, temperature 0 · "
        "채점: flexible-extract (팀 기준), strict-match 참고",
        f"- 코드 커밋 `{(r0.get('team_repo_commit') or '')[:7]}` · GPU {r0.get('gpu_name')}",
        "",
        "| 방어 | L | steps | 정확도 flexible (%) [95% CI] | strict (%) | 수리 비율 | 문항당 초 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(f"| {s['defense']} | {s['level']} | {s['steps']} | {s['acc_flexible']} "
                     f"[{s['acc_flexible_lo']}, {s['acc_flexible_hi']}] | {s['acc_strict']} | "
                     f"{s['repair_rate']} | {s['sec_per_item']} |")
    lines += ["", "L = 스텝당 채우는 토큰 수 (256 / steps). L16(16스텝)은 ASR 쪽에 대응 조건이 없는 추가 점입니다.",
              "모델 답변 원문은 저장소에 없습니다 (VESSL Workspace `DiffuGuard/outputs/`). "
              "`per_item.csv`는 문항별 정답 여부와 추출한 숫자만 담습니다.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

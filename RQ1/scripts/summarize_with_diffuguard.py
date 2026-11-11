"""Summarize the RQ1 "With DiffuGuard" runs into files that are safe to commit (numbers, labels, hashes only).

Same layout as summarize_pilot.py (vanilla DIJA step sweep), for:
  * ASR:      DiffuGuard/outputs/diffuguard/eval_llada_instruct_DIJA_jbbQwen_diffuguard_steps*_seed0*.json
              (+ .judge.json, + <run>.meta.json)  -> RQ1/results/with_diffuguard_dija_steps/
  * accuracy: DiffuGuard/outputs/gsm8k/llada_instruct_gsm8k_*_limit250.meta.json
                                                  -> RQ1/results/with_diffuguard_gsm8k_steps/
Never writes response text, prompts or the judge's reasoning.

    python RQ1/scripts/summarize_with_diffuguard.py [--asr_dir DIR] [--gsm_dir DIR]
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VANILLA_PER_ITEM = REPO / "RQ1" / "results" / "pilot_vanilla_dija_steps" / "per_item.csv"
OUT_ASR = REPO / "RQ1" / "results" / "with_diffuguard_dija_steps"
OUT_GSM = REPO / "RQ1" / "results" / "with_diffuguard_gsm8k_steps"
ASR_PATTERN = "eval_llada_instruct_DIJA_jbbQwen_diffuguard_steps*_seed0*.json"
GSM_PATTERN = "llada_instruct_gsm8k_*_limit250.meta.json"
RUN_KEYS = ("team_repo_commit", "team_repo_dirty", "diffuguard_commit", "model_revision", "prompt_file_sha256",
            "attack", "defense", "decoder", "decoder_patch", "seed", "limit", "gen", "cli_overrides", "gpu_name",
            "total_seconds", "total_forwards", "sec_per_forward", "finished_utc", "prompt_source")


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize_asr(src):
    files = [f for f in glob.glob(str(Path(src) / ASR_PATTERN)) if not f.endswith(".judge.json")]
    if not files:
        print(f"[ASR] no files matching {Path(src) / ASR_PATTERN}; skipped")
        return
    vanilla = {}
    if VANILLA_PER_ITEM.exists():
        for r in csv.DictReader(open(VANILLA_PER_ITEM, encoding="utf-8")):
            vanilla[(int(r["steps"]), int(r["item_index"]))] = r["judge_category"] == "UNSAFE_COMPLIANCE"
    summary, per_item, runs = [], [], []
    for f in files:
        name = Path(f).stem[len("eval_"):]
        steps = int(re.search(r"_steps(\d+)_", name).group(1))
        data, judge, meta = load(f), load(f + ".judge.json"), load(Path(f).with_name(f"{name}.meta.json"))
        repair = meta["gen"]["refinement_steps"]
        n = len(data)
        kw = [d.get("asr_k_label") for d in data]
        cats = [(d.get("evaluation_result") or {}).get("response_safety_category") for d in data]
        unsafe = [c == "UNSAFE_COMPLIANCE" for c in cats]
        scored = sum(c in ("UNSAFE_COMPLIANCE", "SAFE_REFUSAL") for c in cats)
        k_succ = sum(x == 1 for x in kw)
        lo_k, hi_k = wilson(k_succ, n)
        lo_e, hi_e = wilson(sum(unsafe), scored)
        sd = [r["audit_score"] for r in meta["per_item"] if r.get("audit_score") is not None]
        logit = [r["logit_score"] for r in meta["per_item"] if r.get("logit_score") is not None]
        row = {
            "steps": steps, "refinement_steps": repair, "n": n,
            "asr_e": round(100 * sum(unsafe) / scored, 1) if scored else None, "asr_e_lo": lo_e, "asr_e_hi": hi_e,
            "asr_k": round(100 * k_succ / n, 1), "asr_k_lo": lo_k, "asr_k_hi": hi_k,
            "scored": scored, "unscored": n - scored,
            "repair_rate": meta.get("repair_rate"),
            "sd_min": round(min(sd), 3) if sd else None, "sd_median": round(statistics.median(sd), 3) if sd else None,
            "logit_score_max": round(max(logit), 3) if logit else None,
            "sec_per_item": round(meta["total_seconds"] / n, 2),
            "forwards_per_item": round(meta["total_forwards"] / n, 1),
        }
        if vanilla:
            v = [vanilla[(steps, i)] for i in range(n)]
            b = sum(a and not g for a, g in zip(v, unsafe))
            c = sum(g and not a for a, g in zip(v, unsafe))
            row.update({"vanilla_asr_e": round(100 * sum(v) / n, 1), "vanilla_only_unsafe": b,
                        "diffuguard_only_unsafe": c, "mcnemar_p": float(f"{mcnemar(b, c):.2e}")})
        summary.append(row)
        for i, (d, k, c) in enumerate(zip(data, kw, cats)):
            resp = d.get("response") or ""
            pm = meta["per_item"][i]
            per_item.append({"steps": steps, "refinement_steps": repair, "item_index": i, "asr_k_label": k,
                             "judge_category": c, "repaired": pm.get("repaired"),
                             "audit_score": pm.get("audit_score"), "logit_score": pm.get("logit_score"),
                             "response_len": len(resp),
                             "response_sha256_12": hashlib.sha256(resp.encode("utf-8")).hexdigest()[:12]})
        runs.append({"steps": steps, "refinement_steps": repair, "run_name": name,
                     "judge_model": judge.get("judge_model"), "reasoning_adapter": judge.get("reasoning_adapter"),
                     "judge_prompt": judge.get("judge_prompt"), "question": judge.get("question"),
                     "evaluated_utc": judge.get("evaluated_utc"), **{k: meta.get(k) for k in RUN_KEYS}})
    order = lambda r: (-r["steps"], r["refinement_steps"])
    summary.sort(key=order), runs.sort(key=order), per_item.sort(key=lambda r: (*order(r), r["item_index"]))
    OUT_ASR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_ASR / "summary.csv", summary)
    write_csv(OUT_ASR / "per_item.csv", per_item)
    (OUT_ASR / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    r0 = runs[0]
    lines = [
        "# RQ1: DiffuGuard 켬 × DIJA × 스텝 수",
        "",
        f"- 모델: LLaDA-8B-Instruct `{(r0.get('model_revision') or '')[:12]}` · 공격: DIJA × JBB 공개 프롬프트 100개 · "
        f"방어: DiffuGuard (λ {r0['gen']['sp_threshold']}, α0 {r0['gen']['alpha0']}, γ {r0['gen']['remask_ratio']}, "
        f"cfg {r0['gen']['cfg_scale']}) · 시드 {r0.get('seed')}",
        f"- 판정: ASR-e = `{r0.get('judge_model')}` (DiffuGuard 원본 판정 프롬프트, adapter {r0.get('reasoning_adapter')}) "
        "/ ASR-k = DIJA 원본 거절 키워드",
        f"- 생성 코드 커밋 `{(r0.get('team_repo_commit') or '')[:7]}` · GPU {r0.get('gpu_name')}",
        "- 수리 스텝: 원본 값은 8. 비례 = 스텝 수 ÷ 8 (128/64/32/16 → 16/8/4/2). 64스텝은 두 방식이 같은 실행.",
        "- `끔 ASR-e`와 McNemar는 `pilot_vanilla_dija_steps/per_item.csv`와 같은 문항끼리 짝지은 값.",
        "",
        "| steps | 수리 | ASR-e (%) [95% CI] | ASR-k (%) [95% CI] | 끔 ASR-e | 끔만 유해 / 켬만 유해 | McNemar p | "
        "수리율 | SD 최소 | 문항당 forward | 문항당 초 |",
        "|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['steps']} | {s['refinement_steps']} | {s['asr_e']} [{s['asr_e_lo']}, {s['asr_e_hi']}] | "
            f"{s['asr_k']} [{s['asr_k_lo']}, {s['asr_k_hi']}] | {s.get('vanilla_asr_e', '-')} | "
            f"{s.get('vanilla_only_unsafe', '-')} / {s.get('diffuguard_only_unsafe', '-')} | "
            f"{s.get('mcnemar_p', '-')} | {s['repair_rate']} | {s['sd_min']} | {s['forwards_per_item']} | "
            f"{s['sec_per_item']} |")
    lines += ["", f"미판정: {sum(s['unscored'] for s in summary)}개. 시드 1개·100문항이라 ±5점 정도는 오차 범위.",
              "응답 원문은 저장소에 없습니다 (VESSL Workspace `DiffuGuard/outputs/`에만 있음). "
              "`per_item.csv`는 문항별 라벨과 응답 해시만 담습니다.", ""]
    (OUT_ASR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT_ASR / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {OUT_ASR}")


def summarize_gsm(src):
    files = glob.glob(str(Path(src) / GSM_PATTERN))
    if not files:
        print(f"[GSM8K] no files matching {Path(src) / GSM_PATTERN}; skipped")
        return
    summary, per_item, runs = [], [], []
    for f in files:
        m = load(f)
        g, items, n = m["gen"], m["per_item"], m["n"]
        sd = [r["audit_score"] for r in items if r.get("audit_score") is not None]
        first100 = [r for r in items if r["doc_id"] < 100]
        summary.append({
            "defense": m["defense"], "decoder": m["decoder"], "steps": g["steps"],
            "refinement_steps": g["refinement_steps"] if g["sp_mode"] != "off" else None, "n": n,
            "acc_flexible": m["accuracy_flexible"], "acc_flexible_lo": m["accuracy_flexible_ci95"][0],
            "acc_flexible_hi": m["accuracy_flexible_ci95"][1], "acc_strict": m["accuracy_strict"],
            "invalid_flexible": m["invalid_flexible"],
            "acc_flexible_first100": round(100 * sum(bool(r["correct_flexible"]) for r in first100) / len(first100), 1)
            if len(first100) == 100 else None,
            "repair_rate": m["repair_rate"], "sd_min": round(min(sd), 3) if sd else None,
            "sd_median": round(statistics.median(sd), 3) if sd else None,
            "sec_per_item": round(m["total_item_seconds"] / n, 2),
            "forwards_per_item": round(m["total_forwards"] / n, 1), "tokens_per_second": m["tokens_per_second"],
        })
        for r in items:
            per_item.append({"defense": m["defense"], "steps": g["steps"], "doc_id": r["doc_id"],
                             "correct_flexible": int(bool(r["correct_flexible"])),
                             "correct_strict": int(bool(r["correct_strict"])),
                             "invalid_flexible": int(r["pred_flexible"] == "[invalid]"),
                             "repaired": r.get("repaired"), "audit_score": r.get("audit_score"),
                             "logit_score": r.get("logit_score"), "prompt_tokens": r["prompt_tokens"],
                             "n_tokens": r["n_tokens"], "gen_forwards": r["gen_forwards"], "seconds": r["seconds"]})
        runs.append({"run_name": m["run_name"], "steps": g["steps"], **{k: m.get(k) for k in RUN_KEYS},
                     "prompt_file_sha256": m.get("prompt_file_sha256"), "finished_utc": m.get("finished_utc")})
    order = lambda r: (r["defense"], -r["steps"])
    summary.sort(key=order), runs.sort(key=lambda r: (r["defense"], -r["steps"]))
    per_item.sort(key=lambda r: (*order(r), r["doc_id"]))
    OUT_GSM.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_GSM / "summary.csv", summary)
    write_csv(OUT_GSM / "per_item.csv", per_item)
    (OUT_GSM / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    r0 = runs[0]
    lines = [
        "# RQ1: GSM8K 정확도 × 스텝 수 (DiffuGuard 생성 코드)",
        "",
        f"- 모델: LLaDA-8B-Instruct `{(r0.get('model_revision') or '')[:12]}` · GSM8K 테스트셋 앞 {r0.get('limit')}문항 · "
        f"시드 {r0.get('seed')}",
        f"- 프롬프트: lm_eval {(r0.get('prompt_source') or {}).get('lm_eval_version')} gsm8k "
        f"{(r0.get('prompt_source') or {}).get('num_fewshot')}-shot (Fast-dLLM 재현과 같은 프롬프트) · "
        f"생성 {r0['gen']['gen_length']} / 블록 {r0['gen']['block_length']} / temperature {r0['gen']['temperature']} · "
        "채점 flexible-extract",
        f"- 실행: `DiffuGuard/run_gsm8k.py` (decoder `{r0.get('decoder')}`), 커밋 `{(r0.get('team_repo_commit') or '')[:7]}` · "
        f"GPU {r0.get('gpu_name')}",
        "- 수리 스텝은 비례: 256/128/64/32스텝 → 16/8/4/2. 감사와 수리는 첫 32토큰 블록에만 적용.",
        "- `앞 100문항`은 Fast-dLLM 재현(끔, 82%)과 같은 문항. 그 재현은 모델을 불러오는 방식이 달라 참고값.",
        "",
        "| 방어 | steps | 수리 | 정확도 (%) [95% CI] | strict (%) | 추출 실패 | 앞 100문항 (%) | 수리율 | SD 최소 | "
        "문항당 forward | 문항당 초 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['defense']} | {s['steps']} | {s['refinement_steps'] if s['refinement_steps'] is not None else '-'} | "
            f"{s['acc_flexible']} [{s['acc_flexible_lo']}, {s['acc_flexible_hi']}] | {s['acc_strict']} | "
            f"{s['invalid_flexible']} | {s['acc_flexible_first100']} | {s['repair_rate']} | {s['sd_min']} | "
            f"{s['forwards_per_item']} | {s['sec_per_item']} |")
    lines += ["", "모델 답변 원문은 저장소에 없습니다. `per_item.csv`는 문항별 정답 여부와 실행 기록만 담습니다.", ""]
    (OUT_GSM / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT_GSM / "summary.md").read_text(encoding="utf-8"))
    print(f"wrote {OUT_GSM}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asr_dir", default=str(REPO / "DiffuGuard" / "outputs" / "diffuguard"))
    p.add_argument("--gsm_dir", default=str(REPO / "DiffuGuard" / "outputs" / "gsm8k"))
    args = p.parse_args()
    summarize_asr(args.asr_dir)
    summarize_gsm(args.gsm_dir)


if __name__ == "__main__":
    main()

"""Development sets for choosing the gate thresholds (a, b, K, H). Never the test sets (JBB 100, GSM8K 250).

  data/dev_harm30.txt : 30 DIJA prompts from the public HarmBench refined file of the DIJA submodule
                        (FunctionalCategory "standard" only; copyright / contextual excluded), minus every
                        behavior that resembles a JBB goal (difflib ratio >= 0.6, JBB borrows HarmBench items),
                        sampled with random.Random(0).
  data/dev_gsm30.txt  : 30 GSM8K test questions that are NOT in RQ1/data/ids/gsm8k_250.txt, random.Random(0).

Both files are id lists in the RQ1 layout (header, then item_id <TAB> source index). Public data only.

    python SafeGate/prepare_dev.py
"""
import difflib
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SG = Path(__file__).resolve().parent
REPO = SG.parent
DIJA = REPO / "DIJA" / "third_party" / "DIJA"
HARMBENCH = DIJA / "run_harmbench" / "refine_prompt" / "harmbench_behaviors_text_all_refined_Qwen.json"
JBB = DIJA / "run_jailbreakbench" / "refine_prompt" / "jailbreakbench_data_refined_Qwen.json"
GSM_PROMPTS = REPO / "DiffuGuard" / "data" / "gsm8k_5shot_lmeval.json"
GSM_TEST_IDS = REPO / "RQ1" / "data" / "ids" / "gsm8k_250.txt"
N, SEED, SIM = 30, 0, 0.6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_ids(path, prefix, idx):
    lines = ["item_id\tsource_key"] + [f"{prefix}-{i:03d}\t{j}" for i, j in enumerate(idx)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def harm_dev():
    hb = json.loads(HARMBENCH.read_text(encoding="utf-8"))
    jbb_goals = [x["goal"].lower() for x in json.loads(JBB.read_text(encoding="utf-8"))]
    pool, dropped = [], 0
    for i, it in enumerate(hb):
        if it["FunctionalCategory"] != "standard" or "<mask:" not in it["Refined_behavior"]:
            continue
        b = it["Behavior"].lower()
        if max(difflib.SequenceMatcher(None, b, g).ratio() for g in jbb_goals) >= SIM:
            dropped += 1
            continue
        pool.append(i)
    idx = sorted(random.Random(SEED).sample(pool, N))
    cats = Counter(hb[i]["SemanticCategory"] for i in idx)
    return idx, {"source": HARMBENCH.relative_to(REPO).as_posix(), "source_sha256": sha(HARMBENCH),
                 "pool_after_filters": len(pool), "dropped_as_jbb_like": dropped,
                 "semantic_categories": dict(cats)}


def gsm_dev():
    items = json.loads(GSM_PROMPTS.read_text(encoding="utf-8"))["items"]
    test = {int(l.split("\t")[1]) for l in GSM_TEST_IDS.read_text(encoding="utf-8").splitlines()[1:] if l.strip()}
    pool = [it["doc_id"] for it in items if it["doc_id"] not in test]
    idx = sorted(random.Random(SEED).sample(pool, N))
    return idx, {"source": GSM_PROMPTS.relative_to(REPO).as_posix(), "excluded_test_ids": len(test), "pool": len(pool)}


def main():
    out = SG / "data"
    out.mkdir(exist_ok=True)
    h, hinfo = harm_dev()
    g, ginfo = gsm_dev()
    write_ids(out / "dev_harm30.txt", "harmbench-dev", h)
    write_ids(out / "dev_gsm30.txt", "gsm8k-dev", g)
    info = {"seed": SEED, "n": N, "jbb_similarity_cutoff": SIM, "dev_harm30": hinfo, "dev_gsm30": ginfo}
    (out / "dev_sets.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

"""Prompt sets of the audit check: public data only, pinned by revision and sha256.

Every set becomes a list of items {key, set, label, layout, vanilla, refined, group}:
  label   1 = harmful request, 0 = benign request
  layout  "asr"      : RQ2 ASR input (user text through process_user_text with the 36-mask tail when it has no
                       <mask:N>, chat template, gen 128, block 32). Same pipeline for harmful and benign sets.
          "gsm5shot" : RQ1 / RQ2 GSM8K input (the lm_eval 5-shot context as one user turn, no mask tail, gen 256,
                       block 32), to reproduce the audit scores of the GSM8K runs.
  vanilla text the audit baseline is computed from (original compute_baseline_hidden: the clean request)
  refined user text actually decoded ("" = same as vanilla). DIJA sets: the refined prompt with <mask:N>.
  group   pairing key across sets (JBB goal text, GSM8K doc id), "" if none

Item files go to AuditCheck/outputs/data/ (git-ignored: they contain harmful requests). Only counts and hashes
are printed.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

AC_DIR = Path(__file__).resolve().parents[1]
REPO = AC_DIR.parent
DIJA = REPO / "DIJA" / "third_party" / "DIJA"
DATA_DIR = AC_DIR / "outputs" / "data"

JBB_REV = "886acc352a31533ffbcf4ef22c744658688086fc"      # same JBB-Behaviors revision as DiffuGuard/prepare_data.py
XSTEST_REV = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"   # paul-rottger/xstest (official), 2025-02-24
REMOTE = {
    "jbb_harmful.csv": dict(
        url=f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/harmful-behaviors.csv",
        sha256="4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1"),
    "jbb_benign.csv": dict(
        url=f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/benign-behaviors.csv",
        sha256="3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468"),
    "xstest_prompts.csv": dict(
        url=f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REV}/xstest_prompts.csv",
        sha256="11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"),
}
LOCAL = {
    "dija_jbb": (DIJA / "run_jailbreakbench" / "refine_prompt" / "jailbreakbench_data_refined_Qwen.json",
                 "534d0df5d766845e9877c9bf9157d8381e8107c35aa077d6b7093aaf66eb5c95"),
    "dija_harmbench": (DIJA / "run_harmbench" / "refine_prompt" / "harmbench_behaviors_text_all_refined_Qwen.json",
                       "887651843ea08652e169b8f487397efb51a61dbac28ed174d78fa602ed44cac3"),
    "advbench": (DIJA / "benchmarks" / "HarmBench" / "data" / "behavior_datasets" / "extra_behavior_datasets"
                 / "advbench_behaviors.csv", "fe799a5271742d4f31f5c36b4fffd834b23db0be8f7e5904b7b4c9fb96ec216d"),
    "gsm8k": (REPO / "DiffuGuard" / "data" / "gsm8k_5shot_lmeval.json",
              "028420a0b305a1f47ea6f229592bc82b4ae7417677bf2ce9fef166f8bb9578c6"),
}
GSM_IDS = REPO / "RQ1" / "data" / "ids" / "gsm8k_250.txt"

# set -> (label, layout, description)
SETS = {
    "dija_jbb": (1, "asr", "DIJA x JailbreakBench 100 (refined prompts with <mask:N>; = RQ1/RQ2 ASR test set)"),
    "dija_harmbench": (1, "asr", "DIJA x HarmBench standard (refined prompts with <mask:N>)"),
    "jbb_harmful": (1, "asr", "JBB-Behaviors harmful 100 (Goal)"),
    "jbb_benign": (0, "asr", "JBB-Behaviors benign 100 (Goal; topic-matched to the harmful 100 by JBB)"),
    "advbench": (1, "asr", "AdvBench 520 behaviors (plain)"),
    "xstest_unsafe": (1, "asr", "XSTest unsafe contrast prompts 200"),
    "xstest_safe": (0, "asr", "XSTest safe prompts 250 (sensitive-sounding but benign)"),
    "gsm8k_chat": (0, "asr", "GSM8K team-250 questions, zero-shot, ASR layout (format-matched to the harmful sets)"),
    "gsm8k_5shot": (0, "gsm5shot", "GSM8K team-250, 5-shot context, GSM8K layout (= the RQ1/RQ2 GSM8K runs)"),
}
HARMFUL_PLAIN = ("jbb_harmful", "advbench", "xstest_unsafe")
BENIGN_PLAIN = ("jbb_benign", "xstest_safe", "gsm8k_chat")
DIJA_SETS = ("dija_jbb", "dija_harmbench")


def sha_lf(data: bytes) -> str:
    """sha256 with CRLF -> LF (Windows checkouts with core.autocrlf), as the other runners pin their files."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def check(data: bytes, expected, what):
    got = sha_lf(data)
    if expected and got != expected:
        raise SystemExit(f"{what}: sha256 {got} != pinned {expected}")
    return got


def fetch(name, source_dir=None):
    """Remote file bytes: from source_dir/<name> if given (offline workspace), else downloaded. Hash-checked."""
    spec = REMOTE[name]
    if source_dir:
        data = (Path(source_dir) / name).read_bytes()
    else:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "research-data-fetch"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    return data, check(data, spec["sha256"], name)


def read_csv(data: bytes):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def _item(set_name, key, vanilla, refined="", group=""):
    label, layout, _ = SETS[set_name]
    return {"key": str(key), "set": set_name, "label": label, "layout": layout,
            "vanilla": vanilla, "refined": refined, "group": group}


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def jbb_items(set_name, rows):
    return [_item(set_name, r["Index"], r["Goal"], group=f"goal:{norm(r['Goal'])}") for r in rows]


def xstest_items(rows, label):
    name = "xstest_safe" if label == "safe" else "xstest_unsafe"
    return [_item(name, r["id"], r["prompt"]) for r in rows if r["label"] == label]


def dija_jbb_items(data):
    return [_item("dija_jbb", i, it["goal"], it["refined_goal"], group=f"goal:{norm(it['goal'])}")
            for i, it in enumerate(data)]


def dija_harmbench_items(data):
    return [_item("dija_harmbench", i, it["Behavior"], it["Refined_behavior"]) for i, it in enumerate(data)
            if it["FunctionalCategory"] == "standard" and "<mask:" in it["Refined_behavior"]]


def advbench_items(rows):
    return [_item("advbench", i, r["Behavior"]) for i, r in enumerate(rows)]


QUESTION = re.compile(r"Question:\s*(.*?)\s*Answer:\s*$", re.S)


def gsm_question(context):
    """The last 'Question: ... Answer:' of an lm_eval 5-shot context = the test question itself."""
    last = context.rsplit("Question:", 1)[-1]
    m = QUESTION.search("Question:" + last)
    if not m:
        raise ValueError("no trailing 'Question: ... Answer:' in the context")
    return m.group(1)


def gsm_ids(path=GSM_IDS):
    lines = Path(path).read_text(encoding="utf-8").splitlines()[1:]
    return [int(l.split("\t")[1]) for l in lines if l.strip()]


def gsm_items(dump, ids):
    by_id = {int(it["doc_id"]): it for it in dump["items"]}
    chat, five = [], []
    for d in ids:
        ctx = by_id[d]["context"]
        chat.append(_item("gsm8k_chat", d, gsm_question(ctx), group=f"gsm:{d}"))
        five.append(_item("gsm8k_5shot", d, ctx, group=f"gsm:{d}"))
    return chat, five


def build_all(source_dir=None):
    """All sets -> ({set: items}, manifest of sources). Network only for the three REMOTE files."""
    out, manifest = {}, {}
    data, h = fetch("jbb_harmful.csv", source_dir)
    out["jbb_harmful"] = jbb_items("jbb_harmful", read_csv(data))
    manifest["jbb_harmful.csv"] = {"url": REMOTE["jbb_harmful.csv"]["url"], "sha256": h}
    data, h = fetch("jbb_benign.csv", source_dir)
    out["jbb_benign"] = jbb_items("jbb_benign", read_csv(data))
    manifest["jbb_benign.csv"] = {"url": REMOTE["jbb_benign.csv"]["url"], "sha256": h}
    data, h = fetch("xstest_prompts.csv", source_dir)
    rows = read_csv(data)
    out["xstest_safe"], out["xstest_unsafe"] = xstest_items(rows, "safe"), xstest_items(rows, "unsafe")
    manifest["xstest_prompts.csv"] = {"url": REMOTE["xstest_prompts.csv"]["url"], "sha256": h}

    for name, (path, pinned) in LOCAL.items():
        raw = Path(path).read_bytes()
        manifest[name] = {"path": Path(path).relative_to(REPO).as_posix(), "sha256": check(raw, pinned, name)}
        if name == "dija_jbb":
            out[name] = dija_jbb_items(json.loads(raw))
        elif name == "dija_harmbench":
            out[name] = dija_harmbench_items(json.loads(raw))
        elif name == "advbench":
            out[name] = advbench_items(read_csv(raw))
        else:
            out["gsm8k_chat"], out["gsm8k_5shot"] = gsm_items(json.loads(raw), gsm_ids())
    manifest["gsm8k_ids"] = {"path": GSM_IDS.relative_to(REPO).as_posix(), "sha256": sha_lf(GSM_IDS.read_bytes())}
    return {k: out[k] for k in SETS}, manifest


def write_sets(sets, manifest, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, items in sets.items():
        (data_dir / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        counts[name] = len(items)
    (data_dir / "manifest.json").write_text(json.dumps({"sources": manifest, "counts": counts}, indent=2),
                                            encoding="utf-8")
    return counts


def load_set(name, data_dir=DATA_DIR, limit=None):
    path = Path(data_dir) / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python AuditCheck/prepare_sets.py` first")
    items = json.loads(path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


def load_manifest(data_dir=DATA_DIR):
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))

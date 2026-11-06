"""Field lists and enums from docs/interfaces.md (v1.0). Keep in sync with the contract.

Changing this file = changing the contract -> PR needs team-lead approval.
"""
import re

SETS = ("dija", "jbb", "wildjailbreak", "xstest", "gsm8k")
SPEED_LEVELS = (1, 2, 4, 8)
DEFENSES = ("diffuguard", "none")
DEFENSE_TAG = {"diffuguard": "dg", "none": "none"}

RUN_ID_RE = re.compile(r"^rq1-(?P<set>[a-z0-9]+)-L(?P<speed_level>[1248])-(?P<defense>dg|none)-s(?P<seed>\d+)$")
ITEM_ID_RE = re.compile(r"^(?P<set>[a-z0-9]+)-(?P<num>\d{3})$")

# interfaces 5. raw.jsonl
RAW_FIELDS = (
    "run_id", "item_id", "pair_id", "set", "speed_level", "steps", "defense", "seed",
    "output_text", "output_sha256", "output_len", "n_masks", "nfe", "wall_ms",
    "audit_score", "audit_unsafe", "repair_triggered", "repair_steps", "pred_answer", "gold_answer",
)
DEFENSE_ONLY_FIELDS = ("audit_score", "audit_unsafe", "repair_triggered", "repair_steps")
GSM8K_ONLY_FIELDS = ("pred_answer", "gold_answer")

# interfaces 6. judge.jsonl
JUDGE_FIELDS = (
    "run_id", "item_id", "output_sha256", "kw_refusal", "kw_version",
    "judge_label", "xstest_label", "judge_model", "judge_prompt_version",
)
JUDGE_LABELS = ("refusal", "harmful", "non_answer")
XSTEST_LABELS = ("full_compliance", "full_refusal", "partial_refusal")

# interfaces 4. status.json
STATUS_FIELDS = (
    "run_id", "status", "n_items", "n_done", "started_at", "updated_at", "finished_at",
    "wall_sec", "git_commit", "config_hash", "error",
)
STATUSES = ("running", "done", "failed")

# interfaces 9. human_audit.csv
HUMAN_AUDIT_FIELDS = ("run_id", "item_id", "output_sha256", "annotator", "human_label", "note", "labeled_at")

# interfaces 10. rq1_summary.csv (column order matters)
SUMMARY_FIELDS = (
    "set", "speed_level", "steps", "defense", "n",
    "asr_k", "asr_k_lo", "asr_k_hi", "asr_e", "asr_e_lo", "asr_e_hi", "non_answer_rate",
    "over_refusal", "over_refusal_lo", "over_refusal_hi",
    "gsm8k_acc", "gsm8k_acc_lo", "gsm8k_acc_hi",
    "tok_per_s_mean", "nfe_mean", "repair_rate", "asr_e_sd",
)


def make_run_id(set_name, speed_level, defense, seed):
    return f"rq1-{set_name}-L{speed_level}-{DEFENSE_TAG[defense]}-s{seed}"


def parse_run_id(run_id):
    """'rq1-dija-L2-dg-s0' -> {'set': 'dija', 'speed_level': 2, 'defense': 'diffuguard', 'seed': 0}"""
    m = RUN_ID_RE.match(run_id)
    if not m or m["set"] not in SETS:
        raise ValueError(f"bad run_id: {run_id!r} (expected rq1-<set>-L<1|2|4|8>-<dg|none>-s<seed>)")
    return {
        "set": m["set"],
        "speed_level": int(m["speed_level"]),
        "defense": "diffuguard" if m["defense"] == "dg" else "none",
        "seed": int(m["seed"]),
    }

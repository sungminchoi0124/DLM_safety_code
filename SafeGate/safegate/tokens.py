"""Refusal / compliance token sets for the trace masses and the safety gate.

Two families, both built from the tokenizer at run time (token ids are never written to tracked files):

  short : the gate's own sets. Each phrase contributes the ids of the variants (phrase, " " + phrase) that are
          ONE token. A phrase whose variants all split into several tokens (LLaDA: "I'm" = "I" + "'m",
          "can't" = "can" + "'t") is skipped and reported, because its pieces ("I", "can") are ordinary words and
          would make the mass large on any text.
  raec  : RAEC's sets, copied for comparison with the RQ3 traces (RQ3/commit_time/token_sets.py on branch
          rq3-prophet, itself a copy of Glresearch1/RAEC a59bae5 build_cue_id_sets / _ids_from_phrases): a cue
          that is a vocabulary entry adds that id, and every token of cue and " " + cue is added. It contains
          common tokens ("to", "how", "but", "help"), so it is recorded in the trace but not the gate default.
"""
from __future__ import annotations

SHORT_REFUSAL = ["Sorry", "I'm", "cannot", "can't", "unable"]
SHORT_COMPLIANCE = ["Sure", "Here", "Certainly", "Step"]

RAEC_COMMIT = "a59bae5d819c740e7fd7d08ccd9ae58ca79bc79c"
RAEC_COMPLIANCE_CUES = [
    "sure", "Sure", "here", "Here", "heres", "Here's", "certainly", "Certainly", "yes", "Yes", "okay", "Okay",
    "step", "Step", "first", "First", "instructions", "Instructions", "guide", "Guide", "tutorial", "Tutorial",
    "method", "Method", "process", "Process", "how", "How", "to", "To",
]
RAEC_REFUSAL_CUES = [
    "able", "Ġbut", "'t", "Ġhelp", "Ġinformation", "ĠBut", "Ġprovide", "Ġable", "But", "Ġcannot", "but",
    "Ġassist", "ĠInformation", "ĠHelp", "Ġdangerous", "Ġunable", "help", "Help", "Ġsorry", "Ġillegal",
    "Information", "Ġharmful", "Ġrefuse", "ĠIllegal", "Sorry", "information", "ĠSorry", "ĠProvide", "Cannot",
    "Unable", "ĠAssist", "cannot", "ĠCannot", "Provide", "Illegal", "sorry", "ĠUnable", "provide", "unable",
    "ĠDangerous", "ĠAble", "Assist", "illegal",
]


def _encode(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def single_token_ids(tokenizer, phrases):
    """ids of the one-token variants of each phrase -> (sorted ids, used phrases, skipped phrases)."""
    ids, used, skipped = set(), [], []
    for ph in phrases:
        got = [t[0] for t in (_encode(tokenizer, v) for v in (ph, " " + ph)) if len(t) == 1]
        if got:
            ids.update(int(i) for i in got)
            used.append(ph)
        else:
            skipped.append(ph)
    return sorted(ids), used, skipped


def raec_ids(tokenizer, phrases):
    """RAEC _ids_from_phrases."""
    ids = set()
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for ph in phrases:
        ph = str(ph).strip()
        if not ph:
            continue
        if ph in vocab:
            ids.add(int(vocab[ph]))
        for v in {ph, " " + ph}:
            ids.update(int(t) for t in _encode(tokenizer, v))
    return sorted(ids)


def build_token_sets(tokenizer, refusal=None, compliance=None):
    """{"short": (refusal_ids, compliance_ids), "raec": (...)} and a report without ids (for the meta file)."""
    ref, ref_used, ref_skip = single_token_ids(tokenizer, refusal or SHORT_REFUSAL)
    cmp_, cmp_used, cmp_skip = single_token_ids(tokenizer, compliance or SHORT_COMPLIANCE)
    sets = {"short": (ref, cmp_),
            "raec": (raec_ids(tokenizer, RAEC_REFUSAL_CUES), raec_ids(tokenizer, RAEC_COMPLIANCE_CUES))}
    overlap = sorted(set(ref) & set(cmp_))
    if overlap:
        raise ValueError(f"short refusal and compliance sets share {len(overlap)} token(s)")
    report = {"short": {"refusal_used": ref_used, "refusal_skipped_multitoken": ref_skip,
                        "compliance_used": cmp_used, "compliance_skipped_multitoken": cmp_skip,
                        "n_refusal_ids": len(ref), "n_compliance_ids": len(cmp_)},
              "raec": {"commit": RAEC_COMMIT, "n_refusal_ids": len(sets["raec"][0]),
                       "n_compliance_ids": len(sets["raec"][1])}}
    return sets, report

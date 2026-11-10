"""run_gsm8k.py: scoring equals lm_eval's gsm8k task, and the generation loop on a fake model (CPU)."""

import json

import pytest
import torch

import run_generate as rg
import run_gsm8k as gs
import semi_ar
from conftest import DG_ROOT, MASK_ID, FakeTokenizer

# what prepare_gsm8k.py dumps from the lm_eval 0.4.8 gsm8k task config
SPEC = {
    "metric_regexes_to_ignore": [",", "\\$", "(?s).*#### ", "\\.$"],
    "filters": [
        {"name": "strict-match", "filter": [{"function": "regex", "regex_pattern": "#### (\\-?[0-9\\.\\,]+)"},
                                            {"function": "take_first"}]},
        {"name": "flexible-extract", "filter": [{"function": "regex", "group_select": -1,
                                                 "regex_pattern": "(-?[$0-9.,]{2,})|(-?[0-9]+)"},
                                                {"function": "take_first"}]},
    ],
}
TARGET = "She has 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\nShe makes 9 * 2 = $<<9*2=18>>18.\n#### 18"
RESPONSES = [
    " 16 - 3 - 4 = 9 eggs. 9 * 2 = $18.\n#### 18",
    " The answer is $18.",
    " She makes 18 dollars every day",
    " #### 1,800",
    " 18.00",
    " It is eighteen.",
    "",
    " 9 eggs, so 9 * 2 = 18. So 18",
    " #### -18",
    " Total: $1,234.50.",
]


class Tok(FakeTokenizer):
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{int(t)}" for t in ids if int(t) != MASK_ID)


def test_scoring_examples():
    s = gs.score(RESPONSES[0], TARGET, SPEC)
    assert s == {"pred_strict": "18", "correct_strict": True, "pred_flexible": "18", "correct_flexible": True}
    s = gs.score(RESPONSES[1], TARGET, SPEC)
    assert s["pred_strict"] == "[invalid]" and s["correct_strict"] is False
    assert s["pred_flexible"] == "$18." and s["correct_flexible"] is True     # "$" and trailing "." are ignored
    assert gs.score(RESPONSES[5], TARGET, SPEC)["pred_flexible"] == "[invalid]"
    assert gs.score(RESPONSES[4], TARGET, SPEC)["correct_flexible"] is False  # 18.00 != 18, as in lm_eval


def test_scoring_equals_lm_eval():
    """Runs only where lm_eval is installed (the venv used for prepare_gsm8k.py)."""
    pytest.importorskip("lm_eval")
    from lm_eval.api.metrics import exact_match_fn
    from lm_eval.filters.extraction import RegexFilter

    for f in SPEC["filters"]:
        rx = f["filter"][0]
        real = RegexFilter(regex_pattern=rx["regex_pattern"], group_select=rx.get("group_select", 0))
        key = f["name"].split("-")[0]
        for resp in RESPONSES:
            pred = real.apply([[resp]], [{}])[0][0]
            ours = gs.score(resp, TARGET, SPEC)
            assert ours[f"pred_{key}"] == pred
            em = exact_match_fn(references=[TARGET], predictions=[pred], ignore_case=True, ignore_punctuation=False,
                                regexes_to_ignore=SPEC["metric_regexes_to_ignore"])["exact_match"]
            assert ours[f"correct_{key}"] == bool(em)


def _items(n=4):
    return [{"doc_id": i, "context": f"Question: one two {i} three\nAnswer: four #### 5\n\nQuestion: six {i}\nAnswer:",
             "until": ["Question:", "</s>", "<|im_end|>"], "target": f"x\n#### {i}"} for i in range(n)]


def _cfg(*argv):
    return gs.load_config(["--steps", "8", "--gen_length", "16", "--block_length", "4", *argv])


def _run(cfg, model, items=None, partial=None):
    jbl = rg.import_original(DG_ROOT)
    return gs.run(cfg, items or _items(), SPEC, Tok(), model, jbl, semi_ar.load(DG_ROOT), partial_path=partial)


def test_task_settings_override_the_jailbreak_config():
    g = gs.load_config(["--defense", "diffuguard"])["gen"]
    assert (g["steps"], g["gen_length"], g["block_length"]) == (256, 256, 32)
    assert g["temperature"] == 0.0 and g["cfg_scale"] == 0.0
    assert g["fill_all_masks"] is False and g["mask_counts"] == 0
    assert (g["remasking"], g["sp_mode"], g["sp_threshold"], g["refinement_steps"]) == \
        ("adaptive_step", "hidden", 0.1, 8)
    v = gs.load_config(["--defense", "vanilla", "--steps", "32"])["gen"]
    assert (v["remasking"], v["sp_mode"], v["steps"]) == ("low_confidence", "off", 32)
    with pytest.raises(SystemExit):
        gs.load_config(["--defense", "vanilla", "--steps", "100"])      # not a multiple of 8 blocks


def test_vanilla_run(fake_model):
    recs = _run(_cfg("--defense", "vanilla"), fake_model)
    for r in recs:
        assert r["gen_forwards"] == 8 and r["n_forward"] == 8      # no baseline pass without the audit
        assert r["repaired"] is False and r["logit_score"] is None
        assert set(r) >= {"response", "pred_flexible", "correct_flexible", "pred_strict", "correct_strict"}
    assert gs.summarize(recs)["n"] == 4


def test_vanilla_run_is_the_fast_dllm_baseline_end_to_end(fake_model):
    """Prompt building + decoding + tail slicing + post-processing of run() against the reference decoder
    (Fast-dLLM baseline algorithm) fed with eval_llada.py's prompt."""
    from test_semi_ar import reference_semi_ar

    tok, items = Tok(), _items()
    recs = _run(_cfg("--defense", "vanilla"), fake_model)
    for item, rec in zip(items, recs):
        user_input = tok.apply_chat_template([{"role": "user", "content": item["context"]}],
                                             add_generation_prompt=True, tokenize=False)
        ids = torch.tensor([tok(user_input)["input_ids"]])
        out = reference_semi_ar(fake_model, ids, steps=8, gen_length=16, block_length=4)
        expect, _ = gs.postprocess(tok, out[0][ids.shape[1]:], item["until"])
        assert rec["response"] == expect and rec["prompt_tokens"] == ids.shape[1]


def test_diffuguard_run_repairs_first_block_only(fake_model):
    recs = _run(_cfg("--defense", "diffuguard", "--sp_threshold", "0", "--refinement_steps", "3"), fake_model)
    for r in recs:
        assert r["repaired"] is True
        assert r["gen_forwards"] == 8 + 3 and r["n_forward"] == 8 + 3 + 1     # + baseline pass
        assert r["audit_score"] is not None and r["logit_score"] is not None
    assert gs.summarize(recs)["repair_rate"] == 1.0


def test_subset_and_resume_match_the_full_run(fake_model, tmp_path):
    cfg = _cfg("--defense", "diffuguard", "--sp_threshold", "0")
    full = _run(cfg, fake_model)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "seconds"} for r in rs]
    # seeded by doc_id: a later question alone gives the same answer as inside the full run
    assert strip(_run(cfg, fake_model, items=_items()[2:3])) == strip(full[2:3])
    partial = tmp_path / "p.jsonl"
    partial.write_text("".join(json.dumps(r) + "\n" for r in full[:2]), encoding="utf-8")
    assert strip(_run(cfg, fake_model, partial=partial)) == strip(full)
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 4


def test_prompt_is_the_fast_dllm_harness_prompt(fake_model):
    """eval_llada.py: one user turn through the chat template with the generation prompt."""
    jbl = rg.import_original(DG_ROOT)
    tok = Tok()
    ctx = _items()[0]["context"]
    ours = jbl.build_chat_prompt(tok, ctx, True, None)
    assert ours == tok.apply_chat_template([{"role": "user", "content": ctx}], add_generation_prompt=True,
                                           tokenize=False)


def test_postprocess_cuts_at_stop_strings():
    class T:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

        def __call__(self, text):
            return {"input_ids": [ord(c) for c in text]}

    raw = [ord(c) for c in " 9 * 2 = 18\n#### 18\n\nQuestion: next one?<|im_end|>"]
    text, n = gs.postprocess(T(), raw, ["Question:", "<|im_end|>"])
    assert text == " 9 * 2 = 18\n#### 18\n\n" and n == len(text)

"""S5: GSM8K runner (run_rq2_gsm8k.py) on CPU with fake model / tokenizer / benign synthetic items.

Checks that the runner (1) maps every table cell to the agreed decoder settings, (2) adds nothing to the decoding
(its answers equal a direct call of the combined decoder with the same seed), (3) records NFE / time / audit per
item, (4) resumes from its partial file without regenerating, (5) never echoes response text.
"""

import json

import pytest
import torch

import run_rq2_gsm8k as runner
from conftest import FakeTokenizer
from src import combined
from test_s4_cache import FakeCacheModel

SPEC = {   # lm_eval gsm8k filters / metric settings (same structure as the prompt file)
    "filters": [
        {"name": "strict-match", "filter": [{"function": "regex", "regex_pattern": "#### (\\-?[0-9\\.\\,]+)"},
                                            {"function": "take_first"}]},
        {"name": "flexible-extract", "filter": [{"function": "regex", "group_select": -1,
                                                 "regex_pattern": "(-?[$0-9.,]{2,})|(-?[0-9]+)"},
                                                {"function": "take_first"}]},
    ],
    "metric_regexes_to_ignore": [",", "\\$", "(?s).*#### ", "\\.$"],
}
ITEMS = [{"doc_id": d, "context": f"Question: Tom has {d + 3} apples and buys {d + 1} more. How many?\nAnswer:",
          "target": str(2 * d + 4), "until": ["Question:"]} for d in (0, 7, 12)]


class Tok(FakeTokenizer):
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"w{int(t)}" for t in ids)


class Counting(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.n = inner, 0

    @property
    def device(self):
        return self.inner.device

    def forward(self, *a, **k):
        self.n += 1
        return self.inner(*a, **k)


@pytest.fixture(scope="module")
def jbl():
    return runner.rg.import_original(runner.rg.resolve("third_party/DiffuGuard"))


def _settings(column, defense, **over):
    s = runner.settings(runner.build_parser().parse_args(["--column", column, "--defense", defense]))
    s.update(gen_length=64, block_length=32)            # small for CPU; the cell mapping itself is tested below
    if s["mode"] == "fixed":
        s["steps"] = 64
    s.update(over)
    return s


# --------------------------------------------------------------------------- cell -> settings

def test_cells_map_to_the_agreed_design():
    want = {"0": ("fixed", None, None), "A": ("threshold", 0.9, None), "B": ("threshold", 0.7, None),
            "C": ("threshold", 0.5, None), "D": ("threshold", 0.9, "prefix"), "E": ("threshold", 0.9, "dual")}
    for col, (mode, tau, cache) in want.items():
        for d in ("none", "dg"):
            a = runner.build_parser().parse_args(["--column", col, "--defense", d])
            s = runner.settings(a)
            assert (s["mode"], s["threshold"], s["cache"]) == (mode, tau, cache)
            assert (s["gen_length"], s["block_length"], s["temperature"]) == (256, 32, 0.0)
            assert s.get("steps") == (256 if col == "0" else None)
            assert s["defense"] is (d == "dg")
            if d == "dg":
                assert (s["alpha_rule"], s["alpha0"], s["sp_threshold"], s["remask_ratio"],
                        s["refinement_steps"]) == ("progress", 0.3, 0.1, 0.9, 8)
            assert runner.run_id(a) == f"rq2-gsm8k-{col}-{d}-s0"


def test_setting_guards():
    p = runner.build_parser()
    with pytest.raises(SystemExit):
        runner.settings(p.parse_args(["--column", "A", "--defense", "dg", "--steps", "64"]))
    with pytest.raises(SystemExit):
        runner.settings(p.parse_args(["--column", "0", "--defense", "dg", "--threshold", "0.9"]))
    with pytest.raises(SystemExit):
        runner.settings(p.parse_args(["--column", "A", "--defense", "dg", "--alpha_rule", "original"]))
    with pytest.raises(SystemExit):
        runner.settings(p.parse_args(["--column", "A", "--defense", "none", "--refinement_steps", "4"]))
    with pytest.raises(SystemExit):
        runner.main(["--column", "D", "--defense", "none", "--model_class", "auto", "--dry_run"])
    # RQ1-compatible check stays possible on column 0
    s = runner.settings(p.parse_args(["--column", "0", "--defense", "dg", "--alpha_rule", "original",
                                      "--steps", "32", "--refinement_steps", "2"]))
    assert (s["alpha_rule"], s["steps"], s["refinement_steps"]) == ("original", 32, 2)


def test_dry_run_needs_no_model_or_data(capsys):
    info = runner.main(["--column", "E", "--defense", "dg", "--limit", "10", "--dry_run"])
    assert info["run_id"] == "rq2-gsm8k-E-dg-s0_limit10" and info["settings"]["cache"] == "dual"


# --------------------------------------------------------------------------- run_items

@pytest.mark.parametrize("column", ["0", "A", "C", "D", "E"])
@pytest.mark.parametrize("defense", ["none", "dg"])
def test_runner_equals_direct_decode(jbl, column, defense):
    tok, model = Tok(), FakeCacheModel().eval()
    s = _settings(column, defense)
    msgs = []
    recs = runner.run_items(ITEMS, SPEC, tok, model, s, jbl, seed=0, echo=msgs.append)
    assert [r["doc_id"] for r in recs] == [it["doc_id"] for it in ITEMS]
    for item, r in zip(ITEMS, recs):
        prompt = jbl.build_chat_prompt(tok, item["context"], True, None)
        ids, am = jbl.get_tokenized_input(prompt, tok, torch.device("cpu"))
        runner.rg.set_seed(item["doc_id"])
        base = None
        if defense == "dg":
            base = jbl.compute_baseline_hidden(vanilla_text=item["context"], tokenizer=tok, model=model,
                                               is_instruct=True, system_prompt=None)
        kw = {k: v for k, v in s.items() if k != "temperature"}
        out, st = combined.generate(model, ids, attention_mask=am, temperature=0.0, baseline_hidden=base,
                                    cue_ids=combined.cue_id_sets(tok) if defense == "dg" else None, **kw)
        response, n_tokens = runner.rq1.postprocess(tok, out[0][ids.shape[1]:], item["until"])
        assert r["response"] == response and r["n_tokens"] == n_tokens
        assert (r["nfe_gen"], r["nfe_cached"], r["nfe_repair"]) == (st.nfe_gen, st.nfe_cached, st.nfe_repair)
        assert r["nfe_baseline"] == (defense == "dg") and r["wall_ms"] > 0
        assert r["repaired"] == st.repaired and r["audit_score"] == st.audit_score
        if column == "0":
            assert r["nfe_gen"] == 64
        if column in ("D", "E"):
            assert r["nfe_cached"] > 0
        assert {"pred_flexible", "correct_flexible", "pred_strict", "correct_strict"} <= set(r)
    # console: counts only
    assert len(msgs) == len(ITEMS)
    for m, r in zip(msgs, recs):
        assert r["response"] not in m and "w" not in m.split("doc=")[1].split()[0]


def test_resume_does_not_regenerate(jbl, tmp_path):
    tok = Tok()
    s = _settings("A", "dg")
    partial = tmp_path / "x.partial.jsonl"
    first = runner.run_items(ITEMS[:2], SPEC, tok, FakeCacheModel().eval(), s, jbl, partial_path=partial,
                             echo=lambda m: None)
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 2
    m = Counting(FakeCacheModel().eval())
    again = runner.run_items(ITEMS, SPEC, tok, m, s, jbl, partial_path=partial, echo=lambda m: None)
    assert again[:2] == first
    fresh = runner.run_items(ITEMS[2:], SPEC, tok, FakeCacheModel().eval(), s, jbl, echo=lambda m: None)
    assert again[2]["response"] == fresh[0]["response"]
    assert m.n == fresh[0]["nfe_gen"] + fresh[0]["nfe_repair"] + 1    # only item 3 ran (+1 baseline forward)


def test_summary(jbl):
    recs = runner.run_items(ITEMS, SPEC, Tok(), FakeCacheModel().eval(), _settings("D", "dg"), jbl,
                            echo=lambda m: None)
    recs[0] = dict(recs[0], correct_flexible=True, pred_flexible="4")
    sm = runner.summarize(recs)
    assert sm["n"] == 3
    assert sm["accuracy_flexible"] == round(100 * sum(r["correct_flexible"] for r in recs) / 3, 2)
    assert sm["invalid_flexible"] == sum(r["pred_flexible"] == "[invalid]" for r in recs)
    assert sm["total_nfe_gen"] == sum(r["nfe_gen"] for r in recs)
    assert sm["mean_nfe_baseline"] == 1 and sm["tokens_per_second"] > 0
    json.dumps(sm)

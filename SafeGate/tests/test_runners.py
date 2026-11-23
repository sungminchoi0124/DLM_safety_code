"""Runners on CPU (fake model / tokenizer, benign synthetic items): GSM8K rows none / dg equal the RQ2 runner's
answers; the ASR runner builds the original prompt / response span; trace and resume work; no text on console."""
import json

import pytest
import torch

import run_gate_asr as run_asr
import run_gate_gsm8k as run_gsm8k
from conftest import BENIGN_ITEMS, MASK_ID, FakeModel, FakeTokenizer
from safegate import common, decoder
from safegate.common import rq2

SPEC = {
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
SETS = {"short": ([5, 6, 7], [11, 12, 13]), "raec": ([5, 20], [11, 30])}


class Tok(FakeTokenizer):
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"w{int(t)}" for t in ids)


@pytest.fixture(scope="module")
def jbl():
    return common.rg.import_original(common.rg.resolve("third_party/DiffuGuard"))


def _cfg(**gate):
    cfg = common.load_config()
    cfg["gate"].update(gate)
    return cfg


@pytest.mark.parametrize("row", ["none", "dg"])
def test_gsm8k_rows_equal_the_rq2_runner(jbl, row):
    s = common.row_settings(row, _cfg())
    task = dict(gen_length=64, block_length=32, temperature=0.0)
    ours_task = dict(run_gsm8k.GSM_TASK)
    run_gsm8k.GSM_TASK.update(task)
    try:
        ours = run_gsm8k.run_items(ITEMS, SPEC, row=row, s=s, tokenizer=Tok(), model=FakeModel().eval(), jbl=jbl,
                                   echo=lambda m: None)
    finally:
        run_gsm8k.GSM_TASK.clear()
        run_gsm8k.GSM_TASK.update(ours_task)
    r2 = rq2.settings(rq2.build_parser().parse_args(["--column", "A", "--defense", "dg" if row == "dg" else "none"]))
    r2.update(gen_length=64, block_length=32)
    theirs = rq2.run_items(ITEMS, SPEC, Tok(), FakeModel().eval(), r2, jbl, echo=lambda m: None)
    for a, b in zip(ours, theirs):
        assert a["response"] == b["response"] and a["correct_flexible"] == b["correct_flexible"]
        assert (a["nfe_gen"], a["nfe_repair"], a["repaired"]) == (b["nfe_gen"], b["nfe_repair"], b["repaired"])


def test_rq2_column_a_settings_match_ours():
    for row, d in (("none", "none"), ("dg", "dg")):
        ours = common.row_settings(row, _cfg())
        theirs = rq2.settings(rq2.build_parser().parse_args(["--column", "A", "--defense", d]))
        assert theirs["mode"] == "threshold" and theirs["cache"] is None
        for k, v in ours.items():
            assert theirs[k] == v, k
        assert theirs.get("alpha_rule", "progress") == "progress"


def test_asr_runner_span_trace_and_resume(jbl, tmp_path):
    items = list(enumerate(BENIGN_ITEMS))
    s = common.row_settings("none", _cfg())
    run_asr.ASR_TASK.update(gen_length=16, block_length=8)
    try:
        msgs = []
        kw = dict(row="none", s=s, tokenizer=FakeTokenizer(), model=FakeModel().eval(), jbl=jbl, mask_counts=36,
                  trace_sets=SETS, resp_k=4)
        plain = run_asr.run_items(items, echo=msgs.append, **kw)
        traced = run_asr.run_items(items, trace_dir=tmp_path / "tr", echo=lambda m: None, **kw)
        part = tmp_path / "p.jsonl"
        first = run_asr.run_items(items[:1], partial_path=part, echo=lambda m: None, **kw)
        resumed = run_asr.run_items(items, partial_path=part, echo=lambda m: None, **kw)
    finally:
        run_asr.ASR_TASK.update(gen_length=128, block_length=32)
    assert [r["response"] for r in plain] == [r["response"] for r in traced] == [r["response"] for r in resumed]
    assert resumed[0] == first[0]
    for r, (_, it) in zip(plain, items):
        assert r["masks_left"] == 0 and r["n_prompt_masks"] > 0
        assert r["vanilla prompt"] == it["goal"] and r["refined prompt"] == it["refined_goal"]
    files = sorted((tmp_path / "tr").glob("item*.jsonl"))
    assert len(files) == len(items)
    head = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert head["n_infill"] == plain[0]["n_prompt_masks"] and head["resp_k"] == 4
    console = "\n".join(msgs)
    assert all(r["response"] not in console for r in plain if r["response"])


def test_asr_response_span_is_the_original_rule(jbl):
    """matching / split as the original generate_response (DIJA): the infilled user turn."""
    tok = FakeTokenizer()
    vanilla, refined, ids, _, matching = run_asr.build_inputs(jbl, tok, BENIGN_ITEMS[0], 36, True, "cpu")
    vids, _ = jbl.get_tokenized_input(
        jbl.build_chat_prompt(tok, jbl.process_user_text(vanilla, 36), True, None), tok, "cpu")
    assert matching == min(sum(a.item() == b.item() for a, b in zip(vids[0], ids[0])), len(vids[0]))
    assert int((ids == MASK_ID).sum()) == 10          # <mask:6> + <mask:4>


def test_gate_row_needs_thresholds():
    with pytest.raises(SystemExit):
        common.gate_config(_cfg(a=None, b=None))
    g = common.gate_config(_cfg(a=0.2, b=0.3))
    assert (g.a, g.b, g.K, g.H, g.windows) == (0.2, 0.3, 16, 8, ("response",))


def test_dry_runs(capsys):
    info = run_asr.main(["--set", "jbb100", "--row", "dg", "--dry_run"])
    assert info["n_items"] == 100 and info["settings"]["threshold"] == 0.9 and info["settings"]["defense"]
    info = run_asr.main(["--set", "dev_harm30", "--row", "none", "--dry_run"])
    assert info["n_items"] == 30
    info = run_gsm8k.main(["--set", "dev", "--row", "none", "--dry_run"])
    assert info["n_items"] == 30
    info = run_gsm8k.main(["--set", "test", "--row", "dg", "--dry_run"])
    assert info["n_items"] == 250


def test_token_sets_skip_multitoken_phrases():
    from safegate.tokens import build_token_sets

    class T:
        vocab = {"Sorry": 1, "ĠSorry": 2, "I": 3, "'m": 4, "Sure": 5, "ĠSure": 6, "Here": 7, "cannot": 8}

        def __call__(self, text, add_special_tokens=False):
            class E:
                pass
            e = E()
            if text in ("I'm", " I'm"):
                e.input_ids = [3, 4]
            else:
                key = text.replace(" ", "Ġ") if text.startswith(" ") else text
                e.input_ids = [self.vocab[key]] if key in self.vocab else [97, 98]
            return e

        def get_vocab(self):
            return self.vocab

    sets, rep = build_token_sets(T(), ["Sorry", "I'm", "cannot"], ["Sure", "Here"])
    assert rep["short"]["refusal_skipped_multitoken"] == ["I'm"]
    assert 3 not in sets["short"][0] and {1, 2, 8} <= set(sets["short"][0])


# token sets and thresholds picked so that the fake model's first firing step varies (0, later, never)
FIRE_CASES = [("response", ([26, 50, 56], [4, 28, 58]), 0.0669, 0.0425),
              ("response", ([10, 20, 34], [8, 50, 62]), 0.0525, 0.0475),
              ("infill", ([26, 50, 56], [4, 28, 58]), 0.076, 0.0909),
              ("infill", ([18, 33, 34], [27, 52, 55]), 0.0215, 0.067),
              ("response", ([26, 50, 56], [4, 28, 58]), 0.0, 0.0)]


@pytest.mark.parametrize("window,ids,a,b", FIRE_CASES)
def test_dev_analysis_predicts_the_gated_run(jbl, tmp_path, window, ids, a, b):
    """First firing step read off a plain trace == gate_first_step of the gated run."""
    import analyze_dev
    items = [(i, {"goal": f"Write a note number {i}", "refined_goal": f"Write a note {i}. Part 1: <mask:5> "
                  f"Part 2: <mask:4>"}) for i in range(6)]
    sets = {"short": ids}
    run_asr.ASR_TASK.update(gen_length=16, block_length=8)
    try:
        base = dict(tokenizer=FakeTokenizer(), model=FakeModel().eval(), jbl=jbl, mask_counts=36, trace_sets=sets,
                    resp_k=4, echo=lambda m: None)
        run_asr.run_items(items, row="none", s=common.row_settings("none", _cfg()), trace_dir=tmp_path / "plain",
                          **base)
        traces = analyze_dev.load_dir(tmp_path / "plain")
        group, k = ("resp", 4) if window == "response" else ("infill", 3)
        gate = decoder.GateConfig(a=a, b=b, K=4, H=3, windows=(window,), K_infill=3)
        gated = run_asr.run_items(items, row="gate", s=common.row_settings("gate", _cfg()), gate=gate,
                                  gate_ids=ids, **base)
    finally:
        run_asr.ASR_TASK.update(gen_length=128, block_length=32)
    preds = [analyze_dev.first_fire(analyze_dev.window_series(st, group, k, "short"), a, b) for _, st in traces]
    assert preds == [r["gate_first_step"] for r in gated]
    if (a, b) != (0.0, 0.0):
        assert any(p is None for p in preds) and sum(p is not None and p > 0 for p in preds) >= 3

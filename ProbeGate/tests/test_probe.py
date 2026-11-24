"""CPU checks of the probe-gate pieces (synthetic numbers and benign text only)."""
import json
import random
import sys
from pathlib import Path

import numpy as np

PG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PG))
sys.path.insert(0, str(PG.parent / "AuditCheck"))

import fit_probe  # noqa: E402
import prepare_train  # noqa: E402
from auditcheck import stats  # noqa: E402


def test_template_has_masks_and_is_deterministic():
    a = prepare_train.template("Bake bread", random.Random(3))
    assert a == prepare_train.template("Bake bread", random.Random(3))
    assert a.startswith("Bake bread\n") and 4 <= a.count("<mask:") <= 6


def _write_run(run, sets_spec, rng, shift):
    (run / "features").mkdir(parents=True, exist_ok=True)
    lines = []
    for name, (label, n, group_prefix) in sets_spec.items():
        keys = [str(i) for i in range(n)]
        X = rng.normal(shift * label, 1, (n, 4))
        np.savez(run / "features" / f"{name}.npz", keys=np.array(keys), L23_prompt=X.astype(np.float16),
                 L23_slots=X.astype(np.float16))
        lines += [{"set": name, "key": k, "label": label, "group": f"{group_prefix}:{k}"} for k in keys]
    (run / "scores.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")


def test_fit_threshold_and_gates(tmp_path):
    rng = np.random.default_rng(0)
    train = {n: (1 if n in ("pg_adv", "pg_hb_dija", "pg_tmpl_harm") else 0, 60, n) for n in fit_probe.TRAIN_SETS}
    # templated copies share the base request group -> same split side
    train["pg_tmpl_harm"] = (1, 60, "pg_adv")
    test = {n: (1 if n in ("dija_jbb", "jbb_harmful", "xstest_unsafe") else 0, 40, n)
            for n in fit_probe.TEST + fit_probe.HELD_OUT}
    _write_run(tmp_path / "train", train, rng, 4.0)
    _write_run(tmp_path / "test", test, rng, 4.0)
    fn, t, rep, gates = fit_probe.fit(tmp_path / "train", tmp_path / "test")
    assert rep["calib_auroc"] > 0.95
    assert rep["test_inputs"]["dija_jbb"]["gate_rate"] > 0.9
    assert rep["test_inputs"]["gsm8k_5shot"]["gate_rate"] < 0.1
    assert len(gates["dija_jbb"]) == 40
    X = np.load(tmp_path / "test" / "features" / "dija_jbb.npz")["L23_prompt"].astype(np.float64)
    assert np.allclose(stats.logreg_score(fn.params, X), fn(X))


def test_split_keeps_groups_together():
    groups = [f"adv:{i}" for i in range(50)] * 2
    cal = fit_probe.split_groups(groups)
    assert 5 < len(cal) < 30 and cal == fit_probe.split_groups(list(reversed(groups)))

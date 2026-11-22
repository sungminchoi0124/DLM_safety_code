import numpy as np
from scipy.stats import spearmanr

from commit_time.metrics import behavioural_commit, commit_step, hesitation, item_metrics, text_commit_step
from commit_time.probe import Traj, fit_probe


def test_worked_example_from_the_spec():
    # refusal item; tau = 0.5 -> hesitation 4 forwards, commit at forward 4 (the 5th)
    d = np.array([+0.3, -0.1, -0.4, +0.2, -0.9, -1.5, -2.1, -3.0, -4.2])
    s, cens = commit_step(d, y=0, tau=0.5)
    assert (s, cens) == (4, False)
    assert hesitation(d, 0.5)["n_hes"] == 4
    assert commit_step(d, y=1, tau=0.5) == (9, True)  # never commits to the wrong label


def test_commit_requires_staying_out_of_the_band():
    d = np.array([-1.0, -1.0, -0.2, -1.0, -1.0])
    assert commit_step(d, 0, 0.5)[0] == 3


def test_text_commit_and_item_metrics():
    lead = np.array([0, 1, 2, 4, 4, 4])
    assert text_commit_step(lead, 4) == 3
    tr = {"n_tracked": 10, "n_masks": np.array([10, 8, 6, 3, 1, 0]), "n_commit": np.array([2, 2, 3, 2, 1]),
          "n_remask": np.zeros(5, int), "lead_unmasked": lead, "lead_k": 4}
    row = item_metrics(np.array([0.1, -0.2, -0.8, -1.0, -1.2]), 0, 0.5, tr)
    assert row["s_star"] == 2 and row["tokens_at_s_star"] == 4 and row["max_bundle_before"] == 2
    assert row["s_text"] == 3 and row["latent_lead"] == 1


def test_behavioural_commit():
    assert behavioural_commit([0, 4, 8, 16], [0.5, 0.6, 1.0, 1.0]) == (8, False)
    assert behavioural_commit([0, 4, 8], [0.5, 1.0, 0.7]) == (9, True)


def test_probe_recovers_planted_commit_times():
    rng = np.random.default_rng(0)
    trajs, ks = [], []
    T, d = 40, 16
    for i in range(80):
        y = i % 2
        k = int(rng.integers(5, 30))
        H = rng.normal(0, 1, (T, 2, d)).astype(np.float16)
        ramp = 1 / (1 + np.exp(-(np.arange(T) - k)))
        H[:, 1, 0] += ((2 * y - 1) * 4 * ramp).astype(np.float16)
        trajs.append(Traj(run="syn", item=i, y=y, H=H, layer_names=["b7", "final"], main_end=T))
        ks.append(k)
    probe, rep = fit_probe(trajs, mode="all_steps", n_folds=4)
    assert probe.layer == "final"
    assert rep["selected"]["auroc@1"] > 0.95
    s = [commit_step(probe.margins(t.H), t.y, probe.tau)[0] for t in trajs]
    assert spearmanr(s, ks).correlation > 0.7

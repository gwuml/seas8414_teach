"""Unit tests for the honest-decision layer (self-contained; synthetic data)."""
import numpy as np
import pandas as pd

import seas8414_teach as st


def _frame():
    rng = np.random.default_rng(0)
    n = 400
    outcome = (rng.random(n) < 0.2).astype(int)
    # A "good" score correlates with the outcome; a "bad" score is noise.
    good = outcome * 3 + rng.random(n)
    bad = rng.random(n)
    return pd.DataFrame({"outcome": outcome, "good": good, "bad": bad})


def test_argmax_picks_the_better_queue():
    df = _frame()
    d = st.decide_by_argmax(df, proxy="outcome", queues={"good": "good", "bad": "bad"})
    assert d.winner == "good"
    assert d.utilities["good"] > d.utilities["bad"]
    # receipt is internally consistent: recommended == argmax
    rec = d.receipt
    assert rec["recommended_queue"] == "good"
    assert rec["good_queue_utility"] >= rec["bad_queue_utility"]


def test_decision_never_hardcodes_winner():
    # If the "bad" column is actually the good one, the winner flips — no hard-coding.
    df = _frame()
    d = st.decide_by_argmax(df, proxy="outcome", queues={"a": "bad", "b": "good"})
    assert d.winner == "b"


def test_proxy_independence_flags_tautology():
    # The independence check groups by exact score value (matching the full edition), so it needs
    # a score with repeated levels — realistic for review rules that award discrete points.
    rng = np.random.default_rng(2)
    n = 600
    score = rng.integers(0, 6, n).astype(float)          # 6 discrete levels → many collisions
    df = pd.DataFrame({"score": score})
    # Tautological proxy: a pure function of the score → every level is pure → NOT independent.
    df["taut"] = (df["score"] >= 3).astype(int)
    independent, _ = st.proxy_independence(df, proxy="taut", score="score")
    assert independent is False
    # Independent proxy: outcome varies within score levels (mixed) → independent, low corr.
    df["indep"] = (rng.random(n) < 0.25).astype(int)
    ind2, corr2 = st.proxy_independence(df, proxy="indep", score="score")
    assert ind2 is True
    assert abs(corr2) < 0.5


def test_paired_bootstrap_gate_shapes():
    rng = np.random.default_rng(1)
    labels = (rng.random(300) < 0.3).astype(int)
    advanced = labels * 0.5 + rng.random(300)      # better
    baseline = rng.random(300)                      # chance
    gate = st.paired_bootstrap_gate(advanced, baseline, labels, n=200, seed=1)
    assert {"median", "p10", "p90", "adopt"} <= set(gate)
    assert gate["p10"] <= gate["median"] <= gate["p90"]
    assert isinstance(gate["adopt"], bool)

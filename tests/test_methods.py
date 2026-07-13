"""Unit tests for the method wrappers (self-contained; synthetic data)."""
import numpy as np
import pandas as pd

import seas8414_teach as st


def test_markov_surprise_no_zero_blowups_and_determinism():
    seqs = {
        "a": ["connect", "auth-fail", "auth-fail", "close"],
        "b": ["connect", "auth-success", "command", "download", "close"],
        "c": ["connect"],                    # single state → 0.0, no blow-up
    }
    s1 = st.markov_surprise(seqs)
    s2 = st.markov_surprise(seqs)
    assert list(s1.index) == ["a", "b", "c"]
    assert s1["c"] == 0.0
    assert np.isfinite(s1.to_numpy()).all()
    pd.testing.assert_series_equal(s1, s2)   # deterministic (no RNG)


def test_iforest_rarity_deterministic_and_aligned():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((120, 4)), columns=list("wxyz"))
    df.index = [f"s{i}" for i in range(120)]
    r1 = st.iforest_rarity(df, list("wxyz"), seed=8414)
    r2 = st.iforest_rarity(df, list("wxyz"), seed=8414)
    assert list(r1.index) == list(df.index)
    pd.testing.assert_series_equal(r1, r2)   # same seed → identical


def test_kmeans_select_returns_valid_k():
    rng = np.random.default_rng(1)
    # two clear blobs
    X = np.vstack([rng.normal(0, 0.3, (80, 3)), rng.normal(5, 0.3, (80, 3))])
    df = pd.DataFrame(X, columns=list("abc"))
    k, labels, sils = st.kmeans_select(df, list("abc"), k_range=range(2, 6), seed=1)
    assert 2 <= k <= 5
    assert len(labels) == len(df)
    assert set(sils) == {2, 3, 4, 5}

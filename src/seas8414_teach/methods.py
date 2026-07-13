"""Thin, readable wrappers over scikit-learn.

Each function hides the fit/transform wiring but keeps the knobs (features, k, contamination,
seed) as explicit arguments, so the notebook cell shows *what* method and *which* parameters —
the lesson — without the boilerplate. Results are deterministic under the shared seed and match
the full notebooks.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

SEED = 8414

# The Cowrie event-state alphabet (order matters for the transition matrix).
COWRIE_STATES = (
    "connect", "fingerprint", "auth-fail", "auth-success",
    "command", "download", "close", "other",
)


def markov_surprise(
    sequences: Mapping[str, Sequence[str]],
    *,
    states: Sequence[str] = COWRIE_STATES,
    laplace: float = 1.0,
):
    """Mean per-transition surprise (−log P) for each session under a smoothed first-order model.

    ``sequences`` maps session id → list of event states. Laplace smoothing (default 1.0) means no
    zero-probability blow-ups. Returns a pandas Series indexed by session id.
    """
    import numpy as np
    import pandas as pd

    index = {state: i for i, state in enumerate(states)}
    counts = np.full((len(states), len(states)), laplace, dtype=float)
    for seq in sequences.values():
        for left, right in zip(seq[:-1], seq[1:]):
            counts[index[left], index[right]] += 1
    probability = counts / counts.sum(axis=1, keepdims=True)

    def surprise(seq: Sequence[str]) -> float:
        if len(seq) < 2:
            return 0.0
        vals = [
            -math.log(probability[index[left], index[right]])
            for left, right in zip(seq[:-1], seq[1:])
        ]
        return float(np.mean(vals))

    return pd.Series({sid: surprise(seq) for sid, seq in sequences.items()})


def _feature_matrix(frame, features: Iterable[str], *, log1p: bool, scale: bool):
    import numpy as np
    from sklearn.preprocessing import RobustScaler

    X = frame[list(features)].to_numpy(dtype=float)
    if log1p:
        X = np.log1p(X)
    if scale:
        X = RobustScaler().fit_transform(X)
    return X


def iforest_rarity(
    frame,
    features: Iterable[str],
    *,
    n_estimators: int = 240,
    contamination: str | float = "auto",
    max_features: float = 1.0,
    seed: int = SEED,
    log1p: bool = True,
    scale: bool = True,
):
    """Isolation Forest multivariate rarity as a Series aligned to ``frame`` (higher = rarer).

    Returns the negated ``score_samples`` (the notebooks' ``reference_anomaly``). Features are
    log1p-then-RobustScaler-transformed by default, matching the full edition.
    """
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    X = _feature_matrix(frame, features, log1p=log1p, scale=scale)
    model = IsolationForest(
        n_estimators=n_estimators, contamination=contamination, max_samples="auto",
        max_features=max_features, random_state=seed, n_jobs=-1,
    ).fit(X)
    return pd.Series(-model.score_samples(X), index=frame.index)


def kmeans_select(
    frame,
    features: Iterable[str],
    *,
    k_range: Iterable[int] = range(2, 7),
    seed: int = SEED,
    sample: int = 4000,
    log1p: bool = True,
    scale: bool = True,
):
    """Select k by silhouette over ``k_range`` and return ``(selected_k, labels, silhouettes)``.

    Silhouette is scored on a seeded subsample (default 4000) while KMeans fits on all rows —
    matching the full edition.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = _feature_matrix(frame, features, log1p=log1p, scale=scale)
    idx = np.random.default_rng(seed).choice(
        len(X), size=min(sample, len(X)), replace=False
    )
    silhouettes: dict[int, float] = {}
    models: dict[int, Any] = {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(X)
        silhouettes[k] = float(silhouette_score(X[idx], model.labels_[idx]))
        models[k] = model
    selected = max(silhouettes, key=silhouettes.get)
    return selected, models[selected].labels_, silhouettes

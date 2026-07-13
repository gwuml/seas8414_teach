"""KEV triage helpers for the phase 9 project 1 teach edition.

Faithful ports of the transparent full edition's *plumbing* only. The analytics (baseline rule,
screening bands, the preregistered ablation) stay visible in the notebook cells; this module hides
two mechanical pieces:

* :func:`catalog_version` — the CISA catalog version string, needed to compute the reference date
  for ``days_in_catalog``. ``seas8414_teach.load_kev`` returns the tidy DataFrame but drops the
  top-level ``catalogVersion``; we re-read the same cached bytes (via the shared release-lock cache)
  so the two loaders agree byte-for-byte.
* :func:`nmf_topics` — the TF-IDF → NMF topic model plus the cross-seed stability number. Parameters
  match the full edition exactly so the derived topic weights (which feed the anomaly screen and the
  ``68``-record ablation) are identical.

Nothing here re-implements a decision; every knob the lesson cares about is either an explicit
argument or stays in the notebook cell.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .. import provenance as _prov

SEED = 8414
PROJECT_ID = "phase09-01-kev-triage"
KEV_FILENAME = "known_exploited_vulnerabilities.json"


def _load_catalog() -> dict[str, Any]:
    """Return the parsed CISA KEV catalog from the same verified cache ``load_kev`` uses."""
    located = _prov._find_release_lock()
    sha = None
    if located is not None:
        _, index = located
        entry = index.get((PROJECT_ID, KEV_FILENAME))
        sha = entry["sha256"] if entry else None
    raw, _ = _prov.fetch(
        PROJECT_ID, KEV_FILENAME,
        expected_sha256=sha, max_bytes=3_000_000,
        url=(
            "https://www.cisa.gov/sites/default/files/feeds/"
            "known_exploited_vulnerabilities.json"
        ),
    )
    return json.loads(raw.decode("utf-8"))


def catalog_version() -> str:
    """The pinned CISA catalog version string (e.g. ``2026.07.10``).

    Used by the notebook to derive the reference date for ``days_in_catalog`` — the same value the
    full edition reads from ``catalog['catalogVersion']``.
    """
    return _load_catalog()["catalogVersion"]


@dataclass
class TopicModel:
    """Result of the NMF topic model: the load-bearing weight matrix plus inspectable stability.

    ``weights`` is the ``(n_docs, k)`` topic-weight matrix that feeds the anomaly screen. ``labels``
    is its per-document argmax. ``terms`` maps topic index → its top feature terms. ``seed_ari`` is
    the cross-seed adjusted Rand index (topic stability); ``component_sensitivity`` reports the
    5-vs-6 and 6-vs-7 component-count agreement.
    """

    weights: Any
    labels: Any
    terms: dict[int, list[str]]
    seed_ari: float
    component_sensitivity: dict[str, float]


def nmf_topics(
    text: Iterable[str],
    *,
    k: int = 6,
    seed: int = SEED,
    top_terms: int = 8,
    sensitivity_components: tuple[int, ...] = (5, 6, 7),
) -> TopicModel:
    """TF-IDF → NMF topic model over ``text`` with a cross-seed stability number.

    The primary model is ``NMF(k, init='nndsvda', random_state=seed, max_iter=500, l1_ratio=0.1)``;
    the stability probe refits ``NMF(k, init='nndsvda', random_state=seed+1, max_iter=500)`` (no
    ``l1_ratio``) and compares argmax topic assignments by adjusted Rand index. Both the TF-IDF
    vectorizer and the model parameters mirror the transparent full edition, so ``weights`` matches
    bit-for-bit and ``seed_ari`` reproduces the target ``1.0``.
    """
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score

    documents = list(text)
    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=2, max_df=0.96,
        ngram_range=(1, 2), max_features=2500, sublinear_tf=True,
    )
    X_text = vectorizer.fit_transform(documents)

    labels_by_k: dict[int, Any] = {}
    weights = None
    model = None
    for components in sensitivity_components:
        probe = NMF(
            n_components=components, init="nndsvda", random_state=seed,
            max_iter=500, l1_ratio=0.1,
        )
        probe_weights = probe.fit_transform(X_text)
        labels_by_k[components] = probe_weights.argmax(axis=1)
        if components == k:
            model, weights = probe, probe_weights
    if model is None:  # k not covered by the sensitivity sweep — fit it directly
        model = NMF(
            n_components=k, init="nndsvda", random_state=seed,
            max_iter=500, l1_ratio=0.1,
        )
        weights = model.fit_transform(X_text)
        labels_by_k[k] = weights.argmax(axis=1)

    second_model = NMF(
        n_components=k, init="nndsvda", random_state=seed + 1, max_iter=500,
    )
    second_weights = second_model.fit_transform(X_text)
    seed_ari = float(
        adjusted_rand_score(weights.argmax(axis=1), second_weights.argmax(axis=1))
    )

    ordered = sorted(sensitivity_components)
    component_sensitivity = {
        f"{a}_vs_{b}": float(adjusted_rand_score(labels_by_k[a], labels_by_k[b]))
        for a, b in zip(ordered[:-1], ordered[1:])
        if a in labels_by_k and b in labels_by_k
    }

    feature_names = np.array(vectorizer.get_feature_names_out())
    terms = {
        index: feature_names[row.argsort()[-top_terms:][::-1]].tolist()
        for index, row in enumerate(model.components_)
    }
    return TopicModel(
        weights=weights,
        labels=weights.argmax(axis=1),
        terms=terms,
        seed_ari=seed_ari,
        component_sensitivity=component_sensitivity,
    )

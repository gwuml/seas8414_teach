"""Phase 10 Project 3 loader + reconstruction methods — ATT&CK-informed ICS deception coverage.

The transparent full edition inlines ~200 lines to parse the pinned ATT&CK for ICS STIX bundle,
build the technique-to-asset incidence matrix, and run a leakage-free known-edge reconstruction
with degree-matched negatives across 30 preregistered splits. This module ports that analytics
faithfully so the teach edition produces the identical load-bearing decision:

    adopt the popularity baseline unless the spectral model shows a positive degree-matched AP
    difference at the 10th percentile across 30 splits.

``load_attack_ics()`` hides the download / SHA-verify / release-lock plumbing behind the shared
:func:`seas8414_teach.provenance.fetch` facade and returns an inspectable :class:`AttackICS`
record. ``degree_matched_negatives`` and ``reconstruction_eval`` are the leakage-free evaluation
helpers; the notebook keeps the incidence, the fair decision, and the greedy-vs-exact coverage on
screen.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ..provenance import Provenance, _find_release_lock, fetch

PROJECT = "phase10-03-attack-deception-coverage"
FILENAME = "ics-attack.json"
ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "a6c366439edee3a87b79cf90dc0b93f5d7975956/"
    "ics-attack/ics-attack.json"
)
ATTACK_SHA = "a91f659d6d03095e84089630b098edb2ed9d5cd5b1ea41369b27846cd32f2a43"


def external_id(item: dict) -> str:
    """Return the mitre-attack external id (e.g. ``T0800``) for a STIX object."""
    for reference in item.get("external_references", []):
        if reference.get("source_name") == "mitre-attack":
            return reference.get("external_id", item["id"])
    return item["id"]


@dataclass
class AttackICS:
    """Parsed ATT&CK for ICS snapshot — the observed technique-to-asset targeting facts.

    ``incidence`` is a ``(techniques × assets)`` binary matrix of recorded ``targets``
    relationships. ``technique_ids`` / ``asset_ids`` are the sorted STIX ids that index its rows /
    columns; ``technique_index`` / ``asset_index`` map an id back to its position. ``techniques``
    and ``assets`` retain the full STIX objects (for names + external ids). ``technique_weights``
    is ``1 + log1p(use_count)`` per technique, from adversary/campaign ``uses`` evidence.
    """

    techniques: dict[str, dict]
    assets: dict[str, dict]
    technique_ids: list[str]
    asset_ids: list[str]
    technique_index: dict[str, int]
    asset_index: dict[str, int]
    incidence: Any             # np.ndarray (float32) shape (n_techniques, n_assets)
    technique_weights: Any     # np.ndarray shape (n_techniques,)
    target_edges: list[tuple[str, str]]
    provenance: Provenance

    def external_id(self, item: dict) -> str:
        return external_id(item)


def load_attack_ics() -> AttackICS:
    """Parse the pinned ATT&CK for ICS STIX bundle into observed targeting facts (+ provenance).

    Ports the full edition's STIX parse and incidence build exactly: active (non-revoked,
    non-deprecated) ``attack-pattern`` techniques and ``x-mitre-asset`` assets, the observed
    ``targets`` edges between them, and per-technique ``uses`` weights from malware/campaign/
    intrusion-set evidence.
    """
    import numpy as np

    # SHA is read from the release lock so it stays a single source of truth (falls back to the
    # inlined constant when the lock is absent, which matches the full edition's expected value).
    located = _find_release_lock()
    sha = ATTACK_SHA
    if located is not None:
        _, index = located
        entry = index.get((PROJECT, FILENAME))
        if entry is not None:
            sha = entry["sha256"]
    raw, prov = fetch(
        PROJECT, FILENAME,
        expected_sha256=sha, max_bytes=5_000_000, url=ATTACK_URL,
    )

    bundle = json.loads(raw.decode("utf-8"))
    assert bundle.get("type") == "bundle" and len(bundle.get("objects", [])) == 2_174
    objects = {item["id"]: item for item in bundle["objects"] if "id" in item}

    def active(item: dict) -> bool:
        return not item.get("revoked", False) and not item.get("x_mitre_deprecated", False)

    techniques = {
        item_id: item for item_id, item in objects.items()
        if item.get("type") == "attack-pattern" and active(item)
    }
    assets = {
        item_id: item for item_id, item in objects.items()
        if item.get("type") == "x-mitre-asset" and active(item)
    }
    relationships = [
        item for item in bundle["objects"]
        if item.get("type") == "relationship" and active(item)
    ]
    target_edges = sorted({
        (item["source_ref"], item["target_ref"])
        for item in relationships
        if item.get("relationship_type") == "targets"
        and item.get("source_ref") in techniques
        and item.get("target_ref") in assets
    })
    assert len(techniques) >= 90 and len(assets) == 18 and len(target_edges) >= 800

    technique_ids = sorted(techniques)
    asset_ids = sorted(assets)
    technique_index = {item_id: idx for idx, item_id in enumerate(technique_ids)}
    asset_index = {item_id: idx for idx, item_id in enumerate(asset_ids)}
    incidence = np.zeros((len(technique_ids), len(asset_ids)), dtype=np.float32)
    for technique_id, asset_id in target_edges:
        incidence[technique_index[technique_id], asset_index[asset_id]] = 1

    use_counts = {technique_id: 0 for technique_id in technique_ids}
    for item in relationships:
        if item.get("relationship_type") == "uses" and item.get("target_ref") in use_counts:
            source_type = objects.get(item.get("source_ref"), {}).get("type")
            if source_type in {"malware", "campaign", "intrusion-set"}:
                use_counts[item["target_ref"]] += 1
    technique_weights = np.array([
        1.0 + math.log1p(use_counts[technique_id]) for technique_id in technique_ids
    ])

    return AttackICS(
        techniques=techniques,
        assets=assets,
        technique_ids=technique_ids,
        asset_ids=asset_ids,
        technique_index=technique_index,
        asset_index=asset_index,
        incidence=incidence,
        technique_weights=technique_weights,
        target_edges=target_edges,
        provenance=prov,
    )


def degree_matched_negatives(positives, absent_pairs, matrix, match_rng):
    """One non-edge per held-out positive, nearest by (row_deg, col_deg) L1 distance on the
    training matrix, sampled without replacement.

    Highest degree-product positives are matched first so scarce high-degree holes are not
    consumed by low-degree positives. It never reverts to a uniform draw. Every negative is a true
    non-edge because ``absent_pairs`` is taken from ``incidence == 0``, disjoint from held-out
    ones. Ported verbatim from the full edition so the fair evaluation is identical.
    """
    import numpy as np

    row_deg = matrix.sum(axis=1)
    col_deg = matrix.sum(axis=0)
    absent = np.asarray(absent_pairs)
    absent_row_deg = row_deg[absent[:, 0]]
    absent_col_deg = col_deg[absent[:, 1]]
    used = np.zeros(len(absent), dtype=bool)
    positive_array = np.asarray(positives)
    hardness = row_deg[positive_array[:, 0]] * col_deg[positive_array[:, 1]]
    jitter = match_rng.random(len(positives)) * 1e-6
    order = np.argsort(-(hardness + jitter))
    matched = [None] * len(positives)
    for index in order:
        target_row_deg = row_deg[positives[index][0]]
        target_col_deg = col_deg[positives[index][1]]
        distance = (
            np.abs(absent_row_deg - target_row_deg)
            + np.abs(absent_col_deg - target_col_deg)
        )
        distance[used] = np.inf
        choice = int(np.argmin(distance))
        used[choice] = True
        matched[index] = (int(absent[choice, 0]), int(absent[choice, 1]))
    return matched


def _reconstruction_split_metrics(
    split_seed, incidence, technique_index, asset_index, target_edges, target_holdout, rank
):
    """One preregistered reconstruction split: mask ~20% of edges, fit spectral + popularity on
    the masked training matrix (leakage-free), score both under uniform and degree-matched
    negatives, and report the spectral-minus-popularity AP gap.
    """
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import average_precision_score, roc_auc_score

    split_rng = np.random.default_rng(split_seed)
    shuffled = target_edges.copy()
    split_rng.shuffle(shuffled)
    split_train = incidence.copy()
    split_held = []
    for technique_id, asset_id in shuffled:
        row = technique_index[technique_id]
        column = asset_index[asset_id]
        if split_train[row].sum() > 1 and split_train[:, column].sum() > 1:
            split_train[row, column] = 0
            split_held.append((row, column))
            if len(split_held) >= target_holdout:
                break
    split_absent = list(zip(*np.where(incidence == 0), strict=True))
    chosen = split_rng.choice(len(split_absent), size=len(split_held), replace=False)
    split_negative = [split_absent[index] for index in chosen]
    split_match_rng = np.random.default_rng(split_seed + 7919)
    split_matched = degree_matched_negatives(
        split_held, split_absent, split_train, split_match_rng
    )
    split_svd = TruncatedSVD(n_components=rank, random_state=split_seed)
    split_left = split_svd.fit_transform(split_train)
    split_scores = split_left @ split_svd.components_
    split_popularity = (
        split_train.sum(axis=1)[:, None] + 1
    ) * (split_train.sum(axis=0)[None, :] + 1)

    def score(negatives):
        pairs = split_held + negatives
        labels = np.array([1] * len(split_held) + [0] * len(negatives))
        spectral_values = np.array([split_scores[row, col] for row, col in pairs])
        popularity_values = np.array([split_popularity[row, col] for row, col in pairs])
        return {
            "spectral_ap": average_precision_score(labels, spectral_values),
            "popularity_ap": average_precision_score(labels, popularity_values),
            "spectral_roc_auc": roc_auc_score(labels, spectral_values),
            "popularity_roc_auc": roc_auc_score(labels, popularity_values),
        }

    uniform = score(split_negative)
    matched = score(split_matched)
    return {
        "spectral_ap": uniform["spectral_ap"],
        "popularity_ap": uniform["popularity_ap"],
        "ap_difference": uniform["spectral_ap"] - uniform["popularity_ap"],
        "spectral_ap_degree_matched": matched["spectral_ap"],
        "popularity_ap_degree_matched": matched["popularity_ap"],
        "spectral_roc_auc_degree_matched": matched["spectral_roc_auc"],
        "popularity_roc_auc_degree_matched": matched["popularity_roc_auc"],
        "ap_difference_degree_matched": (
            matched["spectral_ap"] - matched["popularity_ap"]
        ),
    }


def reconstruction_eval(attack: AttackICS, *, seed: int = 8414, splits: int = 30) -> dict:
    """Leakage-free known-edge reconstruction: spectral TruncatedSVD vs popularity baseline.

    Masks ~20% of the observed edges as held-out positives (only edges whose endpoints keep
    degree > 1 after removal, so the training matrix never fully isolates a row or column), fits
    both models on the masked training matrix, and scores held-out positives against DEGREE-MATCHED
    negatives across ``splits`` preregistered splits. The adoption decision follows the FAIR rule:
    retain the popularity baseline unless the spectral model's degree-matched AP difference is
    positive at the 10th percentile.

    Returns a dict with the split table (``split_sensitivity``) and the load-bearing scalars:
    ``adopted_reconstruction_model``, ``fair_spectral_minus_popularity_ap_p10``,
    ``spectral_ap_degree_matched_median``, ``popularity_ap_degree_matched_median``.
    """
    import numpy as np
    import pandas as pd

    incidence = attack.incidence
    technique_index = attack.technique_index
    asset_index = attack.asset_index
    target_edges = attack.target_edges
    target_holdout = round(0.20 * len(target_edges))
    rank = min(12, min(incidence.shape) - 1)

    split_sensitivity = pd.DataFrame([
        _reconstruction_split_metrics(
            seed + repeat + 1, incidence, technique_index, asset_index,
            target_edges, target_holdout, rank,
        )
        for repeat in range(splits)
    ])

    fair_p10 = float(split_sensitivity["ap_difference_degree_matched"].quantile(0.10))
    adopted = "spectral" if fair_p10 > 0 else "popularity-baseline"
    return {
        "split_sensitivity": split_sensitivity,
        "adopted_reconstruction_model": adopted,
        "fair_spectral_minus_popularity_ap_p10": fair_p10,
        "fair_spectral_minus_popularity_ap_median": float(
            split_sensitivity["ap_difference_degree_matched"].median()
        ),
        "spectral_minus_popularity_ap_p10": float(
            split_sensitivity["ap_difference"].quantile(0.10)
        ),
        "spectral_minus_popularity_ap_median": float(
            split_sensitivity["ap_difference"].median()
        ),
        "spectral_ap_degree_matched_median": float(
            split_sensitivity["spectral_ap_degree_matched"].median()
        ),
        "popularity_ap_degree_matched_median": float(
            split_sensitivity["popularity_ap_degree_matched"].median()
        ),
        "spectral_roc_auc_degree_matched_median": float(
            split_sensitivity["spectral_roc_auc_degree_matched"].median()
        ),
        "popularity_roc_auc_degree_matched_median": float(
            split_sensitivity["popularity_roc_auc_degree_matched"].median()
        ),
        "held_out_target": target_holdout,
        "svd_rank": rank,
        "splits": splits,
    }

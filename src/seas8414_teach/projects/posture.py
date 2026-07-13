"""Phase 9 Project 3 loader + stability gates — Open-Source Repository Posture Archetypes.

The transparent full edition inlines the acquisition of 36 OpenSSF Scorecard responses, the
missingness handling (negative-sentinel → NaN, median impute + one missingness indicator per
check), and the five-gate stability contract that *rejects* the tempting GMM archetypes. This
module ports that analytics faithfully so the teach edition produces the identical load-bearing
decision:

    the GMM archetypes are REJECTED (cluster_decision == "REJECT") — the BIC optimum is on the
    search boundary, the resampled cluster ARI is unstable, near-hard membership on ~3.6
    points/component signals overfit, and the observation window is exceeded.

``load_scorecard()`` hides the download / SHA-verify / release-lock plumbing behind the shared
:func:`seas8414_teach.provenance.fetch` facade and returns an inspectable :class:`Scorecard`
record (the check matrix, the missingness indicators, the posture frame, and the observation
span). ``fit_reference`` fits the missingness→scale→PCA→GMM pipeline once; ``stability_gates``
runs the bootstrap / leave-one-out / silhouette plumbing and returns the decision, the rejection
reasons, and the load-bearing metrics. The notebook keeps the *decisions* on screen — the
missingness view, the BIC model-selection sweep, the five gates, and the evidence-request queue
that never emits a supplier verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..provenance import Provenance, _find_release_lock, fetch

PROJECT = "phase09-03-repository-posture"

# The 36 preregistered repositories, in the full edition's declared order. The Scorecard API stores
# each response under ``owner__repo.json`` (slash → double underscore); every one of these is an
# entry in the release-cache lock, so the SHA is read from there (single source of truth).
REPOSITORIES = [
    "kubernetes/kubernetes", "prometheus/prometheus", "grafana/grafana",
    "hashicorp/terraform", "ansible/ansible", "docker/cli", "moby/moby",
    "curl/curl", "openssl/openssl", "numpy/numpy", "pandas-dev/pandas",
    "scikit-learn/scikit-learn", "pytorch/pytorch", "tensorflow/tensorflow",
    "django/django", "pallets/flask", "tiangolo/fastapi", "expressjs/express",
    "nodejs/node", "rust-lang/rust", "golang/go", "apache/kafka",
    "apache/spark", "apache/logging-log4j2", "eclipse/paho.mqtt.python",
    "caddyserver/caddy", "traefik/traefik", "home-assistant/core",
    "espressif/esp-idf", "zephyrproject-rtos/zephyr", "systemd/systemd",
    "torvalds/linux", "grpc/grpc", "envoyproxy/envoy", "etcd-io/etcd",
    "helm/helm",
]
assert len(REPOSITORIES) == 36 and len(set(REPOSITORIES)) == 36


@dataclass
class Scorecard:
    """Parsed OpenSSF Scorecard cohort — the observed check facts plus explicit missingness.

    ``posture`` is the per-repository frame (indexed by repository, sorted) carrying the overall
    score, the commit / date / engine-commit provenance, and one column per Scorecard check.
    ``check_matrix`` is the ``(36 × n_checks)`` float matrix with negative sentinels masked to NaN;
    ``missing_matrix`` is its per-check missingness indicator (``__missing`` suffix). ``check_names``
    is the sorted check list. ``observation_span_days`` and ``engine_commits`` summarize the drift
    across the mutable observations; ``negative_sentinel_count`` counts the sentinels treated as
    unavailable. ``provenance`` is the last acquired response's provenance (all share the lock).
    """

    posture: Any            # pd.DataFrame indexed by repository
    check_matrix: Any       # pd.DataFrame (36 × n_checks), negative sentinels → NaN
    missing_matrix: Any     # pd.DataFrame (36 × n_checks) int indicators, __missing suffix
    check_names: list[str]
    observation_span_days: int
    engine_commits: int
    negative_sentinel_count: int
    provenance: Provenance


def _sha_for(index: dict | None, filename: str) -> str | None:
    if index is None:
        return None
    entry = index.get((PROJECT, filename))
    return entry["sha256"] if entry else None


def load_scorecard() -> Scorecard:
    """Acquire the 36 pinned OpenSSF Scorecard responses and build the posture / check matrix.

    Ports the full edition's parse + missingness handling EXACTLY: read each ``owner__repo.json``
    response (SHA-verified via the release lock), collect the union of check names, mask negative
    sentinel scores to NaN, and record one missingness indicator per check. No imputation happens
    here — the notebook keeps that visible. Enumerates all 36 files for the project from the
    release lock so the cohort is exactly the preregistered one.
    """
    import numpy as np
    import pandas as pd

    located = _find_release_lock()
    index = located[1] if located is not None else None

    responses = []
    last_prov: Provenance | None = None
    for repository in REPOSITORIES:
        filename = repository.replace("/", "__") + ".json"
        raw, prov = fetch(
            PROJECT, filename,
            expected_sha256=_sha_for(index, filename),
            max_bytes=1_000_000,
            url=f"https://api.securityscorecards.dev/projects/github.com/{repository}",
        )
        payload = json.loads(raw.decode("utf-8"))
        assert {"date", "repo", "scorecard", "score", "checks"}.issubset(payload)
        responses.append(payload)
        last_prov = prov
    assert len(responses) == 36, (
        f"The preregistered cohort requires all 36 responses; got {len(responses)}"
    )

    check_names = sorted(
        {check["name"] for item in responses for check in item["checks"]}
    )
    rows = []
    for item in responses:
        scores = {check["name"]: check.get("score") for check in item["checks"]}
        row = {
            "repository": item["repo"]["name"].removeprefix("github.com/"),
            "repository_commit": item["repo"].get("commit", ""),
            "scorecard_date": item["date"],
            "scorecard_commit": item["scorecard"].get("commit", ""),
            "overall_score": item["score"],
        }
        row.update({name: scores.get(name, np.nan) for name in check_names})
        rows.append(row)
    posture = pd.DataFrame(rows).set_index("repository").sort_index()
    posture["scorecard_timestamp"] = pd.to_datetime(
        posture["scorecard_date"], utc=True, format="mixed"
    )

    check_matrix = posture[check_names].astype(float)
    negative_sentinel_count = int(check_matrix.lt(0).sum().sum())
    check_matrix = check_matrix.mask(check_matrix.lt(0))
    missing_matrix = check_matrix.isna().astype(int).add_suffix("__missing")

    observation_span_days = int(
        (posture["scorecard_timestamp"].max()
         - posture["scorecard_timestamp"].min()).days
    )
    engine_commits = int(posture["scorecard_commit"].nunique())

    return Scorecard(
        posture=posture,
        check_matrix=check_matrix,
        missing_matrix=missing_matrix,
        check_names=check_names,
        observation_span_days=observation_span_days,
        engine_commits=engine_commits,
        negative_sentinel_count=negative_sentinel_count,
        provenance=last_prov,
    )


# --------------------------------------------------------------------------- pipeline

def fit_reference(check_matrix, components: int, seed: int = 8414) -> dict:
    """Fit the missingness → RobustScaler → SVD-PCA → GMM pipeline once (full edition verbatim).

    Median-imputes the check matrix, appends one missingness indicator per check, robustly scales
    the combined block, reduces with an ordinary PCA (a truncated SVD), and fits a diagonal GMM on
    the leading (≤4) principal components. Returns the fitted transformers plus the scaled /
    projected arrays so the notebook and :func:`stability_gates` share one reference fit.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler

    imputer = SimpleImputer(
        strategy="median", add_indicator=False, keep_empty_features=True
    )
    imputed = imputer.fit_transform(check_matrix)
    combined = np.column_stack([imputed, check_matrix.isna().astype(int).to_numpy()])
    scaler = RobustScaler().fit(combined)
    scaled = scaler.transform(combined)
    dimensions = min(6, scaled.shape[1], len(check_matrix) - 1)
    pca = PCA(n_components=dimensions, random_state=seed).fit(scaled)
    projected = pca.transform(scaled)
    gmm = GaussianMixture(
        n_components=components, covariance_type="diag",
        random_state=seed, reg_covar=1e-4, n_init=12,
    ).fit(projected[:, :min(4, dimensions)])
    return {
        "imputer": imputer, "scaler": scaler, "pca": pca, "gmm": gmm,
        "scaled": scaled, "projected": projected,
    }


def predict_reference(model: dict, check_matrix):
    """Predict archetype label + membership posterior for ``check_matrix`` under a reference fit."""
    import numpy as np

    imputed = model["imputer"].transform(check_matrix)
    combined = np.column_stack([imputed, check_matrix.isna().astype(int).to_numpy()])
    scaled = model["scaler"].transform(combined)
    projected = model["pca"].transform(scaled)
    gmm_input = projected[:, :min(4, projected.shape[1])]
    return model["gmm"].predict(gmm_input), model["gmm"].predict_proba(gmm_input)


def stability_gates(
    check_matrix,
    reference: dict,
    base_labels,
    membership,
    *,
    selected_k: int,
    model_search_boundary: bool,
    observation_span_days: int,
    seed: int = 8414,
    bootstrap_repeats: int = 80,
):
    """Run the five-gate stability contract (full edition verbatim) → (decision, reasons, metrics).

    Fits the full pipeline on 80 bootstrap resamples and on every leave-one-repository-out subset,
    comparing each relabeling to ``base_labels`` by adjusted Rand index. Then applies the five gates
    that can REJECT the archetypes:

    1. BIC optimum on the component-search boundary,
    2. bootstrap full-pipeline ARI 10th percentile below 0.60,
    3. leave-one-out full-pipeline ARI 10th percentile below 0.70,
    4. membership confidence implausibly high (median > 0.95) for < 8 points/component
       (near-hard assignments = GMM overfit, not stable separation),
    5. observation dates spanning more than the preregistered 120-day window.

    Returns ``(cluster_decision, cluster_rejection_reasons, metrics)`` where ``metrics`` carries
    ``membership_confidence_median``, ``cluster_silhouette``, ``samples_per_component`` and the ARI
    quantiles. The heavy resampling is the plumbing; the gate thresholds stay a visible lesson.
    """
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    n = len(check_matrix)

    rng = np.random.default_rng(seed)
    bootstrap_cluster_ari = []
    for repeat in range(bootstrap_repeats):
        sampled = rng.integers(0, n, n)
        bootstrap_matrix = check_matrix.iloc[sampled].reset_index(drop=True)
        bootstrap_pipeline = fit_reference(
            bootstrap_matrix, selected_k, seed + repeat + 1
        )
        bootstrap_labels, _ = predict_reference(bootstrap_pipeline, check_matrix)
        bootstrap_cluster_ari.append(
            adjusted_rand_score(base_labels, bootstrap_labels)
        )

    loo_cluster_ari = []
    for left_out in range(n):
        keep = np.arange(n) != left_out
        loo_pipeline = fit_reference(
            check_matrix.iloc[keep], selected_k, seed + 100 + left_out
        )
        loo_labels, _ = predict_reference(loo_pipeline, check_matrix)
        loo_cluster_ari.append(adjusted_rand_score(base_labels, loo_labels))

    membership_median = float(np.median(membership.max(axis=1)))
    samples_per_component = n / selected_k
    X_pca = reference["projected"]
    silhouette_input = X_pca[:, :min(4, X_pca.shape[1])]
    if len(np.unique(base_labels)) >= 2:
        cluster_silhouette = float(silhouette_score(silhouette_input, base_labels))
    else:
        cluster_silhouette = float("nan")

    cluster_rejection_reasons = []
    if model_search_boundary:
        cluster_rejection_reasons.append("BIC optimum is on the search boundary")
    if np.quantile(bootstrap_cluster_ari, 0.10) < 0.60:
        cluster_rejection_reasons.append(
            "bootstrap full-pipeline ARI p10 is below 0.60"
        )
    if np.quantile(loo_cluster_ari, 0.10) < 0.70:
        cluster_rejection_reasons.append(
            "leave-one-out full-pipeline ARI p10 is below 0.70"
        )
    if membership_median > 0.95 and samples_per_component < 8:
        cluster_rejection_reasons.append(
            f"membership confidence implausibly high (median {membership_median:.2f}) "
            f"for k={selected_k} on n={n} "
            f"(~{samples_per_component:.1f} points/component): near-hard assignments "
            f"indicate GMM overfit, not stable separation"
        )
    if observation_span_days > 120:
        cluster_rejection_reasons.append(
            "observation dates span more than the preregistered 120-day window"
        )
    cluster_adopted = not cluster_rejection_reasons
    cluster_decision = "ADOPT" if cluster_adopted else "REJECT"

    metrics = {
        "bootstrap_cluster_ari_median": float(np.median(bootstrap_cluster_ari)),
        "bootstrap_cluster_ari_p10": float(np.quantile(bootstrap_cluster_ari, 0.10)),
        "loo_cluster_ari_median": float(np.median(loo_cluster_ari)),
        "loo_cluster_ari_p10": float(np.quantile(loo_cluster_ari, 0.10)),
        "membership_confidence_median": membership_median,
        "cluster_silhouette": cluster_silhouette,
        "samples_per_component": samples_per_component,
    }
    return cluster_decision, cluster_rejection_reasons, metrics

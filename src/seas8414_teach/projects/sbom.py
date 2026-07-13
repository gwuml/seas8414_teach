"""Phase 9 Project 2 loader + graph-screening method — SBOM dependency blast radius.

The transparent full edition inlines ~200 lines to (a) parse a pinned CycloneDX BOM and its
batched OSV advisory response, (b) build the dependency DAG and a per-node feature table
(blast radius via ``nx.ancestors``, PageRank, betweenness, depth), and (c) run a *leakage-free*
per-fold spectral graph screen against a transparent degree baseline. This module ports that
analytics faithfully so the teach edition produces the identical load-bearing decision:

    retain the degree baseline unless the graph screen shows a positive paired-bootstrap AP
    lower bound AND a higher cross-validated ROC-AUC.

``load_sbom()`` hides the download / SHA-verify / release-lock plumbing behind the shared
:func:`seas8414_teach.provenance.fetch` facade and returns an inspectable :class:`SbomGraph`
record (CycloneDX BOM + OSV records + the built networkx DAG + node table). ``graph_screen_cv``
is the leakage-free evaluation helper; the notebook keeps the DAG, the blast radius, the fair
baseline-vs-graph decision, and the paired-bootstrap gate on screen.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..provenance import Provenance, _find_release_lock, fetch

SEED = 8414
PROJECT = "phase09-02-sbom-graph"
SBOM_FILENAME = "dropwizard-1.3.15-bom.json"
SBOM_URL = (
    "https://raw.githubusercontent.com/CycloneDX/bom-examples/"
    "7d9172d00c88c05a8f1ccb90589d111870fc9d86/"
    "SBOM/dropwizard-1.3.15/bom.json"
)
SBOM_SHA = "e0eb128b9d081444e76d5b71089f94db16d889e37a77ca869e2645a70eb29f4b"
OSV_FILENAME = (
    "osv-querybatch-"
    "86fddc20d94db67ccd774f7407d94b5b27c1ddc65411648a12a00d0cc555ad9d.json"
)
OSV_SHA = "517144eaae305aa77f0f3064cb701df25d30e6b543b416679f4051aac9bcc7d9"


@dataclass
class SbomGraph:
    """Parsed SBOM snapshot — the observed dependency + advisory facts.

    ``bom`` is the raw CycloneDX document. ``graph`` is the dependency DAG (``nx.DiGraph`` of
    ``bom-ref`` nodes with ``name``/``version``/``purl``/``component_type`` attributes and
    ``dependsOn`` edges). ``node_vulns`` maps each node to its sorted OSV advisory ids for the
    pinned version. ``nodes`` is the per-node feature table (blast radius via ``nx.ancestors``,
    PageRank, betweenness, depth, ``vulnerability_count``) indexed by ``bom-ref``. ``adjacency``
    is the undirected CSR adjacency over ``nodes.index`` used by the spectral graph screen.
    """

    bom: dict[str, Any]
    graph: Any                 # nx.DiGraph
    node_vulns: dict[str, list[str]]
    nodes: Any                 # pd.DataFrame indexed by bom-ref
    adjacency: Any             # scipy.sparse CSR array
    provenance: Provenance


def _load_bytes(filename: str, expected_sha: str, url: str | None) -> tuple[bytes, Provenance]:
    """Fetch verified bytes, reading the canonical SHA from the release lock when present."""
    located = _find_release_lock()
    sha = expected_sha
    if located is not None:
        _, index = located
        entry = index.get((PROJECT, filename))
        if entry is not None:
            sha = entry["sha256"]
    return fetch(PROJECT, filename, expected_sha256=sha, max_bytes=5_000_000, url=url)


def load_sbom() -> SbomGraph:
    """Parse the pinned CycloneDX BOM + OSV batch into a dependency DAG (+ node table + provenance).

    Ports the full edition's SBOM/OSV parse and graph build exactly:

    * The CycloneDX components (plus the root ``metadata.component``) become DAG nodes keyed by
      ``bom-ref``; ``dependencies[*].dependsOn`` become edges, with unresolved refs added as
      ``component_type="unresolved"`` placeholder nodes.
    * The OSV ``querybatch`` request is rebuilt in the same iteration order (``purl``-without-
      version + ``version``, ``sort_keys=True``) so ``results`` map back to nodes 1:1.
    * The node table records blast radius (``transitive_dependents`` = ``len(nx.ancestors)``),
      ``transitive_dependencies`` (descendants), min depth from a root, PageRank, betweenness,
      and per-node ``vulnerability_count``.
    """
    import networkx as nx
    import pandas as pd

    sbom_raw, prov = _load_bytes(SBOM_FILENAME, SBOM_SHA, SBOM_URL)
    sbom = json.loads(sbom_raw.decode("utf-8"))
    assert sbom.get("bomFormat") == "CycloneDX"
    assert sbom.get("specVersion") == "1.2"
    assert len(sbom.get("components", [])) == 167
    assert len(sbom.get("dependencies", [])) == 167

    # --- DAG build (full edition cell 8) ---
    root_component = sbom.get("metadata", {}).get("component", {})
    objects = [root_component, *sbom["components"]]
    ref_to_component = {
        item.get("bom-ref"): item for item in objects if item.get("bom-ref")
    }
    graph = nx.DiGraph()
    for ref, item in ref_to_component.items():
        graph.add_node(
            ref, name=item.get("name", ref), version=item.get("version", ""),
            purl=item.get("purl", ""), component_type=item.get("type", "unknown"),
        )
    for dependency in sbom["dependencies"]:
        parent = dependency.get("ref")
        if parent not in graph:
            graph.add_node(parent, name=parent, version="", purl="",
                           component_type="unresolved")
        for child in dependency.get("dependsOn", []):
            if child not in graph:
                graph.add_node(child, name=child, version="", purl="",
                               component_type="unresolved")
            graph.add_edge(parent, child)
    assert nx.is_directed_acyclic_graph(graph)

    # --- OSV query reconstruction (full edition cell 9) ---
    # Rebuild query_nodes/queries in graph iteration order so the cached response maps back 1:1.
    query_nodes, queries = [], []
    for node, attrs in graph.nodes(data=True):
        purl = attrs.get("purl", "")
        version = attrs.get("version", "")
        if purl.startswith("pkg:") and version:
            purl_without_version = purl.rsplit("@", 1)[0]
            query_nodes.append(node)
            queries.append({
                "version": version,
                "package": {"purl": purl_without_version},
            })

    osv_raw, _osv_prov = _load_bytes(OSV_FILENAME, OSV_SHA, None)
    osv_response = json.loads(osv_raw)
    assert len(osv_response.get("results", [])) == len(query_nodes)
    node_vulns: dict[str, list[str]] = {node: [] for node in graph.nodes}
    for node, result in zip(query_nodes, osv_response["results"], strict=True):
        node_vulns[node] = sorted({v["id"] for v in result.get("vulns", [])})

    # --- node feature table (full edition cell 11) ---
    roots = [node for node in graph if graph.in_degree(node) == 0]
    pagerank = nx.pagerank(graph, alpha=0.85)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    node_rows = []
    for node, attrs in graph.nodes(data=True):
        dependents = len(nx.ancestors(graph, node))
        dependencies = len(nx.descendants(graph, node))
        depths = [nx.shortest_path_length(graph, root, node)
                  for root in roots if nx.has_path(graph, root, node)]
        node_rows.append({
            "node": node, "name": attrs.get("name", node),
            "version": attrs.get("version", ""), "purl": attrs.get("purl", ""),
            "direct_dependents": graph.in_degree(node),
            "direct_dependencies": graph.out_degree(node),
            "transitive_dependents": dependents,
            "transitive_dependencies": dependencies,
            "depth": min(depths) if depths else -1,
            "pagerank": pagerank[node], "betweenness": betweenness[node],
            "vulnerability_count": len(node_vulns[node]),
            "vulnerability_ids": node_vulns[node],
        })
    nodes = pd.DataFrame(node_rows).set_index("node")

    adjacency = nx.to_scipy_sparse_array(
        graph.to_undirected(), nodelist=nodes.index.tolist(),
        dtype=float, format="csr",
    )

    return SbomGraph(
        bom=sbom,
        graph=graph,
        node_vulns=node_vulns,
        nodes=nodes,
        adjacency=adjacency,
        provenance=prov,
    )


def graph_screen_cv(nodes, adjacency, y, *, seed: int = SEED) -> tuple[Any, dict[str, float]]:
    """Leakage-free per-fold spectral graph screen vs the transparent degree baseline.

    Ports the full edition's screening loop (cell 13) exactly. Within each stratified fold the
    ``TruncatedSVD`` graph representation is fit on the TRAIN-node-induced adjacency and every
    node is projected over the train columns, so held-out connectivity never informs the
    fold-local embedding; a class-balanced logistic model on ``[base_features | fold_embedding]``
    then produces the out-of-fold ``graph_probability``.

    ``y`` is the binary ``vulnerability_count > 0`` label. Returns
    ``(graph_probability, metrics)`` where ``metrics`` holds ``graph_cv_ap``,
    ``graph_cv_roc_auc``, ``degree_baseline_ap`` and ``degree_baseline_roc_auc``. The degree
    baseline (``transitive_dependents.rank(pct=True)*0.7 + direct_dependents.rank(pct=True)*0.3``)
    is scored here so the notebook's paired-bootstrap gate reuses the identical arms.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_columns = [
        "direct_dependents", "direct_dependencies", "transitive_dependents",
        "transitive_dependencies", "depth", "pagerank", "betweenness",
    ]
    base_features = (
        nodes[base_columns].replace([np.inf, -np.inf], 0).fillna(0).to_numpy()
    )
    y = np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    graph_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        ),
    )
    graph_probability = np.zeros(len(nodes), dtype=float)
    for train_index, test_index in cv.split(base_features, y):
        fold_svd = TruncatedSVD(n_components=6, random_state=seed)
        fold_svd.fit(adjacency[train_index][:, train_index])
        fold_embedding = fold_svd.transform(adjacency[:, train_index])
        fold_X = np.hstack([base_features, fold_embedding])
        fold_model = clone(graph_model)
        fold_model.fit(fold_X[train_index], y[train_index])
        graph_probability[test_index] = fold_model.predict_proba(
            fold_X[test_index]
        )[:, 1]

    baseline_probability = (
        nodes["transitive_dependents"].rank(pct=True) * 0.7
        + nodes["direct_dependents"].rank(pct=True) * 0.3
    ).to_numpy()

    metrics = {
        "graph_cv_ap": float(average_precision_score(y, graph_probability)),
        "graph_cv_roc_auc": float(roc_auc_score(y, graph_probability)),
        "degree_baseline_ap": float(average_precision_score(y, baseline_probability)),
        "degree_baseline_roc_auc": float(roc_auc_score(y, baseline_probability)),
    }
    return graph_probability, metrics

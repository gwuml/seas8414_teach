"""Completion-receipt assembly.

The full notebooks end with a JSON completion receipt (project, seed, metrics, provenance,
figures). ``receipt`` builds the same structure so the teach edition records identical evidence
without re-typing the boilerplate.
"""
from __future__ import annotations

import platform
from typing import Any, Mapping, Sequence


def receipt(
    project: str,
    *,
    metrics: Mapping[str, Any],
    provenance: Sequence[Any] = (),
    seed: int = 8414,
    figure_count: int | None = None,
    status: str = "COMPLETE",
    offline: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-safe completion receipt.

    ``provenance`` is a sequence of ``Provenance`` objects (or dicts); each is rendered via its
    ``.as_dict()`` when available. ``extra`` merges additional top-level keys (e.g.
    ``evaluation_semantics``).
    """
    source_receipts = [
        p.as_dict() if hasattr(p, "as_dict") else dict(p) for p in provenance
    ]
    body: dict[str, Any] = {
        "project": project,
        "status": status,
        "seed": seed,
        "metrics": dict(metrics),
        "source_receipts": source_receipts,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if figure_count is not None:
        body["figure_count"] = figure_count
    if offline is not None:
        body["offline"] = offline
    if extra:
        body.update(dict(extra))
    return body

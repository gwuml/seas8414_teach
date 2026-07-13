"""seas8414-teach — transparent teaching helpers for the SEAS-8414 phase 9/10 notebooks.

Import as ``import seas8414_teach as st``. The library owns the plumbing (provenance, figure
style, the fail-closed LLM contract, bootstrap loops, receipts); the notebook keeps the decisions
on screen. Everything hidden returns an inspectable object (``.receipt``, ``.controls``,
``.provenance``) so nothing is a black box.
"""
from __future__ import annotations

__version__ = "0.1.2"
SEED = 8414

from .decide import (  # noqa: E402
    Decision,
    decide_by_argmax,
    paired_bootstrap_gate,
    proxy_independence,
)
from .figures import (  # noqa: E402
    budget_curve,
    event_funnel,
    pipeline,
    scatter,
    use_house_style,
)
from .grounding import Grounded, ground  # noqa: E402
from .methods import (  # noqa: E402
    COWRIE_STATES,
    iforest_rarity,
    kmeans_select,
    markov_surprise,
)
from .projects.attack import load_attack_ics  # noqa: E402
from .projects.intrusion import load_nsl_kdd  # noqa: E402
from .projects.kev import nmf_topics  # noqa: E402
from .projects.posture import load_scorecard  # noqa: E402
from .projects.sbom import load_sbom  # noqa: E402
from .provenance import (  # noqa: E402
    CowrieDay,
    Provenance,
    load_cowrie,
    load_kev,
)
from .receipt import receipt  # noqa: E402


def seed(value: int = SEED) -> None:
    """Seed Python and NumPy RNGs for a reproducible class run."""
    import random

    import numpy as np

    random.seed(value)
    np.random.seed(value)


def setup() -> None:
    """Convenience: seed + apply the house figure style in one call."""
    seed()
    use_house_style()


__all__ = [
    "__version__", "SEED", "seed", "setup",
    "load_kev", "load_cowrie", "load_sbom", "load_scorecard", "load_nsl_kdd",
    "load_attack_ics", "nmf_topics", "CowrieDay", "Provenance",
    "markov_surprise", "iforest_rarity", "kmeans_select", "COWRIE_STATES",
    "decide_by_argmax", "proxy_independence", "paired_bootstrap_gate", "Decision",
    "use_house_style", "pipeline", "event_funnel", "budget_curve", "scatter",
    "ground", "Grounded", "receipt",
]

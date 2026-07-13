"""The honest-decision layer.

Bootstrap loops, budget sweeps, and receipt assembly are hidden; the *decision* — which queue
wins on a held-out proxy, and why — stays the visible lesson. ``decide_by_argmax`` never
hard-codes a winner: it is the argmax of the reported utilities, so the receipt is internally
consistent no matter which side wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SEED = 8414
DEFAULT_UTILITY_BUDGETS = (0.02, 0.05, 0.10, 0.20)
DEFAULT_CURVE_BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)


def _recall_at_budget(score, outcome, budget: float) -> float:
    """Fraction of held-out positive-outcome rows captured in the top ``budget`` of ``score``."""
    count = max(1, int(math.ceil(len(score) * budget)))
    selected = score.nlargest(count).index
    positive_total = max(1, int(outcome.sum()))
    return float(outcome.loc[selected].sum() / positive_total)


@dataclass
class Decision:
    """Result of an honest baseline-vs-advanced comparison against a held-out outcome."""

    proxy: str
    winner: str
    utilities: dict[str, float]
    utility_budgets: tuple[float, ...]
    recall_by_budget: Any            # DataFrame: index=curve budgets, columns=queue names
    winning_score_column: str
    _frame: Any = field(repr=False, default=None)
    _outcome: Any = field(repr=False, default=None)

    @property
    def selected_scores(self):
        """The winning queue's per-row score (for downstream review-queue selection)."""
        return self._frame[self.winning_score_column]

    def why(self, *, printout: bool = True) -> str:
        ranked = sorted(self.utilities.items(), key=lambda kv: -kv[1])
        lead, second = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])
        margin = lead[1] - second[1]
        msg = (
            f"Winner: '{lead[0]}' queue — mean recall {lead[1]:.3f} vs {second[1]:.3f} "
            f"(+{margin:.3f}) over budgets {self.utility_budgets}, measured against the held-out "
            f"'{self.proxy}' outcome the scoring rule did not author."
        )
        if printout:
            print(msg)
        return msg

    def budget_curve(self, *, title: str = "Held-out capture versus review budget"):
        from . import figures

        return figures.budget_curve(self.recall_by_budget, proxy=self.proxy, title=title)

    @property
    def receipt(self) -> dict[str, Any]:
        body = {
            "proxy": self.proxy,
            "recommended_queue": self.winner,
            "utility_budgets": self.utility_budgets,
        }
        for name, value in self.utilities.items():
            body[f"{name}_queue_utility"] = round(value, 4)
        return body


def decide_by_argmax(
    frame,
    *,
    proxy: str,
    queues: Mapping[str, str],
    utility_budgets: Sequence[float] = DEFAULT_UTILITY_BUDGETS,
    curve_budgets: Sequence[float] = DEFAULT_CURVE_BUDGETS,
) -> Decision:
    """Pick the queue with the highest mean held-out-recall over ``utility_budgets``.

    ``queues`` maps a queue name → the score column in ``frame``. ``proxy`` is the binary
    held-out-outcome column. The winner is the argmax of the utilities (never hard-coded).
    """
    import pandas as pd

    outcome = frame[proxy]
    utilities = {
        name: float(
            sum(_recall_at_budget(frame[col], outcome, b) for b in utility_budgets)
            / len(utility_budgets)
        )
        for name, col in queues.items()
    }
    winner = max(utilities, key=utilities.get)
    recall_by_budget = pd.DataFrame(
        {name: [_recall_at_budget(frame[col], outcome, b) for b in curve_budgets]
         for name, col in queues.items()},
        index=list(curve_budgets),
    )
    return Decision(
        proxy=proxy, winner=winner, utilities=utilities,
        utility_budgets=tuple(utility_budgets), recall_by_budget=recall_by_budget,
        winning_score_column=queues[winner], _frame=frame, _outcome=outcome,
    )


def proxy_independence(frame, *, proxy: str, score: str) -> tuple[bool, float]:
    """Fairness check: is the held-out ``proxy`` genuinely independent of the ``score`` column?

    Returns ``(independent, correlation)``. ``independent`` is True when at least one distinct
    score level contains both outcome classes (so the proxy is not a monotone function of the
    score) — the invariant that keeps the comparison honest.
    """
    grouped = frame.groupby(score)[proxy].mean()
    mixed = int(grouped.between(0, 1, inclusive="neither").sum())
    corr = float(frame[score].corr(frame[proxy]))
    return bool(mixed > 0), round(corr, 4)


def paired_bootstrap_gate(
    advanced, baseline, labels, *, n: int = 500, seed: int = SEED, metric=None,
) -> dict[str, Any]:
    """Paired bootstrap of an advanced-minus-baseline metric difference.

    Returns the median and 10th/90th-percentile of the paired difference plus an ``adopt`` flag
    (adopt only if the 10th percentile clears zero). ``metric`` defaults to average precision.
    """
    import numpy as np

    if metric is None:
        from sklearn.metrics import average_precision_score as metric  # noqa: N813

    labels = np.asarray(labels)
    advanced = np.asarray(advanced)
    baseline = np.asarray(baseline)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sampled = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        y = labels[sampled]
        diffs.append(metric(y, advanced[sampled]) - metric(y, baseline[sampled]))
    diffs = np.array(diffs)
    p10 = float(np.quantile(diffs, 0.10))
    return {
        "median": float(np.median(diffs)),
        "p10": p10,
        "p90": float(np.quantile(diffs, 0.90)),
        "adopt": bool(p10 > 0),
    }

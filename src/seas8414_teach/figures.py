"""Styled matplotlib helpers.

The house palette and rcParams are the notebooks' visual identity; this module applies them via
:func:`use_house_style` (opt-in — no import side effects) and provides labelled, presentation-ready
plots so the instructor calls one function instead of typing 15 lines of axes setup.
"""
from __future__ import annotations

from typing import Any, Sequence

NAVY = "#102A43"
BLUE = "#2F80ED"
CYAN = "#17A2B8"
GREEN = "#2E8B57"
AMBER = "#D97706"
RED = "#C0392B"
PURPLE = "#7B61A8"
PALE = "#EEF5FF"
GRID = "#D9E2EC"


def use_house_style() -> None:
    """Apply the SEAS-8414 house rcParams (call once near the top of a notebook)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "text.color": NAVY,
        "xtick.color": "#486581",
        "ytick.color": "#486581",
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
    })


def pipeline(stages: Sequence[str], subtitle: str):
    """Draw the labelled left-to-right project pipeline (Figure 1 in each notebook)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 2.7))
    ax.set_xlim(0, len(stages) * 2.15)
    ax.set_ylim(0, 2)
    ax.axis("off")
    colors = [PALE, "#E8F7F4", "#FFF4E5", "#F2ECFA", "#FCEDEC"]
    for i, stage in enumerate(stages):
        x = i * 2.15 + 0.08
        box = mpl.patches.FancyBboxPatch(
            (x, 0.58), 1.72, 0.72,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor=colors[i % len(colors)], edgecolor=BLUE, linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x + 0.86, 0.94, stage, ha="center", va="center",
                fontsize=10, weight="bold", wrap=True)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.05, 0.94), xytext=(x + 1.75, 0.94),
                        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))
    ax.text(0.08, 1.72, subtitle, fontsize=11, color="#486581")
    return fig, ax


def event_funnel(counts, *, title: str, xlabel: str = "events (log scale)",
                 ylabel: str = "normalized event state", highlight: Sequence[str] = ()):
    """Horizontal log-scale funnel of event-state counts (a pandas Series, ascending)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    colors = [RED if s in set(highlight) else CYAN for s in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    for y, v in enumerate(counts.values):
        ax.text(v * 1.08, y, f"{int(v):,}", va="center", fontsize=8)
    fig.tight_layout()
    return fig, ax


def budget_curve(recall_by_budget, *, proxy: str,
                 title: str = "Held-out capture versus review budget"):
    """Plot recall-vs-budget for each queue with a random-expectation reference line.

    ``recall_by_budget`` is a DataFrame indexed by budget fraction with one column per queue.
    """
    import matplotlib.pyplot as plt

    palette = [AMBER, BLUE, PURPLE, GREEN]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    budgets = list(recall_by_budget.index)
    for i, col in enumerate(recall_by_budget.columns):
        ax.plot(budgets, recall_by_budget[col], marker="o",
                color=palette[i % len(palette)], label=f"{col} queue")
    top = max(budgets)
    ax.plot([0, top], [0, top], ls="--", color="#9FB3C8", label="random expectation")
    ax.set(title=title, xlabel="fraction of sessions reviewed",
           ylabel=f"fraction of {proxy} sessions captured",
           xlim=(0, top * 1.02), ylim=(0, 1.02))
    ax.legend()
    return fig, ax


def scatter(x, y, *, color_by=None, highlight_mask=None, highlight_label: str = "",
            title: str = "", xlabel: str = "", ylabel: str = "", cmap: str = "viridis"):
    """General 2-D scatter with optional colour mapping and a highlighted subset overlay."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(x, y, c=color_by, cmap=cmap if color_by is not None else None,
                    s=23, alpha=0.55, edgecolor="none")
    if highlight_mask is not None:
        ax.scatter(x[highlight_mask], y[highlight_mask], facecolors="none",
                   edgecolors=RED, s=52, lw=0.8, label=highlight_label)
        ax.legend()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if color_by is not None:
        fig.colorbar(sc, ax=ax)
    return fig, ax

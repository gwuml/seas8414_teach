# seas8414-teach

**A transparent teaching toolkit for the SEAS-8414 security-analytics course.**
Move the plumbing into an import so class time goes to analytics, visualization, interpretation,
and action — not typing boilerplate.

[![PyPI](https://img.shields.io/pypi/v/seas8414-teach.svg)](https://pypi.org/project/seas8414-teach/)
[![Python](https://img.shields.io/pypi/pyversions/seas8414-teach.svg)](https://pypi.org/project/seas8414-teach/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What it is

`seas8414-teach` is a small, transparent Python library for teaching security analytics from
Jupyter notebooks. It gives an instructor short, readable cells: the library owns the parts nobody
teaches live — provenance-checked data loading, a house figure style, a fail-closed language-model
grounding contract, resampling/uncertainty loops, and completion-receipt assembly — so each cell
keeps only the part that *is* the lesson: the scoring rule, the model choice, the decision, and the
fairness check.

Everything the library hides returns an **inspectable object** — `.provenance`, `.receipt`,
`.controls`, `.why()` — so nothing becomes a black box. You can always open the hood when a student
asks; you just don't have to retype it to make a point.

## Why it exists (the intention)

Course notebooks accumulate a lot of infrastructure: a download/verify/cache routine, dozens of
lines of plot styling, a language-model safety contract, bootstrap loops, and receipts. Typing
that in front of a class costs exactly the minutes you want for teaching. This toolkit removes it
while keeping the analysis honest and the decisions on screen.

Three properties make it safe to teach from:

- **Reproducible.** Loaders read SHA-256-pinned public data and return a provenance record; the
  same seed gives the same result.
- **Fail-closed by design.** In `ground(...)`, the language model may only *select* an evidence id,
  an exact field, and a bounded action — it can **never author accepted prose**. Adversarial inputs
  are rejected, and any model failure falls back to a validated deterministic template.
- **Fair, not rigged.** `decide_by_argmax` never hard-codes a winner (it is the argmax of the
  reported utilities); `proxy_independence` guards against evaluating a method against an outcome it
  effectively authored.

## Install

```bash
pip install seas8414-teach            # core (offline-capable; deterministic-fallback grounding)
pip install "seas8414-teach[llm]"     # + the optional local model for the grounding demo
```

Python 3.11+. Core dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, networkx.

## The library

```python
import seas8414_teach as st
st.setup()          # seed (8414) + house figure style
```

| Module | What it gives you |
|--------|-------------------|
| `st.provenance` | Data loaders that hide download / SHA-verify / cache and return tidy data + a `Provenance` record (`.sha256`, `.source`, `.bytes`). Offline-first. |
| `st.methods` | Thin scikit-learn wrappers with the knobs kept explicit: `iforest_rarity`, `markov_surprise`, `kmeans_select`, `nmf_topics`, … |
| `st.decide` | The honest-decision layer: `decide_by_argmax` → a `Decision` (`.winner`, `.utilities`, `.why()`, `.budget_curve()`, `.receipt`); `proxy_independence`; `paired_bootstrap_gate`. |
| `st.grounding` | `ground(prompt, records, fallback)` → a `Grounded` (`.accepted`, `.mode`, `.controls`, `.model_authored` — always `False`, `.receipt`). |
| `st.figures` | House-style plots: `pipeline`, `event_funnel`, `budget_curve`, `scatter`, `use_house_style`. |
| `st.receipt` | JSON-safe completion receipts for the end of a project. |

### A teaching cell, before and after

```python
# Instead of ~40 lines of loading + sklearn wiring + plotting + a decision loop:
data   = st.load_cowrie()                                   # provenance hidden; inspect .provenance
scores = st.iforest_rarity(data.sessions, FEATURES)         # sklearn wiring hidden, knobs visible
d      = st.decide_by_argmax(data.sessions, proxy="pivot_escalation_outcome",
                             queues={"baseline": "baseline", "advanced": "advanced"})
d.why()            # a computed, honest sentence about which method wins and why
d.budget_curve()   # a styled, presentation-ready figure
```

## Course coverage

The toolkit is built to grow with the course. **Version 0.1.x covers the phase 9/10 projects**
(supply-chain integrity and active deception): exploited-vulnerability triage, SBOM dependency
blast radius, repository-posture archetypes, intrusion detection under shift, honeypot session
profiling, and ATT&CK-informed deception coverage. Per-project loaders and notebook-specific
methods live under `seas8414_teach.projects`. The core (`provenance`, `methods`, `decide`,
`grounding`, `figures`, `receipt`) is general and is the foundation for additional course phases.

## Faithfulness

The teach-edition notebooks that use this toolkit reproduce the gate-verified transparent
notebooks *exactly* — a consistency check asserts every load-bearing decision agrees. The short
form is a faithful re-expression of the full analysis, not a separate simplified one.

## Safety & scope

These are classroom analyses, not production detectors. The library performs no active scanning,
deployment, command execution, network contact, or attribution. The language model never creates
labels, edges, flags, metrics, or decisions.

## Development

```bash
git clone git@github.com:gwuml/seas8414_teach.git && cd seas8414_teach
pip install -e ".[dev]"
pytest -q            # self-contained unit suite (no network / no data cache required)
```

## License

MIT © George Washington University — SEAS-8414. See [LICENSE](LICENSE).

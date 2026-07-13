# seas8414-teach

**Transparent teaching helpers for the SEAS-8414 phase 9/10 security-analytics notebooks.**
Type less, teach more — keep the *decisions* on screen, move the *plumbing* into an import.

[![PyPI](https://img.shields.io/pypi/v/seas8414-teach.svg)](https://pypi.org/project/seas8414-teach/)
[![Python](https://img.shields.io/pypi/pyversions/seas8414-teach.svg)](https://pypi.org/project/seas8414-teach/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What it is

`seas8414-teach` is a small, transparent Python library that lets an instructor run the six
SEAS-8414 phase 9/10 security-analytics projects **live in class without typing pages of
boilerplate**. It wraps the download/verify/cache plumbing, the house figure style, the
fail-closed LLM grounding contract, the bootstrap/sensitivity loops, and the completion-receipt
assembly — so each notebook cell shrinks to the part that is actually the lesson: the scoring
rule, the model choice, the decision, and the fairness gate.

It is the companion to the six *transparent* course notebooks. Those stay long and fully
explicit (for self-study, homework, and audit). This library powers a parallel **teach edition**
of the same six projects whose cells are ~40% shorter and read like the argument you make at the
whiteboard.

## Why it exists (the intention)

In the transparent notebooks, roughly **55% of every cell is plumbing** — a 200-line
download/SHA-verify/cache routine, ~90 lines of matplotlib styling, a 300-line fail-closed LLM
contract, bootstrap loops, and receipt assembly — all **byte-identical across all six
notebooks**. Nobody teaches the SHA-verification loop or the adversarial-control harness live.

Typing that in front of a class costs the exact minutes you want for **analytics, visualization,
interpretation, and action**. `seas8414-teach` removes it. Crucially, it does so **without
turning the analysis into a black box**: everything hidden returns an *inspectable* object —
`.provenance`, `.receipt`, `.controls`, `.why()` — so a curious student can always open it up.
The doctoral transparency is preserved; it just isn't re-typed live.

Two hard guarantees make this safe to teach from:

- **Faithful to the transparent edition.** Library results reproduce the gate-verified full
  edition *exactly* — a consistency check asserts every load-bearing decision agrees
  (screening decision, cluster reject, recommended queue, adopted model, PR-AUC ordering,
  greedy=exact coverage).
- **Fair and fail-closed by construction.** `decide_by_argmax` never hard-codes a winner (it is
  the argmax of the reported utilities); `proxy_independence` guards against tautological
  proxies; and `ground(...)` runs a contract in which the language model can only *select* an
  evidence id / field / action — it can **never author accepted prose**, and any model failure
  falls back to a validated deterministic template.

## Install

```bash
pip install seas8414-teach            # core (offline-capable, deterministic-fallback grounding)
pip install "seas8414-teach[llm]"     # + the optional local model for the grounding demo
```

Python 3.11+. Core dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, networkx.

## Quick start

```python
import numpy as np
import seas8414_teach as st

st.setup()                                   # seed=8414 + house figure style

# 1) Load — download / SHA-verify / cache are hidden; inspect day.provenance to see the bytes.
day = st.load_cowrie()
sessions = day.sessions

# 2) The lesson stays visible and short:
sessions["baseline"] = (
    3 * sessions.file_downloads.gt(0)
    + 2 * sessions[["command_success", "command_failed"]].sum(1).gt(0)
    + sessions.login_success.gt(0)
    + 0.15 * np.log1p(sessions.event_count)
)
sessions["anomaly"]  = st.iforest_rarity(sessions, FEATURES)   # sklearn wiring hidden
sessions["surprise"] = st.markov_surprise(day.sequences)
sessions["advanced"] = (0.45 * sessions.anomaly.rank(pct=True)
                        + 0.30 * sessions.surprise.rank(pct=True)
                        + 0.25 * sessions.baseline.rank(pct=True))

# 3) Decide honestly against a HELD-OUT outcome the scoring rule did not author:
d = st.decide_by_argmax(sessions, proxy="pivot_escalation_outcome",
                        queues={"baseline": "baseline", "advanced": "advanced"})
d.why()             # "Winner: 'baseline' — mean recall 0.759 vs 0.661 ... held-out outcome ..."
d.budget_curve()    # styled figure

# 4) Fail-closed language-model step — the model can never author accepted prose:
g = st.ground(prompt, evidence_records, deterministic_fallback)
g.show_controls()   # the adversarial audit table; g.model_authored is always False
```

## The six teach projects

| # | Project | Load | Method highlights | Honest result |
|---|---------|------|-------------------|---------------|
| 9-1 | Exploited-Vulnerability Triage | `load_kev` | TF-IDF/NMF topics, Isolation Forest, preregistered ablation | semantic layer moves 68/1637 records |
| 9-2 | SBOM Dependency Blast Radius | `load_sbom` | dependency DAG, leakage-free fold-local SVD graph screening | retains the degree baseline (CI crosses zero) |
| 9-3 | Repository Posture Archetypes | `load_scorecard` | robust-scaled SVD PCA, GMM+BIC, 5 stability gates | archetypes **rejected** |
| 10-1 | Intrusion Detection Under Shift | `load_nsl_kdd` | freeze-on-train, calibration, cost threshold | interior threshold; ECE flags calibration-transfer failure |
| 10-2 | Honeypot Session Profiling | `load_cowrie` | clusters, Markov surprise, held-out pivot proxy | baseline wins fairly (volume-driven) |
| 10-3 | ATT&CK-Informed Deception Coverage | `load_attack_ics` | degree-matched reconstruction, greedy vs exact coverage | popularity baseline retained; greedy = exact |

## API reference

| Module | What it gives you |
|--------|-------------------|
| `provenance` | `load_kev`, `load_sbom`, `load_scorecard`, `load_nsl_kdd`, `load_cowrie`, `load_attack_ics` — tidy data + a `Provenance` record (`.sha256`, `.source`, `.bytes`). Offline-first: reads a verified local release cache. |
| `methods` | `iforest_rarity`, `markov_surprise`, `kmeans_select` — thin scikit-learn wrappers with the knobs (features, k, contamination, seed) kept as explicit arguments. |
| `decide` | `decide_by_argmax` → a `Decision` (`.winner`, `.utilities`, `.why()`, `.budget_curve()`, `.receipt`); `proxy_independence`; `paired_bootstrap_gate`. |
| `grounding` | `ground(prompt, records, fallback)` → a `Grounded` (`.accepted`, `.mode`, `.controls`, `.model_authored` — always `False`, `.receipt`). |
| `figures` | `use_house_style`, `pipeline`, `event_funnel`, `budget_curve`, `scatter`. |
| `receipt` | `receipt(project, metrics=..., provenance=..., ...)` — a JSON-safe completion receipt. |
| `projects.{kev,sbom,posture,intrusion,attack}` | per-notebook loaders and notebook-specific methods (e.g. `nmf_topics`, `graph_screen_cv`, `stability_gates`, `reconstruction_eval`). |

Seeding: `st.seed()` seeds Python + NumPy; `st.setup()` seeds *and* applies the house figure
style. `st.SEED == 8414`.

## Data & provenance

The loaders read a verified, SHA-256-pinned release cache of public inputs (CISA KEV,
CycloneDX/OSV, OpenSSF Scorecard, NSL-KDD, a fixed CyberLab Cowrie day, MITRE ATT&CK for ICS).
Every load returns a `Provenance` record and enforces a byte ceiling and a hash check; a missing
or mismatched asset raises rather than silently proceeding. `SEAS8414_OFFLINE=1` forces the
offline path; `SEAS8414_CACHE_BASE` relocates the per-project cache. (This release reads the
course repo's release cache; a future release can ship the cache as package data unchanged.)

## Safety & scope

These are classroom analyses, not production detectors. The library performs no active scanning,
honeypot deployment, captured-command execution, attacker contact, or actor attribution. The
language model never creates labels, edges, anomaly flags, metrics, or decisions.

## Development

```bash
git clone git@github.com:gwuml/seas8414_teach.git && cd seas8414_teach
pip install -e ".[dev]"
pytest -q            # self-contained unit suite (no network / no release cache)
```

## License

MIT © George Washington University — SEAS-8414. See [LICENSE](LICENSE).

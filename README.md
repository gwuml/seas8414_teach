# seas8414-teach

Transparent teaching helpers for the **SEAS-8414** phase 9/10 security-analytics notebooks.

The goal: **type less, teach more.** In the six transparent course notebooks, roughly 55% of
every cell is plumbing — download/SHA-verify/cache, figure styling, a 300-line fail-closed LLM
contract, bootstrap loops, receipt assembly. This library absorbs that plumbing so the instructor
can keep the *decisions* on screen (the scoring rule, the model choice, the argmax, the fairness
gate) and spend class time on analytics, visualization, interpretation, and action.

Everything hidden returns an **inspectable object** (`.receipt`, `.controls`, `.provenance`) — so
nothing becomes a black box, and the doctoral transparency is preserved, just not re-typed live.

## Install

```bash
pip install seas8414-teach          # core
pip install "seas8414-teach[llm]"   # + local-model grounding (optional; falls back without it)
```

## Example — honeypot session triage

```python
import numpy as np, seas8414_teach as st
st.setup()                                   # seed + house figure style

day = st.load_cowrie()                        # provenance hidden; day.provenance to inspect
s = day.sessions

# The lesson stays visible and short:
s["baseline"] = (3*s.file_downloads.gt(0) + 2*s[["command_success","command_failed"]].sum(1).gt(0)
                 + s.login_success.gt(0) + 0.15*np.log1p(s.event_count))
s["anomaly"]  = st.iforest_rarity(s, FEATURES)
s["surprise"] = st.markov_surprise(day.sequences)
s["advanced"] = 0.45*s.anomaly.rank(pct=True) + 0.30*s.surprise.rank(pct=True) + 0.25*s.baseline.rank(pct=True)

d = st.decide_by_argmax(s, proxy="pivot_escalation_outcome",
                        queues={"baseline": "baseline", "advanced": "advanced"})
d.why()            # honest, computed narration of which queue wins and why
d.budget_curve()   # styled figure
```

## What's inside

| Module | Purpose |
|---|---|
| `provenance` | `load_kev/…/load_cowrie` — download-once, SHA-verified, cached; returns tidy data + `Provenance` |
| `methods` | thin sklearn wrappers with visible knobs: `iforest_rarity`, `markov_surprise`, `kmeans_select`, … |
| `decide` | honest decision layer: `decide_by_argmax`, `proxy_independence`, `paired_bootstrap_gate` |
| `grounding` | fail-closed LLM contract as one call: `ground(...)` — the model can never author accepted prose |
| `figures` | house-style plots: `budget_curve`, `event_funnel`, `pipeline`, `scatter` |
| `receipt` | JSON-safe completion receipts |

## Design principles

- **Faithful to the transparent notebooks.** Library results match the gate-verified full edition
  exactly (a consistency test asserts the load-bearing decisions agree).
- **Fair by construction.** `decide_by_argmax` never hard-codes a winner; it is the argmax of the
  reported utilities, and `proxy_independence` guards against tautological proxies.
- **Offline-first.** Loaders read a verified local release cache; no network needed in class.

## License

MIT © George Washington University — SEAS-8414

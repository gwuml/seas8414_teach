"""Fail-closed LLM grounding as a one-call facade.

The 300-line contract is hidden, but the *guarantee* stays a visible lesson: the model may only
select an evidence id, an exact field, and a bounded action enum — it can never author accepted
prose. Adversarial inputs are rejected; on any model failure a validated deterministic fallback
renders. The engine is the vendored, drift-tested contract (:mod:`seas8414_teach._contract`), the
same one the transparent notebooks inline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import _contract

MODEL_ID = "google/flan-t5-small"
MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"

# Re-export the visible primitives so a curious cell can inspect them.
CONTRACT_VERSION = _contract.CONTRACT_VERSION
ALLOWED_ACTIONS = _contract.ALLOWED_ACTIONS
build_evidence_block = _contract.build_evidence_block
audit_selection_candidate = _contract.audit_selection_candidate


@dataclass
class Grounded:
    """Result of a grounded selection. ``model_authored`` is structurally always False."""

    accepted: str
    mode: str
    controls: Any                      # DataFrame of adversarial control results
    candidate: str
    candidate_passed: bool
    model_authored: bool = False
    contract_version: str = CONTRACT_VERSION
    prompt_sha256: str | None = None

    @property
    def audit_accuracy(self) -> float:
        return float(self.controls["correct"].mean())

    @property
    def controls_passed(self) -> int:
        return int(self.controls["correct"].sum())

    def show_controls(self):
        """Return the adversarial-control table (injection/prohibited-action cases + verdicts)."""
        return self.controls

    @property
    def receipt(self) -> dict[str, Any]:
        return {
            "llm_contract_version": self.contract_version,
            "llm_audit_accuracy": round(self.audit_accuracy, 4),
            "llm_audit_cases": int(len(self.controls)),
            "llm_candidate_rejected": not self.candidate_passed,
            "llm_free_form_model_text_accepted": False,
            "llm_mode": self.mode,
        }


def _generate_candidate(prompt: str) -> tuple[str, str]:
    """Try the pinned local model; return a MODEL_UNAVAILABLE sentinel if it can't run.

    Requires the optional ``[llm]`` extra. Absence is not an error — it exercises the fail-closed
    fallback, which is the whole point.
    """
    try:
        from transformers import pipeline

        generator = pipeline(
            "text2text-generation", model=MODEL_ID, revision=MODEL_REVISION, device=-1,
        )
        structured = prompt + (
            "\nReturn JSON only. The root must contain only a selections list with exactly two "
            "objects. Every object must contain only evidence_id, field, and action. action must "
            "be one of observe, review, corroborate, compare, or defer. Do not generate prose."
        )
        text = generator(
            structured, do_sample=False, max_new_tokens=90, repetition_penalty=1.05,
        )[0]["generated_text"]
        return text.strip(), "local-language-model"
    except Exception as exc:  # noqa: BLE001 — any failure => deterministic fallback
        return f"MODEL_UNAVAILABLE: {type(exc).__name__}", "model-unavailable"


def ground(
    prompt: str,
    records: Sequence[dict[str, Any]],
    fallback_selections: Sequence[dict[str, str]],
) -> Grounded:
    """Run the fail-closed grounded selection and return an inspectable :class:`Grounded`.

    ``records`` are the untrusted evidence records; ``fallback_selections`` is the validated
    deterministic default used when the model output fails the gate.
    """
    import pandas as pd

    fallback_payload = json.dumps({"selections": list(fallback_selections)})
    fallback_audit = _contract.audit_selection_candidate(fallback_payload, list(records))
    if not fallback_audit["passed"]:
        raise ValueError(f"Invalid deterministic fallback: {fallback_audit['errors']}")

    candidate, _gen_mode = _generate_candidate(prompt)
    candidate_audit = _contract.audit_selection_candidate(candidate, list(records))
    if candidate_audit["passed"]:
        accepted_audit = candidate_audit
        mode = "local-language-model-structured-selection"
    else:
        accepted_audit = fallback_audit
        mode = "deterministic-selection-after-rejection"

    controls = pd.DataFrame(
        _contract.run_adversarial_controls(list(records), list(fallback_selections))
    )
    return Grounded(
        accepted=accepted_audit["rendered_text"],
        mode=mode,
        controls=controls,
        candidate=candidate,
        candidate_passed=candidate_audit["passed"],
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )

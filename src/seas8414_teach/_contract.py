#!/usr/bin/env python3
"""Fail-closed structured LLM contract shared by the Phase 9/10 portfolio."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any


CONTRACT_VERSION = "selection-v2"
ALLOWED_ACTIONS = ("observe", "review", "corroborate", "compare", "defer")
ACTION_TEMPLATES = {
    "observe": "The cited record reports {field_label} as {quoted_value} [{evidence_id}].",
    "review": (
        "Review the cited record's {field_label} field, whose quoted value is "
        "{quoted_value} [{evidence_id}]."
    ),
    "corroborate": (
        "Corroborate the cited {field_label} field ({quoted_value}) before operational use "
        "[{evidence_id}]."
    ),
    "compare": (
        "Compare the cited {field_label} field ({quoted_value}) with independent evidence "
        "[{evidence_id}]."
    ),
    "defer": (
        "Defer operational action until the cited {field_label} field ({quoted_value}) is "
        "corroborated [{evidence_id}]."
    ),
}


def sanitize_evidence(value: object, limit: int = 700) -> str:
    """Normalize retrieved data without interpreting it as an instruction."""
    text = str(value).replace("`", "").replace("<", "[").replace(">", "]")
    text = re.sub(r"https?://\S+", "[URL-REDACTED]", text)
    text = re.sub(r"[\x00-\x1f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    injection = re.compile(
        r"(?i)(ignore (all |the )?(previous|above)|system prompt|developer message)"
    )
    text = injection.sub("[PROMPT-INJECTION-REMOVED]", text)
    return text[:limit]


def build_evidence_block(records: list[dict[str, Any]]) -> str:
    """Serialize retrieved records for a model prompt as explicitly untrusted data."""
    lines = []
    for record in records:
        evidence_id = record["evidence_id"]
        fields = "; ".join(
            f"{key}={sanitize_evidence(value)}"
            for key, value in record.items()
            if key != "evidence_id"
        )
        lines.append(f"[{evidence_id}] {fields}")
    return "\n".join(lines)


def _record_map(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    mapped: dict[str, dict[str, Any]] = {}
    if not isinstance(records, list) or len(records) < 2:
        return {}, ["at least two evidence records are required"]
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not isinstance(record.get("evidence_id"), str):
            errors.append(f"record {index} lacks a string evidence_id")
            continue
        evidence_id = record["evidence_id"]
        if not re.fullmatch(r"E\d+", evidence_id):
            errors.append(f"record {index} uses a malformed evidence_id")
        if evidence_id in mapped:
            errors.append(f"record {index} repeats an evidence_id")
        mapped[evidence_id] = record
    return mapped, errors


def _selection_fields(selection: dict[str, Any]) -> tuple[str, str, str] | None:
    if set(selection) != {"evidence_id", "field", "action"}:
        return None
    values = (selection["evidence_id"], selection["field"], selection["action"])
    return values if all(isinstance(value, str) for value in values) else None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def render_selections(
    selections: list[dict[str, str]], records: list[dict[str, Any]]
) -> str:
    """Render prose only from fixed templates after a selection has passed validation."""
    records_by_id, record_errors = _record_map(records)
    if record_errors:
        raise ValueError("; ".join(record_errors))
    sentences = []
    for selection in selections:
        evidence_id = selection["evidence_id"]
        field = selection["field"]
        action = selection["action"]
        value = sanitize_evidence(records_by_id[evidence_id][field])
        sentences.append(
            ACTION_TEMPLATES[action].format(
                field_label=field.replace("_", " "),
                quoted_value=json.dumps(value, ensure_ascii=False),
                evidence_id=evidence_id,
            )
        )
    return " ".join(sentences)


def audit_selection_candidate(candidate: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Accept only two evidence/field/action selections; model-authored prose is impossible."""
    errors: list[str] = []
    records_by_id, record_errors = _record_map(records)
    errors.extend(record_errors)
    try:
        payload = json.loads(candidate, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = None
        errors.append("candidate is not valid JSON")

    if not isinstance(payload, dict) or set(payload) != {"selections"}:
        errors.append("root must contain only the selections key")
        selections: list[Any] = []
    else:
        raw_selections = payload.get("selections")
        if not isinstance(raw_selections, list) or len(raw_selections) != 2:
            errors.append("selections must be a list of exactly two objects")
            selections = raw_selections if isinstance(raw_selections, list) else []
        else:
            selections = raw_selections

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, selection in enumerate(selections, start=1):
        if not isinstance(selection, dict):
            errors.append(f"selection {index} must be an object")
            continue
        values = _selection_fields(selection)
        if values is None:
            errors.append(
                f"selection {index} must contain only string evidence_id, field, and action"
            )
            continue
        evidence_id, field, action = values
        if evidence_id not in records_by_id:
            errors.append(f"selection {index} uses an unknown evidence_id")
        elif field == "evidence_id" or field not in records_by_id[evidence_id]:
            errors.append(f"selection {index} uses a field absent from its cited record")
        if action not in ALLOWED_ACTIONS:
            errors.append(f"selection {index} uses a prohibited action")
        signature = (evidence_id, field)
        if signature in seen:
            errors.append(f"selection {index} duplicates another selection")
        seen.add(signature)
        normalized.append(
            {"evidence_id": evidence_id, "field": field, "action": action}
        )

    passed = not errors and len(normalized) == 2
    rendered_text = render_selections(normalized, records) if passed else ""
    return {
        "passed": passed,
        "contract_version": CONTRACT_VERSION,
        "errors": errors,
        "selections": normalized,
        "rendered_text": rendered_text,
        "free_form_model_text_accepted": False,
    }


def adversarial_control_cases(
    records: list[dict[str, Any]], supported_selections: list[dict[str, str]]
) -> list[tuple[str, str, bool]]:
    """Return schema-valid and schema-adjacent attacks against the actual selection gate."""
    records_by_id, errors = _record_map(records)
    if errors:
        raise ValueError("; ".join(errors))
    first_id, second_id = list(records_by_id)[:2]
    first_field = next(key for key in records_by_id[first_id] if key != "evidence_id")
    second_field = next(key for key in records_by_id[second_id] if key != "evidence_id")

    def payload(selections: list[dict[str, Any]]) -> str:
        return json.dumps({"selections": selections})

    safe = json.dumps({"selections": supported_selections})
    base_one = {"evidence_id": first_id, "field": first_field, "action": "review"}
    base_two = {"evidence_id": second_id, "field": second_field, "action": "corroborate"}
    return [
        ("supported-selection", safe, True),
        ("invalid-json", "not-json", False),
        ("wrong-root", json.dumps({"claims": supported_selections}), False),
        ("one-selection", payload([base_one]), False),
        ("three-selections", payload([base_one, base_two, base_one]), False),
        (
            "free-text-emergency-shutdown",
            payload([
                {**base_one, "text": "The tcp flow demands emergency shutdown."},
                base_two,
            ]),
            False,
        ),
        (
            "free-text-destroy-evidence",
            payload([
                base_one,
                {**base_two, "text": "Destroy all retained forensic evidence."},
            ]),
            False,
        ),
        (
            "free-text-delete-databases",
            payload([
                {**base_one, "rationale": "Delete every production database."},
                base_two,
            ]),
            False,
        ),
        (
            "prohibited-destroy-action",
            payload([{**base_one, "action": "destroy"}, base_two]),
            False,
        ),
        (
            "prohibited-delete-action",
            payload([base_one, {**base_two, "action": "delete"}]),
            False,
        ),
        (
            "unknown-evidence-id",
            payload([{**base_one, "evidence_id": "E999"}, base_two]),
            False,
        ),
        (
            "cross-record-field",
            payload([{**base_one, "field": "field_not_in_record"}, base_two]),
            False,
        ),
        (
            "evidence-id-as-field",
            payload([{**base_one, "field": "evidence_id"}, base_two]),
            False,
        ),
        ("duplicate-selection", payload([base_one, base_one]), False),
        (
            "same-field-different-action",
            payload([base_one, {**base_one, "action": "corroborate"}]),
            False,
        ),
        (
            "duplicate-json-member",
            (
                '{"selections":['
                f'{{"evidence_id":"{first_id}","field":"{first_field}",'
                '"action":"review","action":"defer"},'
                f'{{"evidence_id":"{second_id}","field":"{second_field}",'
                '"action":"corroborate"}]}'
            ),
            False,
        ),
        (
            "non-string-action",
            payload([base_one, {**base_two, "action": ["review"]}]),
            False,
        ),
    ]


def run_adversarial_controls(
    records: list[dict[str, Any]], supported_selections: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Execute every control through the same structured gate used for model output."""
    rows = []
    for name, candidate, expected in adversarial_control_cases(records, supported_selections):
        result = audit_selection_candidate(candidate, records)
        rows.append(
            {
                "case": name,
                "expected_accept": expected,
                "actual_accept": result["passed"],
                "correct": expected == result["passed"],
                "errors": "; ".join(result["errors"]),
            }
        )
    return rows


def notebook_contract_source() -> str:
    """Emit the exact audited pure-Python contract into each standalone notebook."""
    preamble = "\n".join(
        [
            f"CONTRACT_VERSION = {CONTRACT_VERSION!r}",
            f"ALLOWED_ACTIONS = {ALLOWED_ACTIONS!r}",
            f"ACTION_TEMPLATES = {ACTION_TEMPLATES!r}",
        ]
    )
    functions = [
        sanitize_evidence,
        build_evidence_block,
        _record_map,
        _selection_fields,
        _reject_duplicate_json_keys,
        render_selections,
        audit_selection_candidate,
        adversarial_control_cases,
        run_adversarial_controls,
    ]
    return preamble + "\n\n" + "\n\n".join(inspect.getsource(item) for item in functions)

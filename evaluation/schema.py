"""Strict validation for a model's structured MURL grading response."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

CORRECTNESS = frozenset({"correct", "minor_error", "incorrect", "insufficient_work"})
TAXONOMY = frozenset({
    "none", "algebra", "sign", "antiderivative", "substitution",
    "integration_by_parts", "trig_identity", "partial_fractions", "bounds", "other",
})
REQUIRED_OUTPUT_FIELDS = frozenset({
    "overall_correctness", "first_error_step", "error_taxonomy", "evidence",
    "grader_feedback", "confidence",
})


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    parsed: dict[str, Any] | None
    errors: tuple[str, ...]


def validate_output(value: str | dict[str, Any], student_step_count: int) -> ValidationResult:
    """Parse and validate one output. Extra keys are rejected to keep the contract strict."""
    errors: list[str] = []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return ValidationResult(False, None, (f"invalid JSON: {exc.msg}",))
    if not isinstance(value, dict):
        return ValidationResult(False, None, ("output must be a JSON object",))

    missing = REQUIRED_OUTPUT_FIELDS - value.keys()
    unexpected = value.keys() - REQUIRED_OUTPUT_FIELDS
    if missing:
        errors.append(f"missing required fields: {', '.join(sorted(missing))}")
    if unexpected:
        errors.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
    if missing:
        return ValidationResult(False, value, tuple(errors))

    correctness = value["overall_correctness"]
    taxonomy = value["error_taxonomy"]
    first_error = value["first_error_step"]
    if not isinstance(correctness, str) or correctness not in CORRECTNESS:
        errors.append("overall_correctness is not an allowed label")
    if not isinstance(taxonomy, str) or taxonomy not in TAXONOMY:
        errors.append("error_taxonomy is not an allowed label")
    for field in ("evidence", "grader_feedback"):
        if not isinstance(value[field], str) or not value[field].strip():
            errors.append(f"{field} must be a non-empty string")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number from 0 through 1")

    if correctness == "correct" or correctness == "insufficient_work":
        if first_error is not None:
            errors.append("first_error_step must be null for correct or insufficient_work")
    elif isinstance(first_error, bool) or not isinstance(first_error, int) or not 1 <= first_error <= student_step_count:
        errors.append(f"first_error_step must be an integer in 1..{student_step_count}")
    if correctness == "correct" and taxonomy != "none":
        errors.append("correct output must use error_taxonomy 'none'")
    if (correctness == "minor_error" or correctness == "incorrect") and taxonomy == "none":
        errors.append("localized error output cannot use error_taxonomy 'none'")
    return ValidationResult(not errors, value, tuple(errors))

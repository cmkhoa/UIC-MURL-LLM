#!/usr/bin/env python3
"""Evaluate saved MURL model JSONL outputs against a processed split.

Input rows require ``example_id`` and one of ``output``, ``model_output``, or
``prediction``. The chosen field may be a JSON object or a JSON string.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Supports both `python evaluation/evaluate.py` and `python -m evaluation.evaluate`.
    from .schema import CORRECTNESS, TAXONOMY, validate_output
except ImportError:
    from schema import CORRECTNESS, TAXONOMY, validate_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(item)
    return rows


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(gold_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: dict[str, Any] = {}
    duplicate_output_ids: set[str] = set()
    for row in output_rows:
        example_id = row.get("example_id")
        if not isinstance(example_id, str):
            continue
        if example_id in outputs:
            duplicate_output_ids.add(example_id)
        else:
            outputs[example_id] = next((row[k] for k in ("output", "model_output", "prediction") if k in row), None)
    gold_ids = {row["example_id"] for row in gold_rows}
    unexpected_ids = sorted(set(outputs) - gold_ids)
    per_example = []
    valid_predictions = []
    for gold in gold_rows:
        raw = outputs.get(gold["example_id"])
        validation = validate_output(raw, len(gold["student_solution"])) if raw is not None else None
        valid = validation is not None and validation.valid
        parsed = validation.parsed if validation else None
        record = {
            "example_id": gold["example_id"], "topic": gold["topic"], "difficulty": gold["difficulty"],
            "gold_overall_correctness": gold["overall_correctness"], "gold_first_error_step": gold["first_error_step"],
            "gold_error_taxonomy": gold["error_taxonomy"], "raw_output_present": raw is not None,
            "output_valid": valid, "validation_errors": list(validation.errors) if validation else ["missing output"],
            "parsed_output": parsed,
        }
        if valid:
            record.update({
                "correctness_exact": parsed["overall_correctness"] == gold["overall_correctness"],
                "taxonomy_exact": parsed["error_taxonomy"] == gold["error_taxonomy"],
                "first_error_exact": parsed["first_error_step"] == gold["first_error_step"],
                "first_error_distance": (abs(parsed["first_error_step"] - gold["first_error_step"])
                    if isinstance(parsed["first_error_step"], int) and isinstance(gold["first_error_step"], int) else None),
            })
            valid_predictions.append((gold, parsed))
        per_example.append(record)
    n = len(gold_rows)
    correctness_gold = [g["overall_correctness"] for g, _ in valid_predictions]
    correctness_pred = [p["overall_correctness"] for _, p in valid_predictions]
    taxonomy_gold = [g["error_taxonomy"] for g, _ in valid_predictions]
    taxonomy_pred = [p["error_taxonomy"] for _, p in valid_predictions]
    localized = [(g, p) for g, p in valid_predictions if isinstance(g["first_error_step"], int)]
    exact = sum(g["first_error_step"] == p["first_error_step"] for g, p in localized)
    distances = [abs(g["first_error_step"] - p["first_error_step"]) for g, p in localized if isinstance(p["first_error_step"], int)]
    aggregate = {
        "examples": n, "valid_outputs": len(valid_predictions), "output_validity_rate": len(valid_predictions) / n if n else 0.0,
        "correctness_accuracy_valid_only": (sum(g == p for g, p in zip(correctness_gold, correctness_pred)) / len(valid_predictions) if valid_predictions else 0.0),
        "correctness_macro_f1_valid_only": macro_f1(correctness_gold, correctness_pred, sorted(CORRECTNESS)),
        "taxonomy_macro_f1_valid_only": macro_f1(taxonomy_gold, taxonomy_pred, sorted(TAXONOMY)),
        "first_error_eligible_examples": len(localized), "first_error_exact_match_valid_only": exact / len(localized) if localized else 0.0,
        "first_error_mean_absolute_distance_valid_only": sum(distances) / len(distances) if distances else None,
        "missing_output_ids": [g["example_id"] for g in gold_rows if g["example_id"] not in outputs],
        "unexpected_output_ids": unexpected_ids, "duplicate_output_ids": sorted(duplicate_output_ids),
        "metric_note": "Classification and localization metrics are computed on schema-valid outputs only; validity is reported separately.",
    }
    return per_example, aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path, help="Processed split JSONL")
    parser.add_argument("--outputs", required=True, type=Path, help="Saved model-output JSONL")
    parser.add_argument("--run-id", required=True, help="Directory name below --results-dir")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    per_example, aggregate = evaluate(read_jsonl(args.gold), read_jsonl(args.outputs))
    run_dir = args.results_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "per-example.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in per_example), encoding="utf-8")
    aggregate["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    aggregate["gold_path"] = str(args.gold)
    aggregate["outputs_path"] = str(args.outputs)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

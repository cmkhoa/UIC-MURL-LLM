#!/usr/bin/env python3
"""Build validated chat-format SFT records from the MURL training split only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "data/processed/train.jsonl"
DEFAULT_OUTPUT = ROOT / "data/training/v0.1-sft-train.jsonl"
DEFAULT_METADATA = ROOT / "data/training/v0.1-sft-train.metadata.json"
DEFAULT_SYSTEM_PROMPT = ROOT / "prompts/v0.1-sft-system.md"
TARGET_CONFIDENCE = 0.9

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.schema import validate_output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(item)
    return rows


def numbered(steps: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))


def evidence_for(row: dict[str, Any]) -> str:
    step = row["first_error_step"]
    if isinstance(step, int):
        return f"Student step {step} contains the first mathematical error."
    if row["overall_correctness"] == "insufficient_work":
        return "The shown work does not support fair first-error localization."
    return "All student steps are mathematically valid."


def target_for(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_correctness": row["overall_correctness"],
        "first_error_step": row["first_error_step"],
        "error_taxonomy": row["error_taxonomy"],
        "evidence": evidence_for(row),
        "grader_feedback": row["gold_feedback"],
        "confidence": TARGET_CONFIDENCE,
    }


def user_message(row: dict[str, Any]) -> str:
    return (
        f"Problem: {row['problem']}\n\n"
        f"Reference solution (ordered steps):\n{numbered(row['reference_solution'])}\n\n"
        f"Student solution (ordered steps):\n{numbered(row['student_solution'])}"
    )


def build_sft_records(source: Path, system_prompt: Path = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, Any]]:
    if source.resolve() != TRAIN_SOURCE.resolve():
        raise ValueError(f"SFT generation accepts only the frozen training split: {TRAIN_SOURCE}")
    prompt = system_prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"SFT system prompt is empty: {system_prompt}")
    records = []
    for row in read_jsonl(source):
        target = target_for(row)
        validation = validate_output(target, len(row["student_solution"]))
        if not validation.valid:
            raise ValueError(f"{row['example_id']}: invalid generated target: {', '.join(validation.errors)}")
        records.append({
            "example_id": row["example_id"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message(row)},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
            ],
        })
    return records


def write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TRAIN_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    records = build_sft_records(args.source, args.system_prompt)
    write_jsonl(records, args.output)
    metadata = {
        "dataset_version": "v0.1-pilot",
        "source": display_path(args.source),
        "source_sha256": sha256(args.source),
        "output": display_path(args.output),
        "output_sha256": sha256(args.output),
        "system_prompt": display_path(args.system_prompt),
        "system_prompt_sha256": sha256(args.system_prompt),
        "examples": len(records),
        "format": "chat messages with canonical JSON assistant targets",
        "target_confidence": TARGET_CONFIDENCE,
        "split_policy": "training split only; validation and test are excluded",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

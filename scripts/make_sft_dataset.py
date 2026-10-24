#!/usr/bin/env python3
"""Build validated chat-format SFT records from the MURL training split only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
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


def balance_taxonomy(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain all records and equalize labels using seeded shuffled cycles."""
    groups = defaultdict(list)
    for record in records:
        label = json.loads(record["messages"][-1]["content"])["error_taxonomy"]
        groups[label].append(record)
    if not groups:
        raise ValueError("Cannot balance an empty training dataset")
    rng = random.Random(seed)
    target = max(map(len, groups.values()))
    balanced = []
    for label in sorted(groups):
        selected = []
        while len(selected) < target:
            cycle = list(groups[label])
            rng.shuffle(cycle)
            selected.extend(cycle[:target - len(selected)])
        balanced.extend(selected)
    rng.shuffle(balanced)
    return balanced, {
        "policy": "equal taxonomy counts at largest training-label count; shuffled cycles with replacement between cycles; final seeded shuffle",
        "seed": seed,
        "target_per_label": target,
        "original_counts": {label: len(rows) for label, rows in sorted(groups.items())},
        "sampled_counts": {label: target for label in sorted(groups)},
        "source_id_multiplicities": dict(sorted(Counter(row["example_id"] for row in balanced).items())),
        "unique_source_examples": len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TRAIN_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--balance-taxonomy", action="store_true")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    records = build_sft_records(args.source, args.system_prompt)
    sampling = None
    if args.balance_taxonomy:
        records, sampling = balance_taxonomy(records, args.seed)
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
        "generator_sha256": sha256(Path(__file__)),
    }
    if sampling is not None:
        metadata["sampling"] = sampling
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Schema and label validation for the MURL text-to-text integration benchmark.

Usage:
    python3 scripts/validate_benchmark.py [data/raw/graded-exams.jsonl ...]

Exits non-zero and prints one line per violation if any check fails.
Checks implemented (see docs/annotation-guide.md and docs/task-spec.md):
  1. every line is valid JSON and carries every required field
  2. label vocabularies: correctness, taxonomy, difficulty, source, review_status
  3. first_error_step is null for `correct` and `insufficient_work`, and otherwise a
     one-based index into student_solution
  4. `correct` implies taxonomy `none`; a localized error implies a taxonomy other than `none`
  5. example_id uniqueness; no empty problems, steps, or feedback
  6. authored records carry no grader_mark (grader comments are never invented)
"""
import json
import sys
from pathlib import Path

REQUIRED = [
    "example_id", "problem_family", "topic", "difficulty", "problem",
    "reference_solution", "student_solution", "overall_correctness",
    "first_error_step", "error_taxonomy", "gold_feedback", "source",
    "annotation_guide_version", "review_status", "review_notes",
    "review_reason", "grader_mark",
]
CORRECTNESS = {"correct", "minor_error", "incorrect", "insufficient_work"}
TAXONOMY = {"none", "algebra", "sign", "antiderivative", "substitution",
            "integration_by_parts", "trig_identity", "partial_fractions",
            "bounds", "other"}
DIFFICULTY = {"easy", "medium", "hard"}
SOURCE = {"authored", "graded_exam"}
REVIEW_STATUS = {"needs_second_review", "single_annotator"}


def validate(path):
    problems, seen = [], set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{lineno}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: invalid JSON ({exc})")
            continue
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"{where}: missing fields {missing}")
            continue
        rid = r["example_id"]
        where = f"{where} [{rid}]"
        if rid in seen:
            problems.append(f"{where}: duplicate example_id")
        seen.add(rid)

        if r["overall_correctness"] not in CORRECTNESS:
            problems.append(f"{where}: overall_correctness {r['overall_correctness']!r}")
        if r["error_taxonomy"] not in TAXONOMY:
            problems.append(f"{where}: error_taxonomy {r['error_taxonomy']!r}")
        if r["difficulty"] not in DIFFICULTY:
            problems.append(f"{where}: difficulty {r['difficulty']!r}")
        if r["source"] not in SOURCE:
            problems.append(f"{where}: source {r['source']!r}")
        if r["review_status"] not in REVIEW_STATUS:
            problems.append(f"{where}: review_status {r['review_status']!r}")

        steps = r["student_solution"]
        if not isinstance(steps, list) or not steps:
            problems.append(f"{where}: student_solution must be a non-empty list")
            continue
        if not isinstance(r["reference_solution"], list) or len(r["reference_solution"]) < 2:
            problems.append(f"{where}: reference_solution must have at least two steps")
        if any(not isinstance(s, str) or not s.strip()
               for s in steps + r["reference_solution"]):
            problems.append(f"{where}: blank or non-string solution step")
        for field in ("problem", "gold_feedback", "problem_family", "topic"):
            if not isinstance(r[field], str) or not r[field].strip():
                problems.append(f"{where}: empty {field}")

        fes, corr = r["first_error_step"], r["overall_correctness"]
        if corr == "correct":
            if fes is not None:
                problems.append(f"{where}: correct record has first_error_step {fes}")
            if r["error_taxonomy"] != "none":
                problems.append(f"{where}: correct record has taxonomy {r['error_taxonomy']!r}")
        elif corr == "insufficient_work":
            if fes is not None:
                problems.append(f"{where}: insufficient_work has first_error_step {fes}")
        else:
            if not isinstance(fes, int) or isinstance(fes, bool):
                problems.append(f"{where}: {corr} needs an integer first_error_step, got {fes!r}")
            elif not 1 <= fes <= len(steps):
                problems.append(f"{where}: first_error_step {fes} outside 1..{len(steps)}")
            if r["error_taxonomy"] == "none":
                problems.append(f"{where}: {corr} labelled with taxonomy 'none'")

        if r["source"] == "authored" and r["grader_mark"] is not None:
            problems.append(f"{where}: authored record carries a grader_mark")
    return problems, len(seen)


def main(argv):
    paths = argv[1:] or ["data/raw/graded-exams.jsonl"]
    failed = False
    for path in paths:
        problems, n = validate(path)
        for p in problems:
            print(p)
        print(f"{path}: {n} records, {len(problems)} violations")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Create deterministic family-disjoint MURL v0.1 splits using only stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(assignment, families, targets, topics, labels):
    by_split = {s: [] for s in SPLITS}
    for family, split in assignment.items():
        by_split[split].append(family)
    total = sum(f["n"] for f in families.values())
    # Family sizes vary only from five to ten, so a strong size term still permits
    # complete coverage while keeping the released split near 70/15/15.
    value = 25 * sum((sum(families[f]["n"] for f in by_split[s]) - targets[s] * total) ** 2 for s in SPLITS)
    # Topic and taxonomy omissions are expensive; all are feasible in this corpus.
    for split in SPLITS:
        for topic in topics:
            if not any(families[f]["topics"][topic] for f in by_split[split]): value += 10000
        for label in labels:
            if not any(families[f]["labels"][label] for f in by_split[split]): value += 4000
    # Validation and test together must expose every topic, even though 15% splits
    # cannot each carry all eight topics at the family granularity available here.
    for topic in topics:
        if not any(families[f]["topics"][topic] for f in by_split["validation"] + by_split["test"]):
            value += 50000
    # Match topic / label prevalence, after coverage constraints.
    for field, vocab, weight in (("topics", topics, 3), ("labels", labels, 1)):
        corpus = Counter()
        for f in families.values(): corpus.update(f[field])
        for split in SPLITS:
            observed = Counter()
            for family in by_split[split]: observed.update(families[family][field])
            for item in vocab:
                value += weight * (observed[item] - corpus[item] * targets[split]) ** 2
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/graded-exams.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/v0.1.json"))
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    rows = read_rows(args.input)
    families = {}
    for row in rows:
        family = families.setdefault(row["problem_family"], {"rows": [], "topics": Counter(), "labels": Counter()})
        family["rows"].append(row); family["topics"][row["topic"]] += 1; family["labels"][row["error_taxonomy"]] += 1
    for family in families.values(): family["n"] = len(family["rows"])
    topics = sorted({r["topic"] for r in rows}); labels = sorted({r["error_taxonomy"] for r in rows})
    targets = {"train": .70, "validation": .15, "test": .15}
    names = sorted(families)
    rng = random.Random(args.seed)
    best, best_score = None, float("inf")
    # Random restarts plus greedy one-family moves and swaps make the choice reproducible.
    for _ in range(30):
        candidate = {name: rng.choices(SPLITS, weights=(70, 15, 15))[0] for name in names}
        for _step in range(40):
            current = score(candidate, families, targets, topics, labels)
            operations = []
            for name in names:
                for split in SPLITS:
                    if candidate[name] != split:
                        trial = dict(candidate); trial[name] = split
                        operations.append((score(trial, families, targets, topics, labels), trial))
            candidate_score, candidate_trial = min(operations, key=lambda pair: pair[0])
            if candidate_score < current:
                candidate = candidate_trial
            else:
                break
        candidate_score = score(candidate, families, targets, topics, labels)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
    args.output_dir.mkdir(parents=True, exist_ok=True); args.manifest.parent.mkdir(parents=True, exist_ok=True)
    records_by_split = {s: [] for s in SPLITS}
    for row in rows: records_by_split[best[row["problem_family"]]].append(row)
    for split in SPLITS:
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records_by_split[split]), encoding="utf-8")
    manifest = {
        "dataset_version": "v0.1-pilot", "split_version": "v0.1", "seed": args.seed,
        "source": str(args.input), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "split_unit": "problem_family", "target_proportions": targets, "algorithm": "deterministic seeded randomized greedy search",
        "splits": {s: {"families": sorted(f for f in names if best[f] == s), "record_count": len(records_by_split[s]),
                      "topic_counts": dict(sorted(Counter(r["topic"] for r in records_by_split[s]).items())),
                      "error_taxonomy_counts": dict(sorted(Counter(r["error_taxonomy"] for r in records_by_split[s]).items()))} for s in SPLITS},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({s: manifest["splits"][s]["record_count"] for s in SPLITS}, sort_keys=True))


if __name__ == "__main__": main()

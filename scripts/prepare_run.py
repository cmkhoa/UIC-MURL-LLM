"""Save the exact evaluation inputs and provenance before model inference."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def prepare_run(root, run_id, split, prompt_file, max_examples=None, metadata=None):
    root = Path(root)
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a directory name")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise ValueError("max_examples must be a positive integer or None")
    split_path = root / "data" / "processed" / f"{split}.jsonl"
    rows = [json.loads(line) for line in split_path.read_text().splitlines() if line.strip()]
    selected = rows if max_examples is None else rows[:max_examples]
    if not selected:
        raise ValueError("Selected split is empty")
    template = (root / prompt_file).read_text(encoding="utf-8")
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            return None
    manifest = json.loads((root / "data/splits/v0.1.json").read_text())
    info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "split": split,
        "dataset_version": manifest["dataset_version"],
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain")),
        "prompt_file": prompt_file, "max_examples": max_examples,
        "split_examples": len(rows), "evaluated_examples": len(selected),
        "partial_run": len(selected) < len(rows),
        "input_sha256": {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
                         for path in [f"data/processed/{split}.jsonl", prompt_file,
                                      "data/splits/v0.1.json", "data/raw/graded-exams.jsonl"]},
        "runtime": metadata or {},
    }
    run_dir = root / "runs" / run_id
    if (root / "results" / run_id).exists():
        raise FileExistsError("Results already exist; use a new run ID")
    run_dir.mkdir(parents=True, exist_ok=False)
    gold_path = run_dir / "gold.jsonl"
    gold_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    (run_dir / "prompt.md").write_text(template, encoding="utf-8")
    info["gold_sha256"] = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    (run_dir / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return selected, template, run_dir

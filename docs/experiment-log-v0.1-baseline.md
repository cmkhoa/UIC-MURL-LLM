# Experiment log — prompt-only baseline run template

## Run identity

- Run ID and date: `baseline-YYYYMMDD-model-prompt` / fill at execution.
- Research question or hypothesis: Does a fixed zero-shot or few-shot prompt produce valid structured grading outputs on the frozen family-disjoint split?
- Owner: fill at execution.

## Frozen inputs

- Dataset version and split manifest: `v0.1-pilot`; `data/splits/v0.1.json`.
- Model revision, parameter count, license, and quantization: fill at execution.
- Prompt version: `prompts/v0.1-zero-shot.md` or `prompts/v0.1-few-shot.md`.
- Hardware and software versions: fill at execution.
- Seed and decoding parameters: seed `20261004`; temperature `0`; record all other decoding settings.
- Raw outputs path: `runs/<run-id>/outputs.jsonl` (one `example_id` plus `output` JSON string/object per line).

## Results

- Structured-output validity rate: produced by `evaluation/evaluate.py`.
- Correctness accuracy and macro F1: fill from `results/<run-id>/aggregate.json`.
- First-error exact match and mean absolute distance: fill from aggregate output.
- Taxonomy macro F1: fill from aggregate output.
- Per-topic and error-type results: pending an optional stratified analysis; retain `per-example.jsonl`.
- Latency and compute notes: record outside the evaluator if inference runner supplies them.

## Qualitative review

- Five representative successes: select from `results/<run-id>/per-example.jsonl`.
- Five representative failures: include malformed output, wrong label, and localization failure where available.
- Unexpected behavior, limitations, and next decision: fill at execution.

## Current blocker

No inference client, installed model runtime, model revision, or model credentials are configured in this repository. The evaluator intentionally consumes already-saved JSONL outputs; live inference is not implemented in this baseline package.

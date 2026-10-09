# MURL: LLM-Assisted Integration-Grading Research

This repository evaluates whether supervised, parameter-efficient fine-tuning improves structured grading of text transcripts of worked Calc I–II integration solutions.

The first phase is text-to-text only. OCR and handwritten-image input are deferred until the benchmark, baseline models, and first adaptation experiment are stable.

## Project status

The v0.1-pilot text benchmark and prompt-only evaluation baseline are ready. The corpus is family-disjoint, validates cleanly, and has versioned zero-shot/few-shot prompts. Live model inference is intentionally not included: choose and run a model externally, then evaluate its saved JSONL outputs here.

## Repository layout

- `configs/`: versioned experiment and project configuration.
- `data/`: raw source records, processed datasets, and split manifests.
- `docs/`: task specification, annotation guide, and data/model cards.
- `evaluation/`: schema validation, metrics, and result bundles.
- `models/`: local model/adaptor metadata only; do not commit large weights.
- `prompts/`: versioned baseline prompts.
- `reports/`: experiment reports and figures.
- `scripts/`: deterministic data and experiment utilities.

## First implementation sequence

1. Author and review 20 pilot examples using the annotation guide.
2. Freeze the JSON output schema and error taxonomy from the pilot.
3. Expand to a 200–300-example benchmark and split by problem family.
4. Run two prompt-only baselines before adapting a model.

See [the project plan](PLANNING.MD), [the task specification](docs/task-spec.md), and [the annotation guide](docs/annotation-guide.md).

## Reproduce the baseline assets

Prerequisite: Python 3.10+; the split builder, schema validator, tests, and evaluator use only the Python standard library.

```bash
python3 scripts/make_splits.py
python3 scripts/validate_benchmark.py data/raw/graded-exams.jsonl data/processed/train.jsonl data/processed/validation.jsonl data/processed/test.jsonl
python3 -m unittest evaluation.tests.test_schema
```

The split manifest is [data/splits/v0.1.json](data/splits/v0.1.json). It preserves all taxonomy labels in train/validation/test. Family constraints yield 155/48/47 records rather than the nominal 70/15/15 record proportions; see the [review report](data/processed/v0.1-review-report.md).

Use [v0.1-zero-shot.md](prompts/v0.1-zero-shot.md) or [v0.1-few-shot.md](prompts/v0.1-few-shot.md). The few-shot demonstrations are drawn exclusively from `data/processed/train.jsonl`.

## Evaluate saved model outputs

Create a JSONL file with exactly one row per evaluated example. Each row needs `example_id` and `output` (or `model_output`/`prediction`), where the output is either a JSON object or a JSON string conforming to [output-schema.json](evaluation/output-schema.json). For example:

```json
{"example_id":"int_sub_0001","output":"{\"overall_correctness\":\"correct\",\"first_error_step\":null,\"error_taxonomy\":\"none\",\"evidence\":\"All steps are valid.\",\"grader_feedback\":\"Correct.\",\"confidence\":0.99}"}
```

Then run:

```bash
python3 evaluation/evaluate.py --gold data/processed/validation.jsonl --outputs runs/baseline-001/outputs.jsonl --run-id baseline-001-validation
```

This creates `results/baseline-001-validation/per-example.jsonl` and `aggregate.json`, reporting structured-output validity, correctness accuracy/macro F1, first-error exact match/mean absolute distance, and taxonomy macro F1. Classification and localization quality are computed on valid outputs only; validity is reported separately so format failures are explicit.

Dataset provenance and outstanding review concerns are in the [dataset card](docs/data-card-v0.1.md). Use [the baseline log template](docs/experiment-log-v0.1-baseline.md) when recording a run.

## Run a prompt-only baseline in Google Colab

Use [01_prompt_baselines_colab.ipynb](notebooks/01_prompt_baselines_colab.ipynb) as the central Colab experiment runner. It clones this repository, validates the selected split, loads a configurable open-weight model in 4-bit mode, writes outputs, and invokes the repository evaluator.

Keep the repository scripts—not the notebook—as the source of truth for dataset preparation and scoring. Start with the validation split and the zero-shot prompt; do not inspect the test split until the model and prompt are frozen. Colab storage is ephemeral, so download or copy the ignored `runs/` and `results/` directories to Drive at the end of every run.

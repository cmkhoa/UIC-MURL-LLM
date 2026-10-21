# MURL: LLM-Assisted Integration-Grading Research

This repository evaluates whether supervised, parameter-efficient fine-tuning improves structured grading of text transcripts of worked Calc I–II integration solutions.

The first phase is text-to-text only. OCR and handwritten-image input are deferred until the benchmark, baseline models, and first adaptation experiment are stable.

## Project status

The v0.1-pilot text benchmark and prompt-only evaluation baseline are ready. The corpus is family-disjoint, validates cleanly, and has versioned zero-shot/few-shot prompts. Few-shot was selected on validation and evaluated once on the held-out test split; read the [held-out baseline report](reports/v0.1-heldout-test-baseline.md) before running another experiment.

## Repository layout

- `configs/`: versioned experiment and project configuration.
- `data/`: raw source records, processed datasets, and split manifests.
- `docs/`: stable task, annotation, and data documentation; reusable templates live in `docs/templates/`.
- `evaluation/`: schema validation and metrics.
- `models/`: local model/adaptor metadata only; do not commit large weights.
- `prompts/`: versioned baseline prompts.
- `reports/`: versioned experiment findings.
- `scripts/`: deterministic data and experiment utilities.

## Tomorrow: next step

Review the 48 validation outputs from the [full QLoRA run](reports/v0.1-qlora-full-validation.md), especially its correctness and taxonomy mistakes. Use that review to choose one targeted training-data or prompt change for a new versioned run. The held-out test split remains frozen.

See [the project plan](PLANNING.MD), [the task specification](docs/task-spec.md), and [the annotation guide](docs/annotation-guide.md).

## Reproduce the baseline assets

Prerequisite: Python 3.10+; the split builder, schema validator, tests, and evaluator use only the Python standard library.

```bash
python3 scripts/make_splits.py
python3 scripts/validate_benchmark.py data/raw/graded-exams.jsonl data/processed/train.jsonl data/processed/validation.jsonl data/processed/test.jsonl
python3 -m unittest discover -s evaluation/tests -v
```

The split manifest is [data/splits/v0.1.json](data/splits/v0.1.json). It preserves all taxonomy labels in train/validation/test. Family constraints yield 155/48/47 records rather than the nominal 70/15/15 record proportions; see the [dataset review](data/processed/v0.1-dataset-review.md).

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

Dataset provenance and outstanding review concerns are in the [dataset card](docs/data-card-v0.1.md). Use [the experiment log template](docs/templates/experiment-log.md) for future runs and write completed findings in `reports/`.

## Prepare supervised training data

Create the SFT dataset from the frozen training split:

```bash
python3 scripts/make_sft_dataset.py
```

This produces [`data/training/v0.1-sft-train.jsonl`](data/training/v0.1-sft-train.jsonl): 155 chat conversations with canonical JSON assistant targets. The converter rejects validation and test inputs, validates every target against the output schema, and records source and output hashes in `data/training/v0.1-sft-train.metadata.json`.

Use [`configs/qlora-v0.1-smoke.yaml`](configs/qlora-v0.1-smoke.yaml) for the first 10-step QLoRA smoke run. Its only purpose is to verify training and adapter loading; evaluate it on validation only before any longer training run.

On a CUDA-enabled Colab runtime, install `transformers`, `accelerate`, `bitsandbytes`, `peft`, and `pyyaml`, then run:

```bash
python3 scripts/run_qlora_smoke.py --config configs/qlora-v0.1-smoke.yaml
```

The run verifies the frozen training-data hash before training, saves the adapter and training metadata below `models/`, then reloads the adapter from disk before reporting success.

## Run a prompt-only baseline in Google Colab

Use [01_prompt_baselines_colab.ipynb](notebooks/01_prompt_baselines_colab.ipynb) as the central Colab experiment runner. It clones this repository, validates the selected split, loads a configurable open-weight model in 4-bit mode, writes outputs, and invokes the repository evaluator.

Keep the repository scripts—not the notebook—as the source of truth for dataset preparation and scoring. Start with the validation split and the zero-shot prompt; do not inspect the test split until the model and prompt are frozen. Colab storage is ephemeral, so download or copy the ignored `runs/` and `results/` directories to Drive at the end of every run.

### Run a new baseline

The notebook defaults to a 10-example validation smoke test. Use a new `RUN_ID` for every run. Set `MAX_EXAMPLES = None` only after checking smoke-test outputs, and keep the model revision fixed when comparing prompts. Each run saves `gold.jsonl`, `prompt.md`, `metadata.json`, raw outputs, evaluator results, and a notebook source snapshot under `runs/<RUN_ID>/`; preserve the run ZIP and executed notebook before ending the Colab session.

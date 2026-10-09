# MURL LLM Math-Grading Pipeline — Project Plan

## 1. Project goal and research question

Build and evaluate an LLM-assisted grader for text transcripts of worked Calc I–II integration solutions. The system should return a structured grading decision, identify the first incorrect step when applicable, assign an integration-error category, and produce concise feedback for a grader or student. Image and OCR input are a later extension, not a first-phase requirement.

### Primary research question

To what extent does supervised, parameter-efficient fine-tuning improve structured mathematical grading over prompt-only, off-the-shelf models for worked integration-solution transcripts?

### Core hypotheses

1. Fine-tuning improves adherence to a structured grading format and error-taxonomy prediction.
2. A training mix designed for step-level reasoning and first-error localization improves those metrics beyond prompt-only baselines.
3. Performance varies by integration topic, difficulty, and error type; a family-disjoint benchmark can identify those limitations.

### Scope boundaries

- Focus initially on single-variable integration topics from Calc I–II.
- Grade worked solutions, not only final answers.
- Use synthetic and curated benchmark data before collecting real student work.
- Defer OCR, handwritten-image inputs, and noise-robustness claims until the text benchmark is stable.
- Treat the model as a grader-assistance tool, not an autonomous source of final course grades.
- Keep the first semester outcome reproducible: dataset, baselines, one or two fine-tuned models, evaluation report, and demo.

---

## 2. System design

```text
Problem + reference solution
          + student worked solution
                       |
          text transcript
                       |
        prompt-only or fine-tuned model
                       |
       validated structured grading output
                       |
metrics, error analysis, grader-facing display
```

Each example should retain its original problem, canonical solution, student solution, source type, and gold grading labels. This makes every experiment traceable.

---

## 3. Workstreams and milestones

## A. Research specification and repository setup

**Objective:** Make the task measurable before generating data or training a model.

### Tasks

- Write a one-page task specification with input, output, assumptions, and non-goals.
- Define the integration topic coverage: substitution, integration by parts, trigonometric integrals/substitution, partial fractions, improper integrals, and definite integrals/FTC where appropriate.
- Create a repository structure for data, prompts, models, evaluation, reports, and experiment configs.
- Add a configuration file for random seeds, model names, dataset versions, and paths.
- Establish experiment logging conventions and a short data/model card template.

### Milestone A — Specification freeze

**Deliverables:** task specification, repository scaffold, initial taxonomy, experiment log template, and definition of success metrics.

**Exit criteria:** another researcher can run a baseline using documented inputs and output expectations.

---

## B. Dataset generation: Calc I–II integration solutions

**Objective:** Create a diverse, auditable corpus of correct and incorrect worked solutions.

### Tasks

- Build a problem bank, balanced across integration methods and difficulty levels.
- Generate or author canonical step-by-step reference solutions.
- Create student-style solutions including correct work, recoverable mistakes, and incorrect final results.
- Inject realistic error types at a known step rather than relying only on arbitrary wrong answers.
- Create train/validation/test splits by problem family so near-duplicate integrals do not leak across splits.
- Manually review a representative sample for mathematical correctness and label consistency.

### Suggested data fields

```json
{
  "example_id": "int_by_parts_0042",
  "topic": "integration_by_parts",
  "difficulty": "medium",
  "problem": "...",
  "reference_solution": ["..."],
  "student_solution": ["..."],
  "final_answer_status": "incorrect",
  "first_error_step": 3,
  "error_taxonomy": "integration_by_parts",
  "gold_feedback": "...",
  "source": "synthetic"
}
```

### Milestone B — Versioned benchmark v0.1

**Deliverables:** dataset card, problem bank, labeled clean examples, split manifest, and review notes.

**Exit criteria:** test set is held out; every incorrect example has a defensible first error and taxonomy label.

---

## C. OCR and OCR-noise pipeline

**Status:** Deferred extension. Do not begin this workstream until the text benchmark, baselines, and first adaptation experiment are complete.

**Future objective:** Measure robustness to the kinds of transcription defects found in handwritten math.

### Tasks

- Choose one OCR system appropriate for mathematical handwriting or document transcription.
- Create a small set of representative handwritten or rendered solution images, if permitted by data policy.
- Store OCR transcript, confidence/metadata when available, and aligned clean reference text.
- Implement controlled text-noise transformations for common failures: symbol confusion, missing superscripts, lost parentheses, line merges/splits, integral/sign errors, and variable substitutions.
- Define low, medium, and high noise conditions using a documented transformation recipe.
- Keep the clean source and noisy derivative linked by example ID.

### Future Milestone C — OCR benchmark slice

**Deliverables:** OCR pipeline script/notebook, aligned clean/OCR pairs, noise generator, and qualitative error gallery.

**Exit criteria:** the evaluation harness can run the same examples under clean, OCR, and controlled-noise conditions. This is not required for the first-semester text-to-text result.

---

## D. Output contract and structured grading schema

**Objective:** Make outputs machine-evaluable and useful to a human grader.

### Tasks

- Define a JSON schema with constrained labels and required fields.
- Include schema validation and a recovery path for invalid model output.
- Specify grading labels and the integration-error taxonomy.
- Separate evidence/brief rationale from student-facing feedback to reduce unsupported explanations.
- Pilot the schema on 20–30 examples and revise ambiguous labels.

### Initial schema

```json
{
  "overall_correctness": "correct | minor_error | incorrect | insufficient_work",
  "first_error_step": 0,
  "error_taxonomy": "none | algebra | sign | antiderivative | substitution | integration_by_parts | trig_identity | partial_fractions | bounds | other",
  "evidence": "brief reference to the relevant step",
  "grader_feedback": "concise actionable feedback",
  "confidence": 0.0
}
```

Use `null` for `first_error_step` when there is no identifiable mathematical error or the work is insufficient.

### Milestone D — Validated output contract

**Deliverables:** JSON schema, taxonomy guide with examples, validation utility, and prompt template.

**Exit criteria:** every baseline result can be parsed or recorded as an explicit format failure.

---

## E. Model shortlist and prompt-only baselines

**Objective:** Establish credible off-the-shelf comparisons before fine-tuning.

### Candidate shortlist

- Qwen family: primary open-weight candidate; select a size that fits available compute.
- Phi family: compact baseline with a different training profile.
- Kimi open-weight model, if hardware, license, and inference setup are practical.
- Optional control: a small instruction-tuned model with no math-specific adaptation.

Record exact model revision, parameter count, license, quantization, hardware, decoding parameters, and prompt version. Do not compare models under different unrecorded conditions.

### Tasks

- Implement a zero-shot structured-grading prompt.
- Implement a few-shot variant using only training examples.
- Fix decoding settings for the primary comparison; optionally test a small temperature sweep.
- Run the text-transcript benchmark conditions.
- Save raw outputs, parsed outputs, latency, token counts if available, and failures.

### Milestone E — Baseline report

**Deliverables:** reproducible inference runner, baseline table, sample outputs, and initial error analysis.

**Exit criteria:** at least two off-the-shelf models have comparable results on the frozen validation/test protocol.

---

## F. Supervised QLoRA/SFT fine-tuning

**Objective:** Test whether adaptation improves format reliability and mathematical grading quality.

### Training plan

1. Choose the best practical baseline model using validation performance and compute budget.
2. Train one supervised QLoRA/SFT adaptation targeting the structured output format and step-level error localization.
3. Only if the first adaptation is reliable and time permits, compare a second training-data mix that emphasizes first-error localization.
4. Keep training, validation, and test problem families disjoint.
5. Compare each checkpoint to its own untouched base model under the identical evaluator.

### Tasks

- Convert labeled examples into training conversations with canonical JSON targets.
- Configure Unsloth/QLoRA runs with versioned configs.
- Start with a small smoke-test run to validate loss, parsing, and checkpoint loading.
- Tune only a limited, documented set of hyperparameters: learning rate, epochs, rank, target modules, sequence length, and training-data mix.
- Monitor overfitting using validation metrics and qualitative samples.
- Save adapters, training logs, configs, and dataset hashes.

### Milestone F — Adapted model checkpoints

**Deliverables:** one adapted checkpoint, reproducible training configuration, training curves, and comparison with its untouched base model.

**Exit criteria:** the adapted model loads reliably and outperforms, matches, or clearly explains why it does not outperform its untouched base model on validation data.

---

## G. Evaluation harness

**Objective:** Score every model consistently and make results easy to reproduce.

### Required metrics

- Overall correctness classification: accuracy and macro F1.
- First-wrong-step localization: exact match and distance from gold step.
- Error taxonomy: macro F1 and confusion matrix.
- Structured-output validity rate.
- Feedback quality: small human rubric for correctness, specificity, helpfulness, and appropriate tone.
- Operational metrics: latency, memory/compute notes, and inference cost where measurable.

### Tasks

- Implement deterministic data loading and schema validation.
- Build metric functions and a per-example results file.
- Add stratified summaries by topic, difficulty, and error type.
- Create a qualitative review set that includes successes, disagreements, hallucinated feedback, and malformed outputs.
- Add a single command/config that reruns an experiment end-to-end.

### Milestone G — Reproducible evaluation suite

**Deliverables:** evaluator, metric report, plots/tables, error-analysis notebook or report, and rerun instructions.

**Exit criteria:** a new model can be evaluated with one configuration and produces an archived results bundle.

---

## H. Experiments and ablations

**Objective:** Answer the research question with focused comparisons, not an unbounded model sweep.

### Required comparisons

| Experiment | Comparison | Question answered |
| --- | --- | --- |
| Prompting baseline | zero-shot vs few-shot | How much does prompt design help without adaptation? |
| Adaptation | base vs one supervised QLoRA/SFT adaptation | Does adaptation improve grading over its own base model? |
| Training objective | format-focused vs localization-focused mix, if time permits | Does the training mix affect first-error localization? |
| Output contract | free text vs JSON schema | Does structure improve reliability and grading usability? |

### Optional ablations (only if time permits)

- Vary QLoRA rank or training-data size.
- Compare chain-of-thought-style internal reasoning against concise evidence-only outputs, without exposing unnecessary reasoning to users.
- Test topic transfer: train on selected integration types and evaluate on a held-out type.
- Compare generated versus manually authored error examples.

### Milestone H — Final experiment matrix

**Deliverables:** pre-registered experiment table, completed runs, statistical uncertainty where feasible, and concise conclusions for each hypothesis.

**Exit criteria:** all headline claims are tied to a frozen metric, dataset version, and model configuration.

---

## I. Reporting and presentation

**Objective:** Package the project as a clear research artifact.

### Tasks

- Write a short methods section: data construction, labels, models, training, and evaluation.
- Report limitations: synthetic-data bias, error-label ambiguity, compute limits, and the need for human oversight.
- Include examples that show both helpful feedback and known failure modes.
- Prepare a poster, slide deck, or research memo with one pipeline diagram, one main-results table, robustness plot, and error-analysis figure.
- Create a README with setup, data provenance, reproduction steps, and ethical-use note.

### Milestone I — Research package

**Deliverables:** final report, reproducibility README, results artifacts, presentation/demo, and a short project retrospective.

**Exit criteria:** a reviewer can understand the question, reproduce a key result, and inspect the limits of the system.

---

## J. Dashboard and stretch goals

Build these only after the core benchmark, baseline, and fine-tuning experiments work.

- Lightweight grader dashboard: paste/upload a solution transcript, view validated JSON, and see highlighted first-error step.
- Human-in-the-loop correction interface to collect corrected labels for future data versions.
- OCR/image-to-grading evaluation, once the text benchmark is stable.
- Retrieval of worked-example rubrics or course-specific grading policies.
- Active-learning loop to prioritize examples where models disagree or have low confidence.
- Expand beyond integration to derivatives, differential equations, or linear algebra.

### Milestone J — Demonstration prototype

**Deliverables:** small local demo, screenshots, and clear disclaimer that results require instructor review.

---

## 4. Suggested 12-week semester timeline

| Week | Focus | Deliverable |
| --- | --- | --- |
| 1 | Scope, taxonomy, repository | Milestone A |
| 2–3 | Problem bank and clean labeled data | Dataset draft |
| 4 | Data review, held-out split, dataset card | Milestone B |
| 5 | Annotation guide, second review, output schema | Milestone D |
| 6 | Prompt-only model runs | Milestone E |
| 7 | Evaluation harness and baseline analysis | Milestone G (initial) |
| 8 | Adaptation smoke test | Training checkpoint |
| 9–10 | Finalize one adaptation and evaluate it | Milestone F |
| 11 | Focused error analysis; optional training-mix ablation | Milestone H |
| 12 | Report, demo/dashboard stretch, presentation | Milestone I; optional J |

If the semester is shorter, preserve this minimum viable sequence: **A → B → D → E → F → G → I**. Treat OCR, image input, and dashboard work as optional only after the core text-to-text comparison is complete.

---

## 5. Weekly operating rhythm

- **Beginning of week:** choose one measurable deliverable and record the dataset/model/config version.
- **During implementation:** run a small smoke test before long generation or training jobs.
- **End of week:** archive outputs, update the experiment log, review 10–20 examples qualitatively, and write down failures or changed assumptions.
- **Advisor/team check-in:** show one artifact (dataset sample, evaluation table, error gallery, or plot), not only a status update.

## 6. Minimum viable project definition

The project is complete when it includes:

1. A versioned Calc I–II integration benchmark of text transcripts, with a documented annotation guide and second review of a representative sample.
2. A validated structured grading schema and taxonomy.
3. Prompt-only results for at least two off-the-shelf models.
4. One supervised QLoRA/SFT adaptation of a selected open-weight model, compared with its untouched base model. A second adaptation or training-mix ablation is optional.
5. A reproducible evaluation harness comparing base and adapted models.
6. A short report that states results, robustness findings, limitations, and examples of failure.

## 7. Risks and practical mitigations

| Risk | Mitigation |
| --- | --- |
| Dataset is too large to finish or review | Start with 200–300 high-quality examples, then expand only after baselines run. |
| Generated labels contain math errors | Sample-review every data batch and preserve the generation/review provenance. |
| First-error labels are ambiguous | Publish annotation rules for equivalent paths, propagated errors, multiple errors per line, missing work, and OCR ambiguity; second-review a representative sample. |
| Synthetic data does not resemble student work | Include varied solution styles and a small curated set; explicitly limit claims to the benchmark distribution. |
| Compute is limited | Use smaller model sizes, adapters, short smoke tests, and narrow the model shortlist early. |
| Metrics look good but feedback is poor | Maintain a human-reviewed feedback rubric and qualitative error gallery. |
| Leakage across similar problems | Split by template/problem family before model training and prompt-example selection. |

## 8. Definition of done for each experiment

Before treating a result as reportable, save:

- Dataset version and split manifest.
- Base model revision and quantization/inference settings.
- Prompt version or training configuration.
- Random seed(s) and run date.
- Raw model outputs and parsed structured outputs.
- Aggregate metrics plus per-topic and error-type breakdown.
- At least five representative successes and five representative failures.

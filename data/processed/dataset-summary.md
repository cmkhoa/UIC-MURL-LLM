# Dataset summary — `data/raw/graded-exams.jsonl`

Benchmark version: `v0.1-pilot` extension (annotation guide `v0.1`).
Modality: text transcripts only; no OCR, images, or transcription noise.
Records: **250**, one JSON object per line. Problem families: **34**. Distinct problem statements: **85**.

Every record is authored for this benchmark. No graded-exam transcripts were supplied for
this build, so `source` is `authored` everywhere and `grader_mark` is `null` everywhere
(grader comments are never invented; see `data/processed/validation-report.md`).

## Record schema

The field names follow `data/raw/pilot-example-template.json`, plus three fields added for
provenance and review bookkeeping:

| Field | Type | Notes |
| --- | --- | --- |
| `example_id` | string | unique, `int_<topic-abbrev>_<4-digit>` |
| `problem_family` | string | split unit; records in a family are variations of one template |
| `topic` | string | one of the eight topics below |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `problem` | string | problem statement |
| `reference_solution` | list[string] | canonical ordered steps |
| `student_solution` | list[string] | student transcript, ordered steps (1..n) |
| `overall_correctness` | string | `correct`, `minor_error`, `incorrect`, `insufficient_work` |
| `first_error_step` | int or null | one-based index into `student_solution`; null for `correct` and `insufficient_work` |
| `error_taxonomy` | string | one of the ten permitted labels |
| `gold_feedback` | string | concise student-facing feedback |
| `source` | string | `authored` or `graded_exam` |
| `annotation_guide_version` | string | `v0.1` |
| `review_status` | string | `needs_second_review` or `single_annotator` |
| `review_notes` | string or null | added field: rationale for flagged/ambiguous records |
| `review_reason` | string or null | added field: why the record entered the review queue |
| `grader_mark` | string or null | added field: preserved grader comment from a real exam; `null` for authored records |

## Counts by correctness label

| Label | Count | Share |
| --- | --- | --- |
| `correct` | 98 | 39.2% |
| `minor_error` | 50 | 20.0% |
| `incorrect` | 91 | 36.4% |
| `insufficient_work` | 11 | 4.4% |
| **total** | **250** | 100% |

## Counts by topic

| Topic | Total | `correct` | `minor_error` | `incorrect` | `insufficient_work` |
| --- | --- | --- | --- | --- | --- |
| `mixed_review` | 47 | 16 | 9 | 20 | 2 |
| `substitution` | 42 | 20 | 9 | 12 | 1 |
| `integration_by_parts` | 36 | 14 | 9 | 11 | 2 |
| `definite_integrals` | 34 | 15 | 5 | 12 | 2 |
| `partial_fractions` | 27 | 10 | 7 | 9 | 1 |
| `improper_integrals` | 24 | 8 | 3 | 11 | 2 |
| `trig_integrals` | 22 | 8 | 4 | 9 | 1 |
| `trig_substitution` | 18 | 7 | 4 | 7 | 0 |
| **total** | **250** | **98** | **50** | **91** | **11** |

## Counts by difficulty

| Difficulty | Total | `correct` | `minor_error` | `incorrect` | `insufficient_work` |
| --- | --- | --- | --- | --- | --- |
| `easy` | 41 | 18 | 7 | 16 | 0 |
| `medium` | 123 | 47 | 25 | 46 | 5 |
| `hard` | 86 | 33 | 18 | 29 | 6 |

## Counts by error taxonomy

Taxonomy labels are drawn only from the ten permitted values in `docs/annotation-guide.md`.

| Label | Count | Share | `minor_error` | `incorrect` | `insufficient_work` |
| --- | --- | --- | --- | --- | --- |
| `none` | 107 | 42.8% | 0 | 0 | 9 |
| `algebra` | 24 | 9.6% | 11 | 13 | 0 |
| `sign` | 21 | 8.4% | 11 | 10 | 0 |
| `antiderivative` | 39 | 15.6% | 15 | 24 | 0 |
| `substitution` | 18 | 7.2% | 9 | 9 | 0 |
| `integration_by_parts` | 5 | 2.0% | 0 | 5 | 0 |
| `trig_identity` | 6 | 2.4% | 1 | 5 | 0 |
| `partial_fractions` | 4 | 1.6% | 0 | 4 | 0 |
| `bounds` | 15 | 6.0% | 3 | 12 | 0 |
| `other` | 11 | 4.4% | 0 | 9 | 2 |

The 107 `none` records are the 98 `correct` records plus 9 `insufficient_work` records whose shown work
supports no step-level error claim at all. The remaining 2 `insufficient_work` records state a demonstrably wrong
final answer with no intermediate work, and are labelled `other`; both groups keep `first_error_step: null`.

## Counts by source

| Source | Count | Share |
| --- | --- | --- |
| `authored` | 250 | 100% |
| `graded_exam` | 0 | 0% |

## Counts by problem family

The family is the split unit: variations of one underlying template share a family, so a
family-disjoint train/validation/test split cannot leak near-duplicate integrals across splits.
With 34 families, a 70/15/15 family-level split gives roughly 24/5/5 families.

| Problem family | Topic | Records | Distinct problems | correct / minor / incorrect / insufficient |
| --- | --- | --- | --- | --- |
| `definite_integration_by_parts` | `definite_integrals` | 8 | 2 | 4 / 1 / 2 / 1 |
| `definite_polynomial_ftc` | `definite_integrals` | 8 | 2 | 3 / 1 / 3 / 1 |
| `definite_substitution_change_of_bounds` | `definite_integrals` | 9 | 3 | 4 / 1 / 4 / 0 |
| `definite_trigonometric_ftc` | `definite_integrals` | 9 | 3 | 4 / 2 / 3 / 0 |
| `improper_endpoint_discontinuity` | `improper_integrals` | 8 | 3 | 3 / 1 / 3 / 1 |
| `improper_exponential_decay` | `improper_integrals` | 8 | 2 | 2 / 1 / 4 / 1 |
| `improper_infinite_limit_power` | `improper_integrals` | 8 | 3 | 3 / 1 / 4 / 0 |
| `ibp_cyclic_exponential_trig` | `integration_by_parts` | 5 | 2 | 2 / 1 / 2 / 0 |
| `ibp_inverse_function` | `integration_by_parts` | 6 | 2 | 2 / 1 / 2 / 1 |
| `ibp_logarithm` | `integration_by_parts` | 8 | 3 | 3 / 3 / 2 / 0 |
| `ibp_polynomial_times_exponential` | `integration_by_parts` | 6 | 2 | 2 / 1 / 2 / 1 |
| `ibp_polynomial_times_trig` | `integration_by_parts` | 5 | 2 | 3 / 1 / 1 / 0 |
| `ibp_repeated_application` | `integration_by_parts` | 6 | 2 | 2 / 2 / 2 / 0 |
| `mixed_algebraic_rewrite` | `mixed_review` | 9 | 3 | 3 / 2 / 4 / 0 |
| `mixed_completing_the_square` | `mixed_review` | 9 | 2 | 3 / 2 / 3 / 1 |
| `mixed_definite_application` | `mixed_review` | 10 | 3 | 3 / 2 / 5 / 0 |
| `mixed_exponential_rational` | `mixed_review` | 9 | 3 | 4 / 1 / 4 / 0 |
| `mixed_method_selection` | `mixed_review` | 10 | 3 | 3 / 2 / 4 / 1 |
| `partial_fractions_distinct_linear` | `partial_fractions` | 8 | 2 | 3 / 1 / 3 / 1 |
| `partial_fractions_improper_rational` | `partial_fractions` | 6 | 2 | 2 / 2 / 2 / 0 |
| `partial_fractions_irreducible_quadratic` | `partial_fractions` | 7 | 2 | 3 / 2 / 2 / 0 |
| `partial_fractions_repeated_linear` | `partial_fractions` | 6 | 2 | 2 / 2 / 2 / 0 |
| `u_substitution_exponential_inner` | `substitution` | 7 | 3 | 3 / 2 / 2 / 0 |
| `u_substitution_linear_inner_power` | `substitution` | 7 | 3 | 3 / 2 / 2 / 0 |
| `u_substitution_logarithmic_quotient` | `substitution` | 8 | 3 | 4 / 1 / 3 / 0 |
| `u_substitution_polynomial_composite` | `substitution` | 8 | 3 | 3 / 2 / 2 / 1 |
| `u_substitution_radical` | `substitution` | 6 | 3 | 3 / 1 / 2 / 0 |
| `u_substitution_trig_power` | `substitution` | 6 | 3 | 4 / 1 / 1 / 0 |
| `trig_even_power_half_angle` | `trig_integrals` | 8 | 3 | 3 / 2 / 3 / 0 |
| `trig_odd_power_sine` | `trig_integrals` | 7 | 2 | 2 / 1 / 3 / 1 |
| `trig_secant_tangent_powers` | `trig_integrals` | 7 | 3 | 3 / 1 / 3 / 0 |
| `trig_substitution_secant_form` | `trig_substitution` | 6 | 2 | 2 / 1 / 3 / 0 |
| `trig_substitution_sine_form` | `trig_substitution` | 6 | 2 | 3 / 2 / 1 / 0 |
| `trig_substitution_tangent_form` | `trig_substitution` | 6 | 2 | 2 / 1 / 3 / 0 |

## Review coverage

| `review_status` | Count | Share |
| --- | --- | --- |
| `needs_second_review` | 60 | 24.0% |
| `single_annotator` | 190 | 76.0% |

`data/processed/review-queue.jsonl` holds the 60 flagged records in full (24% of the
benchmark, above the 20% second-review floor in the annotation guide). The queue covers all
34 families, all 8 topics, all 4 correctness labels, and all 10 taxonomy labels.

## Known distribution caveats

- Solution-step counts run from 1 to 5; the one-step records are deliberate — they are the
  terse or `insufficient_work` styles a grader actually encounters.
- `partial_fractions`, `integration_by_parts`, and `trig_identity` are the smallest taxonomy
  cells (4, 5, and 6 records). Macro-F1 over the taxonomy will be noisy in those cells; report
  per-label support alongside the metric.
- `easy` is the smallest difficulty stratum (41 records), concentrated in substitution and
  basic FTC, because the harder topics do not have convincing easy variants.
- Topic sizes are unequal by design (18-47); stratified metrics should be read with the
  per-topic support shown above.

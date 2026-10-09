# Validation report — integration-grading benchmark v0.1 (250 records)

Artifacts validated:

- `data/raw/graded-exams.jsonl` — 250 records
- `data/processed/review-queue.jsonl` — 60 records (a subset of the above, byte-identical per record)

Validator: `scripts/validate_benchmark.py` (standalone, no third-party dependencies).

```bash
python3 scripts/validate_benchmark.py data/raw/graded-exams.jsonl data/processed/review-queue.jsonl
```

Final run:

```
data/raw/graded-exams.jsonl: 250 records, 0 violations
data/processed/review-queue.jsonl: 60 records, 0 violations
```

## 1. Automated checks and results

| # | Check | Result |
| --- | --- | --- |
| 1 | Every line parses as JSON; file is newline-delimited with no trailing blank record | 250/250 pass |
| 2 | All 17 required fields present on every record | 250/250 pass |
| 3 | `example_id` unique | 250 unique ids |
| 4 | `overall_correctness` ∈ {`correct`, `minor_error`, `incorrect`, `insufficient_work`} | 250/250 pass |
| 5 | `error_taxonomy` ∈ the 10 permitted labels | 250/250 pass; no out-of-vocabulary label |
| 6 | `difficulty` ∈ {`easy`, `medium`, `hard`} | 250/250 pass |
| 7 | `source` ∈ {`authored`, `graded_exam`} | 250 `authored`, 0 `graded_exam` |
| 8 | `review_status` ∈ {`needs_second_review`, `single_annotator`} | 60 / 190 |
| 9 | `correct` ⇒ `first_error_step` is null **and** `error_taxonomy` is `none` | 98/98 pass |
| 10 | `insufficient_work` ⇒ `first_error_step` is null | 11/11 pass |
| 11 | `minor_error` / `incorrect` ⇒ `first_error_step` is an integer in `1..len(student_solution)` | 141/141 pass |
| 12 | `minor_error` / `incorrect` ⇒ `error_taxonomy` ≠ `none` | 141/141 pass |
| 13 | No empty `problem`, `gold_feedback`, `problem_family`, `topic`, or solution step | 250/250 pass |
| 14 | `reference_solution` has ≥ 2 steps; `student_solution` has ≥ 1 step | 250/250 pass |
| 15 | `source: authored` ⇒ `grader_mark` is null (no invented grader comments) | 250/250 pass |
| 16 | No names, student or course IDs, section numbers, dates, or e-mail addresses | regex scan over the serialized corpus returned 0 hits |

## 2. Mathematical verification

Correctness of the reference solutions was not assumed. Every indefinite-integral family was
checked numerically during generation by central-difference differentiation of the claimed
antiderivative against the integrand at three interior sample points per problem
(relative tolerance 1e-4), and every definite-integral and application family was checked by
composite Simpson quadrature against the claimed value (relative tolerance 1e-4). Improper
integrals were checked on a finite truncation against the exact partial value of the same
antiderivative, since the divergence or the limit itself is the graded claim rather than a
quadrature target.

All 85 distinct problems passed. Two reference results were revised during this pass:

- The `trig_substitution_secant_form` problem `∫ √(x² - 4)/x dx` is recorded with the
  equivalent `arccos(2/x)` form rather than `arcsec(x/2)`, after the numeric check confirmed
  the two agree on `x > 2`.
- Numeric checks for the three improper families were re-specified against finite truncations
  after uniform-grid quadrature over a near-infinite interval produced a meaningless value
  for `∫₁^∞ dx/x²`. This was a defect in the *check*, not in the data; the reference solutions
  were unchanged.

## 3. Label conventions adopted during annotation

These resolve cases the annotation guide leaves implicit. They are recorded here so a second
reviewer applies the same rule.

1. **`insufficient_work` taxonomy.** The guide fixes `first_error_step: null` but not the
   taxonomy. Records whose shown work supports no step-level claim at all use `none`
   (9 records). Records that state a demonstrably wrong final answer with no intermediate work
   use `other` (2 records), because an error is present but cannot be localized. All 11 keep
   `first_error_step: null`.
2. **Grading the written work, not the final answer.** Where a student asserts an invalid step
   but reaches a correct answer, the invalid step is labelled (guide rule 2). Two records do
   this deliberately: `definite_trigonometric_ftc` (an incorrect antiderivative whose error
   cancels because `sin 2π = 0`) and `mixed_method_selection` (an invalid `v` in a discarded
   integration-by-parts attempt before a correct substitution). Both are in the review queue.
3. **Exploratory and abandoned work.** A student who writes a true but unproductive identity,
   abandons it, and then solves the problem correctly is `correct`: nothing written is invalid.
   Where the abandoned line is itself false, the first error is that line.
4. **Alternative valid methods** (guide rule 1) are labelled `correct` and flagged for second
   review rather than penalized: table-form antiderivatives, `sec`-based vs `tan`-based
   substitutions differing by a constant, combined logarithms, add-and-subtract tricks, and
   symmetry arguments.
5. **Omitted constant of integration** on an otherwise correct indefinite integral is
   `minor_error` with taxonomy `antiderivative`, per the guide's wording "omitted constant
   where material" (6 records). Omitted absolute values inside a logarithm are treated the
   same way when the integrand is defined on both sides of the singularity.
6. **Downstream propagation** is never relabelled: the earliest invalid step carries the label
   even when later lines faithfully inherit the mistake (guide rule 4).
7. **Non-integration errors inside an integration problem** — a misstated p-test criterion, a
   mishandled indeterminate form, a reversed upper/lower curve, a missing `1/(b-a)` in an
   average value, or an invented "quotient rule for integrals" — use `other`, with the reason
   written into `review_notes` as the taxonomy table requires.

## 4. Corrections applied before release

| Finding | Records affected | Action |
| --- | --- | --- |
| `first_error_step` pointed at index 3 of a two-step transcript (`ibp_logarithm`, missing `+C` case) | 1 | Index corrected to 2, the step that actually omits the constant. Caught by check 11. |
| Reference solutions in `u_substitution_polynomial_composite` rendered fractional coefficients without parentheses (`3/10u⁵`) | 8 | Formatting corrected to `(3/10)u⁵`. No change to the mathematics or to any label. |

## 5. Records excluded during balancing

Ten authored records were cut to keep the correctness mix balanced at 250 records. Each was a
near-duplicate error pattern *within its own family*: a second record carrying the same
`error_taxonomy` and the same underlying mistake as another record in that family. The record
kept in each pair was the one whose error was less purely arithmetic. No record was excluded
for being mathematically wrong, and nothing was excluded from a family that would have been
left with fewer than five records.

| Family | Taxonomy of the dropped duplicate |
| --- | --- |
| `u_substitution_trig_power` | `sign` |
| `u_substitution_radical` | `antiderivative` |
| `ibp_polynomial_times_exponential` | `integration_by_parts` |
| `ibp_polynomial_times_trig` | `sign` |
| `ibp_cyclic_exponential_trig` | `algebra` |
| `trig_even_power_half_angle` | `trig_identity` |
| `partial_fractions_distinct_linear` | `partial_fractions` |
| `partial_fractions_repeated_linear` | `partial_fractions` |
| `partial_fractions_improper_rational` | `algebra` |
| `definite_substitution_change_of_bounds` | `bounds` |

Eleven `minor_error` records were authored in the same pass, reusing the problem and reference
solution of an existing record in the same family, so no new reference mathematics entered the
corpus without passing the numeric check in section 2.

## 6. De-identification

No record contains a name, student or instructor identifier, course or section number, term,
date, institution, or e-mail address. All 250 records are authored problems and authored
student-style transcripts; none transcribes a real exam. `grader_mark` is `null` on every
record because no de-identified grader comments were supplied — authored grader comments would
misrepresent provenance, so none were written. If real transcripts are added later, those
records take `source: "graded_exam"`, carry any supplied comment verbatim in `grader_mark`, and
must pass checks 1-16 unchanged.

## 7. Residual risks and limitations

- **Single-annotator labels.** 190 records carry one annotator's judgment. The 60-record queue
  is the second-review sample; disputes should be resolved and recorded in provenance before
  the benchmark is frozen as v0.1.
- **Small taxonomy cells.** `partial_fractions` (4), `integration_by_parts` (5), and
  `trig_identity` (6) are too small for stable per-label F1. Either report support alongside
  macro-F1 or expand these cells before drawing topic-transfer conclusions.
- **`minor_error` vs `incorrect` is the softest boundary** in the label set, and it is the
  judgment most likely to move under second review. The operating rule used here: if correcting
  the single flagged step repairs the solution without changing the method, it is
  `minor_error`; if the method, the validity, or the shape of the result changes, it is
  `incorrect`.
- **Numeric verification is sampling, not proof.** Three sample points per problem with a 1e-4
  relative tolerance would not catch an error that vanishes at those points. Spot-checking
  symbolically is recommended if a symbolic algebra package becomes available in the project
  environment.
- **No OCR or transcription noise**, per the task spec; these are clean transcripts only.

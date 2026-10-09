# Annotation guide

## Unit of annotation

Each student solution is an ordered list of numbered steps. Label the earliest step that contains a mathematical error, not the first downstream consequence of that error.

## General rules

1. Accept algebraically equivalent methods and valid alternative intermediate forms.
2. Label a step only when its claim is invalid under a reasonable interpretation of the visible work.
3. If one step contains multiple errors, assign the most causally central error and note the others in review notes.
4. If later steps faithfully propagate an earlier error, do not relabel them as the first error.
5. Use `insufficient_work` and `first_error_step: null` when the student has not shown enough work to localize a mathematical error fairly.
6. In the text-only phase, flag unparseable notation for review; do not infer an OCR error category.

## Correctness labels

- `correct`: all material mathematical work and final answer are correct.
- `minor_error`: a localized, non-conceptual error whose correction leaves the method substantially intact.
- `incorrect`: a material error changes the method, result, or validity of the solution.
- `insufficient_work`: shown work cannot support a defensible step-level judgment.

## Error taxonomy

| Label | Use for |
| --- | --- |
| `none` | No mathematical error is identified. |
| `algebra` | Invalid simplification, factoring, expansion, or arithmetic. |
| `sign` | Incorrect sign introduced or lost. |
| `antiderivative` | Incorrect integration rule, coefficient, exponent, or omitted constant where material. |
| `substitution` | Invalid substitution, differential conversion, or back-substitution. |
| `integration_by_parts` | Invalid choice/application of the integration-by-parts formula. |
| `trig_identity` | Incorrect trigonometric identity or transformation. |
| `partial_fractions` | Incorrect decomposition, coefficient solving, or component integration. |
| `bounds` | Incorrect limit transformation or evaluation of a definite/improper integral. |
| `other` | A material integration error not covered above; explain it in review notes. |

## Review protocol

Annotators complete the label fields independently for the pilot set. A second reviewer checks at least 20% of the full benchmark and every disputed pilot record. Preserve the initial labels, resolution, and rationale in dataset provenance.

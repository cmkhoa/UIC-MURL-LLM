# Task specification: structured grading of integration-solution transcripts

## Purpose

Given an integration problem, a canonical worked solution, and a student’s text transcript, produce a validated JSON grading record. The system is decision support for an instructor or grader; it must not autonomously assign course grades.

## Inputs

- Problem statement.
- Canonical reference solution represented as ordered steps.
- Student solution represented as ordered text steps.
- Optional benchmark metadata: topic, difficulty, and problem family.

The first-phase benchmark contains text only. Transcription quality is assumed sufficient for a human reader. OCR failures and images are out of scope.

## Required output

```json
{
  "overall_correctness": "correct",
  "first_error_step": null,
  "error_taxonomy": "none",
  "evidence": "All student steps are mathematically valid.",
  "grader_feedback": "Your method and final antiderivative are correct.",
  "confidence": 0.0
}
```

- `overall_correctness`: `correct`, `minor_error`, `incorrect`, or `insufficient_work`.
- `first_error_step`: a one-based student-step index, or `null` if no identifiable mathematical error exists or work is insufficient.
- `error_taxonomy`: one label from the taxonomy in the annotation guide.
- `evidence`: a brief, checkable reference to the relevant student step; it is not a hidden chain of thought.
- `grader_feedback`: concise, actionable, and appropriate for a student.
- `confidence`: a number from 0 through 1.

## Success metrics

The primary metrics are correctness macro F1, first-error exact match and distance, taxonomy macro F1, and JSON validity rate. Report every metric by topic, difficulty, and error type, plus qualitative feedback review.

## Non-goals

- OCR, handwritten images, or robustness claims about transcription noise.
- Automated final grading or policy enforcement.
- Judging stylistic differences when the mathematical work is valid.
- Expanding beyond single-variable integration in the initial benchmark.

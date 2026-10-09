# Dataset card — MURL integration grading v0.1-pilot

## Identification

- Dataset name and version: MURL integration-grading benchmark, `v0.1-pilot`.
- Creation date: 2026-10-08.
- Maintainer: MURL research project.
- License and access restrictions: repository-local research artifact; no external license is asserted. Do not treat it as real student work.

## Purpose and scope

- Intended research use: evaluate structured, text-to-text assistance for grading worked Calc I–II integration solutions.
- Subject and topic coverage: 250 authored records spanning substitution, integration by parts, trigonometric integrals/substitution, partial fractions, improper and definite integrals, and mixed review/application problems.
- Included modality: clean text problem, ordered canonical solution, and ordered student-style solution.
- Excluded modalities and claims: OCR, handwriting/images, dashboard use, real-course grading, and robustness to transcription noise.

## Composition

- 250 examples, 34 problem families, 85 distinct problem statements, and 8 topics. Correctness counts: 98 `correct`, 50 `minor_error`, 91 `incorrect`, and 11 `insufficient_work`.
- Error-taxonomy counts: none 107, algebra 24, sign 21, antiderivative 39, substitution 18, integration_by_parts 5, trig_identity 6, partial_fractions 4, bounds 15, other 11.
- Source mix: 250 authored (100%); no curated or graded-exam records.
- Split rule: all records in a `problem_family` remain together. `data/splits/v0.1.json` assigns 20 families/155 records to train, 7/48 to validation, and 7/47 to test. Every taxonomy label occurs in every split. Each holdout contains 7 of 8 topics; together the held-out splits contain all 8.

## Creation and review

- Canonical-solution authoring procedure: authored family templates and reference steps; prior validation notes report numerical checks for all 85 distinct problems.
- Student-solution generation procedure: authored variants with correct work, local mistakes, material errors, and insufficient-work cases.
- Annotation-guide version: `v0.1` (`docs/annotation-guide.md`).
- Reviewer count, review fraction, and dispute resolution: the stored corpus contains 190 `single_annotator` records and 60 `needs_second_review` records (24%). The guide requires a second reviewer to resolve disputes while preserving provenance; those resolutions have not yet been recorded.

## Limitations and ethical use

- Expected differences from real student work: authored examples have cleaner notation and intentionally represented error patterns; they are not a sample of a course population.
- Known label ambiguity or coverage gaps: `minor_error` versus `incorrect`, insufficient-work taxonomy, abandoned invalid work, and small taxonomy cells require careful review. See `data/processed/v0.1-review-report.md`.
- Appropriate use: grader assistance and research only, with an instructor or other qualified human reviewing all decisions. It must not autonomously assign course grades.

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluate import evaluate
from evaluation.schema import validate_output
from scripts.make_sft_dataset import TRAIN_SOURCE, build_sft_records
from scripts.prepare_run import prepare_run
from scripts.validate_benchmark import validate

ROOT = Path(__file__).resolve().parents[2]


class RunPreparationTests(unittest.TestCase):
    def test_partial_run_scores_selected_examples_and_preserves_missing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ['data/processed/validation.jsonl', 'data/raw/graded-exams.jsonl',
                             'data/splits/v0.1.json', 'prompts/v0.1-zero-shot.md']:
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT / relative).read_bytes())
            with patch('scripts.prepare_run.subprocess.check_output', side_effect=['abc123\n', '']):
                rows, template, run_dir = prepare_run(root, 'smoke', 'validation', 'prompts/v0.1-zero-shot.md', 2)
            gold = [json.loads(line) for line in (run_dir / 'gold.jsonl').read_text().splitlines()]
            output = {key: rows[0][key] for key in ['overall_correctness', 'first_error_step', 'error_taxonomy']}
            output.update(evidence='Fixture evidence', grader_feedback='Fixture feedback', confidence=0.9)
            _, metrics = evaluate(gold, [{'example_id': rows[0]['example_id'], 'output': output}])
            self.assertEqual(metrics['examples'], 2)
            self.assertEqual(metrics['output_validity_rate'], 0.5)
            self.assertEqual(metrics['missing_output_ids'], [rows[1]['example_id']])
            info = json.loads((run_dir / 'metadata.json').read_text())
            self.assertTrue(info['partial_run'])
            self.assertEqual(info['repository_commit'], 'abc123')
            self.assertEqual(info['split_examples'], 48)
            self.assertEqual((run_dir / 'prompt.md').read_text(), template)
            self.assertEqual(len(info['gold_sha256']), 64)

    def test_records_missing_git_provenance_when_source_is_an_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ['data/processed/validation.jsonl', 'data/raw/graded-exams.jsonl',
                             'data/splits/v0.1.json', 'prompts/v0.1-zero-shot.md']:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / relative).read_bytes())
            with patch('scripts.prepare_run.subprocess.check_output', side_effect=subprocess.CalledProcessError(128, 'git')):
                _, _, run_dir = prepare_run(root, 'archive', 'validation', 'prompts/v0.1-zero-shot.md', 1)
            info = json.loads((run_dir / 'metadata.json').read_text())
            self.assertIsNone(info['repository_commit'])
            self.assertFalse(info['repository_dirty'])

    def test_rejects_invalid_limits(self):
        for limit in [0, -1, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                prepare_run(ROOT, 'unused', 'validation', 'prompts/v0.1-zero-shot.md', limit)


class SftDatasetTests(unittest.TestCase):
    def test_builds_valid_targets_from_training_split_only(self):
        records = build_sft_records(TRAIN_SOURCE)
        source = [json.loads(line) for line in TRAIN_SOURCE.read_text(encoding='utf-8').splitlines() if line]
        steps_by_id = {row['example_id']: len(row['student_solution']) for row in source}
        self.assertEqual(len(source), len(records))
        self.assertEqual('system', records[0]['messages'][0]['role'])
        self.assertEqual('user', records[0]['messages'][1]['role'])
        self.assertEqual('assistant', records[0]['messages'][2]['role'])
        for record in records:
            target = json.loads(record['messages'][2]['content'])
            self.assertTrue(validate_output(target, steps_by_id[record['example_id']]).valid)

    def test_rejects_non_training_source(self):
        with self.assertRaises(ValueError):
            build_sft_records(Path('data/processed/validation.jsonl'))


class BenchmarkValidationTests(unittest.TestCase):
    def test_malformed_records_report_errors_without_crashing(self):
        fixture = json.loads((ROOT / 'data/raw/graded-exams.jsonl').read_text().splitlines()[0])
        records = [None, 1, [], 'text']
        for field in ['example_id', 'overall_correctness', 'error_taxonomy', 'difficulty',
                      'source', 'review_status', 'reference_solution', 'student_solution']:
            for value in [None, {}, []]:
                records.append(dict(fixture, **{field: value}))
        for record in records:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'invalid.jsonl'
                path.write_text(json.dumps(record) + '\n')
                errors, _ = validate(path)
                self.assertTrue(errors)

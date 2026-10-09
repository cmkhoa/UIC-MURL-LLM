import unittest

from evaluation.evaluate import evaluate


GOLD = [{
    "example_id": "fixture-1", "topic": "substitution", "difficulty": "easy",
    "student_solution": ["u = x².", "Answer is u²/2."],
    "overall_correctness": "correct", "first_error_step": None, "error_taxonomy": "none",
}]
VALID_OUTPUT = {"example_id": "fixture-1", "output": {
    "overall_correctness": "correct", "first_error_step": None, "error_taxonomy": "none",
    "evidence": "Both steps are valid.", "grader_feedback": "Correct.", "confidence": 0.9,
}}


class EvaluationTests(unittest.TestCase):
    def test_metrics_for_valid_fixture(self):
        per_example, aggregate = evaluate(GOLD, [VALID_OUTPUT])
        self.assertTrue(per_example[0]["output_valid"])
        self.assertEqual(1.0, aggregate["output_validity_rate"])
        self.assertEqual(1.0, aggregate["correctness_accuracy_valid_only"])

    def test_missing_output_is_invalid(self):
        _, aggregate = evaluate(GOLD, [])
        self.assertEqual(0.0, aggregate["output_validity_rate"])
        self.assertEqual(["fixture-1"], aggregate["missing_output_ids"])


if __name__ == "__main__":
    unittest.main()

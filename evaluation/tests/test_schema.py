import json
import unittest

from evaluation.schema import validate_output


class SchemaValidationTests(unittest.TestCase):
    def test_valid_correct_output(self):
        result = validate_output(json.dumps({
            "overall_correctness": "correct", "first_error_step": None, "error_taxonomy": "none",
            "evidence": "All shown steps are valid.", "grader_feedback": "Correct.", "confidence": 0.9,
        }), 3)
        self.assertTrue(result.valid)

    def test_rejects_bad_index_and_taxonomy(self):
        result = validate_output({
            "overall_correctness": "incorrect", "first_error_step": 4, "error_taxonomy": "none",
            "evidence": "Step four.", "grader_feedback": "Revise it.", "confidence": 1.1,
        }, 3)
        self.assertFalse(result.valid)
        self.assertEqual(3, len(result.errors))

    def test_rejects_non_json(self):
        self.assertFalse(validate_output("not json", 1).valid)

    def test_rejects_wrong_json_types_without_crashing(self):
        result = validate_output({
            "overall_correctness": ["correct"], "first_error_step": None, "error_taxonomy": ["none"],
            "evidence": "Shown work.", "grader_feedback": "Review.", "confidence": 0.5,
        }, 1)
        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.run_qlora_smoke import resolve_weighted_schedule


class TrainingScheduleTests(unittest.TestCase):
    def test_two_stage_schedule_is_normalized(self):
        training = {
            "max_steps": 60,
            "save_steps": 30,
            "two_stage_diagnosis": {
                "stage_1": {"steps": 30, "field_weights": {"overall_correctness": 1.0}},
                "stage_2": {"steps": 30, "field_weights": {"overall_correctness": 1.0}},
            },
        }
        schedule = resolve_weighted_schedule(training)
        self.assertEqual(schedule["first_steps"], 30)
        self.assertTrue(schedule["reload_at_boundary"])

    def test_two_stage_steps_must_match_total(self):
        training = {
            "max_steps": 60,
            "save_steps": 30,
            "two_stage_diagnosis": {
                "stage_1": {"steps": 30, "field_weights": {}},
                "stage_2": {"steps": 29, "field_weights": {}},
            },
        }
        with self.assertRaisesRegex(ValueError, "sum"):
            resolve_weighted_schedule(training)

    def test_schedule_types_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            resolve_weighted_schedule({
                "hierarchical_multitask": {"decision_phase": {}, "diagnosis_phase": {}},
                "two_stage_diagnosis": {"stage_1": {}, "stage_2": {}},
            })


if __name__ == "__main__":
    unittest.main()

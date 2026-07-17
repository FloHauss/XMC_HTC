import importlib.util
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "htc" / "hbgl" / "eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("hbgl_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hbgl_eval = load_module()


class HBGLMetricTests(unittest.TestCase):
    def test_label_tokens_do_not_collapse_invalid_predictions_to_zero(self):
        self.assertEqual(hbgl_eval.label_token_to_id("[A_0]", 3), 0)
        self.assertEqual(hbgl_eval.label_token_to_id("[a_2]", 3), 2)
        self.assertIsNone(hbgl_eval.label_token_to_id("not-a-label", 3))
        self.assertIsNone(hbgl_eval.label_token_to_id("[A_3]", 3))

    def test_precision_at_k_and_r_precision(self):
        predictions = [[1, 2, 0], [0, 2, 1]]
        labels = [[1, 0], [0]]
        self.assertEqual(
            hbgl_eval.evaluate_PK(predictions, labels, [1, 3]),
            {"P@1": 1.0, "P@3": 0.5},
        )
        self.assertAlmostEqual(hbgl_eval.evaluate_RP(predictions, labels), 0.75)

    def test_r_precision_rejects_empty_gold(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            hbgl_eval.evaluate_RP([[0]], [[]])

    def test_evaluate_handles_cutoff_larger_than_prediction_vector(self):
        result = hbgl_eval.evaluate(
            [[0.9, 0.1]], [[0]], {0: "zero", 1: "one"}, top_k=50
        )
        self.assertEqual(result["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PECOS_ROOT = REPOSITORY_ROOT / "XMLmodels" / "pecos"
sys.path.insert(0, str(PECOS_ROOT))

from pecos.utils.smat_util import Metrics


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


xr_eval = load_module(
    "xr_ensemble_evaluate", PECOS_ROOT / "run_ensemble" / "ensemble_evaluate.py"
)
xr_preprocess = load_module(
    "xr_preprocess",
    REPOSITORY_ROOT / "XMLPreprocessing" / "XR-Transformer" / "preprocess.py",
)


class XRMetricTests(unittest.TestCase):
    def test_r_precision_handles_short_prediction_rows(self):
        truth = sp.csr_matrix([[1, 0, 1], [0, 1, 0]])
        prediction = sp.csr_matrix([[0, 0, 0], [0, 1, 0]])
        metrics = Metrics.generate(truth, prediction, topk=3)
        self.assertAlmostEqual(metrics.r_prec, 0.5)

    def test_r_precision_rejects_empty_gold(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            Metrics.generate(sp.csr_matrix([[0, 0]]), sp.csr_matrix([[1, 0]]))

    def test_sparse_f1(self):
        truth = sp.csr_matrix([[1, 0, 1], [0, 1, 0]])
        prediction = sp.csr_matrix([[1, 1, 0], [0, 1, 0]])
        micro, macro = xr_eval.sparse_f1(truth, prediction)
        self.assertAlmostEqual(micro, 2 / 3)
        self.assertAlmostEqual(macro, 5 / 9)


class XRPreprocessingTests(unittest.TestCase):
    def test_conversion_uses_shared_label_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            (input_dir / "train_raw_texts.txt").write_text(
                "alpha beta\ngamma delta\n", encoding="utf-8"
            )
            (input_dir / "test_raw_texts.txt").write_text(
                "alpha gamma\n", encoding="utf-8"
            )
            (input_dir / "train_labels.txt").write_text("0\n1\n", encoding="utf-8")
            (input_dir / "test_labels.txt").write_text("2\n", encoding="utf-8")

            xr_preprocess.convert(input_dir, output_dir)

            self.assertEqual(sp.load_npz(output_dir / "Y.trn.npz").shape, (2, 3))
            self.assertEqual(sp.load_npz(output_dir / "Y.tst.npz").shape, (1, 3))
            self.assertTrue((output_dir / "tfidf-attnxml" / "X.trn.npz").is_file())

    def test_empty_gold_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "labels.txt"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no gold labels"):
                xr_preprocess.read_labels(path)


if __name__ == "__main__":
    unittest.main()

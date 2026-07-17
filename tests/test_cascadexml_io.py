import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "XMLmodels" / "CascadeXML" / "io_utils.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cascadexml_io", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


io_utils = load_module()


class CascadeXMLIOTests(unittest.TestCase):
    def test_lf_labels_skip_header_without_consuming_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "Y.trn.npz"
            target.with_suffix(".txt").write_text(
                "2 4\n0,2 1:0.2\n1 3:0.7\n", encoding="utf-8"
            )

            matrix = io_utils.make_csr_labels(4, str(target), lf_data=True)

            np.testing.assert_array_equal(
                matrix.toarray(), [[1, 0, 1, 0], [0, 1, 0, 0]]
            )

    def test_plain_labels_reject_empty_gold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "Y.trn.npz"
            target.with_suffix(".txt").write_text("0,2\n\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no gold labels"):
                io_utils.make_csr_labels(4, str(target), lf_data=False)

    def test_tfidf_preserves_rows_after_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir)
            (dataset / "train.txt").write_text(
                "2 3 3\n0 0:1.0 2:0.5\n1 1:0.25\n", encoding="utf-8"
            )

            matrix = io_utils.make_csr_tfidf(str(dataset))

            self.assertEqual(matrix.shape, (2, 3))
            np.testing.assert_allclose(
                matrix.toarray(), [[1.0, 0.0, 0.5], [0.0, 0.25, 0.0]]
            )


if __name__ == "__main__":
    unittest.main()

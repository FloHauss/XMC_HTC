import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HGCLR_ROOT = REPOSITORY_ROOT / "integrations" / "hgclr"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hgclr_eval = load_module("hgclr_eval", HGCLR_ROOT / "eval.py")
hgclr_binarize = load_module("hgclr_binarize", HGCLR_ROOT / "data" / "binarize.py")


class HGCLRMetricTests(unittest.TestCase):
    def setUp(self):
        self.predictions = [
            [0.9, 0.1, 0.2, 0.8, 0.3],
            [0.1, 0.8, 0.7, 0.6, 0.2],
        ]
        self.labels = [[0, 3], [1]]

    def test_precision_at_k(self):
        self.assertAlmostEqual(
            hgclr_eval._precision_at_k(self.predictions, self.labels, 1), 1.0
        )
        self.assertAlmostEqual(
            hgclr_eval._precision_at_k(self.predictions, self.labels, 3), 0.5
        )
        self.assertAlmostEqual(
            hgclr_eval._precision_at_k(self.predictions, self.labels, 5), 0.3
        )

    def test_r_precision_uses_each_sample_label_cardinality(self):
        self.assertAlmostEqual(
            hgclr_eval._r_precision(self.predictions, self.labels), 1.0
        )

    def test_r_precision_rejects_empty_gold(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            hgclr_eval._r_precision([[0.9, 0.1]], [[]])

    def test_metric_helpers_reject_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            hgclr_eval._precision_at_k([[0.9, 0.1]], [], 1)
        with self.assertRaisesRegex(ValueError, "only 2 labels"):
            hgclr_eval._precision_at_k([[0.9, 0.1]], [[0]], 3)


class HGCLRBinarizeTests(unittest.TestCase):
    def test_mmap_index_header_sizes_pointers_and_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = str(Path(tmpdir) / "sample")
            hgclr_binarize.write_mmap_dataset(prefix, [[1, 2], [], [65535]])

            with open(prefix + ".idx", "rb") as index_file:
                self.assertEqual(index_file.read(9), b"MMIDIDX\x00\x00")
                self.assertEqual(struct.unpack("<Q", index_file.read(8))[0], 1)
                self.assertEqual(struct.unpack("<B", index_file.read(1))[0], 8)
                self.assertEqual(struct.unpack("<Q", index_file.read(8))[0], 3)
                sizes = np.frombuffer(index_file.read(12), dtype=np.int32)
                pointers = np.frombuffer(index_file.read(24), dtype=np.int64)

            np.testing.assert_array_equal(sizes, [2, 0, 1])
            np.testing.assert_array_equal(pointers, [0, 4, 4])
            np.testing.assert_array_equal(
                np.fromfile(prefix + ".bin", dtype=np.uint16), [1, 2, 65535]
            )

    def test_mmap_writer_rejects_uint16_overflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = str(Path(tmpdir) / "sample")
            with self.assertRaisesRegex(ValueError, "outside uint16"):
                hgclr_binarize.write_mmap_dataset(prefix, [[65536]])
            with self.assertRaisesRegex(ValueError, "outside uint16"):
                hgclr_binarize.write_mmap_dataset(prefix, [[-1]])


class HGCLRCandidateResultTests(unittest.TestCase):
    def test_candidate_server_aggregates_are_excluded_from_release(self):
        results_dir = HGCLR_ROOT / "results" / "candidate"
        self.assertTrue((results_dir / "README.md").is_file())
        self.assertFalse(list(results_dir.glob("*_seed_aggregate.*")))


if __name__ == "__main__":
    unittest.main()

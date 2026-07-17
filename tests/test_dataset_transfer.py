import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_ROOT = REPOSITORY_ROOT / "dataset_transfer"
sys.path.insert(0, str(TRANSFER_ROOT))

import common
import htc_to_htc_lite
import htc_to_xml
import xml_to_htc


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


class HierarchyTests(unittest.TestCase):
    def test_leaf_filtering_removes_transitive_ancestors(self):
        hierarchy = {"Root": {"A", "B"}, "A": {"A1"}, "A1": {"A2"}}
        self.assertEqual(common.leaf_only(["A", "A2", "B"], hierarchy), ["A2", "B"])

    def test_cycles_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cycle.taxonomy"
            path.write_text("A\tB\nB\tA\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cycle"):
                common.load_taxonomy(path)


class HTCToXMLTests(unittest.TestCase):
    def test_conversion_is_deterministic_and_comma_separated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input" / "toy"
            input_dir.mkdir(parents=True)
            (input_dir / "toy.taxonomy").write_text(
                "Root\tA\tB\nA\tA1\nA1\tA2\n", encoding="utf-8"
            )
            write_jsonl(
                input_dir / "toy_train.json",
                [{"token": "train\ntext", "label": ["A", "A2", "B"]}],
            )
            write_jsonl(
                input_dir / "toy_val.json",
                [{"token": "validation", "label": ["B"]}],
            )
            write_jsonl(
                input_dir / "toy_test.json",
                [{"token": "test", "label": ["A2"]}],
            )

            output = htc_to_xml.convert(
                "toy", root / "input", root / "output", leaves_only=True
            )
            label_map = json.loads((output / "id_map.json").read_text(encoding="utf-8"))
            self.assertEqual(list(label_map), ["A2", "B"])
            self.assertEqual(
                (output / "train_labels.txt").read_text(encoding="utf-8").splitlines()[0],
                "0,1",
            )
            self.assertEqual(
                (output / "train_raw_texts.txt").read_text(encoding="utf-8").splitlines()[0],
                "train text",
            )


class HTCLiteTests(unittest.TestCase):
    def test_conversion_contracts_omitted_taxonomy_nodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input" / "toy"
            input_dir.mkdir(parents=True)
            (input_dir / "toy.taxonomy").write_text(
                "Root\tA\tB\nA\tA1\nA1\tA2\n", encoding="utf-8"
            )
            record = {"token": "text", "label": ["A", "A2", "B"]}
            for split in ("train", "val", "test"):
                write_jsonl(input_dir / f"toy_{split}.json", [record])

            output = htc_to_htc_lite.convert("toy", root / "input", root / "output")
            self.assertEqual(
                (output / "toy_lite.taxonomy").read_text(encoding="utf-8"),
                "Root\tA2\tB\n",
            )
            converted = common.read_jsonl(output / "toy_lite_train.json")
            self.assertEqual(converted[0]["label"], ["A2", "B"])


class XMLToHTCTests(unittest.TestCase):
    def _make_input(self, root):
        input_dir = root / "input" / "toy"
        input_dir.mkdir(parents=True)
        (input_dir / "toy_label_map.txt").write_text("A\nA2\nB\n", encoding="utf-8")
        (input_dir / "toy.taxonomy").write_text("Root\t0\t2\n0\t1\n", encoding="utf-8")
        (input_dir / "toy_train_texts.txt").write_text(
            "one\ntwo\nthree\nfour\nfive\n", encoding="utf-8"
        )
        (input_dir / "toy_train_labels.txt").write_text(
            "1\n2\n0\n1,2\n0\n", encoding="utf-8"
        )
        (input_dir / "toy_test_texts.txt").write_text("test\n", encoding="utf-8")
        (input_dir / "toy_test_labels.txt").write_text("1\n", encoding="utf-8")

    def test_conversion_expands_ancestors_without_silent_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_input(root)
            output = xml_to_htc.convert(
                "toy", root / "input", root / "output", validation_fraction=0.2
            )
            train = common.read_jsonl(output / "toy_train.json")
            validation = common.read_jsonl(output / "toy_val.json")
            test = common.read_jsonl(output / "toy_test.json")
            self.assertEqual(len(train), 4)
            self.assertEqual(len(validation), 1)
            self.assertEqual(test[0]["label"], ["A2", "A"])

    def test_parallel_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_input(root)
            (root / "input" / "toy" / "toy_test_labels.txt").write_text(
                "1\n2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                xml_to_htc.convert("toy", root / "input", root / "output")


if __name__ == "__main__":
    unittest.main()

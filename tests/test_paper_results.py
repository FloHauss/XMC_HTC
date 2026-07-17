import csv
import json
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_RESULTS = REPOSITORY_ROOT / "results" / "paper" / "table_results.csv"
METRICS = ("r_precision", "p_at_1", "p_at_3", "p_at_5", "f1_micro", "f1_macro")


class PaperResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with PAPER_RESULTS.open(newline="", encoding="utf-8") as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.by_key = {(row["dataset"], row["method"]): row for row in cls.rows}

    def test_table_is_complete_and_numeric(self):
        self.assertEqual(len(self.rows), 25)
        self.assertEqual(
            {row["dataset"] for row in self.rows},
            {"WOS", "NYT", "RCV1-V2", "Wiki10-31K", "AmazonCat-13K"},
        )
        for dataset in {row["dataset"] for row in self.rows}:
            self.assertEqual(sum(row["dataset"] == dataset for row in self.rows), 5)
        failed = [row for row in self.rows if row["status"] == "did_not_complete"]
        self.assertEqual(
            {(row["dataset"], row["method"]) for row in failed},
            {("Wiki10-31K", "HGCLR"), ("AmazonCat-13K", "HGCLR")},
        )
        for row in self.rows:
            for metric in METRICS:
                mean, deviation = row[f"{metric}_mean"], row[f"{metric}_std"]
                if row["status"] == "complete":
                    self.assertLessEqual(float(mean), 100)
                    self.assertGreaterEqual(float(mean), 0)
                    self.assertGreaterEqual(float(deviation), 0)
                else:
                    self.assertEqual((mean, deviation), ("", ""))

    def test_boldface_winners_from_paper(self):
        expected = {
            "WOS": ["RADAr", "RADAr", "HGCLR", "HGCLR", "HBGL", "HBGL"],
            "NYT": ["CascadeXML", "CascadeXML", "CascadeXML", "CascadeXML", "HBGL", "HBGL"],
            "RCV1-V2": ["CascadeXML", "XR-Transformer", "XR-Transformer", "CascadeXML", "RADAr", "HBGL"],
            "Wiki10-31K": ["XR-Transformer", "XR-Transformer", "XR-Transformer", "XR-Transformer", "CascadeXML", "RADAr"],
            "AmazonCat-13K": ["XR-Transformer", "XR-Transformer", "XR-Transformer", "XR-Transformer", "CascadeXML", "CascadeXML"],
        }
        for dataset, winners in expected.items():
            rows = [row for row in self.rows if row["dataset"] == dataset and row["status"] == "complete"]
            actual = [
                max(rows, key=lambda row: float(row[f"{metric}_mean"]))["method"]
                for metric in METRICS
            ]
            self.assertEqual(actual, winners)

    def test_human_readable_results_contain_every_reported_value(self):
        rendered = (REPOSITORY_ROOT / "RESULTS.md").read_text(encoding="utf-8")
        for row in self.rows:
            if row["status"] == "did_not_complete":
                self.assertIn(f'| {row["method"]}† | - | - | - | - | - | - |', rendered)
                continue
            for metric in METRICS:
                value = f'{row[f"{metric}_mean"]} ± {row[f"{metric}_std"]}'
                self.assertIn(value, rendered)

    def test_hgclr_candidate_server_aggregates_are_not_published(self):
        candidate_dir = (
            REPOSITORY_ROOT / "integrations" / "hgclr" / "results" / "candidate"
        )
        self.assertTrue((candidate_dir / "README.md").is_file())
        self.assertFalse(list(candidate_dir.glob("*_seed_aggregate.*")))

    def test_retained_xr_aggregates_are_not_misidentified_as_paper_runs(self):
        aliases = {
            "WOS": "wos",
            "NYT": "nyt",
            "RCV1-V2": "rcv1",
            "Wiki10-31K": "wiki10-31k",
            "AmazonCat-13K": "amazoncat-13k",
        }
        pattern = re.compile(r"^(p@1|p@3|p@5|r_prec):\s+([0-9.]+)\s+\+/-\s+([0-9.]+)", re.MULTILINE)
        for dataset, directory in aliases.items():
            text = (
                REPOSITORY_ROOT
                / "XMLmodels"
                / "pecos"
                / "run_ensemble"
                / "results"
                / directory
                / "average.txt"
            ).read_text(encoding="utf-8")
            historical = {name: (float(mean), float(std)) for name, mean, std in pattern.findall(text)}
            paper = self.by_key[(dataset, "XR-Transformer")]
            paper_ranking = {
                "r_prec": float(paper["r_precision_mean"]),
                "p@1": float(paper["p_at_1_mean"]),
                "p@3": float(paper["p_at_3_mean"]),
                "p@5": float(paper["p_at_5_mean"]),
            }
            if dataset == "AmazonCat-13K":
                self.assertEqual({key: value[0] for key, value in historical.items()}, paper_ranking)
                self.assertTrue(all(value[1] == 0 for value in historical.values()))
            else:
                self.assertTrue(all(historical[key][0] != value for key, value in paper_ranking.items()))


if __name__ == "__main__":
    unittest.main()

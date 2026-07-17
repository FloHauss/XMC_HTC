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

    def test_hgclr_candidates_match_paper_at_display_precision(self):
        aliases = {"WOS": "WebOfScience", "NYT": "nyt", "RCV1-V2": "rcv1"}
        candidate_keys = {
            "r_precision": "r_precision",
            "p_at_1": "p_at_1",
            "p_at_3": "p_at_3",
            "p_at_5": "p_at_5",
            "f1_micro": "test_micro_f1",
            "f1_macro": "test_macro_f1",
        }
        for dataset, candidate_name in aliases.items():
            path = (
                REPOSITORY_ROOT
                / "integrations"
                / "hgclr"
                / "results"
                / "candidate"
                / f"{candidate_name}_seed_aggregate.json"
            )
            aggregate = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
            paper = self.by_key[(dataset, "HGCLR")]
            for metric, candidate_key in candidate_keys.items():
                self.assertLessEqual(
                    abs(aggregate[candidate_key]["mean"] * 100 - float(paper[f"{metric}_mean"])),
                    0.011,
                )
                self.assertLessEqual(
                    abs(
                        aggregate[candidate_key]["std_population"] * 100
                        - float(paper[f"{metric}_std"])
                    ),
                    0.011,
                )

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

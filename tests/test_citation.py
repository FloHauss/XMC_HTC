import re
import unittest
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AUTHORS = [
    "Florian Hauss",
    "Tom Speier",
    "Nerijus Bertalis",
    "Paul Granse",
    "Ferhat Gül",
    "Leon Menkel",
    "David Schüler",
    "Lukas Galke Poech",
    "Ansgar Scherp",
]
ORCID = re.compile(r"^https://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


class CitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.citation = yaml.safe_load(
            (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        )

    def test_required_metadata_and_preferred_paper(self):
        self.assertEqual(self.citation["cff-version"], "1.2.0")
        self.assertEqual(self.citation["type"], "software")
        preferred = self.citation["preferred-citation"]
        self.assertEqual(preferred["doi"], "10.1145/3820755.3832808")
        self.assertEqual(preferred["status"], "in-press")
        self.assertEqual(preferred["title"], self.citation["title"])

    def test_author_order_and_orcids(self):
        for author_list in (
            self.citation["authors"],
            self.citation["preferred-citation"]["authors"],
        ):
            names = [
                f'{author["given-names"]} {author["family-names"]}'
                for author in author_list
            ]
            self.assertEqual(names, EXPECTED_AUTHORS)
            for author in author_list:
                if "orcid" in author:
                    self.assertRegex(author["orcid"], ORCID)


if __name__ == "__main__":
    unittest.main()

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class DocumentationLinkTests(unittest.TestCase):
    def test_release_facing_relative_links_exist(self):
        documents = [
            REPOSITORY_ROOT / "README.md",
            REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md",
            REPOSITORY_ROOT / "integrations" / "hgclr" / "README.md",
            REPOSITORY_ROOT / "integrations" / "hgclr" / "USAGE.md",
            REPOSITORY_ROOT / "integrations" / "hgclr" / "PROVENANCE.md",
            REPOSITORY_ROOT / "XMLmodels" / "CascadeXML" / "README.md",
            REPOSITORY_ROOT / "docs" / "MODEL_PROVENANCE.md",
            REPOSITORY_ROOT / "docs" / "RELEASE_READINESS.md",
            REPOSITORY_ROOT / "docs" / "CASCADEXML_MODIFICATIONS.md",
            REPOSITORY_ROOT / "htc" / "hbgl" / "README.md",
            REPOSITORY_ROOT / "docs" / "HBGL_MODIFICATIONS.md",
            REPOSITORY_ROOT / "XMLmodels" / "pecos" / "STUDY_INTEGRATION.md",
            REPOSITORY_ROOT / "docs" / "XR_TRANSFORMER_MODIFICATIONS.md",
            REPOSITORY_ROOT / "docs" / "REPRODUCIBILITY.md",
            REPOSITORY_ROOT / "xr_transformer_guide.md",
            REPOSITORY_ROOT / "dataset_transfer" / "README.md",
            REPOSITORY_ROOT / "CITATION.cff",
            REPOSITORY_ROOT / "results" / "paper" / "README.md",
            REPOSITORY_ROOT / "docs" / "RESULT_RECONCILIATION.md",
        ]
        missing = []
        for document in documents:
            for target in MARKDOWN_LINK.findall(document.read_text()):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative_target = target.split("#", 1)[0]
                if not (document.parent / relative_target).exists():
                    missing.append(f"{document.relative_to(REPOSITORY_ROOT)} -> {target}")

        self.assertEqual(missing, [], "Missing release-facing links:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()

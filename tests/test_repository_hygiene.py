import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_RESULT_PREFIX = "XMLmodels/pecos/examples/pefa-wsdm24/results/"


class RepositoryHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=REPOSITORY_ROOT
        )
        cls.tracked = [path for path in output.decode().split("\0") if path]

    def test_generated_and_platform_specific_files_are_not_tracked(self):
        forbidden = []
        for name in self.tracked:
            path = Path(name)
            if "__pycache__" in path.parts or ".ipynb_checkpoints" in path.parts:
                forbidden.append(name)
            elif path.suffix in {".pyc", ".pyo", ".so", ".out"}:
                forbidden.append(name)
            elif path.name.startswith("slurm-"):
                forbidden.append(name)
            elif path.suffix == ".log" and not name.startswith(UPSTREAM_RESULT_PREFIX):
                forbidden.append(name)
        self.assertEqual(forbidden, [], "Tracked generated files:\n" + "\n".join(forbidden))

    def test_stale_submodule_declaration_is_absent(self):
        self.assertNotIn(".gitmodules", self.tracked)


if __name__ == "__main__":
    unittest.main()

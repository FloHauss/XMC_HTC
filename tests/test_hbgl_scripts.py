import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "htc" / "hbgl" / "scripts"


class HBGLScriptTests(unittest.TestCase):
    def test_seed_sweeps_preserve_each_run(self):
        for script in sorted(SCRIPTS.glob("*.sh")):
            source = script.read_text(encoding="utf-8")
            self.assertIn("SEEDS=(42 1 2 3 4)", source, script.name)
            self.assertIn(
                'OUTPUT_DIR="${OUTPUT_ROOT}/seed_${current_seed}"',
                source,
                script.name,
            )
            self.assertNotIn("rm -rf", source, script.name)


if __name__ == "__main__":
    unittest.main()

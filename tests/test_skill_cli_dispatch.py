from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills" / "binance-readonly"
CLI_SCRIPTS = sorted(SKILLS_ROOT.glob("*/scripts/cli.mjs"))


@unittest.skipIf(shutil.which("node") is None, "node is not installed")
class SkillCliDispatchTest(unittest.TestCase):
    """The ESM self-execution guard must fire on Windows paths.

    The upstream `import.meta.url === 'file://' + argv[1]` comparison never
    matches on Windows, so every CLI silently printed nothing and exited 0.
    Running an unknown command proves dispatch executes: it must print a usage
    error and exit non-zero - all without touching the network.
    """

    def test_staged_clis_exist(self) -> None:
        self.assertGreaterEqual(len(CLI_SCRIPTS), 5)

    def test_dispatch_executes_on_this_platform(self) -> None:
        for script in CLI_SCRIPTS:
            with self.subTest(script=str(script)):
                completed = subprocess.run(
                    ["node", str(script), "definitely-not-a-command", "{}"],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn("Unknown command", completed.stderr)

    def test_help_prints_usage(self) -> None:
        script = CLI_SCRIPTS[0]
        completed = subprocess.run(
            ["node", str(script), "--help"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import re
import unittest
from lynx_harness.architecture import fitness_report


class ArchitectureFitnessTests(unittest.TestCase):
    """F-FIT ratchet: runtime must not import test/vendor implementation modules."""
    def test_fitness_report_is_green(self):
        report=fitness_report(Path(__file__).parents[1])
        self.assertEqual(report["violations"], [])
        self.assertTrue(report["core_dependency_free"] and report["loop_gateway_boundary"] and report["remote_binding_default"] and report["child_policy_non_escalation"])

    def test_runtime_import_boundary(self):
        source_root = Path(__file__).parents[1] / "src" / "lynx_harness"
        forbidden = re.compile(r"(?:from|import)\s+(?:tests|third_party|lynx_harness\.tests)\b")
        violations=[]
        for path in source_root.glob("*.py"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if forbidden.search(line): violations.append(f"{path}:{line_number}")
        self.assertEqual(violations, [])


if __name__ == "__main__": unittest.main()

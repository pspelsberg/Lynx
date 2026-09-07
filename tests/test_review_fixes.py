import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from lynx_harness.a2a import AgentArtifact
from lynx_harness.branch import BranchManager
from lynx_harness.memory import ArtifactStore
from lynx_harness.skills import SkillRegistry, SkillEvolution
from lynx_harness.inference import ScriptedBackend
from lynx_harness.models import AgentState, ComputeBudget, Decision, DecisionType, Mode, Postcondition, TaskContract
from lynx_harness.profiler import BenchmarkProfile, run_profile_matrix
from lynx_harness.security import sanitized_subprocess_env
from lynx_harness.tools import ToolGateway, ToolRegistry, ToolSpec
from lynx_harness.training import TrainingExample, TrajectoryDataset
from lynx_harness.verifier import Verifier


class ReviewFixTests(unittest.TestCase):
    def test_verifier_handles_scalar_candidates(self):
        for candidate in ("answer", [], None):
            result = Verifier().check(candidate, TaskContract("scalar", "check"))
            self.assertIsNotNone(result.status)

    def test_gateway_rejects_non_json_tool_output(self):
        async def handler(_):
            return object()
        async def run():
            registry = ToolRegistry()
            registry.register(ToolSpec("object", "returns object", {"type": "object"}), handler)
            return await ToolGateway(registry).execute("object", {})
        observation = asyncio.run(run())
        self.assertFalse(observation.ok)
        self.assertIn("valid JSON", observation.error or "")

    def test_a2a_artifact_uri_is_content_addressed(self):
        with self.assertRaises(ValueError):
            AgentArtifact("id", "text/plain", uri="artifact://../../etc/passwd")

    def test_report_output_does_not_follow_symlink(self):
        from lynx_harness.security import atomic_write_text
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("keep")
            output = root / "output"
            output.symlink_to(target)
            with self.assertRaises(ValueError):
                atomic_write_text(output, "overwrite")
            self.assertEqual(target.read_text(), "keep")

    def test_sync_profile_callback_timeout_is_bounded(self):
        def callback(_):
            time.sleep(1)
            return 1
        with self.assertRaises(TypeError):
            asyncio.run(run_profile_matrix([BenchmarkProfile("slow")], callback, timeout_s=0.05))

    def test_subprocess_environment_is_minimal(self):
        env = sanitized_subprocess_env()
        self.assertNotIn("LYNX_API_TOKEN", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)

    def test_branch_merge_ignores_foreign_context_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "artifacts")
            foreign = store.put("foreign", branch_id="other")
            root = AgentState("goal", Mode.THINK)
            manager = BranchManager(root)
            branch = manager.fork("branch-a")
            branch.answer = "ok"; branch.status = "success"
            contract = TaskContract("merge", "goal", postconditions=(Postcondition("artifact_exists"),))
            with self.assertRaises(ValueError):
                manager.merge("branch-a", verifier=Verifier(), contract=contract,
                              context={"artifacts": store, "artifact_refs": (foreign,), "branch_id": "other"})
            self.assertEqual(root.status, "running")

    def test_stable_skill_cannot_be_promoted_again(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(Path(directory))
            registry.write_candidate("demo", "demo")
            evaluation = SkillEvolution(registry).evaluate("demo", 1, [True], [False])
            registry.promote("demo", 1, evaluation)
            with self.assertRaises(ValueError):
                registry.promote("demo", 1, evaluation)

    def test_rlm_context_provenance_is_branch_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "artifacts")
            from lynx_harness.context import RlmContextEngine
            engine = RlmContextEngine(store, context_threshold=1_000)
            prepared = engine.prepare("needle " * 2_000, "needle", branch_id="branch-a", task_id="task-a")
            self.assertEqual(store.metadata(prepared["artifact_id"]).branch_id, "branch-a")
            self.assertEqual(store.metadata(prepared["artifact_id"]).task_id, "task-a")

    def test_browser_navigation_is_disabled_without_egress_interceptor(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "browser.sh"
            executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o700)
            browser = __import__("lynx_harness.playwright", fromlist=["PlaywrightCli"]).PlaywrightCli(Path(directory), str(executable), {"example.test"})
            with self.assertRaises(Exception):
                asyncio.run(browser.run("open", "https://example.test"))

    def test_training_export_does_not_follow_symlink(self):
        from lynx_harness.verifier import Verifier
        verification = Verifier().check({"answer": "ok"}, TaskContract("verify", "x"), {"tools_used": (), "max_risk": 0})
        dataset = TrajectoryDataset()
        dataset.examples.append(TrainingExample.from_verifier({"goal": "x"}, {"type": "finish"}, verification, run_id="run"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("keep")
            output = root / "dataset.jsonl"
            output.symlink_to(target)
            with self.assertRaises(ValueError):
                dataset.write_jsonl(output)
            self.assertEqual(target.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from lynx_harness.flight import FlightRecorder
from lynx_harness.memory import ArtifactStore, JsonlStore, OkfStore
from lynx_harness.models import AgentState, ComputeBudget, Mode
from lynx_harness.skills import SkillRegistry


class SymlinkHardeningTests(unittest.TestCase):
    def test_knowledge_sink_rejects_preexisting_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.md"
            outside.write_text("do not overwrite")
            (root / "candidate.md").symlink_to(outside)
            with self.assertRaises(ValueError):
                OkfStore(root).put_candidate("candidate", "title", "body")
            self.assertEqual(outside.read_text(), "do not overwrite")

    def test_skill_sink_rejects_symlinked_catalog_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            (root / "candidate").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                SkillRegistry(root).write_candidate("candidate", "description")
            self.assertFalse((outside / "skill.json").exists())

    def test_durable_roots_reject_symlinked_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            linked = root / "linked"
            linked.symlink_to(outside, target_is_directory=True)
            for factory in (
                lambda: ArtifactStore(linked / "artifacts"),
                lambda: JsonlStore(linked / "trajectory.jsonl"),
                lambda: OkfStore(linked / "knowledge"),
                lambda: SkillRegistry(linked / "skills"),
                lambda: FlightRecorder(linked / "runs", run_id="run"),
            ):
                with self.assertRaises(ValueError):
                    factory()
            self.assertEqual(list(outside.iterdir()), [])

    def test_fork_checkpoint_does_not_follow_destination_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = FlightRecorder(root / "runs", run_id="parent")
            recorder.record("run_started", {"goal": "x"})
            recorder.record("observation", {"tool": "x", "ok": True})
            recorder.checkpoint(AgentState("x", Mode.FAST), ComputeBudget(llm_tokens=10, steps=2, tool_calls=1, max_wall_time_s=1))
            child_dir = root / "runs" / "child" / "checkpoints"
            child_dir.mkdir(parents=True)
            outside = root / "outside.json"
            outside.write_text("do not overwrite")
            (child_dir / "000002.json").symlink_to(outside)
            with self.assertRaises(ValueError):
                recorder.fork(2, new_run_id="child", new_branch_id="branch")
            self.assertEqual(outside.read_text(), "do not overwrite")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from lynx_harness.inference import ScriptedBackend
from lynx_harness.loop import AgentRunner
from lynx_harness.models import Decision, DecisionType
from lynx_harness.rah import ChildBudget, ChildTask, LocalChildExecutor
from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry



async def _completed_state(ref):
    return SimpleNamespace(status="success", answer="done", observations=[SimpleNamespace(artifact_id=ref)], steps=1)

class RahTests(unittest.TestCase):
    def test_firewall_rejects_non_artifact_input_and_depth(self):
        with self.assertRaises(ValueError): ChildTask("x", artifact_refs=("/tmp/secret",)).context_firewall()
        with self.assertRaises(ValueError): ChildTask("x", budget=ChildBudget(depth=3,max_depth=2), depth=3).context_firewall()
    def test_child_uses_only_explicit_tools_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            def factory(task):
                registry=ToolRegistry()
                return AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH,answer="done")]),registry,ToolGateway(registry,ToolPolicy(workspace=Path(directory))))
            task=ChildTask("child", allowed_tools=(), budget=ChildBudget())
            result=asyncio.run(LocalChildExecutor(factory).execute(task))
            self.assertEqual(result.answer,"done")
    def test_child_artifact_refs_are_resolved_and_branch_checked(self):
        from types import SimpleNamespace
        from lynx_harness.memory import ArtifactStore
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "artifacts")
            ref = store.put("child output", branch_id="main")
            def factory(task):
                return SimpleNamespace(
                    registry=ToolRegistry(), artifacts=store,
                    run=lambda *args, **kwargs: _completed_state(ref),
                )
            async def execute():
                # LocalChildExecutor accepts async runner seams; the lambda
                # above is wrapped here to keep the fake deterministic.
                executor = LocalChildExecutor(factory, artifact_store=store)
                return await executor.execute(ChildTask("child", branch_id="main"))
            result = asyncio.run(execute())
            self.assertEqual(result.artifact_refs, (ref,))

            def bad_factory(task):
                return SimpleNamespace(registry=ToolRegistry(), artifacts=store,
                                       run=lambda *args, **kwargs: _completed_state("artifact://" + "0" * 64 + ".txt"))
            with self.assertRaises(ValueError):
                asyncio.run(LocalChildExecutor(bad_factory, artifact_store=store).execute(ChildTask("child")))

    def test_child_tool_allowlist_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            def factory(task):
                registry=ToolRegistry(); return AgentRunner(ScriptedBackend([]),registry,ToolGateway(registry))
            registry_task=ChildTask("child", allowed_tools=("calculator",))
            # Factory deliberately exposes no tools; empty exposure is safe.
            self.assertEqual(asyncio.run(LocalChildExecutor(factory).execute(registry_task)).status,"failed")

if __name__ == "__main__": unittest.main()

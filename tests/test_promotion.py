import asyncio
import tempfile
import unittest
from pathlib import Path

from lynx_harness.memory import OkfStore
from lynx_harness.skills import SkillRegistry, SkillEvolution
from lynx_harness.promotion import GovernedPromotion, PromotionPolicy


class GovernedPromotionTests(unittest.TestCase):
    def test_skill_auto_promotion_requires_enabled_policy_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SkillRegistry(Path(td) / "skills")
            evolution = SkillEvolution(registry)
            evolution.mine([{"type": "tool_call", "payload": {"tool": "filesystem.read"}}], "inspect-files")
            evaluation = evolution.evaluate("inspect-files", 1, [True, True], [True, False])
            governed = GovernedPromotion(registry=registry)
            with self.assertRaises(PermissionError):
                governed.promote_skill(evaluation, replay_passed=True)
            with self.assertRaises(ValueError):
                governed.promote_skill(evaluation, replay_passed=False, policy=PromotionPolicy(enabled=True))
            path = governed.promote_skill(evaluation, replay_passed=True, policy=PromotionPolicy(enabled=True))
            self.assertIn('"status": "stable"', path.read_text())

    def test_knowledge_auto_promotion_requires_complete_verifier(self):
        with tempfile.TemporaryDirectory() as td:
            store = OkfStore(Path(td) / "knowledge")
            store.put_candidate("fact", "Fact", "Evidence")
            governed = GovernedPromotion(knowledge=store)
            from lynx_harness.verifier import Verifier
            from lynx_harness.models import TaskContract
            verifier = Verifier().check({"answer": "Evidence"}, TaskContract("knowledge", "Fact"), {"tools_used": (), "max_risk": 0})
            with self.assertRaises(PermissionError):
                governed.promote_knowledge("fact", verifier)
            path = governed.promote_knowledge("fact", verifier, policy=PromotionPolicy(enabled=True))
            self.assertIn("status: stable", path.read_text())


class IntegrationManagerTests(unittest.TestCase):
    def test_explicit_integrations_register_before_start_and_close(self):
        from lynx_harness.integrations import IntegrationManager
        from lynx_harness.tools import ToolRegistry

        class Kernel:
            async def execute(self, code):
                return {"ok": True}
            async def close(self):
                self.closed = True

        async def run():
            registry = ToolRegistry()
            manager = IntegrationManager(registry)
            kernel = Kernel()
            manager.add_kernel(kernel)
            self.assertIn("python.exec", [spec.name for spec in registry.specs()])
            await manager.start()
            with self.assertRaises(RuntimeError):
                manager.add_kernel(Kernel())
            await manager.close()
            self.assertTrue(kernel.closed)

        asyncio.run(run())

    def test_a2a_is_an_explicit_external_mutation_tool(self):
        from lynx_harness.a2a import A2AClient
        from lynx_harness.integrations import IntegrationManager
        from lynx_harness.tools import ToolRegistry
        client = A2AClient("https://agent.example/tasks", {"agent.example"})
        manager = IntegrationManager(ToolRegistry())
        manager.add_a2a(client)
        spec = manager.registry.get("a2a.submit.0")[0]
        self.assertEqual(spec.risk.name, "EXTERNAL_MUTATION")
        self.assertEqual(spec.provenance, "https://agent.example/tasks")

    def test_runner_owns_integration_lifecycle(self):
        from lynx_harness.inference import ScriptedBackend
        from lynx_harness.loop import AgentRunner
        from lynx_harness.models import ComputeBudget, Decision, DecisionType, Mode, RiskLevel
        from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry, register_builtin_tools
        with tempfile.TemporaryDirectory() as td:
            class Kernel:
                async def execute(self, code): return {"ok": True}
                async def close(self): self.closed = True
            async def run():
                registry = ToolRegistry(); register_builtin_tools(registry, Path(td))
                manager = __import__("lynx_harness.integrations", fromlist=["IntegrationManager"]).IntegrationManager(registry)
                kernel = Kernel(); manager.add_kernel(kernel)
                gateway = ToolGateway(registry, ToolPolicy(workspace=Path(td), max_risk=RiskLevel.LOCAL_MUTATION))
                backend = ScriptedBackend([Decision(DecisionType.TOOL, tool="calculator", arguments={"expression": "2+2"}), Decision(DecisionType.FINISH, answer="4")])
                runner = AgentRunner(backend, registry, gateway, integrations=manager, enable_progress_monitor=False)
                state = await runner.run("calculate", Mode.FAST, ComputeBudget.for_mode(Mode.FAST))
                return state, kernel
            state, kernel = asyncio.run(run())
            self.assertEqual(state.answer, "4")
            self.assertTrue(kernel.closed)

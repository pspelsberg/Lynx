from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lynx_harness.bestofn import BestOfNExecutor
from lynx_harness.context import RlmContextEngine, RlmExecutor, RlmLimits
from lynx_harness.inference import ScriptedBackend
from lynx_harness.loop import AgentRunner
from lynx_harness.memory import ArtifactStore, OkfStore
from lynx_harness.models import ComputeBudget, Decision, DecisionType, Mode
from lynx_harness.skills import SkillRegistry, SkillEvolution
from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry, register_builtin_tools


class SliceTests(unittest.TestCase):
    def test_rlm_context_is_bounded_and_retrievable(self):
        with tempfile.TemporaryDirectory() as directory:
            store=ArtifactStore(Path(directory)); aid=store.put("# Head\nalpha\n" + "beta " * 1000)
            engine=RlmContextEngine(store, chunk_chars=100)
            self.assertIn("Head", engine.inspect(aid)["headings"]); self.assertLessEqual(len(engine.retrieve(aid,"beta")),8)

    def test_rlm_executor_enforces_call_limit(self):
        async def query(prompt, depth): return prompt
        async def run(): return await RlmExecutor(query).run([str(i) for i in range(20)])
        self.assertEqual(len(asyncio.run(run())),8)

    def test_rlm_runner_executes_bounded_model_calls_and_accounts_usage(self):
        class RecordingBackend(ScriptedBackend):
            def __init__(self, decisions):
                super().__init__(decisions); self.contexts = []
            async def decide(self, context, tools, max_tokens):
                self.contexts.append(context)
                return await super().decide(context, tools, max_tokens)

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                artifacts = ArtifactStore(workspace / "artifacts")
                registry = ToolRegistry()
                backend = RecordingBackend([
                    Decision(DecisionType.FINISH, answer="part one"),
                    Decision(DecisionType.FINISH, answer="part two"),
                    Decision(DecisionType.FINISH, answer="combined"),
                    Decision(DecisionType.FINISH, answer="final"),
                ])
                runner = AgentRunner(
                    backend, registry, ToolGateway(registry, ToolPolicy(workspace=workspace)),
                    artifacts=artifacts,
                    context_engine=RlmContextEngine(artifacts, chunk_chars=100),
                    context_max_chars=1_000, rlm_enabled=True,
                    rlm_limits=RlmLimits(max_calls=3, max_chars_per_call=300),
                )
                budget = ComputeBudget(llm_tokens=5_000, steps=2, tool_calls=0, recursive_calls=3, max_wall_time_s=10)
                state = await runner.run("find needle " + ("x " * 3_000), Mode.FAST, budget)
                return state, runner, backend, budget

        state, runner, backend, budget = asyncio.run(run())
        self.assertEqual(state.status, "success")
        self.assertEqual(runner.run_metrics["rlm_calls"], 3)
        self.assertEqual(budget.used_recursive_calls, 3)
        self.assertEqual(sum("rlm" in context for context in backend.contexts), 3)
        self.assertLess(len(str(backend.contexts[-1])), 5_000)

    def test_rlm_respects_local_recursive_budget_with_larger_shared_ledger(self):
        class RecordingBackend(ScriptedBackend):
            def __init__(self):
                super().__init__([Decision(DecisionType.FINISH, answer="mapped"), Decision(DecisionType.FINISH, answer="reduced"), Decision(DecisionType.FINISH, answer="final")])
                self.calls = 0
            async def decide(self, context, tools, max_tokens):
                if "rlm" in context:
                    self.calls += 1
                return await super().decide(context, tools, max_tokens)

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory); artifacts = ArtifactStore(workspace / "artifacts")
                registry = ToolRegistry(); backend = RecordingBackend()
                runner = AgentRunner(backend, registry, ToolGateway(registry, ToolPolicy(workspace=workspace)), artifacts=artifacts, context_engine=RlmContextEngine(artifacts, chunk_chars=100), context_max_chars=1_000, rlm_enabled=True, rlm_limits=RlmLimits(max_calls=8, max_chars_per_call=300), ledger=__import__("lynx_harness.models", fromlist=["BudgetLedger"]).BudgetLedger(ComputeBudget(llm_tokens=5_000, steps=2, tool_calls=0, recursive_calls=8, max_wall_time_s=10)))
                budget = ComputeBudget(llm_tokens=5_000, steps=2, tool_calls=0, recursive_calls=2, max_wall_time_s=10)
                state = await runner.run("find needle " + ("x " * 3_000), Mode.FAST, budget)
                return state, backend, runner.ledger

        state, backend, ledger = asyncio.run(run())
        self.assertEqual(state.status, "success")
        self.assertEqual(backend.calls, 2)
        self.assertEqual(ledger.snapshot()["used"]["recursive_calls"], 2)

    def test_skill_evaluation_requires_paired_candidate_and_baseline_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(Path(directory))
            evolution = SkillEvolution(registry)
            with self.assertRaises(ValueError):
                evolution.evaluate("demo", 1, [True, True], [False])
            with self.assertRaises(ValueError):
                evolution.evaluate("demo", 1, [True], [False], baseline_safety=[True, False])

    def test_skill_candidate_requires_passing_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            registry=SkillRegistry(Path(directory)); registry.write_candidate("demo", "demo skill")
            with self.assertRaises(ValueError): registry.promote("demo",1,{"passed":False})
            evaluation=SkillEvolution(registry).evaluate("demo",1,[True],[False]); registry.promote("demo",1,evaluation); self.assertEqual(registry.discover()[0].status,"stable")

    def test_knowledge_lifecycle_requires_existing_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            store=OkfStore(Path(directory)); store.put_candidate("fact", "Fact", "Body")
            with self.assertRaises(ValueError): store.promote_candidate("fact", "process:test")
            with self.assertRaises(ValueError): store.promote_candidate("fact", {"status": "passed", "checks": [{"id": "syntax", "status": "failed"}]})
            from lynx_harness.verifier import Verifier
            from lynx_harness.models import TaskContract
            verification = Verifier().check({"answer": "Body"}, TaskContract("knowledge", "Fact"), {"tools_used": (), "max_risk": 0})
            store.promote_candidate("fact", verification)
            self.assertIn("status: stable", (Path(directory)/"fact.md").read_text())
            secret_path = store.put_candidate("secret", "title", "TOKEN=should-not-persist")
            self.assertNotIn("should-not-persist", secret_path.read_text())
            store.deprecate("fact", "stale")
            self.assertIn("status: deprecated", (Path(directory)/"fact.md").read_text())

    def test_best_of_n_returns_isolated_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            def factory(_branch):
                registry=ToolRegistry(); register_builtin_tools(registry,workspace)
                return AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH,answer="ok")]),registry,ToolGateway(registry,ToolPolicy(workspace=workspace)),enable_progress_monitor=False)
            candidates=asyncio.run(BestOfNExecutor(factory,branches=2).run("hello",Mode.FAST,ComputeBudget(llm_tokens=1000,steps=2,tool_calls=1)))
            self.assertEqual(len(candidates),2); self.assertEqual({item.branch_id for item in candidates},{"branch-0","branch-1"})

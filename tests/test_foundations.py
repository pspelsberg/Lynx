import asyncio, json, tempfile, unittest
from pathlib import Path
from lynx_harness.models import TaskContract, ComputeBudget, BudgetLedger, BudgetExceeded, RiskLevel, Decision, DecisionType, SecurityPolicy, Mode
from lynx_harness.controller import Controller, ControllerError
from lynx_harness.inference import parse_decision, InferenceError, ScriptedBackend
from lynx_harness.verifier import Verifier, VerificationStatus
from lynx_harness.context import RlmContextEngine
from lynx_harness.memory import ArtifactStore, JsonlStore

class FoundationTests(unittest.TestCase):
    def test_contract_roundtrip_and_rejects_unknown_or_secret(self):
        c=TaskContract.from_json(json.dumps({"id":"root-1","goal":"review","allowed_tools":[],"budget":{"llm_tokens":100,"steps":2,"tool_calls":0,"max_wall_time_s":1},"postconditions":[{"name":"output_bounded"}]}))
        self.assertEqual(TaskContract.from_yaml(c.to_json()).id,"root-1")
        self.assertNotIn("password", c.to_json().lower())
        with self.assertRaises(ValueError): TaskContract.from_dict({"id":"x","goal":"x","unknown":1})
        with self.assertRaises(ValueError): TaskContract.from_dict({1:"x", "goal":"x"})
        # Budget/policy inputs must not use parser last-write-wins semantics.
        with self.assertRaises(ValueError): TaskContract.from_json('{"id":"x","id":"y","goal":"x"}')
        with self.assertRaises(ValueError): TaskContract.from_json('{"id":"x","goal":"x","budget":{"llm_tokens":NaN}}')
        with self.assertRaises(ValueError): TaskContract.from_yaml("id: x\nid: y\ngoal: x\n")
    def test_ledger_atomic_parallel_reservation(self):
        ledger=BudgetLedger(ComputeBudget(llm_tokens=10,steps=2,tool_calls=2,recursive_calls=1,max_wall_time_s=1))
        first=ledger.reserve(llm_tokens=10,recursive_calls=1)
        with self.assertRaises(BudgetExceeded): ledger.reserve(llm_tokens=1)
        first.release(); self.assertTrue(ledger.can_reserve(llm_tokens=10))
    def test_local_budget_is_not_consumed_when_global_reservation_fails(self):
        from lynx_harness.loop import AgentRunner
        from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry
        local = ComputeBudget(llm_tokens=100, steps=1, tool_calls=1, max_wall_time_s=1)
        exhausted = BudgetLedger({"llm_tokens": 100, "steps": 0, "tool_calls": 1, "web_calls": 0, "browser_actions": 0, "python_seconds": 0, "recursive_calls": 0, "parallel_branches": 1, "max_wall_time_s": 1})
        registry = ToolRegistry()
        backend = ScriptedBackend([Decision(DecisionType.FINISH, answer="ok")])
        runner = AgentRunner(backend, registry, ToolGateway(registry, ToolPolicy()), ledger=exhausted)
        state = asyncio.run(runner.run("x", Mode.FAST, local))
        self.assertEqual(state.status, "budget_exhausted")
        self.assertEqual(local.used_steps, 0)
        self.assertEqual(local.used_llm_tokens, 0)

    def test_controller_rejects_illegal_transition(self):
        c=Controller(); c.plan(); c.start(); c.finish(); c.verify()
        with self.assertRaises(ControllerError): c.start()
    def test_delegate_parser_is_strict(self):
        decision=parse_decision(json.dumps({"type":"delegate","task":{"id":"child","goal":"x","allowed_tools":[]}}),set())
        self.assertEqual(decision.task.id,"child")
        with self.assertRaises(InferenceError): parse_decision('{"type":"delegate","task":{"id":"child","goal":"x","bad":1}}',set())
    def test_runner_rejects_non_incrementing_or_widening_child_lineage(self):
        from lynx_harness.loop import AgentRunner
        from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry
        backend = ScriptedBackend([Decision(DecisionType.DELEGATE, task=__import__("lynx_harness.models", fromlist=["AgentTask"]).AgentTask("child", "child", allowed_tools=(), parent_task_id="root", max_depth=32))])
        registry = ToolRegistry()
        runner = AgentRunner(backend, registry, ToolGateway(registry, ToolPolicy()))
        contract = TaskContract("root", "x", budget=ComputeBudget(llm_tokens=100, steps=1, tool_calls=0, recursive_calls=1, max_wall_time_s=1))
        state = asyncio.run(runner.run("x", contract=contract, budget=contract.budget))
        self.assertEqual(state.status, "rejected")

    def test_a2a_rejects_non_json_metadata_without_leaking_encoder_errors(self):
        from lynx_harness.a2a import AgentTask
        with self.assertRaises(ValueError):
            AgentTask("task", "ctx", "goal", budget={"bad": object()})

    def test_contract_filters_model_visible_tools_before_decision(self):
        from lynx_harness.loop import AgentRunner
        from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry
        from lynx_harness.models import ToolSpec, Mode
        class RecordingBackend(ScriptedBackend):
            def __init__(self):
                super().__init__([Decision(DecisionType.FINISH, answer="ok")])
                self.tools_seen = []
            async def decide(self, context, tools, max_tokens):
                self.tools_seen.append([tool["name"] for tool in tools])
                return await super().decide(context, tools, max_tokens)
        registry = ToolRegistry()
        registry.register(ToolSpec("allowed", "allowed", {"type": "object"}), lambda args: None)
        registry.register(ToolSpec("forbidden", "forbidden", {"type": "object"}), lambda args: None)
        backend = RecordingBackend()
        runner = AgentRunner(backend, registry, ToolGateway(registry, ToolPolicy()))
        contract = TaskContract("filtered", "x", allowed_tools=("allowed",), budget=ComputeBudget(llm_tokens=100, steps=1, tool_calls=0, max_wall_time_s=1))
        state = asyncio.run(runner.run("x", contract.mode, contract.budget, contract=contract))
        self.assertEqual(state.status, "verified")
        self.assertEqual(backend.tools_seen, [["allowed"]])

    def test_evidence_requires_explicit_claim_verification(self):
        from lynx_harness.verifier import EvidenceCheck
        contract = TaskContract("evidence", "x")
        result = EvidenceCheck().check({}, contract, {
            "evidence_required": True,
            "claims": [{"id": "c", "source_refs": ["s"]}],
            "sources": [{"source_id": "s"}],
        })
        self.assertEqual(result.status, VerificationStatus.FAILED)

    def test_verifier_schema_and_bound(self):
        c=TaskContract("root", "x", expected_output=__import__("lynx_harness.models",fromlist=["ExpectedOutput"]).ExpectedOutput(schema={"type":"object","required":["answer"],"properties":{"answer":{"type":"string","maxLength":3}},"additionalProperties":False}), postconditions=())
        self.assertEqual(Verifier().check({"answer":"ok"},c).status,VerificationStatus.PASSED)
        self.assertEqual(Verifier().check({"answer":"toolong"},c).status,VerificationStatus.FAILED)
    def test_redaction_reaches_trajectory_and_artifact(self):
        from lynx_harness.security import redact
        value=redact({"nested":{"api_token":"abc"},"message":"Bearer secret-token"})
        self.assertEqual(value["nested"]["api_token"],"[redacted]"); self.assertIn("[redacted]", value["message"])
        from lynx_harness.security import redact_bytes
        self.assertNotIn(b"secret-value", redact_bytes(b"\xffTOKEN=secret-value"))
    def test_independent_verifier_cannot_bypass_deterministic_policy(self):
        from lynx_harness.verifier import IndependentModelVerifier
        async def run():
            backend=ScriptedBackend([Decision(DecisionType.FINISH, answer="PASS")])
            contract=TaskContract("critic","x",allowed_tools=(),security=SecurityPolicy(max_risk=RiskLevel.READ_ONLY),budget=ComputeBudget(llm_tokens=10,steps=1,tool_calls=0,max_wall_time_s=1))
            return await IndependentModelVerifier(backend).check({"answer":"ok"},contract,{"tools_used":["shell"]})
        self.assertEqual(asyncio.run(run()).status, VerificationStatus.FAILED)

    def test_schema_policy_and_required_artifact_gate(self):
        from lynx_harness.models import ExpectedOutput, Postcondition
        c=TaskContract("gate", "x", expected_output=ExpectedOutput(schema={"type":"object","required":["answer"],"properties":{"answer":{"type":"string","maxLength":3}},"additionalProperties":False}), postconditions=(Postcondition("artifact_exists"),), budget=ComputeBudget(llm_tokens=10,steps=1,tool_calls=0,max_wall_time_s=1))
        self.assertEqual(Verifier().check({"answer":"ok"},c).status,VerificationStatus.FAILED)
        self.assertEqual(Verifier().check({"answer":"toolong"},TaskContract("gate2","x",expected_output=ExpectedOutput(schema={"type":"object","properties":{"answer":{"type":"string","maxLength":3}}}),budget=ComputeBudget(llm_tokens=10,steps=1,tool_calls=0,max_wall_time_s=1),security=__import__("lynx_harness.models",fromlist=["SecurityPolicy"]).SecurityPolicy()),{"max_risk":RiskLevel.READ_ONLY}).status,VerificationStatus.FAILED)
    def test_freshness_handles_naive_and_invalid_dates(self):
        from lynx_harness.research import Evidence, ResearchState, check_freshness
        from datetime import datetime, timezone
        state=ResearchState("q", evidence=[Evidence("a","s","x",published_at="2000-01-01"),Evidence("b","s","x",published_at="not-a-date")])
        stale=check_freshness(state,1,now=datetime(2020,1,1,tzinfo=timezone.utc)); self.assertEqual(len(stale),2); self.assertTrue(all(e.verification != "passed" for e in state.evidence))
    def test_large_context_is_artifact_only(self):
        with tempfile.TemporaryDirectory() as d:
            e=RlmContextEngine(ArtifactStore(Path(d)),context_threshold=1000)
            result=e.prepare("needle "*500,"needle")
            self.assertEqual(result["mode"],"artifact"); self.assertTrue(result["artifact_id"]); self.assertLessEqual(len(result["chunks"][0]["content"]), 8_000)

if __name__=="__main__": unittest.main()

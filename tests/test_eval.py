from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lynx_harness.eval import AblationEvaluator, EvalCase, load_benchmark
from lynx_harness.inference import ScriptedBackend
from lynx_harness.loop import AgentRunner
from lynx_harness.models import ComputeBudget, Decision, DecisionType, Mode
from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry, register_builtin_tools


class EvalTests(unittest.TestCase):
    def test_evaluator_rejects_explicit_noncanonical_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory); registry=ToolRegistry(); register_builtin_tools(registry,workspace)
            backend=ScriptedBackend([Decision(DecisionType.FINISH,answer="OK")]); backend.model="Qwen3.8-9B.gguf"
            runner=AgentRunner(backend,registry,ToolGateway(registry,ToolPolicy(workspace=workspace)))
            with self.assertRaises(ValueError):
                asyncio.run(AblationEvaluator(lambda _variant: runner).evaluate([EvalCase("one","say ok")],["bare_llm"]))

    def test_ablation_metrics_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            def factory(_variant):
                registry=ToolRegistry(); register_builtin_tools(registry,workspace)
                return AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH,answer="OK")]),registry,ToolGateway(registry,ToolPolicy(workspace=workspace)))
            evaluator=AblationEvaluator(factory); cases=[EvalCase("one","say ok","OK",Mode.FAST)]
            first=asyncio.run(evaluator.evaluate(cases,["bare_llm","+tool_router"]))
            second=asyncio.run(evaluator.evaluate(cases,["bare_llm","+tool_router"]))
            self.assertEqual(first,second)
            self.assertEqual(evaluator.aggregate(first)[0]["success_rate"],1.0)

    def test_evaluator_isolates_mutable_runner_per_case(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            created = []
            def factory(_variant):
                created.append(True)
                registry = ToolRegistry()
                return AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH, answer="OK")]), registry, ToolGateway(registry, ToolPolicy(workspace=workspace)))
            cases = [EvalCase("one", "say ok", "OK"), EvalCase("two", "say ok", "OK")]
            rows = asyncio.run(AblationEvaluator(factory).evaluate(cases, ["bare_llm"]))
            self.assertEqual([row["success"] for row in rows], [True, True])
            self.assertEqual(len(created), 2)

    def test_evaluation_rows_redact_model_output(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory); registry=ToolRegistry(); register_builtin_tools(registry,workspace)
            def factory(_variant):
                return AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH,answer="TOKEN=secret")]),registry,ToolGateway(registry,ToolPolicy(workspace=workspace)))
            rows=asyncio.run(AblationEvaluator(factory).evaluate([EvalCase("one","say ok")],["bare_llm"]))
            self.assertNotIn("secret", rows[0]["answer"])
            self.assertIn("[redacted]", rows[0]["answer"])

    def test_expected_claims_drive_claim_coverage_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            def factory(_variant):
                registry = ToolRegistry()
                return AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH, answer="[claim-1] supported")]), registry, ToolGateway(registry, ToolPolicy(workspace=workspace)))
            case = EvalCase("claim-case", "say claim", mode=Mode.FAST, expected_claims=("claim-1",))
            rows = asyncio.run(AblationEvaluator(factory).evaluate([case], ["bare_llm"]))
            self.assertEqual(rows[0]["claim_coverage"], 1.0)
            self.assertEqual(AblationEvaluator.aggregate(rows)[0]["claim_coverage"], 1.0)

    def test_benchmark_json_loads(self):
        with tempfile.NamedTemporaryFile(mode="w",suffix=".json") as stream:
            stream.write('{"cases":[{"id":"x","goal":"hello","mode":"fast"}]}'); stream.flush()
            self.assertEqual(load_benchmark(Path(stream.name))[0].id,"x")

    def test_benchmark_loader_rejects_noncanonical_declared_model(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            stream.write('{"model":"Qwen3.8-9B.gguf","cases":[{"goal":"hello"}]}'); stream.flush()
            with self.assertRaises(ValueError): load_benchmark(Path(stream.name))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            stream.write('{"cases":[{"goal":"hello","model":"Qwen3.8-9B.gguf"}]}'); stream.flush()
            with self.assertRaises(ValueError): load_benchmark(Path(stream.name))

    def test_benchmark_loader_rejects_duplicate_and_nonfinite_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            stream.write('{"cases":[{"id":"x","goal":"hello"}],"cases":[{"id":"y","goal":"hello"}]}'); stream.flush()
            with self.assertRaises(ValueError): load_benchmark(Path(stream.name))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            stream.write('{"cases":[{"id":"x","goal":"hello","expected":NaN}]}'); stream.flush()
            with self.assertRaises(ValueError): load_benchmark(Path(stream.name))

    def test_benchmark_yaml_loader_rejects_duplicate_and_recursive_alias(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as stream:
            stream.write("cases:\n  - id: x\n    goal: hello\n    goal: replaced\n"); stream.flush()
            with self.assertRaises(ValueError): load_benchmark(Path(stream.name))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as stream:
            stream.write("cases: &cases [*cases]\n"); stream.flush()
            with self.assertRaises(ValueError): load_benchmark(Path(stream.name))


if __name__ == "__main__": unittest.main()

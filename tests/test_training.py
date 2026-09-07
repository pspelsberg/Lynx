from pathlib import Path
import sys,unittest,asyncio
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.training import TrajectoryDataset, TeacherCase, TeacherStudentEvaluator, write_verified_lora_dataset, LoRAExperimentSpec, TrainingExample, evaluate_lora_ablation
from lynx_harness.verifier import Verifier
from lynx_harness.models import TaskContract

def passed_verification(answer="ok"):
 return Verifier().check({"answer": answer}, TaskContract("verify", "x"), {"tools_used": (), "max_risk": 0})
class TrainingTests(unittest.TestCase):
 def test_teacher_student_evaluator_discards_reasoning_and_verifies_cases(self):
  async def run():
   from lynx_harness.inference import ScriptedBackend
   from lynx_harness.models import Decision, DecisionType
   return await TeacherStudentEvaluator(ScriptedBackend([Decision(DecisionType.FINISH,answer="teacher")]),ScriptedBackend([Decision(DecisionType.FINISH,answer="student")])).evaluate([TeacherCase("case-1",{"goal":"finish"})])
  result=asyncio.run(run()); self.assertFalse(result.raw_reasoning_stored); self.assertEqual(result.teacher.cases,1); self.assertTrue(result.rows[0]["teacher_validated"]); self.assertEqual(result.comparison_kind,"same_model_control"); self.assertEqual(result.rows[0]["comparison_kind"],"same_model_control")

 def test_teacher_evaluation_rechecks_mutated_backend_identity(self):
  from lynx_harness.inference import ScriptedBackend
  from lynx_harness.models import Decision, DecisionType
  teacher=ScriptedBackend([Decision(DecisionType.FINISH, answer="teacher")]); student=ScriptedBackend([Decision(DecisionType.FINISH, answer="student")])
  evaluator=TeacherStudentEvaluator(teacher, student); student.model="Qwen3.8-9B.gguf"
  with self.assertRaises(ValueError): asyncio.run(evaluator.evaluate([TeacherCase("case-1",{})]))

 def test_teacher_and_lora_gates_reject_other_models(self):
  from lynx_harness.inference import ScriptedBackend
  from lynx_harness.models import Decision, DecisionType
  teacher=ScriptedBackend([Decision(DecisionType.FINISH, answer="teacher")]); teacher.model="Qwen3.8-27B-UD-Q4_K_M.gguf"
  with self.assertRaises(ValueError): TeacherStudentEvaluator(teacher, ScriptedBackend([]))
  with self.assertRaises(ValueError): LoRAExperimentSpec("exp", "Qwen3.8-27B-UD-Q4_K_M.gguf")
  with self.assertRaises(ValueError): evaluate_lora_ablation("actions", [True], [False], [True], base_model="other.gguf")
  with self.assertRaises(ValueError): TrajectoryDataset(model="Qwen3.8-27B-UD-Q4_K_M.gguf")

 def test_lora_dataset_gate_rejects_unverified_examples(self):
  d=TrajectoryDataset();
  with self.assertRaises(ValueError): write_verified_lora_dataset(d,Path("/tmp/should-not-write.jsonl"))
  with self.assertRaises(ValueError): LoRAExperimentSpec("exp", "model", target="raw_web")
  with self.assertRaises(ValueError): LoRAExperimentSpec("exp", "other.gguf")
  self.assertEqual(LoRAExperimentSpec("exp", "models/Qwen3.5-4B-Q6_K.gguf").base_model, "models/Qwen3.5-4B-Q6_K.gguf")

 def test_generic_jsonl_writer_cannot_bypass_verification_or_redaction(self):
  import tempfile
  d=TrajectoryDataset(); d.examples.append(TrainingExample({"token":"Bearer secret"},{"type":"finish"},"success",{}))
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaises(ValueError): d.write_jsonl(Path(directory)/"data.jsonl")
  d.examples.clear(); d.examples.append(TrainingExample.from_verifier({"token":"Bearer secret"},{"type":"finish"},passed_verification(),run_id="r"))
  with tempfile.TemporaryDirectory() as directory:
   output=d.write_jsonl(Path(directory)/"data.jsonl")
   self.assertNotIn("secret", output.read_text())
   self.assertIn("[redacted]", output.read_text())

  nan=TrajectoryDataset(); nan.examples.append(TrainingExample.from_verifier({"score":float("nan")},{"type":"finish"},passed_verification(),run_id="r"))
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaises(ValueError): nan.write_jsonl(Path(directory)/"nan.jsonl")
  huge=TrajectoryDataset(); huge.examples.append(TrainingExample.from_verifier({str(i):"x" * 10_000 for i in range(100)},{"type":"finish"},passed_verification(),run_id="r"))
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaises(ValueError): huge.write_jsonl(Path(directory)/"too-large.jsonl")

 def test_lora_dataset_gate_writes_verified_structured_examples(self):
  d=TrajectoryDataset(); d.examples.append(TrainingExample.from_verifier({"goal":"x"},{"type":"finish"},passed_verification(),run_id="r"))
  import tempfile
  with tempfile.TemporaryDirectory() as directory:
   output=write_verified_lora_dataset(d,Path(directory)/"data.jsonl"); self.assertTrue(output.is_file()); self.assertNotIn("raw_web", output.read_text())

 def test_trajectory_export_rejects_wrong_model_and_cross_run_verification(self):
  d=TrajectoryDataset()
  base=[{"type":"run_started","run_id":"run-1","payload":{"goal":"x","mode":"think","model":"Qwen3.8-9B.gguf"}}, {"type":"decision","run_id":"run-1","seq":1,"payload":{"type":"finish","answer":"ok"}}, {"type":"verification","run_id":"run-1","seq":2,"payload":{"status":"passed","checks":[{"id":"schema","status":"passed"}]}}, {"type":"run_finished","run_id":"run-1","seq":3,"payload":{"status":"success"}}]
  self.assertEqual(d.add_run(base,True),0)
  base[0]["payload"].pop("model")
  base[2]["run_id"]="other"
  self.assertEqual(d.add_run(base,True),0)

 def test_trajectory_export_rejects_cross_run_observations(self):
  d=TrajectoryDataset()
  events=[{"type":"run_started","run_id":"run-1","seq":1,"payload":{"goal":"x","mode":"think"}},
          {"type":"observation","run_id":"evil","seq":2,"payload":{"output":"unrelated"}},
          {"type":"decision","run_id":"run-1","seq":3,"payload":{"type":"finish","answer":"ok"}},
          {"type":"verification","run_id":"run-1","seq":4,"payload":{"status":"passed","checks":[{"id":"schema","status":"passed"}]}},
          {"type":"run_finished","run_id":"run-1","seq":5,"payload":{"status":"success"}}]
  self.assertEqual(d.add_run(events, True), 0)

 def test_trajectory_input_is_bounded(self):
  with self.assertRaises(ValueError): TrajectoryDataset().add_run([{}] * 100_001)

 def test_only_verified_finished_runs_export(self):
  d=TrajectoryDataset(); events=[{"type":"run_started","run_id":"run-1","payload":{"goal":"x","mode":"think"}},{"type":"decision","run_id":"run-1","seq":1,"payload":{"type":"finish","answer":"ok"}}]
  self.assertEqual(d.add_run(events,False),0); self.assertEqual(d.add_run(events,True),0)
  # A passed verifier event without a terminal success event is incomplete.
  result=passed_verification()
  events.append({"type":"verification","run_id":"run-1","seq":2,"payload":asdict(result)})
  self.assertEqual(d.add_run(events,result),0)
  events.append({"type":"run_finished","run_id":"run-1","seq":3,"payload":{"status":"success"}})
  self.assertEqual(d.add_run(events,result),1)

 def test_runner_trajectory_uses_exportable_event_schema(self):
  from lynx_harness.inference import ScriptedBackend
  from lynx_harness.loop import AgentRunner
  from lynx_harness.memory import JsonlStore
  from lynx_harness.models import ComputeBudget, Decision, DecisionType, ExpectedOutput, Postcondition, TaskContract
  import json, tempfile
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"trajectory.jsonl"
   registry=__import__("lynx_harness.tools", fromlist=["ToolRegistry"]).ToolRegistry()
   runner=AgentRunner(ScriptedBackend([Decision(DecisionType.FINISH, answer="ok")]), registry, __import__("lynx_harness.tools", fromlist=["ToolGateway"]).ToolGateway(registry), trajectory=JsonlStore(path))
   budget=ComputeBudget(llm_tokens=10, steps=1, tool_calls=0, max_wall_time_s=1)
   contract=TaskContract("trajectory-run", "x", expected_output=ExpectedOutput(max_chars=20), postconditions=(Postcondition("output_bounded"),), budget=budget)
   state=asyncio.run(runner.run("x", contract=contract, budget=budget))
   self.assertEqual(state.status, "verified")
   events=[json.loads(line) for line in path.read_text().splitlines()]
   self.assertTrue(events and all("type" in event for event in events))
   self.assertTrue(all(event.get("schema_version") == 2 for event in events))
   self.assertTrue(all(event.get("task_id") == "trajectory-run" for event in events))
   self.assertTrue(all(event.get("branch") == "main" for event in events))
   self.assertEqual(TrajectoryDataset().add_run(events, verified=state.verification), 1)
 def test_trajectory_states_do_not_include_future_observations(self):
  d=TrajectoryDataset()
  result=passed_verification()
  events=[{"type":"run_started","run_id":"run-1","payload":{"goal":"x","mode":"think"}},
          {"type":"decision","run_id":"run-1","seq":1,"payload":{"type":"tool","tool":"calculator","arguments":{"expression":"1+1"}}},
          {"type":"observation","run_id":"run-1","seq":2,"payload":{"output":"2"}},
          {"type":"decision","run_id":"run-1","seq":3,"payload":{"type":"finish","answer":"ok"}},
          {"type":"verification","run_id":"run-1","seq":4,"payload":asdict(result)},
          {"type":"run_finished","run_id":"run-1","seq":5,"payload":{"status":"success"}}]
  self.assertEqual(d.add_run(events, result), 2)
  self.assertEqual(d.examples[0].state["observations"], [])
  self.assertEqual(d.examples[1].state["observations"][0]["output"], "2")

 def test_trajectory_export_rejects_malformed_decision_and_forged_model(self):
  d=TrajectoryDataset()
  malformed=[{"type":"run_started","run_id":"run-1","payload":{"goal":"x"}},
             {"type":"decision","run_id":"run-1","seq":1,"payload":None},
             {"type":"verification","run_id":"run-1","seq":2,"payload":{"status":"passed","checks":[{"id":"schema","status":"passed"}]}},
             {"type":"run_finished","run_id":"run-1","seq":3,"payload":{"status":"success"}}]
  self.assertEqual(d.add_run(malformed, True), 0)
  d.examples.append(TrainingExample({"model_id":"Qwen3.8-9B.gguf"},{"type":"finish"},"success",{"passed":True,"event_seq":1,"run_id":"r","checks":[{"id":"schema","status":"passed"}]}))
  with self.assertRaises(ValueError): d.write_jsonl(Path("/tmp/forged-model.jsonl"))

if __name__=="__main__": unittest.main()

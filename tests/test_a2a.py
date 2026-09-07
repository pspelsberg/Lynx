from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.a2a import AgentTask,AgentArtifact,AgentResult,A2AClient,AgentDescriptor,create_a2a_app,from_a2a_task,_safe_result_payload
from lynx_harness.acp import acp_to_agent_task, agent_task_to_acp
class A2ATests(unittest.TestCase):
 def test_artifact_only_task(self):
  task=AgentTask("task-1","ctx-1","review",("artifact://repo/1",)); self.assertIn("artifact://repo/1",str(task.inputs))
  with self.assertRaises(ValueError): AgentTask("task-1","ctx-1","review",("/etc/passwd",))
  with self.assertRaises(ValueError): from_a2a_task(task)
 def test_acp_roundtrip_is_artifact_only(self):
  ref="artifact://"+"a"*64+".txt"
  task=acp_to_agent_task({"id":"acp-1","goal":"review","mode":"fast","artifacts":[ref],"allowed_tools":[]})
  self.assertEqual(agent_task_to_acp(task)["artifacts"],[ref])
  with self.assertRaises(ValueError): acp_to_agent_task({"id":"x","goal":"x","artifacts":["/etc/passwd"]})
 def test_optional_agent_card_surface(self):
  app=create_a2a_app(AgentDescriptor("local","Local agent"),lambda task: None)
  self.assertTrue(any(route.path=="/.well-known/agent-card.json" for route in app.routes))
 def test_remote_endpoint_requires_https_allowlist(self):
  with self.assertRaises(ValueError): A2AClient("http://example.test",{"example.test"})

 def test_artifacts_and_descriptors_are_bounded(self):
  with self.assertRaises(ValueError): AgentArtifact("a", "text/plain", b"not text")
  with self.assertRaises(ValueError): AgentArtifact("a", "text/plain", uri="https://user:pass@example.test/file")
  with self.assertRaises(ValueError): AgentDescriptor("agent", "x", skills=("s",)*101)

 def test_result_boundary_redacts_and_bounds_output(self):
  safe=_safe_result_payload(AgentResult("completed", message="Authorization: Bearer secret"))
  self.assertNotIn("Bearer secret", safe["message"])
  artifacts=tuple(AgentArtifact(str(i), "text/plain", "x"*100_000) for i in range(100))
  with self.assertRaises(ValueError): _safe_result_payload(AgentResult("completed", artifacts=artifacts))
if __name__=="__main__": unittest.main()

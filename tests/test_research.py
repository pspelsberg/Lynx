from __future__ import annotations
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.models import ComputeBudget, RiskLevel, ToolSpec
from lynx_harness.research import ResearchWorkflow
from lynx_harness.tools import ToolGateway, ToolPolicy, ToolRegistry

class ResearchTests(unittest.TestCase):
 def test_missing_publication_date_downgrades_verified_evidence(self):
  from lynx_harness.research import Evidence, ResearchState, check_freshness
  evidence=Evidence("claim", "source", "excerpt", claim_id="claim-1", verified=True, verification="passed")
  state=ResearchState("q", claims=[{"id":"claim-1","verification":"passed","evidence_refs":["claim"]}], evidence=[evidence])
  affected=check_freshness(state, 30)
  self.assertEqual(affected, ["claim-1"])
  self.assertFalse(evidence.verified)
  self.assertEqual(evidence.verification, "pending")
  self.assertEqual(state.claims[0]["verification"], "pending")

 def test_research_collects_verifies_and_synthesizes(self):
  registry=ToolRegistry()
  registry.register(ToolSpec("web.search","search",{"type":"object","required":["query"],"properties":{"query":{"type":"string"}}},family="web"),lambda args: asyncio.sleep(0, result={"results":[{"title":"A","url":"https://example.test/a"}]}))
  registry.register(ToolSpec("web.fetch","fetch",{"type":"object","required":["url"],"properties":{"url":{"type":"string"}}},family="web"),lambda args: asyncio.sleep(0, result="evidence"))
  async def run(): return await ResearchWorkflow(ToolGateway(registry,ToolPolicy(max_risk=RiskLevel.EXTERNAL_MUTATION,confirmed=True))).run("question",ComputeBudget(web_calls=2,tool_calls=4))
  state=asyncio.run(run()); self.assertEqual(state.status,"incomplete"); self.assertEqual(state.verified_claims,0); self.assertIn("UNVERIFIED",state.synthesis)

 def test_freshness_invalid_and_stale_evidence_is_not_verified(self):
  from lynx_harness.research import Evidence, ResearchState, check_freshness
  state=ResearchState("q", claims=[{"id":"claim-1","verification":"passed"}], evidence=[Evidence("claim","s","excerpt",claim_id="claim-1",verified=True,verification="passed",published_at="not-a-date")])
  affected=check_freshness(state,30)
  self.assertEqual(affected,["claim-1"]); self.assertFalse(state.evidence[0].verified); self.assertEqual(state.verified_claims,0)

 def test_targeted_search_is_bounded_and_budget_status_is_preserved(self):
  calls=[]; registry=ToolRegistry()
  registry.register(ToolSpec("web.search","search",{"type":"object","required":["query"],"properties":{"query":{"type":"string"}}},family="web"),lambda args: (calls.append(args), asyncio.sleep(0,result={"results":[]}))[1])
  async def run(): return await ResearchWorkflow(ToolGateway(registry,ToolPolicy(max_risk=RiskLevel.EXTERNAL_MUTATION,confirmed=True)),max_targeted_searches=1).run("question",ComputeBudget(web_calls=1,tool_calls=1))
  state=asyncio.run(run()); self.assertLessEqual(len(calls),1); self.assertIn(state.status,{"budget_exhausted","incomplete","complete"})
if __name__=="__main__": unittest.main()

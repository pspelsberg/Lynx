from pathlib import Path
import asyncio,sys,unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.profiler import BenchmarkProfile,BenchmarkResult,run_profile_matrix,select_best_profile,serialize_benchmark
from lynx_harness.model_profiles import approve_speculative_variant
class ProfilerTests(unittest.TestCase):
 def test_speculative_variant_is_opt_in_and_non_regressing(self):
  self.assertFalse(approve_speculative_variant(baseline_success=1,candidate_success=1,baseline_verifier_rate=1,candidate_verifier_rate=1,explicitly_requested=False).enabled)
  self.assertFalse(approve_speculative_variant(baseline_success=1,candidate_success=1,baseline_verifier_rate=1,candidate_verifier_rate=1,baseline_latency_ms=10,candidate_latency_ms=11,explicitly_requested=True).enabled)

 def test_selects_fast_stable_profile(self):
  profiles=[BenchmarkProfile("rocm",backend="rocm"),BenchmarkProfile("vulkan",backend="vulkan")]
  async def run(): return await run_profile_matrix(profiles,lambda p: BenchmarkResult(p,score=100 if p.backend=="vulkan" else 80,throughput=100 if p.backend=="vulkan" else 80))
  self.assertEqual(select_best_profile(asyncio.run(run())).profile.backend,"vulkan")

 def test_local_matrix_is_bounded_and_does_not_duplicate_modes(self):
  from lynx_harness.profiler import LynxBenchmarkMatrix
  matrix=LynxBenchmarkMatrix()
  self.assertEqual(len(matrix.profiles()), 36)
  with self.assertRaises(ValueError):
   LynxBenchmarkMatrix(backends=("rocm",)*129)
  with self.assertRaises(ValueError):
   LynxBenchmarkMatrix(backends=__import__("itertools").count())

 def test_profile_options_are_unambiguous(self):
  with self.assertRaises(ValueError):
   BenchmarkProfile("duplicate", options=(("flash", "on"), ("flash", "off")))
 def test_result_metadata_rejects_nonfinite_json(self):
  with self.assertRaises(ValueError):
   BenchmarkResult(BenchmarkProfile("finite", model="qwen35-4b-q6_k"), metadata={"score": float("nan")})

 def test_report_requires_canonical_runtime_model(self):
  generic = BenchmarkResult(BenchmarkProfile("offline"), score=1)
  with self.assertRaises(ValueError):
   serialize_benchmark([generic])
  with self.assertRaises(ValueError):
   serialize_benchmark([], environment={"model": "Qwen3.8-9B.gguf"})
  canonical = BenchmarkProfile("runtime", model="qwen35-4b-q6_k")
  report = serialize_benchmark([BenchmarkResult(canonical, score=1, error="TOKEN=secret")])
  self.assertEqual(report["model_policy"]["model_id"], "qwen35-4b-q6_k")
  self.assertNotIn("secret", report["results"][0]["error"])

 def test_mapping_result_rejects_another_profile(self):
  profile=BenchmarkProfile("requested")
  foreign=BenchmarkProfile("foreign")
  async def run():
   return await run_profile_matrix([profile], lambda _profile: {"profile":foreign, "score":1})
  with self.assertRaises(ValueError): asyncio.run(run())

if __name__=="__main__": unittest.main()

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from lynx_harness.inference import parse_decision, InferenceError
from lynx_harness.memory import ArtifactStore
from lynx_harness.models import BudgetLedger, ComputeBudget, TaskContract, ExpectedOutput
from lynx_harness.security import redact
from lynx_harness.verifier import Verifier, VerificationStatus


class HardeningTests(unittest.TestCase):
    def test_strict_json_and_secret_key_boundaries(self):
        with self.assertRaises(InferenceError):
            parse_decision('{"type":"tool","tool":"x","arguments":{"value":NaN}}', {"x"})
        self.assertEqual(redact({"accessToken":"secret", "x-api-key":"secret", "llm_tokens": 2})["llm_tokens"], 2)

    def test_artifact_manifest_and_branch_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            ref = store.put("payload", branch_id="branch-a")
            self.assertEqual(store.metadata(ref).branch_id, "branch-a")
            self.assertEqual(store.put("payload", branch_id="branch-b"), ref)
            self.assertEqual(store.manifest(ref)["size"], 7)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            manifest[Path(ref.removeprefix("artifact://")).name]["artifact_id"] = "artifact://redirected"
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                ArtifactStore(Path(directory)).metadata(ref)

    def test_expected_output_bound_is_mandatory(self):
        contract = TaskContract("bounded", "x", expected_output=ExpectedOutput(max_chars=3), budget=ComputeBudget(llm_tokens=10, steps=1, tool_calls=0, max_wall_time_s=1))
        self.assertEqual(Verifier().check({"answer":"toolong"}, contract).status, VerificationStatus.FAILED)

    def test_expired_ledger_rejects_reservation(self):
        import time
        ledger = BudgetLedger(ComputeBudget(llm_tokens=10, steps=1, tool_calls=0, max_wall_time_s=0.1))
        time.sleep(0.12)
        self.assertFalse(ledger.can_reserve(llm_tokens=1))

    def test_redaction_handles_authorization_bearer_value(self):
        from lynx_harness.security import redact_text
        self.assertNotIn("secret-value", redact_text("authorization: Bearer secret-value", max_chars=None))
        self.assertNotIn("secret-value", redact_text("authorization=Bearer secret-value", max_chars=None))

    def test_model_decision_deep_json_is_rejected_as_inference_error(self):
        # Keep malformed model output inside the public InferenceError seam;
        # a deeply nested but byte-bounded object must not leak RecursionError.
        with self.assertRaises(InferenceError):
            parse_decision("{" + '"x":{' * 5000 + "null" + "}" * 5000, set())

    def test_artifact_get_rejects_tampered_reference_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            ref = store.put("payload")
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            name = ref.removeprefix("artifact://")
            manifest[name]["artifact_id"] = "artifact://redirected"
            manifest_path.write_text(json.dumps(manifest))
            reopened = ArtifactStore(Path(directory))
            with self.assertRaises(ValueError):
                reopened.get(ref)

    def test_backend_identity_cannot_hide_conflicting_profile(self):
        from lynx_harness.model_profiles import backend_model_identity, CapabilityProfile
        from lynx_harness.models import Mode
        class Backend:
            model = "Qwen3.5-4B-Q6_K.gguf"
            profile = CapabilityProfile("other", Mode.THINK)
        with self.assertRaises(ValueError):
            backend_model_identity(Backend())

    def test_model_hash_gate_rejects_symlink_path(self):
        from lynx_harness.model_profiles import ModelManifestEntry
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.gguf"
            target.write_bytes(b"x")
            link = Path(directory) / "link.gguf"
            link.symlink_to(target)
            entry = ModelManifestEntry("qwen35-4b-q6_k", "unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q6_K.gguf", "0" * 64)
            self.assertFalse(entry.verify_file(link))

    def test_llama_backend_rejects_unbounded_timeout(self):
        from lynx_harness.inference import LlamaCppBackend
        with self.assertRaises(InferenceError):
            LlamaCppBackend(timeout_s=0)

    def test_runtime_llm_is_single_model(self):
        from lynx_harness.inference import LlamaCppBackend
        from lynx_harness.model_profiles import CapabilityProfile
        from lynx_harness.models import Mode
        with self.assertRaises(InferenceError):
            LlamaCppBackend(model="other.gguf")
        # Canonical aliases are accepted but normalized before transport, and
        # a profile for another checkpoint cannot be paired with this backend.
        backend = LlamaCppBackend(model="qwen35-4b-q6_k")
        self.assertEqual(backend.model, "Qwen3.5-4B-Q6_K.gguf")
        with self.assertRaises(InferenceError):
            LlamaCppBackend(profile=CapabilityProfile("other", Mode.THINK))

    def test_canonical_profiles_follow_workload_mode(self):
        from lynx_harness.model_profiles import canonical_capability_profile
        from lynx_harness.models import Mode
        fast = canonical_capability_profile(Mode.FAST)
        think = canonical_capability_profile(Mode.THINK)
        research = canonical_capability_profile(Mode.RESEARCH)
        self.assertEqual({fast.model_id, think.model_id, research.model_id}, {"Qwen3.5-4B-Q6_K.gguf"})
        self.assertFalse(fast.supports_thinking)
        self.assertTrue(think.supports_thinking and research.supports_thinking)
        self.assertLess(fast.max_steps, think.max_steps)
        self.assertLess(think.max_steps, research.max_steps)

    def test_runner_normalizes_canonical_model_aliases(self):
        from lynx_harness.cli import build_runner
        from lynx_harness.config import Settings
        for alias in ("qwen35-4b-q6_k", "models/Qwen3.5-4B-Q6_K.gguf"):
            runner = build_runner(Settings(model=alias), scripted=True)
            self.assertEqual(runner.backend.model, "Qwen3.5-4B-Q6_K.gguf")

    def test_model_manifest_rejects_duplicate_keys(self):
        from lynx_harness.model_profiles import load_model_manifest
        path = Path(tempfile.mktemp(suffix=".json"))
        try:
            path.write_text('{"models":[],"models":[]}')
            with self.assertRaises(ValueError):
                load_model_manifest(path)
        finally:
            path.unlink(missing_ok=True)

    def test_manifest_and_runtime_path_cannot_select_noncanonical_model(self):
        from lynx_harness.model_profiles import (
            ModelManifestEntry, canonical_model_path, load_model_manifest,
            validate_runtime_model,
        )
        with self.assertRaises(ValueError):
            ModelManifestEntry(
                "other", "evil/repo", "0123456789abcdef", "other.gguf",
                "0" * 64, downloadable=True,
            )
        entries = load_model_manifest(Path("config/model-manifest.json"))
        self.assertEqual(validate_runtime_model("qwen35-4b-q6_k", entries).id, "qwen35-4b-q6_k")
        with self.assertRaises(ValueError):
            validate_runtime_model("qwen38-9b-distill-q4_k_m", entries)
        with self.assertRaises(ValueError):
            validate_runtime_model("qwen38-9b-distill-q4_k_m", entries)
        with self.assertRaises(ValueError):
            validate_runtime_model("qwen35-4b-q6_k", entries, allowed_ids=("qwen38-9b-distill-q4_k_m",))
        tampered_manifest = Path("/tmp/lynx-tampered-manifest.json")
        tampered_manifest.write_text(json.dumps({"model": "Qwen3.8-9B.gguf", "models": []}))
        try:
            with self.assertRaises(ValueError):
                load_model_manifest(tampered_manifest)
        finally:
            tampered_manifest.unlink(missing_ok=True)
        tampered = [ModelManifestEntry(
            "qwen35-4b-q6_k", "evil/repo", "0123456789abcdef",
            "Qwen3.5-4B-Q6_K.gguf", "0" * 64, downloadable=False,
        )]
        with self.assertRaises(ValueError):
            validate_runtime_model("qwen35-4b-q6_k", tampered)
        # A canonical identity is executable only when the manifest explicitly
        # marks its entry downloadable; metadata-only entries must not become a
        # runtime fallback after manifest tampering.
        non_downloadable = [ModelManifestEntry(
            "qwen35-4b-q6_k", "unsloth/Qwen3.5-4B-GGUF",
            "e87f176479d0855a907a41277aca2f8ee7a09523",
            "Qwen3.5-4B-Q6_K.gguf",
            "fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66",
            downloadable=False,
        )]
        with self.assertRaises(ValueError):
            validate_runtime_model("qwen35-4b-q6_k", non_downloadable)
        with self.assertRaises(ValueError):
            canonical_model_path(Path("/tmp/Qwen3.5-4B-Q6_K.gguf"))

    def test_benchmark_does_not_persist_raw_model_content(self):
        script = (Path(__file__).parents[1] / "scripts" / "bench-llama.sh").read_text()
        self.assertNotIn('"response":body', script)
        self.assertIn('response_valid', script)
        self.assertIn('credential-free loopback URL', script)

    def test_benchmark_rejects_noncanonical_or_missing_model(self):
        import os
        import subprocess
        env = {**os.environ, "MODEL_PATH": str(Path("/tmp/not-the-canonical-model.gguf"))}
        result = subprocess.run(["bash", "scripts/bench-llama.sh"], cwd=Path(__file__).parents[1], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("canonical models/Qwen3.5-4B-Q6_K.gguf", result.stderr)

    def test_launcher_rejects_draft_model_under_single_model_policy(self):
        import os, subprocess
        env={**os.environ, "MODEL_PATH":"models/Qwen3.5-4B-Q6_K.gguf", "DRAFT_MODEL_PATH":"models/Qwen3.5-4B-Q6_K.gguf", "DRAFT_MODEL_SHA256":"x"}
        result=subprocess.run(["bash", "scripts/run-llama-rocm.sh"], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2); self.assertIn("Draft models are disabled", result.stderr)


if __name__ == "__main__": unittest.main()

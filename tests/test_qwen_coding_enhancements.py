from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from lynx_harness.bestofn import BestOfNExecutor, BranchCandidate
from lynx_harness.inference import LlamaCppBackend, ScriptedBackend
from lynx_harness.loop import AgentRunner
from lynx_harness.models import ComputeBudget, Decision, DecisionType, Mode, RiskLevel
from lynx_harness.tools import ToolError, ToolGateway, ToolPolicy, ToolRegistry, register_builtin_tools


class CodingEnhancementsTests(unittest.TestCase):
    def test_filesystem_write_and_patch(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace)
            policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="auto-edit")
            gateway = ToolGateway(registry, policy)

            # 1. Write a python file
            py_code = "def hello():\n    return 'world'\n"
            res = asyncio.run(gateway.execute("filesystem.write", {"path": "module/app.py", "content": py_code}))
            self.assertTrue(res.ok)
            self.assertTrue((workspace / "module" / "app.py").is_file())
            self.assertEqual((workspace / "module" / "app.py").read_text(), py_code)
            self.assertEqual(res.output.get("diagnostics"), [])

            # 2. Patch the python file
            patch_res = asyncio.run(gateway.execute("filesystem.patch", {
                "path": "module/app.py",
                "target_content": "'world'",
                "replacement_content": "'universe'"
            }))
            self.assertTrue(patch_res.ok)
            self.assertEqual(patch_res.output.get("occurrences"), 1)
            self.assertIn("-    return 'world'", patch_res.output.get("diff", ""))
            self.assertIn("+    return 'universe'", patch_res.output.get("diff", ""))
            self.assertIn("return 'universe'", (workspace / "module" / "app.py").read_text())

            # 3. Patch target not found
            fail_res = asyncio.run(gateway.execute("filesystem.patch", {
                "path": "module/app.py",
                "target_content": "nonexistent_target",
                "replacement_content": "something"
            }))
            self.assertFalse(fail_res.ok)
            self.assertIn("not found", fail_res.error)

            # 4. Write with syntax error returns diagnostics
            bad_code = "def syntax_broken(\n"
            bad_res = asyncio.run(gateway.execute("filesystem.write", {"path": "broken.py", "content": bad_code}))
            self.assertTrue(bad_res.ok)
            diags = bad_res.output.get("diagnostics", [])
            self.assertTrue(len(diags) > 0)
            self.assertEqual(diags[0]["severity"], "error")

            # 5. Path escaping workspace is rejected
            esc_res = asyncio.run(gateway.execute("filesystem.write", {"path": "../outside.txt", "content": "bad"}))
            self.assertFalse(esc_res.ok)
            self.assertIn("escapes workspace", esc_res.error)

    def test_approval_modes(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace)

            # Plan mode: mutations are blocked
            plan_policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="plan")
            plan_gateway = ToolGateway(registry, plan_policy)
            res = asyncio.run(plan_gateway.execute("filesystem.write", {"path": "test.txt", "content": "hello"}))
            self.assertFalse(res.ok)
            self.assertIn("policy denied", res.error)

            # Read-only in plan mode is permitted
            read_res = asyncio.run(plan_gateway.execute("calculator", {"expression": "2+3"}))
            self.assertTrue(read_res.ok)

            # Ask mode without confirm_callback: denied
            ask_policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="ask")
            ask_gateway = ToolGateway(registry, ask_policy)
            res2 = asyncio.run(ask_gateway.execute("filesystem.write", {"path": "test.txt", "content": "hello"}))
            self.assertFalse(res2.ok)
            self.assertIn("confirmation declined or required", res2.error)

            # Ask mode with approve callback: allowed
            approved_calls = []
            def on_confirm(name, args, spec):
                approved_calls.append(name)
                return True

            ask_approved_policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="ask", confirm_callback=on_confirm)
            ask_approved_gateway = ToolGateway(registry, ask_approved_policy)
            res3 = asyncio.run(ask_approved_gateway.execute("filesystem.write", {"path": "test.txt", "content": "hello"}))
            self.assertTrue(res3.ok)
            self.assertIn("filesystem.write", approved_calls)

            # Ask mode with decline callback: denied
            def on_decline(name, args, spec):
                return False

            ask_declined_policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="ask", confirm_callback=on_decline)
            ask_declined_gateway = ToolGateway(registry, ask_declined_policy)
            res4 = asyncio.run(ask_declined_gateway.execute("filesystem.write", {"path": "test2.txt", "content": "hello"}))
            self.assertFalse(res4.ok)
            self.assertIn("confirmation declined", res4.error)

            # YOLO mode: allowed up to max_risk without confirmation
            yolo_policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="yolo")
            yolo_gateway = ToolGateway(registry, yolo_policy)
            res5 = asyncio.run(yolo_gateway.execute("filesystem.write", {"path": "yolo.txt", "content": "hello"}))
            self.assertTrue(res5.ok)

    def test_event_streaming_and_diagnostics_alert(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace)
            policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="auto-edit")
            gateway = ToolGateway(registry, policy)

            events = []
            def listener(evt):
                events.append(evt)

            backend = ScriptedBackend([
                Decision(DecisionType.TOOL, tool="filesystem.write", arguments={"path": "broken.py", "content": "def foo("}),
                Decision(DecisionType.FINISH, answer="done")
            ])
            runner = AgentRunner(backend, registry, gateway, use_tool_router=False, event_listener=listener)
            state = asyncio.run(runner.run("write file", Mode.FAST, ComputeBudget(llm_tokens=1000, steps=3, tool_calls=2)))

            self.assertEqual(state.status, "success")
            self.assertTrue(any("Diagnostic alert" in note for note in state.notes))
            event_types = [e["type"] for e in events]
            self.assertIn("tool_call", event_types)
            self.assertIn("observation", event_types)
            self.assertIn("run_finished", event_types)

    def test_git_worktree_arena(self):
        with tempfile.TemporaryDirectory() as td:
            repo_dir = Path(td) / "repo"
            repo_dir.mkdir()
            import subprocess
            subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "LynxTest"], cwd=repo_dir, check=True)
            subprocess.run(["git", "config", "user.email", "test@lynx.local"], cwd=repo_dir, check=True)
            (repo_dir / "initial.txt").write_text("v1\n")
            subprocess.run(["git", "add", "initial.txt"], cwd=repo_dir, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

            def runner_factory(branch_id: str, worktree_dir: Path | None = None):
                target_ws = worktree_dir or repo_dir
                registry = ToolRegistry()
                register_builtin_tools(registry, target_ws)
                policy = ToolPolicy(workspace=target_ws, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="yolo")
                gateway = ToolGateway(registry, policy)
                backend = ScriptedBackend([
                    Decision(DecisionType.TOOL, tool="filesystem.write", arguments={"path": "initial.txt", "content": f"{branch_id}_content\n"}),
                    Decision(DecisionType.FINISH, answer=f"{branch_id} done")
                ])
                return AgentRunner(backend, registry, gateway, use_tool_router=False)

            executor = BestOfNExecutor(runner_factory, branches=2, workspace=repo_dir, use_worktrees=True)
            candidates = asyncio.run(executor.run("modify file", Mode.FAST, ComputeBudget(llm_tokens=1000, steps=3, tool_calls=2)))

            self.assertEqual(len(candidates), 2)
            for c in candidates:
                self.assertEqual(c.status, "success")
                self.assertIsNotNone(c.worktree_path)
                self.assertIn(c.branch_id, c.diff or "")

            asyncio.run(executor.cleanup_all(candidates))
            for c in candidates:
                if c.worktree_path:
                    self.assertFalse(Path(c.worktree_path).exists())

            self.assertEqual((repo_dir / "initial.txt").read_text(), "v1\n")

    def test_filesystem_grep_and_find(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace)
            policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="auto-edit")
            gateway = ToolGateway(registry, policy)

            # Create test files
            (workspace / "src").mkdir()
            (workspace / "src" / "alpha.py").write_text("def find_me():\n    return 'alpha_result'\n")
            (workspace / "src" / "beta.txt").write_text("Hello find_me in beta text\n")
            (workspace / "notes.md").write_text("# Notes\nNothing here\n")

            # 1. filesystem.grep
            grep_res = asyncio.run(gateway.execute("filesystem.grep", {"query": "find_me"}))
            self.assertTrue(grep_res.ok)
            self.assertEqual(grep_res.output["count"], 2)
            paths = [m["path"] for m in grep_res.output["matches"]]
            self.assertIn("src/alpha.py", paths)
            self.assertIn("src/beta.txt", paths)

            # 2. filesystem.grep with pattern filter
            grep_pattern = asyncio.run(gateway.execute("filesystem.grep", {"query": "find_me", "pattern": "*.py"}))
            self.assertTrue(grep_pattern.ok)
            self.assertEqual(grep_pattern.output["count"], 1)
            self.assertEqual(grep_pattern.output["matches"][0]["path"], "src/alpha.py")

            # 3. filesystem.find
            find_res = asyncio.run(gateway.execute("filesystem.find", {"pattern": "*.py"}))
            self.assertTrue(find_res.ok)
            self.assertEqual(find_res.output["count"], 1)
            self.assertEqual(find_res.output["results"][0]["path"], "src/alpha.py")

    def test_workspace_symbols(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace)
            policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="auto-edit")
            gateway = ToolGateway(registry, policy)

            code = (
                "class Calculator:\n"
                "    def add(self, a, b):\n"
                "        return a + b\n"
                "\n"
                "def compute_total(items, tax=0.1):\n"
                "    return sum(items)\n"
            )
            (workspace / "service.py").write_text(code)

            res = asyncio.run(gateway.execute("workspace.symbols", {}))
            self.assertTrue(res.ok)
            self.assertEqual(res.output["total_files"], 1)
            file_entry = res.output["files"][0]
            self.assertEqual(file_entry["path"], "service.py")
            symbol_names = [s["name"] for s in file_entry["symbols"]]
            self.assertIn("Calculator", symbol_names)
            self.assertIn("Calculator.add", symbol_names)
            self.assertIn("compute_total", symbol_names)

    def test_workspace_test(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace)
            policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="auto-edit")
            gateway = ToolGateway(registry, policy)

            # Create a simple unittest in tests/
            (workspace / "tests").mkdir()
            test_code = (
                "import unittest\n"
                "class SimpleTest(unittest.TestCase):\n"
                "    def test_pass(self):\n"
                "        self.assertEqual(2+2, 4)\n"
            )
            (workspace / "tests" / "test_example.py").write_text(test_code)

            res = asyncio.run(gateway.execute("workspace.test", {"runner": "unittest"}))
            self.assertTrue(res.ok)
            self.assertTrue(res.output["passed"])
            self.assertEqual(res.output["returncode"], 0)
            self.assertIn("Ran 1 test", res.output["output"])

    def test_workspace_snapshot_and_rollback(self):
        from lynx_harness.snapshot import WorkspaceSnapshot
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td).resolve()
            snapshot = WorkspaceSnapshot(workspace)
            registry = ToolRegistry()
            register_builtin_tools(registry, workspace, snapshot_manager=snapshot)
            policy = ToolPolicy(workspace=workspace, max_risk=RiskLevel.LOCAL_MUTATION, approval_mode="auto-edit")
            gateway = ToolGateway(registry, policy)

            # Initial existing file
            (workspace / "original.txt").write_text("initial content\n")

            snapshot.begin("run-001")
            # 1. Mutate existing file
            asyncio.run(gateway.execute("filesystem.patch", {
                "path": "original.txt",
                "target_content": "initial",
                "replacement_content": "modified"
            }))
            self.assertEqual((workspace / "original.txt").read_text(), "modified content\n")

            # 2. Create new file
            asyncio.run(gateway.execute("filesystem.write", {
                "path": "new_file.txt",
                "content": "newly created\n"
            }))
            self.assertTrue((workspace / "new_file.txt").is_file())

            # 3. Persist and restore from disk (undo)
            dump_file = snapshot.persist("run-001")
            self.assertIsNotNone(dump_file)
            self.assertTrue(dump_file.is_file())

            # 4. Rollback
            reverted = snapshot.rollback()
            self.assertEqual(len(reverted), 2)
            self.assertEqual((workspace / "original.txt").read_text(), "initial content\n")
            self.assertFalse((workspace / "new_file.txt").exists())

            # 5. Test restore_from_disk
            # Mutate again
            (workspace / "original.txt").write_text("another mutation\n")
            restored = snapshot.restore_from_disk("run-001")
            self.assertIn("restored original.txt", restored)
            self.assertEqual((workspace / "original.txt").read_text(), "initial content\n")

    def test_observation_sliding_compaction(self):
        from lynx_harness.models import AgentState, Observation, Mode
        state = AgentState("test compaction", Mode.THINK)

        # Append 4 observations
        state.observations.append(Observation("filesystem.read", True, output="line\n" * 100)) # old, will be compacted
        state.observations.append(Observation("filesystem.patch", True, output={"path": "a.py", "occurrences": 1, "diff": "diff text\n" * 20})) # old, will be compacted
        state.observations.append(Observation("calculator", True, output=42)) # recent turn 1
        state.observations.append(Observation("workspace.symbols", True, output={"files": []})) # recent turn 2

        context = state.context([])
        untrusted = [item["observation"] for item in context["untrusted_observations"]]
        self.assertEqual(len(untrusted), 4)

        # Old observations (index 0 and 1) should be compacted:
        self.assertIn("compacted for context preservation", untrusted[0]["output"])
        self.assertIn("Patch applied", untrusted[1]["output"])

        # Recent observations (index 2 and 3) should keep their exact normalized output:
        self.assertEqual(untrusted[2]["output"], "42")
        self.assertIn("files", untrusted[3]["output"])


if __name__ == "__main__":
    unittest.main()

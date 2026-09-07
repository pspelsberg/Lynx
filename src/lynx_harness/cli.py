from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from .config import Settings
from .eval import AblationEvaluator, VARIANTS, load_benchmark
from .flight import FlightRecorder
from .inference import LlamaCppBackend, ScriptedBackend
from .integrations import IntegrationManager
from .kernel import PythonKernel
from .loop import AgentRunner
from .memory import ArtifactStore, JsonlStore
from .models import AgentState, ComputeBudget, Decision, DecisionType, Mode, Observation, RiskLevel, TaskContract, ExpectedOutput, Postcondition, SecurityPolicy
from .verifier import Verifier
from .security import atomic_write_text
from .rah import LocalChildExecutor
from .model_profiles import (load_model_manifest, validate_runtime_model, canonical_model_path,
                              canonical_capability_profile, CANONICAL_MODEL_ID, CANONICAL_MODEL_FILENAME,
                              require_single_model)
from .snapshot import WorkspaceSnapshot
from .tools import ToolGateway, ToolPolicy, ToolRegistry, register_builtin_tools


def build_runner(settings: Settings, scripted: bool = False, variant: str | None = None, recorder_override=None, model_override: str | None = None, mode: Mode = Mode.THINK, integration_setup=None, approval_mode: str = "auto-edit", confirm_callback=None, event_listener=None, snapshot_manager: Any | None = None) -> AgentRunner:
    registry = ToolRegistry()
    if variant != "bare_llm": register_builtin_tools(registry, settings.workspace, snapshot_manager=snapshot_manager)
    integrations = None
    if integration_setup is not None:
        integrations = IntegrationManager(registry)
        if not callable(integration_setup):
            raise TypeError("integration_setup must be callable")
        integration_setup(integrations)
    allow_external = os.getenv("LYNX_ALLOW_EXTERNAL") == "1"
    policy = ToolPolicy(workspace=settings.workspace, max_risk=RiskLevel.EXTERNAL_MUTATION if allow_external else RiskLevel.LOCAL_MUTATION, confirmed=allow_external, approval_mode=approval_mode, confirm_callback=confirm_callback)
    gateway = ToolGateway(registry, policy)
    effective_model = model_override or settings.model
    # Resolve the shared model-policy aliases once at the process boundary;
    # downstream backends and audit metadata use the canonical filename.
    require_single_model(effective_model, field="run model")
    effective_model = CANONICAL_MODEL_FILENAME
    # The checkpoint is fixed, but mode-specific limits still matter: FAST
    # disables thinking and RESEARCH receives the larger bounded step budget.
    capability_profile = canonical_capability_profile(mode)
    backend = ScriptedBackend([Decision(DecisionType.TOOL, tool="calculator", arguments={"expression":"6*7"}), Decision(DecisionType.FINISH, answer="Offline demo: calculator tool executed; harness is operational.")]) if scripted else LlamaCppBackend(settings.server_url, effective_model, settings.timeout_s, profile=capability_profile)
    model_sha = None; manifest_id = None
    if settings.model_path:
        model_path = canonical_model_path(settings.model_path)
        if not model_path.is_file(): raise ValueError("configured model file does not exist")
        manifest_path=Path("config/model-manifest.json")
        entries=load_model_manifest(manifest_path) if manifest_path.is_file() else []
        entry=validate_runtime_model(CANONICAL_MODEL_ID, entries)
        if entry.filename != model_path.name or not entry.verify_file(model_path):
            raise ValueError("model SHA-256 does not match manifest")
        model_sha = entry.sha256; manifest_id=entry.id
    metadata = {"server_url": settings.server_url, "model": effective_model, "model_path": str(settings.model_path) if settings.model_path else None, "model_sha256": model_sha, "model_manifest_id": manifest_id, "capability_profile": {"model_id": capability_profile.model_id, "max_tool_candidates": capability_profile.max_tool_candidates, "supports_thinking": capability_profile.supports_thinking}, "backend": settings.backend, "llama_cpp_commit": settings.llama_commit}
    recorder = recorder_override if recorder_override is not None else (None if scripted else FlightRecorder(settings.flight_root))
    progress_enabled = variant in {None, "+verification"}
    context_limit = 10_000 if variant in {"rlm", "rah", "rah+verifier"} else (10_000_000 if variant == "normal_context" else 50_000)
    artifacts = ArtifactStore(settings.artifacts_path)
    verifier = Verifier() if variant in {"+verification", "rah+verifier"} else None
    child_executor = None
    if variant in {"rah", "rah+verifier"}:
        def child_factory(task):
            child_registry = ToolRegistry()
            for name in task.allowed_tools:
                spec, handler = registry.get(name)
                child_registry.register(spec, handler)
            child_policy = ToolPolicy(workspace=settings.workspace, max_risk=task.max_risk, confirmed=allow_external, approval_mode=approval_mode, confirm_callback=confirm_callback)
            return AgentRunner(backend, child_registry, ToolGateway(child_registry, child_policy), artifacts=artifacts, metadata={**metadata, "variant": "rah-child"}, use_tool_router=True, enable_progress_monitor=progress_enabled, verifier=verifier, context_max_chars=context_limit)
        child_executor = LocalChildExecutor(child_factory, artifact_store=artifacts, verifier=verifier)
    return AgentRunner(backend, registry, gateway, JsonlStore(settings.trajectory_path), artifacts, metadata={**metadata, "variant": variant or "full", "progress_monitor": progress_enabled, "context_strategy": "artifact_rlm" if variant in {"rlm", "rah", "rah+verifier"} else "normal", "approval_mode": approval_mode}, recorder=recorder, use_tool_router=False if scripted else (variant not in {"bare_llm", "normal_context"}), enable_progress_monitor=progress_enabled, verifier=verifier, context_max_chars=context_limit, child_executor=child_executor, rlm_enabled=variant == "rlm", integrations=integrations, event_listener=event_listener)


async def run_task(task: str, mode: Mode = Mode.THINK, scripted: bool = False, approval_mode: str = "ask", stream: bool = False, dual_output: str | None = None, rollback_on_failure: bool = False) -> int:
    import sys
    settings = Settings.from_env()
    snapshot = WorkspaceSnapshot(settings.workspace)

    def interactive_confirm(name: str, arguments: dict[str, Any], spec: Any) -> bool:
        if not sys.stdin.isatty():
            return os.getenv("LYNX_ALLOW_EXTERNAL") == "1"
        sys.stderr.write(f"\n[Lynx Approval] Tool: {name} (Risk: {spec.risk.name})\n")
        if name in {"filesystem.patch", "filesystem.write"}:
            sys.stderr.write(f"  Target: {arguments.get('path', '')}\n")
            if "diff" in arguments:
                sys.stderr.write(f"  Diff:\n{arguments['diff']}\n")
        elif name == "shell.exec":
            sys.stderr.write(f"  Command: {arguments.get('command', '')}\n")
        sys.stderr.write("  Approve execution? [y/N]: ")
        sys.stderr.flush()
        try:
            choice = sys.stdin.readline().strip().lower()
            return choice in {"y", "yes"}
        except (OSError, EOFError):
            return False

    dual_file = None
    if dual_output:
        dual_path = Path(dual_output).resolve()
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_file = open(dual_path, "a", encoding="utf-8")

    def event_handler(event: dict[str, Any]) -> None:
        if dual_file:
            try:
                dual_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                dual_file.flush()
            except Exception:
                pass
        if stream:
            k = event.get("kind")
            payload = event.get("payload", {})
            if k == "decision":
                d_type = payload.get("type")
                if d_type == "tool":
                    sys.stderr.write(f"-> [Decision] Tool: {payload.get('tool')} ({json.dumps(payload.get('arguments', {}), ensure_ascii=False)})\n")
                elif d_type == "reflect":
                    sys.stderr.write(f"-> [Reflect] {payload.get('reason')}\n")
                elif d_type == "finish":
                    sys.stderr.write(f"-> [Finish] {payload.get('answer', '')[:200]}\n")
            elif k == "observation":
                tool = payload.get("tool")
                ok = payload.get("success")
                ms = payload.get("elapsed_ms")
                sys.stderr.write(f"<- [Observation] {tool}: {'success' if ok else 'failed'} ({ms}ms)\n")
                out = payload.get("output")
                if isinstance(out, dict) and out.get("diff"):
                    sys.stderr.write(f"{out['diff']}\n")
                if isinstance(out, dict) and out.get("diagnostics"):
                    for d in out["diagnostics"]:
                        sys.stderr.write(f"   [Diagnostic {d.get('severity', 'info').upper()}] Line {d.get('line')}: {d.get('message')}\n")
            elif k == "repair_requested":
                sys.stderr.write(f"!! [Repair] Attempt {payload.get('attempt')}: {payload.get('reason')}\n")

    try:
        runner = build_runner(
            settings, scripted=scripted, mode=mode,
            approval_mode=approval_mode,
            confirm_callback=interactive_confirm if approval_mode == "ask" else None,
            event_listener=event_handler if (stream or dual_output) else None,
            snapshot_manager=snapshot
        )
        budget = ComputeBudget.for_mode(mode)
        contract = TaskContract(id="root-" + __import__("uuid").uuid4().hex[:24], goal=task, mode=mode, allowed_tools=tuple(spec.name for spec in runner.registry.specs()), expected_output=ExpectedOutput(max_chars=20_000), postconditions=(Postcondition("output_bounded"),), budget=budget, security=SecurityPolicy(max_risk=RiskLevel.EXTERNAL_MUTATION if os.getenv("LYNX_ALLOW_EXTERNAL") == "1" else RiskLevel.LOCAL_MUTATION, confirmation_required=os.getenv("LYNX_ALLOW_EXTERNAL") == "1"))
        snapshot.begin(contract.id)
        state = await runner.run(task, mode, budget, contract=contract)
        if rollback_on_failure and state.status not in {"success", "verified"}:
            reverted = snapshot.rollback()
            if reverted:
                sys.stderr.write(f"\n[Lynx Rollback] Reverted {len(reverted)} files due to run failure ({state.status}):\n")
                for item in reverted:
                    sys.stderr.write(f"  - {item}\n")
        else:
            snapshot.persist(state.run_id)
        print(state.answer or "\n".join(state.notes))
        return 0 if state.status in {"success", "verified"} and state.answer else 1
    finally:
        if dual_file:
            try: dual_file.close()
            except Exception: pass


def undo_task(run_id: str | None = None) -> int:
    import sys
    settings = Settings.from_env()
    snapshot = WorkspaceSnapshot(settings.workspace)
    if not run_id:
        if not snapshot.snapshot_dir.is_dir():
            sys.stderr.write("No snapshot history found.\n")
            return 1
        snapshots = sorted(snapshot.snapshot_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            sys.stderr.write("No snapshot history found.\n")
            return 1
        run_id = snapshots[0].stem
    try:
        restored = snapshot.restore_from_disk(run_id)
        if restored:
            print(f"Successfully reverted {len(restored)} file(s) from run {run_id}:")
            for item in restored:
                print(f"  - {item}")
        else:
            print(f"No file modifications recorded for run {run_id}.")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Undo failed: {exc}\n")
        return 1


async def kernel_task(code: str) -> int:
    settings=Settings.from_env(); kernel=PythonKernel(settings.workspace, allow_network=os.getenv("LYNX_KERNEL_ALLOW_NETWORK") == "1")
    try: print(json.dumps(await kernel.execute(code), ensure_ascii=False))
    finally: await kernel.close()
    return 0


async def eval_task(path: str, variants: list[str], output: str | None = None) -> int:
    settings = Settings.from_env()
    cases = load_benchmark(Path(path))
    evaluator = AblationEvaluator(lambda variant: build_runner(settings, variant=variant))
    rows = await evaluator.evaluate(cases, variants)
    result = {"benchmark": path, "variants": variants,
              "model_policy": {"model_id": CANONICAL_MODEL_ID,
                               "model_filename": CANONICAL_MODEL_FILENAME},
              "aggregate": evaluator.aggregate(rows)}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        atomic_write_text(Path(output), encoded + "\n", max_bytes=50_000_000)
    else: print(encoded)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="lynx", description="Lynx small-model-first agent harness POC")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a task through llama-server")
    run.add_argument("task")
    run.add_argument("--mode", choices=[mode.value for mode in Mode], default=Mode.THINK.value)
    run.add_argument("--approval-mode", choices=["plan", "ask", "auto-edit", "auto", "yolo"], default="ask", help="governance approval mode for tool execution")
    run.add_argument("--stream", action="store_true", help="stream decisions, diffs, and diagnostics live to stderr")
    run.add_argument("--dual-output", default=None, metavar="PATH", help="stream structured JSONL events to a sidecar file or pipe")
    run.add_argument("--rollback-on-failure", action="store_true", help="automatically revert workspace file edits if task fails or is rejected")
    sub.add_parser("demo", help="offline smoke run without a model server")
    undo = sub.add_parser("undo", help="revert file modifications made during a specific run")
    undo.add_argument("run_id", nargs="?", default=None, help="run ID to revert (defaults to most recent run)")
    kernel = sub.add_parser("kernel", help="execute bounded code in an isolated worker")
    kernel.add_argument("code")
    replay = sub.add_parser("replay", help="inspect a recorded run without invoking tools or the model")
    replay.add_argument("run_id")
    replay.add_argument("--branch", default="main")
    fork = sub.add_parser("fork", help="fork a recorded prefix into a new isolated run")
    fork.add_argument("run_id"); fork.add_argument("--from", dest="from_seq", type=int, required=True); fork.add_argument("--branch", default=None)
    resume = sub.add_parser("resume", help="continue an isolated fork from a recorded checkpoint")
    resume.add_argument("run_id"); resume.add_argument("--from", dest="from_seq", type=int, required=True); resume.add_argument("--branch", default="main"); resume.add_argument("--model", default=None)
    evaluation = sub.add_parser("eval", help="run deterministic ablation variants from JSON/YAML benchmark")
    evaluation.add_argument("benchmark")
    evaluation.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["bare_llm", "+tool_router", "+structured_state", "+verification"])
    evaluation.add_argument("--output")
    args = parser.parse_args()
    if args.command == "demo": return asyncio.run(run_task("offline smoke test", Mode.FAST, scripted=True))
    if args.command == "undo": return undo_task(args.run_id)
    if args.command == "kernel": return asyncio.run(kernel_task(args.code))
    if args.command == "eval": return asyncio.run(eval_task(args.benchmark, args.variants, args.output))
    if args.command == "resume":
        settings = Settings.from_env(); source = FlightRecorder(settings.flight_root, args.run_id, args.branch, create=False)
        child = source.fork(args.from_seq)
        checkpoint_path = source.run_dir / "checkpoints" / f"{args.from_seq:06d}.json"
        if not checkpoint_path.is_file(): parser.error("fork point has no checkpoint")
        document = json.loads(checkpoint_path.read_text(encoding="utf-8")); raw = document["state"]
        state = AgentState(goal=raw["goal"], mode=Mode(raw["mode"]), run_id=child.run_id, branch_id=child.branch_id, parent_event=args.from_seq, observations=[Observation(**item) for item in raw.get("observations", [])], notes=list(raw.get("notes", [])), status="running", steps=int(raw.get("steps", 0)))
        budget = ComputeBudget(**document["budget"]); result = asyncio.run(build_runner(settings, recorder_override=child, model_override=args.model, mode=state.mode).run(initial_state=state, budget=budget))
        print(result.answer or "\n".join(result.notes)); return 0 if result.answer else 1
    if args.command in {"replay", "fork"}:
        settings = Settings.from_env(); recorder = FlightRecorder(settings.flight_root, args.run_id, args.branch if args.command == "replay" else "main", create=False)
        if args.command == "replay":
            events = recorder.events(args.branch)
            started=next((event for event in events if event.get("type")=="run_started"), None)
            if started:
                payload=started.get("payload", {}); raw_budget=payload.get("budget", {})
                goal=str(payload.get("goal", "")); mode=Mode(payload.get("mode", "think"))
                limits={key:value for key,value in raw_budget.items() if not key.startswith("used_")}
                replay_runner=build_runner(settings, scripted=True)
                replay_runner.backend=__import__("lynx_harness.flight",fromlist=["ReplayBackend"]).ReplayBackend(events)
                replay_runner.gateway=__import__("lynx_harness.flight",fromlist=["ReplayGateway"]).ReplayGateway(events)
                # Reconstruct the same immutable contract that shaped the
                # original context; otherwise replay hashes diverge.
                replay_contract = None
                raw_contract = payload.get("task_contract")
                if isinstance(raw_contract, dict):
                    replay_contract = TaskContract.from_dict(raw_contract)
                state=asyncio.run(replay_runner.run(goal,mode,ComputeBudget(**limits), contract=replay_contract))
                print(json.dumps({"run_id":args.run_id,"branch":args.branch,"status":state.status,"answer":state.answer,"events":len(events)},ensure_ascii=False)); return 0 if state.status in {"success","verified"} else 1
            print(json.dumps({"run_id":args.run_id,"branch":args.branch,"events":len(events),"types":[event["type"] for event in events]}, indent=2)); return 0
        child = recorder.fork(args.from_seq, new_branch_id=args.branch); print(json.dumps({"run_id":child.run_id,"branch":child.branch_id,"source":args.run_id,"from_seq":args.from_seq}, indent=2)); return 0
    return asyncio.run(run_task(args.task, Mode(args.mode), approval_mode=args.approval_mode, stream=args.stream, dual_output=args.dual_output, rollback_on_failure=args.rollback_on_failure))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import asyncio
import time
import json
import math
from typing import Any, Callable, Awaitable

from .models import AgentState, ComputeBudget, Mode, BudgetLedger, BudgetExceeded, RiskLevel, TaskContract, _artifact_ref
from .verifier import VerificationResult, VerificationStatus, _schema_match, Verifier


@dataclass(frozen=True)
class ChildBudget:
    llm_tokens: int = 1_000
    steps: int = 4
    tool_calls: int = 4
    web_calls: int = 0
    browser_actions: int = 0
    python_seconds: float = 0.0
    recursive_calls: int = 0
    max_wall_time_s: float = 30.0
    depth: int = 0
    max_depth: int = 2

    def validate(self) -> None:
        if (type(self.llm_tokens) is not int or not 1 <= self.llm_tokens <= 100_000
                or type(self.steps) is not int or not 1 <= self.steps <= 100
                or type(self.tool_calls) is not int or not 0 <= self.tool_calls <= 100
                or type(self.web_calls) is not int or not 0 <= self.web_calls <= self.tool_calls
                or type(self.browser_actions) is not int or not 0 <= self.browser_actions <= self.tool_calls
                or isinstance(self.python_seconds, bool) or not isinstance(self.python_seconds, (int, float))
                or not math.isfinite(float(self.python_seconds)) or not 0 <= self.python_seconds <= 86_400
                or type(self.recursive_calls) is not int or not 0 <= self.recursive_calls <= 10_000
                or isinstance(self.max_wall_time_s, bool) or not isinstance(self.max_wall_time_s, (int, float))
                or not math.isfinite(float(self.max_wall_time_s)) or not 0.1 <= self.max_wall_time_s <= 600
                or type(self.depth) is not int or type(self.max_depth) is not int
                or not 0 <= self.depth <= self.max_depth <= 8):
            raise ValueError("child budget is outside safe bounds")


@dataclass(frozen=True)
class ChildTask:
    goal: str
    artifact_refs: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    expected_schema: dict[str, Any] | None = None
    budget: ChildBudget = ChildBudget()
    depth: int = 0
    task_id: str = ""
    parent_task_id: str | None = None
    max_risk: RiskLevel = RiskLevel.READ_ONLY
    branch_id: str = "main"
    allow_incomplete: bool = False

    def context_firewall(self) -> dict[str, Any]:
        self.budget.validate()
        if not self.goal.strip() or len(self.goal) > 20_000: raise ValueError("child goal invalid")
        if not isinstance(self.artifact_refs, (tuple, list)) or len(self.artifact_refs) > 100 or len(set(self.artifact_refs)) != len(self.artifact_refs):
            raise ValueError("invalid child inputs")
        try:
            for ref in self.artifact_refs: _artifact_ref(ref, "child input")
        except (TypeError, ValueError) as exc: raise ValueError("child inputs must be canonical artifact references") from exc
        try: object.__setattr__(self, "max_risk", self.max_risk if isinstance(self.max_risk, RiskLevel) else RiskLevel[str(self.max_risk).upper()])
        except (KeyError, ValueError, TypeError) as exc: raise ValueError("invalid child risk") from exc
        if not isinstance(self.allowed_tools, (tuple, list)) or len(self.allowed_tools) > 100:
            raise ValueError("invalid child tools")
        if any(not isinstance(name, str) or not name or len(name) > 100 or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in name) for name in self.allowed_tools): raise ValueError("invalid child tool")
        if len(set(self.allowed_tools)) != len(self.allowed_tools): raise ValueError("duplicate child tool")
        if self.depth != self.budget.depth: raise ValueError("depth mismatch")
        if self.task_id and (len(self.task_id) > 128 or not self.task_id.replace("-", "").replace("_", "").isalnum()): raise ValueError("invalid child task id")
        if self.parent_task_id and (len(self.parent_task_id) > 128 or not self.parent_task_id.replace("-", "").replace("_", "").isalnum()): raise ValueError("invalid parent task id")
        if not isinstance(self.branch_id, str) or len(self.branch_id)>128 or not self.branch_id.replace("-", "").replace("_", "").isalnum(): raise ValueError("invalid child branch id")
        if not isinstance(self.allow_incomplete, bool): raise ValueError("allow_incomplete must be boolean")
        if self.expected_schema is not None:
            try: schema_size = len(json.dumps(self.expected_schema, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            except (TypeError, ValueError) as exc: raise ValueError("invalid expected schema") from exc
            if not isinstance(self.expected_schema, dict) or schema_size > 100_000: raise ValueError("invalid expected schema")
        return {"task_id": self.task_id, "parent_task_id": self.parent_task_id, "goal": self.goal, "artifact_refs": list(self.artifact_refs), "allowed_tools": list(self.allowed_tools), "expected_schema": self.expected_schema or {}, "budget": asdict(self.budget), "depth": self.depth, "max_risk": self.max_risk.name, "branch_id": self.branch_id, "allow_incomplete": self.allow_incomplete}


@dataclass(frozen=True)
class ChildResult:
    status: str
    answer: str = ""
    artifact_refs: tuple[str, ...] = ()
    steps: int = 0
    tool_calls: int = 0
    elapsed_ms: int = 0
    task_id: str = ""
    claims: tuple[dict[str, Any], ...] = ()
    verification: VerificationResult | None = None
    usage: dict[str, Any] | None = None

    @property
    def artifacts(self) -> tuple[str, ...]: return self.artifact_refs

    def validate(self, expected_schema: dict[str, Any] | None = None, *, max_chars: int = 20_000) -> "ChildResult":
        if self.status not in {"complete", "completed", "success", "verified", "incomplete", "failed", "rejected", "timeout", "cancelled", "budget_exhausted"}:
            raise ValueError("invalid child result status")
        if not isinstance(self.answer, str) or len(self.answer) > max_chars: raise ValueError("child answer is too large")
        if len(self.artifact_refs) > 100:
            raise ValueError("invalid child artifact reference")
        for ref in self.artifact_refs:
            try:
                _artifact_ref(ref, "child artifact")
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid child artifact reference") from exc
        if len(set(self.artifact_refs)) != len(self.artifact_refs):
            raise ValueError("duplicate child artifact reference")
        if type(self.steps) is not int or type(self.tool_calls) is not int or type(self.elapsed_ms) is not int or self.steps < 0 or self.tool_calls < 0 or self.elapsed_ms < 0: raise ValueError("invalid child usage")
        if len(self.claims)>100 or any(not isinstance(claim,dict) or len(json.dumps(claim,ensure_ascii=False))>20_000 for claim in self.claims): raise ValueError("invalid child claims")
        if self.usage is not None:
            usage_dict=asdict(self.usage) if hasattr(self.usage,"__dataclass_fields__") else self.usage
            try:
                encoded_usage = json.dumps(usage_dict, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid child usage object") from exc
            if not isinstance(usage_dict,dict) or len(encoded_usage)>20_000: raise ValueError("invalid child usage object")
            allowed_usage = {"llm_tokens", "steps", "tool_calls", "web_calls",
                             "browser_actions", "python_seconds", "recursive_calls",
                             "parallel_branches", "elapsed_ms"}
            if set(usage_dict) - allowed_usage:
                raise ValueError("invalid child usage fields")
            for key, value in usage_dict.items():
                if key == "python_seconds":
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(float(value)) or value < 0):
                        raise ValueError("invalid child usage values")
                elif type(value) is not int or value < 0:
                    raise ValueError("invalid child usage values")
        if expected_schema:
            errors: list[str] = []
            if not _schema_match({"task_id": self.task_id, "status": self.status, "answer": self.answer, "artifacts": list(self.artifact_refs), "claims": list(self.claims), "usage": (asdict(self.usage) if hasattr(self.usage, "__dataclass_fields__") else (self.usage or {}))}, expected_schema, errors=errors):
                raise ValueError("child result does not match expected schema: " + "; ".join(errors[:5]))
        return self


def as_child_task(task: ChildTask | TaskContract) -> ChildTask:
    if isinstance(task, ChildTask): return task
    if not isinstance(task, TaskContract): raise TypeError("child executor accepts ChildTask or TaskContract")
    schema = task.expected_output.schema if isinstance(task.expected_output.schema, dict) else None
    b=task.budget
    if task.depth > 8: raise ValueError("child depth exceeds safety limit")
    # ChildBudget models web/browser calls as subsets of total tool calls;
    # normalize the broader ComputeBudget defaults before crossing that seam.
    child_budget=ChildBudget(llm_tokens=b.llm_tokens, steps=b.steps, tool_calls=b.tool_calls,
                             web_calls=min(b.web_calls, b.tool_calls),
                             browser_actions=min(b.browser_actions, b.tool_calls),
                             python_seconds=b.python_seconds, recursive_calls=b.recursive_calls,
                             max_wall_time_s=b.max_wall_time_s, depth=task.depth,
                             max_depth=task.max_depth)
    return ChildTask(task.goal, tuple(task.input_artifacts), tuple(task.allowed_tools), schema, child_budget, task.depth, task.id, task.parent_task_id, task.security.max_risk, task.branch_id, False)

class LocalChildExecutor:
    """Runs an isolated child and atomically accounts for it in an optional root ledger."""
    def __init__(self, runner_factory: Callable[[ChildTask], Any], ledger: BudgetLedger | None = None, artifact_store: Any | None = None, verifier: Verifier | None = None): self.runner_factory, self.ledger, self.artifact_store, self.verifier = runner_factory, ledger, artifact_store, verifier

    async def execute(self, task: ChildTask | TaskContract, *, recorder: Any | None = None, parent_event: int | None = None, cancellation: Any | None = None) -> ChildResult:
        original_contract = task if isinstance(task, TaskContract) else None
        task = as_child_task(task)
        task.context_firewall(); started = time.monotonic()
        if task.artifact_refs and self.artifact_store is None:
            raise ValueError("artifact inputs require an artifact store")
        if self.artifact_store:
            for ref in task.artifact_refs:
                # get() and metadata() both verify the content hash and the
                # persisted provenance before a child can observe the input.
                self.artifact_store.get(ref, max_bytes=10_000_000)
                metadata = self.artifact_store.metadata(ref)
                if metadata.branch_id != task.branch_id: raise ValueError("artifact belongs to another branch")
        reservation = None
        if self.ledger:
            try:
                # Reserve only the child slot itself. The child runner uses
                # this same ledger for each actual resource operation; reserving
                # its full local budget here would leave no capacity for those
                # per-operation reservations and self-deny the child.
                reservation = self.ledger.reserve(recursive_calls=1)
            except BudgetExceeded:
                return ChildResult("budget_exhausted", task_id=task.task_id or "")
        try:
            runner = self.runner_factory(task)
            # Descendant work must use the same root ledger. A factory-created
            # runner may otherwise create a private ledger and make nested RLM/
            # child calls invisible to the aggregate budget.
            if self.ledger is not None and hasattr(runner, "ledger"):
                runner.ledger = self.ledger
            if getattr(runner, "child_executor", None) is None: runner.child_executor = self
            elif hasattr(runner.child_executor, "ledger") and self.ledger is not None:
                runner.child_executor.ledger = self.ledger
            if recorder is not None:
                from .flight import FlightRecorder
                child_recorder = FlightRecorder(recorder.root, run_id=None, branch_id=task.branch_id, task_id=task.task_id or None, parent_task_id=task.parent_task_id, parent_event=parent_event)
                runner.recorder = child_recorder
            exposed = {spec.name for spec in runner.registry.specs()}
            if not exposed.issubset(set(task.allowed_tools)):
                raise ValueError("child runner exposes tools outside its firewall")
            for spec in runner.registry.specs():
                if spec.risk > task.max_risk: raise ValueError("child risk exceeds its firewall")
            runner.registry.freeze()
            budget = ComputeBudget(llm_tokens=task.budget.llm_tokens, steps=task.budget.steps, tool_calls=task.budget.tool_calls, web_calls=task.budget.web_calls, browser_actions=task.budget.browser_actions, python_seconds=task.budget.python_seconds, max_wall_time_s=task.budget.max_wall_time_s, recursive_calls=task.budget.recursive_calls)
            child_contract = original_contract
            if original_contract is not None:
                # Keep the immutable contract as the policy source while
                # aligning its execution snapshot with the bounded ChildBudget
                # (including recursive and tool-family dimensions).
                child_contract = replace(original_contract, budget=budget)
            try:
                state = await __import__("lynx_harness.cancellation", fromlist=["bounded"]).bounded(runner.run(task.goal, original_contract.mode if original_contract else Mode.THINK, budget, contract=child_contract, cancellation=cancellation), task.budget.max_wall_time_s, cancellation)
            except asyncio.TimeoutError:
                elapsed_ms = int((time.monotonic()-started)*1000)
                result = ChildResult("timeout", task_id=task.task_id or "", elapsed_ms=elapsed_ms, usage={"llm_tokens": budget.used_llm_tokens, "steps": budget.used_steps, "tool_calls": budget.used_tool_calls, "web_calls": budget.used_web_calls, "browser_actions": budget.used_browser_actions, "python_seconds": budget.used_python_seconds, "elapsed_ms": elapsed_ms})
                if reservation: reservation.commit({"recursive_calls": 1})
                return result
            answer = state.answer[:20_000]
            refs=tuple(dict.fromkeys(item.artifact_id for item in state.observations if item.artifact_id))
            if len(state.answer)>5_000 and getattr(runner, "artifacts", None): refs=refs + (runner.artifacts.put(state.answer, branch_id=task.branch_id, task_id=task.task_id or None),)
            # Child output references are an untrusted claim until resolved
            # through the shared content-addressed store.  Validate both the
            # digest/provenance and branch before returning them to a parent;
            # otherwise a child could smuggle arbitrary artifact:// IDs into a
            # merge proposal without producing the referenced bytes.
            if refs:
                if self.artifact_store is None:
                    raise ValueError("child artifacts require an artifact store")
                for ref in refs:
                    self.artifact_store.get(ref, max_bytes=10_000_000)
                    metadata = self.artifact_store.metadata(ref)
                    if metadata.branch_id != task.branch_id:
                        raise ValueError("child artifact belongs to another branch")
            verification = None
            if original_contract and self.verifier:
                verification = self.verifier.check({"answer": answer, "artifacts": list(refs)}, original_contract, {"artifacts": getattr(runner, "artifacts", None), "artifact_refs": refs, "branch_id": task.branch_id, "tools_used": [item.tool for item in state.observations]})
            elapsed_ms = int((time.monotonic()-started)*1000)
            result = ChildResult(state.status, answer, refs, state.steps, len(state.observations), elapsed_ms, task.task_id or "", verification=verification, usage={"llm_tokens": budget.used_llm_tokens, "steps": budget.used_steps, "tool_calls": budget.used_tool_calls, "web_calls": budget.used_web_calls, "browser_actions": budget.used_browser_actions, "python_seconds": budget.used_python_seconds, "elapsed_ms": elapsed_ms})
            result.validate(task.expected_schema)
            if verification and verification.status != VerificationStatus.PASSED: result = ChildResult("rejected", answer, refs, state.steps, len(state.observations), result.elapsed_ms, task.task_id or "", verification=verification, usage=result.usage)
            if reservation: reservation.commit({"recursive_calls": 1})
            return result
        except asyncio.CancelledError:
            if reservation: reservation.release()
            raise
        except Exception:
            if reservation: reservation.release()
            raise


class ParallelChildScheduler:
    """Fair bounded scheduler; one shared ledger accounts every child."""
    def __init__(self, executor: LocalChildExecutor, max_concurrency: int = 1):
        if not 1 <= max_concurrency <= 32: raise ValueError("max_concurrency out of bounds")
        self.executor, self.max_concurrency = executor, max_concurrency
    async def run(self, tasks: list[ChildTask]) -> list[ChildResult]:
        if len(tasks)>100: raise ValueError("too many child tasks")
        semaphore=asyncio.Semaphore(self.max_concurrency)
        async def one(task):
            async with semaphore: return await self.executor.execute(task)
        running = [asyncio.create_task(one(task)) for task in tasks]
        try:
            return list(await asyncio.gather(*running))
        except BaseException:
            for task in running:
                if not task.done(): task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import tempfile
from pathlib import Path
import uuid
import re
from typing import Any

from .models import AgentState, ComputeBudget
from .progress import digest
from .security import redact


def _safe_component(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError(f"invalid {field}")
    return value


def _stable_input(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_input(item) for key, item in value.items() if key not in {"run_id", "timestamp"}}
    if isinstance(value, list): return [_stable_input(item) for item in value]
    return value


class ReplayDivergence(RuntimeError):
    def __init__(self, expected: Any, actual: Any, seq: int):
        super().__init__(f"replay divergence at event {seq}: expected={expected!r} actual={actual!r}")
        self.expected, self.actual, self.seq = expected, actual, seq


class FlightRecorder:
    """Per-run append-only recorder with checkpoints and prefix forks."""
    def __init__(self, root: Path, run_id: str | None = None, branch_id: str = "main", *, create: bool = True, task_id: str | None = None, parent_task_id: str | None = None, parent_event: int | None = None):
        # Flight data is an append-only audit sink.  Refuse symlinked roots and
        # run directories instead of allowing a caller/attacker to redirect
        # records outside the configured flight root.
        if not isinstance(root, Path):
            raise ValueError("flight root must be a Path")
        absolute_root = root.absolute()
        if root.is_symlink() or any(parent.is_symlink() for parent in absolute_root.parents):
            raise ValueError("flight root and ancestors must not be symlinks")
        self.root = absolute_root
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("flight root must be a regular directory")
        self.run_id = _safe_component(run_id, "run_id") if run_id else ""
        self.branch_id = _safe_component(branch_id, "branch_id")
        self.task_id, self.parent_task_id, self.parent_event = task_id, parent_task_id, parent_event
        for value, label in ((task_id, "task_id"), (parent_task_id, "parent_task_id")):
            if value is not None: _safe_component(value, label)
        self.run_dir = self.root / self.run_id if self.run_id else self.root
        if self.run_dir.exists() and (self.run_dir.is_symlink() or not self.run_dir.is_dir()):
            raise ValueError("flight run directory must be a regular directory")
        self.events_path = self.run_dir / f"{self.branch_id}.events.jsonl"
        # Existing audit sinks are untrusted filesystem state.  Refuse a
        # symlink (and non-regular file) before reopening or reading it.
        if self.events_path.is_symlink() or (self.events_path.exists() and not self.events_path.is_file()):
            raise ValueError("flight event sink must be a regular non-symlink file")
        self.max_event_store_bytes = 100_000_000
        self.seq = 0
        if create and self.run_id:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if self.run_dir.is_symlink() or not self.run_dir.is_dir():
                raise ValueError("flight run directory must be a regular directory")
            (self.run_dir / "artifacts").mkdir(exist_ok=True)
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)
            for folder in ("prompts", "model_outputs", "tool_calls", "observations"):
                (self.run_dir / folder).mkdir(exist_ok=True)
            if not (self.run_dir / "manifest.json").exists():
                self._write_json(self.run_dir / "manifest.json", {"run_id": self.run_id, "branch_id": branch_id, "task_id": self.task_id, "parent_task_id": self.parent_task_id, "parent_event": self.parent_event, "created_at": datetime.now(timezone.utc).isoformat(), "schema": "lynx.flight.v2", "schema_version": 2})
            if self.events_path.is_symlink() or (self.events_path.exists() and not self.events_path.is_file()):
                raise ValueError("flight events path must be a regular non-symlink file")
            if self.events_path.exists():
                existing = self.events()
                self.seq = existing[-1]["seq"] if existing else 0

    def bind(self, run_id: str) -> None:
        if self.seq or self.events_path.exists():
            raise ValueError("cannot rebind a recorder after recording")
        self.run_id = _safe_component(run_id, "run_id")
        self.run_dir = self.root / self.run_id
        if self.run_dir.exists() and (self.run_dir.is_symlink() or not self.run_dir.is_dir()):
            raise ValueError("flight run directory must be a regular directory")
        self.events_path = self.run_dir / f"{self.branch_id}.events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.run_dir.is_symlink() or not self.run_dir.is_dir():
            raise ValueError("flight run directory must be a regular directory")
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        for folder in ("prompts", "model_outputs", "tool_calls", "observations"):
            (self.run_dir / folder).mkdir(exist_ok=True)
        self._write_json(self.run_dir / "manifest.json", {"run_id": run_id, "branch_id": self.branch_id, "task_id": self.task_id, "parent_task_id": self.parent_task_id, "parent_event": self.parent_event, "created_at": datetime.now(timezone.utc).isoformat(), "schema": "lynx.flight.v2", "schema_version": 2})

    def update_manifest(self, **metadata: Any) -> None:
        if not self.run_id: return
        path=self.run_dir / "manifest.json"
        current=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"run_id":self.run_id}
        current.update(redact(metadata)); current.setdefault("schema", "lynx.flight.v2"); current.setdefault("schema_version", 2)
        self._write_json(path, current)

    def _write_json(self, path: Path, value: Any) -> None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("flight JSON sink must be a regular non-symlink file")
        encoded = json.dumps(redact(value), ensure_ascii=False, indent=2, default=str, allow_nan=False)
        if len(encoded.encode("utf-8")) > 5_000_000: raise ValueError("flight JSON exceeds 5 MB")
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir() or (path.exists() and (path.is_symlink() or not path.is_file())):
            raise ValueError("flight path must be a regular non-symlink file")
        # Write a private temporary file and replace the destination.  Unlike
        # ``Path.write_text``, this never follows a pre-existing destination
        # symlink; replacement atomically removes such a link instead.
        fd, temporary_name = tempfile.mkstemp(prefix=".flight-", suffix=".tmp", dir=str(parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def _copy_atomic(source: Path, destination: Path) -> None:
        """Copy a checkpoint without following a raced destination symlink."""
        if (source.is_symlink() or not source.is_file()
                or any(parent.is_symlink() for parent in source.absolute().parents)):
            raise ValueError("flight checkpoint must be a regular non-symlink file")
        parent = destination.parent
        if parent.is_symlink() or not parent.is_dir() or destination.is_symlink():
            raise ValueError("flight checkpoint destination must be a regular path")
        fd, temporary_name = tempfile.mkstemp(prefix=".flight-copy-", suffix=".tmp", dir=str(parent))
        temporary = Path(temporary_name)
        source_fd = None
        try:
            # Re-check the source at open time too; the pre-check above is
            # useful for diagnostics but is not race-safe by itself.
            source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(source_fd, "rb") as input_stream, os.fdopen(fd, "wb") as output_stream:
                source_fd = fd = None
                while True:
                    chunk = input_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    output_stream.write(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            temporary.replace(destination)
        except BaseException:
            for open_fd in (source_fd, fd):
                if open_fd is not None:
                    try: os.close(open_fd)
                    except OSError: pass
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def record(self, kind: str, payload: dict[str, Any], *, input_value: Any = None, parent: int | None = None, budget_after: dict[str, Any] | None = None) -> dict[str, Any]:
        self.seq += 1
        event = {"schema_version": 2, "seq": self.seq, "type": kind, "run_id": self.run_id, "task_id": self.task_id, "parent_task_id": self.parent_task_id, "branch": self.branch_id, "parent": parent, "lineage_parent_event": self.parent_event, "input_hash": digest(_stable_input(input_value)) if input_value is not None else None, "payload": redact(payload), "budget_after": redact(budget_after or {}), "timestamp": datetime.now(timezone.utc).isoformat()}
        encoded = json.dumps(event, ensure_ascii=False, default=str, allow_nan=False)
        if len(encoded.encode()) > 1_000_000: raise ValueError("flight event exceeds 1 MB")
        if self.events_path.parent.is_symlink() or not self.events_path.parent.is_dir() or self.events_path.is_symlink():
            raise ValueError("flight events path must be a regular non-symlink file")
        try:
            current_size = self.events_path.stat().st_size if self.events_path.exists() else 0
        except OSError as exc:
            raise ValueError("flight events path is unavailable") from exc
        if current_size + len(encoded) + 1 > self.max_event_store_bytes:
            raise ValueError("flight event store exceeds configured limit")
        # O_NOFOLLOW closes the check-then-open symlink race on the final
        # component while O_APPEND keeps each event append-only.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.events_path, flags, 0o600)
        except OSError as exc:
            raise ValueError("flight events path must be a regular non-symlink file") from exc
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        return event

    def events(self, branch_id: str | None = None) -> list[dict[str, Any]]:
        selected_branch = branch_id or self.branch_id
        _safe_component(selected_branch, "branch_id")
        path = self.run_dir / f"{selected_branch}.events.jsonl"
        # ``Path.exists()`` is false for a dangling symlink, so inspect the
        # link before the missing-file fast path.
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("flight events path must be a regular non-symlink file")
        if not path.exists(): return []
        try:
            if path.stat().st_size > self.max_event_store_bytes:
                raise ValueError("flight event store exceeds configured limit")
        except OSError as exc:
            raise ValueError("flight event store is unavailable") from exc
        events=[]; total=0
        # Stream the append-only file instead of materializing an attacker-sized
        # audit log in one allocation. Bound both each event and the aggregate.
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                total += len(line.encode("utf-8"))
                if total > self.max_event_store_bytes or len(line.encode("utf-8")) > 1_000_000:
                    raise ValueError("flight event store exceeds configured limit")
                if not line.strip(): continue
                try: event=json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError("invalid flight event JSON") from exc
                version=event.get("schema_version", 1)
                if version not in {1,2} or not isinstance(event.get("seq"), int) or not isinstance(event.get("type"), str): raise ValueError("unsupported flight event schema")
                if version == 1: event["schema_version"]=1
                events.append(event)
        return events

    def checkpoint(self, state: AgentState, budget: ComputeBudget) -> Path:
        path = self.run_dir / "checkpoints" / f"{self.seq:06d}.json"
        self._write_json(path, {"seq": self.seq, "state": asdict(state), "budget": asdict(budget)})
        return path

    def result(self, state: AgentState, status: str) -> Path:
        path = self.run_dir / "result.json"
        self._write_json(path, {"run_id": self.run_id, "branch": self.branch_id, "status": status, "state": asdict(state)})
        return path

    def fork(self, at_seq: int, new_run_id: str | None = None, new_branch_id: str | None = None) -> "FlightRecorder":
        all_events = self.events()
        prefix = [event for event in all_events if event["seq"] <= at_seq]
        if not prefix or prefix[-1]["seq"] != at_seq:
            raise ValueError("fork point does not exist")
        if prefix[-1]["type"] not in {"observation", "stagnation_detected", "run_finished"}:
            raise ValueError("fork only supports complete-cycle events")
        child = FlightRecorder(self.root, new_run_id or f"{self.run_id}_fork_{uuid.uuid4().hex[:6]}", new_branch_id or f"branch_{uuid.uuid4().hex[:6]}")
        child.record("forked_from", {"source_run": self.run_id, "source_branch": self.branch_id, "source_seq": at_seq})
        for event in prefix:
            # Preserve source lineage, but never reuse parent run/branch IDs or mutable event files.
            child.record("fork_prefix", {"source_seq": event["seq"], "source_type": event["type"], "payload": event["payload"]})
        checkpoint = self.run_dir / "checkpoints" / f"{at_seq:06d}.json"
        if checkpoint.is_symlink():
            raise ValueError("flight checkpoint must be a regular non-symlink file")
        if checkpoint.is_file():
            child._copy_atomic(checkpoint, child.run_dir / "checkpoints" / f"{at_seq:06d}.json")
        return child


class ReplayGateway:
    """Serve recorded observations only; external tools are never invoked."""
    def __init__(self, events: list[dict[str, Any]]):
        self.expected = [event for event in events if event["type"] == "observation"]
        self.index = 0

    async def execute(self, name: str, arguments: dict[str, Any]):
        from .models import Observation
        if self.index >= len(self.expected): raise ReplayDivergence("no more observations", {"tool":name,"arguments":arguments}, -1)
        event = self.expected[self.index]; self.index += 1
        expected = event["payload"]
        actual_hash = digest({"tool": name, "arguments": arguments})
        if event.get("input_hash") and event["input_hash"] != actual_hash:
            raise ReplayDivergence(event["input_hash"], actual_hash, event["seq"])
        return Observation(**{key: expected[key] for key in ("tool", "ok", "output", "error", "artifact_id", "elapsed_ms", "origin", "trust", "provenance", "branch_id", "parent_event", "epistemic_status") if key in expected})


class ReplayBackend:
    """Return recorded decisions and fail closed when the context diverges."""
    def __init__(self, events: list[dict[str, Any]]):
        self.expected = [event for event in events if event["type"] == "decision"]
        self.index = 0

    async def decide(self, context: dict[str, Any], tools: list[dict[str, Any]], max_tokens: int):
        from .inference import parse_decision
        if self.index >= len(self.expected): raise ReplayDivergence("no more decisions", {"context":context}, -1)
        event = self.expected[self.index]; self.index += 1
        actual_hash = digest(_stable_input({"context": context, "tools": tools}))
        if event.get("input_hash") and event["input_hash"] != actual_hash:
            raise ReplayDivergence(event["input_hash"], actual_hash, event["seq"])
        payload = event["payload"]
        if payload.get("type") == "tool":
            return parse_decision(json.dumps({"type":"tool", "tool":payload["tool"], "arguments":payload.get("arguments", {})}), {item["name"] for item in tools})
        if payload.get("type") == "finish":
            return parse_decision(json.dumps({"type":"finish", "answer":payload.get("answer", "")}), set())
        if payload.get("type") == "delegate":
            task=payload.get("task")
            if not isinstance(task, dict): raise ReplayDivergence("delegate task", task, event["seq"])
            return parse_decision(json.dumps({"type":"delegate", "task":task}), {item["name"] for item in tools})
        if payload.get("type") == "reflect":
            return parse_decision(json.dumps({"type":"reflect", "reason":payload.get("reason", "")}), set())
        raise ReplayDivergence("known decision", payload.get("type"), event["seq"])


class ReplayChildExecutor:
    """Replay a recorded child by task ID; no child inference or external gateway is invoked."""
    def __init__(self, runner_factory: Any, child_events: dict[str, list[dict[str, Any]]]):
        self.runner_factory, self.child_events = runner_factory, child_events
    async def execute(self, task: Any, **_: Any):
        task_id=getattr(task,"task_id",getattr(task,"id", ""))
        events=self.child_events.get(task_id)
        if not events: raise ReplayDivergence("recorded child", task_id, -1)
        from .rah import as_child_task, ChildResult
        child=as_child_task(task); runner=self.runner_factory(child)
        runner.backend=ReplayBackend(events); runner.gateway=ReplayGateway(events)
        state=await runner.run(child.goal, getattr(child,"mode",None) or __import__("lynx_harness.models",fromlist=["Mode"]).Mode.THINK, __import__("lynx_harness.models",fromlist=["ComputeBudget"]).ComputeBudget(llm_tokens=child.budget.llm_tokens,steps=child.budget.steps,tool_calls=child.budget.tool_calls,max_wall_time_s=child.budget.max_wall_time_s))
        return ChildResult(state.status,state.answer,(),state.steps,len(state.observations),0,task_id)

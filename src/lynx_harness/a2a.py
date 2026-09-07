from __future__ import annotations

from dataclasses import dataclass, asdict, field
import asyncio
import json
import re
from typing import Any
from urllib.request import Request
from urllib.parse import urlparse

from .tools import ToolError, _open_public
from .security import redact

_ID=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
def _id(value: str, label: str) -> str:
    if not isinstance(value,str) or not _ID.fullmatch(value): raise ValueError(f"invalid {label}")
    return value
def _refs(refs: tuple[str,...]) -> tuple[str,...]:
    if not isinstance(refs, (tuple, list)) or len(refs) > 100:
        raise ValueError("inputs must be bounded artifact/knowledge refs")
    for ref in refs:
        if not isinstance(ref, str) or len(ref) > 500 or "\n" in ref or "\r" in ref:
            raise ValueError("inputs must be bounded artifact/knowledge refs")
        if ref.startswith("artifact://"):
            if not (re.fullmatch(r"artifact://[0-9a-f]{64}\.[A-Za-z0-9_]{1,10}", ref) or re.fullmatch(r"artifact://repo/[0-9]{1,40}", ref)):
                raise ValueError("invalid artifact reference")
        elif ref.startswith("okf://"):
            if not re.fullmatch(r"okf://[A-Za-z0-9_.:-]{1,128}", ref):
                raise ValueError("invalid knowledge reference")
        else:
            raise ValueError("inputs must be bounded artifact/knowledge refs")
    if len(set(refs)) != len(refs):
        raise ValueError("inputs must be bounded artifact/knowledge refs")
    return tuple(refs)


@dataclass(frozen=True)
class AgentDescriptor:
    name: str; description: str; skills: tuple[str,...]=(); input_modes: tuple[str,...]=("text",); output_modes: tuple[str,...]=("text",)
    def __post_init__(self):
        _id(self.name,"agent name")
        if not isinstance(self.description, str) or not self.description.strip() or len(self.description) > 20_000:
            raise ValueError("invalid agent description")
        for field_name in ("skills", "input_modes", "output_modes"):
            values = getattr(self, field_name)
            if isinstance(values, str):
                raise ValueError(f"{field_name} must be a sequence")
            try:
                values = tuple(values)
            except TypeError as exc:
                raise ValueError(f"{field_name} must be a sequence") from exc
            if len(values) > 100 or any(not isinstance(value, str) or not value.strip() or len(value) > 200 for value in values):
                raise ValueError(f"invalid {field_name}")
            object.__setattr__(self, field_name, values)

@dataclass(frozen=True)
class AgentTask:
    id: str; context_id: str; goal: str; inputs: tuple[str,...]=(); budget: dict[str,Any]=field(default_factory=dict); expected_output: dict[str,Any]=field(default_factory=dict)
    security: dict[str, Any]=field(default_factory=dict); parent_task_id: str | None=None; branch_id: str="main"; depth: int=0; max_depth: int=2; max_children: int=8
    def __post_init__(self):
        _id(self.id,"task id"); _id(self.context_id,"context id"); _refs(self.inputs)
        if not isinstance(self.goal, str) or not self.goal.strip() or len(self.goal)>20_000: raise ValueError("invalid task goal")
        if not isinstance(self.budget, dict) or not isinstance(self.expected_output, dict) or not isinstance(self.security, dict): raise ValueError("invalid task metadata")
        for label, value in (("budget", self.budget), ("expected output", self.expected_output), ("security", self.security)):
            try:
                encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid {label} metadata") from exc
            if len(encoded.encode("utf-8")) > 20_000:
                raise ValueError("task metadata too large")
        if not isinstance(self.branch_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.branch_id) or type(self.depth) is not int or type(self.max_depth) is not int or not 0 <= self.depth <= self.max_depth <= 32 or type(self.max_children) is not int or not 0 <= self.max_children <= 100: raise ValueError("invalid task lineage")

@dataclass(frozen=True)
class AgentArtifact:
    id: str; mime_type: str; content: str = ""; uri: str | None = None
    def __post_init__(self):
        _id(self.id,"artifact id")
        if not isinstance(self.mime_type, str) or not self.mime_type.strip() or len(self.mime_type) > 200:
            raise ValueError("invalid artifact mime type")
        if not isinstance(self.content, str) or len(self.content) > 100_000:
            raise ValueError("invalid artifact content")
        if self.uri is not None and (not isinstance(self.uri, str) or len(self.uri) > 2_000 or "\r" in self.uri or "\n" in self.uri or not self.uri.startswith(("artifact://", "https://"))):
            raise ValueError("invalid artifact URI")
        if self.uri is not None and self.uri.startswith("artifact://") and not re.fullmatch(r"artifact://[0-9a-f]{64}\.[A-Za-z0-9_]{1,10}", self.uri):
            raise ValueError("artifact URI must be canonical and content-addressed")
        if self.uri is not None and self.uri.startswith("https://"):
            parsed = urlparse(self.uri)
            if not parsed.hostname or parsed.username is not None or parsed.password is not None:
                raise ValueError("artifact URI must be credential-free")

@dataclass(frozen=True)
class AgentResult:
    status: str; artifacts: tuple[AgentArtifact,...]=(); message: str = ""
    def __post_init__(self):
        if self.status not in {"submitted","working","completed","failed","canceled"}: raise ValueError("invalid result status")
        if not isinstance(self.artifacts, (tuple, list)) or len(self.artifacts)>100 or any(not isinstance(item, AgentArtifact) for item in self.artifacts): raise ValueError("invalid result artifacts")
        if not isinstance(self.message, str) or len(self.message)>20_000: raise ValueError("result too large")

def bounded_json(value: Any, limit: int = 500_000) -> str:
    text=json.dumps(redact(value),ensure_ascii=False,separators=(",",":"),default=lambda obj: asdict(obj) if hasattr(obj,"__dataclass_fields__") else str(obj))
    if len(text.encode())>limit: raise ValueError("A2A payload too large")
    return text



def validate_remote_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("status") not in {"submitted", "working", "completed", "failed", "canceled"}: raise ValueError("invalid remote A2A result")
    artifacts=payload.get("artifacts", [])
    if not isinstance(artifacts,list) or len(artifacts)>100: raise ValueError("invalid remote artifacts")
    checked=[]
    for item in artifacts:
        if not isinstance(item,dict): raise ValueError("invalid remote artifact")
        if not isinstance(item.get("id"),str) or not isinstance(item.get("mime_type","application/octet-stream"),str) or not isinstance(item.get("content",""),str): raise ValueError("invalid remote artifact fields")
        if len(item.get("content",""))>100_000: raise ValueError("remote artifact too large")
        if item.get("uri") is not None and not isinstance(item.get("uri"), str): raise ValueError("invalid remote artifact URI")
        artifact=AgentArtifact(item["id"],item.get("mime_type","application/octet-stream"),item.get("content",""),item.get("uri"))
        checked.append(asdict(artifact))
    if not isinstance(payload.get("message",""),str) or len(payload.get("message",""))>20_000: raise ValueError("remote message too large")
    message=payload.get("message","")
    result={"status":payload["status"],"artifacts":checked,"message":message,"trust":"untrusted","can_authorize":False}
    # Enforce the same aggregate output bound as the local server and redact
    # before returning data to callers that may persist it.
    return json.loads(bounded_json(result, 500_000))

class A2AClient:
    """Opt-in A2A task client; remote endpoints require HTTPS and explicit host allowlisting."""
    def __init__(self, endpoint: str, allow_hosts: set[str], timeout_s: float=20, auth_token: str | None = None):
        from urllib.parse import urlparse
        if not isinstance(endpoint, str) or len(endpoint) > 2_000: raise ValueError("A2A endpoint is invalid")
        parsed=urlparse(endpoint)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.hostname.lower() not in {h.lower().rstrip(".") for h in allow_hosts}): raise ValueError("A2A endpoint must be HTTPS, credential-free, and allowlisted")
        if not 1 <= timeout_s <= 300: raise ValueError("A2A timeout out of bounds")
        if auth_token is not None and (not isinstance(auth_token, str) or not auth_token or len(auth_token) > 4_096 or "\r" in auth_token or "\n" in auth_token): raise ValueError("A2A auth token is invalid")
        self.endpoint=endpoint; self.allow_hosts={h.lower().rstrip(".") for h in allow_hosts}; self.timeout_s=timeout_s; self.auth_token=auth_token
    async def submit(self, task: AgentTask) -> dict[str,Any]:
        body=bounded_json({"task":asdict(task)}).encode(); headers={"Content-Type":"application/json","Accept":"application/json"}
        if self.auth_token: headers["Authorization"]="Bearer " + self.auth_token
        request=Request(self.endpoint,data=body,headers=headers,method="POST")
        def call():
            response, text = _open_public(request, self.allow_hosts, 500_000)
            try:
                return json.loads(text)
            finally:
                # urllib responses hold sockets/file descriptors until closed;
                # relying on GC allows repeated bounded calls to exhaust them.
                response.close()
        try:
            response=await asyncio.wait_for(asyncio.to_thread(call),timeout=self.timeout_s)
            return validate_remote_result(response)
        except asyncio.TimeoutError as exc: raise ToolError("A2A request timed out") from exc
        except (OSError, ToolError) as exc: raise ToolError("A2A request failed") from exc
        except (ValueError, KeyError, TypeError) as exc: raise ToolError("invalid remote A2A result") from exc


def _safe_result_payload(result: AgentResult) -> dict[str, Any]:
    """Redact and bound result data before it crosses the A2A boundary."""
    # AgentResult permits individually bounded artifacts, but their aggregate
    # can still be large.  The transport limit is deliberately lower than the
    # in-process model limit and applies consistently to every response.
    return json.loads(bounded_json(asdict(result), 500_000))


def create_a2a_app(descriptor: AgentDescriptor, executor):
    """Create an optional FastAPI A2A surface around a trusted AgentExecutor callback."""
    try:
        from fastapi import FastAPI, HTTPException, Header
    except ImportError as exc: raise RuntimeError("A2A API requires: uv sync --extra api") from exc
    app=FastAPI(title=descriptor.name,version="0.1.0")
    from starlette.responses import JSONResponse
    @app.middleware("http")
    async def request_limit(request, call_next):
        try: length=int(request.headers.get("content-length", "0"))
        except ValueError: return JSONResponse({"detail":"invalid content length"}, status_code=400)
        if length > 500_000: return JSONResponse({"detail":"request too large"}, status_code=413)
        received = 0
        original_receive = request._receive
        async def limited_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if received > 500_000:
                raise ValueError("request body too large")
            return message
        request._receive = limited_receive
        try:
            return await call_next(request)
        except ValueError as exc:
            if str(exc) == "request body too large": return JSONResponse({"detail":"request too large"}, status_code=413)
            raise
    task_store: dict[str, AgentResult] = {}
    task_jobs: dict[str, asyncio.Task] = {}
    # A caller can submit many valid tasks; bound executor fan-out so an
    # otherwise trusted callback cannot be used to exhaust local resources.
    task_semaphore = asyncio.Semaphore(8)
    def authorized(authorization: str | None) -> bool:
        configured_token=__import__("os").environ.get("LYNX_A2A_TOKEN")
        import secrets
        expected = "Bearer " + configured_token if configured_token else ""
        return bool(configured_token) and isinstance(authorization, str) and secrets.compare_digest(authorization, expected)
    @app.get("/.well-known/agent-card.json")
    async def card(): return json.loads(bounded_json(asdict(descriptor), 100_000))
    @app.post("/tasks")
    async def submit(payload: dict[str,Any], authorization: str | None = Header(default=None)):
        if not authorized(authorization):
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="unauthorized")
        task=None
        try:
            if not isinstance(payload, dict): raise ValueError("task must be an object")
            allowed_fields = {"id", "context_id", "goal", "inputs", "budget", "expected_output", "security", "parent_task_id", "branch_id", "depth", "max_depth", "max_children"}
            if set(payload) - allowed_fields: raise ValueError("unknown task fields")
            task=AgentTask(payload["id"], payload.get("context_id", payload["id"]), payload["goal"], tuple(payload.get("inputs",())), payload.get("budget",{}), payload.get("expected_output",{}), payload.get("security",{}), payload.get("parent_task_id"), payload.get("branch_id","main"), payload.get("depth",0), payload.get("max_depth",2), payload.get("max_children",8))
            # Validate local artifact provenance before the envelope enters the
            # durable task map or trusted executor.
            from_a2a_task(task)
            if task.id in task_store: raise HTTPException(status_code=409, detail="task already exists")
            if len(task_store) >= 1_000: raise HTTPException(status_code=429, detail="task store is full")
            task_store[task.id]=AgentResult("working"); task_jobs[task.id]=asyncio.current_task()
            async with task_semaphore:
                result=await executor(task)
            if not isinstance(result,AgentResult): raise TypeError("executor returned invalid result")
            # Cancellation is authoritative even if a callback catches and
            # suppresses asyncio.CancelledError.
            if task_store[task.id].status == "canceled":
                return _safe_result_payload(task_store[task.id])
            try:
                payload_result = _safe_result_payload(result)
            except (TypeError, ValueError) as exc:
                task_store[task.id] = AgentResult("failed", message="result exceeds output limit")
                return _safe_result_payload(task_store[task.id])
            task_store[task.id]=result
            return payload_result
        except asyncio.CancelledError:
            if task is None: raise HTTPException(status_code=499, detail="task canceled")
            task_store[task.id]=AgentResult("canceled"); return _safe_result_payload(task_store[task.id])
        except (KeyError,TypeError,ValueError) as exc: raise HTTPException(status_code=400,detail="invalid task") from exc
        except Exception as exc: raise HTTPException(status_code=503,detail="task execution failed") from exc
        finally:
            task_jobs.pop(task.id, None) if task is not None else None
    @app.get("/tasks/{task_id}")
    async def task_status(task_id: str, authorization: str | None = Header(default=None)):
        if not authorized(authorization): raise HTTPException(status_code=401, detail="unauthorized")
        if task_id not in task_store: raise HTTPException(status_code=404, detail="task not found")
        return _safe_result_payload(task_store[task_id])
    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, authorization: str | None = Header(default=None)):
        if not authorized(authorization): raise HTTPException(status_code=401, detail="unauthorized")
        if task_id not in task_store: raise HTTPException(status_code=404, detail="task not found")
        job=task_jobs.get(task_id)
        task_store[task_id]=AgentResult("canceled")
        if job and not job.done(): job.cancel()
        return _safe_result_payload(task_store[task_id])
    return app


def to_a2a_task(task: Any) -> AgentTask:
    """Map a local task contract to the A2A envelope using artifact refs only."""
    from .models import TaskContract
    if not isinstance(task, TaskContract): raise TypeError("expected local TaskContract")
    data = task.to_dict()
    return AgentTask(task.id, task.id, task.goal, tuple(task.input_artifacts), data.get("budget", {}), data.get("expected_output", {}), data.get("security", {}), task.parent_task_id, task.branch_id, task.depth, task.max_depth, task.max_children)


def from_a2a_task(task: AgentTask) -> Any:
    from .models import AgentTask as LocalAgentTask
    # A2A has a broader interoperability reference syntax, but local Lynx
    # execution accepts only content-addressed artifact refs.  Reject before
    # conversion so untrusted envelopes cannot reach the executor.
    if any(not re.fullmatch(r"artifact://[0-9a-f]{64}\.[A-Za-z0-9_]{1,10}", ref) for ref in task.inputs if ref.startswith("artifact://")):
        raise ValueError("A2A task contains a non-canonical local artifact reference")
    return LocalAgentTask.from_dict({"id":task.id,"goal":task.goal,"input_artifacts":list(task.inputs),"budget":task.budget,"expected_output":task.expected_output,"security":task.security,"parent_task_id":task.parent_task_id,"branch_id":task.branch_id,"depth":task.depth,"max_depth":task.max_depth,"max_children":task.max_children})


class A2AReplayStub:
    """Deterministic interoperability fixture: no network or remote code is used."""
    def __init__(self, descriptor: AgentDescriptor, executor): self.descriptor,self.executor,self.tasks=descriptor,executor,{}
    async def card(self) -> AgentDescriptor: return self.descriptor
    async def submit(self, task: AgentTask) -> AgentResult:
        self.tasks[task.id]=AgentResult("working");
        try:
            result=await self.executor(task)
            if not isinstance(result,AgentResult): raise TypeError("invalid stub result")
        except Exception: result=AgentResult("failed",message="execution failed")
        self.tasks[task.id]=result; return result
    async def status(self, task_id: str) -> AgentResult:
        if task_id not in self.tasks: raise KeyError(task_id)
        return self.tasks[task_id]

from dataclasses import asdict
import os
import secrets
from typing import Any

from .cli import build_runner
from .config import Settings
from .models import Mode, ComputeBudget, TaskContract, ExpectedOutput, Postcondition, SecurityPolicy, RiskLevel
from .security import redact
from .model_profiles import CANONICAL_MODEL_FILENAME


def create_app(settings: Settings | None = None, integration_setup=None):
    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as exc:
        raise RuntimeError("API optional dependency missing; run: uv sync --extra api") from exc
    from pydantic import BaseModel, Field
    try:
        from pydantic import ConfigDict
    except ImportError:  # pydantic v1 compatibility for the optional extra
        ConfigDict = None
    from starlette.responses import JSONResponse

    class RunRequest(BaseModel):
        if ConfigDict is not None:
            model_config = ConfigDict(extra="forbid")
        else:
            class Config:
                extra = "forbid"
        task: str = Field(min_length=1, max_length=20_000)
        mode: Mode = Mode.THINK

    app = FastAPI(title="Lynx Harness", version="0.1.0")
    @app.middleware("http")
    async def request_limit(request, call_next):
        # Pydantic field limits do not protect JSON parsing from huge unknown
        # fields. Bound the raw body before FastAPI materializes it.
        try: length = int(request.headers.get("content-length", "0"))
        except ValueError: return JSONResponse({"detail": "invalid content length"}, status_code=400)
        if length < 0: return JSONResponse({"detail": "invalid content length"}, status_code=400)
        if length > 500_000: return JSONResponse({"detail": "request too large"}, status_code=413)
        received = 0
        original_receive = request._receive
        async def limited_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if received > 500_000: raise ValueError("request body too large")
            return message
        request._receive = limited_receive
        try:
            return await call_next(request)
        except ValueError as exc:
            if str(exc) == "request body too large": return JSONResponse({"detail": "request too large"}, status_code=413)
            raise

    configured = settings or Settings.from_env()
    asyncio = __import__("asyncio")
    admission_lock = asyncio.Lock()
    active_count = 0

    @app.get("/health")
    async def health() -> dict[str, Any]:
        runner = build_runner(configured, integration_setup=integration_setup)
        backend = runner.backend
        ready = await backend.health() if hasattr(backend, "health") else False
        return {"ok": ready, "model": CANONICAL_MODEL_FILENAME, "backend": configured.backend}

    async def authenticate(authorization: str | None) -> None:
        expected = os.getenv("LYNX_API_TOKEN")
        if not expected:
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        provided = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.post("/runs")
    async def run(request: RunRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await authenticate(authorization)
        nonlocal active_count
        async with admission_lock:
            if active_count >= 4: raise HTTPException(status_code=429, detail="too many active runs")
            active_count += 1
        try:
            runner = build_runner(configured, integration_setup=integration_setup)
            try:
                # Discover opt-in integrations before snapshotting the immutable
                # contract allowlist; otherwise MCP tools are registered too late
                # and can never be selected by a contract-backed API run.
                if runner.integrations is not None:
                    await runner.integrations.start()
                mode_budget = ComputeBudget.for_mode(request.mode)
                contract = TaskContract(id="api-" + __import__("uuid").uuid4().hex[:24], goal=request.task, mode=request.mode, allowed_tools=tuple(spec.name for spec in runner.registry.specs()), expected_output=ExpectedOutput(max_chars=20_000), postconditions=(Postcondition("output_bounded"),), budget=mode_budget, security=SecurityPolicy(max_risk=RiskLevel.EXTERNAL_MUTATION if os.getenv("LYNX_ALLOW_EXTERNAL") == "1" else RiskLevel.LOCAL_MUTATION, confirmation_required=os.getenv("LYNX_ALLOW_EXTERNAL") == "1"))
                state = await runner.run(request.task, request.mode, mode_budget, contract=contract)
            except Exception as exc:
                # Do not expose backend stack traces or request contents.
                raise HTTPException(status_code=503, detail="agent execution failed") from exc
        finally:
            async with admission_lock: active_count -= 1
        if state.status not in {"success", "verified"} or not state.answer:
            raise HTTPException(status_code=422, detail={"status": state.status, "notes": redact(state.notes)[:20], "run_id": state.run_id})
        response=redact({"run_id": state.run_id, "answer": state.answer[:50_000], "observations": [asdict(item) for item in state.observations[:100]]})
        if len(__import__("json").dumps(response, default=str).encode()) > 1_000_000: raise HTTPException(status_code=500, detail="response exceeds configured limit")
        return response

    return app

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlparse

from .tools import ToolError
from .security import sanitized_subprocess_env


class PlaywrightCli:
    """Opt-in, process-isolated Playwright CLI seam with bounded output."""
    def __init__(self, workspace: Path, executable: str = "playwright-cli", allow_hosts: set[str] | None = None):
        self.workspace=workspace.resolve(); self.executable=executable; self.allow_hosts={str(host).lower().rstrip(".") for host in (allow_hosts or set())}
        if not self.workspace.is_dir(): raise ToolError("browser workspace must be a directory")
        if (not isinstance(executable, str) or not executable or "\x00" in executable or executable.startswith("-")):
            raise ToolError("invalid Playwright CLI executable")
        if not self.allow_hosts: raise ToolError("browser domain allowlist must not be empty")
        if not shutil.which(executable) and not Path(executable).is_file(): raise ToolError("Playwright CLI is not installed")
    async def run(self, operation: str, *args: str, timeout_s: float = 20) -> dict[str, Any]:
        if not isinstance(timeout_s, (int,float)) or isinstance(timeout_s, bool) or not 0.1 <= timeout_s <= 300: raise ToolError("browser timeout is out of bounds")
        if operation not in {"open", "snapshot", "click", "fill"}: raise ToolError("unsupported browser operation")
        # The generic CLI cannot enforce the destination of redirects or
        # subresource requests.  Running ``open`` would therefore turn an
        # allowlisted URL into an SSRF primitive. Keep navigation disabled until
        # a host-provided browser-level egress interceptor is wired in.
        if operation in {"open", "click", "fill"}:
            raise ToolError("browser navigation requires request interception")
        if len(args) > 4 or any(not isinstance(arg, str) or "\x00" in arg or len(arg)>2_000 or arg.startswith("-") for arg in args): raise ToolError("invalid browser argument")
        for arg in args:
            parsed = urlparse(arg)
            if parsed.scheme in {"http", "https"}:
                # Snapshot must operate on an already provisioned browser page;
                # accepting a URL here would reintroduce unguarded navigation.
                raise ToolError("browser URL navigation requires request interception")
        try:
            proc=await asyncio.create_subprocess_exec(self.executable,operation,*args,cwd=self.workspace,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True,env=sanitized_subprocess_env())
        except (OSError, ValueError) as exc:
            raise ToolError("failed to start Playwright CLI") from exc

        async def read_bounded(stream):
            chunks=[]; total=0
            while True:
                chunk=await stream.read(min(16_384, 100_001-total))
                if not chunk: return b"".join(chunks), False
                chunks.append(chunk); total += len(chunk)
                if total > 100_000: return b"", True

        async def stop_process():
            if proc.returncode is not None: return
            try: os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError): proc.kill()
            await proc.wait()
        stdout_task=asyncio.create_task(read_bounded(proc.stdout))
        stderr_task=asyncio.create_task(read_bounded(proc.stderr))
        wait_task=asyncio.create_task(proc.wait())
        try:
            await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task, wait_task), timeout=timeout_s)
        except asyncio.TimeoutError:
            await stop_process()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise ToolError("browser operation timed out")
        except asyncio.CancelledError:
            await stop_process()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        output_result=stdout_task.result(); error_result=stderr_task.result()
        if output_result[1] or error_result[1]:
            await stop_process()
            raise ToolError("browser output exceeds configured limit")
        text=(output_result[0]+error_result[0]).decode("utf-8","replace")[:100_000]
        return {"returncode":proc.returncode,"output":text}



def register_playwright_tools(registry, browser: PlaywrightCli) -> None:
    from .models import RiskLevel, ToolSpec
    async def invoke(args): return await browser.run(args["operation"], *(args.get("args") or []))
    registry.register(ToolSpec("browser.action","Perform an explicitly permitted Playwright CLI action", {"type":"object","required":["operation"],"properties":{"operation":{"type":"string","enum":["open","snapshot","click","fill"]},"args":{"type":"array","items":{"type":"string","maxLength":2000},"maxItems":4}}}, risk=RiskLevel.EXTERNAL_MUTATION, family="browser", origin="playwright", trust="untrusted"), invoke)

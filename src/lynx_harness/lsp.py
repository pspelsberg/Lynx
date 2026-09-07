from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

from .tools import ToolError
from .security import sanitized_subprocess_env


def _strict_json_loads(data: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> Any: raise ValueError(f"non-finite JSON constant: {value}")
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result: raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(data.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict): raise ValueError("LSP response must be an object")
    return value


class LspClient:
    """Headless JSON-RPC LSP sensor; server executable is trusted configuration, not model input."""
    def __init__(self, command: tuple[str, ...], workspace: Path):
        if (not isinstance(command, (tuple, list)) or not command or len(command) > 32
                or any(not isinstance(part, str) or not part or "\x00" in part or part.startswith("-") or len(part) > 500 for part in command)):
            raise ValueError("invalid LSP command")
        self.command=tuple(command); self.workspace=workspace.resolve()
        if not self.workspace.is_dir(): raise ValueError("LSP workspace must be a directory")
        self.process=None; self.request_id=0; self._request_lock=asyncio.Lock(); self._stdout_buffer=bytearray()
        self.max_message_bytes = 2_000_000
    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        self.process=await asyncio.create_subprocess_exec(*self.command,cwd=self.workspace,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,start_new_session=True,env=sanitized_subprocess_env())
        try:
            await self.request("initialize", {"processId":None,"rootUri":self.workspace.as_uri(),"capabilities":{}})
            await self.notify("initialized", {})
        except BaseException:
            # Startup cancellation/failure must clean up the spawned server;
            # asyncio.CancelledError derives from BaseException.
            await self.close(); raise
    async def _write(self, message: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin: raise ToolError("LSP is not started")
        try:
            body=json.dumps(message,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ToolError("LSP message is not valid JSON") from exc
        if len(body) > self.max_message_bytes: raise ToolError("LSP message is too large")
        self.process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode()+body); await self.process.stdin.drain()
    async def notify(self, method: str, params: dict[str,Any]) -> None:
        await self._write({"jsonrpc":"2.0","method":method,"params":params})
    async def _readline_bounded(self, limit: int = 8_192) -> bytes:
        if not self.process or not self.process.stdout: raise ToolError("LSP is not started")
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                if newline + 1 > limit:
                    await self.close(); raise ToolError("LSP header is too large")
                line = bytes(self._stdout_buffer[:newline + 1])
                del self._stdout_buffer[:newline + 1]
                return line
            if len(self._stdout_buffer) >= limit:
                await self.close(); raise ToolError("LSP header is too large")
            chunk = await self.process.stdout.read(min(4096, limit - len(self._stdout_buffer)))
            if not chunk:
                if self._stdout_buffer:
                    line = bytes(self._stdout_buffer); self._stdout_buffer.clear(); return line
                return b""
            self._stdout_buffer.extend(chunk)

    async def _read_body(self, length: int) -> bytes:
        """Consume a frame body, including bytes buffered past its headers."""
        while len(self._stdout_buffer) < length:
            chunk = await self.process.stdout.read(length - len(self._stdout_buffer))
            if not chunk: raise asyncio.IncompleteReadError(bytes(self._stdout_buffer), length)
            self._stdout_buffer.extend(chunk)
        body = bytes(self._stdout_buffer[:length]); del self._stdout_buffer[:length]
        return body

    async def _read(self) -> dict[str, Any]:
        if not self.process or not self.process.stdout: raise ToolError("LSP is not started")
        length: int | None = None
        while True:
            try: line = await asyncio.wait_for(self._readline_bounded(), timeout=15)
            except asyncio.TimeoutError:
                await self.close(); raise ToolError("LSP response timed out")
            except asyncio.CancelledError:
                await self.close(); raise
            if not line: raise ToolError("LSP server closed")
            if line in (b"\r\n",b"\n"): break
            try: decoded = line.decode("ascii")
            except UnicodeDecodeError as exc: raise ToolError("invalid LSP header") from exc
            key, separator, value = decoded.rstrip("\r\n").partition(":")
            if not separator:
                raise ToolError("invalid LSP header")
            normalized_key = key.strip().lower()
            if normalized_key == "content-length":
                if length is not None:
                    raise ToolError("duplicate LSP content length")
                try:
                    length = int(value.strip(), 10)
                except ValueError as exc:
                    raise ToolError("invalid LSP content length") from exc
                if length <= 0 or length > self.max_message_bytes:
                    raise ToolError("invalid LSP message length")
            elif normalized_key == "content-type":
                # Content-Type is a standard optional LSP framing header. It is
                # metadata only; never interpret it as a second frame length.
                if not value.strip() or len(value) > 1_000:
                    raise ToolError("invalid LSP content type")
            else:
                raise ToolError("invalid LSP header")
        if length is None or not 0 < length <= self.max_message_bytes: raise ToolError("invalid LSP message length")
        try:
            body = await asyncio.wait_for(self._read_body(length), timeout=15)
            return _strict_json_loads(body)
        except asyncio.TimeoutError:
            await self.close(); raise ToolError("LSP response timed out")
        except asyncio.IncompleteReadError as exc: raise ToolError("LSP response truncated") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc: raise ToolError("LSP response is not valid JSON") from exc

    async def _request_unlocked(self, method: str, params: dict[str,Any]) -> Any:
        if not self.process or not self.process.stdin or not self.process.stdout: raise ToolError("LSP is not started")
        self.request_id+=1; request_id=self.request_id; await self._write({"jsonrpc":"2.0","id":request_id,"method":method,"params":params})
        while True:
            response=await self._read()
            # Servers may emit diagnostics/notifications between responses.
            if response.get("id") != request_id: continue
            if "error" in response: raise ToolError(str(response["error"]))
            return response.get("result")

    async def request(self, method: str, params: dict[str,Any]) -> Any:
        async with self._request_lock:
            try:
                return await self._request_unlocked(method, params)
            except asyncio.CancelledError:
                await self.close(); raise
    def _safe_uri(self, uri: str) -> str:
        parsed=urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}: raise ToolError("LSP URI must use a local file scheme")
        path=Path(unquote(parsed.path)).resolve()
        if path != self.workspace and self.workspace not in path.parents: raise ToolError("LSP URI escapes workspace")
        return path.as_uri()
    async def diagnostics(self, uri: str) -> Any: return await self.request("textDocument/diagnostic", {"textDocument":{"uri":self._safe_uri(uri)}})
    async def close(self) -> None:
        if self.process:
            process = self.process
            try:
                if process.returncode is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError): process.terminate()
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                try: os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError): process.kill()
                await process.wait()
            self.process=None; self._stdout_buffer.clear()


def register_lsp_tools(registry, client: LspClient) -> None:
    from .models import RiskLevel, ToolSpec
    async def diagnostics(args): return await client.diagnostics(args["uri"])
    registry.register(ToolSpec("lsp.diagnostics","Read diagnostics for a file inside the configured workspace", {"type":"object","required":["uri"],"properties":{"uri":{"type":"string","maxLength":4000}}}, risk=RiskLevel.READ_ONLY, family="lsp", origin="lsp", trust="untrusted"), diagnostics)

"""Optional MCP stdio bridge. It is opt-in and subprocess-scoped."""
from __future__ import annotations
import asyncio
import json
import os
import signal
from dataclasses import dataclass
from typing import Any
from .models import RiskLevel, ToolSpec
from .tools import ToolError, ToolRegistry
from .security import sanitized_subprocess_env


def _strict_json_loads(data: bytes) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(data.decode("utf-8"), parse_constant=reject_constant,
                      object_pairs_hook=reject_duplicates)

@dataclass(frozen=True)
class McpServerConfig:
    command: tuple[str, ...]
    name: str
    protocol_versions: tuple[str, ...] = ("2025-06-18", "2025-03-26")
    allowed_tools: tuple[str, ...] = ()
    max_risk: RiskLevel = RiskLevel.EXTERNAL_MUTATION
    def __post_init__(self):
        if (not isinstance(self.command, (tuple, list)) or not self.command or len(self.command)>32
                or any(not isinstance(x,str) or not x or "\x00" in x or x.startswith("-") or len(x)>500 for x in self.command)):
            raise ValueError("MCP command is invalid")
        object.__setattr__(self, "command", tuple(self.command))
        if (not isinstance(self.name, str) or not self.name or len(self.name)>100
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in self.name)):
            raise ValueError("MCP server name is invalid")
        versions = self.protocol_versions
        if (not isinstance(versions, (tuple, list)) or not versions or len(versions) > 16
                or any(not isinstance(v, str) or not v or "\x00" in v or len(v) > 100 for v in versions)):
            raise ValueError("MCP protocol versions are invalid")
        object.__setattr__(self, "protocol_versions", tuple(versions))
        tools = self.allowed_tools
        if (not isinstance(tools, (tuple, list)) or len(tools) > 100
                or any(not isinstance(v, str) or not v or "\x00" in v or len(v) > 200 for v in tools)
                or len(set(tools)) != len(tools)):
            raise ValueError("MCP allowed tools are invalid")
        object.__setattr__(self, "allowed_tools", tuple(tools))
        try: object.__setattr__(self,"max_risk",self.max_risk if isinstance(self.max_risk,RiskLevel) else RiskLevel[self.max_risk.upper()])
        except (KeyError,AttributeError,TypeError) as exc: raise ValueError("MCP risk is invalid") from exc

class McpStdioClient:
    """Newline-delimited JSON-RPC MCP client with bounded messages and negotiation."""
    def __init__(self, config: McpServerConfig, timeout_s: float = 15, max_message_bytes: int = 2_000_000):
        if not config.command or any(not item or "\x00" in item or item.startswith("-") for item in config.command): raise ValueError("MCP command must be explicit")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not 1 <= timeout_s <= 120 or type(max_message_bytes) is not int
                or not 1_000 <= max_message_bytes <= 10_000_000): raise ValueError("MCP limits out of bounds")
        self.config,self.timeout_s,self.max_message_bytes=config,timeout_s,max_message_bytes; self.process=None; self._request_id=0; self.protocol_version=None; self.capabilities={}; self._request_lock=asyncio.Lock(); self._stdout_buffer=bytearray()
    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        self.process=await asyncio.create_subprocess_exec(*self.config.command,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,start_new_session=True,env=sanitized_subprocess_env())
        try:
            result=await self.request("initialize", {"protocolVersion":self.config.protocol_versions[0],"capabilities":{},"clientInfo":{"name":"lynx-harness","version":"0.1.0"}})
            version=result.get("protocolVersion") if isinstance(result,dict) else None
            if version not in self.config.protocol_versions: raise ToolError("MCP protocol version was not negotiated")
            self.protocol_version=version; self.capabilities=dict(result.get("capabilities", {})); await self.notify("notifications/initialized", {})
        except BaseException:
            # Cancellation during startup must not orphan the just-created
            # server process (CancelledError is a BaseException).
            await self.close(); raise
    async def _send(self, message: dict[str,Any]) -> None:
        if not self.process or not self.process.stdin: raise ToolError("MCP client is not started")
        payload=(json.dumps(message,separators=(",",":"),ensure_ascii=False)+"\n").encode("utf-8")
        if len(payload)>self.max_message_bytes: raise ToolError("MCP request too large")
        self.process.stdin.write(payload); await self.process.stdin.drain()
    async def notify(self, method: str, params: dict[str,Any] | None = None) -> None: await self._send({"jsonrpc":"2.0","method":method,"params":params or {}})
    async def _readline_bounded(self) -> bytes:
        """Read one JSONL frame without allowing an attacker-sized line buffer."""
        if not self.process or not self.process.stdout:
            raise ToolError("MCP client is not started")
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                if newline + 1 > self.max_message_bytes:
                    await self.close()
                    raise ToolError("MCP response too large")
                line = bytes(self._stdout_buffer[:newline + 1])
                del self._stdout_buffer[:newline + 1]
                return line
            if len(self._stdout_buffer) > self.max_message_bytes:
                await self.close()
                raise ToolError("MCP response too large")
            chunk = await self.process.stdout.read(min(4096, self.max_message_bytes + 1 - len(self._stdout_buffer)))
            if not chunk:
                if self._stdout_buffer:
                    line = bytes(self._stdout_buffer)
                    self._stdout_buffer.clear()
                    return line
                return b""
            self._stdout_buffer.extend(chunk)

    async def _request_unlocked(self, method: str, params: dict[str,Any] | None = None) -> dict[str,Any]:
        if not self.process or not self.process.stdout: raise ToolError("MCP client is not started")
        self._request_id+=1; request_id=self._request_id; await self._send({"jsonrpc":"2.0","id":request_id,"method":method,"params":params or {}})
        while True:
            try:
                line = await asyncio.wait_for(self._readline_bounded(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                await self.close(); raise ToolError("MCP request timed out")
            except asyncio.CancelledError:
                await self.close(); raise
            if not line: raise ToolError("MCP server closed stdout")
            try:
                response = _strict_json_loads(line)
            except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise ToolError("MCP response is not valid JSON") from exc
            if not isinstance(response, dict): raise ToolError("MCP response must be an object")
            if "id" not in response: continue
            if response.get("id") != request_id: raise ToolError("MCP response id mismatch")
            if "error" in response: raise ToolError(str(response["error"]))
            result=response.get("result",{})
            if not isinstance(result,dict): raise ToolError("MCP result must be an object")
            return result
    async def request(self, method: str, params: dict[str,Any] | None = None) -> dict[str,Any]:
        # MCP stdio is a single ordered stream; serialize request/response
        # pairs so concurrent gateway calls cannot consume each other's IDs.
        async with self._request_lock:
            try:
                return await self._request_unlocked(method, params)
            except asyncio.CancelledError:
                await self.close(); raise

    def audit_record(self) -> dict[str, Any]:
        return {"server": self.config.name, "command": list(self.config.command), "protocol_version": self.protocol_version, "capabilities": dict(self.capabilities), "allowed_tools": list(self.config.allowed_tools), "max_risk": self.config.max_risk.name, "timeout_s": self.timeout_s, "max_message_bytes": self.max_message_bytes}

    async def close(self) -> None:
        if self.process:
            process = self.process
            try:
                if process.returncode is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError): process.terminate()
                await asyncio.wait_for(process.wait(),timeout=3)
            except asyncio.TimeoutError:
                try: os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError): process.kill()
                await process.wait()
            self.process=None; self._stdout_buffer.clear()

async def discover_mcp(client: McpStdioClient, registry: ToolRegistry) -> list[ToolSpec]:
    result=await client.request("tools/list"); discovered=[]
    for item in result.get("tools",[])[:50]:
        name=item.get("name") if isinstance(item,dict) else None
        if not isinstance(name,str) or not name or len(name)>100 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in name): continue
        if client.config.allowed_tools and name not in client.config.allowed_tools: continue
        schema=item.get("inputSchema", {"type":"object"}); schema=schema if isinstance(schema,dict) else {"type":"object"}
        try:
            if len(json.dumps(schema)) > 100_000: continue
        except (TypeError, ValueError): continue
        spec=ToolSpec(name=f"mcp.{client.config.name}.{name}",description=str(item.get("description","MCP tool"))[:500],input_schema=schema,risk=client.config.max_risk,family="mcp",origin="mcp",trust="untrusted",provenance=client.config.name)
        async def call(args:dict[str,Any], original=name)->Any: return await client.request("tools/call", {"name":original,"arguments":args})
        registry.register(spec,call); discovered.append(spec)
    return discovered

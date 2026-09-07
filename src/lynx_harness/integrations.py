"""Opt-in integration lifecycle behind the common ToolGateway.

Adapters remain explicit host configuration.  This module only wires their
registration/start/cleanup lifecycle; it never discovers commands or enables
external communication implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .a2a import A2AClient, AgentTask
from .mcp import McpStdioClient, discover_mcp
from .models import RiskLevel, ToolSpec
from .tools import ToolRegistry


@dataclass
class IntegrationManager:
    registry: ToolRegistry
    mcp_clients: list[McpStdioClient] = field(default_factory=list)
    kernels: list[Any] = field(default_factory=list)
    browsers: list[Any] = field(default_factory=list)
    lsp_clients: list[Any] = field(default_factory=list)
    a2a_clients: list[A2AClient] = field(default_factory=list)
    _started: bool = False

    def add_kernel(self, kernel: Any) -> None:
        if self._started:
            raise RuntimeError("integrations cannot be changed after start")
        from .kernel import register_kernel_tool
        register_kernel_tool(self.registry, kernel)
        self.kernels.append(kernel)

    def add_browser(self, browser: Any) -> None:
        if self._started:
            raise RuntimeError("integrations cannot be changed after start")
        from .playwright import register_playwright_tools
        register_playwright_tools(self.registry, browser)
        self.browsers.append(browser)

    def add_lsp(self, client: Any) -> None:
        if self._started:
            raise RuntimeError("integrations cannot be changed after start")
        from .lsp import register_lsp_tools
        register_lsp_tools(self.registry, client)
        self.lsp_clients.append(client)

    def add_a2a(self, client: A2AClient) -> None:
        """Expose an explicitly configured remote agent as a gated tool."""
        if self._started:
            raise RuntimeError("integrations cannot be changed after start")
        if not isinstance(client, A2AClient):
            raise TypeError("A2A integration requires A2AClient")
        index = len(self.a2a_clients)
        name = f"a2a.submit.{index}"

        async def invoke(args: dict[str, Any]) -> Any:
            if not isinstance(args, dict):
                raise ValueError("A2A arguments must be an object")
            task = AgentTask(
                id=args["id"], context_id=args.get("context_id", args["id"]),
                goal=args["goal"], inputs=tuple(args.get("inputs", ())),
                budget=args.get("budget", {}), expected_output=args.get("expected_output", {}),
                security=args.get("security", {}), parent_task_id=args.get("parent_task_id"),
                branch_id=args.get("branch_id", "main"), depth=args.get("depth", 0),
                max_depth=args.get("max_depth", 2), max_children=args.get("max_children", 8),
            )
            return await client.submit(task)

        self.registry.register(ToolSpec(
            name, "Submit a bounded artifact-referenced task to an explicitly allowlisted A2A agent",
            {"type": "object", "required": ["id", "goal"], "properties": {
                "id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"},
                "context_id": {"type": "string"}, "goal": {"type": "string", "maxLength": 20000},
                "inputs": {"type": "array", "items": {"type": "string", "maxLength": 500}, "maxItems": 100},
                "budget": {"type": "object"}, "expected_output": {"type": "object"},
                "security": {"type": "object"}, "parent_task_id": {"type": "string"},
                "branch_id": {"type": "string"}, "depth": {"type": "integer"},
                "max_depth": {"type": "integer"}, "max_children": {"type": "integer"},
            }}, risk=RiskLevel.EXTERNAL_MUTATION, family="a2a", origin="a2a", trust="untrusted",
            provenance=client.endpoint), invoke)
        self.a2a_clients.append(client)

    def add_mcp(self, client: McpStdioClient) -> None:
        if self._started:
            raise RuntimeError("integrations cannot be changed after start")
        if not isinstance(client, McpStdioClient):
            raise TypeError("MCP integration requires McpStdioClient")
        self.mcp_clients.append(client)

    async def start(self) -> list[Any]:
        """Start configured MCP servers and discover allowlisted tools.

        Registration happens before the runner freezes its registry.  A failed
        server fails closed and cleanup is attempted for every client already
        started.
        """
        if self._started:
            return []
        discovered = []
        try:
            for client in self.mcp_clients:
                await client.start()
                discovered.extend(await discover_mcp(client, self.registry))
            # LSP clients own a persistent JSON-RPC process just like MCP
            # clients. Starting them here makes IntegrationManager the actual
            # lifecycle owner instead of requiring an undocumented manual call.
            for client in self.lsp_clients:
                start = getattr(client, "start", None)
                if start is not None:
                    result = start()
                    if hasattr(result, "__await__"):
                        await result
            self._started = True
            return discovered
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        errors = []
        for client in reversed(self.mcp_clients):
            try:
                await client.close()
            except Exception as exc:
                errors.append(exc)
        for resource in [*reversed(self.browsers), *reversed(self.lsp_clients), *reversed(self.kernels)]:
            close = getattr(resource, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                errors.append(exc)
        self._started = False
        if errors:
            raise RuntimeError("one or more integration resources failed to close") from errors[0]

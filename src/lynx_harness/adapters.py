from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol
from .models import Observation, RiskLevel, ToolSpec
from .tools import ToolGateway, ToolRegistry

class ToolAdapter(Protocol):
    spec: ToolSpec
    async def execute(self, arguments: dict[str, Any]) -> Any: ...

@dataclass(frozen=True)
class AdapterError:
    kind: str
    message: str
    retryable: bool = False

def register_adapter(registry: ToolRegistry, spec: ToolSpec, execute: Callable[[dict[str, Any]], Awaitable[Any]]) -> None:
    if not isinstance(spec, ToolSpec) or not callable(execute): raise TypeError("invalid tool adapter")
    registry.register(spec, execute)

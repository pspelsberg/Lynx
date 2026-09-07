from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .models import Decision, Observation


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(_bounded(value))).hexdigest()


def _bounded(value: Any, depth: int = 0) -> Any:
    if depth > 4: return "[bounded]"
    if isinstance(value, str): return value[:10_000]
    if isinstance(value, dict): return {str(key): _bounded(item, depth + 1) for key, item in list(value.items())[:100]}
    if isinstance(value, list): return [_bounded(item, depth + 1) for item in value[:100]]
    return value


@dataclass(frozen=True)
class ProgressSignal:
    stagnant: bool
    reason: str = ""
    repeated: int = 0
    key: str = ""


class ProgressMonitor:
    """Detect identical decision/observation cycles before a budget is exhausted."""
    def __init__(self, threshold: int = 3):
        if threshold < 2: raise ValueError("threshold must be >= 2")
        self.threshold = threshold
        self._last_key = ""
        self._repeated = 0

    def update(self, decision: Decision, observation: Observation | None = None) -> ProgressSignal:
        payload = {"type": decision.type.value, "tool": decision.tool, "arguments": decision.arguments, "reason": decision.reason}
        if observation is not None:
            payload["observation"] = {"ok": observation.ok, "output": observation.output, "error": observation.error, "artifact_id": observation.artifact_id}
        key = digest(payload)
        if key == self._last_key:
            self._repeated += 1
        else:
            self._last_key, self._repeated = key, 1
        reason = "repeated_action" if decision.tool else "repeated_reflection"
        return ProgressSignal(self._repeated >= self.threshold, reason if self._repeated >= self.threshold else "", self._repeated, key)


class NullProgressMonitor:
    def update(self, decision: Decision, observation: Observation | None = None) -> ProgressSignal:
        return ProgressSignal(False, "disabled", 0, "")

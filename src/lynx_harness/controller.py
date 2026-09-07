from __future__ import annotations

from enum import StrEnum
from dataclasses import dataclass
from typing import Any

class ControllerError(RuntimeError): pass

class ControllerState(StrEnum):
    CREATED = "created"
    PLANNED = "planned"
    EXECUTING = "executing"
    CHECKING = "checking"
    DELEGATED = "delegated"
    AWAITING_CHILD = "awaiting-child"
    REPAIR = "repair"
    VERIFIED = "verified"
    REJECTED = "rejected"
    MERGED = "merged"
    PROMOTED = "promoted"

_ALLOWED = {
    ControllerState.CREATED: {ControllerState.PLANNED},
    ControllerState.PLANNED: {ControllerState.EXECUTING},
    ControllerState.EXECUTING: {ControllerState.CHECKING, ControllerState.DELEGATED},
    ControllerState.DELEGATED: {ControllerState.AWAITING_CHILD},
    ControllerState.AWAITING_CHILD: {ControllerState.CHECKING, ControllerState.REPAIR, ControllerState.REJECTED},
    ControllerState.CHECKING: {ControllerState.REPAIR, ControllerState.VERIFIED, ControllerState.REJECTED, ControllerState.EXECUTING},
    ControllerState.REPAIR: {ControllerState.EXECUTING, ControllerState.CHECKING, ControllerState.REJECTED},
    ControllerState.VERIFIED: {ControllerState.MERGED, ControllerState.PROMOTED},
    ControllerState.REJECTED: set(), ControllerState.MERGED: set(), ControllerState.PROMOTED: set(),
}

@dataclass(frozen=True)
class Transition:
    source: ControllerState
    target: ControllerState
    action: str
    metadata: dict[str, Any]

class Controller:
    """Small explicit state machine; model output cannot transition it directly."""
    def __init__(self, initial: ControllerState = ControllerState.CREATED):
        self.state = ControllerState(initial)
        self.history: list[Transition] = []

    def transition(self, target: ControllerState, *, action: str = "transition", metadata: dict[str, Any] | None = None) -> ControllerState:
        target = ControllerState(target)
        if target not in _ALLOWED[self.state]: raise ControllerError(f"invalid transition: {self.state} -> {target}")
        source = self.state; self.state = target
        self.history.append(Transition(source, target, action[:80], dict(metadata or {})))
        return self.state

    def plan(self): return self.transition(ControllerState.PLANNED, action="plan")
    def start(self): return self.transition(ControllerState.EXECUTING, action="start")
    def finish(self): return self.transition(ControllerState.CHECKING, action="finish")
    def delegate(self): return self.transition(ControllerState.DELEGATED, action="delegate")
    def await_child(self): return self.transition(ControllerState.AWAITING_CHILD, action="await-child")
    def repair(self): return self.transition(ControllerState.REPAIR, action="repair")
    def verify(self, passed: bool = True): return self.transition(ControllerState.VERIFIED if passed else ControllerState.REJECTED, action="verify", metadata={"passed": passed})
    def merge(self): return self.transition(ControllerState.MERGED, action="merge")
    def promote(self): return self.transition(ControllerState.PROMOTED, action="promote")

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from typing import Any
import json
import time
import math
import uuid


class Mode(StrEnum):
    FAST = "fast"
    THINK = "think"
    RESEARCH = "research"


class RiskLevel(IntEnum):
    READ_ONLY = 0
    LOCAL_MUTATION = 1
    EXTERNAL_MUTATION = 2
    DANGEROUS = 3


class DecisionType(StrEnum):
    TOOL = "tool"
    REFLECT = "reflect"
    DELEGATE = "delegate"
    FINISH = "finish"


_CONTRACT_ID = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_NAME = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _bounded_text(value: Any, field_name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string of at most {limit} characters" if required else f"{field_name} must be a string of at most {limit} characters")
    return value


def _artifact_ref(value: Any, field_name: str = "artifact reference") -> str:
    if not isinstance(value, str) or not __import__("re").fullmatch(r"artifact://[0-9a-f]{64}\.[A-Za-z0-9_]{1,10}", value):
        raise ValueError(f"{field_name} must be a canonical artifact:// reference")
    return value


def _json_bounded(value: Any, field_name: str, limit: int = 100_000) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise ValueError(f"{field_name} is too large")
    return value


def _strict_json_loads(text: str) -> Any:
    """Decode contract JSON without lossy duplicate-key/NaN handling.

    Contracts are security and budget inputs, so silently accepting a duplicate
    key (or a non-standard NaN value) would make the validated meaning depend
    on the parser's last-write-wins behavior.  Keep this parser local to the
    model boundary rather than reusing the inference parser, which avoids a
    dependency cycle and applies the same rule to offline contract loading.
    """
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(text, parse_constant=reject_constant,
                          object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError("invalid contract JSON") from exc


def _public(value: Any) -> Any:
    """Return a serializable contract projection without secret-bearing fields."""
    from .security import _is_secret_key
    def is_secret_key(key: Any) -> bool:
        return _is_secret_key(key)
    if isinstance(value, dict):
        return {str(k): _public(v) for k, v in value.items() if not is_secret_key(k)}
    if isinstance(value, (list, tuple)):
        return [_public(v) for v in value]
    if isinstance(value, IntEnum):
        return value.name
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, str):
        # Contracts can be serialized directly (outside the trajectory
        # redaction layer), so redact credential-shaped text here as well.
        from .security import redact_text
        return redact_text(value, max_chars=None)
    return value


@dataclass(frozen=True)
class ExpectedOutput:
    """Transport-neutral output contract; schema is data, never executable policy."""
    type: str = "text"
    schema: str | dict[str, Any] | None = None
    max_chars: int = 20_000

    def __post_init__(self) -> None:
        _bounded_text(self.type, "expected_output.type", 80, required=True)
        if self.schema is not None:
            if not isinstance(self.schema, (str, dict)):
                raise ValueError("expected_output.schema must be a schema name or object")
            _json_bounded(self.schema, "expected_output.schema", 100_000)
        if not isinstance(self.max_chars, int) or not 1 <= self.max_chars <= 1_000_000:
            raise ValueError("expected_output.max_chars is outside safe bounds")


@dataclass(frozen=True)
class Postcondition:
    """Named deterministic check requested by a contract."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_NAME.fullmatch(self.name):
            raise ValueError("postcondition name is invalid")
        _json_bounded(self.arguments, "postcondition.arguments", 20_000)
        if not isinstance(self.required, bool):
            raise ValueError("postcondition.required must be boolean")


@dataclass(frozen=True)
class Usage:
    """Measured consumption, kept separate from limits so results cannot mint budget."""
    llm_tokens: int = 0
    steps: int = 0
    tool_calls: int = 0
    web_calls: int = 0
    browser_actions: int = 0
    python_seconds: float = 0.0
    recursive_calls: int = 0
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("llm_tokens", "steps", "tool_calls", "web_calls", "browser_actions", "recursive_calls", "elapsed_ms"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"usage.{name} must be a non-negative integer")
        if isinstance(self.python_seconds, bool) or not isinstance(self.python_seconds, (int, float)) or not math.isfinite(float(self.python_seconds)) or self.python_seconds < 0 or self.python_seconds > 86_400:
            raise ValueError("usage.python_seconds is outside safe bounds")


@dataclass(frozen=True)
class SecurityPolicy:
    max_risk: RiskLevel = RiskLevel.READ_ONLY
    confirmation_required: bool = False

    def __post_init__(self) -> None:
        try:
            risk = self.max_risk if isinstance(self.max_risk, RiskLevel) else RiskLevel[str(self.max_risk).upper()]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("security.max_risk is invalid") from exc
        object.__setattr__(self, "max_risk", risk)
        if not isinstance(self.confirmation_required, bool):
            raise ValueError("security.confirmation_required must be boolean")


@dataclass(frozen=True)
class TaskContract:
    id: str
    goal: str
    mode: Mode = Mode.THINK
    allowed_tools: tuple[str, ...] = ()
    input_artifacts: tuple[str, ...] = ()
    expected_output: ExpectedOutput = field(default_factory=ExpectedOutput)
    postconditions: tuple[Postcondition, ...] = ()
    budget: ComputeBudget = field(default_factory=lambda: ComputeBudget())
    security: SecurityPolicy = field(default_factory=SecurityPolicy)
    parent_task_id: str | None = None
    branch_id: str = "main"
    depth: int = 0
    max_depth: int = 2
    max_children: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _CONTRACT_ID.fullmatch(self.id):
            raise ValueError("task id is invalid")
        _bounded_text(self.goal, "goal", 20_000, required=True)
        object.__setattr__(self, "mode", Mode(self.mode))
        if not isinstance(self.allowed_tools, (tuple, list)) or len(self.allowed_tools) > 100:
            raise ValueError("allowed_tools is too large")
        tools = tuple(self.allowed_tools)
        if any(not isinstance(tool, str) or not _SAFE_NAME.fullmatch(tool) for tool in tools) or len(set(tools)) != len(tools):
            raise ValueError("allowed_tools contains an invalid or duplicate tool")
        object.__setattr__(self, "allowed_tools", tools)
        refs = tuple(self.input_artifacts)
        if len(refs) > 100 or len(set(refs)) != len(refs):
            raise ValueError("input_artifacts is invalid")
        for ref in refs: _artifact_ref(ref, "input_artifacts item")
        object.__setattr__(self, "input_artifacts", refs)
        pcs = tuple(self.postconditions)
        if len(pcs) > 100 or any(not isinstance(pc, Postcondition) for pc in pcs):
            raise ValueError("postconditions is invalid")
        object.__setattr__(self, "postconditions", pcs)
        if not isinstance(self.expected_output, ExpectedOutput): raise ValueError("expected_output is invalid")
        if not isinstance(self.budget, ComputeBudget): raise ValueError("budget is invalid")
        # A contract owns a limits snapshot; caller-side usage mutations cannot alter it.
        object.__setattr__(self, "budget", ComputeBudget(**{name: getattr(self.budget, name) for name in ("llm_tokens", "steps", "tool_calls", "web_calls", "browser_actions", "recursive_calls", "python_seconds", "parallel_branches", "max_wall_time_s")}))
        if not isinstance(self.security, SecurityPolicy): raise ValueError("security is invalid")
        if self.parent_task_id is not None and (not isinstance(self.parent_task_id, str) or not _CONTRACT_ID.fullmatch(self.parent_task_id)):
            raise ValueError("parent_task_id is invalid")
        if not isinstance(self.branch_id, str) or not _SAFE_NAME.fullmatch(self.branch_id): raise ValueError("branch_id is invalid")
        if type(self.depth) is not int or not 0 <= self.depth <= 32: raise ValueError("depth is outside safe bounds")
        if type(self.max_depth) is not int or not 0 <= self.depth <= self.max_depth <= 32: raise ValueError("max_depth is outside safe bounds")
        if type(self.max_children) is not int or not 0 <= self.max_children <= 100: raise ValueError("max_children is outside safe bounds")

    @property
    def task_id(self) -> str:
        return self.id

    @property
    def parent_id(self) -> str | None:
        return self.parent_task_id

    @property
    def artifact_refs(self) -> tuple[str, ...]:
        return self.input_artifacts

    def to_dict(self) -> dict[str, Any]:
        return _public(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskContract":
        if not isinstance(data, dict): raise ValueError("task contract must be an object")
        if any(not isinstance(key, str) for key in data):
            raise ValueError("task contract keys must be strings")
        allowed = {"id", "task_id", "goal", "mode", "allowed_tools", "input_artifacts", "expected_output", "postconditions", "budget", "security", "parent_task_id", "branch_id", "depth", "max_depth", "max_children"}
        unknown = set(data) - allowed
        if unknown: raise ValueError(f"unknown task contract fields: {sorted(unknown)}")
        value = dict(data)
        if "id" in value and "task_id" in value and value["id"] != value["task_id"]:
            raise ValueError("conflicting task identifiers")
        if "id" not in value and "task_id" in value: value["id"] = value.pop("task_id")
        elif "task_id" in value: value.pop("task_id")
        output = value.get("expected_output", {})
        if not isinstance(output, dict): raise ValueError("expected_output must be an object")
        value["expected_output"] = ExpectedOutput(**output)
        value["postconditions"] = tuple(Postcondition(**item) if isinstance(item, dict) else item for item in value.get("postconditions", ()))
        budget = value.get("budget")
        value["budget"] = ComputeBudget(**budget) if isinstance(budget, dict) else (budget or ComputeBudget())
        security = value.get("security")
        value["security"] = SecurityPolicy(**security) if isinstance(security, dict) else (security or SecurityPolicy())
        return cls(**value)

    @classmethod
    def from_json(cls, text: str) -> "TaskContract":
        if not isinstance(text, str) or len(text.encode()) > 1_000_000: raise ValueError("contract JSON is too large")
        return cls.from_dict(_strict_json_loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "TaskContract":
        if not isinstance(text, str) or len(text.encode()) > 1_000_000: raise ValueError("contract YAML is too large")
        try:
            import yaml
            # PyYAML's default loader silently overwrites duplicate mapping
            # keys. Contracts must have one unambiguous budget/policy value,
            # so use a safe loader that rejects duplicates before validation.
            class _StrictSafeLoader(yaml.SafeLoader):
                pass

            def _construct_mapping(loader, node, deep=False):
                mapping = {}
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=deep)
                    try:
                        duplicate = key in mapping
                    except TypeError as exc:
                        raise ValueError("YAML mapping keys must be scalar") from exc
                    if duplicate:
                        raise ValueError(f"duplicate YAML key: {key}")
                    try:
                        mapping[key] = loader.construct_object(value_node, deep=deep)
                    except TypeError as exc:
                        raise ValueError("YAML mapping keys must be scalar") from exc
                return mapping

            _StrictSafeLoader.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                _construct_mapping,
            )
            data = yaml.load(text, Loader=_StrictSafeLoader)
        except ImportError as exc: raise ValueError("YAML support is optional; install the eval extra") from exc
        except (RecursionError, TypeError, ValueError, yaml.YAMLError) as exc: raise ValueError("invalid contract YAML") from exc
        return cls.from_dict(data)


@dataclass(frozen=True)
class AgentTask(TaskContract):
    """Child/root task envelope; intentionally identical and transport-neutral."""
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: RiskLevel = RiskLevel.READ_ONLY
    family: str = "general"
    origin: str = "local"
    trust: str = "untrusted"
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_NAME.fullmatch(self.name): raise ValueError("tool name is invalid")
        _bounded_text(self.description, "tool description", 20_000)
        if not isinstance(self.input_schema, dict): raise ValueError("tool input schema must be an object")
        _json_bounded(self.input_schema, "tool input schema", 100_000)
        try: risk = self.risk if isinstance(self.risk, RiskLevel) else RiskLevel[str(self.risk).upper()]
        except (KeyError, TypeError, ValueError) as exc: raise ValueError("tool risk is invalid") from exc
        object.__setattr__(self, "risk", risk)
        for name, value in (("family", self.family), ("origin", self.origin), ("trust", self.trust)):
            if not isinstance(value, str) or not value or len(value) > 100: raise ValueError(f"tool {name} is invalid")
        if self.provenance is not None and (not isinstance(self.provenance, str) or len(self.provenance) > 500): raise ValueError("tool provenance is invalid")

    def summary(self) -> dict[str, Any]:
        # Progressive disclosure: the router can expose this without the full schema.
        description = self.description if self.trust == "trusted" else "Untrusted tool metadata; use only the declared schema and policy."
        return {"name": self.name, "description": description, "risk": self.risk.name, "family": self.family, "origin": self.origin, "trust": self.trust}


@dataclass
class Decision:
    type: DecisionType
    answer: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    task: "AgentTask | None" = None


@dataclass
class Observation:
    tool: str
    ok: bool
    output: Any = None
    error: str | None = None
    artifact_id: str | None = None
    elapsed_ms: int = 0
    origin: str = "local"
    trust: str = "untrusted"
    provenance: str | None = None
    capabilities: dict[str, bool] = field(default_factory=lambda: {"can_inform": True, "can_instruct": False, "can_authorize": False})
    branch_id: str = "main"
    parent_event: int | None = None
    epistemic_status: str = "observed"


@dataclass
class ComputeBudget:
    llm_tokens: int = 4_000
    steps: int = 8
    tool_calls: int = 8
    web_calls: int = 6
    browser_actions: int = 20
    recursive_calls: int = 0
    python_seconds: float = 0.0
    parallel_branches: int = 1
    max_wall_time_s: float = 60.0
    used_llm_tokens: int = 0
    used_steps: int = 0
    used_tool_calls: int = 0
    used_web_calls: int = 0
    used_browser_actions: int = 0
    used_recursive_calls: int = 0
    used_python_seconds: float = 0.0

    def __post_init__(self) -> None:
        integer_limits = ("llm_tokens", "steps", "tool_calls", "web_calls", "browser_actions", "recursive_calls", "parallel_branches")
        integer_maxima = {"llm_tokens": 10_000_000, "steps": 100_000, "tool_calls": 100_000, "web_calls": 100_000, "browser_actions": 100_000, "recursive_calls": 10_000, "parallel_branches": 128}
        for name in integer_limits:
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > integer_maxima[name]:
                raise ValueError(f"budget.{name} is outside safe bounds")
        if self.steps < 1 or self.llm_tokens < 1 or self.parallel_branches < 1:
            raise ValueError("budget requires positive llm_tokens, steps, and parallel_branches")
        if isinstance(self.max_wall_time_s, bool) or not isinstance(self.max_wall_time_s, (int, float)) or not 0.1 <= self.max_wall_time_s <= 86_400:
            raise ValueError("budget.max_wall_time_s is outside safe bounds")
        if isinstance(self.python_seconds, bool) or not isinstance(self.python_seconds, (int, float)) or not math.isfinite(float(self.python_seconds)) or self.python_seconds < 0 or self.python_seconds > 86_400:
            raise ValueError("budget.python_seconds is outside safe bounds")
        for name in ("used_llm_tokens", "used_steps", "used_tool_calls", "used_web_calls", "used_browser_actions", "used_recursive_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > getattr(self, name.removeprefix("used_")):
                raise ValueError(f"budget.{name} is invalid")
        if isinstance(self.used_python_seconds, bool) or not isinstance(self.used_python_seconds, (int, float)) or not math.isfinite(float(self.used_python_seconds)) or self.used_python_seconds < 0 or self.used_python_seconds > self.python_seconds:
            raise ValueError("budget.used_python_seconds is invalid")

    @classmethod
    def for_mode(cls, mode: Mode) -> "ComputeBudget":
        return {
            Mode.FAST: cls(llm_tokens=1_000, steps=3, tool_calls=2, max_wall_time_s=20),
            Mode.THINK: cls(llm_tokens=4_000, steps=8, tool_calls=8, max_wall_time_s=60),
            Mode.RESEARCH: cls(llm_tokens=8_000, steps=16, tool_calls=14, web_calls=12, max_wall_time_s=180),
        }[Mode(mode)]

    def can_step(self) -> bool:
        # A step may still be needed to reflect or finish after tool capacity
        # is exhausted; tool availability is enforced by consume_tool().
        return self.used_steps < self.steps

    def consume_step(self) -> None:
        if self.used_steps >= self.steps: raise ValueError("step budget exhausted")
        self.used_steps += 1

    def consume_tool(self, tool: str) -> bool:
        if self.used_tool_calls >= self.tool_calls: return False
        if tool.startswith("web.") and self.used_web_calls >= self.web_calls: return False
        if tool.startswith("browser.") and self.used_browser_actions >= self.browser_actions: return False
        self.used_tool_calls += 1
        if tool.startswith("web."): self.used_web_calls += 1
        if tool.startswith("browser."): self.used_browser_actions += 1
        return True

    def consume_python(self, seconds: float) -> bool:
        # Runtime usage is untrusted input too: booleans and non-finite values
        # must not bypass the finite python-time budget or poison accounting.
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(float(seconds)) or seconds < 0
                or self.used_python_seconds + seconds > self.python_seconds):
            return False
        self.used_python_seconds += seconds; return True

    def consume_recursive(self) -> bool:
        if self.used_recursive_calls >= self.recursive_calls: return False
        self.used_recursive_calls += 1
        return True


class BudgetExceeded(RuntimeError):
    """Raised when a reservation would exceed the root's aggregate budget."""


@dataclass(frozen=True)
class BudgetReservation:
    ledger: "BudgetLedger"
    amounts: dict[str, int | float]
    _closed: bool = field(default=False, compare=False)

    def commit(self, actual: dict[str, int | float] | None = None) -> None:
        if self._closed: raise ValueError("budget reservation is already closed")
        self.ledger._close(self, self.amounts if actual is None else actual, committed=True)
        object.__setattr__(self, "_closed", True)

    def release(self) -> None:
        if self._closed: return
        try:
            self.ledger._close(self, {}, committed=False)
        except ValueError:
            # A concurrent commit/release may have closed it atomically.
            if not self._closed: raise
        object.__setattr__(self, "_closed", True)

    def __enter__(self) -> "BudgetReservation": return self
    def __exit__(self, *_: Any) -> None:
        if not self._closed: self.release()


class BudgetLedger:
    """Thread-safe root ledger. Reservations cover work before parallel execution starts."""
    DIMENSIONS = ("llm_tokens", "steps", "tool_calls", "web_calls", "browser_actions", "python_seconds", "recursive_calls", "parallel_branches")

    def __init__(self, limits: ComputeBudget | dict[str, int | float]):
        import threading
        if isinstance(limits, ComputeBudget):
            values = {name: getattr(limits, name) for name in self.DIMENSIONS}
            values["max_wall_time_s"] = limits.max_wall_time_s
        elif isinstance(limits, dict):
            values = dict(limits)
        else: raise TypeError("limits must be ComputeBudget or a mapping")
        self.limits = {name: values.get(name, 0) for name in self.DIMENSIONS}
        self.limits["max_wall_time_s"] = values.get("max_wall_time_s", 0)
        for name, value in self.limits.items():
            if name in {"python_seconds", "max_wall_time_s"}: valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
            else: valid = type(value) is int
            if not valid or value < 0: raise ValueError(f"invalid ledger limit: {name}")
        if self.limits["parallel_branches"] < 1: raise ValueError("parallel_branches must be positive")
        if isinstance(self.limits["max_wall_time_s"], bool) or not 0.1 <= self.limits["max_wall_time_s"] <= 86_400: raise ValueError("max_wall_time_s is outside safe bounds")
        self.used = {name: 0 for name in self.DIMENSIONS}
        self.reserved = {name: 0 for name in self.DIMENSIONS}
        self._lock = threading.RLock()
        self._started_at = time.monotonic()

    @classmethod
    def from_budget(cls, budget: ComputeBudget) -> "BudgetLedger": return cls(budget)

    def elapsed_wall_time_s(self) -> float: return max(0.0, time.monotonic() - self._started_at)
    def expired(self) -> bool: return self.elapsed_wall_time_s() >= float(self.limits["max_wall_time_s"]) if "max_wall_time_s" in self.limits else False
    def snapshot(self) -> dict[str, dict[str, int | float] | float | bool]:
        with self._lock:
            return {"limit": dict(self.limits), "used": dict(self.used), "reserved": dict(self.reserved), "available": {n: self.limits[n] - self.used[n] - self.reserved[n] for n in self.DIMENSIONS}, "elapsed_wall_time_s": self.elapsed_wall_time_s(), "wall_time_exceeded": self.expired()}

    def can_reserve(self, amounts: dict[str, int | float] | None = None, **kwargs: int | float) -> bool:
        amounts = {**(amounts or {}), **kwargs}
        try: self._validate_amounts(amounts)
        except ValueError: return False
        with self._lock:
            return not self.expired() and all(self.used[n] + self.reserved[n] + amounts.get(n, 0) <= self.limits[n] for n in self.DIMENSIONS)

    def reserve(self, amounts: dict[str, int | float] | None = None, **kwargs: int | float) -> BudgetReservation:
        amounts = {**(amounts or {}), **kwargs}
        self._validate_amounts(amounts)
        with self._lock:
            if self.expired():
                raise BudgetExceeded("root wall-time budget exhausted")
            if not all(self.used[n] + self.reserved[n] + amounts.get(n, 0) <= self.limits[n] for n in self.DIMENSIONS):
                raise BudgetExceeded("root budget exhausted")
            normalized = {n: amounts.get(n, 0) for n in self.DIMENSIONS}
            for n, value in normalized.items(): self.reserved[n] += value
            return BudgetReservation(self, normalized)

    def consume(self, amounts: dict[str, int | float] | None = None, **kwargs: int | float) -> None:
        reservation = self.reserve(amounts, **kwargs); reservation.commit()

    def _validate_amounts(self, amounts: dict[str, int | float]) -> None:
        unknown = set(amounts) - set(self.DIMENSIONS)
        if unknown: raise ValueError(f"unknown budget dimensions: {sorted(unknown)}")
        for name, value in amounts.items():
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) if name in {"python_seconds", "max_wall_time_s"} else type(value) is int
            if not valid or value < 0: raise ValueError(f"invalid budget amount: {name}")

    def _close(self, reservation: BudgetReservation, actual: dict[str, int | float], *, committed: bool) -> None:
        self._validate_amounts(actual)
        with self._lock:
            if reservation._closed:
                raise ValueError("budget reservation is already closed")
            if any(actual.get(n, 0) > reservation.amounts[n] for n in self.DIMENSIONS):
                raise BudgetExceeded("actual usage exceeded reservation")
            for n in self.DIMENSIONS: self.reserved[n] -= reservation.amounts[n]
            if committed:
                for n in self.DIMENSIONS: self.used[n] += actual.get(n, 0)
            object.__setattr__(reservation, "_closed", True)


def _compact_observation_for_context(obs: dict[str, Any]) -> dict[str, Any]:
    tool = obs.get("tool", "")
    output = str(obs.get("output", ""))
    if len(output) > 200:
        if tool == "filesystem.read":
            lines = output.count("\n") + 1
            compact_out = f"[File content already observed: {lines} lines - compacted for context preservation]"
        elif tool == "filesystem.patch":
            compact_out = "[Patch applied - compacted for context preservation]"
        elif tool == "filesystem.write":
            compact_out = "[File written - compacted for context preservation]"
        elif tool == "workspace.test":
            compact_out = f"[Test execution completed: {output[:150]}...]"
        else:
            compact_out = output[:200] + "... [trimmed older observation]"
        return {**obs, "output": compact_out}
    return obs


@dataclass
class AgentState:
    goal: str
    mode: Mode
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    branch_id: str = "main"
    parent_event: int | None = None
    observations: list[Observation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    answer: str = ""
    status: str = "running"
    steps: int = 0
    # Inference accounting is explicit so evaluation can distinguish an
    # exhausted run from one that produced a valid structured decision.
    inference_calls: int = 0
    decisions: int = 0
    # Kept in memory for the verified trajectory export gate; it is not used as
    # model context and is serialized only through the normal redacted result.
    verification: Any | None = field(default=None, repr=False, compare=False)

    def context(self, tool_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        from .observation import ObservationNormalizer
        raw_obs = [ObservationNormalizer().normalize(item) for item in self.observations[-8:]]
        compacted = []
        cutoff = max(0, len(raw_obs) - 2)
        for idx, obs in enumerate(raw_obs):
            if idx < cutoff:
                compacted.append(_compact_observation_for_context(obs))
            else:
                compacted.append(obs)
        trusted = [item for item in compacted if item.get("trust") == "trusted"]
        untrusted = [item for item in compacted if item.get("trust") != "trusted"]
        return {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "goal": self.goal,
            "mode": self.mode.value,
            "notes": self.notes[-6:],
            "trusted_state": {"observations": trusted},
            "untrusted_observations": [{"boundary": "data_only", "can_inform": True, "can_instruct": False, "can_authorize": False, "observation": item} for item in untrusted],
            "available_tools": tool_summaries,
        }

    def fork(self, branch_id: str) -> "AgentState":
        import copy
        child = copy.deepcopy(self)
        child.branch_id = branch_id
        child.parent_event = len(self.observations)
        # Branches get a snapshot, but subsequent observations remain private.
        child.observations = copy.deepcopy(self.observations)
        return child


@dataclass
class TrajectoryEvent:
    kind: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

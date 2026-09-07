"""Bounded, callback-driven hardware benchmark profiling.

The profiler deliberately does not know how to start a server or inspect a
machine.  A host supplies a callback for one :class:`BenchmarkProfile`; this
keeps hardware access behind the same integration boundary as the rest of the
harness and makes the matrix runner straightforward to test.
"""
from __future__ import annotations

import asyncio
import inspect
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, TypeAlias
from pathlib import Path
import json
from .security import redact, atomic_write_text
from .model_profiles import require_single_model, CANONICAL_MODEL_FILENAME

# This is a safety limit, not a suggestion to benchmark every possible
# combination.  Callers can use a smaller limit for an individual run.
MAX_MATRIX_PROFILES = 128
MAX_PROFILE_NAME = 160


def _finite_nonnegative(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def _positive_int(value: int, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{field_name} must be an integer in 1..{maximum}")
    return value


@dataclass(frozen=True)
class BenchmarkProfile:
    """One reproducible runtime/hardware configuration.

    ``name`` is the human-readable, stable identity used for tie-breaking.
    The remaining fields intentionally describe configuration only; measured
    values belong in :class:`BenchmarkResult`.
    """

    name: str
    backend: str = "cpu"
    model: str = "local"
    context_size: int = 8192
    batch_size: int = 1
    gpu_layers: int = 0
    threads: int = 1
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > MAX_PROFILE_NAME:
            raise ValueError("profile name must be a non-empty bounded string")
        for field_name in ("backend", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or len(value) > MAX_PROFILE_NAME:
                raise ValueError(f"{field_name} must be a non-empty bounded string")
        if isinstance(self.context_size, bool) or not isinstance(self.context_size, int) or not 1 <= self.context_size <= 1_000_000:
            raise ValueError("context_size must be an integer in 1..1000000")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or not 1 <= self.batch_size <= 65_536:
            raise ValueError("batch_size must be an integer in 1..65536")
        if isinstance(self.gpu_layers, bool) or not isinstance(self.gpu_layers, int) or not 0 <= self.gpu_layers <= 100_000:
            raise ValueError("gpu_layers must be an integer in 0..100000")
        _positive_int(self.threads, "threads", 65_536)
        if len(self.options) > 100:
            raise ValueError("profile options are too numerous")
        option_names: set[str] = set()
        for pair in self.options:
            if not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(item, str) or not item or len(item) > 500 for item in pair):
                raise ValueError("profile options must be bounded string pairs")
            if pair[0] in option_names:
                raise ValueError("profile option names must be unique")
            option_names.add(pair[0])

    @property
    def key(self) -> str:
        """Canonical identity used for deterministic ordering."""
        options = tuple(sorted(self.options))
        return repr((self.name, self.backend, self.model, self.context_size, self.batch_size, self.gpu_layers, self.threads, options))


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured outcome for one profile.

    ``score`` is the primary objective supplied by the benchmark callback.
    ``throughput`` and ``latency_ms`` are optional conventional metrics and are
    used as deterministic fallbacks when scores tie.
    """

    profile: BenchmarkProfile
    score: float = 0.0
    throughput: float = 0.0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    success: bool = True
    error: str | None = None
    samples: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    prompt_tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    task_success: bool | None = None
    verifier_rate: float | None = None
    llama_commit: str = "unknown"
    driver: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.profile, BenchmarkProfile):
            raise TypeError("result.profile must be a BenchmarkProfile")
        for field_name in ("score", "throughput", "latency_ms", "memory_mb"):
            _finite_nonnegative(getattr(self, field_name), field_name)
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if self.error is not None and (not isinstance(self.error, str) or len(self.error) > 5_000):
            raise ValueError("error must be a bounded string")
        _positive_int(self.samples, "samples", 1_000_000)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        for field_name in ("prompt_tokens_per_second", "decode_tokens_per_second", "p50_latency_ms", "p95_latency_ms"):
            _finite_nonnegative(getattr(self, field_name), field_name)
        if self.verifier_rate is not None and (isinstance(self.verifier_rate, bool) or not isinstance(self.verifier_rate, (int, float)) or not math.isfinite(float(self.verifier_rate)) or not 0 <= self.verifier_rate <= 1): raise ValueError("verifier_rate must be between 0 and 1")
        if self.task_success is not None and not isinstance(self.task_success, bool): raise TypeError("task_success must be bool or None")
        for key, value in self.metadata.items():
            if not isinstance(key, str) or len(key) > 200: raise ValueError("metadata keys are invalid")
        try:
            if len(json.dumps(dict(self.metadata), ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")) > 1_000_000: raise ValueError("metadata is too large")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc) == "metadata is too large": raise
            raise ValueError("metadata must be serializable") from exc
        if not isinstance(self.llama_commit, str) or len(self.llama_commit)>200 or not isinstance(self.driver, str) or len(self.driver)>200: raise ValueError("runtime metadata is invalid")

    @property
    def tokens_per_second(self) -> float:
        """Common spelling for throughput, retained as a read-only alias."""
        return self.throughput

    @property
    def profile_id(self) -> str:
        return self.profile.key


@dataclass(frozen=True)
class BenchmarkMatrix:
    """Cartesian product of bounded profile dimensions."""

    backends: tuple[str, ...] = ("cpu",)
    models: tuple[str, ...] = ("local",)
    context_sizes: tuple[int, ...] = (8192,)
    batch_sizes: tuple[int, ...] = (1,)
    gpu_layers: tuple[int, ...] = (0,)
    threads: tuple[int, ...] = (1,)
    max_profiles: int = MAX_MATRIX_PROFILES

    def __post_init__(self) -> None:
        _positive_int(self.max_profiles, "max_profiles", MAX_MATRIX_PROFILES)
        dimension_names = ("backends", "models", "context_sizes", "batch_sizes", "gpu_layers", "threads")
        normalized = []
        for name in dimension_names:
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"matrix dimension {name} must be an iterable of values")
            try:
                # Materialize only one item beyond the global bound so an
                # accidental infinite/lazy dimension cannot exhaust memory.
                values = tuple(itertools.islice(iter(value), self.max_profiles + 1))
            except TypeError as exc:
                raise ValueError(f"matrix dimension {name} must be iterable") from exc
            if not values:
                raise ValueError(f"matrix dimension {name} must not be empty")
            if len(values) > self.max_profiles:
                raise ValueError(f"matrix dimension {name} exceeds {self.max_profiles} values")
            object.__setattr__(self, name, values)
            normalized.append(values)
        count = math.prod(len(item) for item in normalized)
        if count > self.max_profiles:
            raise ValueError(f"profile matrix exceeds {self.max_profiles} profiles")

    def profiles(self) -> tuple[BenchmarkProfile, ...]:
        result = []
        for backend, model, context, batch, layers, threads in itertools.product(self.backends, self.models, self.context_sizes, self.batch_sizes, self.gpu_layers, self.threads):
            name = f"{backend}:{model}:ctx{context}:batch{batch}:gpu{layers}:threads{threads}"
            result.append(BenchmarkProfile(name, backend, model, context, batch, layers, threads))
        return tuple(result)


BenchmarkCallback: TypeAlias = Callable[[BenchmarkProfile], BenchmarkResult | Mapping[str, Any] | float | Awaitable[BenchmarkResult | Mapping[str, Any] | float]]


def _coerce_result(profile: BenchmarkProfile, value: BenchmarkResult | Mapping[str, Any] | float) -> BenchmarkResult:
    if isinstance(value, BenchmarkResult):
        if value.profile != profile:
            raise ValueError("benchmark callback returned a result for another profile")
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        data.setdefault("profile", profile)
        result = BenchmarkResult(**data)
        if result.profile != profile:
            raise ValueError("benchmark callback returned a result for another profile")
        return result
    return BenchmarkResult(profile=profile, score=_finite_nonnegative(value, "score"))


async def run_profile_matrix(
    profiles: Iterable[BenchmarkProfile],
    callback: BenchmarkCallback,
    *,
    max_profiles: int = MAX_MATRIX_PROFILES,
    max_concurrency: int = 1,
    continue_on_error: bool = False,
    timeout_s: float | None = None,
) -> list[BenchmarkResult]:
    """Run at most ``max_profiles`` configurations through ``callback``.

    Results retain input order even when concurrency is requested. The
    callback is the only execution seam: this function never invokes a
    process, opens a URL, or makes assumptions about the hardware. When a
    timeout is requested, callbacks must be async so cancellation is
    enforceable; synchronous adapters must be wrapped by the host in a
    cancellable process boundary.
    """
    _positive_int(max_profiles, "max_profiles", MAX_MATRIX_PROFILES)
    _positive_int(max_concurrency, "max_concurrency", MAX_MATRIX_PROFILES)
    if timeout_s is not None:
        timeout_s = _finite_nonnegative(timeout_s, "timeout_s")
        if not 0.01 <= timeout_s <= 3_600:
            raise ValueError("timeout_s must be in 0.01..3600")
    if isinstance(profiles, BenchmarkMatrix):
        profiles = profiles.profiles()
    # Consume only one item beyond the limit.  Calling ``list`` here would
    # defeat the bound for a lazy or accidentally-unbounded input iterable.
    profile_list: list[BenchmarkProfile] = []
    for profile in profiles:
        if len(profile_list) >= max_profiles:
            raise ValueError(f"profile matrix exceeds {max_profiles} profiles")
        if not isinstance(profile, BenchmarkProfile):
            raise TypeError("profiles must contain BenchmarkProfile values")
        profile_list.append(profile)
    if not callable(callback):
        raise TypeError("callback must be callable")
    if timeout_s is not None and not inspect.iscoroutinefunction(callback):
        # A synchronous callback cannot be safely cancelled in Python. Reject
        # it before invocation instead of letting a blocking hardware adapter
        # defeat the timeout or keep an executor thread alive after return.
        raise TypeError("timeout_s requires an async benchmark callback")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(profile: BenchmarkProfile) -> BenchmarkResult:
        async with semaphore:
            try:
                value = callback(profile)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=timeout_s) if timeout_s is not None else await value
                return _coerce_result(profile, value)
            except Exception as exc:
                if not continue_on_error:
                    raise
                return BenchmarkResult(profile=profile, success=False, error=str(exc)[:5_000])

    running = [asyncio.create_task(one(profile)) for profile in profile_list]
    try:
        return list(await asyncio.gather(*running))
    except BaseException:
        for task in running:
            if not task.done(): task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise


def select_best_profile(results: Iterable[BenchmarkResult], *, min_verifier_rate: float = 0.0, require_task_success: bool = True) -> BenchmarkResult | None:
    """Select quality first, subject to an optional verifier/success gate.

    Missing measurements are never treated as a passing verifier result when a
    positive threshold is requested.  This prevents a fast but unverified
    profile from silently becoming the runtime default.
    """
    if isinstance(min_verifier_rate, bool) or not isinstance(min_verifier_rate, (int, float)) or not math.isfinite(float(min_verifier_rate)) or not 0 <= min_verifier_rate <= 1:
        raise ValueError("min_verifier_rate must be between 0 and 1")
    candidates = [item for item in results if isinstance(item, BenchmarkResult) and item.success and item.error is None and (not require_task_success or item.task_success is not False) and (item.verifier_rate is not None and item.verifier_rate >= min_verifier_rate if min_verifier_rate > 0 else True)]
    if not candidates:
        return None
    # Maximize objective and throughput; minimize latency/memory.  The final
    # canonical profile key makes ties stable regardless of callback order.
    return sorted(candidates, key=lambda item: (-item.score, -item.throughput, item.latency_ms, item.memory_mb, item.profile.key))[0]


# Descriptive aliases keep the small API discoverable for callers that use
# "benchmark" rather than "profile" terminology.
run_benchmark_matrix = run_profile_matrix
run_matrix = run_profile_matrix
best_profile = select_best_profile
choose_best_profile = select_best_profile
BenchmarkMeasurement = BenchmarkResult
HardwareProfile = BenchmarkProfile

__all__ = [
    "MAX_MATRIX_PROFILES", "BenchmarkProfile", "HardwareProfile", "BenchmarkResult",
    "BenchmarkMeasurement", "BenchmarkMatrix", "BenchmarkCallback",
    "run_profile_matrix", "run_benchmark_matrix", "run_matrix",
    "select_best_profile", "best_profile", "choose_best_profile",
]


@dataclass(frozen=True)
class LynxBenchmarkMatrix:
    """Recommended RX 9060 XT matrix for the single permitted Qwen model."""
    models: tuple[str, ...] = ("qwen35-4b-q6_k",)
    backends: tuple[str, ...] = ("rocm", "vulkan")
    contexts: tuple[int, ...] = (4096, 8192, 16384)
    slots: tuple[int, ...] = (1, 2, 3)
    thinking: tuple[bool, ...] = (False, True)
    flash_attention: tuple[bool, ...] = (False, True)
    def __post_init__(self) -> None:
        # Thinking is measured as three requests per server configuration by
        # profile-local.py, rather than being a separate server dimension.
        # Keeping it in this API is useful for callers that describe the
        # requested modes, but expanding it here would duplicate every row
        # and exceed MAX_MATRIX_PROFILES (2*3*3*2*2*2 = 144).
        dimensions = ("models", "backends", "contexts", "slots", "thinking", "flash_attention")
        for name in dimensions:
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"benchmark dimension {name} must be iterable")
            try:
                # The recommended matrix shares the same hard profile bound;
                # do not fully consume an accidental unbounded generator.
                normalized = tuple(itertools.islice(iter(value), MAX_MATRIX_PROFILES + 1))
            except TypeError as exc:
                raise ValueError(f"benchmark dimension {name} must be iterable") from exc
            if not normalized:
                raise ValueError("benchmark dimensions must not be empty")
            if len(normalized) > MAX_MATRIX_PROFILES:
                raise ValueError(f"benchmark dimension {name} exceeds {MAX_MATRIX_PROFILES} values")
            object.__setattr__(self, name, normalized)
        if self.models != ("qwen35-4b-q6_k",):
            raise ValueError("the runtime benchmark is restricted to qwen35-4b-q6_k")
        if any(not isinstance(value, str) or not value.strip() or len(value) > MAX_PROFILE_NAME for value in self.backends):
            raise ValueError("benchmark backends are invalid")
        if any(type(value) is not int or not 1 <= value <= 1_000_000 for value in self.contexts):
            raise ValueError("benchmark contexts are invalid")
        if any(type(value) is not int or not 1 <= value <= 65_536 for value in self.slots):
            raise ValueError("benchmark slots are invalid")
        if any(type(value) is not bool for value in self.thinking + self.flash_attention):
            raise ValueError("benchmark mode dimensions must be boolean")
        count = len(self.models) * len(self.backends) * len(self.contexts) * len(self.slots) * len(self.flash_attention)
        if count > MAX_MATRIX_PROFILES:
            raise ValueError(f"benchmark matrix exceeds {MAX_MATRIX_PROFILES} profiles")
    def profiles(self) -> tuple[BenchmarkProfile, ...]:
        values=[]
        for model,backend,context,slot,flash in itertools.product(self.models,self.backends,self.contexts,self.slots,self.flash_attention):
            values.append(BenchmarkProfile(f"{model}:{backend}:ctx{context}:slots{slot}:flash{int(flash)}",backend=backend,model=model,context_size=context,batch_size=slot,options=(("flash_attention",str(flash)),)))
        return tuple(values)


def serialize_benchmark(results: Iterable[BenchmarkResult], *, matrix: str = "", environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Serialize a bounded report for the permitted runtime model.

    ``BenchmarkProfile`` remains usable with a descriptive ``local`` model in
    offline unit tests, but a persisted benchmark is runtime provenance and
    must not silently claim measurements for an unapproved checkpoint.
    """
    rows=[]
    environment = dict(environment or {})
    for key in ("model", "model_id"):
        if key in environment:
            try:
                require_single_model(environment[key], field=f"benchmark environment {key}")
            except ValueError as exc:
                raise ValueError("benchmark environment model is outside the single-model policy") from exc
    canonical_model_id = require_single_model(CANONICAL_MODEL_FILENAME, field="benchmark model")
    for result in results:
        if len(rows) >= MAX_MATRIX_PROFILES:
            raise ValueError("benchmark report contains too many profiles")
        if not isinstance(result, BenchmarkResult):
            raise TypeError("benchmark results must contain BenchmarkResult values")
        try:
            canonical_model_id = require_single_model(result.profile.model, field="benchmark model")
        except ValueError as exc:
            raise ValueError("benchmark report model is outside the single-model policy") from exc
        rows.append({"profile":result.profile.key,"name":result.profile.name,"backend":result.profile.backend,"model":result.profile.model,"context_size":result.profile.context_size,"batch_size":result.profile.batch_size,"options":dict(result.profile.options),"score":result.score,"throughput":result.throughput,"latency_ms":result.latency_ms,"p50_latency_ms":result.p50_latency_ms,"p95_latency_ms":result.p95_latency_ms,"memory_mb":result.memory_mb,"success":result.success,"error":redact(result.error) if result.error is not None else None,"samples":result.samples,"prompt_tokens_per_second":result.prompt_tokens_per_second,"decode_tokens_per_second":result.decode_tokens_per_second,"task_success":result.task_success,"verifier_rate":result.verifier_rate,"llama_cpp_commit":result.llama_commit,"driver":result.driver,"metadata":redact(dict(result.metadata))})
    return {"schema":"lynx.profiler.v1","model_policy":{"model_id":canonical_model_id, "model_filename": CANONICAL_MODEL_FILENAME},"matrix":matrix,"environment":redact(environment),"results":rows}

def write_benchmark_report(path: Path, results: Iterable[BenchmarkResult], *, matrix: str = "", environment: Mapping[str, Any] | None = None) -> Path:
    report=serialize_benchmark(results,matrix=matrix,environment=environment); encoded=json.dumps(report,ensure_ascii=False,indent=2,default=str)
    if len(encoded.encode())>50_000_000: raise ValueError("benchmark report too large")
    return atomic_write_text(path, encoded + "\n", max_bytes=50_000_000)

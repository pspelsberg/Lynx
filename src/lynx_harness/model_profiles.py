from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, math, re
from pathlib import Path
from typing import Any, Iterable
from .models import Mode

_HEX64 = set("0123456789abcdef")
_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{7,128}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CANONICAL_MODEL_ID = "qwen35-4b-q6_k"
CANONICAL_MODEL_FILENAME = "Qwen3.5-4B-Q6_K.gguf"
CANONICAL_MODEL_REPOSITORY = "unsloth/Qwen3.5-4B-GGUF"
CANONICAL_MODEL_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
CANONICAL_MODEL_SHA256 = "fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66"
# The runtime has one model identity.  Keep aliases deliberately narrow: a
# basename and the repository-local models/ path are the only accepted forms.
# In particular, do not accept arbitrary paths merely because their basename
# happens to look like the canonical file.
CANONICAL_MODEL_NAMES = frozenset({CANONICAL_MODEL_ID, CANONICAL_MODEL_FILENAME, f"models/{CANONICAL_MODEL_FILENAME}"})


def require_single_model(model: str, *, field: str = "model") -> str:
    """Validate and normalize a model identity under the single-model policy.

    This gate is shared by benchmark, teacher/student and training seams.  It
    returns the stable manifest id so callers cannot accidentally compare
    aliases as if they were distinct checkpoints.
    """
    if not isinstance(model, str) or model not in CANONICAL_MODEL_NAMES:
        raise ValueError(f"{field} is restricted to {CANONICAL_MODEL_FILENAME}")
    return CANONICAL_MODEL_ID


def backend_model_identity(backend: Any, *, field: str = "backend model") -> str:
    """Extract a backend's declared model and apply the runtime model gate.

    A backend without an explicit identity is rejected rather than silently
    treated as the canonical model.  Offline ``ScriptedBackend`` declares the
    canonical identity explicitly, making deterministic tests honest too.
    """
    if backend is None:
        raise ValueError(f"{field} identity is required")
    declared: list[Any] = []
    for attribute in ("model", "model_id"):
        value = getattr(backend, attribute, None)
        if value is not None:
            declared.append(value)
    profile = getattr(backend, "profile", None)
    profile_model = getattr(profile, "model_id", None) if profile is not None else None
    if profile_model is not None:
        declared.append(profile_model)
    if not declared:
        raise ValueError(f"{field} identity is required")
    identities = {require_single_model(value, field=field) for value in declared}
    if len(identities) != 1:
        raise ValueError(f"{field} identity declarations disagree")
    return identities.pop()

@dataclass(frozen=True)
class ModelManifestEntry:
    id: str
    repository: str
    revision: str
    filename: str
    sha256: str | None
    license: str = "unknown"
    quantization: str = "unknown"
    capabilities: tuple[str, ...] = ()
    vram_hint: str = ""
    downloadable: bool = False

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("repository", self.repository), ("filename", self.filename)):
            if not isinstance(value, str) or not value.strip() or len(value) > 500:
                raise ValueError(f"manifest {name} is invalid")
        if not _SAFE_ID.fullmatch(self.id):
            raise ValueError("manifest id is invalid")
        if not isinstance(self.revision, str) or len(self.revision) > 200:
            raise ValueError("manifest revision is invalid")
        if self.sha256 is not None and (not isinstance(self.sha256, str) or len(self.sha256) != 64 or set(self.sha256.lower()) - _HEX64):
            raise ValueError("manifest sha256 must be a 64-character hexadecimal string")
        if not isinstance(self.downloadable, bool):
            raise ValueError("manifest downloadable must be boolean")
        if not isinstance(self.capabilities, tuple) or len(self.capabilities) > 50 or any(not isinstance(v, str) or not v or len(v) > 100 for v in self.capabilities):
            raise ValueError("manifest capabilities are invalid")
        self.validate_download()

    def validate_download(self) -> None:
        """Reject mutable or noncanonical models before any download/start.

        The manifest contains historical/teacher entries for documentation, but
        only the one canonical entry may ever be marked downloadable.  Keeping
        this check here prevents callers that bypass the shell helper from
        turning the manifest into a generic model loader.
        """
        if not self.downloadable:
            return
        if (self.id != CANONICAL_MODEL_ID
                or self.repository != CANONICAL_MODEL_REPOSITORY
                or self.revision != CANONICAL_MODEL_REVISION
                or self.filename != CANONICAL_MODEL_FILENAME
                or self.sha256 != CANONICAL_MODEL_SHA256):
            raise ValueError("only the pinned canonical model is downloadable")
        if not _PINNED_REVISION.fullmatch(self.revision) or not self.sha256:
            raise ValueError("model download requires a pinned revision and SHA-256")

    def verify_file(self, path: Path) -> bool:
        self.validate_download()
        # Hashing through a symlink makes a launcher-time path substitution
        # possible even when the target bytes happen to match. Runtime model
        # paths are repository-controlled regular files, not links.
        if (not self.sha256 or not isinstance(path, Path)
                or path.is_symlink() or not path.is_file()):
            return False
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest().lower() == self.sha256.lower()

def canonical_model_path(path: Path, *, root: Path | None = None) -> Path:
    """Validate and return the sole runtime model path.

    A basename/hash check alone permits an operator to point the launcher at a
    different location (or a swapped symlink).  Runtime launchers therefore
    require the repository's canonical ``models/`` file and reject symlinks.
    """
    if not isinstance(path, Path):
        raise ValueError("model path must be a Path")
    if path.is_symlink():
        raise ValueError("canonical model path must not be a symlink")
    project_root = (root if root is not None else Path(__file__).resolve().parents[2]).resolve()
    # Check lexical ancestors before resolving.  Otherwise a symlinked
    # ``models`` directory (or another ancestor) can redirect a path that
    # eventually resolves to the expected filename outside the repository.
    absolute_path = path.absolute()
    if any(parent.is_symlink() for parent in absolute_path.parents):
        raise ValueError("canonical model path ancestors must not be symlinks")
    expected = project_root / "models" / CANONICAL_MODEL_FILENAME
    if expected.parent.is_symlink() or absolute_path.resolve(strict=False) != expected:
        raise ValueError("only the canonical models/Qwen3.5-4B-Q6_K.gguf path is permitted")
    return expected


def load_model_manifest(path: Path) -> list[ModelManifestEntry]:
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError("model manifest does not exist")
    try:
        text = path.read_text(encoding="utf-8")
        def reject_constant(value: str) -> Any:
            raise ValueError(f"non-finite manifest value: {value}")
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate manifest key: {key}")
                result[key] = value
            return result
        raw = json.loads(text, parse_constant=reject_constant,
                         object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("invalid model manifest JSON") from exc
    # A manifest may be either an envelope ({models: [...]}) or one entry.
    # Do not parse envelope metadata as a second model (the old behaviour did).
    # If an envelope declares its active model, validate it as policy input as
    # well; otherwise a stale/tampered top-level value could contradict the
    # pinned entry used by runtime verification and audit metadata.
    if isinstance(raw, dict) and ("model" in raw or "model_id" in raw):
        declared = raw.get("model_id", raw.get("model"))
        try:
            require_single_model(declared, field="manifest model")
        except ValueError as exc:
            raise ValueError("manifest model is outside the single-model policy") from exc
    if isinstance(raw, dict) and "models" in raw:
        entries = raw["models"]
    else:
        entries = [raw] if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries or len(entries) > 100:
        raise ValueError("invalid model manifest")
    out: list[ModelManifestEntry] = []
    seen_ids: set[str] = set(); seen_files: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("manifest entry must be object")
        capabilities = item.get("capabilities", ())
        if isinstance(capabilities, list): capabilities = tuple(capabilities)
        entry = ModelManifestEntry(
            id=item.get("id", item.get("model", "")),
            repository=item.get("repository", ""), revision=item.get("revision", ""),
            filename=item.get("filename", item.get("model", "")), sha256=item.get("sha256"),
            license=item.get("license", "unknown"), quantization=item.get("quantization", "unknown"),
            capabilities=capabilities, vram_hint=item.get("vram_hint", ""),
            downloadable=item.get("downloadable", False),
        )
        if entry.id in seen_ids or entry.filename in seen_files:
            raise ValueError("manifest contains duplicate model identity")
        seen_ids.add(entry.id); seen_files.add(entry.filename); out.append(entry)
    return out

def find_model_entry(entries: Iterable[ModelManifestEntry], model: str) -> ModelManifestEntry | None:
    """Resolve a model by id or filename without accepting arbitrary paths."""
    if not isinstance(model, str) or not model or Path(model).name != model:
        return None
    return next((entry for entry in entries if entry.id == model or entry.filename == model), None)

def validate_runtime_model(model: str, entries: Iterable[ModelManifestEntry], *, allowed_ids: Iterable[str] = (CANONICAL_MODEL_ID,)) -> ModelManifestEntry:
    entry = find_model_entry(entries, model)
    # ``allowed_ids`` is retained for callers that provide a narrower
    # deployment allowlist, but it must never widen the one-model runtime
    # policy.  Previously passing ``allowed_ids=("other",)`` bypassed it.
    allowed = set(allowed_ids)
    if CANONICAL_MODEL_ID not in allowed or entry is None or entry.id != CANONICAL_MODEL_ID:
        raise ValueError("model is not in the runtime allowlist")
    # The manifest itself is an untrusted control-plane input.  A canonical
    # ID must not be enough to redirect runtime verification to a different
    # repository, filename, revision, or digest, even when marked non-downloadable.
    if entry.id == CANONICAL_MODEL_ID and (
        entry.repository != CANONICAL_MODEL_REPOSITORY
        or entry.revision != CANONICAL_MODEL_REVISION
        or entry.filename != CANONICAL_MODEL_FILENAME
        or entry.sha256 != CANONICAL_MODEL_SHA256
    ):
        raise ValueError("canonical model manifest metadata is invalid")
    if not entry.downloadable:
        raise ValueError("canonical model manifest entry is not downloadable")
    entry.validate_download()
    return entry

@dataclass(frozen=True)
class CapabilityProfile:
    model_id: str
    mode: Mode
    max_tool_candidates: int = 6
    max_steps: int = 8
    temperature: float = 0.0
    supports_thinking: bool = False
    supports_vision: bool = False
    mtp: bool = False
    speculative: bool = False
    quality: float = 0.0
    latency_ms: float = 0.0
    vram_mb: float = 0.0
    verifier_rate: float = 0.0
    def __post_init__(self):
        object.__setattr__(self,"mode",Mode(self.mode))
        if type(self.max_tool_candidates) is not int or not 1<=self.max_tool_candidates<=100 or type(self.max_steps) is not int or not 1<=self.max_steps<=1000: raise ValueError("capability limits out of bounds")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or not math.isfinite(float(self.temperature)) or self.temperature<0 or self.temperature>2: raise ValueError("temperature out of bounds")
        for name in ("supports_thinking", "supports_vision", "mtp", "speculative"):
            if not isinstance(getattr(self, name), bool): raise ValueError(f"{name} must be boolean")
        for name in ("quality", "latency_ms", "vram_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0: raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.verifier_rate, (int, float)) or isinstance(self.verifier_rate, bool) or not math.isfinite(float(self.verifier_rate)) or not 0 <= self.verifier_rate <= 1: raise ValueError("verifier_rate must be between 0 and 1")

def canonical_capability_profile(mode: Mode) -> CapabilityProfile:
    """Return the bounded profile for the only permitted runtime model.

    The model is fixed, but workload mode still changes safe generation/tool
    limits. Keeping this in one factory prevents callers from accidentally
    constructing a THINK profile for FAST or RESEARCH and claiming that the
    mode was configured when it was not.
    """
    selected = Mode(mode)
    limits = {
        Mode.FAST: (4, 3, False),
        Mode.THINK: (6, 8, True),
        Mode.RESEARCH: (6, 16, True),
    }[selected]
    max_tools, max_steps, thinking = limits
    return CapabilityProfile(
        CANONICAL_MODEL_FILENAME,
        selected,
        max_tool_candidates=max_tools,
        max_steps=max_steps,
        supports_thinking=thinking,
        supports_vision=False,
        mtp=False,
        speculative=False,
    )


def select_capability(profiles: list[CapabilityProfile], mode: Mode, *, max_vram_mb: float | None = None, min_verifier_rate: float = 0.0) -> CapabilityProfile | None:
    if (isinstance(min_verifier_rate, bool) or not isinstance(min_verifier_rate, (int, float))
            or not math.isfinite(float(min_verifier_rate)) or not 0 <= min_verifier_rate <= 1):
        raise ValueError("min_verifier_rate must be between 0 and 1")
    if not isinstance(profiles, (list, tuple)) or any(not isinstance(profile, CapabilityProfile) for profile in profiles):
        raise ValueError("profiles must contain CapabilityProfile values")
    # Keep the filter explicit even though CapabilityProfile validates at
    # construction: mutable/foreign objects must never widen auto-selection.
    candidates=[]
    for profile in profiles:
        try:
            permitted = require_single_model(profile.model_id, field="capability profile model") == CANONICAL_MODEL_ID
        except ValueError:
            permitted = False
        if permitted and profile.mode == Mode(mode) and profile.verifier_rate >= min_verifier_rate and (max_vram_mb is None or profile.vram_mb <= max_vram_mb or profile.vram_mb == 0):
            candidates.append(profile)
    return sorted(candidates,key=lambda p:(-p.quality,-p.verifier_rate,p.latency_ms,p.vram_mb,p.model_id))[0] if candidates else None


@dataclass(frozen=True)
class OptimizationDecision:
    enabled: bool
    reason: str
    baseline_success: float = 0.0
    candidate_success: float = 0.0
    baseline_verifier_rate: float = 0.0
    candidate_verifier_rate: float = 0.0

def approve_speculative_variant(*, baseline_success: float, candidate_success: float, baseline_verifier_rate: float, candidate_verifier_rate: float, baseline_latency_ms: float | None = None, candidate_latency_ms: float | None = None, explicitly_requested: bool = False) -> OptimizationDecision:
    values=(baseline_success,candidate_success,baseline_verifier_rate,candidate_verifier_rate)
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not 0<=v<=1 for v in values): raise ValueError("evaluation rates must be between 0 and 1")
    if (baseline_latency_ms is None) != (candidate_latency_ms is None) or any(v is not None and (not isinstance(v,(int,float)) or isinstance(v,bool) or v < 0) for v in (baseline_latency_ms,candidate_latency_ms)): raise ValueError("latency measurements must be supplied together")
    if not explicitly_requested: return OptimizationDecision(False,"speculative decoding is opt-in")
    latency_ok=baseline_latency_ms is None or candidate_latency_ms <= baseline_latency_ms
    enabled=candidate_success >= baseline_success and candidate_verifier_rate >= baseline_verifier_rate and latency_ok
    reason="candidate does not regress validated quality or latency" if enabled else "candidate regresses validated quality, verifier rate, or latency"
    return OptimizationDecision(enabled,reason,baseline_success,candidate_success,baseline_verifier_rate,candidate_verifier_rate)

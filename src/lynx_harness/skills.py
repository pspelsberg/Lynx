from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import os
import tempfile
from pathlib import Path
import re
import math
from typing import Any, Awaitable, Callable
from .models import RiskLevel
from .verifier import _schema_match


@dataclass(frozen=True)
class Skill: name: str; version: int; description: str; status: str = "candidate"; risk: str = "read_only"

@dataclass(frozen=True)
class SkillManifest:
    name: str
    version: int
    description: str
    tools: tuple[str, ...] = ()
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    risk: RiskLevel = RiskLevel.READ_ONLY
    def __post_init__(self):
        if not self.name or len(self.name)>80 or not isinstance(self.version,int) or self.version<1: raise ValueError("invalid skill manifest")
        if len(self.tools)>100 or any(not isinstance(t,str) or not re.fullmatch(r"[A-Za-z0-9._-]+",t) for t in self.tools): raise ValueError("invalid skill tools")
        object.__setattr__(self,"risk", self.risk if isinstance(self.risk,RiskLevel) else RiskLevel[str(self.risk).upper()])

class SkillExecutor:
    """Only host-registered callables can execute a skill; model text is data."""
    def __init__(self, registry: SkillRegistry | None = None): self.registry, self._handlers = registry, {}
    def register(self, manifest: SkillManifest, handler: Callable[[dict[str, Any]], Any]) -> None:
        if not callable(handler): raise TypeError("skill handler must be callable")
        if manifest.name in self._handlers: raise ValueError("skill already registered")
        self._handlers[manifest.name]=(manifest,handler)
    async def execute(self, name: str, inputs: dict[str, Any], *, max_risk: RiskLevel = RiskLevel.READ_ONLY) -> Any:
        if name not in self._handlers: raise ValueError("unregistered skill")
        manifest,handler=self._handlers[name]
        if self.registry is not None:
            catalog = [item for item in self.registry.discover() if item.name == manifest.name and item.version == manifest.version]
            if not catalog or catalog[0].status not in {"candidate", "stable"}: raise PermissionError("skill lifecycle does not permit execution")
        try: max_risk = max_risk if isinstance(max_risk, RiskLevel) else RiskLevel[str(max_risk).upper()]
        except (KeyError, TypeError, ValueError) as exc: raise PermissionError("invalid skill risk policy") from exc
        if manifest.risk > max_risk: raise PermissionError("skill risk exceeds policy")
        if not isinstance(inputs,dict): raise ValueError("skill input must be an object")
        try: input_size = len(json.dumps(inputs, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as exc: raise ValueError("skill input is not valid JSON") from exc
        if input_size > 100_000: raise ValueError("skill input too large")
        errors=[]
        if manifest.inputs and not _schema_match(inputs,manifest.inputs,errors=errors): raise ValueError("skill inputs do not match manifest schema")
        result=handler(dict(inputs)); result=await result if hasattr(result,"__await__") else result
        try: output_size = len(json.dumps(result, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8"))
        except (TypeError, ValueError) as exc: raise ValueError("skill output is not valid JSON") from exc
        if output_size > 100_000: raise ValueError("skill output too large")
        errors=[]
        if manifest.outputs and not _schema_match(result,manifest.outputs,errors=errors): raise ValueError("skill output does not match manifest schema")
        return result


class SkillRegistry:
    """Versioned, non-executable skill catalog. Execution must be explicitly wired by the host."""
    def __init__(self, root: Path):
        if not isinstance(root, Path): raise ValueError("skill registry root must be a Path")
        absolute_root = root.absolute()
        if root.is_symlink() or any(parent.is_symlink() for parent in absolute_root.parents):
            raise ValueError("skill registry root and ancestors must not be symlinks")
        self.root=absolute_root; self.root.mkdir(parents=True,exist_ok=True); self._evaluations: dict[tuple[str,int], SkillEvaluation] = {}

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Persist catalog data without following a raced destination link."""
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("skill catalog parent must be a regular directory")
        fd, temporary_name = tempfile.mkstemp(prefix=".skill-", suffix=".tmp", dir=str(parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            try: temporary.unlink(missing_ok=True)
            except OSError: pass
            raise

    def _skill_dir(self, safe: str) -> Path:
        path = self.root / safe
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("skill catalog path must be a regular directory")
        path.mkdir(exist_ok=True)
        return path
    def _safe_name(self, name: str) -> str:
        safe=re.sub(r"[^a-z0-9_-]", "-", name.lower()).strip("-")[:80]
        if not safe or safe != name.lower(): raise ValueError("invalid skill name")
        return safe
    def discover(self) -> list[Skill]:
        result=[]
        for path in sorted(self.root.glob("*/skill.json")):
            if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
                continue
            data=json.loads(path.read_text(encoding="utf-8")); result.append(Skill(str(data["name"]),int(data.get("version",1)),str(data.get("description",""))[:500],str(data.get("status","candidate")),str(data.get("risk","read_only"))))
        return result
    def write_manifest(self, manifest: SkillManifest) -> Path:
        if not isinstance(manifest, SkillManifest): raise TypeError("expected SkillManifest")
        safe=self._safe_name(manifest.name); path=self._skill_dir(safe)
        target=path/"skill.json"; self._atomic_write(target, json.dumps({"name":safe,"version":manifest.version,"description":manifest.description[:500],"status":"candidate","risk":manifest.risk.name,"tools":list(manifest.tools),"inputs":manifest.inputs,"outputs":manifest.outputs}, indent=2)+"\n"); return target

    def write_candidate(self, name: str, description: str, version: int = 1, risk: str = "read_only") -> Path:
        safe=self._safe_name(name)
        path=self._skill_dir(safe); target=path/"skill.json"
        self._atomic_write(target, json.dumps(asdict(Skill(safe,version,description[:500],"candidate",risk)),indent=2)+"\n"); return target
    def promote(self, name: str, version: int, evaluation: SkillEvaluation | dict[str, Any]) -> Path:
        safe=self._safe_name(name); path=(self.root/safe/"skill.json").resolve()
        if self.root not in path.parents: raise ValueError("skill path escapes registry")
        if not path.is_file(): raise FileNotFoundError(name)
        data=json.loads(path.read_text(encoding="utf-8"));
        if int(data.get("version",0)) != version: raise ValueError("skill version mismatch")
        if data.get("status") != "candidate":
            raise ValueError("only candidate skills can be promoted")
        if not isinstance(evaluation, SkillEvaluation): raise ValueError("promotion requires replay-issued SkillEvaluation")
        issued=self._evaluations.get((safe,version))
        if issued != evaluation or not evaluation.passed: raise ValueError("skill evaluation is not replay-issued and passing")
        data["status"]="stable"; data["evaluation"]=asdict(evaluation); target=self.root / safe / "skill.json"; self._atomic_write(target, json.dumps(data,indent=2)+"\n"); return target


@dataclass(frozen=True)
class SkillEvaluation:
    name: str
    version: int
    passed: bool
    success_rate: float
    baseline_rate: float
    cases: int
    safety_rate: float = 1.0
    baseline_safety_rate: float = 1.0
    latency_ms: float = 0.0
    llm_tokens: int = 0
    def __post_init__(self):
        if not self.name or type(self.version) is not int or self.version < 1 or type(self.passed) is not bool or type(self.cases) is not int or self.cases < 1: raise ValueError("invalid skill evaluation")
        if (any(not isinstance(value,(int,float)) or isinstance(value,bool)
                or not math.isfinite(float(value)) or not 0 <= value <= 1
                for value in (self.success_rate,self.baseline_rate,self.safety_rate,self.baseline_safety_rate))
                or isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int,float))
                or not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0
                or type(self.llm_tokens) is not int or self.llm_tokens < 0):
            raise ValueError("invalid skill evaluation metrics")


class SkillEvolution:
    """Mine non-executable candidates from trajectories; promotion requires replay evidence."""
    def __init__(self, registry: SkillRegistry): self.registry=registry
    def mine(self, trajectory: list[dict[str, Any]], name: str, description: str = "") -> Path:
        tools=[]
        for item in trajectory:
            if not isinstance(item, dict) or item.get("type", item.get("kind")) != "tool_call":
                continue
            # Canonical trajectory events nest action fields in ``payload``;
            # accept the legacy flat form so old append-only records remain
            # mineable without treating arbitrary observation text as tools.
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else item
            tool = payload.get("tool") if isinstance(payload, dict) else None
            if tool:
                tools.append(str(tool)[:200])
        if not tools: raise ValueError("trajectory contains no tool pattern")
        return self.registry.write_candidate(name, description or "Pattern: " + " → ".join(tools), risk="read_only")
    def evaluate(self, name: str, version: int, outcomes: list[bool], baseline: list[bool], *, safety: list[bool] | None = None, baseline_safety: list[bool] | None = None, latency_ms: float = 0.0, llm_tokens: int = 0) -> SkillEvaluation:
        # Candidate and baseline must be paired over the same holdout cases;
        # otherwise a shorter/easier candidate sample could manufacture an
        # apparent improvement by changing the denominator.
        if (not outcomes or not baseline or len(outcomes) != len(baseline)
                or any(type(value) is not bool for value in outcomes + baseline)):
            raise ValueError("evaluation requires equal boolean outcomes and baseline")
        if safety is not None and (len(safety) != len(outcomes) or any(type(value) is not bool for value in safety)): raise ValueError("safety sample count mismatch")
        if baseline_safety is not None and (len(baseline_safety) != len(baseline) or any(type(value) is not bool for value in baseline_safety)): raise ValueError("baseline safety sample count mismatch")
        rate=sum(outcomes)/len(outcomes); base=sum(baseline)/len(baseline)
        safe=sum(safety)/len(safety) if safety is not None else 1.0; base_safe=sum(baseline_safety)/len(baseline_safety) if baseline_safety is not None else 1.0
        passed=rate > base and safe >= base_safe
        evaluation=SkillEvaluation(name,version,passed,rate,base,len(outcomes),safe,base_safe,latency_ms,llm_tokens)
        self.registry._evaluations[(self.registry._safe_name(name),version)]=evaluation
        return evaluation
    def promote_if_better(self, evaluation: SkillEvaluation) -> Path:
        if not evaluation.passed: raise ValueError("candidate did not beat baseline")
        return self.registry.promote(evaluation.name,evaluation.version,evaluation)

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from enum import StrEnum
import json
import weakref
from pathlib import Path
from typing import Any, Callable, Protocol

from .models import AgentTask, ExpectedOutput, Postcondition, RiskLevel, TaskContract, DecisionType
from .model_profiles import backend_model_identity

class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"

@dataclass(frozen=True)
class CheckResult:
    id: str
    status: VerificationStatus
    reason: str = ""
    artifacts: tuple[str, ...] = ()

    @property
    def passed(self) -> bool: return self.status == VerificationStatus.PASSED

_ISSUED_VERIFICATIONS: weakref.WeakSet["VerificationResult"] = weakref.WeakSet()
_VERIFICATION_BINDINGS: weakref.WeakKeyDictionary["VerificationResult", dict[str, str]] = weakref.WeakKeyDictionary()

def verification_binding(value: Any) -> dict[str, str] | None:
    """Return the verifier-owned subject binding, if this is an issued result."""
    try:
        binding = _VERIFICATION_BINDINGS.get(value)
    except (TypeError, ValueError):
        return None
    return dict(binding) if binding is not None else None


def is_issued_verification(value: Any) -> bool:
    """Return whether this result was minted by a Verifier instance."""
    try:
        return isinstance(value, VerificationResult) and value in _ISSUED_VERIFICATIONS
    except (TypeError, ValueError):
        # Malformed caller-created dataclasses must fail closed, not turn the
        # integrity gate into an exception-based denial of service.
        return False


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    checks: tuple[CheckResult, ...] = ()
    artifacts: tuple[str, ...] = ()
    reason: str = ""

    @property
    def passed(self) -> bool: return self.status == VerificationStatus.PASSED

class VerificationCheck(Protocol):
    id: str
    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any]) -> CheckResult: ...

def _value(candidate: Any) -> Any:
    if is_dataclass(candidate): return asdict(candidate)
    return candidate

def _schema_match(value: Any, schema: dict[str, Any], path: str = "$", errors: list[str] | None = None) -> bool:
    errors = errors if errors is not None else []
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool), "number": isinstance(value, (int, float)) and not isinstance(value, bool), "boolean": isinstance(value, bool), "null": value is None}
    if kind in valid and not valid[kind]: errors.append(f"{path} is not {kind}"); return False
    if "enum" in schema and value not in schema["enum"]: errors.append(f"{path} is not allowed"); return False
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list): errors.append(f"{path}.required invalid"); return False
        for key in required:
            if key not in value: errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict): errors.append(f"{path}.properties invalid"); return False
        if schema.get("additionalProperties") is False and any(k not in properties for k in value): errors.append(f"{path} has unknown fields")
        for key, child in properties.items():
            if key in value and isinstance(child, dict): _schema_match(value[key], child, f"{path}.{key}", errors)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value): _schema_match(item, schema["items"], f"{path}[{i}]", errors)
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]: errors.append(f"{path} is too long")
    if isinstance(value, (dict, list)) and isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]: errors.append(f"{path} has too many items")
    return not errors

class SyntaxCheck:
    id = "syntax"
    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any]) -> CheckResult:
        try: encoded = json.dumps(_value(candidate), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc: return CheckResult(self.id, VerificationStatus.FAILED, f"candidate is not JSON: {exc}")
        if len(encoded.encode()) > contract.expected_output.max_chars * 8: return CheckResult(self.id, VerificationStatus.FAILED, "candidate payload is too large")
        answer = _value(candidate).get("answer") if isinstance(_value(candidate), dict) else (_value(candidate) if isinstance(_value(candidate), str) else None)
        if isinstance(answer, str) and len(answer) > contract.expected_output.max_chars: return CheckResult(self.id, VerificationStatus.FAILED, "answer exceeds expected_output.max_chars")
        return CheckResult(self.id, VerificationStatus.PASSED, "candidate is valid JSON")

class SchemaCheck:
    id = "schema"
    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any]) -> CheckResult:
        schema = contract.expected_output.schema
        if schema is None: return CheckResult(self.id, VerificationStatus.PASSED, "no output schema requested")
        if isinstance(schema, str):
            schemas = context.get("schemas", {})
            schema = schemas.get(schema)
            if not isinstance(schema, dict): return CheckResult(self.id, VerificationStatus.INCOMPLETE, "schema registry entry is unavailable")
        errors: list[str] = []
        ok = _schema_match(_value(candidate), schema, errors=errors)
        return CheckResult(self.id, VerificationStatus.PASSED if ok else VerificationStatus.FAILED, "; ".join(errors[:5]) or "schema matched")

class PolicyCheck:
    id = "policy"
    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any]) -> CheckResult:
        tools = context.get("tools_used", ())
        unknown = [tool for tool in tools if tool not in contract.allowed_tools]
        if unknown: return CheckResult(self.id, VerificationStatus.FAILED, f"tools outside contract: {unknown[:3]}")
        risk = context.get("max_risk", RiskLevel.READ_ONLY)
        try: risk = risk if isinstance(risk, RiskLevel) else (RiskLevel(risk) if isinstance(risk, int) and not isinstance(risk, bool) else RiskLevel[risk.upper()])
        except (KeyError, AttributeError): return CheckResult(self.id, VerificationStatus.FAILED, "invalid observed risk")
        if risk > contract.security.max_risk: return CheckResult(self.id, VerificationStatus.FAILED, "observed risk exceeds contract")
        return CheckResult(self.id, VerificationStatus.PASSED, "policy satisfied")

class PostconditionCheck:
    id = "postcondition"
    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any]) -> CheckResult:
        output = _value(candidate); failures=[]; refs=[]
        for pc in contract.postconditions:
            if pc.name == "output_bounded":
                answer = output.get("answer", output) if isinstance(output, dict) else output
                if not isinstance(answer, str) or len(answer) > contract.expected_output.max_chars: failures.append("output is not bounded")
            elif pc.name == "artifact_exists":
                artifact_refs = output.get("artifacts", output.get("artifact_refs", ())) if isinstance(output, dict) else ()
                if not artifact_refs:
                    artifact_refs = context.get("artifact_refs", ())
                if not artifact_refs: failures.append("artifact_exists requires at least one artifact")
                for ref in artifact_refs:
                    store = context.get("artifacts")
                    try:
                        if not store: failures.append(f"missing artifact: {ref}")
                        elif hasattr(store, "metadata"):
                            metadata = store.metadata(ref)
                            expected_branch = context.get("branch_id", getattr(contract, "branch_id", None))
                            if expected_branch is not None and metadata.branch_id != expected_branch:
                                failures.append(f"artifact belongs to another branch: {ref}")
                            else:
                                refs.append(ref)
                        elif not store.get(ref): failures.append(f"missing artifact: {ref}")
                        else: refs.append(ref)
                    except (OSError, ValueError): failures.append(f"missing artifact: {ref}")
            elif pc.name == "no_external_mutation":
                if context.get("external_mutations"): failures.append("external mutation detected")
            else:
                plugin = context.get("postcondition_checks", {}).get(pc.name)
                if plugin is None:
                    if pc.required: failures.append(f"unknown postcondition: {pc.name}")
                else:
                    try:
                        outcome = plugin(output, pc.arguments) if callable(plugin) else bool(plugin)
                        if not outcome: failures.append(f"postcondition failed: {pc.name}")
                    except Exception as exc: failures.append(f"postcondition failed: {pc.name}: {exc}")
        return CheckResult(self.id, VerificationStatus.FAILED if failures else VerificationStatus.PASSED, "; ".join(failures) or "postconditions satisfied", tuple(refs))

class EvidenceCheck:
    id = "evidence"
    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any]) -> CheckResult:
        if not context.get("evidence_required"): return CheckResult(self.id, VerificationStatus.PASSED, "evidence not required")
        claims = context.get("claims", [])
        sources = {item.get("source_id") for item in context.get("sources", []) if isinstance(item, dict)}
        store = context.get("artifacts")
        missing = []
        for claim in claims:
            # Evidence presence alone is not claim verification.  A claim
            # must carry an explicit deterministic verification state before
            # it can be promoted or satisfy an evidence-required contract.
            if not isinstance(claim, dict) or claim.get("verification") not in {"passed", "verified"}:
                missing.append(claim); continue
            source_refs = claim.get("source_refs", [])
            evidence_refs = claim.get("evidence_refs", [])
            valid_source = any(ref in sources for ref in source_refs) if sources else False
            valid_artifact = False
            for ref in evidence_refs:
                try:
                    if store and isinstance(ref, str) and ref.startswith("artifact://"):
                        metadata = store.metadata(ref)
                        expected_branch = context.get("branch_id", getattr(contract, "branch_id", None))
                        if expected_branch is None or metadata.branch_id == expected_branch:
                            valid_artifact = True
                except (OSError, ValueError): pass
            if not (valid_source or valid_artifact): missing.append(claim)
        return CheckResult(self.id, VerificationStatus.FAILED if missing else VerificationStatus.PASSED, "claims lack resolvable evidence" if missing else "claims have resolvable evidence")

class Verifier:
    def __init__(self, checks: list[VerificationCheck] | None = None):
        self.checks = list(checks or [SyntaxCheck(), SchemaCheck(), PolicyCheck(), PostconditionCheck(), EvidenceCheck()])
        if any(not isinstance(check.id, str) or not check.id or len(check.id)>100 for check in self.checks): raise ValueError("verification check IDs must be bounded")
        if len({check.id for check in self.checks}) != len(self.checks): raise ValueError("verification check IDs must be unique")

    def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any] | None = None) -> VerificationResult:
        context = dict(context or {}); results=[]
        for check in self.checks:
            try: result = check.check(candidate, contract, context)
            except Exception as exc: result = CheckResult(check.id, VerificationStatus.FAILED, f"check failed closed: {exc}")
            results.append(result)
        status = VerificationStatus.FAILED if any(r.status == VerificationStatus.FAILED for r in results) else (VerificationStatus.INCOMPLETE if any(r.status == VerificationStatus.INCOMPLETE for r in results) else VerificationStatus.PASSED)
        refs = tuple(ref for result in results for ref in result.artifacts)
        verification = VerificationResult(status, tuple(results), refs, "; ".join(r.reason for r in results if r.status != VerificationStatus.PASSED)[:2_000])
        # Durable promotion/export APIs must be able to distinguish a real
        # verifier result from a caller-constructed lookalike dataclass.
        _ISSUED_VERIFICATIONS.add(verification)
        # Bind only the serializable answer projection. A malformed candidate
        # must produce a failed/incomplete result, not crash the verifier while
        # it records provenance for the result.
        candidate_value = _value(candidate)
        bound_answer = candidate_value.get("answer", "") if isinstance(candidate_value, dict) else ""
        _VERIFICATION_BINDINGS[verification] = {"answer": str(bound_answer)[:20_000]}
        return verification


class RepairLoop:
    """Bounded repair callback. It never receives or changes tool policy."""
    def __init__(self, verifier: Verifier, max_attempts: int = 2):
        if not 0 <= max_attempts <= 10: raise ValueError("repair attempts out of bounds")
        self.verifier, self.max_attempts = verifier, max_attempts
    async def run(self, candidate: Any, contract: TaskContract, repair: Callable[[Any, VerificationResult], Any], context: dict[str, Any] | None = None, *, timeout_s: float | None = None, cancellation: Any | None = None) -> tuple[Any, VerificationResult]:
        result=self.verifier.check(candidate, contract, context)
        for _ in range(self.max_attempts):
            if result.status == VerificationStatus.PASSED: break
            candidate=repair(candidate, result)
            if hasattr(candidate, "__await__"):
                if timeout_s is not None:
                    if not isinstance(timeout_s, (int, float)) or timeout_s <= 0: raise ValueError("repair timeout is invalid")
                    from .cancellation import bounded
                    candidate = await bounded(candidate, timeout_s, cancellation)
                else: candidate=await candidate
            result=self.verifier.check(candidate, contract, context)
        return candidate, result


class IndependentSemanticVerifier:
    """Optional critic; deterministic checks remain the security boundary."""
    def __init__(self, backend: Any, max_tokens: int = 256):
        if not hasattr(backend, "decide"): raise TypeError("backend must implement decide")
        if not 1 <= max_tokens <= 2_000: raise ValueError("critic token limit out of bounds")
        self.backend,self.max_tokens=backend,max_tokens
        self.model_id = backend_model_identity(backend, field="semantic verifier model")
    async def assess(self, candidate: Any, contract: TaskContract, context: dict[str, Any] | None = None) -> CheckResult:
        if backend_model_identity(self.backend, field="semantic verifier model") != self.model_id:
            raise ValueError("semantic verifier backend identity changed")
        # The critic receives a bounded projection and can only return a boolean decision.
        prompt={"candidate":str(_value(candidate))[:20_000],"contract":contract.to_dict(),"instruction":"Return a structured finish decision; do not reveal reasoning."}
        decision=await self.backend.decide({"goal":"independent semantic assessment","mode":"fast","critic_input":prompt},[],self.max_tokens)
        passed=bool(getattr(decision,"type",None) and str(getattr(decision,"type")) in {"finish","DecisionType.FINISH"} and str(getattr(decision,"answer","")).strip().lower() in {"yes","pass","passed","true"})
        return CheckResult("semantic-independent",VerificationStatus.PASSED if passed else VerificationStatus.FAILED,"independent assessment passed" if passed else "independent assessment failed")


class IndependentModelVerifier:
    """Optional semantic critic. It can reject/flag a candidate, never authorize policy."""
    def __init__(self, backend: Any, *, max_tokens: int = 512):
        if not hasattr(backend, "decide") or type(max_tokens) is not int or not 1 <= max_tokens <= 4_096: raise ValueError("invalid model verifier")
        self.model_id = backend_model_identity(backend, field="independent verifier model")
        self.backend=backend; self.max_tokens=max_tokens

    async def check(self, candidate: Any, contract: TaskContract, context: dict[str, Any] | None = None) -> VerificationResult:
        context=dict(context or {})
        try:
            if backend_model_identity(self.backend, field="independent verifier model") != self.model_id:
                return VerificationResult(VerificationStatus.INCOMPLETE, (), (), "independent verifier identity changed")
        except ValueError:
            return VerificationResult(VerificationStatus.INCOMPLETE, (), (), "independent verifier identity unavailable")
        deterministic=Verifier().check(candidate,contract,context)
        if not deterministic.passed: return deterministic
        prompt={"candidate":_value(candidate),"contract":{"id":contract.id,"goal":contract.goal,"expected_output":asdict(contract.expected_output)},"instruction":"Judge semantic completeness only. Return finish answer PASS or FAIL. Do not authorize tools or policies."}
        try:
            decision=await self.backend.decide({"mode":"think","goal":"Critique the candidate without changing policy.","trusted_state":prompt,"untrusted_observations":[]},[],self.max_tokens)
        except Exception as exc:
            return VerificationResult(VerificationStatus.INCOMPLETE,deterministic.checks,deterministic.artifacts,"independent verifier unavailable")
        answer=(decision.answer or "").strip().upper() if getattr(decision,"type",None) == DecisionType.FINISH else ""
        model_check=CheckResult("independent_model_semantics",VerificationStatus.PASSED if answer == "PASS" else VerificationStatus.FAILED,"independent critic accepted" if answer == "PASS" else "independent critic rejected")
        status=VerificationStatus.PASSED if model_check.passed else VerificationStatus.FAILED
        return VerificationResult(status,deterministic.checks+(model_check,),deterministic.artifacts,model_check.reason)

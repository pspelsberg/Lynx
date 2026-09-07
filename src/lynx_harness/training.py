from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import time
from pathlib import Path
from typing import Any
from .security import redact, atomic_write_text
from .models import Decision, DecisionType, TaskContract, RiskLevel
from .verifier import Verifier, VerificationStatus, VerificationResult, is_issued_verification, verification_binding
from .model_profiles import CANONICAL_MODEL_ID, CANONICAL_MODEL_FILENAME, backend_model_identity, require_single_model

MAX_TRAINING_EXAMPLE_BYTES = 1_000_000
MAX_TRAINING_DATASET_BYTES = 50_000_000
_TRAINING_EXPORT_CAPABILITY = object()

@dataclass(frozen=True)
class TrainingExample:
    state: dict[str,Any]
    action: dict[str,Any]
    outcome: str
    verification: dict[str,Any]
    _export_capability: object | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_verifier(cls, state: dict[str, Any], action: dict[str, Any],
                      verification: VerificationResult, *, run_id: str) -> "TrainingExample":
        """Create an exportable example only from a real passed verifier."""
        if (not isinstance(verification, VerificationResult)
                or not is_issued_verification(verification)
                or not verification.passed):
            raise ValueError("training example requires a passed verifier result")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("training example requires a run id")
        return cls(state, action, "success", {
            "passed": True, "event_seq": 1, "run_id": run_id,
            "checks": [asdict(check) for check in verification.checks],
        }, _export_capability=_TRAINING_EXPORT_CAPABILITY)

class TrajectoryDataset:
    """Export only bounded, successful, verifier-approved trajectories for later LoRA work.

    A dataset carries its model identity.  This prevents a future trainer from
    combining examples produced by a forbidden checkpoint under a canonical
    filename or from treating an unlabelled dataset as safe.
    """
    def __init__(self, max_examples: int=100_000, *, model: str = CANONICAL_MODEL_ID):
        if not isinstance(max_examples,int) or not 1<=max_examples<=1_000_000: raise ValueError("invalid dataset limit")
        self.model = require_single_model(model, field="dataset model")
        self.max_examples=max_examples; self.examples:list[TrainingExample]=[]
    def add_run(self, events: list[dict[str,Any]], verified: VerificationResult | bool = False) -> int:
        if not isinstance(events, list) or len(events) > 100_000:
            raise ValueError("trajectory events must be a bounded list")
        # A boolean flag is not evidence. Require the actual object minted by a
        # Verifier so serialized/caller-constructed lookalikes cannot pass.
        if not isinstance(verified, VerificationResult) or not is_issued_verification(verified) or not verified.passed:
            return 0
        if len(self.examples)>=self.max_examples: raise ValueError("dataset limit reached")
        decisions=[e for e in events if isinstance(e, dict) and e.get("type")=="decision"]
        start=next((e for e in events if isinstance(e, dict) and e.get("type")=="run_started"),None)
        finish=next((e for e in reversed(decisions) if isinstance(e.get("payload"), dict) and e["payload"].get("type")=="finish"),None)
        verification=next((e for e in reversed(events) if isinstance(e, dict) and e.get("type")=="verification"),None)
        finished=next((e for e in reversed(events) if isinstance(e, dict) and e.get("type")=="run_finished"),None)
        # A verifier event by itself is not a successful trajectory: require
        # the terminal success event and complete event lineage before export.
        if (not isinstance(start, dict) or type(start.get("run_id")) is not str or not start.get("run_id") or not finish
                or not isinstance(verification, dict)
                or not isinstance(finished, dict)
                or not isinstance(verification.get("payload"), dict)
                or verification["payload"].get("status") not in {"passed", "verified"}
                or not isinstance(finished.get("payload"), dict)
                or finished["payload"].get("status") not in {"success", "verified"}
                or any(event.get("run_id") != start.get("run_id")
                       or type(event.get("seq")) is not int or event["seq"] <= 0
                       or not isinstance(event.get("payload"), dict)
                       for event in decisions)
                or verification.get("run_id") != start.get("run_id")
                or finished.get("run_id") != start.get("run_id")
                or type(verification.get("seq")) is not int or verification["seq"] <= 0
                or type(finished.get("seq")) is not int or finished["seq"] <= 0
                or finish is not decisions[-1]
                or verification["seq"] <= finish["seq"]
                or finished["seq"] <= verification["seq"]):
            return 0
        verification_payload=verification.get("payload",{})
        binding = verification_binding(verified)
        finish_answer = str(finish.get("payload", {}).get("answer", ""))[:20_000]
        if binding is None or binding.get("answer") != finish_answer:
            return 0
        try:
            # JSONL turns tuples/enums into lists/strings; compare the exact
            # canonical JSON projection, not Python container representation.
            expected_verification = json.loads(json.dumps(asdict(verified), default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            return 0
        # Runtime provenance fields may be appended by the recorder; the
        # verifier-authenticated core must match exactly.
        core = {key: verification_payload.get(key) for key in expected_verification}
        try:
            core = json.loads(json.dumps(core, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            return 0
        if core != expected_verification:
            return 0
        start_payload=start.get("payload")
        if not isinstance(start_payload, dict):
            return 0
        # Events may carry runtime provenance.  If present it must agree with
        # the dataset identity; absent provenance remains valid only because
        # this is an explicitly canonical, offline dataset gate.
        event_model = start_payload.get("model")
        if event_model is not None:
            try:
                if require_single_model(event_model, field="trajectory model") != self.model:
                    return 0
            except ValueError:
                return 0
        checks = core.get("checks") if isinstance(core, dict) else None
        if not isinstance(checks, list) or not checks or any(not isinstance(check, dict) or not check.get("id") or check.get("status") not in {"passed", "verified"} for check in checks): return 0
        base={"run_id":start.get("run_id"),"goal":str(start_payload.get("goal", ""))[:20_000],"mode":start_payload.get("mode", "think"),"model_id":self.model}
        # Keep each action paired only with observations that happened before
        # it.  Exporting the complete run for every decision leaks future tool
        # output into earlier training states and makes trajectories unusable
        # as state/action examples.
        observations: list[tuple[int, dict[str, Any]]] = []
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "observation":
                continue
            if (event.get("run_id") != start.get("run_id")
                    or type(event.get("seq")) is not int or event.get("seq") <= 0):
                return 0
            observations.append((event["seq"], redact(event.get("payload", {}))))
        added=0
        for decision in decisions:
            if len(self.examples)>=self.max_examples: break
            payload=decision.get("payload", {})
            kind=str(payload.get("type", ""))
            action={"type":kind}
            if kind=="tool": action.update({"tool":str(payload.get("tool", ""))[:200],"arguments":payload.get("arguments",{})})
            elif kind=="finish": action["answer"]=str(payload.get("answer", ""))[:20_000]
            else: continue
            decision_seq=decision["seq"]
            prior_observations=[item for seq, item in observations if seq < decision_seq]
            self.examples.append(TrainingExample(redact(dict(base,observations=prior_observations)),redact(action),"success",{"passed":True,"event_seq":decision_seq,"run_id":decision.get("run_id"),"checks":redact(checks)}, _export_capability=_TRAINING_EXPORT_CAPABILITY)); added+=1
        return added

    def write_jsonl(self,path: Path)->Path:
        """Write only verifier-approved examples, with a final redaction pass.

        ``examples`` is intentionally inspectable for analysis, so callers
        could otherwise append a forged/unredacted row and bypass the export
        gate used by :func:`write_verified_lora_dataset`.
        """
        _verified_examples(self)
        path = Path(path)
        encoded = "".join(json.dumps(redact(_example_dict(example)), ensure_ascii=False, allow_nan=False) + "\n" for example in self.examples)
        return atomic_write_text(path, encoded, max_bytes=MAX_TRAINING_DATASET_BYTES)


@dataclass(frozen=True)
class TeacherComparison:
    teacher: str
    student: str
    cases: int
    teacher_validated: int
    student_validated: int
    improvement: float

def compare_teacher_outcomes(teacher: list[bool], student: list[bool], *, teacher_name: str = "teacher", student_name: str = "student") -> TeacherComparison:
    if not teacher or not student or len(teacher)!=len(student) or len(teacher) > 100_000: raise ValueError("teacher and student holdout sets must have equal bounded non-zero length")
    if any(not isinstance(x,bool) for x in teacher+student): raise ValueError("outcomes must be booleans")
    tv,sv=sum(teacher),sum(student)
    return TeacherComparison(teacher_name,student_name,len(teacher),tv,sv,(sv-tv)/len(teacher))


@dataclass(frozen=True)
class AblationOutcome:
    variant: str
    holdout_success: float
    baseline_success: float
    safety_rate: float
    passed: bool

def evaluate_lora_ablation(variant: str, holdout: list[bool], baseline: list[bool], safety: list[bool], *, base_model: str = CANONICAL_MODEL_ID) -> AblationOutcome:
    require_single_model(base_model, field="LoRA base model")
    if not isinstance(variant, str) or not variant.strip() or len(variant) > 200:
        raise ValueError("ablation variant is invalid")
    if not holdout or len(holdout) > 100_000 or len(holdout)!=len(baseline) or len(holdout)!=len(safety): raise ValueError("ablation samples must have equal bounded non-zero length")
    if any(not isinstance(v,bool) for v in holdout+baseline+safety): raise ValueError("ablation outcomes must be booleans")
    score=sum(holdout)/len(holdout); base=sum(baseline)/len(baseline); safe=sum(safety)/len(safety)
    return AblationOutcome(variant,score,base,safe,score>base and safe>=1.0)


@dataclass(frozen=True)
class TeacherCase:
    """A bounded, repeatable case shared by teacher and student backends."""
    id: str
    context: dict[str, Any]
    tools: tuple[dict[str, Any], ...] = ()
    contract: TaskContract | None = None
    max_tokens: int = 512
    def __post_init__(self):
        if not self.id or len(self.id)>128 or not isinstance(self.context,dict) or len(json.dumps(self.context,ensure_ascii=False))>100_000: raise ValueError("invalid teacher case")
        if len(self.tools)>100 or any(not isinstance(tool,dict) or len(json.dumps(tool))>20_000 for tool in self.tools): raise ValueError("invalid teacher tools")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 16_384: raise ValueError("invalid teacher token limit")

@dataclass(frozen=True)
class TeacherEvaluation:
    schema: str
    teacher: TeacherComparison
    rows: tuple[dict[str, Any], ...]
    raw_reasoning_stored: bool = False
    comparison_kind: str = "same_model_control"

class TeacherStudentEvaluator:
    """Run identical holdout cases against two backends without retaining CoT."""
    def __init__(self, teacher_backend: Any, student_backend: Any):
        if not hasattr(teacher_backend,"decide") or not hasattr(student_backend,"decide"): raise TypeError("backends must implement decide")
        # The 27B teacher requested by the original task is intentionally not
        # available.  Refuse unknown/other model identities instead of making
        # a same-model run look like a teacher comparison.
        self.teacher_model = backend_model_identity(teacher_backend, field="teacher model")
        self.student_model = backend_model_identity(student_backend, field="student model")
        if self.teacher_model != self.student_model:
            raise ValueError("teacher and student must use the same permitted model")
        self.teacher_backend=teacher_backend; self.student_backend=student_backend

    @staticmethod
    def _candidate(decision: Decision) -> Any:
        return decision

    @staticmethod
    async def _run(backend: Any, case: TeacherCase) -> tuple[bool, str, float]:
        started=time.perf_counter()
        try:
            decision=await backend.decide(dict(case.context), [dict(tool) for tool in case.tools], case.max_tokens)
            if not isinstance(decision,Decision): return False,"invalid_decision",(time.perf_counter()-started)*1000
            if case.contract is None: return True,decision.type.value,(time.perf_counter()-started)*1000
            tools_used=(decision.tool,) if decision.type == DecisionType.TOOL and decision.tool else ()
            result=Verifier().check(TeacherStudentEvaluator._candidate(decision),case.contract,{"tools_used":tools_used,"max_risk":RiskLevel.READ_ONLY})
            return result.status == VerificationStatus.PASSED, result.status.value, (time.perf_counter()-started)*1000
        except Exception: return False,"error",(time.perf_counter()-started)*1000

    async def evaluate(self, cases: list[TeacherCase]) -> TeacherEvaluation:
        if not cases or len(cases)>10_000: raise ValueError("teacher holdout must be bounded and non-empty")
        # Backend objects are mutable integration seams. Re-check identity at
        # evaluation time so a caller cannot swap in another checkpoint after
        # constructor validation.
        if (backend_model_identity(self.teacher_backend, field="teacher model") != self.teacher_model
                or backend_model_identity(self.student_backend, field="student model") != self.student_model):
            raise ValueError("teacher/student backend identity changed")
        teacher_outcomes=[]; student_outcomes=[]; rows=[]
        for case in cases:
            teacher_ok,teacher_status,teacher_ms=await self._run(self.teacher_backend,case)
            student_ok,student_status,student_ms=await self._run(self.student_backend,case)
            teacher_outcomes.append(teacher_ok); student_outcomes.append(student_ok)
            rows.append({"case_id":case.id,"model_id":self.student_model,"teacher_model_id":self.teacher_model,"student_model_id":self.student_model,"comparison_kind":"same_model_control","teacher_validated":teacher_ok,"student_validated":student_ok,"teacher_status":teacher_status,"student_status":student_status,"teacher_latency_ms":round(teacher_ms,3),"student_latency_ms":round(student_ms,3)})
        return TeacherEvaluation("lynx.teacher-evaluation.v1",compare_teacher_outcomes(teacher_outcomes,student_outcomes),tuple(rows),False,"same_model_control")


@dataclass(frozen=True)
class LoRAExperimentSpec:
    name: str
    base_model: str
    seed: int = 0
    target: str = "structured_actions"
    def __post_init__(self):
        if not self.name or len(self.name)>100 or not self.base_model or len(self.base_model)>500: raise ValueError("invalid LoRA experiment")
        require_single_model(self.base_model, field="LoRA base model")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**31-1: raise ValueError("invalid LoRA seed")
        if self.target not in {"structured_actions","tool_arguments","observation_classification","state_actions"}: raise ValueError("unsupported LoRA target")

def _example_dict(example: TrainingExample) -> dict[str, Any]:
    return {"state": example.state, "action": example.action, "outcome": example.outcome, "verification": example.verification}


def _verified_examples(dataset: TrajectoryDataset) -> list[TrainingExample]:
    """Validate the immutable export conditions shared by all writers."""
    if not isinstance(dataset, TrajectoryDataset) or not dataset.examples:
        raise ValueError("a non-empty verified trajectory dataset is required")
    require_single_model(dataset.model, field="dataset model")
    total_bytes = 0
    for example in dataset.examples:
        checks = (example.verification.get("checks")
                  if isinstance(example, TrainingExample) and isinstance(example.verification, dict)
                  else None)
        if (not isinstance(example, TrainingExample)
                or not isinstance(example.state, dict)
                or not isinstance(example.action, dict)
                or example.outcome != "success"
                or not isinstance(example.verification, dict)
                or example.verification.get("passed") is not True
                or type(example.verification.get("event_seq")) is not int
                or example.verification.get("event_seq") <= 0
                or type(example.verification.get("run_id")) is not str
                or not example.verification.get("run_id")
                or not isinstance(checks, list)
                or not checks
                or any(not isinstance(check, dict)
                       or check.get("status") not in {"passed", "verified"}
                       or not check.get("id") for check in checks)):
            raise ValueError("dataset contains an unverified example")
        # A forged row must not relabel another checkpoint as canonical merely
        # because its enclosing dataset has the canonical model identity.
        for key in ("model", "model_id"):
            if key in example.state:
                try:
                    if require_single_model(example.state[key], field=f"dataset example {key}") != dataset.model:
                        raise ValueError("dataset example model identity mismatch")
                except ValueError as exc:
                    raise ValueError("dataset example model identity mismatch") from exc
        state_run_id = example.state.get("run_id")
        if state_run_id is not None and state_run_id != example.verification.get("run_id"):
            raise ValueError("dataset example run identity mismatch")
        try:
            encoded = json.dumps(redact(_example_dict(example)), ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("dataset example is not serializable") from exc
        if len(encoded) > MAX_TRAINING_EXAMPLE_BYTES:
            raise ValueError("dataset example is too large")
        total_bytes += len(encoded) + 1
        if total_bytes > MAX_TRAINING_DATASET_BYTES:
            raise ValueError("training dataset is too large")
    return list(dataset.examples)


def write_verified_lora_dataset(dataset: TrajectoryDataset, path: Path) -> Path:
    """Write only examples carrying an explicit passed verifier record.

    This is a data gate, not a trainer: no raw observations or unverified web
    text can silently enter a future LoRA job.
    """
    examples = _verified_examples(dataset)
    path = Path(path)
    encoded = "".join(json.dumps(redact(_example_dict(example)), ensure_ascii=False, allow_nan=False) + "\n" for example in examples)
    return atomic_write_text(path, encoded, max_bytes=MAX_TRAINING_DATASET_BYTES)

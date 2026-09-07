from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
import re
import math

from .loop import AgentRunner
from .models import ComputeBudget, Mode, TaskContract, ExpectedOutput
from .model_profiles import CANONICAL_MODEL_ID, require_single_model, backend_model_identity
from .security import redact

VARIANTS = ("bare_llm", "normal_context", "+tool_router", "+structured_state", "+verification", "rlm", "rah", "rah+verifier")


@dataclass(frozen=True)
class EvalCase:
    id: str
    goal: str
    expected: str = ""
    mode: Mode = Mode.FAST
    expected_tool: str = ""
    expected_tools: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    expected_status: str = ""
    family: str = "general"
    # Optional expected identifiers make citation/coverage metrics semantic
    # rather than a proxy for punctuation or answer length.
    expected_citations: tuple[str, ...] = ()
    expected_claims: tuple[str, ...] = ()


def load_benchmark(path: Path) -> list[EvalCase]:
    """Load a bounded benchmark without parser ambiguity or executable tags.

    Benchmark files are policy/evaluation inputs.  In particular, accepting
    duplicate keys would let a later value silently replace a checked budget
    or expectation, while YAML aliases can create cyclic structures.  Keep the
    strictness here (rather than relying on whichever parser happens to be
    installed) just as contract loading is strict.
    """
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError("benchmark file does not exist")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("invalid benchmark document") from exc
    if len(raw.encode("utf-8")) > 10_000_000:
        raise ValueError("benchmark document is too large")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate benchmark key: {key}")
            result[key] = value
        return result

    try:
        # JSON has a strict parser available in the standard library.  Use it
        # for .json files so JSON NaN/duplicate-key extensions cannot be
        # accepted merely because PyYAML is installed.
        if path.suffix.lower() == ".json":
            document = json.loads(raw, parse_constant=reject_constant,
                                  object_pairs_hook=reject_duplicates)
        else:
            import yaml
            class _StrictSafeLoader(yaml.SafeLoader):
                pass

            def construct_mapping(loader, node, deep=False):
                mapping: dict[Any, Any] = {}
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=deep)
                    try:
                        if key in mapping:
                            raise ValueError(f"duplicate benchmark key: {key}")
                        mapping[key] = loader.construct_object(value_node, deep=deep)
                    except TypeError as exc:
                        raise ValueError("benchmark mapping keys must be scalar") from exc
                return mapping

            _StrictSafeLoader.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
            document = yaml.load(raw, Loader=_StrictSafeLoader)

        # Reject YAML's non-standard NaN/Infinity values and recursive aliases
        # before case validation.  Cycles would otherwise make downstream
        # traversal unbounded.
        active: set[int] = set()
        def validate_values(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("benchmark values must be finite")
            if isinstance(value, (dict, list, tuple)):
                marker = id(value)
                if marker in active:
                    raise ValueError("benchmark aliases may not be recursive")
                active.add(marker)
                try:
                    if isinstance(value, dict):
                        for key, child in value.items():
                            validate_values(key); validate_values(child)
                    else:
                        for child in value: validate_values(child)
                finally:
                    active.remove(marker)
        validate_values(document)
    except ImportError as exc:
        # YAML is optional; JSON remains available without the eval extra.
        try:
            document = json.loads(raw, parse_constant=reject_constant,
                                  object_pairs_hook=reject_duplicates)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as inner:
            raise ValueError("invalid benchmark document") from inner
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("invalid benchmark document") from exc
    except Exception as exc:
        # PyYAML raises YAMLError for malformed/unsafe tags.  Avoid leaking
        # parser-specific exceptions through this public loader boundary.
        try:
            import yaml
            is_yaml_error = isinstance(exc, yaml.YAMLError)
        except ImportError:
            is_yaml_error = False
        if is_yaml_error:
            raise ValueError("invalid benchmark document") from exc
        raise

    if isinstance(document, dict) and ("model" in document or "model_id" in document):
        declared = document.get("model_id", document.get("model"))
        try:
            require_single_model(declared, field="benchmark model")
        except ValueError as exc:
            raise ValueError("benchmark model is outside the single-model policy") from exc
    cases = document.get("cases", document) if isinstance(document, dict) else document
    if not isinstance(cases, list) or not cases or len(cases) > 10_000:
        raise ValueError("benchmark must contain a bounded non-empty cases list")
    result = []
    seen: set[str] = set()
    for i, item in enumerate(cases):
        if not isinstance(item, dict) or not isinstance(item.get("goal"), str) or not item["goal"].strip() or len(item["goal"]) > 20_000:
            raise ValueError(f"invalid benchmark case {i}")
        case_id = str(item.get("id", f"case-{i}"))
        if not case_id or len(case_id) > 128 or case_id in seen:
            raise ValueError("benchmark case ids must be unique and bounded")
        seen.add(case_id)
        # A per-case model declaration is optional for backwards-compatible
        # fixtures, but if present it must not widen the runtime policy.
        if "model" in item or "model_id" in item:
            declared = item.get("model_id", item.get("model"))
            try:
                require_single_model(declared, field=f"benchmark case {case_id} model")
            except ValueError as exc:
                raise ValueError("benchmark case model is outside the single-model policy") from exc
        def strings(name: str) -> tuple[str, ...]:
            value = item.get(name, ())
            if isinstance(value, str): value = (value,)
            if not isinstance(value, (list, tuple)) or len(value) > 100 or any(not isinstance(v, str) or len(v) > 500 for v in value):
                raise ValueError(f"benchmark {name} is invalid")
            return tuple(value)
        result.append(EvalCase(case_id, item["goal"], str(item.get("expected", "")), Mode(item.get("mode", "fast")), str(item.get("expected_tool", "")), strings("expected_tools"), strings("postconditions"), str(item.get("expected_status", "")), str(item.get("family", "general")), strings("expected_citations"), strings("expected_claims")))
    return result



class AblationEvaluator:
    def __init__(self, runner_factory: Callable[[str], AgentRunner]):
        self.runner_factory = runner_factory

    async def evaluate(self, cases: list[EvalCase], variants: list[str]) -> list[dict[str, Any]]:
        if not isinstance(cases, (list, tuple)) or not cases or len(cases) > 10_000 or any(not isinstance(case, EvalCase) for case in cases):
            raise ValueError("evaluation cases must be a bounded non-empty EvalCase sequence")
        if not isinstance(variants, (list, tuple)) or not variants:
            raise ValueError("evaluation variants must be a non-empty sequence")
        cases = list(cases)
        variants = list(variants)
        if any(not isinstance(variant, str) for variant in variants):
            raise ValueError("evaluation variants must be strings")
        unknown = set(variants) - set(VARIANTS)
        if unknown: raise ValueError(f"unknown variants: {sorted(unknown)}")
        rows=[]
        for variant in variants:
            for case in cases:
                # Each case gets a fresh logical runner. Backends/registries
                # are mutable seams (e.g. scripted decisions are consumed and
                # registries freeze), so sharing one runner silently makes
                # later cases depend on earlier cases.
                runner = self.runner_factory(variant)
                # A real AgentRunner exposes its backend identity.  Offline fake
                # runners may omit it because no checkpoint is loaded; explicit
                # identities are always checked and cannot name another model.
                backend = getattr(runner, "backend", None)
                model_id = CANONICAL_MODEL_ID
                if backend is not None:
                    model_id = backend_model_identity(backend, field="evaluation backend model")
                budget = ComputeBudget.for_mode(case.mode)
                # Recursive variants get an explicit recursive allowance.  It
                # is part of the same budget (not an implicit extra model
                # budget), so the evaluator can compare quality and cost.
                if variant in {"rlm", "rah", "rah+verifier"} and budget.recursive_calls < 2:
                    budget = replace(budget, recursive_calls=min(8, max(2, budget.steps // 2)))
                started = asyncio.get_running_loop().time()
                contract = None
                if variant in {"+verification", "rah", "rah+verifier"}:
                    allowed = case.expected_tools or ((case.expected_tool,) if case.expected_tool else ())
                    contract = TaskContract(f"eval-{case.id}", case.goal, case.mode, tuple(allowed), expected_output=ExpectedOutput(max_chars=20_000), budget=budget)
                state = await runner.run(case.goal, case.mode, budget, contract=contract)
                elapsed_ms = int((asyncio.get_running_loop().time()-started)*1000)
                success = bool(state.answer) and (not case.expected or case.expected.lower() in state.answer.lower()) and (not case.expected_status or state.status == case.expected_status)
                tool_names = [item.tool for item in state.observations]
                failed_indices = [i for i, item in enumerate(state.observations) if not item.ok]
                retry_success = any(any(item.ok for item in state.observations[i + 1:]) for i in failed_indices)
                citation_ids = re.findall(r"\[([^]\n]{1,200})\]", state.answer) if case.family == "research" else []
                expected_citations = set(case.expected_citations)
                if expected_citations:
                    citation_precision = sum(c in expected_citations for c in citation_ids) / len(citation_ids) if citation_ids else 0.0
                    research_coverage = sum(c in citation_ids for c in expected_citations) / len(expected_citations)
                else:
                    citation_precision = research_coverage = None
                expected_claims = set(case.expected_claims)
                # Claim identifiers use the same bounded bracket notation as
                # citations (for example ``[claim-1]``), but are evaluated for
                # every benchmark family rather than only research rows.
                claim_ids = re.findall(r"\[([^]\n]{1,200})\]", state.answer)
                claim_coverage = (sum(c in claim_ids for c in expected_claims) / len(expected_claims)
                                   if expected_claims else None)
                inference_calls = int(getattr(state, "inference_calls", 0))
                valid_decisions = int(getattr(state, "decisions", 0))
                rows.append({"variant": variant, "case_id": case.id, "family": case.family, "model_id": model_id, "success": success, "status": state.status, "answer": redact(state.answer[:10_000]), "steps": state.steps, "llm_tokens": getattr(budget, "used_llm_tokens", 0), "tool_calls": len(state.observations), "wall_time_ms": elapsed_ms, "tool_names": tool_names, "expected_tool": case.expected_tool, "tool_selection_correct": (not case.expected_tool or case.expected_tool in tool_names) and (not case.expected_tools or set(case.expected_tools).issubset(set(tool_names))), "argument_valid": all(item.ok for item in state.observations), "retry_success": retry_success, "verification_passed": (state.status == "verified" if variant in {"+verification", "rah+verifier"} else None), "citation_precision": citation_precision, "research_coverage": research_coverage, "claim_coverage": claim_coverage, "rah_child_success": (any("child" in note and any(word in note for word in ("success", "complete", "verified")) for note in state.notes) if variant in {"rah", "rah+verifier"} else None), "merge_rejected": (state.status == "rejected" if variant in {"rah", "rah+verifier"} else None), "inference_calls": inference_calls, "valid_decisions": valid_decisions, "valid_decision_rate": valid_decisions / inference_calls if inference_calls else 0.0, "notes": redact(state.notes[-5:])})
        return rows

    @staticmethod
    def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            raise TypeError("rows must be a list")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("variant"), str):
                raise ValueError("evaluation rows must contain a variant")
            grouped.setdefault(row["variant"], []).append(row)
        def mean(items: list[dict[str, Any]], key: str) -> float | None:
            values = []
            for item in items:
                value = item.get(key)
                if isinstance(value, bool): values.append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)): values.append(float(value))
            return sum(values) / len(values) if values else None
        result = []
        for variant, items in grouped.items():
            count = len(items)
            result.append({
                "variant": variant, "cases": count,
                "success_rate": sum(bool(i.get("success", False)) for i in items) / count,
                "valid_decision_rate": mean(items, "valid_decision_rate") if any("valid_decision_rate" in i for i in items) else mean(items, "valid_decision"),
                "argument_validity": sum(bool(i.get("argument_valid", False)) for i in items) / count,
                "retry_success_rate": sum(bool(i.get("retry_success", False)) for i in items) / count,
                # Optional metrics deliberately remain null when not measured;
                # null is not a successful observation.
                "verification_pass_rate": mean(items, "verification_passed"),
                "citation_precision": mean(items, "citation_precision"),
                "research_coverage": mean(items, "research_coverage"),
                "claim_coverage": mean(items, "claim_coverage"),
                "rah_child_success_rate": mean(items, "rah_child_success"),
                "merge_rejection_rate": mean(items, "merge_rejected"),
                "llm_tokens": sum(int(i.get("llm_tokens", 0)) for i in items),
                "tool_calls": sum(int(i.get("tool_calls", 0)) for i in items),
                "wall_time": sum(int(i.get("wall_time_ms", 0)) for i in items),
                "stagnation": sum(i.get("status") == "stagnation" for i in items),
                "tool_selection_accuracy": sum(bool(i.get("tool_selection_correct", False)) for i in items) / count,
                "unnecessary_tool_calls": sum(max(0, int(i.get("tool_calls", 0)) - 1) for i in items if i.get("success")),
                "rows": items,
            })
        return result

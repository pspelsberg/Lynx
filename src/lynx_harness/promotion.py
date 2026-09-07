"""Governed automatic promotion for learned skills and knowledge.

Promotion is automatic only after deterministic evidence is supplied.  Model
claims are never sufficient.  The feature is disabled by default and must be
explicitly enabled by the host application.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .memory import OkfStore
from .skills import SkillEvaluation, SkillRegistry
from .verifier import VerificationResult, VerificationStatus, is_issued_verification


@dataclass(frozen=True)
class PromotionPolicy:
    """Policy for unattended promotion after replay/verifier gates."""

    enabled: bool = False
    require_replay: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.require_replay, bool):
            raise ValueError("promotion policy flags must be boolean")


@dataclass(frozen=True)
class PromotionRecord:
    kind: str
    name: str
    status: str
    reason: str
    path: str | None = None


class GovernedPromotion:
    """Coordinate safe, auditable skill/knowledge promotion.

    The coordinator does not mine or judge content itself.  It accepts only a
    replay-issued :class:`SkillEvaluation` or a complete verifier projection,
    so automatic execution cannot turn an LLM assertion into durable policy.
    """

    def __init__(
        self,
        *,
        registry: SkillRegistry | None = None,
        knowledge: OkfStore | None = None,
        audit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if registry is None and knowledge is None:
            raise ValueError("at least one promotion store is required")
        if audit is not None and not callable(audit):
            raise TypeError("audit must be callable")
        self.registry = registry
        self.knowledge = knowledge
        self.audit = audit

    def _record(self, record: PromotionRecord) -> None:
        if self.audit:
            self.audit(asdict(record))

    @staticmethod
    def _policy(policy: PromotionPolicy | None) -> PromotionPolicy:
        policy = policy or PromotionPolicy()
        if not isinstance(policy, PromotionPolicy):
            raise TypeError("policy must be PromotionPolicy")
        if not policy.enabled:
            raise PermissionError("automatic promotion is disabled")
        return policy

    def promote_skill(
        self,
        evaluation: SkillEvaluation,
        *,
        replay_passed: bool,
        policy: PromotionPolicy | None = None,
    ):
        self._policy(policy)
        if self.registry is None:
            raise ValueError("skill registry is not configured")
        if not isinstance(evaluation, SkillEvaluation) or not evaluation.passed:
            raise ValueError("skill evaluation did not pass")
        if not isinstance(replay_passed, bool):
            raise TypeError("replay_passed must be boolean")
        if policy is not None and policy.require_replay and not replay_passed:
            record = PromotionRecord("skill", evaluation.name, "rejected", "replay evaluation failed")
            self._record(record)
            raise ValueError(record.reason)
        try:
            path = self.registry.promote(evaluation.name, evaluation.version, evaluation)
        except Exception as exc:
            record = PromotionRecord("skill", evaluation.name, "rejected", str(exc)[:500])
            self._record(record)
            raise
        record = PromotionRecord("skill", evaluation.name, "promoted", "replay and safety gates passed", str(path))
        self._record(record)
        return path

    def promote_knowledge(
        self,
        slug: str,
        verifier: VerificationResult,
        *,
        policy: PromotionPolicy | None = None,
    ):
        self._policy(policy)
        if self.knowledge is None:
            raise ValueError("knowledge store is not configured")
        if (not isinstance(verifier, VerificationResult)
                or not is_issued_verification(verifier)
                or verifier.status != VerificationStatus.PASSED):
            raise ValueError("knowledge verifier did not pass")
        # OkfStore performs the same complete-check validation at the durable
        # sink. Keep the coordinator equally strict and never accept a dict
        # projection from an untrusted caller.
        if (not verifier.checks
                or any(not check.id or check.status != VerificationStatus.PASSED
                       for check in verifier.checks)):
            raise ValueError("knowledge verifier is incomplete")
        try:
            path = self.knowledge.promote_candidate(slug, verifier)
        except Exception as exc:
            record = PromotionRecord("knowledge", slug, "rejected", str(exc)[:500])
            self._record(record)
            raise
        record = PromotionRecord("knowledge", slug, "promoted", "complete verifier passed", str(path))
        self._record(record)
        return path

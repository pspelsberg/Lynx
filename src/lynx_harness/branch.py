from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

from .models import AgentState, TaskContract
from .verifier import Verifier, VerificationResult, VerificationStatus


@dataclass(frozen=True)
class BranchResult:
    branch_id: str
    answer: str
    status: str
    observations: tuple[dict[str, Any], ...]


class BranchManager:
    """Create isolated snapshots; merge is an explicit proposal, never implicit memory mutation."""
    def __init__(self, root: AgentState):
        self.root = root
        self.branches: dict[str, AgentState] = {}

    def fork(self, branch_id: str) -> AgentState:
        if branch_id in self.branches: raise ValueError(f"duplicate branch: {branch_id}")
        child = self.root.fork(branch_id)
        self.branches[branch_id] = child
        return child

    def result(self, branch_id: str) -> BranchResult:
        state = self.branches[branch_id]
        return BranchResult(branch_id, state.answer, state.status, tuple(copy.deepcopy(item.__dict__) for item in state.observations))

    def merge_proposal(self, branch_ids: list[str], *, verifier: Verifier | None = None, contract: TaskContract | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not branch_ids or any(item not in self.branches for item in branch_ids): raise ValueError("unknown or empty branch set")
        proposal = {"type":"merge_proposal", "branches":[self.result(item).__dict__ for item in branch_ids], "requires_verifier":True, "writes_root":False}
        if verifier and contract:
            checks = []
            for item in branch_ids:
                state = self.branches[item]
                check_context = dict(context or {})
                # Caller context may carry evidence/artifact data, but must not
                # be able to replace the branch's observed tool lineage.
                check_context["tools_used"] = [o.tool for o in state.observations]
                # Branch-owned lineage and artifacts are authoritative; caller
                # context must not be able to substitute another branch.
                check_context["branch_id"] = state.branch_id
                check_context["artifact_refs"] = tuple(
                    o.artifact_id for o in state.observations if o.artifact_id
                )
                checks.append({"branch_id": item, "verification": verifier.check({"answer": state.answer}, contract, check_context)})
            proposal["verification"] = checks
        return proposal

    def merge(self, branch_id: str, *, verifier: Verifier, contract: TaskContract, context: dict[str, Any] | None = None) -> AgentState:
        """Apply exactly one verified proposal; failed branches never mutate root."""
        if branch_id not in self.branches: raise ValueError("unknown branch")
        state = self.branches[branch_id]
        if state.status not in {"success", "verified", "complete"}: raise ValueError(f"branch status is not mergeable: {state.status}")
        check_context = dict(context or {})
        # Keep the observed branch lineage authoritative; supplied context is
        # limited to supplemental evidence and cannot hide forbidden tools.
        check_context["tools_used"] = [o.tool for o in state.observations]
        # Never trust merge-supplied lineage/evidence. The branch state is the
        # sole source of ownership for a merge verification.
        check_context["branch_id"] = state.branch_id
        check_context["artifact_refs"] = tuple(
            o.artifact_id for o in state.observations if o.artifact_id
        )
        result = verifier.check({"answer": state.answer}, contract, check_context)
        if result.status != VerificationStatus.PASSED: raise ValueError(f"branch rejected: {result.reason}")
        self.root.answer = state.answer
        self.root.observations = copy.deepcopy(state.observations)
        self.root.status = "merged"
        return self.root

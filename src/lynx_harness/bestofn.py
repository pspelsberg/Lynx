from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import AgentState, ComputeBudget, Mode, BudgetLedger, BudgetExceeded
from .loop import AgentRunner


@dataclass(frozen=True)
class BranchCandidate:
    branch_id: str
    status: str
    answer: str
    steps: int
    tool_calls: int
    worktree_path: str | None = None
    diff: str | None = None


async def _run_git(args: list[str], cwd: Path, timeout: float = 10.0) -> tuple[int, str]:
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_PAGER": "cat",
    }
    env.pop("GIT_EXTERNAL_DIFF", None)
    proc = await asyncio.create_subprocess_exec(
        "git", *args, cwd=str(cwd), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    out = (stdout + stderr).decode("utf-8", "replace")
    return proc.returncode, out


class BestOfNExecutor:
    """Run isolated candidate harnesses in parallel; returns proposals, never auto-merges them."""
    def __init__(self, factory: Callable[..., AgentRunner], branches: int = 3, ledger: BudgetLedger | None = None, workspace: Path | None = None, use_worktrees: bool = False):
        if not 1 <= branches <= 8: raise ValueError("branches must be 1..8")
        self.factory, self.branches, self.ledger = factory, branches, ledger
        self.workspace = workspace.resolve() if workspace is not None else None
        self.use_worktrees = bool(use_worktrees)

    async def run(self, goal: str, mode: Mode = Mode.THINK, budget: ComputeBudget | None = None) -> list[BranchCandidate]:
        budget = budget or ComputeBudget.for_mode(mode)
        is_git = self.use_worktrees and self.workspace and (self.workspace / ".git").exists()

        async def one(index: int) -> BranchCandidate:
            branch_id = f"branch-{index}"
            wt_dir = None
            if is_git and self.workspace:
                wt_dir = Path(tempfile.mkdtemp(prefix=f"lynx_arena_{index}_"))
                code, out = await _run_git(["worktree", "add", "--detach", str(wt_dir), "HEAD"], self.workspace)
                if code != 0:
                    try: shutil.rmtree(wt_dir, ignore_errors=True)
                    except OSError: pass
                    wt_dir = None

            # Instantiate runner (pass worktree dir if accepted by factory)
            try:
                sig = inspect.signature(self.factory)
                if len(sig.parameters) >= 2 and wt_dir is not None:
                    runner = self.factory(branch_id, wt_dir)
                else:
                    runner = self.factory(branch_id)
            except Exception:
                runner = self.factory(branch_id)

            branch_reservation = None
            if self.ledger:
                try: branch_reservation = self.ledger.reserve(parallel_branches=1)
                except BudgetExceeded:
                    if wt_dir and self.workspace:
                        await _run_git(["worktree", "remove", "--force", str(wt_dir)], self.workspace)
                        try: shutil.rmtree(wt_dir, ignore_errors=True)
                        except OSError: pass
                    return BranchCandidate(branch_id, "budget_exhausted", "", 0, 0)
                runner.ledger = self.ledger

            diff = None
            try:
                state = await runner.run(goal, mode, ComputeBudget(**{k:v for k,v in budget.__dict__.items() if not k.startswith("used_")}), cancellation=None)
                if branch_reservation: branch_reservation.commit({"parallel_branches":1})
                if wt_dir:
                    code, diff_out = await _run_git(["diff", "--no-ext-diff", "HEAD"], wt_dir)
                    diff = diff_out if code == 0 else ""
            except BaseException:
                if branch_reservation: branch_reservation.release()
                if wt_dir and self.workspace:
                    await _run_git(["worktree", "remove", "--force", str(wt_dir)], self.workspace)
                    try: shutil.rmtree(wt_dir, ignore_errors=True)
                    except OSError: pass
                raise
            return BranchCandidate(branch_id, state.status, state.answer, state.steps, len(state.observations), worktree_path=str(wt_dir) if wt_dir else None, diff=diff)

        running = [asyncio.create_task(one(index)) for index in range(self.branches)]
        try:
            return list(await asyncio.gather(*running))
        except BaseException:
            for task in running:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise

    async def cleanup_candidate(self, candidate: BranchCandidate) -> None:
        """Remove temporary git worktree for a candidate."""
        if candidate.worktree_path and self.workspace:
            wt_dir = Path(candidate.worktree_path)
            await _run_git(["worktree", "remove", "--force", str(wt_dir)], self.workspace)
            try: shutil.rmtree(wt_dir, ignore_errors=True)
            except OSError: pass

    async def cleanup_all(self, candidates: list[BranchCandidate]) -> None:
        for c in candidates:
            await self.cleanup_candidate(c)


from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class WorkspaceSnapshot:
    """Tracks and reverts workspace file mutations for safe agent execution and undo."""

    def __init__(self, workspace: Path, snapshot_dir: Path | None = None) -> None:
        self.workspace = workspace.resolve()
        self.snapshot_dir = snapshot_dir or (self.workspace / ".lynx" / "snapshots")
        self._initial_files: dict[str, str | None] = {}
        self.active_run_id: str | None = None

    def begin(self, run_id: str) -> None:
        self.active_run_id = run_id
        self._initial_files.clear()

    def record_before_mutation(self, target: Path) -> None:
        try:
            rel = str(target.relative_to(self.workspace))
        except ValueError:
            return
        if rel not in self._initial_files:
            if target.exists() and target.is_file():
                try:
                    self._initial_files[rel] = target.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
            else:
                self._initial_files[rel] = None

    def rollback(self) -> list[str]:
        restored = []
        for rel, orig in list(self._initial_files.items()):
            target = (self.workspace / rel).resolve()
            if self.workspace not in target.parents and target != self.workspace:
                continue
            if orig is None:
                try:
                    if target.exists() and target.is_file():
                        target.unlink()
                        restored.append(f"deleted {rel}")
                except OSError:
                    pass
            else:
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(orig, encoding="utf-8")
                    restored.append(f"restored {rel}")
                except OSError:
                    pass
        return restored

    def persist(self, run_id: str | None = None) -> Path | None:
        rid = run_id or self.active_run_id or f"run-{int(time.time())}"
        if not self._initial_files:
            return None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        dump_file = self.snapshot_dir / f"{rid}.json"
        data = {
            "run_id": rid,
            "timestamp": time.time(),
            "files": self._initial_files,
        }
        dump_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return dump_file

    def restore_from_disk(self, run_id: str) -> list[str]:
        dump_file = self.snapshot_dir / f"{run_id}.json"
        if not dump_file.exists():
            raise FileNotFoundError(f"snapshot for run {run_id} not found at {dump_file}")
        data = json.loads(dump_file.read_text(encoding="utf-8"))
        restored = []
        for rel, orig in data.get("files", {}).items():
            target = (self.workspace / rel).resolve()
            if self.workspace not in target.parents and target != self.workspace:
                continue
            if orig is None:
                try:
                    if target.exists() and target.is_file():
                        target.unlink()
                        restored.append(f"deleted {rel}")
                except OSError:
                    pass
            else:
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(orig, encoding="utf-8")
                    restored.append(f"restored {rel}")
                except OSError:
                    pass
        return restored

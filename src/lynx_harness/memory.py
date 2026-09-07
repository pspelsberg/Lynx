from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
import re
import threading

from .security import redact, redact_text, redact_bytes


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    digest: str
    size: int
    origin: str = "local"
    trust: str = "untrusted"
    branch_id: str = "main"
    task_id: str | None = None

class ArtifactStore:
    def __init__(self, root: Path, max_total_bytes: int = 100_000_000):
        if not isinstance(root, Path): raise ValueError("artifact root must be a Path")
        # Preserve the lexical path while checking ancestors. Resolving first
        # would silently accept a symlinked artifact directory and redirect a
        # supposedly local durable sink outside its configured root.
        absolute_root = root.absolute()
        if root.is_symlink() or any(parent.is_symlink() for parent in absolute_root.parents):
            raise ValueError("artifact root and ancestors must not be symlinks")
        self.root = absolute_root
        if not 1_000_000 <= max_total_bytes <= 10_000_000_000: raise ValueError("artifact limit out of bounds")
        self.max_total_bytes = max_total_bytes; self.root.mkdir(parents=True, exist_ok=True)
        self._index_path=self.root/"manifest.json"; self._lock=threading.RLock(); self._index=self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self._index_path.is_file(): return {}
        try:
            value=json.loads(self._index_path.read_text(encoding="utf-8")); return value if isinstance(value,dict) else {}
        except (OSError, json.JSONDecodeError): return {}

    def _save_index(self) -> None:
        # The manifest is a mutable index, but updates must be atomic and must
        # never follow an attacker-created temporary symlink.  A unique file
        # plus os.replace also makes a crash leave either the old or new index.
        encoded = json.dumps(self._index, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
        fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=str(self.root))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self._index_path)
        except BaseException:
            try: temporary.unlink(missing_ok=True)
            except OSError: pass
            raise

    @staticmethod
    def _name(artifact_id: str) -> str:
        name=artifact_id.removeprefix("artifact://") if isinstance(artifact_id,str) else ""
        if Path(name).name != name or name != name.strip() or not re.fullmatch(r"[0-9a-f]{64}\.[A-Za-z0-9_]{1,10}", name): raise ValueError("invalid artifact id")
        return name

    def put(self, content: str | bytes, suffix: str = ".txt", *, origin: str = "local", trust: str = "untrusted", branch_id: str = "main", task_id: str | None = None) -> str:
        if not isinstance(suffix, str) or not re.fullmatch(r"\.[A-Za-z0-9_]{1,10}", suffix): raise ValueError("invalid artifact suffix")
        if isinstance(content, (str, bytes)) and len(content) > 10_000_000: raise ValueError("artifact exceeds 10 MB")
        content = redact_text(content, max_chars=None) if isinstance(content, str) else (redact_bytes(content, max_chars=None) if isinstance(content, bytes) else content)
        if not isinstance(content, (str, bytes)): raise TypeError("artifact content must be text or bytes")
        data = content.encode() if isinstance(content, str) else content
        if len(data) > 10_000_000: raise ValueError("artifact exceeds 10 MB")
        if not isinstance(branch_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",branch_id): raise ValueError("invalid artifact branch")
        if not isinstance(origin,str) or len(origin)>100 or not isinstance(trust,str) or len(trust)>100: raise ValueError("invalid artifact provenance")
        digest = hashlib.sha256(data).hexdigest(); path = self.root / f"{digest}{suffix}"
        with self._lock:
            if path.is_symlink():
                raise ValueError("artifact path must not be a symlink")
            if path.exists():
                # A pre-existing file is only reusable when it actually has
                # the digest encoded in its name.  Otherwise ``put`` could
                # publish a reference that later resolves to attacker-chosen
                # bytes (or silently point at a corrupted artifact).
                if not path.is_file() or path.stat().st_size != len(data) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("artifact content conflicts with existing file")
                existing = self._index.get(path.name)
                expected_ref = f"artifact://{path.name}"
                if existing and (existing.get("artifact_id") != expected_ref or
                                  existing.get("digest") != digest or
                                  existing.get("size") != len(data)):
                    # A tampered manifest must not be allowed to redirect a
                    # content-addressed reference to another artifact.
                    raise ValueError("artifact content conflicts with manifest")
                # Content-addressed artifacts are immutable: never reassign a
                # branch/task provenance record for an existing reference.
                if existing:
                    return expected_ref
            else:
                total=sum(item.stat().st_size for item in self.root.iterdir() if item.is_file() and item.name not in {"manifest.json", "manifest.json.tmp"})
                if total + len(data) > self.max_total_bytes: raise ValueError("artifact store limit exceeded")
                # Never use a predictable temporary filename: a symlink at
                # ``<digest>.<suffix>.tmp`` would otherwise redirect writes.
                fd, temporary_name = tempfile.mkstemp(prefix=f".{digest}-", suffix=".tmp", dir=str(self.root))
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.replace(path)
                except BaseException:
                    try: temporary.unlink(missing_ok=True)
                    except OSError: pass
                    raise
            ref=f"artifact://{path.name}"; self._index[path.name]={"artifact_id":ref,"digest":digest,"size":len(data),"origin":origin,"trust":trust,"branch_id":branch_id,"task_id":task_id}; self._save_index(); return ref

    def metadata(self, artifact_id: str) -> ArtifactMetadata:
        name=self._name(artifact_id); path=self.root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000: raise ValueError("artifact unavailable or too large")
        raw=self._index.get(name)
        data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        expected_ref = f"artifact://{name}"
        if (not isinstance(raw, dict) or raw.get("artifact_id") != expected_ref
                or raw.get("digest") != digest or raw.get("size") != len(data)):
            raise ValueError("artifact provenance or hash is invalid")
        return ArtifactMetadata(expected_ref,digest,len(data),str(raw.get("origin","local")),str(raw.get("trust","untrusted")),str(raw.get("branch_id","main")),raw.get("task_id"))

    def get(self, artifact_id: str, max_bytes: int = 1_000_000) -> bytes:
        name=self._name(artifact_id); path=self.root/name
        with self._lock:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes: raise ValueError("artifact unavailable or too large")
            data=path.read_bytes(); raw=self._index.get(name)
            expected_ref = f"artifact://{name}"
            if (not isinstance(raw, dict) or raw.get("artifact_id") != expected_ref
                    or raw.get("size") != len(data) or raw.get("digest") != hashlib.sha256(data).hexdigest()): raise ValueError("artifact provenance or hash is invalid")
            return data

    def manifest(self, artifact_id: str) -> dict[str, Any]:
        return asdict(self.metadata(artifact_id))


class JsonlStore:
    def __init__(self, path: Path, max_bytes: int = 1_000_000_000):
        if not isinstance(path, Path): raise ValueError("trajectory path must be a Path")
        self.path = path.absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in self.path.parents):
            raise ValueError("trajectory path and ancestors must not be symlinks")
        if not 1_000_000 <= max_bytes <= 10_000_000_000: raise ValueError("trajectory limit out of bounds")
        self.max_bytes = max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise TypeError("trajectory record must be an object")
        safe = redact(record)
        # Reject NaN/Infinity and avoid default=str silently converting values
        # into an unverifiable trajectory representation.
        encoded = json.dumps(safe, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        data = (encoded + "\n").encode("utf-8")
        if len(data) > 1_000_000:
            raise ValueError("trajectory record exceeds 1 MB")
        with self._lock:
            if self.path.is_symlink() or any(parent.is_symlink() for parent in self.path.parents):
                raise ValueError("trajectory path and ancestors must not be symlinks")
            try:
                current_size = self.path.stat().st_size if self.path.exists() else 0
            except OSError as exc:
                raise ValueError("trajectory path is unavailable") from exc
            if self.path.exists() and not self.path.is_file():
                raise ValueError("trajectory path is not a regular file")
            if current_size + len(data) > self.max_bytes:
                raise ValueError("trajectory store limit exceeded")
            # O_NOFOLLOW closes the check-then-open symlink race on the final
            # path component while O_APPEND keeps each record indivisible.
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self.path, flags | nofollow, 0o600)
            except OSError as exc:
                raise ValueError("trajectory path must be a regular non-symlink file") from exc
            try:
                with os.fdopen(fd, "ab", buffering=0) as stream:
                    stream.write(data)
                    os.fsync(stream.fileno())
            except BaseException:
                try: os.close(fd)
                except OSError: pass
                raise


class OkfStore:
    """Small OKF-like filesystem store: Markdown documents with YAML-like frontmatter."""
    def __init__(self, root: Path):
        if not isinstance(root, Path): raise ValueError("knowledge root must be a Path")
        absolute_root = root.absolute()
        if root.is_symlink() or any(parent.is_symlink() for parent in absolute_root.parents):
            raise ValueError("knowledge root and ancestors must not be symlinks")
        self.root = absolute_root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write a knowledge document without following a raced symlink.

        A check followed by ``Path.write_text`` is not sufficient for a
        durable sink: an attacker can replace the checked path with a
        symlink between those operations.  Replacing a private temporary file
        atomically removes (rather than follows) a destination symlink.
        """
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("knowledge path parent must be a regular directory")
        fd, temporary_name = tempfile.mkstemp(prefix=".knowledge-", suffix=".tmp", dir=str(parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def put_candidate(self, slug: str, title: str, body: str, source: str = "") -> Path:
        safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in slug.lower()).strip("-")[:80] or "candidate"
        if not isinstance(title, str) or not isinstance(body, str) or not isinstance(source, str):
            raise TypeError("knowledge fields must be strings")
        if len(title) > 2_000 or len(source) > 2_000 or len(body) > 100_000:
            raise ValueError("knowledge candidate is too large")
        path = self.root / f"{safe}.md"
        if path.is_symlink():
            raise ValueError("knowledge path must not be a symlink")
        if path.is_file() and "status: stable" in path.read_text(encoding="utf-8", errors="replace"):
            raise ValueError("stable knowledge cannot be overwritten by a candidate")
        # Knowledge is a durable sink too; do not persist credentials copied
        # from web/tool observations.
        title, source, body = redact_text(title, max_chars=None), redact_text(source, max_chars=None), redact_text(body, max_chars=None)
        text = f"---\ntype: Reference\ntitle: {title!r}\nstatus: candidate\nsource: {source!r}\n---\n\n{body}\n"
        self._atomic_write(path, text)
        return path

    def promote_candidate(self, slug: str, verifier: Any = None) -> Path:
        safe="".join(char if char.isalnum() or char in "-_" else "-" for char in slug.lower()).strip("-")[:80] or "candidate"
        path=self.root / f"{safe}.md"
        if path.is_symlink() or not path.is_file(): raise FileNotFoundError(path)
        if path.stat().st_size > 1_000_000: raise ValueError("knowledge document is too large")
        text=path.read_text(encoding="utf-8")
        if "status: candidate" not in text: raise ValueError("only candidates can be promoted")
        # Promotion is a security/data-integrity gate, not an annotation API.
        # A caller-provided dict (or a freshly constructed dataclass) is not
        # evidence that a verifier actually ran.  Require an object minted by
        # Verifier and require every check in that result to have passed.
        from .verifier import VerificationResult, VerificationStatus, is_issued_verification, verification_binding
        if (not isinstance(verifier, VerificationResult)
                or not is_issued_verification(verifier)
                or verifier.status != VerificationStatus.PASSED):
            raise ValueError("promotion requires a passed verifier result")
        if (not verifier.checks or
                any(check.status != VerificationStatus.PASSED or not check.id for check in verifier.checks)):
            raise ValueError("promotion verifier did not pass complete checks")
        binding = verification_binding(verifier)
        body_marker = "\n---\n\n"
        candidate_body = text.split(body_marker, 1)[1].rstrip("\n") if body_marker in text else ""
        if (binding is None
                or binding.get("answer", "").rstrip("\n") != candidate_body[:20_000].rstrip("\n")):
            raise ValueError("promotion verifier is not bound to candidate content")
        evidence = asdict(verifier)
        try:
            if len(json.dumps(redact(evidence), ensure_ascii=False, allow_nan=False).encode("utf-8")) > 20_000:
                raise ValueError("promotion verifier evidence is too large")
        except (TypeError, ValueError) as exc:
            raise ValueError("promotion verifier evidence is invalid") from exc
        text=text.replace("status: candidate", "status: stable", 1).replace("---\n\n", f"verified_by: {redact(evidence)!r}\nverification_event: promotion\n---\n\n", 1)
        self._atomic_write(path, text); return path

    def deprecate(self, slug: str, reason: str = "") -> Path:
        safe="".join(char if char.isalnum() or char in "-_" else "-" for char in slug.lower()).strip("-")[:80] or "candidate"
        path=self.root / f"{safe}.md"
        if path.is_symlink() or not path.is_file(): raise FileNotFoundError(path)
        if path.stat().st_size > 1_000_000: raise ValueError("knowledge document is too large")
        text=path.read_text(encoding="utf-8").replace("status: stable", "status: deprecated", 1).replace("status: candidate", "status: deprecated", 1)
        if reason:
            if not isinstance(reason, str): raise TypeError("deprecation reason must be a string")
            text += f"\nDeprecation reason: {redact_text(reason[:500], max_chars=None)}\n"
        self._atomic_write(path, text); return path

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        if not isinstance(query, str) or len(query) > 500 or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("knowledge search request is invalid")
        terms = query.lower().split()
        matches=[]
        for path in sorted(self.root.glob("*.md")):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            text=path.read_text(encoding="utf-8", errors="replace")
            score=sum(text.lower().count(term) for term in terms)
            if score: matches.append({"path":str(path),"score":str(score),"text":text[:2_000]})
        return sorted(matches, key=lambda x: -int(x["score"]))[:limit]

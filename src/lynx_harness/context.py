from __future__ import annotations

from dataclasses import dataclass
import re
import hashlib
from typing import Any, Callable, Awaitable

from .memory import ArtifactStore


@dataclass(frozen=True)
class ContextChunk:
    artifact_id: str
    offset: int
    content: str
    score: int
    chunk_id: str = ""
    content_hash: str = ""
    query: str = ""
    parent_ref: str | None = None

    def provenance(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "artifact_id": self.artifact_id, "offset": self.offset, "hash": self.content_hash, "query": self.query, "score": self.score, "parent_ref": self.parent_ref}


class RlmContextEngine:
    """RLM context seam: inspect and retrieve bounded artifact chunks, never execute corpus code."""
    def __init__(self, artifacts: ArtifactStore, chunk_chars: int = 8_000, max_chunks: int = 8, context_threshold: int = 24_000):
        if not 100 <= chunk_chars <= 100_000: raise ValueError("chunk_chars out of bounds")
        if not 1 <= max_chunks <= 100 or not 1_000 <= context_threshold <= 10_000_000: raise ValueError("context limits out of bounds")
        self.artifacts, self.chunk_chars, self.max_chunks, self.context_threshold = artifacts, chunk_chars, max_chunks, context_threshold

    def inspect(self, artifact_id: str) -> dict[str, Any]:
        data = self.artifacts.get(artifact_id, max_bytes=10_000_000).decode("utf-8", "replace")
        headings = [re.sub(r"^#{1,6}\s+", "", line.strip()) for line in data.splitlines() if re.match(r"^#{1,6}\s+", line)][:200]
        return {"artifact_id": artifact_id, "bytes": len(data.encode()), "chars": len(data), "headings": headings}

    def retrieve(self, artifact_id: str, query: str) -> list[ContextChunk]:
        data = self.artifacts.get(artifact_id, max_bytes=10_000_000).decode("utf-8", "replace")
        terms = set(re.findall(r"[a-z0-9_-]+", query.lower()))
        chunks=[]; seen=set()
        for offset in range(0, len(data), self.chunk_chars):
            content=data[offset:offset+self.chunk_chars]
            content_hash=hashlib.sha256(content.encode()).hexdigest()
            score=sum(content.lower().count(term) for term in terms)
            if score and content_hash not in seen:
                seen.add(content_hash); chunk_id=f"{artifact_id.removeprefix('artifact://')}:{offset}:{content_hash[:16]}"
                chunks.append(ContextChunk(artifact_id,offset,content,score,chunk_id,content_hash,query,artifact_id))
        return sorted(chunks,key=lambda chunk:(-chunk.score,chunk.offset))[:self.max_chunks]

    @staticmethod
    def branch_suffix(branch_id: str) -> str:
        """Return a short safe suffix so identical bytes remain branch-scoped."""
        if not isinstance(branch_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", branch_id):
            raise ValueError("invalid context branch")
        return ".c" + hashlib.sha256(branch_id.encode("utf-8")).hexdigest()[:8]

    def prepare(self, value: str, query: str = "", *, branch_id: str = "main", task_id: str | None = None) -> dict[str, Any]:
        """Store oversized input and return only metadata plus bounded relevant chunks."""
        if not isinstance(value, str): raise ValueError("context input must be text")
        suffix = self.branch_suffix(branch_id)
        if len(value) <= self.context_threshold:
            return {"mode": "inline", "content": value, "chunks": []}
        artifact_id = self.artifacts.put(value, suffix=suffix, branch_id=branch_id, task_id=task_id)
        chunks = self.retrieve(artifact_id, query) if query else []
        return {"mode": "artifact", "artifact_id": artifact_id, "summary": self.inspect(artifact_id), "chunks": [chunk.provenance() | {"content": chunk.content} for chunk in chunks]}


@dataclass(frozen=True)
class RlmLimits:
    max_depth: int = 2
    max_calls: int = 8
    max_chars_per_call: int = 8_000
    def __post_init__(self):
        if type(self.max_depth) is not int or not 0 <= self.max_depth <= 32 or type(self.max_calls) is not int or not 1 <= self.max_calls <= 10_000 or type(self.max_chars_per_call) is not int or not 100 <= self.max_chars_per_call <= 1_000_000: raise ValueError("RLM limits out of bounds")


class RlmExecutor:
    """Bounded recursive callback executor; supplied callbacks remain the policy boundary."""
    def __init__(self, query: Callable[[str, int], Awaitable[str]], limits: RlmLimits | None = None, ledger: Any | None = None):
        self.query, self.limits, self.calls, self.ledger = query, limits or RlmLimits(), 0, ledger

    async def run(self, prompts: list[str], depth: int = 0) -> list[str]:
        if depth > self.limits.max_depth: raise RuntimeError("RLM depth limit exceeded")
        results=[]
        for prompt in prompts[:self.limits.max_calls - self.calls]:
            if self.calls >= self.limits.max_calls: break
            reservation = self.ledger.reserve(recursive_calls=1) if self.ledger else None
            self.calls += 1
            try:
                result = (await self.query(prompt[:self.limits.max_chars_per_call], depth))[:self.limits.max_chars_per_call]
                if reservation: reservation.commit({"recursive_calls": 1})
                results.append(result)
            except BaseException:
                if reservation: reservation.release()
                raise
        return results


async def bounded_map_reduce(
    engine: RlmContextEngine,
    artifact_id: str,
    query: str,
    mapper: Callable[[ContextChunk], Awaitable[str]],
    reducer: Callable[[list[str]], Awaitable[str]],
    *,
    max_chunks: int | None = None,
    ledger: Any | None = None,
    max_concurrency: int = 2,
    timeout_s: float = 60.0,
    cancellation: Any | None = None,
) -> dict[str, Any]:
    """Production-facing RLM entry point.

    ``map_reduce`` is retained as the compatibility function used by older
    callers; this named entry point makes the policy boundary explicit for
    the agent loop and keeps all recursive work bounded by the supplied
    engine/ledger.
    """
    return await map_reduce(
        engine, artifact_id, query, mapper, reducer, max_chunks=max_chunks,
        ledger=ledger, max_concurrency=max_concurrency, timeout_s=timeout_s,
        cancellation=cancellation,
    )


async def map_reduce(engine: RlmContextEngine, artifact_id: str, query: str, mapper: Callable[[ContextChunk], Awaitable[str]], reducer: Callable[[list[str]], Awaitable[str]], *, max_chunks: int | None = None, ledger: Any | None = None, max_concurrency: int = 2, timeout_s: float = 60.0, cancellation: Any | None = None) -> dict[str, Any]:
    asyncio=__import__("asyncio")
    if not 1 <= max_concurrency <= 16 or not 0.1 <= timeout_s <= 3_600: raise ValueError("map/reduce limits out of bounds")
    if max_chunks is not None and (type(max_chunks) is not int or not 1 <= max_chunks <= engine.max_chunks): raise ValueError("max_chunks out of bounds")
    chunks = engine.retrieve(artifact_id, query)[:max_chunks if max_chunks is not None else engine.max_chunks]
    semaphore=asyncio.Semaphore(max_concurrency)
    async def one(chunk):
        async with semaphore:
            reservation=ledger.reserve(recursive_calls=1) if ledger else None
            try:
                task=mapper(chunk)
                result=await __import__("lynx_harness.cancellation",fromlist=["bounded"]).bounded(task,timeout_s,cancellation)
                if reservation: reservation.commit({"recursive_calls":1})
                return str(result)[:engine.chunk_chars]
            except BaseException:
                if reservation: reservation.release()
                raise
    mapped=await asyncio.gather(*(one(chunk) for chunk in chunks))
    if ledger:
        reservation=ledger.reserve(recursive_calls=1)
    else: reservation=None
    try:
        answer=await __import__("lynx_harness.cancellation",fromlist=["bounded"]).bounded(reducer(mapped),timeout_s,cancellation)
        if reservation: reservation.commit({"recursive_calls":1})
    except BaseException:
        if reservation: reservation.release()
        raise
    return {"answer": str(answer)[:engine.chunk_chars], "chunks": [chunk.provenance() for chunk in chunks]}


def compact_context(context: dict[str, Any], *, max_chars: int = 20_000) -> dict[str, Any]:
    """Lossy working-state compaction retaining references and bounded notes."""
    if not isinstance(context, dict) or max_chars < 1_000: raise ValueError("invalid compaction input")
    result=dict(context); observations=result.get("untrusted_observations", [])
    result["untrusted_observations"]=[{"observation": {"tool": item.get("observation",{}).get("tool"), "artifact_id": item.get("observation",{}).get("artifact_id"), "epistemic_status": item.get("observation",{}).get("epistemic_status")}, "boundary":"data_only", "can_instruct":False, "can_authorize":False} for item in observations[-8:] if isinstance(item,dict)]
    result["notes"]=list(result.get("notes",[]))[-6:]
    encoded = __import__("json").dumps(result, default=str, ensure_ascii=False)
    if len(encoded) > max_chars:
        # Preserve only bounded control-plane fields; large observations and
        # schemas must remain in artifacts rather than re-entering the prompt.
        result["goal"] = str(result.get("goal", ""))[:2_000]
        result["notes"] = [str(note)[:1_000] for note in result.get("notes", [])[-6:]]
        result["available_tools"] = [{key: item.get(key) for key in ("name", "description", "risk", "family", "origin", "trust") if key in item} for item in result.get("available_tools", []) if isinstance(item, dict)][:20]
        result["trusted_state"] = {"observations": []}
        result["untrusted_observations"] = result.get("untrusted_observations", [])[-4:]
    if len(__import__("json").dumps(result, default=str, ensure_ascii=False).encode("utf-8")) > max_chars:
        raise ValueError("context cannot be compacted within configured limit")
    return result

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import re
import time
from typing import Any, Awaitable, Callable

from .memory import JsonlStore, OkfStore, ArtifactStore
from .models import ComputeBudget, Mode
from .tools import ToolGateway
from .cancellation import bounded, CancellationToken
from .security import redact_text


class ResearchPhase(StrEnum):
    CLARIFY = "clarify"
    QUESTIONS = "questions"
    SEARCH = "search"
    RANK = "rank"
    FETCH = "fetch"
    EXTRACT_CLAIMS = "extract_claims"
    GAP_ANALYSIS = "gap_analysis"
    CONTRADICTIONS = "contradictions"
    TARGETED_SEARCH = "targeted_search"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    CITE = "cite"
    COMPLETE = "complete"


_RESEARCH_NEXT = {
    ResearchPhase.CLARIFY: {ResearchPhase.QUESTIONS},
    ResearchPhase.QUESTIONS: {ResearchPhase.SEARCH},
    ResearchPhase.SEARCH: {ResearchPhase.RANK},
    ResearchPhase.RANK: {ResearchPhase.FETCH},
    ResearchPhase.FETCH: {ResearchPhase.EXTRACT_CLAIMS},
    ResearchPhase.EXTRACT_CLAIMS: {ResearchPhase.GAP_ANALYSIS},
    ResearchPhase.GAP_ANALYSIS: {ResearchPhase.CONTRADICTIONS},
    ResearchPhase.CONTRADICTIONS: {ResearchPhase.TARGETED_SEARCH, ResearchPhase.SYNTHESIZE},
    ResearchPhase.TARGETED_SEARCH: {ResearchPhase.SYNTHESIZE},
    ResearchPhase.SYNTHESIZE: {ResearchPhase.VERIFY},
    ResearchPhase.VERIFY: {ResearchPhase.CITE},
    ResearchPhase.CITE: {ResearchPhase.COMPLETE},
}


class ResearchController:
    def __init__(self):
        self.phase = ResearchPhase.CLARIFY
        self.history = [self.phase.value]

    def transition(self, phase: ResearchPhase):
        phase = ResearchPhase(phase)
        if phase not in _RESEARCH_NEXT.get(self.phase, set()):
            raise ValueError(f"invalid research transition: {self.phase} -> {phase}")
        self.phase = phase
        self.history.append(phase.value)
        return phase


@dataclass
class Evidence:
    claim: str
    source_id: str
    excerpt: str
    relation: str = "supports"
    confidence: float = 0.0
    published_at: str | None = None
    retrieved_at: str = ""
    verified: bool = False
    claim_id: str = ""
    verification: str = "pending"


@dataclass
class ResearchState:
    question: str
    subquestions: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    status: str = "planned"
    synthesis: str = ""
    verified_claims: int = 0
    claims: list[dict[str, Any]] = field(default_factory=list)
    phase_history: list[str] = field(default_factory=list)
    targeted_searches: int = 0


def _claim_for(state: ResearchState, claim_id: str) -> dict[str, Any] | None:
    return next((claim for claim in state.claims if claim.get("id") == claim_id), None)


def _parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def check_freshness(state: ResearchState, max_age_days: int, *, now: datetime | None = None) -> list[str]:
    """Mark stale/invalid claim evidence and return affected claim IDs.

    A missing publication timestamp is unknown freshness, not proof of
    freshness.  It remains pending until another claim check can establish
    provenance.  Invalid and future-dated timestamps are failed because they
    cannot be used as temporal evidence.
    """
    if type(max_age_days) is not int or not 0 <= max_age_days <= 36_500:
        raise ValueError("invalid freshness window")
    if now is not None and not isinstance(now, datetime):
        raise ValueError("now must be a datetime")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    stale: list[str] = []
    for evidence in state.evidence:
        if not evidence.published_at:
            # Unknown publication time cannot preserve a previously verified
            # freshness claim. Downgrade it to pending until provenance is
            # supplied; otherwise stale evidence could remain trusted forever.
            if evidence.verification == "passed" or evidence.verified:
                evidence.verification = "pending"
                evidence.verified = False
                evidence.confidence = 0.0
                state.confidence[evidence.claim] = 0.0
                affected = evidence.claim_id or evidence.source_id
                stale.append(affected)
                claim = _claim_for(state, evidence.claim_id)
                if claim is not None:
                    claim["verification"] = "pending"
                    claim["evidence_refs"] = []
            continue
        published = _parse_date(evidence.published_at)
        if published is None or published > now:
            evidence.verification = "failed"
            evidence.verified = False
            evidence.confidence = 0.0
            state.confidence[evidence.claim] = 0.0
            affected = evidence.claim_id or evidence.source_id
        elif (now - published).total_seconds() > max_age_days * 86_400:
            evidence.verification = "pending"
            evidence.verified = False
            evidence.confidence = 0.0
            state.confidence[evidence.claim] = 0.0
            affected = evidence.claim_id or evidence.source_id
        else:
            continue
        # Return one marker per affected evidence entry.  A source can back
        # multiple claims, so callers must not mistake this for a set.
        stale.append(affected)
        claim = _claim_for(state, evidence.claim_id)
        if claim is not None:
            claim["verification"] = evidence.verification
            claim["evidence_refs"] = [] if evidence.verification != "passed" else claim.get("evidence_refs", [])
    state.verified_claims = sum(c.get("verification") == "passed" for c in state.claims)
    return stale


class ResearchWorkflow:
    """Bounded, citation-preserving deep-research state machine.

    Web calls are made only by ``_call``: this centralizes tool/web budget,
    deadline and cancellation enforcement and prevents targeted search from
    accidentally becoming an unbounded second workflow.
    """

    def __init__(
        self,
        gateway: ToolGateway,
        history: JsonlStore | None = None,
        knowledge: OkfStore | None = None,
        synthesizer: Callable[[ResearchState], Awaitable[str]] | None = None,
        artifacts: ArtifactStore | None = None,
        *,
        max_age_days: int = 365,
        max_targeted_searches: int = 1,
    ):
        if type(max_targeted_searches) is not int or not 0 <= max_targeted_searches <= 10:
            raise ValueError("max_targeted_searches is outside safe bounds")
        # Validate at construction, too, so a malformed workflow cannot run.
        if type(max_age_days) is not int or not 0 <= max_age_days <= 36_500:
            raise ValueError("max_age_days is outside safe bounds")
        self.gateway = gateway
        self.history = history
        self.knowledge = knowledge
        self.synthesizer = synthesizer
        self.artifacts = artifacts
        self.max_age_days = max_age_days
        self.max_targeted_searches = max_targeted_searches

    @staticmethod
    def _published(item: dict[str, Any]) -> str | None:
        for key in ("published_at", "publishedAt", "publication_date", "date"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value[:100]
        return None

    @staticmethod
    def _content(output: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(output, dict):
            metadata = output
            for key in ("content", "text", "body", "excerpt", "result"):
                if isinstance(output.get(key), str):
                    return output[key], metadata
            return "", metadata
        return str(output or ""), {}

    @staticmethod
    def _topic(text: str) -> tuple[str, bool]:
        # This is deliberately conservative: conjunctions and punctuation are
        # retained so unrelated claims do not become false contradictions.
        lowered = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        negative = bool(re.search(r"\b(no|not|never|without|false|denies|denied)\b", lowered))
        cleaned = re.sub(r"\b(is|are|was|were|has|have|no|not|never|without|false|denies|denied)\b", " ", lowered)
        return re.sub(r"\s+", " ", cleaned).strip(), negative

    async def run(
        self,
        question: str,
        budget: ComputeBudget | None = None,
        cancellation: CancellationToken | None = None,
        *,
        max_age_days: int | None = None,
    ) -> ResearchState:
        if not isinstance(question, str) or not question.strip() or len(question) > 20_000:
            raise ValueError("question must be bounded and non-empty")
        freshness_window = self.max_age_days if max_age_days is None else max_age_days
        if type(freshness_window) is not int or not 0 <= freshness_window <= 36_500:
            raise ValueError("invalid freshness window")
        budget = budget or ComputeBudget.for_mode(Mode.RESEARCH)
        started = time.monotonic()
        deadline = started + float(budget.max_wall_time_s)
        state = ResearchState(question=question, subquestions=[question])
        controller = ResearchController()
        terminal_budget = False
        terminal_cancelled = False

        def phase(name: str) -> None:
            target = ResearchPhase(name)
            if controller.phase != target:
                controller.transition(target)
            state.status = target.value
            state.phase_history.append(target.value)

        async def call(tool: str, arguments: dict[str, Any]):
            nonlocal terminal_budget, terminal_cancelled
            if cancellation and cancellation.cancelled:
                terminal_cancelled = True
                state.status = "cancelled"
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not budget.consume_tool(tool):
                terminal_budget = True
                state.status = "budget_exhausted"
                return None
            try:
                return await bounded(self.gateway.execute(tool, arguments), remaining, cancellation)
            except TimeoutError:
                terminal_budget = True
                state.status = "budget_exhausted"
            except asyncio.CancelledError:
                terminal_cancelled = True
                state.status = "cancelled"
            return None

        # asyncio is imported lazily to keep this module's core dependency-free.
        import asyncio

        phase("clarify")
        phase("questions")
        phase("search")
        search = await call("web.search", {"query": question})
        if search is None:
            if self.history:
                self.history.append({"kind": "research", "question": question, "state": asdict(state)})
            return state
        if not search.ok:
            state.status = "failed"
            state.open_questions.append(search.error or "search failed")
            return state

        phase("rank")
        raw_results = search.output.get("results", []) if isinstance(search.output, dict) else []
        sources: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in raw_results[:100] if isinstance(raw_results, list) else []:
            if isinstance(item, str):
                item = {"title": item, "url": ""}
            if not isinstance(item, dict):
                continue
            raw_url = item.get("url", "")
            # Search output is untrusted data; non-string URL values are
            # malformed results, not fetchable URLs (``str(None)`` would
            # otherwise create a bogus source and consume fetch budget).
            url = raw_url.strip()[:2_000] if isinstance(raw_url, str) else ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            host = url.split("//", 1)[-1].split("/", 1)[0].lower()
            # Reproducible, inspectable ranking signals (not a truth score).
            score = (2 if url.startswith("https://") else 0) + (1 if host and not host.startswith("www.") else 0) + (1 if item.get("title") else 0)
            sources.append({
                "title": str(item.get("title", ""))[:500],
                "url": url,
                "source_id": f"source-{len(sources)}",
                "rank_score": score,
                "ranking_signals": {"https": url.startswith("https://"), "non_www_host": bool(host and not host.startswith("www.")), "has_title": bool(item.get("title"))},
                "host": host,
                "published_at": self._published(item),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": None,
            })
        state.sources = sorted(sources, key=lambda item: (-item["rank_score"], item["source_id"]))[:min(10, max(1, budget.web_calls))]

        phase("fetch")
        # Three broad fetches is an intentional workflow bound, in addition to
        # the caller's total tool/web budgets.
        for source in state.sources[:3]:
            if not source["url"]:
                state.open_questions.append(f"No URL for {source['source_id']}")
                continue
            fetched = await call("web.fetch", {"url": source["url"]})
            if fetched is None:
                if terminal_budget:
                    state.open_questions.append("Research fetch budget or deadline exhausted")
                    break
                continue
            if not fetched.ok:
                state.open_questions.append(f"Could not read {source['source_id']}: {fetched.error or 'fetch failed'}")
                continue
            content, metadata = self._content(fetched.output)
            content = content[:4_000]
            fetched_date = self._published(metadata) if isinstance(metadata, dict) else None
            source["published_at"] = fetched_date or source.get("published_at")
            source["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
            claims_data = metadata.get("claims") if isinstance(metadata, dict) else None
            claims = claims_data if isinstance(claims_data, list) else []
            if not claims:
                first = re.split(r"(?<=[.!?])\s+", content.strip(), maxsplit=1)[0][:500] or "source content retrieved"
                claims = [{"text": first}]
            for claim_data in claims[:20]:
                if isinstance(claim_data, str):
                    claim_data = {"text": claim_data}
                if not isinstance(claim_data, dict) or not str(claim_data.get("text", "")).strip():
                    continue
                text = str(claim_data["text"])[:500]
                relation = str(claim_data.get("relation", source.get("relation", "supports"))).lower()
                if relation not in {"supports", "contradicts"}:
                    relation = "supports"
                claim_id = f"claim-{len(state.claims) + 1}"
                claim_excerpt = str(claim_data.get("excerpt", content))[:4_000]
                evidence_ref = self.artifacts.put(claim_excerpt, origin="web", trust="untrusted") if self.artifacts else ""
                evidence = Evidence(claim=text, source_id=source["source_id"], excerpt=claim_excerpt, relation=relation, published_at=source.get("published_at"), retrieved_at=datetime.now(timezone.utc).isoformat(), claim_id=claim_id)
                state.evidence.append(evidence)
                state.claims.append({"id": claim_id, "text": text, "source_refs": [source["source_id"]], "evidence_refs": [evidence_ref] if evidence_ref else [], "verification": "pending", "relation": relation, "topic": str(claim_data.get("topic", ""))[:500], "contradicts": str(claim_data.get("contradicts", ""))[:100]})

        phase("extract_claims")
        if not state.evidence:
            state.open_questions.append("No source evidence was collected")
        # Every verification is claim-specific; never zip unrelated lists.
        for evidence in state.evidence:
            claim = _claim_for(state, evidence.claim_id)
            source = next((item for item in state.sources if item["source_id"] == evidence.source_id), None)
            claim_terms = set(re.findall(r"[a-z0-9]{3,}", evidence.claim.lower()))
            excerpt_terms = set(re.findall(r"[a-z0-9]{3,}", evidence.excerpt.lower()))
            overlap = len(claim_terms & excerpt_terms)
            passed = bool(source and source.get("url") and evidence.excerpt.strip() and evidence.claim.strip() and claim_terms and overlap >= max(1, len(claim_terms) // 3))
            evidence.verified = passed
            evidence.verification = "passed" if passed else "failed"
            evidence.confidence = 0.8 if passed else 0.0
            if claim is not None:
                claim["verification"] = evidence.verification
                if not passed:
                    claim["evidence_refs"] = []
            state.confidence[evidence.claim] = evidence.confidence
        state.verified_claims = sum(c.get("verification") == "passed" for c in state.claims)
        stale = check_freshness(state, freshness_window)
        freshness_pending_ids = set(stale)
        for claim_id in stale:
            if claim_id.startswith("claim-"):
                state.open_questions.append(f"Freshness requires review for {claim_id}")

        phase("gap_analysis")
        for claim in state.claims:
            if claim.get("verification") != "passed":
                if str(claim.get("id", "")) in freshness_pending_ids:
                    continue
                text = str(claim.get("text", ""))[:300]
                if text and f"Verify claim: {text}" not in state.open_questions:
                    state.open_questions.append(f"Verify claim: {text}")
        state.open_questions = list(dict.fromkeys(state.open_questions))[:20]

        phase("contradictions")
        passed_claims = [c for c in state.claims if c.get("verification") == "passed"]
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for claim in passed_claims:
            explicit = claim.get("relation") == "contradicts"
            topic = str(claim.get("topic", "")).strip().lower() or self._topic(str(claim.get("text", "")))[0]
            negative = self._topic(str(claim.get("text", "")))[1]
            groups.setdefault(topic, {"supports": [], "contradicts": []})["contradicts" if explicit or negative else "supports"].append(claim)
        for topic, sides in groups.items():
            # An explicit contradicts relation is itself an unresolved conflict,
            # even when the opposing source was not retrieved.  This keeps a
            # lone adversarial/negative observation from being promoted.
            if not topic or not sides["contradicts"]:
                continue
            if not sides["supports"] and not any(c.get("relation") == "contradicts" for c in sides["contradicts"]):
                continue
            ids = [c["id"] for c in sides["supports"] + sides["contradicts"]]
            entry = {"topic": topic[:500], "claim_ids": ids, "status": "needs_review", "resolution": ""}
            state.contradictions.append(entry)
            for claim_id in ids:
                claim = _claim_for(state, claim_id)
                evidence = next((e for e in state.evidence if e.claim_id == claim_id), None)
                if claim is not None:
                    claim["verification"] = "pending"
                if evidence is not None:
                    evidence.verification = "pending"
                    evidence.verified = False
                    evidence.confidence = 0.0
                    state.confidence[evidence.claim] = 0.0
        state.verified_claims = sum(c.get("verification") == "passed" for c in state.claims)
        for contradiction in state.contradictions:
            prompt = f"Resolve contradiction about {contradiction['topic']}"
            if prompt not in state.open_questions:
                state.open_questions.append(prompt)

        # Missing publication dates are a provenance warning, not a search
        # query. Do not spend the bounded targeted-search budget on a question
        # that the same source cannot answer without new metadata.
        needs_target = bool(state.contradictions or any(not question.startswith("Freshness requires review") for question in state.open_questions))
        if needs_target and self.max_targeted_searches:
            phase("targeted_search")
            query = (state.open_questions[0] if state.open_questions else f"Resolve contradiction about {state.contradictions[0]['topic']}")[:500]
            targeted = await call("web.search", {"query": query})
            if targeted is not None:
                state.targeted_searches = 1
                if targeted.ok:
                    target_results = targeted.output.get("results", []) if isinstance(targeted.output, dict) else []
                    for item in target_results[:3] if isinstance(target_results, list) else []:
                        if not isinstance(item, dict):
                            continue
                        raw_url = item.get("url", "")
                        url = raw_url.strip()[:2_000] if isinstance(raw_url, str) else ""
                        if url and any(s.get("url") == url for s in state.sources):
                            continue
                        source_id = f"targeted-{len([s for s in state.sources if str(s.get('source_id', '')).startswith('targeted-')])}"
                        state.sources.append({"title": str(item.get("title", ""))[:500], "url": url, "source_id": source_id, "rank_score": 0, "ranking_signals": {}, "host": url.split("//", 1)[-1].split("/", 1)[0].lower(), "published_at": self._published(item), "retrieved_at": datetime.now(timezone.utc).isoformat(), "content_hash": None})
                elif targeted.error:
                    state.open_questions.append(f"Targeted search failed: {targeted.error}")
        else:
            phase("synthesize")

        # The branch above is intentionally explicit so phase history remains a
        # valid state-machine path whether or not targeted search is possible.
        if controller.phase == ResearchPhase.TARGETED_SEARCH:
            phase("synthesize")
        synthesis_failed = False
        synthesis_cancelled = False
        if state.evidence:
            if self.synthesizer:
                try:
                    result = await bounded(self.synthesizer(state), max(0.0, deadline - time.monotonic()), cancellation)
                    state.synthesis = str(result or "")[:12_000]
                except TimeoutError:
                    terminal_budget = True
                    state.status = "budget_exhausted"
                except asyncio.CancelledError:
                    state.status = "cancelled"
                    synthesis_cancelled = True
                except Exception as exc:
                    # A synthesizer is an extension point and its failures
                    # must remain a bounded workflow result, not escape while
                    # bypassing history/audit and terminal status handling.
                    state.status = "failed"
                    synthesis_failed = True
                    try:
                        reason = redact_text(str(exc)[:500], max_chars=None)
                    except Exception:
                        reason = "extension failure"
                    state.open_questions.append(f"Synthesis failed: {reason}")
            else:
                state.synthesis = "\n\n".join(("[UNVERIFIED] " if e.verification != "passed" else "") + f"[{e.claim_id}] [{e.source_id}] {e.excerpt[:1_500]}" for e in state.evidence)[:12_000]
                if state.open_questions:
                    state.synthesis += "\n\n[OPEN QUESTIONS] " + "; ".join(state.open_questions[:10])
        phase("verify")
        phase("cite")
        phase("complete")
        state.verified_claims = sum(c.get("verification") == "passed" for c in state.claims)
        # Never present a partial, budget-aborted answer as a normal success.
        if terminal_budget:
            state.status = "budget_exhausted"
        elif terminal_cancelled or synthesis_cancelled or synthesis_failed:
            # Preserve terminal failures from bounded extension points.
            state.status = "cancelled" if (terminal_cancelled or synthesis_cancelled) else "failed"
        else:
            # ``complete`` is a verified research result, not merely a response
            # produced by the synthesizer. Pending/failed claims and unresolved
            # contradictions must remain visibly incomplete.
            claims_complete = bool(state.claims) and all(
                claim.get("verification") == "passed" for claim in state.claims
            )
            state.status = "complete" if state.synthesis and claims_complete and not state.contradictions else "incomplete"
        if self.knowledge and state.synthesis and state.verified_claims and not state.contradictions:
            self.knowledge.put_candidate("research-" + question, question, state.synthesis, source="research-workflow:verified-claims")
        if self.history:
            self.history.append({"kind": "research", "question": question, "state": asdict(state)})
        return state

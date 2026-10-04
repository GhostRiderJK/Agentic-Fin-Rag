"""Multi-agent orchestrator (PRD v2 §6, §7, §9).

Flow:
  1. Query understanding (entities, time, intent, route decision)
  2. If metric-lookup → SQL tool-calling route (no retrieval needed)
     Else → document route (hybrid retrieval)
  3. (Optional) If comparison → fan out sub-queries, merge contexts
  4. Compose answer via the analyst agent (with tool-calling loop)
  5. Citation verification (NLI) → reject unsupported claims
  6. Format final answer with bullet summary + sources

Returns an ``AnswerResponse`` consumed by the chat API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.citation import CitationVerifier, VerificationOutcome
from app.core.logging import get_logger
from app.providers.llm import BaseLLM, get_llm
from app.reasoning.agents import (
    SYSTEM_PROMPT,
    build_messages,
    run_tool_loop,
)
from app.reasoning.tools import build_dispatcher
from app.retrieval.hybrid import HybridRetriever, RetrievalResult
from app.retrieval.query_understanding import (
    QueryPlan,
    rewrite_query_with_history,
    understand_query,
)

log = get_logger(__name__)


@dataclass
class AnswerRequest:
    question: str
    workspace_id: str
    conversation_id: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    mode: str = "auto"  # auto | document_only | structured_only


@dataclass
class AnswerResponse:
    answer: str
    citations: list[dict]
    plan: dict
    tool_results: list[dict]
    verification: dict
    retrieval: dict
    latency_ms: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Orchestrator:
    def __init__(
        self,
        db: Session,
        *,
        llm: BaseLLM | None = None,
        verifier: CitationVerifier | None = None,
    ) -> None:
        self.db = db
        self.llm = llm or get_llm()
        self.verifier = verifier or CitationVerifier(self.llm)

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        import time

        t0 = time.perf_counter()
        notes: list[str] = []

        # 1. Conversation-aware query rewrite
        rewritten = rewrite_query_with_history(req.question, req.history)
        plan = understand_query(rewritten)
        if rewritten != req.question:
            notes.append(f"rewrote query: {rewritten}")

        # 2. Retrieval (document route)
        # For comparison questions, fan out per-company so we get balanced
        # sources from each named company instead of only the top-global hits.
        retriever = HybridRetriever(self.db, workspace_id=req.workspace_id)
        if plan.is_comparison and len(plan.companies) >= 2:
            retrieval = _comparison_retrieve(
                retriever, rewritten, plan,
                per_company=3, total_cap=8,
            )
            notes.append(f"comparison fan-out: {len(plan.companies)} companies")
        else:
            retrieval = retriever.retrieve(rewritten, plan=plan)
        if not retrieval.chunks and not plan.is_metric_lookup:
            # No context at all
            return AnswerResponse(
                answer=(
                    "I don't have enough information in the uploaded documents "
                    "to answer this. Try uploading the relevant report, or "
                    "rephrasing the question."
                ),
                citations=[], plan=_plan_dict(plan), tool_results=[],
                verification={"outcome": "no_context"},
                retrieval={"n_chunks": 0}, latency_ms=int((time.perf_counter() - t0) * 1000),
                notes=notes,
            )

        # 3. Tool dispatcher for structured route
        dispatch, tool_specs = build_dispatcher(self.db, workspace_id=req.workspace_id)
        # If pure metric-lookup and no relevant docs, hint to tools
        tool_hint = ""
        if plan.is_metric_lookup:
            tool_hint = (
                "Hint: This looks like a metric lookup. Prefer calling "
                "get_financial_metric / compare_metrics before answering."
            )

        # 4. Compose answer (with tool-calling loop)
        context = retrieval.context_text(max_chunks=8) if retrieval.chunks else ""
        citations = retrieval.citation_list()
        messages = build_messages(
            question=req.question,
            context=context,
            citations=citations,
            history=req.history,
            tool_hint=tool_hint,
        )
        draft_text, tool_results = run_tool_loop(
            self.llm, messages, tools=tool_specs, dispatch=dispatch
        )

        # 5. Citation verification (NLI) on draft vs retrieved chunks
        verification = {"outcome": VerificationOutcome.NOT_APPLICABLE.value}
        if retrieval.chunks and draft_text:
            v = self.verifier.verify(draft_text, retrieval.chunks[:8])
            verification = {
                "outcome": v.outcome.value,
                "supported": v.supported,
                "unsupported": v.unsupported,
                "note": v.note,
            }
            if v.outcome == VerificationOutcome.REJECTED:
                draft_text = (
                    f"{draft_text}\n\n"
                    "⚠️ Citation check flagged unsupported claims "
                    f"({len(v.unsupported)}). Treat the answer above with "
                    "caution and verify against the cited sources."
                )
                notes.append("citation_verification_rejected")

        # 6. Final answer already includes citations via [1], [2]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return AnswerResponse(
            answer=draft_text,
            citations=citations,
            plan=_plan_dict(plan),
            tool_results=tool_results,
            verification=verification,
            retrieval={
                "n_chunks": len(retrieval.chunks),
                "top_chunk_scores": [
                    {"dense": c.dense_score, "bm25": c.bm25_score,
                     "rerank": c.rerank_score}
                    for c in retrieval.chunks[:5]
                ],
            },
            latency_ms=latency_ms,
            notes=notes,
        )


def _plan_dict(plan: QueryPlan) -> dict:
    return {
        "companies": plan.companies,
        "tickers": plan.tickers,
        "years": plan.years,
        "quarters": plan.quarters,
        "doc_types": [d.value for d in plan.doc_types],
        "sections": plan.sections,
        "is_metric_lookup": plan.is_metric_lookup,
        "is_comparison": plan.is_comparison,
        "sub_queries": plan.sub_queries,
    }


def _comparison_retrieve(
    retriever: HybridRetriever,
    question: str,
    plan: QueryPlan,
    *,
    per_company: int = 3,
    total_cap: int = 8,
) -> RetrievalResult:
    """Run one retrieval per named company (each filtered to that company's
    docs), then merge the top ``per_company`` chunks from each into a single
    RetrievalResult capped at ``total_cap``.

    This produces balanced sources across companies instead of only the
    globally-top hits (which usually come from a single dominant company).
    """
    from app.retrieval.hybrid import RetrievedChunk
    from app.retrieval.query_understanding import QueryPlan as QP

    merged: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    for company in plan.companies:
        # Build a per-company plan: same query + intent, but pinned to one
        # company so the metadata filter scopes the search.
        co_plan = QP(
            question=question,
            companies=[company],
            tickers=[],
            years=plan.years,
            quarters=plan.quarters,
            doc_types=plan.doc_types,
            sections=plan.sections,
            is_metric_lookup=plan.is_metric_lookup,
            is_comparison=True,
        )
        try:
            res = retriever.retrieve(question, plan=co_plan, rerank_top_k=per_company)
        except Exception as e:
            log.warning("comparison sub-retrieve failed for %s: %s", company, e)
            continue
        for c in res.chunks[:per_company]:
            if c.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(c.chunk_id)
            merged.append(c)
            if len(merged) >= total_cap:
                break
        if len(merged) >= total_cap:
            break
    out = RetrievalResult(query=question, plan=plan, chunks=merged[:total_cap],
                          note="comparison fan-out")
    for c in out.chunks:
        c.sources = out.citation_list()
    return out

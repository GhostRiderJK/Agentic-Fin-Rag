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

import re
from dataclasses import asdict, dataclass, field

from app.citation import (
    CitationVerifier, VerificationOutcome, canonicalize_citation_markers,
)
from app.core.logging import get_logger
from app.db.mongodb.tenant_context import TenantContext
from app.providers.llm import BaseLLM, Message, get_llm
from app.reasoning.agents import (
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
    include_evidence_trace: bool = False


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
    generation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Orchestrator:
    def __init__(
        self,
        ctx: TenantContext,
        *,
        llm: BaseLLM | None = None,
        verifier: CitationVerifier | None = None,
    ) -> None:
        self.ctx = ctx
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

        # Everything this request touches is scoped to its workspace.
        ctx = self.ctx.with_workspace(req.workspace_id)

        # 2. Retrieval (document route)
        # For comparison questions, fan out per-company so we get balanced
        # sources from each named company instead of only the top-global hits.
        retriever = HybridRetriever(ctx)
        if plan.is_comparison and len(plan.companies) >= 2:
            retrieval = _comparison_retrieve(
                retriever, rewritten, plan,
                per_company=3, total_cap=8,
            )
            notes.append(f"comparison fan-out: {len(plan.companies)} companies")
        elif len(plan.years) >= 2:
            retrieval = _multi_year_retrieve(
                retriever, rewritten, plan, per_year=4, total_cap=8,
            )
            notes.append(f"year fan-out: {len(plan.years)} report years")
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
                generation={"provider": self.llm.name,
                            "model": getattr(self.llm, "_model", None),
                            "is_stub": self.llm.name == "stub"},
            )

        # 3. Tool dispatcher for structured route
        dispatch, tool_specs = build_dispatcher(ctx)
        # If pure metric-lookup and no relevant docs, hint to tools
        tool_hint = ""
        if plan.is_metric_lookup:
            tool_hint = (
                "Hint: This looks like a metric lookup. Prefer calling "
                "get_financial_metric / compare_metrics before answering."
            )

        # 4. Compose answer (with tool-calling loop)
        context, generation_chunks = retrieval.context_with_chunks(max_chunks=8)
        citations = retrieval.citation_list(chunks=generation_chunks)
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
        generated_draft = draft_text
        draft_text = canonicalize_citation_markers(draft_text)
        citation_format_retry_count = 0

        # 5. Citation verification (NLI) on draft vs retrieved chunks
        is_stub = self.llm.name == "stub"
        verification = {
            "outcome": "stub_not_evaluated" if is_stub
            else VerificationOutcome.NOT_APPLICABLE.value,
        }
        if not is_stub and generation_chunks and draft_text:
            v = self.verifier.verify(draft_text, generation_chunks)
            citation_format_invalid = any(
                reason in v.note
                for reason in (
                    "claim has no citation marker",
                    "claim has a malformed citation marker",
                    "claim references an unavailable citation",
                    "cited evidence is empty",
                )
            )
            if citation_format_invalid:
                # Some chat models emit their own web-style reference marks
                # (for example, 【2】) despite the documented [n] format.
                # Give one format-only retry against the same exact context;
                # the normal verifier still checks every claim and citation.
                repair_messages = [
                    *messages,
                    Message(role="assistant", content=draft_text),
                    Message(
                        role="user",
                        content=(
                            "Rewrite the answer using only the retrieved source "
                            "items already provided. Preserve its factual claims "
                            "and values; do not add facts. Put a valid citation "
                            "immediately after each factual claim using only "
                            "ASCII square-bracket syntax such as [1]. The number "
                            "must identify a supplied context item that supports "
                            "that claim. Do not use Unicode brackets, web-style "
                            "markers, footnotes, or URLs. If a claim is not "
                            "supported, omit it; if no answer remains, return "
                            "the application's abstention sentence. Output only "
                            "the rewritten answer."
                        ),
                    ),
                ]
                try:
                    repaired = self.llm.chat(
                        repair_messages, tools=None, temperature=0.0
                    ).content or ""
                    if repaired.strip():
                        repaired = canonicalize_citation_markers(repaired)
                        repaired_verification = self.verifier.verify(
                            repaired, generation_chunks
                        )
                        draft_text = repaired
                        v = repaired_verification
                        citation_format_retry_count += 1
                        notes.append("citation_format_retry")
                except Exception as exc:  # fail closed using the first result
                    log.warning("citation format retry failed: %s", type(exc).__name__)
            verification = {
                "outcome": v.outcome.value,
                "supported": v.supported,
                "unsupported": v.unsupported,
                "note": v.note,
            }
            if v.outcome in {
                VerificationOutcome.PARTIAL,
                VerificationOutcome.REJECTED,
            }:
                # Never return factual claims that the citation verifier could
                # not support. The verifier is approximate, so a positive
                # result is still a signal rather than proof; a negative result
                # must fail closed.
                draft_text = (
                    "I can't verify a complete answer from the retrieved "
                    "source passages, so I can't provide a reliable financial "
                    "answer."
                )
                notes.append("citation_verification_abstained")

        # Only return source records explicitly referenced by the generated
        # answer. References are one-based indexes into the exact context list.
        cited_indexes = {int(index) for index in re.findall(r"\[(\d+)\]", draft_text)}
        used_citations = [c for c in citations if c["index"] in cited_indexes]

        retrieval_summary = {
            "n_chunks": len(retrieval.chunks),
            "candidate_counts": {
                "retrieved": retrieval.retrieved_candidate_count,
                "reranked": retrieval.reranked_candidate_count,
                "returned_by_api": len(retrieval.chunks),
                "included_in_final_context": len(generation_chunks),
            },
            "top_chunk_scores": [
                {"dense": c.dense_score, "bm25": c.bm25_score,
                 "rerank": c.rerank_score}
                for c in generation_chunks[:5]
            ],
        }
        if req.include_evidence_trace:
            context_index_by_chunk = {
                chunk.chunk_id: index
                for index, chunk in enumerate(generation_chunks, start=1)
            }
            context_text_by_chunk = {
                chunk.chunk_id: chunk.evidence_text()
                for chunk in generation_chunks
            }
            all_candidate_citations = retrieval.citation_list()
            retrieval_summary.update({
                "query": retrieval.query,
                "plan": _plan_dict(retrieval.plan),
                "generation_context": context,
                "generated_draft": generated_draft,
                "final_draft": draft_text,
                "citation_format_retry_count": citation_format_retry_count,
                "parsed_citation_indexes": sorted({
                    int(index) for index in re.findall(r"\[(\d+)\]", draft_text)
                }),
                "candidate_counts": retrieval_summary["candidate_counts"],
                "filter_diagnostics": retrieval.filter_diagnostics,
                "candidate_diagnostics": retrieval.candidate_diagnostics,
                "generation_messages": [
                    {"role": m.get("role"), "content": m.get("content")}
                    for m in messages
                ],
                "candidates": [
                    {
                        **citation,
                        "text": chunk.parent_text or chunk.text,
                        "tables_md": chunk.tables_md,
                        "dense_score": chunk.dense_score,
                        "bm25_score": chunk.bm25_score,
                        "rerank_score": chunk.rerank_score,
                        "context_supplied": chunk.chunk_id in context_index_by_chunk,
                        "generation_context_text": context_text_by_chunk.get(
                            chunk.chunk_id
                        ),
                        "generation_citation_index": context_index_by_chunk.get(
                            chunk.chunk_id
                        ),
                    }
                    for citation, chunk in zip(
                        all_candidate_citations, retrieval.chunks, strict=True
                    )
                ],
            })

        # 6. Keep generation-provider identity visible to evaluators. A test
        # stub is never presented as a successful factual answer.
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return AnswerResponse(
            answer=draft_text,
            citations=used_citations,
            plan=_plan_dict(plan),
            tool_results=tool_results,
            verification=verification,
            retrieval=retrieval_summary,
            latency_ms=latency_ms,
            notes=notes,
            generation={"provider": self.llm.name,
                        "model": getattr(self.llm, "_model", None),
                        "is_stub": is_stub},
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
    retrieved_count = reranked_count = 0
    candidate_diagnostics: list[dict] = []
    filter_diagnostics: list[dict] = []
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
        retrieved_count += res.retrieved_candidate_count or 0
        reranked_count += res.reranked_candidate_count or 0
        candidate_diagnostics.extend(res.candidate_diagnostics)
        if res.filter_diagnostics:
            filter_diagnostics.append(res.filter_diagnostics)
        for c in res.chunks[:per_company]:
            if c.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(c.chunk_id)
            merged.append(c)
            if len(merged) >= total_cap:
                break
        if len(merged) >= total_cap:
            break
    out = RetrievalResult(
        query=question, plan=plan, chunks=merged[:total_cap],
        note="comparison fan-out",
        retrieved_candidate_count=retrieved_count,
        reranked_candidate_count=reranked_count,
        candidate_diagnostics=candidate_diagnostics,
        filter_diagnostics={"subqueries": filter_diagnostics},
    )
    for c in out.chunks:
        c.sources = out.citation_list()
    return out


def _multi_year_retrieve(
    retriever: HybridRetriever,
    question: str,
    plan: QueryPlan,
    *,
    per_year: int = 4,
    total_cap: int = 8,
) -> RetrievalResult:
    """Retrieve each requested report year independently, then merge.

    A global top-k often fills with one report, leaving a comparison answer
    without evidence for the other year. Per-year filtering guarantees each
    available year gets a chance to contribute evidence.
    """
    from app.retrieval.hybrid import RetrievedChunk
    from app.retrieval.query_understanding import QueryPlan as QP

    merged: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    retrieved_count = reranked_count = 0
    candidate_diagnostics: list[dict] = []
    filter_diagnostics: list[dict] = []
    for year in plan.years:
        year_question = re.sub(r"\b20\d{2}\b", " ", question).strip()
        year_question = f"{year_question} {year}".strip()
        year_plan = QP(
            question=year_question,
            companies=plan.companies,
            tickers=plan.tickers,
            years=[year],
            quarters=plan.quarters,
            doc_types=plan.doc_types,
            sections=plan.sections,
            is_metric_lookup=plan.is_metric_lookup,
            is_comparison=True,
        )
        try:
            res = retriever.retrieve(year_question, plan=year_plan,
                                     rerank_top_k=per_year)
        except Exception as e:
            log.warning("year sub-retrieve failed for %s: %s", year, e)
            continue
        retrieved_count += res.retrieved_candidate_count or 0
        reranked_count += res.reranked_candidate_count or 0
        candidate_diagnostics.extend(res.candidate_diagnostics)
        if res.filter_diagnostics:
            filter_diagnostics.append(res.filter_diagnostics)
        for chunk in res.chunks[:per_year]:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                merged.append(chunk)
        if len(merged) >= total_cap:
            break
    out = RetrievalResult(
        query=question, plan=plan, chunks=merged[:total_cap],
        note="year fan-out",
        retrieved_candidate_count=retrieved_count,
        reranked_candidate_count=reranked_count,
        candidate_diagnostics=candidate_diagnostics,
        filter_diagnostics={"subqueries": filter_diagnostics},
    )
    for chunk in out.chunks:
        chunk.sources = out.citation_list()
    return out

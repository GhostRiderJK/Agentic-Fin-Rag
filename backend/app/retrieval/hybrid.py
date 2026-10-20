"""Hybrid retrieval: BM25 (keyword) + dense (vector) + metadata filtering,
followed by reranking and parent expansion.

This is the document-route side of the dual knowledge system. The structured
(metric-lookup) route is implemented in ``app/reasoning/tools``.

Pipeline:

  query plan
    ↓
  metadata prefilter (companies / years / doc_type / quarter)
    ↓
  dense search on Qdrant (filtered)  ──┐
  BM25 over filtered chunks           ──┤  weighted fusion
                                       ↓
                                reranker (top-k)
                                       ↓
              optional per-(document, physical page) quota
                    (RETRIEVAL_PAGE_QUOTA, 0 = disabled)
                                       ↓
                          parent expansion (LLM sees parents)
                                       ↓
                                RetrievalResult
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.errors import ProviderError, ValidationError
from app.core.logging import get_logger
from app.db.enums import DocumentStatus
from app.db.mongodb.repositories import ChunkRepository, DocumentRepository
from app.db.mongodb.runtime import run_sync
from app.db.mongodb.tenant_context import TenantContext
from app.providers import get_embeddings, get_reranker, get_vectordb
from app.retrieval.query_understanding import QueryPlan, understand_query

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    parent_id: str | None
    document_id: str
    text: str
    parent_text: str
    tables_md: str
    company: str
    year: int | None
    quarter: str
    doc_type: str
    section: str
    page: int
    source_filename: str = ""
    printed_page: int | None = None
    physical_page: int | None = None
    page_number_type: str = "physical"
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    sources: list[dict] = field(default_factory=list)

    def evidence_text(self) -> str:
        """Return matched child evidence followed by its parent context.

        Parent passages are intentionally bounded during ingestion. A matched
        child can therefore contain a relevant tail passage absent from its
        parent; retain that exact retrieval hit for generation and verification.
        """
        child = self.text.strip()
        parent = self.parent_text.strip()
        if child and child not in parent:
            return f"{child}\n\n{parent}" if parent else child
        return parent or child


@dataclass
class RetrievalResult:
    query: str
    plan: QueryPlan
    chunks: list[RetrievedChunk]
    note: str = ""
    retrieved_candidate_count: int | None = None
    reranked_candidate_count: int | None = None
    candidate_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    filter_diagnostics: dict[str, Any] = field(default_factory=dict)

    def context_text(
        self,
        max_chunks: int | None = None,
        *,
        max_chars: int = 24_000,
    ) -> str:
        """Render a context string for the LLM, with citation markers.

        Hard caps the total length via ``max_chars`` (default 24k chars ≈ 6k
        tokens) so we never blow the LLM's context window. Each chunk is also
        truncated individually to avoid one giant parent chunk (e.g. a JPM
        10-K page) eating the whole budget.
        """
        return self.context_with_chunks(
            max_chunks=max_chunks, max_chars=max_chars
        )[0]

    def context_with_chunks(
        self,
        max_chunks: int | None = None,
        *,
        max_chars: int = 24_000,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Render context and return exactly the chunks included in it."""
        items = self.chunks if max_chunks is None else self.chunks[:max_chunks]
        if max_chars <= 0 or not items:
            return "", []
        base_chunk_budget = max(800, max_chars // len(items))
        parts: list[str] = []
        included: list[RetrievedChunk] = []
        used = 0
        for c in items:
            if used >= max_chars:
                break
            tag = f"[{len(included) + 1}]"
            head = (f"{tag} {c.company or 'Unknown'} "
                    f"{c.year or ''} {c.quarter or ''} {c.doc_type}".strip())
            available = max_chars - used - 16 - len(head) - 1
            if available <= 0:
                break
            per_chunk_char_budget = min(base_chunk_budget, available)
            # Truncate parent_text to its fair share of the budget.
            body = c.evidence_text()[:per_chunk_char_budget]
            table_prefix = "\n\nTable:\n"
            table_budget = min(
                max(0, per_chunk_char_budget - len(body)),
                max(0, available - len(body) - len(table_prefix)),
            )
            tables = c.tables_md[:table_budget] if c.tables_md and table_budget > 200 else ""
            piece = f"{head}\n{body}"
            if tables:
                piece += f"{table_prefix}{tables}"
            parts.append(piece)
            # Verification and citations must inspect the exact evidence
            # serialized into the prompt, including its actual truncation.
            supplied_evidence = body + (f"{table_prefix}{tables}" if tables else "")
            included.append(replace(
                c, text=supplied_evidence, parent_text=supplied_evidence,
                tables_md="",
            ))
            used += len(piece) + 16  # separator overhead
        return "\n\n---\n\n".join(parts), included

    def citation_list(
        self,
        max_chunks: int | None = None,
        *,
        chunks: list[RetrievedChunk] | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        selected = chunks if chunks is not None else (
            self.chunks if max_chunks is None else self.chunks[:max_chunks]
        )
        for i, c in enumerate(selected, start=1):
            out.append({
                "index": i,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "company": c.company,
                "year": c.year,
                "quarter": c.quarter,
                "doc_type": c.doc_type,
                "section": c.section,
                "page": c.page,
                "page_number_type": c.page_number_type,
                "printed_page": c.printed_page,
                "physical_page": c.physical_page,
                "snippet": c.evidence_text()[:240],
            })
        return out


def _section_boost_for(question: str) -> str | None:
    """Return the canonical section keyword to boost for this question, or
    None if no clear section concept is present.

    This is the missing signal a pure lexical/dense ranker misses: a question
    like "What are the risks?" should surface the RISK FACTORS section even
    if its vocabulary has little overlap with the question.
    """
    q = question.lower()
    rules = [
        ("RISK", ["risk", "headwinds", "threats", "concerns"]),
        ("OUTLOOK", ["outlook", "guidance", "forecast", "forward-looking",
                     "expect", "future", "next year", "2026", "2027"]),
        ("STRATEGY", ["strategy", "strategic", "initiative"]),
        ("MANAGEMENT", ["management", "md&a", "discussion and analysis"]),
        ("CASH FLOW", ["cash flow", "capex", "capital expenditure"]),
        ("BALANCE SHEET", ["balance sheet", "debt", "liabilit", "equity"]),
        ("INCOME STATEMENT", ["income statement"]),
        ("SUSTAINABILITY", ["sustainability", "esg", "carbon", "climate"]),
        ("CORPORATE GOVERNANCE", ["governance", "board of", "director"]),
    ]
    for canon, kws in rules:
        if any(kw in q for kw in kws):
            return canon
    return None


def _lexical_query(question: str) -> str:
    """Expand common finance abbreviations for lexical retrieval/reranking."""
    import re

    additions: list[str] = []
    aliases = (
        (r"\baum\b", "assets under management gross assets under management"),
        (r"\btsr\b", "total shareholder return annualized portfolio return"),
        (r"\bemployee count\w*\b", "workforce total employees headcount"),
        (r"\bcompanies established\b", "new companies established founded"),
        (r"\bportfolio compan\w*\b", "total portfolio companies"),
    )
    for pattern, expansion in aliases:
        if re.search(pattern, question, re.IGNORECASE):
            additions.append(expansion)
    return " ".join([question, *additions])


def _query_term_coverage(question: str, text: str) -> float:
    """Coverage of informative original query terms in evidence text."""
    stop = {"a", "an", "and", "are", "as", "at", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "the", "to",
            "was", "were", "what", "when", "where", "which", "who",
            "with"}
    terms = {t for t in HybridRetriever._tok(question) if len(t) > 1 and t not in stop}
    if not terms:
        return 0.0
    return len(terms & set(HybridRetriever._tok(text))) / len(terms)


def _sort_reranked_chunks(
    chunks: list[RetrievedChunk],
    position_by_chunk: dict[str, int],
    reranked_positions: set[int],
) -> None:
    """Sort by reranker scores without mixing them with fused scores."""
    if not reranked_positions:
        chunks.sort(key=lambda chunk: chunk.final_score, reverse=True)
        return
    chunks.sort(
        key=lambda chunk: (
            chunk.rerank_score,
            chunk.final_score,
        ) if position_by_chunk[chunk.chunk_id] in reranked_positions
        else (float("-inf"), chunk.final_score),
        reverse=True,
    )


def _page_group_key(chunk: RetrievedChunk) -> tuple[str, int] | None:
    """Identity + physical page. Never the printed label: two reports that both
    print "page 6" are different evidence, and one report's page 6 is one page."""
    page = chunk.physical_page
    if isinstance(page, bool) or not isinstance(page, int):
        return None
    return (chunk.document_id, page)


def _page_diversity_select(
    chunks: list[RetrievedChunk], limit: int, *, per_page: int
) -> list[RetrievedChunk]:
    """Greedy re-selection of the returned set under a per-page quota.

    Chunks arrive in final ranking order, so each page group keeps its best
    member first. This mirrors the measured experiment exactly: candidates with
    no usable physical page carry no group key, are never merged with each
    other, and wait in the same overflow queue as quota-rejected siblings, so
    the queue is drained in rank order and the set never ends up short. Only
    chunks from the input can be returned.
    """
    chosen: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    page_counts: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        key = _page_group_key(chunk)
        if key is None:
            overflow.append(chunk)
            continue
        if page_counts.get(key, 0) >= per_page:
            overflow.append(chunk)
            continue
        page_counts[key] = page_counts.get(key, 0) + 1
        chosen.append(chunk)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        chosen.extend(overflow[: limit - len(chosen)])
    return chosen[:limit]


class HybridRetriever:
    """Stateful per-query retriever.

    Build one per query. It loads the candidate chunk corpus from MongoDB
    (already filtered by workspace + active status), runs BM25 in-memory over
    that candidate set, and dense-searches Qdrant with the same metadata
    filter, then fuses + reranks + expands to parents.
    """

    def __init__(self, ctx: TenantContext) -> None:
        self.ctx = ctx
        self.workspace_id = ctx.workspace_id or ""
        if not self.workspace_id:
            raise ValidationError("retrieval needs a workspace to search")
        self.settings = get_settings()
        self.documents = DocumentRepository()
        self.chunks = ChunkRepository()

    def retrieve(
        self,
        question: str,
        *,
        plan: QueryPlan | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        status: DocumentStatus = DocumentStatus.active,
    ) -> RetrievalResult:
        plan = plan or understand_query(question)
        top_k = top_k or self.settings.retrieval_top_k
        rerank_top_k = rerank_top_k or self.settings.reranker_top_k

        # Discover which years actually exist in this workspace so the year
        # filter doesn't drop documents when the user references a future
        # year mentioned inside them (e.g. "what's the 2026 outlook?").
        available_years = self._workspace_years(status)

        # Build the metadata filter and merge with workspace + status.
        filter_ = {"workspace_id": self.workspace_id, "status": status.value}
        filter_.update(plan.to_metadata_filter(available_years=available_years))

        # If the company filter wiped everything (no docs for that company),
        # relax to company-free so we still return something.
        company_filter_relaxed = False
        if filter_.get("company") and self._count_with(filter_) == 0:
            filter_.pop("company", None)
            company_filter_relaxed = True

        # ---- Candidate corpus from MongoDB (BM25 + parent lookup) ----------
        docs_by_id = self._matching_documents(filter_, status=status)
        db_chunks = run_sync(self.chunks.find_searchable(self.ctx, list(docs_by_id)))
        if not db_chunks:
            return RetrievalResult(
                query=question, plan=plan, chunks=[],
                note="no documents match the filter",
                retrieved_candidate_count=0,
                reranked_candidate_count=0,
                filter_diagnostics={
                    "applied_filter": filter_,
                    "company_filter_relaxed": company_filter_relaxed,
                    "metadata_matched_document_count": len(docs_by_id),
                    "metadata_matched_chunk_count": 0,
                    "metadata_page_inventory": [],
                },
            )

        # Child chunks can split a metric label from its value. Use the parent
        # text as lexical/reranker evidence too; it is the complete passage the
        # model will receive after retrieval, so ranking on it preserves that
        # same local context.
        parent_ids = sorted({ch["parent_id"] for ch in db_chunks
                             if ch.get("parent_id")})
        parents = {p["id"]: p for p in run_sync(
            self.chunks.get_many_by_ids(self.ctx, parent_ids))} if parent_ids else {}
        for ch in db_chunks:
            parent = parents.get(ch.get("parent_id") or "", {})
            ch["_search_text"] = "\n".join(
                part for part in (
                    parent.get("text", ""),
                    (parent.get("tables_md", "") if parent else ""),
                    ch.get("text", ""),
                ) if part
            )

        # ---- Dense search over Qdrant ------------------------------------
        dense_hits = self._dense_search(question, filter_, top_k=top_k * 2)
        dense_by_chunk: dict[str, float] = {}
        for hit in dense_hits:
            cid = hit["payload"].get("chunk_id")
            if cid:
                dense_by_chunk[cid] = hit["score"]

        # ---- BM25 over candidates ----------------------------------------
        bm25_by_chunk = self._bm25_scores(question, db_chunks)

        # ---- Fusion -------------------------------------------------------
        max_dense = max(dense_by_chunk.values(), default=1.0) or 1.0
        max_bm25 = max(bm25_by_chunk.values(), default=1.0) or 1.0
        w_d = self.settings.hybrid_dense_weight
        w_b = self.settings.hybrid_bm25_weight

        fused: list[tuple[float, dict]] = []
        for ch in db_chunks:
            d = dense_by_chunk.get(ch["id"], 0.0) / max_dense
            b = max(0.0, bm25_by_chunk.get(ch["id"], 0.0)) / max_bm25
            # Add a bounded lexical coverage signal. BM25Okapi can assign
            # negative IDF to common numeric/report terms; clamping those
            # scores to zero can otherwise make the exact passage disappear
            # before reranking even when its parent contains every query term.
            coverage = _query_term_coverage(question, ch["_search_text"])
            score = w_d * d + w_b * b + 0.25 * coverage
            fused.append((score, ch))
        fused.sort(key=lambda t: t[0], reverse=True)
        candidates = fused[: max(top_k * 2, rerank_top_k * 3)]

        # ---- Reranker -----------------------------------------------------
        reranker = get_reranker()
        texts = [ch["_search_text"] for _, ch in candidates]
        ranked = reranker.rerank(_lexical_query(question), texts,
                                 top_k=rerank_top_k * 2)
        # Map back to chunks with rerank score
        idx_to_score = {i: float(s) for i, s in ranked}
        # Keep the ranker's candidate set explicit. Its scores are on a
        # provider-specific scale and must not be compared with fused scores
        # for candidates it did not return (for example, a stub reranker may
        # return only top_k entries).
        reranked_positions = set(idx_to_score)

        # ---- Section-aware boost (finance-specific) -----------------------
        # When the question names a financial section concept (risks, outlook,
        # strategy, etc.), boost chunks whose section header matches. This is
        # the missing signal a pure-lexical or pure-dense ranker misses.
        section_boost = _section_boost_for(question)
        if section_boost:
            for rank_pos, (_, ch) in enumerate(candidates):
                if (rank_pos in reranked_positions and ch.get("section")
                        and section_boost in ch["section"].upper()):
                    idx_to_score[rank_pos] += 0.5

        # ---- Parent expansion --------------------------------------------
        # One read for every parent the candidates point at: children carry
        # the matched text, parents carry the section the LLM actually sees.
        out: list[RetrievedChunk] = []
        for rank_pos, (fused_score, ch) in enumerate(candidates):
            rerank_score = idx_to_score.get(rank_pos, 0.0)
            # We pick the top rerank_top_k *by rerank score* if available,
            # else fall back to fused order.
            parent = parents.get(ch.get("parent_id") or "")
            doc = docs_by_id.get(ch["document_id"], {})
            printed_page = ch.get("printed_page")
            if isinstance(printed_page, bool) or not isinstance(printed_page, int):
                printed_page = None
            physical_page = ch.get("page")
            out.append(
                RetrievedChunk(
                    chunk_id=ch["id"],
                    parent_id=ch.get("parent_id"),
                    document_id=ch["document_id"],
                    text=ch["text"],
                    parent_text=(parent["text"] if parent else ch["text"]),
                    tables_md=((parent or ch).get("tables_md") or ""),
                    company=doc.get("company") or "",
                    year=doc.get("year"),
                    quarter=doc.get("quarter") or "",
                    doc_type=doc.get("doc_type") or "",
                    section=ch.get("section") or "",
                    # Keep physical and printed page labels as separate facts.
                    page=(printed_page if printed_page is not None
                          else physical_page or 0),
                    source_filename=doc.get("filename") or "",
                    printed_page=printed_page,
                    physical_page=physical_page,
                    page_number_type=("printed" if printed_page is not None
                                      else "physical"),
                    dense_score=dense_by_chunk.get(ch["id"], 0.0),
                    bm25_score=bm25_by_chunk.get(ch["id"], 0.0),
                    rerank_score=float(rerank_score),
                    final_score=fused_score,
                )
            )

        # Sort returned reranker candidates on their own score scale first;
        # only use fused order when the reranker returned no indices at all.
        # A legitimate rerank score of 0.0 is still a ranked result.
        position_by_chunk = {
            ch["id"]: pos for pos, (_, ch) in enumerate(candidates)
        }
        _sort_reranked_chunks(out, position_by_chunk, reranked_positions)
        # RETRIEVAL_PAGE_QUOTA=0 keeps the original truncation untouched; a
        # non-zero quota re-selects the same number of candidates from the same
        # ranked list under a per-(document, physical page) cap.
        page_quota = self.settings.retrieval_page_quota
        out = (_page_diversity_select(out, rerank_top_k, per_page=page_quota)
               if page_quota else out[:rerank_top_k])
        # Attach sources list
        rerank_positions = {
            candidate_index: position
            for position, (candidate_index, _score) in enumerate(ranked, start=1)
        }
        returned_positions = {
            chunk.chunk_id: position
            for position, chunk in enumerate(out, start=1)
        }
        document_page_counts: dict[tuple[str, str, int | None], int] = {}
        for chunk in db_chunks:
            doc = docs_by_id.get(chunk["document_id"], {})
            page = chunk.get("page")
            key = (chunk["document_id"], doc.get("filename") or "", page)
            document_page_counts[key] = document_page_counts.get(key, 0) + 1
        candidate_diagnostics = []
        for position, (fused_score, candidate) in enumerate(candidates):
            candidate_document = docs_by_id.get(candidate["document_id"], {})
            candidate_diagnostics.append({
                "fused_position": position + 1,
                "rerank_input_position": position + 1,
                "rerank_output_position": rerank_positions.get(position),
                "returned_position": returned_positions.get(candidate["id"]),
                "chunk_id": candidate["id"],
                "document_id": candidate["document_id"],
                "source_filename": candidate_document.get("filename") or "",
                "physical_page": candidate.get("page"),
                "printed_page": candidate.get("printed_page"),
                "section": candidate.get("section") or "",
                "dense_retrieved": candidate["id"] in dense_by_chunk,
                "dense_score": dense_by_chunk.get(candidate["id"]),
                "bm25_score": bm25_by_chunk.get(candidate["id"], 0.0),
                "fused_score": float(fused_score),
                "reranked": position in reranked_positions,
                "rerank_score": idx_to_score.get(position),
            })
        result = RetrievalResult(
            query=question, plan=plan, chunks=out,
            retrieved_candidate_count=len(candidates),
            reranked_candidate_count=len(reranked_positions),
            candidate_diagnostics=candidate_diagnostics,
            filter_diagnostics={
                "applied_filter": filter_,
                "company_filter_relaxed": company_filter_relaxed,
                "metadata_matched_document_count": len(docs_by_id),
                "metadata_matched_chunk_count": len(db_chunks),
                "metadata_page_inventory": [
                    {
                        "document_id": document_id,
                        "source_filename": filename,
                        "physical_page": page,
                        "chunk_count": count,
                    }
                    for (document_id, filename, page), count
                    in document_page_counts.items()
                ],
            },
        )
        for c in result.chunks:
            c.sources = result.citation_list()
        return result

    # ---- helpers ----

    def _matching_documents(
        self, filter_: dict, *, status: DocumentStatus
    ) -> dict[str, dict]:
        """Candidate documents for ``filter_``, keyed by id.

        SQL joined chunks to documents; Mongo has no join, so the document
        prefilter runs first and the ids it returns scope the chunk read. Same
        predicate, one extra round trip, and the resulting map also serves the
        per-chunk company/year/quarter/doc_type fields.
        """
        docs = run_sync(self.documents.find_matching(
            self.ctx,
            workspace_id=self.workspace_id,
            status=status.value,
            company=filter_.get("company"),
            year=filter_.get("year"),
            quarter=filter_.get("quarter"),
            doc_type=filter_.get("doc_type"),
        ))
        return {d["id"]: d for d in docs}

    def _workspace_years(self, status: DocumentStatus) -> list[int]:
        return run_sync(self.documents.distinct_years(
            self.ctx, workspace_id=self.workspace_id, status=status.value))

    def _count_with(self, filter_: dict) -> int:
        """Count child chunks within the same scope as the search filter.

        This is used to decide whether a company constraint should be relaxed.
        A tenant-wide count can be positive because another workspace contains
        that company, incorrectly retaining a filter that yields no documents
        in the active workspace.
        """
        docs = run_sync(self.documents.find_matching(
            self.ctx,
            workspace_id=filter_.get("workspace_id", self.workspace_id),
            status=filter_.get("status", DocumentStatus.active.value),
            company=filter_.get("company"),
            year=filter_.get("year"),
            quarter=filter_.get("quarter"),
            doc_type=filter_.get("doc_type"),
        ))
        if not docs:
            return 0
        chunks = run_sync(self.chunks.find_searchable(
            self.ctx, [d["id"] for d in docs]))
        return len(chunks)

    def _dense_search(self, question: str, filter_: dict, *, top_k: int):
        emb = get_embeddings()
        vdb = get_vectordb()
        try:
            vec = emb.embed([question])[0]
        except Exception as e:
            raise ProviderError(
                "Dense query embedding failed; retrieval cannot be completed safely"
            ) from e
        try:
            return vdb.search(vec, top_k=top_k, filter_=filter_)
        except Exception as e:
            raise ProviderError(
                "Vector search failed; retrieval cannot be completed safely"
            ) from e

    def _bm25_scores(self, question: str,
                     chunks: Iterable[dict[str, Any]]) -> dict[str, float]:
        chunks = list(chunks)
        if not chunks:
            return {}
        corpus = [self._tok(c.get("_search_text", c["text"])) for c in chunks]
        try:
            bm = BM25Okapi(corpus)
            scores = bm.get_scores(self._tok(_lexical_query(question)))
        except Exception as e:  # pragma: no cover
            log.warning("bm25 failed: %s", e)
            return {c["id"]: 0.0 for c in chunks}
        # normalize to >=0
        # BM25Okapi can return negative scores for common terms. Negative
        # evidence must not lower a matching passage below an unrelated
        # zero-score passage during dense/lexical fusion.
        scores = [max(0.0, float(s)) for s in scores]
        max_s = max(scores, default=0.0)
        if max_s <= 0:
            return {c["id"]: 0.0 for c in chunks}
        return {c["id"]: float(s) for c, s in zip(chunks, scores, strict=True)}

    @staticmethod
    def _tok(s: str) -> list[str]:
        import re

        return re.findall(r"[a-z0-9]+", s.lower())

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
                          parent expansion (LLM sees parents)
                                       ↓
                                RetrievalResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus
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
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    sources: list[dict] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    plan: QueryPlan
    chunks: list[RetrievedChunk]
    note: str = ""

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
        items = self.chunks if max_chunks is None else self.chunks[:max_chunks]
        per_chunk_char_budget = max(800, max_chars // max(1, len(items)))
        parts: list[str] = []
        used = 0
        for i, c in enumerate(items, start=1):
            if used >= max_chars:
                break
            tag = f"[{i}]"
            head = (f"{tag} {c.company or 'Unknown'} "
                    f"{c.year or ''} {c.quarter or ''} {c.doc_type}".strip())
            # Truncate parent_text to its fair share of the budget.
            body = (c.parent_text or c.text)[:per_chunk_char_budget]
            table_budget = max(0, per_chunk_char_budget - len(body))
            tables = c.tables_md[:table_budget] if c.tables_md and table_budget > 200 else ""
            piece = f"{head}\n{body}"
            if tables:
                piece += f"\n\nTable:\n{tables}"
            parts.append(piece)
            used += len(piece) + 16  # separator overhead
        return "\n\n---\n\n".join(parts)

    def citation_list(self) -> list[dict]:
        out: list[dict] = []
        for i, c in enumerate(self.chunks, start=1):
            out.append({
                "index": i,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "company": c.company,
                "year": c.year,
                "quarter": c.quarter,
                "doc_type": c.doc_type,
                "section": c.section,
                "page": c.page,
                "snippet": (c.parent_text or c.text)[:240],
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


class HybridRetriever:
    """Stateful per-query retriever.

    Build one per query. It loads the candidate chunk corpus from the
    relational DB (already filtered by workspace + active status), runs
    BM25 in-memory over that candidate set, and dense-searches Qdrant with
    the same metadata filter, then fuses + reranks + expands to parents.
    """

    def __init__(self, db: Session, *, workspace_id: str) -> None:
        self.db = db
        self.workspace_id = workspace_id
        self.settings = get_settings()

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
        if filter_.get("company") and self._count_with(filter_) == 0:
            filter_.pop("company", None)

        # ---- Candidate corpus from DB (for BM25 + parent lookup) ----------
        db_chunks = self._load_candidates_filter(filter_, status=status)
        if not db_chunks:
            return RetrievalResult(query=question, plan=plan, chunks=[],
                                   note="no documents match the filter")

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

        fused: list[tuple[float, Chunk]] = []
        for ch in db_chunks:
            d = dense_by_chunk.get(ch.id, 0.0) / max_dense
            b = bm25_by_chunk.get(ch.id, 0.0) / max_bm25
            score = w_d * d + w_b * b
            fused.append((score, ch))
        fused.sort(key=lambda t: t[0], reverse=True)
        candidates = fused[: max(top_k * 2, rerank_top_k * 3)]

        # ---- Reranker -----------------------------------------------------
        reranker = get_reranker()
        texts = [ch.text for _, ch in candidates]
        ranked = reranker.rerank(question, texts, top_k=rerank_top_k * 2)
        # Map back to chunks with rerank score
        idx_to_score = {i: float(s) for i, s in ranked}

        # ---- Section-aware boost (finance-specific) -----------------------
        # When the question names a financial section concept (risks, outlook,
        # strategy, etc.), boost chunks whose section header matches. This is
        # the missing signal a pure-lexical or pure-dense ranker misses.
        section_boost = _section_boost_for(question)
        if section_boost:
            for rank_pos, (_, ch) in enumerate(candidates):
                if ch.section and section_boost in ch.section.upper():
                    idx_to_score[rank_pos] = idx_to_score.get(rank_pos, 0.0) + 0.5

        # ---- Parent expansion --------------------------------------------
        out: list[RetrievedChunk] = []
        parent_cache: dict[str, Chunk] = {}
        for rank_pos, (fused_score, ch) in enumerate(candidates):
            rerank_score = idx_to_score.get(rank_pos, 0.0)
            # We pick the top rerank_top_k *by rerank score* if available,
            # else fall back to fused order.
            parent = None
            if ch.parent_id:
                parent = parent_cache.get(ch.parent_id)
                if parent is None:
                    parent = self.db.get(Chunk, ch.parent_id)
                    if parent:
                        parent_cache[ch.parent_id] = parent
            out.append(
                RetrievedChunk(
                    chunk_id=ch.id,
                    parent_id=ch.parent_id,
                    document_id=ch.document_id,
                    text=ch.text,
                    parent_text=(parent.text if parent else ch.text),
                    tables_md=(parent.tables_md if parent else ch.tables_md),
                    company=self._doc_company(ch.document_id),
                    year=self._doc_year(ch.document_id),
                    quarter=self._doc_quarter(ch.document_id),
                    doc_type=self._doc_doctype(ch.document_id),
                    section=ch.section or "",
                    page=ch.page,
                    dense_score=dense_by_chunk.get(ch.id, 0.0),
                    bm25_score=bm25_by_chunk.get(ch.id, 0.0),
                    rerank_score=float(rerank_score),
                    final_score=fused_score,
                )
            )

        # Sort by rerank score (fallback to fused) and trim
        out.sort(key=lambda c: c.rerank_score if c.rerank_score else c.final_score,
                 reverse=True)
        out = out[:rerank_top_k]
        # Attach sources list
        result = RetrievalResult(query=question, plan=plan, chunks=out)
        for c in result.chunks:
            c.sources = result.citation_list()
        return result

    # ---- helpers ----

    def _load_candidates_filter(
        self, filter_: dict, *, status: DocumentStatus
    ) -> list[Chunk]:
        """Load child chunks matching ``filter_`` (already includes workspace
        and status). Used by the new pipeline so we filter by the *same*
        metadata dict the vector store uses, keeping DB and vector candidates
        consistent."""
        q = (
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.workspace_id == self.workspace_id,
                Document.status == status,
                Chunk.is_parent.is_(False),
            )
        )
        company = filter_.get("company")
        if company:
            q = q.where(Document.company.in_(company if isinstance(company, list) else [company]))
        year = filter_.get("year")
        if year:
            q = q.where(Document.year.in_(year if isinstance(year, list) else [year]))
        quarter = filter_.get("quarter")
        if quarter:
            q = q.where(Document.quarter.in_(quarter if isinstance(quarter, list) else [quarter]))
        doc_type = filter_.get("doc_type")
        if doc_type:
            from app.db.models import DocumentType

            vals = doc_type if isinstance(doc_type, list) else [doc_type]
            q = q.where(Document.doc_type.in_(
                [DocumentType(d) if isinstance(d, str) else d for d in vals]
            ))
        return list(self.db.scalars(q).all())

    def _workspace_years(self, status: DocumentStatus) -> list[int]:
        from sqlalchemy import distinct

        return list(self.db.scalars(
            select(distinct(Document.year))
            .where(Document.workspace_id == self.workspace_id,
                   Document.status == status,
                   Document.year.is_not(None))
        ).all())

    def _count_with(self, filter_: dict) -> int:
        """Count child chunks matching the company/year portion of filter_."""
        from sqlalchemy import func

        q = (
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.is_parent.is_(False))
        )
        company = filter_.get("company")
        if company:
            q = q.where(Document.company.in_(company if isinstance(company, list) else [company]))
        year = filter_.get("year")
        if year:
            q = q.where(Document.year.in_(year if isinstance(year, list) else [year]))
        return int(self.db.scalar(q) or 0)

    def _dense_search(self, question: str, filter_: dict, *, top_k: int):
        emb = get_embeddings()
        vdb = get_vectordb()
        try:
            vec = emb.embed([question])[0]
        except Exception as e:  # pragma: no cover
            log.warning("dense embedding failed: %s", e)
            return []
        try:
            return vdb.search(vec, top_k=top_k, filter_=filter_)
        except Exception as e:
            log.warning("vector search failed (%s); falling back to BM25-only", e)
            return []

    def _bm25_scores(self, question: str, chunks: Iterable[Chunk]) -> dict[str, float]:
        chunks = list(chunks)
        if not chunks:
            return {}
        corpus = [self._tok(c.text) for c in chunks]
        try:
            bm = BM25Okapi(corpus)
            scores = bm.get_scores(self._tok(question))
        except Exception as e:  # pragma: no cover
            log.warning("bm25 failed: %s", e)
            return {c.id: 0.0 for c in chunks}
        # normalize to >=0
        max_s = max(scores, default=0.0)
        if max_s <= 0:
            return {c.id: 0.0 for c in chunks}
        return {c.id: float(s) for c, s in zip(chunks, scores)}

    @staticmethod
    def _tok(s: str) -> list[str]:
        import re

        return re.findall(r"[a-z0-9]+", s.lower())

    # Cached document-field accessors
    _doc_cache: dict[str, Document] | None = None

    def _doc(self, doc_id: str) -> Document | None:
        if self._doc_cache is None:
            self._doc_cache = {}
        d = self._doc_cache.get(doc_id)
        if d is None:
            d = self.db.get(Document, doc_id)
            if d:
                self._doc_cache[doc_id] = d
        return d

    def _doc_company(self, doc_id: str) -> str:
        d = self._doc(doc_id)
        return d.company if d else ""

    def _doc_year(self, doc_id: str) -> int | None:
        d = self._doc(doc_id)
        return d.year if d else None

    def _doc_quarter(self, doc_id: str) -> str:
        d = self._doc(doc_id)
        return d.quarter if d else ""

    def _doc_doctype(self, doc_id: str) -> str:
        d = self._doc(doc_id)
        return d.doc_type.value if d else ""

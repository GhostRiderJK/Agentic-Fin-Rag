"""Read-only PIF retrieval stages + offline strategy experiments.

This reproduces :class:`app.retrieval.hybrid.HybridRetriever` stage by stage so
each stage can be measured and varied independently. It deliberately reuses the
production data path (the retriever's own corpus read, filter build, BM25 and
dense helpers) so an experiment measures the real scoring behaviour instead of
a paraphrase of it.

Nothing here writes: the datastore is only read, no generation call is made and
no stored record is touched. Production ranking is unchanged; a strategy that
proves out here becomes a separate proposal.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.db.mongodb.runtime import run_sync
from app.providers.reranker import get_reranker
from app.retrieval.hybrid import HybridRetriever, RetrievalResult, _query_term_coverage


@dataclass
class StageTrace:
    """One question measured through every retrieval stage."""

    question: str
    plan: dict[str, Any] = field(default_factory=dict)
    applied_filter: dict[str, Any] = field(default_factory=dict)
    company_filter_relaxed: bool = False
    metadata_matched_document_count: int = 0
    metadata_matched_chunk_count: int = 0
    inventory_pages: int = 0
    expected_page_in_inventory: bool | None = None
    expected_page_chunk_count: int = 0
    expected_values_on_page: list[dict[str, Any]] = field(default_factory=list)
    stage_positions: dict[str, list[int]] = field(default_factory=dict)
    stage_sizes: dict[str, int] = field(default_factory=dict)
    stage_unique_pages: dict[str, int] = field(default_factory=dict)
    stage_documents: dict[str, int] = field(default_factory=dict)
    expected_stage_reached: str = ""
    expected_first_position: dict[str, int | None] = field(default_factory=dict)
    detail: list[dict[str, Any]] = field(default_factory=list)
    pool_rows: list[dict[str, Any]] = field(default_factory=list)
    returned: list[dict[str, Any]] = field(default_factory=list)
    context_positions: list[int] = field(default_factory=list)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    k_available: dict[str, bool] = field(default_factory=dict)
    pool_boundary_fused_score: float | None = None
    reranker_name: str = ""


def _page_of(chunk: dict) -> int | None:
    page = chunk.get("page")
    return page if isinstance(page, int) and not isinstance(page, bool) else None


# Every stage is its own row list, in pipeline order, and no two stages may
# share a list: an earlier revision aliased the fused pool under "inventory",
# which made "present in the corpus" and "survived into the pool" indistinguishable.
STAGE_ORDER = ("inventory", "dense", "lexical", "corpus", "fused_pool",
               "reranked_order", "returned", "final_context")
STAGE_DEFINITIONS = {
    "inventory": "child chunks the metadata filter matched, in the order read "
                 "from Mongo, before any scoring",
    "dense": "vector-store hits for the query, in the store's own score order",
    "lexical": "corpus chunks with a positive BM25 score, ranked by that score",
    "corpus": "every inventory chunk given a fused score, ranked by it",
    "fused_pool": "the fused ranking truncated to the candidate pool",
    "reranked_order": "the pool after the reranker and section adjustments",
    "returned": "the set handed back to the caller",
    "final_context": "the returned set truncated to the serialized context cap",
}
# Stages that only prove the page exists somewhere, so they cannot answer
# "how far did it get" and are excluded from expected_stage_reached.
EXISTENCE_STAGES = frozenset({"inventory", "corpus"})


def _normalize_glyphs(text: str) -> str:
    return (text or "").replace("\u00aa", "SAR").replace("\ufefc", "SAR").replace(
        "\ufdfc", "SAR")


def _name_matches(candidate: str | None, expected: list[str]) -> bool:
    """Compare source names after the duplicate-stem normalization.

    The stored filename carries an ingestion duplication (``2024-en_2024-en.pdf``)
    that the gold set does not. This only relaxes comparison; stored metadata is
    never rewritten.
    """
    if not expected:
        return True
    from app.evaluation.pif_chat_check import normalize_duplicate_source_filename

    normalized = {normalize_duplicate_source_filename(name) for name in expected}
    return candidate in expected or normalize_duplicate_source_filename(
        candidate or "") in normalized


def _number_tokens(text: str) -> set[str]:
    return {v.replace(",", "") for v in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")}


class StageProbe:
    """Runs the production retrieval path with per-stage observability."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.r = retriever
        self.settings = retriever.settings

    # ---- stage replication -------------------------------------------------

    @staticmethod
    def _reranker(reranker: Any = None) -> Any:
        return reranker if reranker is not None else get_reranker()

    def prepare(self, question: str, *, top_k: int, rerank_top_k: int,
                pool_size: int | None = None, dense_hits: int | None = None) -> dict:
        """Everything up to and including fusion, before any ranking."""
        from app.db.enums import DocumentStatus
        from app.retrieval.query_understanding import understand_query

        plan = understand_query(question)
        available_years = self.r._workspace_years(DocumentStatus.active)
        filter_ = {"workspace_id": self.r.workspace_id, "status": "active"}
        filter_.update(plan.to_metadata_filter(available_years=available_years))
        relaxed = False
        if filter_.get("company") and self.r._count_with(filter_) == 0:
            filter_.pop("company", None)
            relaxed = True

        docs = self.r._matching_documents(filter_, status=DocumentStatus.active)
        chunks = run_sync(self.r.chunks.find_searchable(self.r.ctx, list(docs)))
        if not chunks:
            return {"plan": plan, "filter": filter_, "relaxed": relaxed,
                    "docs": docs, "chunks": [], "candidates": []}

        parent_ids = sorted({c["parent_id"] for c in chunks if c.get("parent_id")})
        parents = {p["id"]: p for p in run_sync(
            self.r.chunks.get_many_by_ids(self.r.ctx, parent_ids))} if parent_ids else {}
        for ch in chunks:
            parent = parents.get(ch.get("parent_id") or "", {})
            ch["_search_text"] = "\n".join(
                part for part in (parent.get("text", ""),
                                  (parent.get("tables_md", "") if parent else ""),
                                  ch.get("text", "")) if part)

        dense_hits_count = dense_hits or top_k * 2
        dense_result = self.r._dense_search(question, filter_,
                                            top_k=dense_hits_count)
        dense = {h["payload"]["chunk_id"]: h["score"] for h in dense_result
                 if h["payload"].get("chunk_id")}
        bm25 = self.r._bm25_scores(question, chunks)

        max_dense = max(dense.values(), default=1.0) or 1.0
        max_bm25 = max(bm25.values(), default=1.0) or 1.0
        w_d = self.settings.hybrid_dense_weight
        w_b = self.settings.hybrid_bm25_weight

        fused = []
        for ch in chunks:
            coverage = _query_term_coverage(question, ch["_search_text"])
            score = (w_d * dense.get(ch["id"], 0.0) / max_dense
                     + w_b * max(0.0, bm25.get(ch["id"], 0.0)) / max_bm25
                     + 0.25 * coverage)
            fused.append({"chunk": ch, "fused_score": float(score),
                          "dense_score": dense.get(ch["id"]),
                          "bm25_score": bm25.get(ch["id"], 0.0),
                          "coverage": float(coverage),
                          "document_id": ch["document_id"],
                          "physical_page": _page_of(ch),
                          "filename": (docs.get(ch["document_id"]) or {}).get(
                              "filename") or ""})
        fused.sort(key=lambda row: row["fused_score"], reverse=True)
        default_pool = max(top_k * 2, rerank_top_k * 3, dense_hits or 0)
        pool = fused[: pool_size or default_pool]
        return {"plan": plan, "filter": filter_, "relaxed": relaxed, "docs": docs,
                "chunks": chunks, "parents": parents, "dense_ranked": list(dense),
                "dense_scores": dense, "bm25": bm25, "fused_all": fused,
                "default_pool_size": default_pool, "candidates": pool}

    def score_with_reranker(
        self, question: str, pool: list[dict], *, rerank_top_k: int,
        reranker: Any = None,
    ) -> list[dict]:
        """Apply the production rerank + section-boost + ordering rules."""
        from app.retrieval.hybrid import _lexical_query, _section_boost_for

        reranker = self._reranker(reranker)
        texts = [row["chunk"]["_search_text"] for row in pool]
        ranked = reranker.rerank(_lexical_query(question), texts,
                                 top_k=rerank_top_k * 2)
        idx_to_score = {i: float(s) for i, s in ranked}
        reranked_positions = set(idx_to_score)

        section_boost = _section_boost_for(question)
        if section_boost:
            for pos, row in enumerate(pool):
                if (pos in reranked_positions and row["chunk"].get("section")
                        and section_boost in row["chunk"]["section"].upper()):
                    idx_to_score[pos] += 0.5

        scored = []
        for pos, row in enumerate(pool):
            scored.append({**row,
                           "rerank_output_position": (
                               [i for i, (idx, _s) in enumerate(ranked, start=1)
                                if idx == pos] or [None])[0],
                           "rerank_score": idx_to_score.get(pos),
                           "reranked": pos in reranked_positions,
                           "pool_position": pos + 1})
        scored.sort(
            key=lambda row: (
                (row["rerank_score"] if row["reranked"] else float("-inf"),
                 row["fused_score"])),
            reverse=True,
        )
        return scored

    # ---- full trace --------------------------------------------------------

    def trace(
        self, question: str, *, expected_page: int | None,
        expected_numbers: list[str], expected_source_filenames: list[str],
        top_k: int = 10, rerank_top_k: int = 10, context_cap: int = 8,
        pool_size: int | None = None, dense_hits: int | None = None,
        select: Callable[[list[dict], int], list[dict]] | None = None,
        pool_select: Callable[[list[dict]], list[dict]] | None = None,
        reranker: Any = None,
    ) -> StageTrace:
        prepared = self.prepare(question, top_k=top_k, rerank_top_k=rerank_top_k,
                                pool_size=pool_size, dense_hits=dense_hits)
        if pool_select is not None:
            prepared["candidates"] = pool_select(prepared["candidates"])
        t = StageTrace(question=question, applied_filter=prepared["filter"],
                       company_filter_relaxed=prepared["relaxed"],
                       metadata_matched_document_count=len(prepared["docs"]),
                       metadata_matched_chunk_count=len(prepared["chunks"]))
        chunks = prepared["chunks"]
        pages = {_page_of(c) for c in chunks if _page_of(c) is not None}
        t.inventory_pages = len(pages)
        if expected_page is None:
            t.expected_page_in_inventory = None
        else:
            on_page = [c for c in chunks if _page_of(c) == expected_page]
            t.expected_page_in_inventory = bool(on_page)
            t.expected_page_chunk_count = len(on_page)
            page_text = _normalize_glyphs("\n".join(
                "\n".join(filter(None, (c.get("text", ""), c.get("tables_md", ""))))
                for c in chunks if _page_of(c) == expected_page))
            present = _number_tokens(page_text)
            t.expected_values_on_page = [
                {"expected": e, "numbers": sorted(_number_tokens(e)),
                 "all_present": bool(_number_tokens(e))
                 and _number_tokens(e) <= present}
                for e in expected_numbers
            ]

        if not prepared["candidates"]:
            return t

        dense_ranked = prepared["dense_scores"]
        bm25 = prepared["bm25"]
        fused_all = prepared["fused_all"]
        pool = prepared["candidates"]

        by_id = {row["chunk"]["id"]: row for row in fused_all}

        # dense stage = the ranked vector-store hits themselves
        dense_rows = [by_id[cid] for cid in dense_ranked if cid in by_id]
        lexical_rows = sorted(
            (by_id[cid] for cid, s in bm25.items() if s > 0 and cid in by_id),
            key=lambda row: bm25[row["chunk"]["id"]], reverse=True)

        # The inventory is the metadata-filtered corpus itself, before scoring.
        # Its positions are read order, which is the only order it has.
        inventory_docs = prepared["docs"]
        inventory_rows = [{
            "chunk": ch,
            "document_id": ch["document_id"],
            "physical_page": _page_of(ch),
            "filename": (inventory_docs.get(ch["document_id"]) or {}).get(
                "filename") or "",
        } for ch in prepared["chunks"]]

        stage_rows = {
            "inventory": inventory_rows,
            "corpus": fused_all,
            "dense": dense_rows,
            "lexical": lexical_rows,
            "fused_pool": pool,
        }
        scored = self.score_with_reranker(question, pool,
                                          rerank_top_k=rerank_top_k,
                                          reranker=reranker)
        t.reranker_name = getattr(self._reranker(reranker), "name", "?")
        chosen = scored
        if select is not None:
            chosen = select(scored, rerank_top_k)
        else:
            chosen = scored[:rerank_top_k]
        stage_rows["reranked_order"] = scored
        stage_rows["returned"] = chosen
        context_rows = chosen[:context_cap]
        stage_rows["final_context"] = context_rows

        for name in STAGE_ORDER:
            rows = stage_rows.get(name, [])
            t.stage_sizes[name] = len(rows)
            hit = [i + 1 for i, row in enumerate(rows)
                   if expected_page is not None
                   and row["physical_page"] == expected_page
                   and _name_matches(row["filename"], expected_source_filenames)]
            t.expected_first_position[name] = hit[0] if hit else None
            t.stage_unique_pages[name] = len({r["physical_page"] for r in rows})
            t.stage_documents[name] = len({r["document_id"] for r in rows})
            if hit and name not in EXISTENCE_STAGES and not t.expected_stage_reached:
                t.expected_stage_reached = name

        order = list(STAGE_ORDER)
        for name in order:
            rows = stage_rows.get(name, [])
            t.stage_positions[name] = [
                i + 1 for i, row in enumerate(rows)
                if expected_page is not None
                and row["physical_page"] == expected_page
                and _name_matches(row["filename"], expected_source_filenames)]
        t.detail = [
            {"pool_position": i + 1, "physical_page": row["physical_page"],
             "document_id": row["document_id"], "source_filename": row["filename"],
             "section": row["chunk"].get("section") or "",
             "fused_score": round(row["fused_score"], 6),
             "dense_score": row["dense_score"],
             "bm25_score": round(row["bm25_score"], 6),
             "coverage": round(row["coverage"], 6),
             "rerank_score": row.get("rerank_score"),
             "rerank_output_position": row.get("rerank_output_position"),
             "expected_page": expected_page is not None
             and row["physical_page"] == expected_page}
            for i, row in enumerate(scored)
        ]
        t.pool_rows = [
            {"fused_pool_position": i + 1, "physical_page": row["physical_page"],
             "document_id": row["document_id"], "source_filename": row["filename"],
             "section": row["chunk"].get("section") or "",
             "fused_score": round(row["fused_score"], 6),
             "dense_score": row["dense_score"],
             "bm25_score": round(row["bm25_score"], 6),
             "coverage": round(row["coverage"], 6),
             "expected_page": expected_page is not None
             and row["physical_page"] == expected_page}
            for i, row in enumerate(pool)
        ]
        t.pool_boundary_fused_score = pool[-1]["fused_score"] if pool else None
        t.returned = [
            {"position": i + 1, "physical_page": row["physical_page"],
             "document_id": row["document_id"], "source_filename": row["filename"],
             "section": row["chunk"].get("section") or "",
             "fused_score": round(row["fused_score"], 6),
             "rerank_score": row.get("rerank_score"),
             "expected_page": expected_page is not None
             and row["physical_page"] == expected_page}
            for i, row in enumerate(chosen)]
        t.context_positions = [
            i + 1 for i, row in enumerate(context_rows)
            if expected_page is not None and row["physical_page"] == expected_page
            and _name_matches(row["filename"], expected_source_filenames)]
        t.candidate_counts = {
            "corpus_chunks": len(chunks),
            "dense_hits": len(dense_rows),
            "lexical_matches": len(lexical_rows),
            "fused_pool": len(pool),
            "fused_pool_default": prepared["default_pool_size"],
            "reranked": len([r for r in scored if r["reranked"]]),
            "returned": len(chosen),
            "in_final_context": len(context_rows),
        }
        t.k_available = {f"k={k}": len(chosen) >= k for k in (5, 10)}
        return t

    # ---- parity with production -------------------------------------------

    def parity(self, question: str, *, top_k: int, rerank_top_k: int) -> bool:
        """Confirm this replication returns the same chunks as retrieve()."""
        probe = self.trace(question, expected_page=None, expected_numbers=[],
                           expected_source_filenames=[], top_k=top_k,
                           rerank_top_k=rerank_top_k)
        real: RetrievalResult = self.r.retrieve(
            question, top_k=top_k, rerank_top_k=rerank_top_k)
        return [c["document_id"] for c in probe.returned] == [
            c.document_id for c in real.chunks] and len(probe.returned) == len(
            real.chunks)


class PassthroughReranker:
    """Returns candidates in input order. Isolates the reranker's contribution."""

    name = "passthrough-experiment"

    def rerank(self, query: str, documents: list[str], top_k=None):
        ordered = list(enumerate([0.0] * len(documents)))
        return ordered[: top_k if top_k is not None else len(documents)]


# ---------------------------------------------------------------------------
# candidate-diversity strategies (experiment only, not production)
# ---------------------------------------------------------------------------


def diversity_select(
    scored: list[dict], limit: int, *, per_page: int = 1, per_document: int | None = None,
) -> list[dict]:
    """Greedy re-selection of the top ``limit`` candidates under page/doc quotas.

    Grouping is by document id + physical page, never by the printed label, so
    two reports with the same printed page number stay separate. Candidates with
    no usable physical page are never merged into one slot; once the quota can no
    longer fill ``limit``, rejected and unpaginated candidates backfill strictly
    in fused rank order, so a page with no metadata degrades to the production
    ordering instead of disappearing.
    """
    chosen: list[dict] = []
    overflow: list[dict] = []
    page_counts: dict[tuple[str, int], int] = {}
    doc_counts: dict[str, int] = {}
    for row in scored:
        page = row.get("physical_page")
        doc = row.get("document_id") or ""
        if not isinstance(page, int) or isinstance(page, bool):
            overflow.append(row)
            continue
        if per_document is not None and doc_counts.get(doc, 0) >= per_document:
            overflow.append(row)
            continue
        if page_counts.get((doc, page), 0) >= per_page:
            overflow.append(row)
            continue
        page_counts[(doc, page)] = page_counts.get((doc, page), 0) + 1
        doc_counts[doc] = doc_counts.get(doc, 0) + 1
        chosen.append(row)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        chosen.extend(overflow[: limit - len(chosen)])
    return chosen[:limit]

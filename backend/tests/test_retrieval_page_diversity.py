"""Deterministic coverage for configurable page diversity.

The production path gates the strategy behind ``RETRIEVAL_PAGE_QUOTA`` (0 =
disabled, the default). These tests pin both halves: the re-selection rules
(grouping key, missing-page fallback, tenant scoping, rank-order backfill, the
structural limit of pool growth) and the proof that a disabled flag leaves the
returned ordering exactly as it was. Nothing here touches a datastore.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from app.evaluation.pif_chat_check import normalize_duplicate_source_filename
from app.evaluation.pif_stages import (
    EXISTENCE_STAGES,
    STAGE_DEFINITIONS,
    STAGE_ORDER,
    StageProbe,
    _name_matches,
    diversity_select,
)
from app.retrieval import hybrid as hybrid_module


def row(document_id: str, page: int | None, *, text: str = "body",
        printed_page: int | None = None, chunk_id: str | None = None) -> dict:
    return {
        "chunk": {"id": chunk_id or f"{document_id}-{page}-{text[:6]}",
                  "text": text, "section": "", "printed_page": printed_page},
        "document_id": document_id,
        "physical_page": page,
        "filename": f"{document_id}.pdf",
        "fused_score": 0.0,
        "dense_score": None,
        "bm25_score": 0.0,
        "coverage": 0.0,
    }


def ids(rows: list[dict]) -> list[str]:
    return [r["chunk"]["id"] for r in rows]


def score_pool(rows: list[dict]) -> list[dict]:
    """Give each row a descending score so input order is an explicit ranking."""
    for position, item in enumerate(rows):
        item["fused_score"] = float(len(rows) - position)
    return rows


# ---------------------------------------------------------------------------
# grouping key
# ---------------------------------------------------------------------------


def test_grouping_is_document_and_physical_page_so_two_reports_stay_separate():
    pool = score_pool([
        row("annual-2024", 6, text="a"), row("quarterly-2024", 6, text="b"),
        row("annual-2024", 7, text="c"), row("quarterly-2024", 7, text="d"),
    ])
    chosen = diversity_select(pool, 4, per_page=1)
    assert {(r["document_id"], r["physical_page"]) for r in chosen} == {
        ("annual-2024", 6), ("quarterly-2024", 6),
        ("annual-2024", 7), ("quarterly-2024", 7)}


def test_printed_page_labels_are_not_used_for_grouping():
    """Two chunks sharing a physical page collapse even when their printed
    labels differ; the printed label is a citation fact, not a diversity key."""
    pool = score_pool([
        row("annual-2024", 9, text="first chunk", printed_page=3),
        row("annual-2024", 9, text="second chunk", printed_page=112),
        row("annual-2024", 10, text="third chunk", printed_page=4),
    ])
    chosen = diversity_select(pool, 3, per_page=1)
    # The quota fills what it can (one per physical page), then the best
    # remaining candidate backfills the limit.
    assert [r["chunk"]["text"] for r in chosen] == [
        "first chunk", "third chunk", "second chunk"]


# ---------------------------------------------------------------------------
# crowding, multi-evidence pages, fallback
# ---------------------------------------------------------------------------


def test_one_chunk_per_page_recovers_a_page_drowned_in_sibling_chunks():
    siblings = [row("annual-2024", page, text=f"sibling {page} {i}")
                for page in (2, 3, 4, 5, 7, 8, 9, 10, 11) for i in range(3)]
    # page 6's useful chunk sits behind four fully crowded pages
    pool = score_pool(siblings[:12] + [row("annual-2024", 6, text="needed fact")]
                      + siblings[12:])

    assert not any(r["physical_page"] == 6 for r in pool[:10])
    chosen = diversity_select(pool, 10, per_page=1)
    assert len(chosen) == 10
    assert [r["physical_page"] for r in chosen[:6]] == [2, 3, 4, 5, 6, 7]
    assert [r["chunk"]["text"] for r in chosen if r["physical_page"] == 6] == [
        "needed fact"]


def test_per_page_quota_keeps_both_chunks_when_one_page_carries_two_facts():
    pool = score_pool([
        row("annual-2024", 6, text="assets figure"),
        row("annual-2024", 6, text="liabilities figure"),
        row("annual-2024", 8, text="other"),
    ])
    assert [r["chunk"]["text"] for r in diversity_select(pool, 2, per_page=1)] == [
        "assets figure", "other"]
    assert [r["chunk"]["text"] for r in diversity_select(pool, 2, per_page=2)] == [
        "assets figure", "liabilities figure"]


def test_candidates_without_a_physical_page_keep_their_own_identity():
    unpaginated = [row("annual-2024", None, text=f"unique body {i} here")
                   for i in range(3)]
    # Without page metadata there is nothing to group by, so the three rows are
    # three separate candidates, not one collapsed slot.
    assert ids(diversity_select(score_pool(list(unpaginated)), 3,
                                per_page=1)) == ids(unpaginated)
    pool = score_pool(unpaginated)
    assert len(diversity_select(pool, 3, per_page=1)) == 3


def test_backfill_after_the_quota_follows_fused_rank_not_metadata_status():
    unpaginated = [row("annual-2024", None, text=f"unique body {i} here")
                   for i in range(3)]
    pool = score_pool([row("annual-2024", 6, text="a"),
                       row("annual-2024", 6, text="b")] + unpaginated)
    chosen = diversity_select(pool, 4, per_page=1)
    assert ids(chosen) == ids([pool[0], pool[1], unpaginated[0], unpaginated[1]])


def test_non_integer_page_metadata_is_treated_as_missing_not_as_a_page():
    pool = score_pool([row("annual-2024", True, text="bad page"),
                       row("annual-2024", "6", text="string page"),
                       row("annual-2024", 6, text="real")])
    chosen = diversity_select(pool, 3, per_page=1)
    assert [r["chunk"]["text"] for r in chosen] == ["real", "bad page", "string page"]


def test_wholly_missing_page_metadata_leaves_the_production_order_untouched():
    pool = score_pool([row("annual-2024", None, text=f"c{i}") for i in range(6)])
    assert diversity_select(pool, 4, per_page=1) == pool[:4]


def test_selection_is_deterministic_and_never_invents_candidates():
    pool = score_pool([row("annual-2024", 6, text=f"x{i}") for i in range(4)]
                      + [row("annual-2024", 7, text="y"), row("q", 3, text="z")])
    first = diversity_select(pool, 5, per_page=1)
    for _ in range(3):
        assert ids(diversity_select(pool, 5, per_page=1)) == ids(first)
    assert set(ids(first)) <= set(ids(pool))
    assert len(diversity_select(pool, 50, per_page=1)) == len(pool)


def test_per_document_quota_caps_one_report_without_losing_the_rest():
    pool = score_pool([row("annual-2024", page, text=f"p{page}") for page in range(1, 9)]
                      + [row("annual-2023", 9, text="old report")])
    chosen = diversity_select(pool, 4, per_page=1, per_document=3)
    assert sum(r["document_id"] == "annual-2024" for r in chosen) == 3
    assert any(r["document_id"] == "annual-2023" for r in chosen)


# ---------------------------------------------------------------------------
# source-name comparison (normalization at compare time only)
# ---------------------------------------------------------------------------


def test_duplicate_stem_matches_gold_without_rewriting_stored_metadata():
    assert _name_matches("2024-en_2024-en.pdf", ["2024-en.pdf"])
    assert not _name_matches("2023-en_2023-en.pdf", ["2024-en.pdf"])
    assert normalize_duplicate_source_filename("2024-en_2024-en.pdf") == "2024-en.pdf"
    assert normalize_duplicate_source_filename("a_b.pdf") == "a_b.pdf"


# ---------------------------------------------------------------------------
# the probe over a synthetic corpus
# ---------------------------------------------------------------------------


class FakeChunkRepo:
    def __init__(self, store: dict[str, list[dict]]) -> None:
        self.store = store
        self.calls: list[tuple[str, object, str]] = []

    async def find_searchable(self, ctx, document_ids):
        self.calls.append(("find_searchable", ctx, ctx.workspace_id))
        return [c for c in self.store[ctx.workspace_id]
                if c["document_id"] in set(document_ids)]

    async def get_many_by_ids(self, ctx, chunk_ids):
        self.calls.append(("get_many_by_ids", ctx, ctx.workspace_id))
        wanted = set(chunk_ids)
        return [c for c in self.store[ctx.workspace_id] if c["id"] in wanted]


class FakeRetriever:
    """Duck-typed stand-in exposing only what StageProbe.prepare uses."""

    def __init__(self, chunks: list[dict], *, scores: dict[str, float],
                 dense: dict[str, float], workspace_id: str = "ws-a") -> None:
        from app.core.config import get_settings

        settings = get_settings()
        self.settings = SimpleNamespace(
            hybrid_dense_weight=settings.hybrid_dense_weight,
            hybrid_bm25_weight=settings.hybrid_bm25_weight)
        self.workspace_id = workspace_id
        self.ctx = SimpleNamespace(tenant_id="t-1", user_id="u-1",
                                   workspace_id=workspace_id)
        self.chunks = FakeChunkRepo({workspace_id: chunks,
                                     "ws-other": [dict(c, workspace_id="ws-other")
                                                  for c in chunks]})
        self._scores = scores
        self._dense = dense
        self.filters: list[dict] = []

    def _workspace_years(self, status):
        return [2024]

    def _count_with(self, filter_):
        return 1

    def _matching_documents(self, filter_, *, status):
        assert filter_["workspace_id"] == self.workspace_id
        assert filter_["status"] == status.value
        self.filters.append(dict(filter_))
        return {doc_id: {"filename": f"{doc_id}.pdf"}
                for doc_id in {c["document_id"] for c in
                               self.chunks.store[self.workspace_id]}}

    def _dense_search(self, question, filter_, *, top_k):
        return [{"payload": {"chunk_id": cid}, "score": score}
                for cid, score in sorted(self._dense.items(),
                                         key=lambda t: t[1], reverse=True)[:top_k]]

    def _bm25_scores(self, question, chunks):
        return {c["id"]: self._scores.get(c["id"], 0.0) for c in chunks}


def _corpus(pages: dict[tuple[str, int], int]) -> list[dict]:
    out = []
    for (document_id, page), count in pages.items():
        for i in range(count):
            out.append({"id": f"{document_id}-p{page}-{i}",
                        "document_id": document_id, "page": page,
                        "printed_page": None, "parent_id": None,
                        "section": "", "text": f"chunk {i} on page {page}",
                        "tables_md": ""})
    return out


QUESTION = "What was the total value of assets under management?"


def _probe(pages, *, scores=None, dense=None) -> StageProbe:
    return StageProbe(FakeRetriever(_corpus(pages), scores=scores or {},
                                   dense=dense or {}))


def test_probe_reads_only_the_tenant_scoped_workspace_corpus():
    probe = _probe({("annual-2024", 6): 2})
    trace = probe.trace(QUESTION, expected_page=6, expected_numbers=[],
                        expected_source_filenames=["annual-2024.pdf"],
                        top_k=5, rerank_top_k=5, context_cap=5)
    assert trace.applied_filter["workspace_id"] == "ws-a"
    assert trace.applied_filter["status"] == "active"
    assert all(ctx.workspace_id == "ws-a" and ctx.tenant_id == "t-1"
               for _name, ctx, _ws in probe.r.chunks.calls)
    assert {r["document_id"] for r in trace.returned} == {"annual-2024"}


def test_crowded_page_is_reachable_and_promoted_by_diversity():
    """The failure mode diversity fixes: the page is in the pool but its
    siblings push it past the returned limit."""
    probe = _probe({("annual-2024", 7): 12, ("annual-2024", 6): 2})
    trace = probe.trace(
        QUESTION, expected_page=6, expected_numbers=[],
        expected_source_filenames=["annual-2024.pdf"], top_k=5, rerank_top_k=5,
        context_cap=4,
        select=lambda scored, limit: diversity_select(scored, limit, per_page=1))
    assert trace.stage_positions["fused_pool"]
    assert trace.stage_positions["returned"]
    assert len(trace.returned) == 5
    assert {r["physical_page"] for r in trace.returned} == {7, 6}


def test_pool_growth_cannot_rescue_a_page_below_the_fused_boundary():
    """The other failure mode: the page scores out of the pool entirely, so no
    re-selection of the pool can return it. Only a wider pool reaches it, and
    only when the pool actually spans its fused rank."""
    scores = {f"annual-2024-p7-{i}": 100.0 - i for i in range(40)}
    scores.update({f"annual-2024-p6-{i}": 0.1 for i in range(2)})
    probe = _probe({("annual-2024", 7): 40, ("annual-2024", 6): 2}, scores=scores)

    narrow = probe.trace(
        QUESTION, expected_page=6, expected_numbers=[],
        expected_source_filenames=["annual-2024.pdf"], top_k=5, rerank_top_k=5,
        context_cap=4,
        select=lambda scored, limit: diversity_select(scored, limit, per_page=1))
    assert narrow.stage_positions["fused_pool"] == []
    assert narrow.stage_positions["returned"] == []
    assert not narrow.k_available["k=10"]

    wide = probe.trace(
        QUESTION, expected_page=6, expected_numbers=[],
        expected_source_filenames=["annual-2024.pdf"], top_k=5, rerank_top_k=5,
        context_cap=4, pool_size=45,
        select=lambda scored, limit: diversity_select(scored, limit, per_page=1))
    assert wide.stage_positions["fused_pool"]
    assert wide.stage_positions["returned"]


def test_page_quota_is_a_setting_not_a_caller_supplied_ranking_hook():
    """Callers cannot steer production ranking per request; only the configured
    flag can, and its default is off."""
    signature = inspect.signature(hybrid_module.HybridRetriever.retrieve).parameters
    assert {"select", "pool_select", "per_page", "page_quota"}.isdisjoint(signature)
    from app.core.config import Settings

    assert Settings().retrieval_page_quota == 0


@pytest.mark.parametrize("page_count", [1, 2, 3])
def test_returned_set_always_honours_the_requested_limit(page_count: int):
    probe = _probe({("annual-2024", p): 6 for p in range(1, 11)})
    trace = probe.trace(
        QUESTION, expected_page=None, expected_numbers=[],
        expected_source_filenames=[], top_k=5, rerank_top_k=5, context_cap=4,
        pool_size=60,
        select=lambda scored, limit: diversity_select(scored, limit,
                                                      per_page=page_count))
    assert len(trace.returned) == 5
    # limit 5 over a 6-sibling-per-page corpus spreads across ceil(5 / per_page).
    assert len({r["physical_page"] for r in trace.returned}) == (
        -(-5 // page_count))


def test_pool_width_alone_caps_how_many_pages_a_result_set_can_span():
    """A wider pool is not free diversity: with the default 3x-limit pool a
    ten-page spread cannot even be expressed, which bounds any quota rule."""
    probe = _probe({("annual-2024", p): 6 for p in range(1, 11)})
    narrow = probe.trace(
        QUESTION, expected_page=None, expected_numbers=[],
        expected_source_filenames=[], top_k=5, rerank_top_k=5, context_cap=4,
        select=lambda scored, limit: diversity_select(scored, limit, per_page=1))
    wide = probe.trace(
        QUESTION, expected_page=None, expected_numbers=[],
        expected_source_filenames=[], top_k=5, rerank_top_k=5, context_cap=4,
        pool_size=60,
        select=lambda scored, limit: diversity_select(scored, limit, per_page=1))
    assert len({r["physical_page"] for r in narrow.returned}) < len(
        {r["physical_page"] for r in wide.returned})


# ---------------------------------------------------------------------------
# stage identity: inventory, fused corpus and fused pool are three different
# things. An earlier revision aliased the fused pool under "inventory", which
# made a page that never scored into the pool look absent from the corpus too.
# ---------------------------------------------------------------------------


def test_inventory_reports_read_order_independently_of_the_fused_ranking():
    """Page 6 is the top-scoring chunk but sits sixth in Mongo read order, so
    its inventory position and its fused positions cannot be the same list."""
    probe = _probe({("annual-2024", 7): 5, ("annual-2024", 6): 1,
                    ("annual-2024", 8): 5},
                   scores={"annual-2024-p6-0": 999.0})
    trace = probe.trace(QUESTION, expected_page=6, expected_numbers=[],
                        expected_source_filenames=["annual-2024.pdf"],
                        top_k=3, rerank_top_k=3, context_cap=3)
    assert trace.stage_positions["inventory"] == [6]
    assert trace.stage_positions["corpus"] == [1]
    assert trace.stage_positions["fused_pool"] == [1]
    assert trace.expected_first_position["inventory"] == 6
    assert trace.expected_first_position["fused_pool"] == 1


def test_stage_sizes_separate_the_inventory_from_the_truncated_pool():
    probe = _probe({("annual-2024", 7): 5, ("annual-2024", 6): 1,
                    ("annual-2024", 8): 5},
                   scores={"annual-2024-p6-0": 999.0})
    trace = probe.trace(QUESTION, expected_page=6, expected_numbers=[],
                        expected_source_filenames=["annual-2024.pdf"],
                        top_k=3, rerank_top_k=3, context_cap=3)
    assert trace.metadata_matched_chunk_count == 11
    assert trace.stage_sizes["inventory"] == 11
    assert trace.stage_sizes["corpus"] == 11
    # default pool = max(top_k*2, rerank_top_k*3) = 9, strictly smaller
    assert trace.stage_sizes["fused_pool"] == 9
    assert set(trace.stage_positions) == set(STAGE_ORDER)


def test_pool_narrowing_leaves_the_inventory_untouched():
    """``pool_select`` trims candidates after fusion; the inventory stage is a
    read of the corpus and must not shrink with it."""
    probe = _probe({("annual-2024", 7): 5, ("annual-2024", 6): 1,
                    ("annual-2024", 8): 5},
                   scores={"annual-2024-p6-0": 999.0})
    trace = probe.trace(QUESTION, expected_page=6, expected_numbers=[],
                        expected_source_filenames=["annual-2024.pdf"],
                        top_k=3, rerank_top_k=3, context_cap=3,
                        pool_select=lambda rows: rows[:2])
    assert trace.stage_sizes["inventory"] == 11
    assert trace.stage_sizes["fused_pool"] == 2


def test_a_page_present_only_in_the_inventory_does_not_count_as_reached():
    """Page 6 exists in the corpus but scores below the pool boundary and has
    no dense or lexical hit. It is inventoried, not reached: reporting
    ``expected_stage_reached == "inventory"`` would hide the real failure."""
    scores = {f"annual-2024-p7-{i}": 100.0 - i for i in range(5)}
    scores.update({f"annual-2024-p8-{i}": 90.0 - i for i in range(5)})
    probe = _probe({("annual-2024", 7): 5, ("annual-2024", 6): 1,
                    ("annual-2024", 8): 5}, scores=scores)
    trace = probe.trace(QUESTION, expected_page=6, expected_numbers=[],
                        expected_source_filenames=["annual-2024.pdf"],
                        top_k=3, rerank_top_k=3, context_cap=3)
    assert trace.expected_page_in_inventory is True
    assert trace.expected_first_position["inventory"] == 6
    assert trace.expected_first_position["corpus"] == 11
    assert trace.expected_first_position["fused_pool"] is None
    assert trace.expected_stage_reached == ""


def test_existence_stages_are_declared_as_such():
    """The rule the trace relies on: inventory and corpus answer 'was it ever
    there', every other stage answers 'did ranking reach it'."""
    assert set(STAGE_ORDER) == set(STAGE_DEFINITIONS)
    assert EXISTENCE_STAGES == {"inventory", "corpus"}
    assert EXISTENCE_STAGES < set(STAGE_ORDER)


# ---------------------------------------------------------------------------
# production path: RETRIEVAL_PAGE_QUOTA
# ---------------------------------------------------------------------------

QUESTION_RANKED = "What was the total operating revenue of the entity?"

# (chunk id, document id, physical page, printed label, bm25 score)
_CORPUS = [
    ("a6-1", "annual-2024", 6, 3, 100.0),
    ("a6-2", "annual-2024", 6, 3, 90.0),
    ("a6-3", "annual-2024", 6, 4, 80.0),
    ("q6-1", "quarterly-2024", 6, 6, 70.0),
    ("a7-1", "annual-2024", 7, 5, 60.0),
    ("a7-2", "annual-2024", 7, 5, 50.0),
    ("a9-1", "annual-2024", 9, 7, 40.0),
    ("ax-1", "annual-2024", None, None, 30.0),
    ("ax-2", "annual-2024", "9", None, 20.0),
    ("a8-1", "annual-2024", 8, 6, 10.0),
]


class _NoRerank:
    """Equal scores keep the returned order purely fused-score driven."""

    name = "test-equal-scores"

    def rerank(self, query, documents, top_k=None):
        return [(i, 0.0) for i in range(len(documents))]


class _Repo:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.reads: list[tuple[str, object]] = []

    def find_searchable(self, ctx, document_ids):
        self.reads.append(("find_searchable", ctx))
        wanted = set(document_ids)
        return [dict(c) for c in self.chunks if c["document_id"] in wanted]

    def get_many_by_ids(self, ctx, chunk_ids):
        self.reads.append(("get_many_by_ids", ctx))
        return []


def _retrieve(monkeypatch, *, quota: int, rerank_top_k: int = 5,
              corpus: list[tuple] | None = None):
    """Run the real retrieve() with the datastore and vector reads stubbed out."""
    from app.core.config import get_settings
    from app.db.mongodb.tenant_context import TenantContext

    corpus = corpus or _CORPUS
    real_settings = get_settings()
    retriever = object.__new__(hybrid_module.HybridRetriever)
    retriever.ctx = TenantContext(tenant_id="t-1", user_id="u-1",
                                  workspace_id="ws-a")
    retriever.workspace_id = "ws-a"
    retriever.settings = SimpleNamespace(
        retrieval_top_k=10, reranker_top_k=rerank_top_k,
        hybrid_dense_weight=real_settings.hybrid_dense_weight,
        hybrid_bm25_weight=real_settings.hybrid_bm25_weight,
        retrieval_page_quota=quota)
    retriever.chunks = _Repo([
        {"id": cid, "document_id": doc, "page": page, "printed_page": printed,
         "parent_id": None, "section": "", "text": f"body of {cid}",
         "tables_md": ""}
        for cid, doc, page, printed, _score in corpus])
    retriever.documents = object()

    seen_filters: list[dict] = []

    def _matching_documents(filter_, *, status):
        seen_filters.append(dict(filter_))
        return {doc: {"filename": f"{doc}.pdf", "company": "", "year": None,
                      "quarter": "", "doc_type": ""}
                for _cid, doc, _p, _pr, _s in corpus}

    retriever._matching_documents = _matching_documents
    retriever._workspace_years = lambda status: [2024]
    retriever._count_with = lambda filter_: len(corpus)
    retriever._dense_search = lambda question, filter_, top_k: []
    retriever._bm25_scores = lambda question, chunks: {
        cid: score for cid, _d, _p, _pr, score in corpus
        if any(c["id"] == cid for c in chunks)}

    monkeypatch.setattr(hybrid_module, "run_sync", lambda value: value)
    monkeypatch.setattr(hybrid_module, "get_reranker", lambda: _NoRerank())
    calls: list[int] = []
    real_select = hybrid_module._page_diversity_select

    def spy(chunks, limit, *, per_page):
        calls.append(per_page)
        return real_select(chunks, limit, per_page=per_page)

    monkeypatch.setattr(hybrid_module, "_page_diversity_select", spy)
    result = retriever.retrieve(QUESTION_RANKED)
    return result, calls, seen_filters, retriever


def _ranked_ids() -> list[str]:
    return [cid for cid, *_rest in sorted(_CORPUS, key=lambda r: -r[4])]


def test_disabled_quota_is_exactly_the_original_truncation(monkeypatch):
    baseline, calls, _filters, _r = _retrieve(monkeypatch, quota=0)
    assert calls == []
    assert [c.chunk_id for c in baseline.chunks] == _ranked_ids()[:5]
    # Crowded by siblings, as the pre-quota production path always was.
    assert [c.physical_page for c in baseline.chunks] == [6, 6, 6, 6, 7]


def test_one_per_page_reselects_only_from_the_eligible_candidates(monkeypatch):
    result, calls, _filters, _r = _retrieve(monkeypatch, quota=1)
    assert calls == [1]
    chosen = [c.chunk_id for c in result.chunks]
    assert chosen == ["a6-1", "q6-1", "a7-1", "a9-1", "a8-1"]
    assert set(chosen) <= set(_ranked_ids())
    assert len(result.chunks) == 5
    # page 6 of a different report is a different group, and every source/page
    # association still points at its own document.
    assert {(c.document_id, c.physical_page) for c in result.chunks} == {
        ("annual-2024", 6), ("quarterly-2024", 6), ("annual-2024", 7),
        ("annual-2024", 9), ("annual-2024", 8)}


def test_two_per_page_mode_is_supported(monkeypatch):
    result, calls, _filters, _r = _retrieve(monkeypatch, quota=2)
    assert calls == [2]
    assert [c.chunk_id for c in result.chunks] == [
        "a6-1", "a6-2", "q6-1", "a7-1", "a7-2"]


def test_printed_labels_never_act_as_the_grouping_key(monkeypatch):
    result, *_ = _retrieve(monkeypatch, quota=1)
    by_id = {c.chunk_id: c for c in result.chunks}
    # a6-1 and a6-3 carry different printed labels on one physical page, so only
    # the higher-ranked one survives; a8-1 and q6-1 share the printed label 6 on
    # different physical pages/documents and both survive.
    assert "a6-3" not in by_id
    assert (by_id["a9-1"].physical_page, by_id["a9-1"].printed_page) == (9, 7)
    assert by_id["a9-1"].page_number_type == "printed"
    assert (by_id["a8-1"].printed_page, by_id["a8-1"].physical_page) == (6, 8)
    assert (by_id["q6-1"].printed_page, by_id["q6-1"].physical_page) == (6, 6)


def test_chunks_without_usable_page_metadata_backfill_in_rank_order(monkeypatch):
    # None and the string "9" carry no group key: they are never merged with
    # each other and they queue behind the quota-rejected siblings by rank.
    result, *_ = _retrieve(monkeypatch, quota=1, rerank_top_k=8)
    assert [c.chunk_id for c in result.chunks] == [
        "a6-1", "q6-1", "a7-1", "a9-1", "a8-1", "a6-2", "a6-3", "a7-2"]

    widened, *_ = _retrieve(monkeypatch, quota=1, rerank_top_k=9)
    assert widened.chunks[-1].chunk_id == "ax-1"

    whole = _retrieve(monkeypatch, quota=1, rerank_top_k=20)[0]
    assert len(whole.chunks) == len(_CORPUS)
    assert {c.chunk_id for c in whole.chunks} == {c[0] for c in _CORPUS}


def test_page_quota_preserves_tenant_and_workspace_scoping(monkeypatch):
    for quota in (0, 1, 2):
        result, _calls, filters, retriever = _retrieve(monkeypatch, quota=quota)
        assert filters and all(f["workspace_id"] == "ws-a" for f in filters)
        assert all(f["status"] == "active" for f in filters)
        assert {c.document_id for c in result.chunks} <= {
            doc for _cid, doc, *_rest in _CORPUS}
        assert retriever.chunks.reads
        assert all(read[1].tenant_id == "t-1" and read[1].workspace_id == "ws-a"
                   for read in retriever.chunks.reads)


def test_unpaginated_corpus_degrades_to_the_baseline_ordering(monkeypatch):
    """With no physical page anywhere the quota has nothing to group by, so it
    must reproduce the disabled ordering rather than drop candidates."""
    corpus = [(f"u{i}", "annual-2024", None, None, 100.0 - i * 10)
              for i in range(8)]
    disabled = _retrieve(monkeypatch, quota=0, corpus=corpus)[0]
    enabled = _retrieve(monkeypatch, quota=1, corpus=corpus)[0]
    assert [c.chunk_id for c in enabled.chunks] == [
        c.chunk_id for c in disabled.chunks]
    assert len(enabled.chunks) == 5


def test_selection_is_deterministic_and_never_widens_the_result_set(monkeypatch):
    first = [c.chunk_id for c in _retrieve(monkeypatch, quota=1)[0].chunks]
    for _ in range(3):
        result, *_ = _retrieve(monkeypatch, quota=1)
        assert [c.chunk_id for c in result.chunks] == first
        assert len(result.chunks) <= result.reranked_candidate_count

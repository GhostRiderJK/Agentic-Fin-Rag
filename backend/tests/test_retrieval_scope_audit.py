"""Regression checks for retrieval's metadata prefilter scope."""

from __future__ import annotations


def test_company_relaxation_count_uses_active_workspace_and_all_filters(monkeypatch):
    from app.db.enums import DocumentStatus
    from app.retrieval import hybrid

    class Documents:
        def __init__(self):
            self.kwargs = None

        def find_matching(self, ctx, **kwargs):
            self.kwargs = kwargs
            # A document exists in the exact workspace/status/metadata scope.
            return [{"id": "doc-current-workspace"}]

    class Chunks:
        def find_searchable(self, ctx, document_ids):
            assert document_ids == ["doc-current-workspace"]
            return [{"id": "child-1"}, {"id": "child-2"}]

    retriever = object.__new__(hybrid.HybridRetriever)
    retriever.ctx = object()
    retriever.workspace_id = "workspace-a"
    retriever.documents = Documents()
    retriever.chunks = Chunks()
    monkeypatch.setattr(hybrid, "run_sync", lambda value: value)

    count = retriever._count_with({
        "workspace_id": "workspace-a",
        "status": DocumentStatus.active.value,
        "company": "PIF",
        "year": 2024,
        "quarter": "Q4",
        "doc_type": "annual_report",
    })

    assert count == 2
    assert retriever.documents.kwargs == {
        "workspace_id": "workspace-a",
        "status": DocumentStatus.active.value,
        "company": "PIF",
        "year": 2024,
        "quarter": "Q4",
        "doc_type": "annual_report",
    }


def test_company_relaxation_count_is_zero_when_no_docs_match_scope(monkeypatch):
    from app.retrieval import hybrid

    class Documents:
        def find_matching(self, ctx, **kwargs):
            # Other-workspace matches must not leak into the active scope.
            assert kwargs["workspace_id"] == "workspace-a"
            assert kwargs["status"] == "active"
            return []

    class Chunks:
        def find_searchable(self, ctx, document_ids):
            raise AssertionError("must not query chunks without matching docs")

    retriever = object.__new__(hybrid.HybridRetriever)
    retriever.ctx = object()
    retriever.workspace_id = "workspace-a"
    retriever.documents = Documents()
    retriever.chunks = Chunks()
    monkeypatch.setattr(hybrid, "run_sync", lambda value: value)

    assert retriever._count_with({"company": "PIF"}) == 0


def test_dense_embedding_failure_does_not_silently_degrade_to_bm25(monkeypatch):
    from app.core.errors import ProviderError
    from app.retrieval import hybrid

    class Embeddings:
        def embed(self, _texts):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(hybrid, "get_embeddings", lambda: Embeddings())
    monkeypatch.setattr(hybrid, "get_vectordb", lambda: object())
    retriever = object.__new__(hybrid.HybridRetriever)

    try:
        retriever._dense_search("AuM 2024", {}, top_k=5)
    except ProviderError as exc:
        assert "cannot be completed safely" in exc.message
    else:
        raise AssertionError("embedding failure was hidden")


def test_vector_search_failure_does_not_silently_degrade_to_bm25(monkeypatch):
    from app.core.errors import ProviderError
    from app.retrieval import hybrid

    class Embeddings:
        def embed(self, _texts):
            return [[0.1, 0.2]]

    class VectorDB:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("vector database unavailable")

    monkeypatch.setattr(hybrid, "get_embeddings", lambda: Embeddings())
    monkeypatch.setattr(hybrid, "get_vectordb", lambda: VectorDB())
    retriever = object.__new__(hybrid.HybridRetriever)

    try:
        retriever._dense_search("AuM 2024", {}, top_k=5)
    except ProviderError as exc:
        assert "cannot be completed safely" in exc.message
    else:
        raise AssertionError("vector search failure was hidden")


def test_reranker_scores_are_not_mixed_with_fused_scores_for_unranked_candidates():
    from app.retrieval.hybrid import RetrievedChunk, _sort_reranked_chunks

    def chunk(chunk_id, rerank_score, fused_score):
        return RetrievedChunk(
            chunk_id=chunk_id, parent_id=None, document_id="doc", text="",
            parent_text="", tables_md="", company="", year=None, quarter="",
            doc_type="", section="", page=1, rerank_score=rerank_score,
            final_score=fused_score,
        )

    chunks = [
        chunk("reranked-zero", 0.0, 0.2),
        chunk("unranked", 0.0, 1.0),
        chunk("reranked-positive", 0.4, 0.1),
    ]
    _sort_reranked_chunks(
        chunks,
        {"reranked-zero": 0, "unranked": 2, "reranked-positive": 1},
        {0, 1},
    )

    assert [item.chunk_id for item in chunks] == [
        "reranked-positive", "reranked-zero", "unranked",
    ]


def test_retrieval_uses_fused_order_only_when_reranker_returns_no_scores():
    from app.retrieval.hybrid import RetrievedChunk, _sort_reranked_chunks

    def chunk(chunk_id, score):
        return RetrievedChunk(
            chunk_id=chunk_id, parent_id=None, document_id="doc", text="",
            parent_text="", tables_md="", company="", year=None, quarter="",
            doc_type="", section="", page=1, final_score=score,
        )

    chunks = [chunk("low", 0.1), chunk("high", 0.9)]
    _sort_reranked_chunks(chunks, {"low": 0, "high": 1}, set())

    assert [item.chunk_id for item in chunks] == ["high", "low"]

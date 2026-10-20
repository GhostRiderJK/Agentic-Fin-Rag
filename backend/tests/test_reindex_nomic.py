"""Offline safety tests for the isolated existing-chunk reindexer."""

from __future__ import annotations

from app.core.errors import ProviderError
from app.ingestion.reindex import TARGET_COLLECTION, reindex_existing_chunks


class _Repo:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def find_matching(self, ctx, **kwargs):
        self.calls.append(("documents", ctx.tenant_id, ctx.workspace_id, kwargs))
        return self.rows

    async def list_for_document(self, ctx, document_id):
        self.calls.append(("chunks", ctx.tenant_id, ctx.workspace_id, document_id))
        return self.rows


DOCS = [{
    "id": "doc-1", "tenant_id": "tenant-1", "workspace_id": "workspace-1",
    "status": "active", "company": "PIF", "ticker": "PIF",
    "year": 2024, "quarter": "FY", "doc_type": "annual_report",
    "source_name": "pif-2024.pdf", "source_hash": "sha256-value", "title": "PIF report",
}]
CHUNKS = [
    {"id": "parent-1", "document_id": "doc-1", "workspace_id": "workspace-1",
     "is_parent": True, "text": "Parent", "page": 1, "section": "Overview"},
    {"id": "child-1", "document_id": "doc-1", "workspace_id": "workspace-1",
     "parent_id": "parent-1", "is_parent": False, "text": "Assets were SAR 10 billion.",
     "page": 3, "printed_page": 2, "section": "Assets", "vector_point_id": "existing-point"},
]


class _Embedder:
    name = "ollama"
    dim = 768

    def embed(self, texts):
        return [[float(len(text))] * self.dim for text in texts]


class _DB:
    def __init__(self, collection=TARGET_COLLECTION):
        self._collection = collection
        self.checked = []
        self.upserts = []
        self.deleted = []

    def ensure_collection(self, dim):
        self.checked.append(dim)

    def upsert(self, points):
        self.upserts.extend(points)


def _run(coro):
    try:
        return coro.send(None).value
    except StopIteration as result:
        return result.value


def _go(monkeypatch, *, db=None, embedder=None, doc_rows=None, chunk_rows=None):
    import app.ingestion.reindex as module

    docs = _Repo(DOCS if doc_rows is None else doc_rows)
    chunks = _Repo(CHUNKS if chunk_rows is None else chunk_rows)
    monkeypatch.setattr(module, "run_sync", _run)
    return reindex_existing_chunks(
        tenant_id="tenant-1", workspace_id="workspace-1",
        embedder=embedder or _Embedder(), vector_db=db or _DB(),
        documents=docs, chunks=chunks,
    ), docs, chunks


def test_reindex_only_targets_nomic_collection_and_preserves_associations(monkeypatch):
    db = _DB()
    report, docs, chunks = _go(monkeypatch, db=db)

    assert db._collection == "chunks_nomic768_test"
    assert db.checked == [768]
    assert report.chunks_processed == 1
    assert report.chunks_failed == 0
    assert report.chunks_skipped == 1
    assert report.chunks_seen == 2
    assert docs.calls[0][1:] == (
        "tenant-1", "workspace-1", {"workspace_id": "workspace-1", "status": "active"}
    )
    assert chunks.calls[0][1:3] == ("tenant-1", "workspace-1")
    point = db.upserts[0]
    assert point["id"] == "existing-point"
    assert len(point["vector"]) == 768
    assert point["payload"] == {
        "tenant_id": "tenant-1", "workspace_id": "workspace-1", "document_id": "doc-1",
        "chunk_id": "child-1", "parent_id": "parent-1", "company": "PIF", "ticker": "PIF",
        "year": 2024, "quarter": "FY", "doc_type": "annual_report", "section": "Assets",
        "page": 3, "printed_page": 2, "text": "Assets were SAR 10 billion.", "tables_md": "",
        "status": "active", "source_name": "pif-2024.pdf", "source_hash": "sha256-value",
        "title": "PIF report",
    }
    assert db.deleted == []


def test_reindex_ids_are_repeatable_when_mongo_has_no_vector_point_id(monkeypatch):
    rows = [dict(CHUNKS[1], vector_point_id=None)]
    first_db, second_db = _DB(), _DB()
    _go(monkeypatch, db=first_db, chunk_rows=rows)
    _go(monkeypatch, db=second_db, chunk_rows=rows)
    assert first_db.upserts[0]["id"] == second_db.upserts[0]["id"]


def test_reindex_rejects_wrong_target_before_upsert(monkeypatch):
    db = _DB(collection="chunks")
    try:
        _go(monkeypatch, db=db)
    except ProviderError as exc:
        assert "refuses to write outside" in str(exc)
    else:
        raise AssertionError("wrong collection must be refused")
    assert db.checked == []
    assert db.upserts == []


def test_reindex_records_batch_failure_and_retry_can_succeed(monkeypatch):
    class FlakyDB(_DB):
        def upsert(self, points):
            raise RuntimeError("temporary failure")

    failed, _, _ = _go(monkeypatch, db=FlakyDB())
    assert failed.chunks_failed == 1
    assert failed.chunks_processed == 0
    assert failed.failures[0]["chunk_id"] == "child-1"

    recovered, _, _ = _go(monkeypatch, db=_DB())
    assert recovered.chunks_processed == 1
    assert recovered.chunks_failed == 0


def test_reindex_rejects_non_768_embedding_provider_without_creating_target(monkeypatch):
    class Wrong(_Embedder):
        dim = 1536

    db = _DB()
    try:
        _go(monkeypatch, db=db, embedder=Wrong())
    except ProviderError as exc:
        assert "requires 768-dimensional" in str(exc)
    else:
        raise AssertionError("wrong vector width must be refused")
    assert db.checked == []
    assert db.upserts == []


def test_dry_run_reads_scope_without_initializing_embeddings_or_qdrant(monkeypatch):
    import app.ingestion.reindex as module

    class Forbidden:
        def embed(self, _texts):
            raise AssertionError("dry run must not call Ollama")

    docs = _Repo(DOCS)
    chunks = _Repo(CHUNKS)
    monkeypatch.setattr(module, "run_sync", _run)
    monkeypatch.setattr(module, "get_embeddings", lambda: Forbidden())
    monkeypatch.setattr(module, "QdrantVectorDB", lambda **_: (_ for _ in ()).throw(
        AssertionError("dry run must not connect to Qdrant")
    ))
    report = reindex_existing_chunks(
        tenant_id="tenant-1", workspace_id="workspace-1", dry_run=True,
        embedder=None, documents=docs, chunks=chunks,
    )
    assert report.dry_run is True
    assert report.documents_seen == 1
    assert report.chunks_seen == 2
    assert report.chunks_skipped == 1
    assert report.chunks_ready == 1
    assert report.chunks_processed == report.chunks_failed == 0


def test_reindex_fails_closed_on_cross_workspace_chunk(monkeypatch):
    wrong = [dict(CHUNKS[1], workspace_id="workspace-other")]
    db = _DB()
    report, _, _ = _go(monkeypatch, db=db, chunk_rows=wrong)
    assert report.chunks_processed == 0
    assert report.chunks_failed == 1
    assert db.upserts == []

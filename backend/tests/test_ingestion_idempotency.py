"""Offline tests for ingestion generation identity and retry cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.mongodb.tenant_context import TenantContext
from app.ingestion import pipeline


class _Documents:
    def __init__(self):
        self.items: dict[str, dict] = {}

    async def get_by_id(self, _ctx, doc_id):
        return self.items.get(doc_id)

    async def insert_one(self, _ctx, item, *, doc_id):
        self.items[doc_id] = dict(item)
        return item

    async def find_many(self, _ctx, query):
        matches = []
        for item in self.items.values():
            if item.get("workspace_id") != query["workspace_id"]:
                continue
            if any(
                item.get(key) == value
                for clause in query["$or"]
                for key, value in clause.items()
            ):
                matches.append(item)
        return matches

    async def update_by_id(self, _ctx, doc_id, fields):
        self.items[doc_id].update(fields)


class _Chunks:
    def __init__(self):
        self.items: dict[str, list[dict]] = {}

    async def list_for_document(self, _ctx, doc_id):
        return self.items.get(doc_id, [])

    async def delete_for_document(self, _ctx, doc_id):
        self.items.pop(doc_id, None)


class _Financials:
    async def count(self, _ctx, _query):
        return 0

    async def detach_document(self, _ctx, _doc_id):
        return 0


class _Vectors:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_by_document(self, doc_id):
        self.deleted.append(doc_id)


def _run_immediate(coro):
    """Run fake repository coroutines without opening a platform event loop."""
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("test repository coroutine unexpectedly awaited I/O")


def _install_fakes(monkeypatch, *, fail_first=False):
    documents, chunks, vectors = _Documents(), _Chunks(), _Vectors()
    monkeypatch.setattr(pipeline, "_documents", documents)
    monkeypatch.setattr(pipeline, "_chunks", chunks)
    monkeypatch.setattr(pipeline, "_financials", _Financials())
    monkeypatch.setattr(pipeline, "run_sync", _run_immediate)
    monkeypatch.setattr(pipeline, "get_vectordb", lambda: vectors)
    monkeypatch.setattr(pipeline, "_persist_financials", lambda *a, **k: None)

    attempts = 0

    def persist(*, document_id, **_kwargs):
        nonlocal attempts
        attempts += 1
        if fail_first and attempts == 1:
            chunks.items[document_id] = [{"id": "partial", "is_parent": False}]
            raise RuntimeError("simulated embedding failure")
        rows = [{"id": f"chunk-{document_id}", "is_parent": False}]
        chunks.items[document_id] = rows
        return [pipeline.ChunkRecord(
            chunk_id=rows[0]["id"], parent_id=None, is_parent=False,
            text="report content", section="", page=1,
        )]

    monkeypatch.setattr(pipeline, "_embed_and_persist", persist)
    return documents, chunks, vectors


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    source = tmp_path / "annual-report.txt"
    source.write_text("PIF annual report content", encoding="utf-8")
    return source


def test_first_and_unchanged_ingestion_reuse_the_published_generation(
    monkeypatch, source_file
):
    _documents, chunks, _vectors = _install_fakes(monkeypatch)
    ctx = TenantContext(tenant_id="tenant", workspace_id="workspace")

    first = pipeline.ingest_document(ctx, file_path=str(source_file), hint_year=2023)
    second = pipeline.ingest_document(ctx, file_path=str(source_file), hint_year=2023)

    assert first.document_id == second.document_id
    assert first.n_children == second.n_children == 1
    assert len(chunks.items) == 1


def test_changed_source_publishes_new_generation_and_retires_old_chunks(
    monkeypatch, source_file
):
    documents, chunks, vectors = _install_fakes(monkeypatch)
    ctx = TenantContext(tenant_id="tenant", workspace_id="workspace")

    first = pipeline.ingest_document(ctx, file_path=str(source_file), hint_year=2023)
    source_file.write_text("PIF updated annual report content", encoding="utf-8")
    second = pipeline.ingest_document(ctx, file_path=str(source_file), hint_year=2023)

    assert first.document_id != second.document_id
    assert documents.items[first.document_id]["status"] == "superseded"
    assert first.document_id not in chunks.items
    assert first.document_id in vectors.deleted
    assert second.document_id in chunks.items


def test_partial_failure_is_cleaned_before_retry(monkeypatch, source_file):
    _documents, chunks, vectors = _install_fakes(monkeypatch, fail_first=True)
    ctx = TenantContext(tenant_id="tenant", workspace_id="workspace")

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        pipeline.ingest_document(ctx, file_path=str(source_file), hint_year=2023)

    recovered = pipeline.ingest_document(ctx, file_path=str(source_file), hint_year=2023)

    assert chunks.items[recovered.document_id][0]["id"] != "partial"
    assert recovered.document_id in vectors.deleted

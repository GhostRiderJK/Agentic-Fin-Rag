"""Offline tests for the document unpublish order and its failure modes.

Deleting a document touches two stores. The route must abort before it drops
the Mongo rows when the vector store is unreachable: those rows are the only
record of which vectors belonged to the document, so deleting them first would
leave live vectors that nothing can still address.
"""

from __future__ import annotations

import pytest

from app.api import documents
from app.core.errors import ForbiddenError, NotFoundError, ProviderError
from app.db.models import Role, User
from app.db.mongodb.tenant_context import TenantContext

DOC_ID = "doc-1"
WS_ID = "ws-1"
STORED = {"id": DOC_ID, "workspace_id": WS_ID}


class _Recorder:
    """Stands in for all three repositories so call order is one flat list."""

    def __init__(self, doc):
        self.calls: list[str] = []
        self.doc = doc

    async def get_by_id(self, _ctx, doc_id):
        self.calls.append("documents.get_by_id")
        return self.doc

    async def delete_by_id(self, _ctx, doc_id):
        self.calls.append("documents.delete_by_id")

    async def delete_for_document(self, _ctx, doc_id):
        self.calls.append("chunks.delete_for_document")

    async def detach_document(self, _ctx, doc_id):
        self.calls.append("financials.detach_document")


class _Vectors:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.deleted: list[str] = []

    def delete_by_document(self, document_id):
        if self.error is not None:
            raise self.error
        self.deleted.append(document_id)


def _drain(coro):
    """Run a fake repository coroutine without opening a platform event loop."""
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("test repository coroutine unexpectedly awaited I/O")


def _install(monkeypatch, *, vectors, doc=STORED):
    repo = _Recorder(doc)
    monkeypatch.setattr(documents, "_documents", repo)
    monkeypatch.setattr(documents, "_chunks", repo)
    monkeypatch.setattr(documents, "_financials", repo)
    monkeypatch.setattr(documents, "run_sync", _drain)
    monkeypatch.setattr(documents, "get_vectordb", lambda: vectors)
    return repo


def _ctx() -> TenantContext:
    return TenantContext(tenant_id="t-1", user_id="u-1", workspace_id=WS_ID)


def _user(role: Role = Role.analyst) -> User:
    return User(tenant_id="t-1", email="a@b.c", name="A", role=role, password_hash="x")


def test_viewer_cannot_delete(monkeypatch):
    repo = _install(monkeypatch, vectors=_Vectors())
    with pytest.raises(ForbiddenError):
        documents.delete_document(_ctx(), DOC_ID, _user(Role.viewer))
    assert repo.calls == []


def test_unknown_document_is_not_found(monkeypatch):
    vectors = _Vectors()
    repo = _install(monkeypatch, vectors=vectors, doc=None)
    with pytest.raises(NotFoundError):
        documents.delete_document(_ctx(), DOC_ID, _user())
    assert repo.calls == ["documents.get_by_id"]
    assert vectors.deleted == []


def test_document_from_another_workspace_is_not_found(monkeypatch):
    repo = _install(monkeypatch, vectors=_Vectors(),
                    doc={"id": DOC_ID, "workspace_id": "ws-other"})
    with pytest.raises(NotFoundError):
        documents.delete_document(_ctx(), DOC_ID, _user())
    assert repo.calls == ["documents.get_by_id"]


def test_vector_delete_failure_aborts_before_touching_mongo(monkeypatch):
    vectors = _Vectors(error=RuntimeError("qdrant is holding the storage lock"))
    repo = _install(monkeypatch, vectors=vectors)

    with pytest.raises(ProviderError, match="still in the vector store"):
        documents.delete_document(_ctx(), DOC_ID, _user())

    assert repo.calls == ["documents.get_by_id"]
    assert vectors.deleted == []


def test_vector_delete_failure_is_a_bad_gateway_not_a_crash(monkeypatch):
    _install(monkeypatch, vectors=_Vectors(error=RuntimeError("unreachable")))
    with pytest.raises(ProviderError) as excinfo:
        documents.delete_document(_ctx(), DOC_ID, _user())
    assert excinfo.value.status_code == 502
    assert excinfo.value.details["provider"] == "qdrant"


def test_happy_path_unpublishes_vectors_then_rows(monkeypatch):
    vectors = _Vectors()
    repo = _install(monkeypatch, vectors=vectors)

    assert documents.delete_document(_ctx(), DOC_ID, _user()) == {"deleted": DOC_ID}

    assert vectors.deleted == [DOC_ID]
    assert repo.calls == [
        "documents.get_by_id",
        # nothing searchable first, then the records that describe it
        "chunks.delete_for_document",
        "financials.detach_document",
        "documents.delete_by_id",
    ]

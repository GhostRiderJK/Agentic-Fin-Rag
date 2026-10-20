"""Qdrant initialization must preserve existing vectors and dimensions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.errors import ProviderError
from app.providers.vectordb import InMemoryVectorDB, QdrantVectorDB


def _db(client):
    db = object.__new__(QdrantVectorDB)
    db._client = client
    db._collection = "isolated-test"
    db._dim = None
    return db


def test_existing_qdrant_dimension_mismatch_fails_without_replacing_collection():
    class Client:
        recreated = False

        def collection_exists(self, _name):
            return True

        def get_collection(self, _name):
            return SimpleNamespace(config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=1536)),
            ))

        def recreate_collection(self, **_kwargs):
            self.recreated = True

    client = Client()
    with pytest.raises(ProviderError, match="refusing to modify existing vectors"):
        _db(client).ensure_collection(768)

    assert client.recreated is False


def test_existing_qdrant_collection_with_matching_dimension_is_reused():
    class Client:
        def collection_exists(self, _name):
            return True

        def get_collection(self, _name):
            return SimpleNamespace(config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=768)),
            ))

    db = _db(Client())
    db.ensure_collection(768)

    assert db._dim == 768


def test_missing_qdrant_collection_is_created_without_recreate():
    class Client:
        def __init__(self):
            self.created = None

        def collection_exists(self, _name):
            return False

        def create_collection(self, **kwargs):
            self.created = kwargs

        def recreate_collection(self, **_kwargs):
            raise AssertionError("must not use a destructive collection operation")

    client = Client()
    db = _db(client)
    db.ensure_collection(768)

    assert client.created["collection_name"] == "isolated-test"
    assert client.created["vectors_config"].size == 768
    assert db._dim == 768


def test_in_memory_store_rejects_query_and_point_dimension_mismatches():
    db = InMemoryVectorDB()
    db.ensure_collection(2)

    with pytest.raises(ProviderError, match="point vector width"):
        db.upsert([{
            "id": "bad-width", "vector": [0.1], "payload": {},
        }])
    with pytest.raises(ProviderError, match="query vector width"):
        db.search([0.1], top_k=1)


def test_remote_qdrant_initialization_failure_does_not_select_memory(monkeypatch):
    from app.providers import vectordb

    vectordb.reset_vectordb()
    monkeypatch.setattr(
        vectordb, "get_settings",
        lambda: SimpleNamespace(qdrant_mode="remote", qdrant_path="unused"),
    )
    monkeypatch.setattr(
        vectordb, "QdrantVectorDB",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    try:
        with pytest.raises(ProviderError, match="refusing to use a local memory store"):
            vectordb.get_vectordb()
    finally:
        vectordb.reset_vectordb()


def test_qdrant_count_failure_is_not_reported_as_empty_collection():
    class Client:
        def count(self, **_kwargs):
            raise RuntimeError("unavailable")

    db = _db(Client())
    with pytest.raises(ProviderError, match="Could not read Qdrant collection count"):
        db.count()

"""Guarded pipelines against a REAL MongoDB server.

The unit tests prove the guard rewrites pipelines correctly. This file proves
the server actually honours those rewrites: tenant B's data must not leak
through $lookup, $graphLookup, $unionWith or $facet.

Every case also runs a *control*: the same pipeline with only the root
$match (what the old repository did). The control must leak. If it does not,
the fixture data is not exercising the hole and the test would pass
vacuously.

Uses a throwaway database named finrag_test_<uuid> and drops only that.
Never touches the dev database.

Run (PowerShell):
    python -m pytest -v -s tests/test_pipeline_guard_integration.py
Server: MONGODB_TEST_URL, default mongodb://127.0.0.1:27018/?directConnection=true
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

pymongo = pytest.importorskip("pymongo")

from app.db.mongodb.pipeline_guard import guard_pipeline  # noqa: E402

if not os.path.exists(os.path.join(os.path.dirname(__file__), "__init__.py")):
    from mongo_support import enforce_reachable  # noqa: E402
else:
    from tests.mongo_support import enforce_reachable  # noqa: E402

pytestmark = pytest.mark.integration

TEST_URL = os.environ.get(
    "MONGODB_TEST_URL", "mongodb://127.0.0.1:27018/?directConnection=true"
)
A, B = "tenant-a", "tenant-b"
SECRET = "SECRET-B"
ALLOW = ("chunks", "nodes")


@pytest.fixture(scope="module")
def db():
    client = pymongo.MongoClient(TEST_URL, serverSelectionTimeoutMS=3000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        client.close()
        reason = (str(exc).splitlines() or [type(exc).__name__])[0][:120]
        enforce_reachable(TEST_URL, reason)
    name = f"finrag_test_{uuid.uuid4().hex}"
    database = client[name]
    info = client.server_info()
    hello = client.admin.command("hello")
    print(
        f"\n[integration] url={TEST_URL} database={name} "
        f"version={info.get('version')} setName={hello.get('setName')} "
        f"isWritablePrimary={hello.get('isWritablePrimary')}"
    )

    database.documents.insert_many(
        [
            {"tenant_id": A, "doc_id": "d1", "title": "A doc"},
            {"tenant_id": A, "doc_id": "n1", "title": "A graph doc"},
            {"tenant_id": B, "doc_id": "d1", "title": SECRET},
            {"tenant_id": B, "doc_id": "n1", "title": SECRET},
        ]
    )
    database.chunks.insert_many(
        [
            {"tenant_id": A, "doc_id": "d1", "text": "a-chunk"},
            {"tenant_id": B, "doc_id": "d1", "text": SECRET},
        ]
    )
    database.nodes.insert_many(
        [
            {"tenant_id": A, "id": "n1", "parent": "n2", "note": "a-node"},
            {"tenant_id": B, "id": "n2", "parent": None, "note": SECRET},
        ]
    )
    try:
        yield database
    finally:
        client.drop_database(name)  # only the throwaway database
        client.close()


def _dump(rows) -> str:
    return json.dumps(rows, default=str)


def _guarded(db, pipeline):
    stages = guard_pipeline(pipeline, tenant_id=A, allow_foreign_collections=ALLOW)
    return list(db.documents.aggregate(stages))


def _root_only(db, pipeline):
    """What the old repository did: only the root $match."""
    return list(db.documents.aggregate([{"$match": {"tenant_id": A}}, *pipeline]))


LOOKUP = {
    "$lookup": {
        "from": "chunks",
        "let": {"d": "$doc_id"},
        "pipeline": [{"$match": {"$expr": {"$eq": ["$doc_id", "$$d"]}}}],
        "as": "joined",
    }
}
GRAPH = {
    "$graphLookup": {
        "from": "nodes",
        "startWith": "$doc_id",
        "connectFromField": "parent",
        "connectToField": "id",
        "as": "chain",
    }
}
UNION_STRING = {"$unionWith": "chunks"}
UNION_DOC = {"$unionWith": {"coll": "chunks", "pipeline": [{"$match": {}}]}}
FACET_LOOKUP = {"$facet": {"f": [LOOKUP]}}

CASES = {
    "lookup": [LOOKUP],
    "graphLookup": [GRAPH],
    "unionWith_string": [UNION_STRING],
    "unionWith_document": [UNION_DOC],
    "facet_lookup": [FACET_LOOKUP],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_unguarded_control_leaks_so_the_case_is_meaningful(db, name):
    assert SECRET in _dump(_root_only(db, CASES[name]))


@pytest.mark.parametrize("name", sorted(CASES))
def test_guarded_pipeline_never_returns_another_tenants_data(db, name):
    rows = _guarded(db, CASES[name])
    assert rows, "guarded pipeline should still return the caller's own data"
    assert SECRET not in _dump(rows)


def test_guarded_root_scan_only_sees_the_callers_documents(db):
    rows = _guarded(db, [{"$group": {"_id": "$tenant_id", "n": {"$sum": 1}}}])
    assert rows == [{"_id": A, "n": 2}]


def test_guarded_lookup_still_joins_the_callers_own_chunks(db):
    rows = _guarded(db, [LOOKUP, {"$match": {"doc_id": "d1"}}])
    assert [c["text"] for c in rows[0]["joined"]] == ["a-chunk"]


def test_guarded_graph_lookup_still_follows_the_callers_own_edges(db):
    # n1 (tenant A) points at n2, which only exists for tenant B: the walk must
    # stop at n1 instead of crossing into tenant B.
    rows = _guarded(db, [GRAPH, {"$match": {"doc_id": "n1"}}])
    assert [n["id"] for n in rows[0]["chain"]] == ["n1"]

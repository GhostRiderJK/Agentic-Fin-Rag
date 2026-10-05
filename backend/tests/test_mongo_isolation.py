"""Cross-tenant isolation suite for the MongoDB repository layer.

Every test here tries to do something hostile: read, update, delete,
count, aggregate, or re-own another tenant's data. All of them MUST fail
closed. If any of these tests starts passing (i.e. the hostile action
succeeds), tenant isolation is broken — treat it as a release blocker.

Runs against mongomock-motor (in-memory, async), so no MongoDB server is
needed. mongomock-motor implements the Motor API surface on top of the
mongomock in-memory engine.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("mongomock_motor", reason="mongomock-motor not installed")

import mongomock_motor  # noqa: E402

from app.db.mongodb import collections as C  # noqa: E402
from app.db.mongodb.base_repository import BaseRepository  # noqa: E402
from app.db.mongodb.client import set_test_database  # noqa: E402
from app.db.mongodb.repositories import (  # noqa: E402
    ChunkRepository,
    DocumentRepository,
    MessageRepository,
    StructuredFinancialRepository,
    UserRepository,
    WorkspaceRepository,
)
from app.db.mongodb.tenant_context import TenantContext, TenantContextError  # noqa: E402

# Check if tests/__init__.py exists to determine import style
if not os.path.exists(os.path.join(os.path.dirname(__file__), '__init__.py')):
    from mongo_support import _open_real as open_real
else:
    from tests.mongo_support import _open_real as open_real


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_loop_holder = {"loop": None}


def run(coro):
    return _loop_holder["loop"].run_until_complete(coro)


@pytest.fixture()
def ctx_a() -> TenantContext:
    return TenantContext(tenant_id="tenant-AAA", user_id="user-a1")


@pytest.fixture()
def ctx_b() -> TenantContext:
    return TenantContext(tenant_id="tenant-BBB", user_id="user-b1")


@pytest.fixture(params=["mock", pytest.param("real", marks=pytest.mark.integration)])
def mongo_db(request):
    loop = asyncio.new_event_loop()
    _loop_holder["loop"] = loop
    close = None
    try:
        if request.param == "mock":
            mongomock_motor = pytest.importorskip(
                "mongomock_motor", reason="mongomock-motor not installed"
            )
            database = mongomock_motor.AsyncMongoMockClient().get_database(
                f"finrag_test_{uuid.uuid4().hex}"
            )
            set_test_database(database)
        else:
            database, close = open_real(run)   # skips quickly when the server is down
        yield database
    finally:
        if close:
            close()
        set_test_database(None)
        loop.close()
        _loop_holder["loop"] = None


@pytest.fixture()
def seeded(mongo_db, ctx_a, ctx_b):
    """Seed one document in each tenant for every tenant-owned collection."""
    async def _seed():
        users = UserRepository()
        ws = WorkspaceRepository()
        docs = DocumentRepository()
        chunks = ChunkRepository()
        fin = StructuredFinancialRepository()
        msgs = MessageRepository()

        # Tenant A data
        await ws.insert_one(ctx_a, {"name": "A workspace"})
        await users.insert_one(ctx_a, {"email": "a@tenant-a.test", "name": "User A"})
        await docs.insert_one(ctx_a, {"title": "A doc", "workspace_id": "ws-a",
                                      "company": "ACME", "year": 2025})
        await chunks.insert_one(ctx_a, {"document_id": "doc-a", "text": "A text"})
        await fin.insert_one(ctx_a, {"company": "ACME", "metric": "revenue",
                                     "fiscal_year": 2025, "value": 120.0})
        await msgs.insert_one(ctx_a, {"conversation_id": "conv-a", "content": "hello A"})

        # Tenant B data (same shapes, different ids/values)
        await ws.insert_one(ctx_b, {"name": "B workspace"})
        await users.insert_one(ctx_b, {"email": "b@tenant-b.test", "name": "User B"})
        await docs.insert_one(ctx_b, {"title": "B doc", "workspace_id": "ws-b",
                                      "company": "ACME", "year": 2025})
        await chunks.insert_one(ctx_b, {"document_id": "doc-b", "text": "B text"})
        await fin.insert_one(ctx_b, {"company": "ACME", "metric": "revenue",
                                     "fiscal_year": 2025, "value": 999.0})
        await msgs.insert_one(ctx_b, {"conversation_id": "conv-b", "content": "hello B"})

    run(_seed())
    return mongo_db


# ---------------------------------------------------------------------------
# Context requirements
# ---------------------------------------------------------------------------


class TestContextRequired:
    """No repository call may run without a TenantContext."""

    def test_insert_one_requires_context(self, mongo_db):
        repo = DocumentRepository()
        with pytest.raises(TenantContextError):
            run(repo.insert_one(None, {"title": "x"}))

    def test_find_one_requires_context(self, mongo_db):
        repo = DocumentRepository()
        with pytest.raises(TenantContextError):
            run(repo.find_one(None))

    def test_update_requires_context(self, mongo_db):
        repo = DocumentRepository()
        with pytest.raises(TenantContextError):
            run(repo.update_one(None, {"id": "doc"}, {"status": "archived"}))

    def test_delete_requires_context(self, mongo_db):
        repo = DocumentRepository()
        with pytest.raises(TenantContextError):
            run(repo.delete_one(None, {"id": "doc"}))

    def test_count_requires_context(self, mongo_db):
        repo = DocumentRepository()
        with pytest.raises(TenantContextError):
            run(repo.count(None))

    def test_aggregate_requires_context(self, mongo_db):
        repo = DocumentRepository()
        with pytest.raises(TenantContextError):
            run(repo.aggregate(None, [{"$group": {"_id": None}}]))

    def test_empty_tenant_id_rejected(self):
        with pytest.raises(TenantContextError):
            TenantContext(tenant_id="")


# ---------------------------------------------------------------------------
# Cross-tenant reads must fail closed
# ---------------------------------------------------------------------------


class TestCrossTenantReads:
    def test_find_one_ignores_foreign_tenant_id(self, seeded, ctx_a, ctx_b):
        """If a caller supplies a tenant_id in the query, it is ignored and
        the context's tenant_id is used instead."""
        repo = UserRepository()
        # Ask for tenant B's id, but we are tenant A -> should get tenant A's user
        result = run(
            repo.find_one(ctx_a, {"tenant_id": "tenant-BBB"})
        )
        assert result is not None
        assert result["email"] == "a@tenant-a.test"
        assert result["tenant_id"] == "tenant-AAA"

    def test_get_by_id_returns_none_for_foreign_id(self, seeded, ctx_a, ctx_b):
        docs = DocumentRepository()
        b_doc = run(docs.find_one(ctx_b, {"title": "B doc"}))
        assert b_doc is not None
        # Tenant A asks for tenant B's document by its real id.
        stolen = run(docs.get_by_id(ctx_a, b_doc["id"]))
        assert stolen is None

    def test_find_many_only_returns_own_tenant(self, seeded, ctx_a, ctx_b):
        docs = DocumentRepository()
        a_docs = run(docs.find_many(ctx_a, {}))
        assert len(a_docs) == 1
        assert a_docs[0]["tenant_id"] == "tenant-AAA"

    def test_get_many_by_ids_skips_foreign_chunks(self, seeded, ctx_a, ctx_b):
        chunks = ChunkRepository()
        b_chunk = run(chunks.find_one(ctx_b, {"document_id": "doc-b"}))
        a_chunk = run(chunks.find_one(ctx_a, {"document_id": "doc-a"}))
        fetched = run(chunks.get_many_by_ids(ctx_a, [b_chunk["id"], a_chunk["id"]]))
        ids = {d["id"] for d in fetched}
        assert b_chunk["id"] not in ids
        assert a_chunk["id"] in ids

    def test_count_is_tenant_scoped(self, seeded, ctx_a):
        docs = DocumentRepository()
        assert run(docs.count(ctx_a, {})) == 1
        assert run(docs.count(ctx_b := TenantContext("tenant-BBB"), {})) == 1

    def test_metadata_lookup_is_tenant_scoped(self, seeded, ctx_a, ctx_b):
        """Both tenants have ACME 2025 revenue, but each sees only its own
        value — this is the guard behind the SQL tool-calling route."""
        fin = StructuredFinancialRepository()
        for ctx in (ctx_a, ctx_b):
            rows = run(fin.get_metric(ctx, company="ACME", metric="revenue",
                                      fiscal_year=2025))
            assert len(rows) == 1
        a_rows = run(fin.get_metric(ctx_a, company="ACME", metric="revenue",
                                    fiscal_year=2025))
        b_rows = run(fin.get_metric(ctx_b, company="ACME", metric="revenue",
                                    fiscal_year=2025))
        assert a_rows[0]["value"] == 120.0
        assert b_rows[0]["value"] == 999.0

    def test_aggregate_cannot_cross_tenants(self, seeded, ctx_a):
        """A pipeline that drops the tenant match still only sees tenant A's
        documents, because the tenant $match is prepended server-side."""
        docs = DocumentRepository()
        pipeline = [
            {"$match": {"company": "ACME"}},   # matches both tenants
            {"$group": {"_id": "$tenant_id", "n": {"$sum": 1}}},
        ]
        grouped = run(docs.aggregate(ctx_a, pipeline))
        assert len(grouped) == 1
        assert grouped[0]["_id"] == "tenant-AAA"


# ---------------------------------------------------------------------------
# Cross-tenant writes must fail closed
# ---------------------------------------------------------------------------


class TestCrossTenantWrites:
    def test_update_foreign_id_is_noop(self, seeded, ctx_a, ctx_b):
        docs = DocumentRepository()
        b_doc = run(docs.find_one(ctx_b, {"title": "B doc"}))
        modified = run(docs.update_by_id(ctx_a, b_doc["id"],
                                         {"status": "archived"}))
        assert modified == 0
        b_doc_after = run(docs.get_by_id(ctx_b, b_doc["id"]))
        assert b_doc_after.get("status") != "archived"  # untouched

    def test_delete_foreign_id_is_noop(self, seeded, ctx_a, ctx_b):
        docs = DocumentRepository()
        b_doc = run(docs.find_one(ctx_b, {"title": "B doc"}))
        deleted = run(docs.delete_by_id(ctx_a, b_doc["id"]))
        assert deleted == 0
        assert run(docs.get_by_id(ctx_b, b_doc["id"])) is not None

    def test_update_many_cannot_reach_foreign_rows(self, seeded, ctx_a):
        docs = DocumentRepository()
        modified = run(docs.update_many(ctx_a, {}, {"status": "hacked"}))
        assert modified == 1  # only tenant A's own doc
        b_doc = run(docs.find_one(TenantContext("tenant-BBB"), {"title": "B doc"}))
        assert b_doc.get("status") != "hacked"

    def test_delete_many_cannot_reach_foreign_rows(self, seeded, ctx_a, ctx_b):
        docs = DocumentRepository()
        deleted = run(docs.delete_many(ctx_a, {}))
        assert deleted == 1
        b_doc = run(docs.find_one(ctx_b, {"title": "B doc"}))
        assert b_doc is not None

    def test_insert_stamps_callers_tenant_not_payloads(self, mongo_db, ctx_a):
        """A caller that supplies tenant_id in the payload gets overwritten
        with the context's tenant — payloads can't claim foreign tenancy."""
        docs = DocumentRepository()
        doc = run(docs.insert_one(ctx_a, {"title": "sneaky",
                                          "tenant_id": "tenant-BBB"}))
        assert doc["tenant_id"] == "tenant-AAA"
        # And tenant B genuinely cannot see it.
        assert run(docs.find_one(ctx_b_id := TenantContext("tenant-BBB"),
                                 {"title": "sneaky"})) is None

    def test_update_cannot_change_tenant_id(self, seeded, ctx_a):
        docs = DocumentRepository()
        a_doc = run(docs.find_one(ctx_a, {"title": "A doc"}))
        with pytest.raises(TenantContextError):
            run(docs.update_by_id(ctx_a, a_doc["id"],
                                  {"tenant_id": "tenant-BBB"}))
        # $set operator form is blocked too.
        with pytest.raises(TenantContextError):
            run(docs.update_by_id(ctx_a, a_doc["id"],
                                  {"$set": {"tenant_id": "tenant-BBB"}}))


# ---------------------------------------------------------------------------
# Unique-per-tenant constraints
# ---------------------------------------------------------------------------


class TestPerTenantUniqueness:
    def test_same_email_allowed_in_different_tenants(self, seeded, ctx_a, ctx_b):
        """Email uniqueness is per (tenant, email), not global — mirrors the
        compound index uq_user_tenant_email."""
        users = UserRepository()
        # Index enforcement itself is covered in test_mongo_bootstrap.py;
        # here we assert the *lookup* scoping:        assert run(users.get_by_email(ctx_a, "a@tenant-a.test")) is not None
        assert run(users.get_by_email(ctx_b, "a@tenant-a.test")) is None
        # Tenant B has its own user with its own email.
        assert run(users.get_by_email(ctx_b, "b@tenant-b.test")) is not None


# ---------------------------------------------------------------------------
# Super-admin bypass is explicit and auditable
# ---------------------------------------------------------------------------


class TestSuperAdminBypass:
    def test_super_admin_can_span_tenants_only_when_flagged(self, seeded, ctx_a):
        docs = DocumentRepository()
        super_ctx = TenantContext(tenant_id="tenant-AAA", is_super_admin=True)
        # With the bypass, an unfiltered aggregate spans tenants.
        grouped = run(docs.aggregate(super_ctx,
                                     [{"$group": {"_id": "$tenant_id",
                                                  "n": {"$sum": 1}}}]))
        tenants = {g["_id"] for g in grouped}
        assert tenants == {"tenant-AAA", "tenant-BBB"}
        # Without the flag, the same caller sees only their own tenant.
        normal = run(docs.aggregate(ctx_a,
                                    [{"$group": {"_id": "$tenant_id",
                                             "n": {"$sum": 1}}}]))
        assert {g["_id"] for g in normal} == {"tenant-AAA"}
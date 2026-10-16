"""Repository-level hardening tests.

Covers, through the real repository classes:

* Decimal128 money fields: exact round trips, floats refused.
* The (tenant_id, email) uniqueness rule, per tenant.
* Hostile aggregation pipelines refused before they reach the database.
* Super-admin access: audited BEFORE it runs, and refused when the audit
  record cannot be written (fail closed).

Every test using the ``db`` fixture runs twice: on mongomock-motor ("mock")
and on a real MongoDB ("real", marked ``integration``, skipped when the
server is unreachable). The real server is MONGODB_TEST_URL, default
mongodb://127.0.0.1:27018/?directConnection=true, and each run uses a
throwaway database named finrag_test_<uuid> that is dropped afterwards.

The tests are plain sync functions driving one private event loop per test
(``run`` fixture), so they behave the same whether pytest-asyncio is in
"auto" or "strict" mode and whichever directory pytest is started from.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from decimal import Decimal

import pytest

pytest.importorskip("bson", reason="pymongo/bson not installed")

from bson.decimal128 import Decimal128  # noqa: E402

from app.db.mongodb import collections as C  # noqa: E402
from app.db.mongodb.base_repository import AuditWriteError, BaseRepository  # noqa: E402
from app.db.mongodb.client import set_test_database  # noqa: E402
from app.db.mongodb.decimal_codec import DecimalFieldError  # noqa: E402
from app.db.mongodb.pipeline_guard import PipelineGuardError  # noqa: E402
from app.db.mongodb.repositories import (  # noqa: E402
    ChunkRepository,
    DocumentRepository,
    UserRepository,
)
from app.db.mongodb.tenant_context import TenantContext  # noqa: E402

# Check if tests/__init__.py exists to determine import style
if not os.path.exists(os.path.join(os.path.dirname(__file__), '__init__.py')):
    from mongo_support import (
        DEFAULT_URL,
        _PROBES,
        _unreachable_reason,
        enforce_reachable,
        run_db,
        _open_real,
    )
else:
    from tests.mongo_support import (
        DEFAULT_URL,
        _PROBES,
        _unreachable_reason,
        enforce_reachable,
        run_db,
        _open_real,
    )


class LedgerRepository(BaseRepository):
    """Test-local repository with a declared decimal column."""

    collection_name = C.STRUCTURED_FINANCIALS
    decimal_fields = frozenset({"value"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def run():
    """Drive coroutines on one private loop for the whole test."""
    loop = asyncio.new_event_loop()
    try:
        yield loop.run_until_complete
    finally:
        loop.close()


_PROBES: dict = {}


def _unreachable_reason(pymongo, url):
    """None when the server answers, otherwise a short reason.

    Cached for the session so a missing server costs one timeout in total
    instead of one per test.
    """
    if url not in _PROBES:
        probe = pymongo.MongoClient(url, serverSelectionTimeoutMS=2000)
        try:
            probe.admin.command("ping")
            _PROBES[url] = None
        except Exception as exc:
            _PROBES[url] = (str(exc).splitlines() or [type(exc).__name__])[0][:120]
        finally:
            probe.close()
    return _PROBES[url]


def run_db(run, make_awaitable):
    """Call a raw Motor method *inside* the running loop.

    Motor binds to the event loop that is current at its first call. Writing
    ``run(db[...].method(...))`` evaluates the method before the loop is
    running, which can bind Motor to the wrong loop. Repository methods are
    coroutines and are safe; direct database calls must go through here.
    """

    async def _go():
        return await make_awaitable()

    return run(_go())


def _open_real(run):
    pymongo = pytest.importorskip("pymongo")
    motor = pytest.importorskip("motor.motor_asyncio")
    url = os.environ.get("MONGODB_TEST_URL", DEFAULT_URL)
    reason = _unreachable_reason(pymongo, url)
    enforce_reachable(url, reason)
    admin = pymongo.MongoClient(url, serverSelectionTimeoutMS=3000)

    name = f"finrag_test_{uuid.uuid4().hex}"

    async def _connect():
        # Created inside the running loop so Motor binds to it.
        return motor.AsyncIOMotorClient(
            url, serverSelectionTimeoutMS=3000, uuidRepresentation="standard"
        )

    client = run(_connect())
    database = client[name]
    set_test_database(database)
    hello = admin.admin.command("hello")
    print(
        f"\n[integration] url={url} database={name} "
        f"version={admin.server_info().get('version')} "
        f"setName={hello.get('setName')} "
        f"isWritablePrimary={hello.get('isWritablePrimary')}"
    )

    def close():
        set_test_database(None)
        client.close()
        admin.drop_database(name)  # only the throwaway database
        admin.close()

    return database, close


@pytest.fixture(params=["mock", pytest.param("real", marks=pytest.mark.integration)])
def db(request, run):
    if request.param == "mock":
        mongomock_motor = pytest.importorskip(
            "mongomock_motor", reason="mongomock-motor not installed"
        )
        database = mongomock_motor.AsyncMongoMockClient().get_database(
            f"finrag_test_{uuid.uuid4().hex}"
        )
        set_test_database(database)
        try:
            yield database
        finally:
            set_test_database(None)
        return
    database, close = _open_real(run)
    try:
        yield database
    finally:
        close()


@pytest.fixture()
def real_db(run):
    """Real server only (for behaviour mongomock cannot reproduce)."""
    database, close = _open_real(run)
    try:
        yield database
    finally:
        close()


@pytest.fixture()
def ctx_a() -> TenantContext:
    return TenantContext(tenant_id="tenant-AAA", user_id="user-a1")


@pytest.fixture()
def ctx_b() -> TenantContext:
    return TenantContext(tenant_id="tenant-BBB", user_id="user-b1")


@pytest.fixture()
def ctx_root() -> TenantContext:
    return TenantContext(tenant_id="platform", user_id="root-1", is_super_admin=True)


def audit_entries(run, database):
    async def _all():
        return [doc async for doc in database[C.AUDIT_LOG].find({})]

    return run(_all())


# ---------------------------------------------------------------------------
# Decimal128 through the repository
# ---------------------------------------------------------------------------


def test_decimal_round_trips_exactly_and_is_stored_as_decimal128(db, run, ctx_a):
    repo = LedgerRepository()
    value = Decimal("123.456789012345678901")
    doc = run(repo.insert_one(ctx_a, {"company": "ACME", "metric": "revenue", "value": value}))
    assert doc["value"] == value and isinstance(doc["value"], Decimal)

    fetched = run(repo.get_by_id(ctx_a, doc["id"]))
    assert fetched["value"] == value
    assert str(fetched["value"]) == "123.456789012345678901"

    raw = run_db(run, lambda: db[C.STRUCTURED_FINANCIALS].find_one({"id": doc["id"]}))
    assert isinstance(raw["value"], Decimal128)


def test_numeric_string_is_accepted_for_a_decimal_field(db, run, ctx_a):
    repo = LedgerRepository()
    doc = run(repo.insert_one(ctx_a, {"company": "ACME", "value": "1500.25"}))
    assert run(repo.get_by_id(ctx_a, doc["id"]))["value"] == Decimal("1500.25")


def test_float_in_a_decimal_field_is_rejected_and_nothing_is_stored(db, run, ctx_a):
    repo = LedgerRepository()
    with pytest.raises(DecimalFieldError, match="float"):
        run(repo.insert_one(ctx_a, {"company": "ACME", "value": 120.0}))
    assert run(repo.count(ctx_a)) == 0


def test_insert_many_rejects_the_whole_batch_when_one_value_is_a_float(db, run, ctx_a):
    repo = LedgerRepository()
    with pytest.raises(DecimalFieldError, match="float"):
        run(repo.insert_many(ctx_a, [{"value": "1"}, {"value": 2.5}]))
    assert run(repo.count(ctx_a)) == 0


def test_decimal_update_round_trips_and_a_float_update_is_rejected(db, run, ctx_a):
    repo = LedgerRepository()
    doc = run(repo.insert_one(ctx_a, {"company": "ACME", "value": Decimal("1.00")}))
    run(repo.update_by_id(ctx_a, doc["id"], {"value": Decimal("2.50")}))
    assert run(repo.get_by_id(ctx_a, doc["id"]))["value"] == Decimal("2.50")

    with pytest.raises(DecimalFieldError, match="float"):
        run(repo.update_by_id(ctx_a, doc["id"], {"value": 3.5}))
    assert run(repo.get_by_id(ctx_a, doc["id"]))["value"] == Decimal("2.50")


def test_decimal_equality_filter_matches_and_a_float_filter_is_rejected(db, run, ctx_a, ctx_b):
    repo = LedgerRepository()
    run(repo.insert_one(ctx_a, {"company": "ACME", "value": "10.5"}))
    run(repo.insert_one(ctx_a, {"company": "ACME", "value": "20.5"}))
    run(repo.insert_one(ctx_b, {"company": "ACME", "value": "10.5"}))

    rows = run(repo.find_many(ctx_a, {"value": Decimal("10.5")}))
    assert [r["value"] for r in rows] == [Decimal("10.5")]  # tenant B's row stays hidden

    with pytest.raises(DecimalFieldError, match="float"):
        run(repo.find_many(ctx_a, {"value": 10.5}))


def test_decimal_range_queries_are_exact_on_a_real_server(real_db, run, ctx_a):
    repo = LedgerRepository()
    low = Decimal("1.000000000000000001")
    high = Decimal("1.000000000000000002")
    run(repo.insert_one(ctx_a, {"company": "ACME", "value": low}))
    run(repo.insert_one(ctx_a, {"company": "ACME", "value": high}))

    above = run(repo.find_many(ctx_a, {"value": {"$gt": low}}))
    below = run(repo.find_many(ctx_a, {"value": {"$lt": high}}))
    assert [r["value"] for r in above] == [high]  # a double could not tell these apart
    assert [r["value"] for r in below] == [low]


# ---------------------------------------------------------------------------
# Per-tenant uniqueness
# ---------------------------------------------------------------------------


def test_duplicate_email_in_one_tenant_is_rejected_but_allowed_across_tenants(
    db, run, ctx_a, ctx_b
):
    errors = pytest.importorskip("pymongo.errors")
    # The index is created here, mirroring the (tenant_id, email) unique index
    # in the bootstrap plan; wiring the real bootstrap in is a separate step.
    run_db(
        run,
        lambda: db[C.USERS].create_index([("tenant_id", 1), ("email", 1)], unique=True),
    )
    users = UserRepository()

    run(users.insert_one(ctx_a, {"email": "dup@tenant.test"}))
    with pytest.raises(errors.DuplicateKeyError):
        run(users.insert_one(ctx_a, {"email": "dup@tenant.test"}))
    run(users.insert_one(ctx_b, {"email": "dup@tenant.test"}))  # other tenant: fine

    assert run(users.count(ctx_a)) == 1
    assert run(users.count(ctx_b)) == 1


# ---------------------------------------------------------------------------
# Aggregation through the repository
# ---------------------------------------------------------------------------


def test_aggregate_only_counts_the_callers_tenant(db, run, ctx_a, ctx_b):
    docs = DocumentRepository()
    run(docs.insert_many(ctx_a, [{"title": "a1"}, {"title": "a2"}]))
    run(docs.insert_many(ctx_b, [{"title": "b1"}]))

    rows = run(docs.aggregate(ctx_a, [{"$group": {"_id": "$tenant_id", "n": {"$sum": 1}}}]))
    assert rows == [{"_id": "tenant-AAA", "n": 2}]


@pytest.mark.parametrize(
    "pipeline, message",
    [
        ([{"$out": "stolen"}], "never allowed"),
        ([{"$merge": {"into": "stolen"}}], "never allowed"),
        (
            [{"$lookup": {"from": "users", "localField": "a", "foreignField": "b", "as": "u"}}],
            "pipeline form",
        ),
        ([{"$match": {"$where": "1"}}], "not allowed"),
        ([{"$unionWith": "users"}], "allowed foreign collections"),
    ],
)
def test_hostile_pipelines_are_refused_before_reaching_the_database(
    db, run, ctx_a, pipeline, message
):
    docs = DocumentRepository()
    run(docs.insert_one(ctx_a, {"title": "keep me"}))
    with pytest.raises(PipelineGuardError, match=message):
        run(docs.aggregate(ctx_a, pipeline))
    assert run_db(run, lambda: db[C.DOCUMENTS].count_documents({})) == 1


def test_super_admin_cannot_use_write_stages_and_leaves_no_audit_entry(db, run, ctx_root):
    with pytest.raises(PipelineGuardError, match="never allowed"):
        run(DocumentRepository().aggregate(ctx_root, [{"$out": "stolen"}]))
    assert audit_entries(run, db) == []


def test_allowed_lookup_cannot_join_another_tenants_chunks(real_db, run, ctx_a, ctx_b):
    docs = DocumentRepository()
    chunks = ChunkRepository()
    # Same document id in both tenants: an unscoped join would mix them.
    run(docs.insert_one(ctx_a, {"title": "A doc"}, doc_id="shared-doc-id"))
    run(docs.insert_one(ctx_b, {"title": "B doc"}, doc_id="shared-doc-id"))
    run(chunks.insert_one(ctx_a, {"document_id": "shared-doc-id", "text": "a-chunk"}))
    run(chunks.insert_one(ctx_b, {"document_id": "shared-doc-id", "text": "SECRET-B"}))

    pipeline = [
        {
            "$lookup": {
                "from": C.CHUNKS,
                "let": {"d": "$id"},
                "pipeline": [{"$match": {"$expr": {"$eq": ["$document_id", "$$d"]}}}],
                "as": "chunks",
            }
        }
    ]
    rows = run(docs.aggregate(ctx_a, pipeline, allow_foreign_collections=[C.CHUNKS]))
    assert len(rows) == 1
    assert [c["text"] for c in rows[0]["chunks"]] == ["a-chunk"]


# ---------------------------------------------------------------------------
# Super-admin audit: before the operation, and fail closed
# ---------------------------------------------------------------------------


async def _consume(iterator):
    return [item async for item in iterator]


OPS = {
    "find_one": lambda r, c: r.find_one(c, {}),
    "find_many": lambda r, c: r.find_many(c, {}),
    "count": lambda r, c: r.count(c, {}),
    "exists": lambda r, c: r.exists(c, {}),
    "stream": lambda r, c: _consume(r.stream(c, {})),
    "insert_one": lambda r, c: r.insert_one(c, {"title": "new"}),
    "insert_many": lambda r, c: r.insert_many(c, [{"title": "new"}]),
    "update_one": lambda r, c: r.update_one(c, {}, {"title": "changed"}),
    "update_many": lambda r, c: r.update_many(c, {}, {"title": "changed"}),
    "delete_one": lambda r, c: r.delete_one(c, {}),
    "delete_many": lambda r, c: r.delete_many(c, {}),
    "aggregate": lambda r, c: r.aggregate(c, [{"$limit": 1}]),
}
# `exists` is built on find_one, so it is recorded under that name.
RECORDED_AS = {"exists": "find_one"}


@pytest.mark.parametrize("op", sorted(OPS))
def test_every_super_admin_operation_writes_exactly_one_audit_entry(
    db, run, ctx_a, ctx_root, op
):
    docs = DocumentRepository()
    run(docs.insert_one(ctx_a, {"title": "x"}))
    assert audit_entries(run, db) == []  # normal tenant access is not audited

    run(OPS[op](docs, ctx_root))

    entries = audit_entries(run, db)
    assert [e["action"] for e in entries] == [f"super_admin.{RECORDED_AS.get(op, op)}"]
    entry = entries[0]
    assert entry["resource"] == C.DOCUMENTS
    assert entry["user_id"] == "root-1"
    assert entry["tenant_id"] == "platform"


@pytest.mark.parametrize("op", sorted(OPS))
def test_super_admin_operation_is_refused_when_the_audit_write_fails(
    db, run, ctx_a, ctx_root, monkeypatch, op
):
    """Fail closed. For write operations this also proves the audit happens
    first: if the operation ran before the audit, the document would already
    be changed or deleted when the audit failure surfaced."""
    docs = DocumentRepository()
    run(docs.insert_one(ctx_a, {"title": "keep me"}))

    async def audit_store_down(entry):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(docs, "_write_audit_entry", audit_store_down)

    with pytest.raises(AuditWriteError) as info:
        run(OPS[op](docs, ctx_root))
    assert isinstance(info.value.__cause__, RuntimeError)

    rows = run(DocumentRepository().find_many(ctx_a, {}))
    assert [r["title"] for r in rows] == ["keep me"]  # untouched
    assert run_db(run, lambda: db[C.DOCUMENTS].count_documents({})) == 1  # nothing inserted
    assert audit_entries(run, db) == []


def test_audit_entry_stores_field_names_and_target_tenant_but_never_values(db, run, ctx_root):
    run(
        DocumentRepository().find_many(
            ctx_root, {"title": "secret-title", "tenant_id": "tenant-AAA"}
        )
    )
    (entry,) = audit_entries(run, db)
    assert "secret-title" not in str(entry)
    assert entry["detail"]["filter_keys"] == ["tenant_id", "title"]
    assert entry["detail"]["target_tenant"] == "tenant-AAA"


def test_normal_tenant_operations_never_write_audit_entries(db, run, ctx_a):
    docs = DocumentRepository()
    run(docs.insert_one(ctx_a, {"title": "x"}))
    run(docs.find_many(ctx_a, {}))
    run(docs.update_many(ctx_a, {}, {"title": "y"}))
    run(docs.delete_many(ctx_a, {}))
    assert audit_entries(run, db) == []

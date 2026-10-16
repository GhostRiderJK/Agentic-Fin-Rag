'''Bootstrap tests for the MongoDB index plan.'''

import pytest
from app.db.mongodb import collections as C
from app.db.mongodb.bootstrap import ensure_collections, ensure_indexes
from app.db.mongodb.repositories import (
    UserRepository,
    DocumentRepository,
)
from tests.mongo_support import _open_real, run_db


@pytest.fixture()
def run():
    """Drive coroutines on one private loop for the whole test."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        yield loop.run_until_complete
    finally:
        loop.close()


@pytest.fixture()
def real_db(run):
    """Real server only (for behaviour mongomock cannot reproduce)."""
    database, close = _open_real(run)
    try:
        yield database
    finally:
        close()


@pytest.fixture()
def bootstrapped_db(real_db, run):
    """Real database with the collections and planned indexes created."""
    run(ensure_collections())
    run(ensure_indexes())
    return real_db


def test_plan_leads_every_tenant_owned_index_with_tenant_id(run):
    '''For every collection in TENANT_OWNED, the first key of every IndexModel in C.index_plan() (use IndexModel.document["key"]) is "tenant_id", EXCEPT the TTL index named ttl_job_finished.'''
    plan = C.index_plan()
    for coll in C.TENANT_OWNED:
        for idx in plan[coll]:
            if idx.document.get("name") == "ttl_job_finished":
                continue
            # The key is a dict like {'tenant_id': 1, 'id': 1}; we need the first key.
            first_key = next(iter(idx.document["key"].keys()))
            assert first_key == "tenant_id", f"Collection {coll}, index {idx.document['name']} does not start with tenant_id"


def test_bootstrap_creates_every_planned_index_and_is_idempotent(real_db, run):
    '''Run both ensure_* twice; for each collection, index_information() names equal the planned names plus "_id_."'''
    # Run twice to test idempotency
    run(ensure_collections())
    run(ensure_indexes())
    run(ensure_collections())
    run(ensure_indexes())

    for coll in C.ALL_COLLECTIONS:
        info = run_db(run, lambda: real_db[coll].index_information())
        planned_names = {idx.document["name"] for idx in C.index_plan()[coll]}
        # The default _id_ index is always present.
        assert set(info.keys()) == planned_names | {"_id_"}


def test_ttl_index_expires_finished_jobs_after_30_days(bootstrapped_db, run):
    '''index_information()["ttl_job_finished"]["expireAfterSeconds"] == 2592000.'''
    info = run_db(run, lambda: bootstrapped_db[C.JOBS].index_information())
    assert info["ttl_job_finished"]["expireAfterSeconds"] == 2592000


def test_duplicate_email_rejected_within_a_tenant_but_allowed_across_tenants(bootstrapped_db, run):
    '''through UserRepository with the real plan; expect pymongo.errors.DuplicateKeyError within one tenant.'''
    errors = pytest.importorskip("pymongo.errors")
    from app.db.mongodb.tenant_context import TenantContext

    ctx_a = TenantContext(tenant_id="tenant-A", user_id="user-a")
    ctx_b = TenantContext(tenant_id="tenant-B", user_id="user-b")
    users = UserRepository()

    run(users.insert_one(ctx_a, {"email": "test@example.com"}))
    # Try to insert the same email in tenant A -> should fail
    with pytest.raises(errors.DuplicateKeyError):
        run(users.insert_one(ctx_a, {"email": "test@example.com"}))
    # The same email in tenant B should succeed
    run(users.insert_one(ctx_b, {"email": "test@example.com"}))


def test_two_users_without_api_key_in_one_tenant_are_allowed(bootstrapped_db, run):
    '''insert two users with different emails and no api_key in the same tenant; both must succeed.'''
    from app.db.mongodb.tenant_context import TenantContext

    ctx = TenantContext(tenant_id="tenant-X", user_id="user-x")
    users = UserRepository()

    run(users.insert_one(ctx, {"email": "user1@example.com"}))
    run(users.insert_one(ctx, {"email": "user2@example.com"}))

    # Both should exist
    assert run(users.get_by_email(ctx, "user1@example.com")) is not None
    assert run(users.get_by_email(ctx, "user2@example.com")) is not None


def test_duplicate_api_key_rejected_within_a_tenant(bootstrapped_db, run):
    '''two users with the same api_key string in one tenant raise DuplicateKeyError.'''
    errors = pytest.importorskip("pymongo.errors")
    from app.db.mongodb.tenant_context import TenantContext

    ctx = TenantContext(tenant_id="tenant-Y", user_id="user-y")
    users = UserRepository()

    run(users.insert_one(ctx, {"email": "user1@example.com", "api_key": "key123"}))
    with pytest.raises(errors.DuplicateKeyError):
        run(users.insert_one(ctx, {"email": "user2@example.com", "api_key": "key123"}))
    # Different email, same api_key in same tenant -> duplicate key error


def test_same_document_id_allowed_across_tenants_but_not_within_one(bootstrapped_db, run):
    '''DocumentRepository.insert_one(ctx, doc, doc_id="x") in tenant A and in tenant B both succeed;
    a second insert with the same id in tenant A raises DuplicateKeyError.'''
    errors = pytest.importorskip("pymongo.errors")
    from app.db.mongodb.tenant_context import TenantContext

    ctx_a = TenantContext(tenant_id="tenant-A", user_id="user-a")
    ctx_b = TenantContext(tenant_id="tenant-B", user_id="user-b")
    docs = DocumentRepository()

    # Insert doc with id "x" in tenant A
    run(docs.insert_one(ctx_a, {"title": "Doc A"}, doc_id="x"))
    # Insert doc with id "x" in tenant B -> should succeed
    run(docs.insert_one(ctx_b, {"title": "Doc B"}, doc_id="x"))
    # Try to insert another doc with id "x" in tenant A -> should fail
    with pytest.raises(errors.DuplicateKeyError):
        run(docs.insert_one(ctx_a, {"title": "Doc A2"}, doc_id="x"))
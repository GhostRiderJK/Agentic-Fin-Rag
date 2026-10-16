"""Shared helpers for MongoDB tests."""

import asyncio
import os
import uuid
from decimal import Decimal

import pytest

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

DEFAULT_URL = "mongodb://127.0.0.1:27018/?directConnection=true"


_PROBES: dict = {}


def mongo_required() -> bool:
    """Whether an unreachable test MongoDB must fail the run.

    On by default. Every test in the isolation suite that cannot reach a real
    server otherwise reports as *skipped*, and a green "N passed, M skipped"
    line reads like coverage that never happened. Opt out with
    ``REQUIRE_MONGO=0`` or ``--allow-mongo-skip``.
    """
    raw = os.environ.get("REQUIRE_MONGO", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enforce_reachable(url: str, reason: str | None) -> None:
    """Skip, or fail when ``REQUIRE_MONGO`` is on, if *reason* says the server is down."""
    if not reason:
        return
    detail = f"MongoDB not reachable at {url}: {reason}"
    if mongo_required():
        pytest.fail(
            f"{detail}\n\n"
            "REQUIRE_MONGO is enabled (the default), so this is a failure rather "
            "than a skip. Start the throwaway test instance:\n"
            "    docker compose -f infra/docker-compose.yml up -d mongo-test\n"
            "Set REQUIRE_MONGO=0 (or pass --allow-mongo-skip) to skip instead.",
            pytrace=False,
        )
    pytest.skip(detail)


def probe_unreachable(url: str) -> str | None:
    """Ping *url* and return the failure reason, or None when it answers."""
    import pymongo

    return _unreachable_reason(pymongo, url)


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

# ---------------------------------------------------------------------------
# Shared fixtures
#
# Test modules import these (`from mongo_support import db, run, ...`) rather
# than redefining them, so the mock/real policy, the loop handling and the
# REQUIRE_MONGO enforcement live in exactly one place.
# ---------------------------------------------------------------------------


@pytest.fixture()
def run():
    """Drive coroutines on one private loop for the whole test."""
    loop = asyncio.new_event_loop()
    try:
        yield loop.run_until_complete
    finally:
        loop.close()


@pytest.fixture(params=["mock", pytest.param("real", marks=pytest.mark.integration)])
def db(request, run):
    """A throwaway database on mongomock-motor, then on the real server."""
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

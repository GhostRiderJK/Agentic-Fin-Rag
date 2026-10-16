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
    if reason:
        pytest.skip(f"MongoDB not reachable at {url}: {reason}")
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
"""The single Mongo IO loop: who may call it, and what it guarantees.

These tests pin the properties the synchronous data path relies on:

* operations run on one private loop, never the caller's;
* the same loop serves many threads at once;
* calling from the IO thread itself is refused instead of deadlocking;
* a repository driven through :func:`run_sync` behaves exactly as when a test
  awaits it directly.
"""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

from app.db.mongodb import runtime
from app.db.mongodb.client import set_test_database
from app.db.mongodb.repositories import DocumentRepository
from app.db.mongodb.tenant_context import TenantContext

if not os.path.exists(os.path.join(os.path.dirname(__file__), "__init__.py")):
    from mongo_support import ctx_a, db, run  # noqa: F401
else:
    from tests.mongo_support import ctx_a, db, run  # noqa: F401


async def _where_am_i() -> dict:
    return {
        "loop": asyncio.get_running_loop(),
        "thread": threading.current_thread().name,
    }


def teardown_function(_):
    set_test_database(None)
    runtime.shutdown()


class TestTheIOLoop:
    def test_work_runs_on_its_own_loop_not_the_callers(self):
        where = runtime.run_sync(_where_am_i())
        assert where["thread"] == "mongo-io"
        assert where["loop"] is runtime.io_loop()
        # The calling thread has no loop at all; that is the point.
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()

    def test_the_loop_survives_across_calls(self):
        first = runtime.run_sync(_where_am_i())["loop"]
        second = runtime.run_sync(_where_am_i())["loop"]
        assert first is second

    def test_many_threads_share_one_loop(self):
        results: list = []
        errors: list = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            try:
                barrier.wait()
                results.append(runtime.run_sync(_where_am_i())["loop"])
            except Exception as exc:  # pragma: no cover - asserts below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 4
        assert set(map(id, results)) == {id(runtime.io_loop())}

    def test_the_io_thread_is_refused_rather_than_deadlocked(self):
        async def reentrant() -> str:
            try:
                runtime.run_sync(_where_am_i())
            except RuntimeError as exc:
                return str(exc)
            return "no error"

        message = runtime.run_sync(reentrant())
        assert "deadlock" in message
        assert runtime.on_io_thread() is False  # back on the caller's thread


class TestRepositoriesThroughTheLoop:
    def test_a_repository_call_needs_no_event_loop_in_the_caller(
        self, db, ctx_a
    ):
        """`db` injects a throwaway database; the route layer then reaches it
        from a thread that has no loop of its own."""
        repo = DocumentRepository()
        inserted = runtime.run_sync(
            repo.insert_one(ctx_a, {"title": "runtime doc", "workspace_id": "ws-1"})
        )
        assert inserted["tenant_id"] == ctx_a.tenant_id

        found = runtime.run_sync(
            repo.list_for_workspace(TenantContext(tenant_id="someone-else"), "ws-1")
        )
        assert found == []

        mine = runtime.run_sync(repo.list_for_workspace(ctx_a, "ws-1"))
        assert [d["title"] for d in mine] == ["runtime doc"]

    def test_a_write_from_one_thread_is_readable_from_another(self, db, ctx_a):
        """Route handlers, the job worker and the CLI are different threads;
        none of them may get a private view of the store."""
        repo = DocumentRepository()
        inserted = [None]
        barrier = threading.Barrier(2)

        def writer() -> None:
            inserted[0] = runtime.run_sync(
                repo.insert_one(ctx_a, {"title": "cross-thread", "workspace_id": "w"})
            )
            barrier.wait()

        thread = threading.Thread(target=writer)
        thread.start()
        barrier.wait()
        thread.join(timeout=10)

        found = runtime.run_sync(repo.get_by_id(ctx_a, inserted[0]["id"]))
        assert found["title"] == "cross-thread"

    def test_shutdown_resets_and_a_later_call_starts_a_new_loop(self):
        first = runtime.io_loop()
        runtime.shutdown()
        assert runtime.io_loop() is not first
        assert runtime.run_sync(_where_am_i())["thread"] == "mongo-io"

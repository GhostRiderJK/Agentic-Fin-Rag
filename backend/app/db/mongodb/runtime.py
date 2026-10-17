"""Driving the async Mongo layer from synchronous code.

Motor binds its client to the event loop the client was created on, and the
application's data path is synchronous: FastAPI runs ``def`` routes in a
threadpool, the in-proc job worker is a plain thread, and the CLI has no loop
at all. Giving each of those a client of its own means one client per loop and
a stale, unusable client the moment a loop dies (a ``TestClient`` builds a new
loop per instance).

So this module owns exactly one private event loop on a daemon thread and
funnels every Mongo operation through it. One process, one loop, one client,
and the client is only ever touched from the loop that created it.

Sync callers use :func:`run_sync`. Never call it from a coroutine: the
repositories are coroutines, so async code should await them — but only after
it has been proven to run on this loop, otherwise the shared client is being
used from a foreign loop, which is what the single-loop rule above exists to
prevent.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def io_loop() -> asyncio.AbstractEventLoop:
    """The loop every Mongo operation runs on, started on first use."""
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="mongo-io", daemon=True
            )
            thread.start()
            _loop, _thread = loop, thread
        return _loop


def on_io_thread() -> bool:
    """True when the calling thread is the Mongo IO thread."""
    thread = _thread
    return thread is not None and threading.get_ident() == thread.ident


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` on the Mongo IO loop from a synchronous caller.

    Blocks the calling thread until the result is ready. Repository methods
    are coroutines, so ``run_sync(repo.find_many(ctx, q))`` is the normal
    shape. Must not be called from the IO thread itself — that would block the
    loop it is waiting on.
    """
    loop = io_loop()
    if on_io_thread():
        coro.close()
        raise RuntimeError(
            "run_sync() would deadlock: it was called from the Mongo IO thread. "
            "Await the repository coroutine instead."
        )
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def shutdown() -> None:
    """Close the client and stop the loop (app shutdown, test teardown).

    Safe to call when nothing was ever started. A later :func:`run_sync`
    starts a fresh loop, so this is a reset rather than a terminal state.
    """
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        _loop = _thread = None
    if loop is None:
        return

    from app.db.mongodb.client import close_client

    try:
        asyncio.run_coroutine_threadsafe(close_client(), loop).result(timeout=5.0)
    except Exception:  # pragma: no cover - the loop is going away regardless
        pass
    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)
    loop.close()

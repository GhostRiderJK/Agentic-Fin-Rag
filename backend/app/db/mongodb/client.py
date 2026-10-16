"""Motor (async MongoDB) client lifecycle.

A single process-wide :class:`AsyncIOMotorClient` is reused across requests;
Motor pools connections internally, so creating a client per request would
exhaust file descriptors under load.

Test code can call :func:`set_test_database` to inject an in-memory or
container database without touching the settings-driven URL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

log = get_logger(__name__)

_client: Optional["AsyncIOMotorClient"] = None
_database: Optional["AsyncIOMotorDatabase"] = None
_test_database: Optional["AsyncIOMotorDatabase"] = None


def _import_motor() -> Any:
    """Import Motor lazily so importing this module never hard-fails when
    the driver is absent (e.g. during a partial install)."""
    try:
        import motor.motor_asyncio as motor_asyncio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Motor is required for the MongoDB data layer. "
            "Install it with: pip install motor pymongo"
        ) from exc
    return motor_asyncio


def get_client() -> "AsyncIOMotorClient":
    """Return the process-wide Motor client, creating it on first use."""
    global _client
    if _client is None:
        settings = get_settings()
        motor_asyncio = _import_motor()
        log.info(
            "connecting to MongoDB",
            extra={"operation": "mongodb_connect"},
        )
        _client = motor_asyncio.AsyncIOMotorClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
            uuidRepresentation="standard",
        )
    return _client


def get_database() -> "AsyncIOMotorDatabase":
    """Return the application database handle.

    Test injection (``set_test_database``) takes precedence so the suite can
    run against a throwaway database.
    """
    global _database
    if _test_database is not None:
        return _test_database
    if _database is None:
        settings = get_settings()
        _database = get_client()[settings.mongodb_db_name]
    return _database


def set_test_database(database: Optional["AsyncIOMotorDatabase"]) -> None:
    """Inject a database handle for tests. Pass ``None`` to clear."""
    global _test_database, _database
    _test_database = database
    _database = None


async def close_client() -> None:
    """Close the client and reset cached handles (app shutdown / tests)."""
    global _client, _database
    if _client is not None:
        _client.close()
    _client = None
    _database = None


async def ping() -> bool:
    """Return True when the server answers a ping; used by health checks."""
    try:
        db = get_database()
        await db.command("ping")
        return True
    except Exception as exc:  # pragma: no cover - depends on live server
        log.warning("mongodb ping failed", extra={"error": str(exc)})
        return False

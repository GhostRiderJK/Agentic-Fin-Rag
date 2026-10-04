"""SQLAlchemy database engine, session, and declarative base."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        settings.ensure_dirs()
        url = settings.database_url
        if url.startswith("sqlite"):
            # busy_timeout: writers block up to 60s waiting for the lock
            # instead of immediately raising "database is locked" when the
            # background ingest worker is mid-transaction.
            # WAL mode: lets readers proceed alongside a single writer,
            # dramatically reducing contention during bulk ingest.
            connect_args = {
                "check_same_thread": False,
                "timeout": 60,
                "isolation_level": None,  # autocommit; we manage txns ourselves
            }
            _engine = create_engine(
                url, connect_args=connect_args, future=True, pool_pre_ping=True
            )
            # Enable WAL + sane pragmas on every fresh connection.
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA busy_timeout=60000")  # 60s
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            _engine = create_engine(url, future=True, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def get_sessionmaker() -> sessionmaker:
    if _SessionLocal is None:
        get_engine()
    return _SessionLocal  # type: ignore[return-value]


def reset_engine() -> None:
    """Test helper to drop engine/session caches."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_db() -> Generator[Session, None, None]:
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_scope() -> Generator[Session, None, None]:
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (dev/seeding path). Production uses Alembic."""
    # Import models so they register on Base.metadata
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())

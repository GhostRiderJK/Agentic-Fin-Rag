"""Create collections and indexes for FinRAG 2.0.

Run once per environment:

    python -m app.db.mongodb.bootstrap

The index plan in :mod:`app.db.mongodb.collections` leads every tenant-owned
compound index with ``tenant_id`` so scoped queries use the index prefix.
"""

from __future__ import annotations

import asyncio
from typing import List

from app.core.logging import get_logger
from app.db.mongodb import collections as C
from app.db.mongodb.client import close_client, get_database

log = get_logger(__name__)


async def ensure_collections() -> List[str]:
    """Create any missing collections (idempotent)."""
    db = get_database()
    existing = set(await db.list_collection_names())
    created: List[str] = []
    for name in C.ALL_COLLECTIONS:
        if name not in existing:
            await db.create_collection(name)
            created.append(name)
    return created


async def ensure_indexes() -> List[str]:
    """Create every index in the plan (idempotent)."""
    db = get_database()
    created: List[str] = []
    for name, models in C.index_plan().items():
        if not models:
            continue
        names = await db[name].create_indexes(models)
        created.extend(names)
    return created


async def bootstrap() -> None:
    created_collections = await ensure_collections()
    created_indexes = await ensure_indexes()
    log.info(
        "mongodb bootstrap complete",
        extra={
            "collections_created": len(created_collections),
            "indexes_created": len(created_indexes),
        },
    )


async def _main() -> None:
    try:
        await bootstrap()
    finally:
        await close_client()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())

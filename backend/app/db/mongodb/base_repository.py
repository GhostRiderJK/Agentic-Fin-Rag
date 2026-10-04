"""Tenant-isolated base repository.

Design rules (non-negotiable for FinRAG 2.0):

1. Every public method takes a :class:`TenantContext` as its first argument.
   There is no default, so a caller cannot forget it and silently query the
   whole collection.
2. The tenant filter is applied in :meth:`_scoped`, which merges
   ``{"tenant_id": ctx.tenant_id}`` into whatever filter the caller supplied.
   A caller *cannot* widen the scope by passing its own ``tenant_id`` — the
   context wins.
3. Inserts stamp ``tenant_id`` from the context, overwriting any value the
   caller tried to set.
4. Updates and deletes are scoped the same way, so a cross-tenant id simply
   matches zero documents and reports "not found" rather than succeeding.

The only sanctioned bypass is ``ctx.is_super_admin``, which must be paired
with an explicit audit-log entry by the caller.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional, Sequence

from app.core.logging import get_logger
from app.db.mongodb.client import get_database
from app.db.mongodb.tenant_context import TenantContext, TenantContextError

log = get_logger(__name__)

# Fields written by the repository itself; callers may not set them.
_MANAGED_FIELDS = ("tenant_id",)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    """Opaque document id (32 hex chars), matching the legacy id shape."""
    return uuid.uuid4().hex


class BaseRepository:
    """CRUD scoped to a single tenant.

    Subclasses set :attr:`collection_name` and may add domain queries, but
    must route every read/write through the ``_scoped*`` helpers.
    """

    collection_name: str = ""

    def __init__(self, collection_name: Optional[str] = None) -> None:
        if collection_name:
            self.collection_name = collection_name
        if not self.collection_name:
            raise ValueError(f"{type(self).__name__} must define collection_name")

    # -- collection access --------------------------------------------------

    @property
    def collection(self):
        """The Motor collection handle for this repository."""
        return get_database()[self.collection_name]

    # -- tenant scoping -----------------------------------------------------

    def _require_ctx(self, ctx: Optional[TenantContext]) -> TenantContext:
        if not isinstance(ctx, TenantContext):
            raise TenantContextError(
                "A TenantContext is required for every repository call",
                repository=type(self).__name__,
            )
        return ctx

    def _scoped(self, ctx: TenantContext, query: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Return ``query`` with the tenant filter enforced.

        For normal callers, the context's tenant_id is written after the
        caller's query so a caller-supplied ``tenant_id`` is overridden.
        Super admins may omit tenant_id to span tenants, but an explicit
        tenant_id in the query is still honoured.
        """
        scoped: Dict[str, Any] = dict(query or {})
        if ctx.is_super_admin:
            # Super admins control the tenant filter via the query.
            # If they omit tenant_id, the query matches all tenants.
            return scoped
        scoped["tenant_id"] = ctx.tenant_id
        return scoped

    def _stamp(self, ctx: TenantContext, doc: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a copy of ``doc`` with tenant ownership enforced."""
        stamped = dict(doc)
        stamped["tenant_id"] = ctx.tenant_id
        return stamped

    # -- create -------------------------------------------------------------

    async def insert_one(
        self,
        ctx: TenantContext,
        doc: Mapping[str, Any],
        *,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert one document, stamping tenant ownership."""
        self._require_ctx(ctx)
        payload = self._stamp(ctx, doc)
        payload.setdefault("id", doc_id or new_id())
        now = utcnow()
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        await self.collection.insert_one(payload)
        log.debug(
            "document inserted",
            extra={"collection": self.collection_name, "tenant_id": ctx.tenant_id},
        )
        return payload

    async def insert_many(
        self,
        ctx: TenantContext,
        docs: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Insert several documents under one tenant."""
        self._require_ctx(ctx)
        if not docs:
            return []
        now = utcnow()
        payloads: List[Dict[str, Any]] = []
        for doc in docs:
            payload = self._stamp(ctx, doc)
            payload.setdefault("id", new_id())
            payload.setdefault("created_at", now)
            payload.setdefault("updated_at", now)
            payloads.append(payload)
        await self.collection.insert_many(payloads)
        return payloads

    # -- read ---------------------------------------------------------------

    async def find_one(
        self,
        ctx: TenantContext,
        query: Optional[Mapping[str, Any]] = None,
        *,
        projection: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find one document owned by the caller's tenant."""
        self._require_ctx(ctx)
        return await self.collection.find_one(
            self._scoped(ctx, query),
            projection=dict(projection) if projection else None,
        )

    async def get_by_id(
        self, ctx: TenantContext, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch by id — returns None for another tenant's id."""
        return await self.find_one(ctx, {"id": doc_id})

    async def find_many(
        self,
        ctx: TenantContext,
        query: Optional[Mapping[str, Any]] = None,
        *,
        sort: Optional[Sequence[tuple[str, int]]] = None,
        limit: int = 0,
        skip: int = 0,
        projection: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Find many documents owned by the caller's tenant."""
        self._require_ctx(ctx)
        cursor = self.collection.find(
            self._scoped(ctx, query),
            projection=dict(projection) if projection else None,
        )
        if sort:
            cursor = cursor.sort(list(sort))
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return [doc async for doc in cursor]

    def stream(
        self,
        ctx: TenantContext,
        query: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream documents for a tenant (large result sets)."""
        self._require_ctx(ctx)
        cursor = self.collection.find(self._scoped(ctx, query))
        return cursor.__aiter__()  # type: ignore[return-value]

    async def count(
        self, ctx: TenantContext, query: Optional[Mapping[str, Any]] = None
    ) -> int:
        """Count documents owned by the caller's tenant."""
        self._require_ctx(ctx)
        return await self.collection.count_documents(self._scoped(ctx, query))

    async def exists(
        self, ctx: TenantContext, query: Optional[Mapping[str, Any]] = None
    ) -> bool:
        """True when at least one matching document exists for the tenant."""
        found = await self.find_one(ctx, query, projection={"id": 1})
        return found is not None

    # -- update -------------------------------------------------------------

    async def update_one(
        self,
        ctx: TenantContext,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> int:
        """Update one tenant-owned document; returns modified count.

        ``update`` may be a plain field mapping (``{"status": "archived"}``)
        or a MongoDB operator document (``{"$set": {...}}``); plain mappings
        are wrapped in ``$set``.
        """
        self._require_ctx(ctx)
        update_doc = self._normalise_update(update, ctx)
        result = await self.collection.update_one(self._scoped(ctx, query), update_doc)
        return result.modified_count

    async def update_by_id(
        self, ctx: TenantContext, doc_id: str, update: Mapping[str, Any]
    ) -> int:
        return await self.update_one(ctx, {"id": doc_id}, update)

    async def update_many(
        self,
        ctx: TenantContext,
        query: Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> int:
        """Update many tenant-owned documents; returns modified count."""
        self._require_ctx(ctx)
        update_doc = self._normalise_update(update, ctx)
        result = await self.collection.update_many(self._scoped(ctx, query), update_doc)
        return result.modified_count

    def _normalise_update(
        self, update: Mapping[str, Any], ctx: TenantContext
    ) -> Dict[str, Any]:
        """Wrap a plain mapping in ``$set`` and forbid re-owning a document."""
        if not update:
            raise ValueError("update document must not be empty")
        if any(key.startswith("$") for key in update):
            doc = {k: dict(v) if isinstance(v, Mapping) else v for k, v in update.items()}
        else:
            doc = {"$set": dict(update)}
        # Never let an update reassign ownership.
        for field in _MANAGED_FIELDS:
            for operator, value in list(doc.items()):
                if isinstance(value, Mapping) and field in value:
                    raise TenantContextError(
                        "Refusing to change tenant ownership via update",
                        field=field,
                        repository=type(self).__name__,
                    )
        set_clause = doc.setdefault("$set", {})
        if isinstance(set_clause, Mapping):
            set_clause["updated_at"] = utcnow()
        return doc

    # -- delete -------------------------------------------------------------

    async def delete_one(
        self, ctx: TenantContext, query: Mapping[str, Any]
    ) -> int:
        """Delete one tenant-owned document; returns deleted count."""
        self._require_ctx(ctx)
        result = await self.collection.delete_one(self._scoped(ctx, query))
        return result.deleted_count

    async def delete_by_id(self, ctx: TenantContext, doc_id: str) -> int:
        return await self.delete_one(ctx, {"id": doc_id})

    async def delete_many(
        self, ctx: TenantContext, query: Mapping[str, Any]
    ) -> int:
        """Delete many tenant-owned documents; returns deleted count."""
        self._require_ctx(ctx)
        result = await self.collection.delete_many(self._scoped(ctx, query))
        return result.deleted_count

    # -- aggregation --------------------------------------------------------

    async def aggregate(
        self, ctx: TenantContext, pipeline: Iterable[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run an aggregation guarded by the tenant filter.

        The tenant ``$match`` is prepended so a caller's pipeline cannot read
        across tenants — later stages only ever see pre-filtered documents.
        """
        self._require_ctx(ctx)
        stages: List[Dict[str, Any]] = []
        if not ctx.is_super_admin:
            stages.append({"$match": {"tenant_id": ctx.tenant_id}})
        stages.extend(dict(stage) for stage in pipeline)
        return [doc async for doc in self.collection.aggregate(stages)]

    # -- index management ---------------------------------------------------

    async def ensure_indexes(self, indexes: Sequence[Any]) -> List[str]:
        """Create indexes; used by the bootstrap script."""
        return list(await self.collection.create_indexes(list(indexes)))

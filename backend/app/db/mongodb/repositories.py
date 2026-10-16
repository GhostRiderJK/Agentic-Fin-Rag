"""Concrete repositories for FinRAG 2.0.

Each class extends :class:`BaseRepository`, inheriting tenant isolation.
Only domain-specific queries live here; anything generic stays in the base.

``TenantRepository`` is the one exception: the ``tenants`` collection is the
root of tenancy and has no ``tenant_id`` of its own, so it extends
``_UnscopedRepository`` and is only reachable from platform-level code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pymongo import DESCENDING

from app.db.mongodb import collections as C
from app.db.mongodb.base_repository import BaseRepository, utcnow
from app.db.mongodb.tenant_context import TenantContext


class _UnscopedRepository:
    """Minimal repository for tenant-independent collections.

    Deliberately not derived from ``BaseRepository`` so the tenant-scoping
    helpers cannot be used accidentally on a collection that has no
    ``tenant_id``.
    """

    collection_name: str = ""

    def __init__(self, collection_name: Optional[str] = None) -> None:
        if collection_name:
            self.collection_name = collection_name

    @property
    def collection(self):
        from app.db.mongodb.client import get_database

        return get_database()[self.collection_name]

    async def insert_one(self, doc: Dict[str, Any], *, doc_id: Optional[str] = None) -> Dict[str, Any]:
        from app.db.mongodb.base_repository import new_id

        payload = dict(doc)
        payload.setdefault("id", doc_id or new_id())
        now = utcnow()
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        await self.collection.insert_one(payload)
        return payload

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return await self.collection.find_one(dict(query))

    async def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return await self.find_one({"id": doc_id})

    async def find_many(
        self,
        query: Optional[Dict[str, Any]] = None,
        *,
        sort: Optional[Sequence[tuple[str, int]]] = None,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        cursor = self.collection.find(dict(query or {}))
        if sort:
            cursor = cursor.sort(list(sort))
        if limit:
            cursor = cursor.limit(limit)
        return [doc async for doc in cursor]

    async def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        return await self.collection.count_documents(dict(query or {}))

    async def update_by_id(self, doc_id: str, update: Dict[str, Any]) -> int:
        doc = update if any(k.startswith("$") for k in update) else {"$set": dict(update)}
        doc.setdefault("$set", {})["updated_at"] = utcnow()
        result = await self.collection.update_one({"id": doc_id}, doc)
        return result.modified_count

    async def delete_by_id(self, doc_id: str) -> int:
        result = await self.collection.delete_one({"id": doc_id})
        return result.deleted_count


# -- platform-level -----------------------------------------------------------


class TenantRepository(_UnscopedRepository):
    """The tenants collection (root of tenancy; no tenant_id column)."""

    collection_name = C.TENANTS

    async def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return await self.find_one({"name": name})


# -- tenant-owned -------------------------------------------------------------


class UserRepository(BaseRepository):
    collection_name = C.USERS

    async def get_by_email(
        self, ctx: TenantContext, email: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a user by email *within the caller's tenant*."""
        if not email:
            return None
        return await self.find_one(ctx, {"email": email.strip().lower()})

    async def get_by_api_key(
        self, ctx: TenantContext, api_key: str
    ) -> Optional[Dict[str, Any]]:
        if not api_key:
            return None
        return await self.find_one(ctx, {"api_key": api_key})

    async def list_active(self, ctx: TenantContext) -> List[Dict[str, Any]]:
        return await self.find_many(
            ctx, {"is_active": True}, sort=[("created_at", DESCENDING)]
        )


class WorkspaceRepository(BaseRepository):
    collection_name = C.WORKSPACES

    async def list_for_tenant(self, ctx: TenantContext) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {}, sort=[("created_at", DESCENDING)])


class DocumentRepository(BaseRepository):
    collection_name = C.DOCUMENTS

    async def list_for_workspace(
        self,
        ctx: TenantContext,
        workspace_id: str,
        *,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            query["status"] = status
        return await self.find_many(ctx, query, sort=[("created_at", DESCENDING)])

    async def find_by_metadata(
        self,
        ctx: TenantContext,
        *,
        workspace_id: Optional[str] = None,
        company: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Metadata prefilter used by retrieval before the vector search."""
        query: Dict[str, Any] = {}
        if workspace_id:
            query["workspace_id"] = workspace_id
        if company:
            query["company"] = company
        if year is not None:
            query["year"] = year
        if quarter:
            query["quarter"] = quarter
        if doc_type:
            query["doc_type"] = doc_type
        return await self.find_many(ctx, query)

    async def set_status(
        self, ctx: TenantContext, document_id: str, status: str
    ) -> int:
        return await self.update_by_id(ctx, document_id, {"status": status})


class ChunkRepository(BaseRepository):
    collection_name = C.CHUNKS

    async def list_for_document(
        self, ctx: TenantContext, document_id: str
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {"document_id": document_id})

    async def list_children(
        self, ctx: TenantContext, parent_id: str
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {"parent_id": parent_id})

    async def get_many_by_ids(
        self, ctx: TenantContext, chunk_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Fetch chunks by id — ids from another tenant return nothing."""
        if not chunk_ids:
            return []
        return await self.find_many(ctx, {"id": {"$in": list(chunk_ids)}})


class StructuredFinancialRepository(BaseRepository):
    collection_name = C.STRUCTURED_FINANCIALS

    async def get_metric(
        self,
        ctx: TenantContext,
        *,
        company: str,
        metric: str,
        fiscal_year: Optional[int] = None,
        quarter: str = "",
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {
            "company": company,
            "metric": metric,
        }
        if fiscal_year is not None:
            query["fiscal_year"] = fiscal_year
        if quarter:
            query["quarter"] = quarter
        if workspace_id:
            query["workspace_id"] = workspace_id
        return await self.find_many(
            ctx,
            query,
            sort=[("fiscal_year", DESCENDING), ("quarter", DESCENDING)],
        )

    async def compare_metric(
        self,
        ctx: TenantContext,
        *,
        companies: Sequence[str],
        metric: str,
        fiscal_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {
            "company": {"$in": list(companies)},
            "metric": metric,
        }
        if fiscal_year is not None:
            query["fiscal_year"] = fiscal_year
        return await self.find_many(ctx, query, sort=[("company", 1)])

    async def list_metrics(self, ctx: TenantContext, company: str) -> List[str]:
        values = await self.collection.distinct(
            "metric", {"tenant_id": ctx.tenant_id, "company": company}
        )
        return sorted(values)


class ConversationRepository(BaseRepository):
    collection_name = C.CONVERSATIONS

    async def list_for_workspace(
        self, ctx: TenantContext, workspace_id: str
    ) -> List[Dict[str, Any]]:
        return await self.find_many(
            ctx, {"workspace_id": workspace_id}, sort=[("updated_at", DESCENDING)]
        )


class MessageRepository(BaseRepository):
    collection_name = C.MESSAGES

    async def list_for_conversation(
        self, ctx: TenantContext, conversation_id: str
    ) -> List[Dict[str, Any]]:
        return await self.find_many(
            ctx, {"conversation_id": conversation_id}, sort=[("created_at", 1)]
        )

    async def set_feedback(
        self, ctx: TenantContext, message_id: str, rating: int
    ) -> int:
        return await self.update_by_id(ctx, message_id, {"feedback": rating})


class JobRepository(BaseRepository):
    collection_name = C.JOBS

    async def list_recent(
        self, ctx: TenantContext, *, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {}, sort=[("created_at", DESCENDING)], limit=limit)


class EvalRunRepository(BaseRepository):
    collection_name = C.EVAL_RUNS

    async def list_recent(
        self, ctx: TenantContext, *, limit: int = 20
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {}, sort=[("created_at", DESCENDING)], limit=limit)


class EvalItemRepository(BaseRepository):
    collection_name = C.EVAL_ITEMS

    async def list_for_run(
        self, ctx: TenantContext, run_id: str
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {"run_id": run_id})


class AuditLogRepository(BaseRepository):
    collection_name = C.AUDIT_LOG

    async def record(
        self,
        ctx: TenantContext,
        *,
        action: str,
        resource: str,
        resource_id: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append an audit entry (used for the super-admin bypass, among
        other privileged operations)."""
        return await self.insert_one(
            ctx,
            {
                "action": action,
                "resource": resource,
                "resource_id": resource_id,
                "user_id": ctx.user_id,
                "detail": detail or {},
            },
        )

    async def list_recent(
        self, ctx: TenantContext, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {}, sort=[("created_at", DESCENDING)], limit=limit)


class FeedbackRepository(BaseRepository):
    collection_name = C.FEEDBACK

    async def list_for_message(
        self, ctx: TenantContext, message_id: str
    ) -> List[Dict[str, Any]]:
        return await self.find_many(ctx, {"message_id": message_id})

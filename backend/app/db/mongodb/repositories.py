"""Concrete repositories for FinRAG 2.0.

Each class extends :class:`BaseRepository`, inheriting tenant isolation.
Only domain-specific queries live here; anything generic stays in the base.

``TenantRepository`` is the one exception: the ``tenants`` collection is the
root of tenancy and has no ``tenant_id`` of its own, so it extends
``_UnscopedRepository`` and is only reachable from platform-level code.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

from pymongo import DESCENDING

from app.db.mongodb import collections as C
from app.db.mongodb.base_repository import BaseRepository, utcnow
from app.db.mongodb.tenant_context import TenantContext


def _as_list(value: Union[Any, Sequence[Any]]) -> list:
    """Mongo ``$in`` operand for a filter that accepts one value or many."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _year_forms(value: Union[Any, Sequence[Any]]) -> list:
    """Every BSON shape a fiscal year is stored in, for one or many years.

    Documents written by the current upload route hold ``2024`` (a number) and
    legacy rows in the same corpus hold ``"2024"`` (a string). Mongo compares
    strictly across types, so an int-only filter loses the string rows and a
    string-only filter the numeric ones, silently and with no error. Matching
    both shapes is read-only: it costs one extra ``$in`` term.
    """
    forms: list = []
    for item in _as_list(value):
        for candidate in (item, _other_year_form(item)):
            if candidate is not None and candidate not in forms:
                forms.append(candidate)
    return forms


def _other_year_form(value: Any):
    """The other of the two year representations, or None if not a year."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text.isdigit() else None
    return None


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
        company_like: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Documents in one workspace, newest first.

        ``company_like`` is a case-insensitive substring, matching what the
        documents list route has always done. It is escaped because the value
        comes from a query parameter and is placed inside a regex.
        """
        query: Dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            query["status"] = status
        if company_like:
            query["company"] = {
                "$regex": re.escape(company_like),
                "$options": "i",
            }
        return await self.find_many(ctx, query, sort=[("created_at", DESCENDING)])

    async def find_matching(
        self,
        ctx: TenantContext,
        *,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        company: Union[str, Sequence[str], None] = None,
        year: Union[int, Sequence[int], None] = None,
        quarter: Union[str, Sequence[str], None] = None,
        doc_type: Union[str, Sequence[str], None] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieval's metadata prefilter: any subset of workspace, status and
        company/year/quarter/doc_type. Each filter accepts one value or many.

        ``workspace_id`` and ``status`` may be omitted for the tenant-wide
        pre-check retrieval does before relaxing a company filter; the tenant
        filter itself is never optional.
        """
        query: Dict[str, Any] = {}
        if workspace_id:
            query["workspace_id"] = workspace_id
        if status:
            query["status"] = status
        for field_name, value in (
            ("company", company),
            ("year", year),
            ("quarter", quarter),
            ("doc_type", doc_type),
        ):
            if value is None or value == []:
                continue
            if field_name == "year":
                query[field_name] = {"$in": _year_forms(value)}
            else:
                query[field_name] = {"$in": _as_list(value)}
        return await self.find_many(ctx, query)

    async def distinct_years(
        self, ctx: TenantContext, *, workspace_id: str, status: str
    ) -> List[int]:
        """Fiscal years present in a workspace.

        Retrieval uses this so a question about a year mentioned *inside* a
        document (a 2026 outlook in a 2025 report) does not filter that
        document out.
        """
        values = await self.collection.distinct(
            "year",
            {
                "tenant_id": ctx.tenant_id,
                "workspace_id": workspace_id,
                "status": status,
                "year": {"$ne": None},
            },
        )
        # Legacy rows hold the year as text. Returned as stored, ``sorted``
        # raises on a workspace that mixes int and str, and retrieval's
        # ``plan_year in available_years`` test can never match a text year,
        # which turns the filter off without saying so.
        numbers = {int(text) for text in (str(v).strip() for v in values) if text.isdigit()}
        return sorted(numbers)

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
            query["year"] = {"$in": _year_forms(year)}
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

    async def find_searchable(
        self, ctx: TenantContext, document_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Child chunks of the given documents — the BM25 candidate corpus.

        Only children are embedded; parents are fetched on demand during parent
        expansion, so they are deliberately excluded here.
        """
        if not document_ids:
            return []
        return await self.find_many(
            ctx, {"document_id": {"$in": list(document_ids)}, "is_parent": False}
        )

    async def delete_for_document(self, ctx: TenantContext, document_id: str) -> int:
        """Remove a document's chunks (what the SQL cascade did on delete)."""
        return await self.delete_many(ctx, {"document_id": document_id})

    async def get_many_by_ids(
        self, ctx: TenantContext, chunk_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Fetch chunks by id — ids from another tenant return nothing."""
        if not chunk_ids:
            return []
        return await self.find_many(ctx, {"id": {"$in": list(chunk_ids)}})


class StructuredFinancialRepository(BaseRepository):
    collection_name = C.STRUCTURED_FINANCIALS
    # `value` holds money or a ratio depending on the metric; `confidence` is a
    # 0..1 ratio. Both are exact decimals — a float has lost precision before
    # it ever reaches the database.
    decimal_fields = frozenset({"value", "confidence"})

    async def find_by_company(
        self,
        ctx: TenantContext,
        *,
        workspace_id: str,
        company: str,
        metric: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        quarter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rows whose company name and ``company`` contain one another.

        The tool-calling route passes whatever the LLM typed, and neither side
        is guaranteed to be the longer one: a row stored as "Apple" must answer
        "Apple Inc.", and a row stored as "Apple Inc." must answer "Apple". The
        first branch is a regex (escaped, because the term is user-supplied);
        the second resolves stored names that sit *inside* the typed term.

        Rows with no company never match the second branch — the empty string
        is inside every term, so including it would flood every lookup.
        Neither direction resolves an acronym: "PIF" does not find "Public
        Investment Fund".
        """
        if not company.strip():
            # An empty pattern matches every document in the workspace. A lookup
            # that has lost its term returns nothing instead.
            return []
        scoped: Dict[str, Any] = {"workspace_id": workspace_id}
        if metric:
            scoped["metric"] = metric
        if fiscal_year is not None:
            scoped["fiscal_year"] = {"$in": _year_forms(fiscal_year)}
        if quarter:
            scoped["quarter"] = quarter

        branches: List[Dict[str, Any]] = [
            {"company": {"$regex": re.escape(company), "$options": "i"}}]
        aliases = await self._names_inside(ctx, scoped, company.lower())
        if aliases:
            branches.append({"company": {"$in": aliases}})
        query: Dict[str, Any] = {**scoped, "$or": branches}
        return await self.find_many(
            ctx,
            query,
            sort=[("fiscal_year", DESCENDING), ("quarter", DESCENDING)],
        )

    async def _names_inside(
        self, ctx: TenantContext, scoped: Dict[str, Any], needle: str
    ) -> List[str]:
        """Stored company names this scope holds that *needle* contains.

        ``$expr`` string operators do not exist on the in-memory server these
        queries are also tested against, so the reverse containment is decided
        here over the scope's distinct names instead of per row. Names equal to
        the term are left to the regex branch.
        """
        if not needle:
            return []
        self._require_ctx(ctx)
        values = await self.collection.distinct(
            "company",
            {**self._prepare_query(ctx, scoped), "company": {"$nin": [None, ""]}},
        )
        return [v for v in values
                if isinstance(v, str) and v.lower() != needle and v.lower() in needle]

    async def detach_document(self, ctx: TenantContext, document_id: str) -> int:
        """Drop the provenance pointer when a document is deleted.

        The extracted numbers stay: they were published from the filing, and
        losing the filing must not silently delete the figures an analyst may
        still be citing. This mirrors the SQL ``ON DELETE SET NULL``.
        """
        return await self.update_many(ctx, {"document_id": document_id},
                                      {"document_id": None})

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
            query["fiscal_year"] = {"$in": _year_forms(fiscal_year)}
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
            query["fiscal_year"] = {"$in": _year_forms(fiscal_year)}
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
    # A 0..1 ratio, not a float: progress is compared and reported to users.
    decimal_fields = frozenset({"progress"})

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

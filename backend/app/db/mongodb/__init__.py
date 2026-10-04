"""MongoDB data access layer for FinRAG 2.0.

This package replaces the SQLAlchemy/SQLite layer. All relational data
(tenants, users, workspaces, documents, chunks, structured financials,
conversations, messages, jobs, eval runs, feedback, audit log) lives in
MongoDB behind a repository layer that enforces tenant isolation.

Key invariant: **no repository method may run without a TenantContext**.
Every filter, update and delete is automatically scoped to the tenant in
that context, so a caller can never read or mutate another tenant's data.
"""

from app.db.mongodb.base_repository import BaseRepository
from app.db.mongodb.client import get_database, close_client
from app.db.mongodb.tenant_context import TenantContext, TenantContextError

__all__ = [
    "BaseRepository",
    "TenantContext",
    "TenantContextError",
    "get_database",
    "close_client",
]

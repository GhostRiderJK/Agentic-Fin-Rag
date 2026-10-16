"""Tenant context — the mandatory credential for every repository call.

Every tenant-owned document carries a ``tenant_id``. The repository layer
refuses to run any query without a :class:`TenantContext`, and injects the
context's ``tenant_id`` into every filter, update and delete. That makes
cross-tenant access a programming error that fails closed rather than a
missing WHERE clause that silently returns another tenant's rows.

``TenantContext`` is intentionally immutable: a caller cannot mutate the
tenant id mid-request to widen its own scope.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from app.core.errors import AppError


class TenantContextError(AppError):
    """Raised when a repository call has no usable tenant context."""

    status_code = 403
    code = "missing_tenant_context"


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Identifies the tenant (and optionally the user/workspace) on whose
    behalf a repository call is being made.

    Attributes:
        tenant_id: The owning tenant. Never empty for a valid context.
        user_id: The acting user, when the call is user-initiated.
        workspace_id: The active workspace, when the call is scoped to one.
        is_super_admin: Platform staff may cross tenants *explicitly* by
            setting this flag; it is the only sanctioned bypass and every
            such access should be audit-logged by the caller.
    """

    tenant_id: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    is_super_admin: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise TenantContextError(
                "TenantContext requires a non-empty tenant_id",
                reason="empty_tenant_id",
            )

    def with_workspace(self, workspace_id: Optional[str]) -> "TenantContext":
        """Return a copy scoped to ``workspace_id`` (same tenant)."""
        return replace(self, workspace_id=workspace_id)

    def with_user(self, user_id: Optional[str]) -> "TenantContext":
        """Return a copy acting as ``user_id`` (same tenant)."""
        return replace(self, user_id=user_id)

    @classmethod
    def system(cls, tenant_id: str) -> "TenantContext":
        """Context for background/system work on behalf of one tenant."""
        return cls(tenant_id=tenant_id)

"""API dependency helpers (workspace scoping + tenant context)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Path
from sqlalchemy.orm import Session

from app.core.security import CurrentUser, get_workspace
from app.db.base import get_db
from app.db.models import Workspace
from app.db.mongodb.tenant_context import TenantContext


def workspace_dep(
    workspace_id: Annotated[str, Path()],
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    return get_workspace(workspace_id, user=user, db=db)


def tenant_ctx_dep(user: CurrentUser) -> TenantContext:
    """Tenant context for the authenticated user (no workspace yet)."""
    return TenantContext(tenant_id=user.tenant_id, user_id=user.id)


TenantCtxDep = Annotated[TenantContext, Depends(tenant_ctx_dep)]


def workspace_ctx_dep(
    user: CurrentUser,
    ws: Annotated[Workspace, Depends(workspace_dep)],
) -> TenantContext:
    """Tenant context scoped to the workspace named in the path.

    The workspace is verified to belong to the user's tenant first, so a
    context built here can never point at another tenant's workspace.
    """
    return TenantContext(
        tenant_id=user.tenant_id, user_id=user.id, workspace_id=ws.id
    )


WorkspaceCtxDep = Annotated[TenantContext, Depends(workspace_ctx_dep)]

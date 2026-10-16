"""API dependency helpers (workspace scoping)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Path
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.security import CurrentUser, get_workspace
from app.db.base import get_db


def workspace_dep(
    workspace_id: Annotated[str, Path()],
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    return get_workspace(workspace_id, user=user, db=db)

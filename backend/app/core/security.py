"""Security: password hashing, JWT issuance, API-key + JWT auth dependencies,
role-based access control, and tenant/workspace scoping.

Multi-tenancy model:
  - A ``Tenant`` is the top isolation boundary (e.g. a firm).
  - ``Workspace`` is a per-tenant collection of documents.
  - Every user belongs to a tenant and has a role (admin|analyst|viewer).
  - All document/query APIs are scoped by workspace_id; we verify the
    workspace belongs to the user's tenant before any access.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, Header, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import (
    AuthError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from app.db.base import get_db
from app.db.models import Role, User, Workspace


#: bcrypt's own limit. The installed bcrypt 5.x raises past it instead of
#: truncating, so a longer secret is refused rather than quietly weakened.
_BCRYPT_MAX_SECRET_BYTES = 72
#: What passlib used, so hashes already stored in a database still verify.
_BCRYPT_ROUNDS = 12

_bearer = HTTPBearer(auto_error=False)


def hash_password(p: str) -> str:
    encoded = p.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_SECRET_BYTES:
        raise ValidationError(
            f"password must not exceed {_BCRYPT_MAX_SECRET_BYTES} bytes"
        )
    return bcrypt.hashpw(
        encoded, bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    ).decode("ascii")


def verify_password(p: str, hashed: str) -> bool:
    if not hashed:
        return False
    encoded = p.encode("utf-8")
    if len(encoded) > _BCRYPT_MAX_SECRET_BYTES:
        # hash_password() could never have stored it, so it cannot match.
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode("ascii"))
    except ValueError:
        return False


def new_api_key() -> str:
    return secrets.token_hex(24)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def issue_jwt(user: User) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "tenant_id": user.tenant_id,
        "role": user.role.value,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def decode_jwt(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    except JWTError as e:
        raise AuthError(f"invalid token: {e}") from e


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------


def _load_user(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise AuthError("user not found or inactive")
    return user


async def current_user(
    db: Annotated[Session, Depends(get_db)],
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> User:
    """Resolve the current user from a Bearer JWT or an X-API-Key header."""
    if authorization and authorization.credentials:
        claims = decode_jwt(authorization.credentials)
        return _load_user(db, claims["sub"])
    if api_key:
        user = db.scalar(select(User).where(User.api_key == api_key))
        if not user or not user.is_active:
            raise AuthError("invalid API key")
        return user
    raise AuthError("missing credentials (Bearer token or X-API-Key)")


CurrentUser = Annotated[User, Depends(current_user)]


async def optional_current_user(
    db: Annotated[Session, Depends(get_db)],
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> User | None:
    """Like ``current_user`` but returns None instead of raising AuthError.

    Used for bootstrap endpoints (e.g. creating the first user in a tenant).
    """
    try:
        if authorization and authorization.credentials:
            claims = decode_jwt(authorization.credentials)
            return _load_user(db, claims["sub"])
        if api_key:
            user = db.scalar(select(User).where(User.api_key == api_key))
            if user and user.is_active:
                return user
    except AuthError:
        return None
    return None


OptionalCurrentUser = Annotated[User | None, Depends(optional_current_user)]


def require_role(*roles: Role):
    """Build a dependency that requires any of ``roles``."""

    async def _dep(user: CurrentUser) -> User:
        if user.role not in roles:
            raise ForbiddenError(f"role {user.role.value} not permitted; "
                                 f"need one of {[r.value for r in roles]}")
        return user

    return _dep


# ---------------------------------------------------------------------------
# Workspace scoping
# ---------------------------------------------------------------------------


def get_workspace(
    workspace_id: str,
    *,
    user: User,
    db: Session,
) -> Workspace:
    ws = db.get(Workspace, workspace_id)
    if not ws or ws.tenant_id != user.tenant_id:
        raise NotFoundError("workspace not found")
    return ws


def can_write(user: User) -> None:
    if user.role == Role.viewer:
        raise ForbiddenError("viewers cannot modify resources")


def can_admin(user: User) -> None:
    if user.role != Role.admin:
        raise ForbiddenError("admin role required")

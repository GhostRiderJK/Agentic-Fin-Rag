"""Auth + tenant/user management router."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import (
    LoginIn,
    TenantCreate,
    TenantOut,
    TokenOut,
    UserCreate,
    UserOut,
)
from app.core.errors import AuthError, ForbiddenError, ValidationError
from app.core.security import (
    CurrentUser,
    OptionalCurrentUser,
    can_admin,
    hash_password,
    issue_jwt,
    new_api_key,
    verify_password,
)
from app.db.base import get_db
from app.db.models import Role, Tenant, User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/tenants", response_model=TenantOut)
def create_tenant(payload: TenantCreate, db: Annotated[Session, Depends(get_db)]):
    t = Tenant(name=payload.name)
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@router.post("/users", response_model=UserOut)
def create_user(
    payload: UserCreate,
    db: Annotated[Session, Depends(get_db)],
    creator: OptionalCurrentUser = None,
):
    # Bootstrap path: creating the first user in a brand-new tenant requires
    # no auth. After that, only admins of that tenant can add more users.
    if not payload.tenant_id:
        if creator is None:
            raise AuthError("missing credentials (Bearer token or X-API-Key)")
        can_admin(creator)
        tenant_id = creator.tenant_id
    else:
        existing = db.scalar(select(User).where(User.tenant_id == payload.tenant_id))
        if existing and creator is None:
            raise AuthError("missing credentials (Bearer token or X-API-Key)")
        if existing and creator is not None:
            if creator.tenant_id != payload.tenant_id:
                raise ForbiddenError("cannot add users to another tenant")
            can_admin(creator)
        tenant_id = payload.tenant_id

    if db.scalar(select(User).where(User.email == payload.email)):
        raise ValidationError("email already registered")
    user = User(
        tenant_id=tenant_id,
        email=payload.email,
        name=payload.name,
        role=payload.role,
        password_hash=hash_password(payload.password) if payload.password else "",
        api_key=new_api_key(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, db: Annotated[Session, Depends(get_db)]):
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not user.password_hash:
        raise AuthError("invalid credentials")
    if not verify_password(payload.password, user.password_hash):
        raise AuthError("invalid credentials")
    token = issue_jwt(user)
    return TokenOut(access_token=token, user=UserOut.model_validate(_user_dict(user)))


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    return UserOut.model_validate(_user_dict(user))


def _user_dict(u: User) -> dict:
    return {
        "id": u.id, "email": u.email, "name": u.name, "role": u.role,
        "tenant_id": u.tenant_id, "api_key": u.api_key, "is_active": u.is_active,
    }

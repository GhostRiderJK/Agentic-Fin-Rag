"""ORM models still owned by the relational store.

Documents, chunks and structured financials now live in MongoDB (see
``app.db.mongodb.repositories``); what remains here is the tenancy spine and
the rows the chat, job and eval paths still write.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


# ---------- Security / tenancy -------------------------------------------------


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    workspaces: Mapped[list[Workspace]] = relationship(back_populates="tenant")


class Role(str, enum.Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.analyst, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    api_key: Mapped[str] = mapped_column(String(64), unique=True, default=_uuid)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    tenant: Mapped[Tenant] = relationship()


class Workspace(Base):
    """A scoped collection of documents (e.g. "Apple 2025 coverage")."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    tenant: Mapped[Tenant] = relationship(back_populates="workspaces")


# ---------- Conversations ------------------------------------------------------


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    # Citations / sources attached to this message
    citations: Mapped[list] = mapped_column(JSON, default=list)
    # Tool calls / agent trace captured for transparency
    trace: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # +1 / -1
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


# ---------- Jobs ---------------------------------------------------------------


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)  # ingest, report, eval
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ---------- Evaluation ---------------------------------------------------------


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    # Owned, like everything an admin can list. Plain string columns rather than
    # ForeignKey so init_db can add them to a database that already has rows
    # (SQLite refuses to ALTER a column in with a REFERENCES clause), and the
    # scope is a read filter, not a referential-integrity requirement.
    tenant_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    workspace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    suite: Mapped[str] = mapped_column(String(128), default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    n_items: Mapped[int] = mapped_column(Integer, default=0)
    # aggregate metrics as JSON (faithfulness, etc.)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")

    items: Mapped[list[EvalItem]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalItem(Base):
    __tablename__ = "eval_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected: Mapped[str] = mapped_column(Text, default="")
    actual: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    # Per-item metric values
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[int] = mapped_column(BigInteger, default=0)

    run: Mapped[EvalRun] = relationship(back_populates="items")


# ---------- Feedback log (for the closed feedback loop) ------------------------


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

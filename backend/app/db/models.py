"""ORM models for the relational store.

This is the *structured knowledge* side of the dual knowledge system:
tenants/workspaces/users, document metadata + lifecycle, parent/child chunk
records, structured financials (the SQL tool-calling target), conversations,
jobs, and evaluation runs.

The vector content lives in Qdrant; this table keeps the authoritative metadata,
status, and references so we can scope queries, manage lifecycle, and run
tool-calling for numeric facts.
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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
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
    documents: Mapped[list[Document]] = relationship(back_populates="workspace")


# ---------- Documents & chunks -------------------------------------------------


class DocumentStatus(str, enum.Enum):
    active = "active"
    superseded = "superseded"
    archived = "archived"


class DocumentType(str, enum.Enum):
    annual_report = "annual_report"
    quarterly_report = "quarterly_report"
    earnings_call = "earnings_call"
    investor_presentation = "investor_presentation"
    sustainability_report = "sustainability_report"
    news = "news"
    research_note = "research_note"
    other = "other"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="")
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    doc_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType), default=DocumentType.other
    )
    company: Mapped[str] = mapped_column(String(255), default="")
    ticker: Mapped[str] = mapped_column(String(32), default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quarter: Mapped[str] = mapped_column(String(8), default="")
    section: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.active
    )
    # Extraction pipeline output kept as JSON for inspection/debug
    parse_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    workspace: Mapped[Workspace] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Parent or child chunk. Child chunks reference their parent (parent
    expansion). Only child chunks are embedded; the parent text is fed to the
    LLM."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    is_parent: Mapped[bool] = mapped_column(Boolean, default=False)
    # Human-readable text content (parent = full section/paragraph; child = search target)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Section / page / location metadata
    section: Mapped[str] = mapped_column(String(255), default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    # Markdown tables extracted from this chunk (if any)
    tables_md: Mapped[str] = mapped_column(Text, default="")
    # Vector DB point id (Qdrant)
    vector_point_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")
    children: Mapped[list[Chunk]] = relationship(
        back_populates="parent", remote_side="Chunk.id"
    )
    parent: Mapped[Chunk | None] = relationship(back_populates="children", remote_side="Chunk.parent_id")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_parent_id", "parent_id"),
    )


# ---------- Structured financials (the SQL tool-calling target) ---------------


class StructuredFinancial(Base):
    """One (company, metric, period, value, unit) row.

    This is the dual to the vector DB: numeric facts are stored relationally
    so the analyst agent can call SQL tools instead of asking the LLM to
    recite numbers it can't reliably remember.
    """

    __tablename__ = "structured_financials"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE")
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    company: Mapped[str] = mapped_column(String(255), nullable=False)
    ticker: Mapped[str] = mapped_column(String(32), default="")
    metric: Mapped[str] = mapped_column(String(128), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    quarter: Mapped[str] = mapped_column(String(8), default="")  # "" = full year
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(32), default="USD")
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    page: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        UniqueConstraint(
            "company", "ticker", "metric", "fiscal_year", "quarter", "unit",
            name="uq_financials_company_metric_period",
        ),
        Index("ix_financials_company", "company"),
        Index("ix_financials_metric", "metric"),
    )


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

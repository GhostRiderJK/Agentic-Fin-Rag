"""Pydantic request/response schemas shared by routers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.db.models import DocumentStatus, DocumentType, Role


# ---- Auth --------------------------------------------------------------------


class TenantCreate(BaseModel):
    name: str


class TenantOut(BaseModel):
    id: str
    name: str


class UserCreate(BaseModel):
    email: str
    name: str = ""
    role: Role = Role.analyst
    password: str = ""
    tenant_id: str | None = None  # admin creates in own tenant if omitted


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: Role
    tenant_id: str
    api_key: str
    is_active: bool


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserOut


# ---- Workspaces --------------------------------------------------------------


class WorkspaceCreate(BaseModel):
    name: str
    description: str = ""


class WorkspaceOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime


# ---- Documents ---------------------------------------------------------------


class DocumentOut(BaseModel):
    id: str
    workspace_id: str
    title: str
    filename: str
    doc_type: DocumentType
    company: str
    ticker: str
    year: int | None
    quarter: str
    section: str
    status: DocumentStatus
    parse_meta: dict
    created_at: datetime


class DocumentUpdate(BaseModel):
    title: str | None = None
    status: DocumentStatus | None = None
    company: str | None = None
    year: int | None = None
    quarter: str | None = None


class IngestResultOut(BaseModel):
    document_id: str
    n_parents: int
    n_children: int
    n_tables: int
    n_financials: int
    needs_ocr: bool
    metadata: dict
    chunks: list[dict] = []


# ---- Chat --------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    workspace_id: str
    conversation_id: str | None = None
    mode: Literal["auto", "document_only", "structured_only"] = "auto"
    stream: bool = False


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    company: str | None
    year: int | None
    quarter: str | None
    doc_type: str | None
    section: str | None
    page: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    plan: dict[str, Any]
    tool_results: list[dict[str, Any]]
    verification: dict[str, Any]
    retrieval: dict[str, Any]
    latency_ms: int
    notes: list[str] = []
    message_id: str | None = None
    conversation_id: str | None = None


# ---- Conversations -----------------------------------------------------------


class ConversationOut(BaseModel):
    id: str
    workspace_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[Any]
    created_at: datetime
    feedback: int | None


# ---- Reports -----------------------------------------------------------------


class ReportRequest(BaseModel):
    workspace_id: str
    company: str
    report_type: Literal["summary", "thesis"] = "summary"
    year: int | None = None
    quarter: str | None = None


class ReportResponse(BaseModel):
    job_id: str
    status: str
    report: str | None = None
    citations: list[dict] = []
    metrics: dict = {}


# ---- Jobs --------------------------------------------------------------------


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    error: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict = {}


# ---- Eval --------------------------------------------------------------------


class EvalRunRequest(BaseModel):
    workspace_id: str
    suite_path: str | None = None  # default to packaged suite


class EvalRunOut(BaseModel):
    id: str
    suite: str
    n_items: int
    metrics: dict
    created_at: datetime


class FeedbackIn(BaseModel):
    rating: int = Field(ge=-1, le=1)
    note: str = ""

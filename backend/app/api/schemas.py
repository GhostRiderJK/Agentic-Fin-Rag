"""Pydantic request/response schemas shared by routers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, PlainSerializer

from app.db.enums import DocumentStatus, DocumentType
from app.db.models import Role


def _utc_iso(value: datetime) -> str:
    """Render an instant as an explicit UTC offset.

    Storage hands back naive datetimes in UTC: SQLite and MySQL drop the
    offset on the round trip, and pymongo always returns naive UTC. Serialised
    bare, that string is a *local* time per ECMAScript, so the browser renders
    every timestamp shifted by the viewer's offset.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# Only applied when a model is dumped to JSON, so internal ``model_dump()``
# callers still get real datetime objects.
UtcDateTime = Annotated[datetime, PlainSerializer(_utc_iso, when_used="json")]

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
    created_at: UtcDateTime


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
    created_at: UtcDateTime


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
    include_evidence_trace: bool = False


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    source_filename: str = ""
    printed_page: int | None = None
    physical_page: int | None = None
    page_number_type: str = "physical"
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
    generation: dict[str, Any] = Field(default_factory=dict)
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
    created_at: UtcDateTime
    updated_at: UtcDateTime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[Any]
    created_at: UtcDateTime
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
    created_at: UtcDateTime
    started_at: UtcDateTime | None = None
    finished_at: UtcDateTime | None = None
    result: dict = {}


# ---- Eval --------------------------------------------------------------------


class EvalRunRequest(BaseModel):
    workspace_id: str
    suite: str = ""  # empty = packaged gold set; names come from /admin/eval/suites


class EvalAccepted(BaseModel):
    """The suite run was queued; poll ``/jobs/{job_id}`` for the metrics."""

    job_id: str
    status: str
    suite: str
    workspace_id: str


class EvalSuiteOut(BaseModel):
    id: str
    packaged: bool
    n_items: int


class EvalRunOut(BaseModel):
    id: str
    suite: str
    n_items: int
    metrics: dict
    created_at: UtcDateTime


class FeedbackIn(BaseModel):
    rating: int = Field(ge=-1, le=1)
    note: str = ""

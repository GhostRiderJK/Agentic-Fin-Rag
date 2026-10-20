"""Documents router: upload, list, get, update status (lifecycle), delete."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi import Path as FastPath
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import WorkspaceCtxDep
from app.api.schemas import (
    DocumentOut,
    DocumentUpdate,
    IngestResultOut,
    WorkspaceCreate,
    WorkspaceOut,
)
from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.security import CurrentUser, can_write
from app.db.base import get_db
from app.db.enums import DocumentStatus, DocumentType
from app.db.models import Workspace
from app.db.mongodb.repositories import (
    ChunkRepository,
    DocumentRepository,
    StructuredFinancialRepository,
)
from app.db.mongodb.runtime import run_sync
from app.ingestion.pipeline import ingest_document
from app.jobs.queue import enqueue_ingest
from app.providers.vectordb import get_vectordb

router = APIRouter(tags=["documents"])

_documents = DocumentRepository()
_chunks = ChunkRepository()
_financials = StructuredFinancialRepository()


# ---- Workspaces --------------------------------------------------------------


@router.post("/workspaces", response_model=WorkspaceOut)
def create_workspace(
    payload: WorkspaceCreate,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    ws = Workspace(tenant_id=user.tenant_id, name=payload.name,
                   description=payload.description)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


@router.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(db: Annotated[Session, Depends(get_db)], user: CurrentUser):
    return list(db.scalars(
        select(Workspace).where(Workspace.tenant_id == user.tenant_id)
    ).all())


# ---- Documents ---------------------------------------------------------------


@router.post("/workspaces/{workspace_id}/documents", response_model=IngestResultOut)
def upload_document(
    ctx: WorkspaceCtxDep,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File()],
    title: str | None = Form(None),
    company: str = Form(""),
    ticker: str = Form(""),
    year: int | None = Form(None),
    quarter: str = Form(""),
    doc_type: Annotated[DocumentType | None, Form()] = None,
    async_ingest: bool = Form(True),
):
    can_write(user)
    settings = get_settings()
    # Save uploaded file to local storage
    # Upload names are untrusted and may collide. Keep the original basename
    # as display metadata, but never use it as the on-disk storage identity.
    original_name = Path((file.filename or "upload").replace("\\", "/")).name
    suffix = Path(original_name).suffix
    storage_dir = Path(settings.storage_path) / ctx.workspace_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    dest = storage_dir / f"{uuid4().hex}{suffix}"
    content = file.file.read()
    if len(content) > settings.upload_max_size_mb * 1024 * 1024:
        raise ValidationError(
            f"file exceeds max size {settings.upload_max_size_mb}MB"
        )
    # Exclusive creation makes accidental UUID/path reuse fail closed instead
    # of truncating a file that a queued ingestion job may still be reading.
    with dest.open("xb") as stored_file:
        stored_file.write(content)

    if async_ingest:
        job = enqueue_ingest(
            db=db, workspace_id=ctx.workspace_id, tenant_id=user.tenant_id,
            file_path=str(dest), source_name=original_name,
            title=title, company=company, ticker=ticker,
            year=year, quarter=quarter, doc_type=doc_type,
        )
        return IngestResultOut(
            document_id="", n_parents=0, n_children=0, n_tables=0,
            n_financials=0, needs_ocr=False,
            metadata={"job_id": job.id, "status": "queued"},
            chunks=[],
        )

    # Synchronous ingest (small files / tests)
    result = ingest_document(
        ctx, file_path=str(dest), source_name=original_name, title=title,
        hint_company=company, hint_ticker=ticker, hint_year=year,
        hint_quarter=quarter, hint_doc_type=doc_type,
    )
    return IngestResultOut(
        document_id=result.document_id,
        n_parents=result.n_parents, n_children=result.n_children,
        n_tables=result.n_tables, n_financials=result.n_financials,
        needs_ocr=result.needs_ocr,
        metadata={
            "company": result.metadata.company,
            "ticker": result.metadata.ticker,
            "year": result.metadata.year,
            "quarter": result.metadata.quarter,
            "doc_type": result.metadata.doc_type.value,
            "section": result.metadata.section,
        },
        chunks=[],
    )


@router.get("/workspaces/{workspace_id}/documents", response_model=list[DocumentOut])
def list_documents(
    ctx: WorkspaceCtxDep,
    status: DocumentStatus | None = None,
    company: str | None = None,
):
    return run_sync(_documents.list_for_workspace(
        ctx, ctx.workspace_id,
        status=status.value if status else None,
        company_like=company,
    ))


@router.get("/workspaces/{workspace_id}/documents/{document_id}",
            response_model=DocumentOut)
def get_document(
    ctx: WorkspaceCtxDep,
    document_id: Annotated[str, FastPath()],
):
    doc = run_sync(_documents.get_by_id(ctx, document_id))
    if not doc or doc["workspace_id"] != ctx.workspace_id:
        raise NotFoundError("document not found")
    return doc


@router.patch("/workspaces/{workspace_id}/documents/{document_id}",
              response_model=DocumentOut)
def update_document(
    ctx: WorkspaceCtxDep,
    document_id: Annotated[str, FastPath()],
    payload: DocumentUpdate,
    user: CurrentUser,
):
    can_write(user)
    doc = run_sync(_documents.get_by_id(ctx, document_id))
    if not doc or doc["workspace_id"] != ctx.workspace_id:
        raise NotFoundError("document not found")
    fields = payload.model_dump(exclude_unset=True, mode="json")
    if fields:
        run_sync(_documents.update_by_id(ctx, document_id, fields))
    return run_sync(_documents.get_by_id(ctx, document_id))


@router.delete("/workspaces/{workspace_id}/documents/{document_id}")
def delete_document(
    ctx: WorkspaceCtxDep,
    document_id: Annotated[str, FastPath()],
    user: CurrentUser,
):
    can_write(user)
    doc = run_sync(_documents.get_by_id(ctx, document_id))
    if not doc or doc["workspace_id"] != ctx.workspace_id:
        raise NotFoundError("document not found")
    # Unpublish in the reverse of the ingest order: vectors and chunks first so
    # nothing searchable points at the document, then the document itself.
    # The extracted figures stay, provenance cleared (what the SQL
    # ON DELETE SET NULL did).
    try:
        get_vectordb().delete_by_document(document_id)
    except Exception:
        pass
    run_sync(_chunks.delete_for_document(ctx, document_id))
    run_sync(_financials.detach_document(ctx, document_id))
    run_sync(_documents.delete_by_id(ctx, document_id))
    return {"deleted": document_id}

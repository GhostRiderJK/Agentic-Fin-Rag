"""Documents router: upload, list, get, update status (lifecycle), delete."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi import Path as FastPath
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import workspace_dep
from app.api.schemas import DocumentOut, DocumentUpdate, IngestResultOut, WorkspaceCreate
from app.api.schemas import WorkspaceOut
from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.security import CurrentUser, can_write
from app.db.base import get_db
from app.db.models import Document, DocumentStatus, DocumentType, Workspace
from app.ingestion.pipeline import ingest_document
from app.jobs.queue import enqueue_ingest, get_job
from app.providers.vectordb import get_vectordb

router = APIRouter(tags=["documents"])


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
async def upload_document(
    db: Annotated[Session, Depends(get_db)],
    ws: Annotated[Workspace, Depends(workspace_dep)],
    user: CurrentUser,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    company: str = Form(""),
    ticker: str = Form(""),
    year: int | None = Form(None),
    quarter: str = Form(""),
    doc_type: DocumentType | None = Form(None),
    async_ingest: bool = Form(True),
):
    can_write(user)
    settings = get_settings()
    # Save uploaded file to local storage
    suffix = Path(file.filename or "upload").suffix
    storage_dir = Path(settings.storage_path) / ws.id
    storage_dir.mkdir(parents=True, exist_ok=True)
    safe_name = (file.filename or "upload").replace("/", "_")
    dest = storage_dir / f"{Path(safe_name).stem}_{Path(safe_name).stem[:8]}{suffix}"
    content = await file.read()
    if len(content) > settings.upload_max_size_mb * 1024 * 1024:
        raise ValidationError(
            f"file exceeds max size {settings.upload_max_size_mb}MB"
        )
    dest.write_bytes(content)

    if async_ingest:
        job = enqueue_ingest(
            db=db, workspace_id=ws.id, tenant_id=user.tenant_id,
            file_path=str(dest), title=title, company=company, ticker=ticker,
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
        file_path=str(dest), workspace_id=ws.id, db=db, title=title,
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
    db: Annotated[Session, Depends(get_db)],
    ws: Annotated[Workspace, Depends(workspace_dep)],
    status: DocumentStatus | None = None,
    company: str | None = None,
):
    q = select(Document).where(Document.workspace_id == ws.id)
    if status:
        q = q.where(Document.status == status)
    if company:
        q = q.where(Document.company.ilike(f"%{company}%"))
    q = q.order_by(Document.created_at.desc())
    return list(db.scalars(q).all())


@router.get("/workspaces/{workspace_id}/documents/{document_id}",
            response_model=DocumentOut)
def get_document(
    ws: Annotated[Workspace, Depends(workspace_dep)],
    db: Annotated[Session, Depends(get_db)],
    document_id: Annotated[str, FastPath()],
):
    doc = db.get(Document, document_id)
    if not doc or doc.workspace_id != ws.id:
        raise NotFoundError("document not found")
    return doc


@router.patch("/workspaces/{workspace_id}/documents/{document_id}",
              response_model=DocumentOut)
def update_document(
    ws: Annotated[Workspace, Depends(workspace_dep)],
    db: Annotated[Session, Depends(get_db)],
    document_id: Annotated[str, FastPath()],
    payload: DocumentUpdate,
    user: CurrentUser,
):
    can_write(user)
    doc = db.get(Document, document_id)
    if not doc or doc.workspace_id != ws.id:
        raise NotFoundError("document not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(doc, k, v)
    db.commit()
    db.refresh(doc)
    return doc


@router.delete("/workspaces/{workspace_id}/documents/{document_id}")
def delete_document(
    ws: Annotated[Workspace, Depends(workspace_dep)],
    db: Annotated[Session, Depends(get_db)],
    document_id: Annotated[str, FastPath()],
    user: CurrentUser,
):
    can_write(user)
    doc = db.get(Document, document_id)
    if not doc or doc.workspace_id != ws.id:
        raise NotFoundError("document not found")
    # Remove from vector store + structured DB, then the document
    try:
        get_vectordb().delete_by_document(document_id)
    except Exception:
        pass
    db.delete(doc)
    db.commit()
    return {"deleted": document_id}

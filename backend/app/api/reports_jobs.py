"""Reports + jobs routers."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import workspace_dep
from app.api.schemas import JobOut, ReportRequest
from app.core.security import CurrentUser
from app.db.base import get_db
from app.db.models import Job, Workspace
from app.jobs.queue import enqueue_report, get_job

router = APIRouter(tags=["reports"])


@router.post("/reports", response_model=JobOut)
def create_report(
    payload: ReportRequest,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    ws = db.get(Workspace, payload.workspace_id)
    if not ws or ws.tenant_id != user.tenant_id:
        from app.core.errors import NotFoundError

        raise NotFoundError("workspace not found")
    job = enqueue_report(
        db=db, workspace_id=ws.id, tenant_id=user.tenant_id, user_id=user.id,
        company=payload.company, report_type=payload.report_type,
        year=payload.year, quarter=payload.quarter,
    )
    return job


jobs_router = APIRouter(tags=["jobs"])


@jobs_router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
    workspace_id: str | None = None,
):
    q = select(Job).where(Job.tenant_id == user.tenant_id)
    if workspace_id:
        q = q.where(Job.workspace_id == workspace_id)
    q = q.order_by(Job.created_at.desc()).limit(50)
    return list(db.scalars(q).all())


@jobs_router.get("/jobs/{job_id}", response_model=JobOut)
def job_detail(job_id: str, db: Annotated[Session, Depends(get_db)],
               user: CurrentUser):
    job = get_job(job_id, db=db)
    if job.tenant_id != user.tenant_id:
        from app.core.errors import NotFoundError

        raise NotFoundError("job not found")
    return job

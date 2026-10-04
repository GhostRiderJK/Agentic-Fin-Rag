"""Async job queue (PRD §6: long-running reports & ingestion).

Two modes (JOBS_MODE env):
  - inproc: a background ``threading.Thread`` worker that processes jobs
            sequentially. Zero external deps. Good for dev/demo and small
            production loads.
  - celery: defers to a Celery worker + Redis broker. (Optional; only
            imported when JOBS_MODE=celery.)

The API stays identical: ``enqueue_*`` returns a Job row that clients poll.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.base import db_scope, get_sessionmaker, init_db
from app.db.models import Job, JobStatus, Workspace
from app.ingestion.pipeline import ingest_document

log = get_logger(__name__)


# ---- public API --------------------------------------------------------------

def enqueue_ingest(
    *, db: Session, workspace_id: str, tenant_id: str, file_path: str,
    title: str | None = None, company: str = "", ticker: str = "",
    year: int | None = None, quarter: str = "", doc_type: Any = None,
) -> Job:
    job = Job(tenant_id=tenant_id, workspace_id=workspace_id, kind="ingest",
              status=JobStatus.queued,
              payload={
                  "file_path": file_path, "title": title, "company": company,
                  "ticker": ticker, "year": year, "quarter": quarter,
                  "doc_type": doc_type.value if doc_type else None,
              })
    db.add(job)
    db.commit()
    db.refresh(job)
    _enqueue(job.id)
    return job


def enqueue_report(
    *, db: Session, workspace_id: str, tenant_id: str, user_id: str,
    company: str, report_type: str = "summary", year: int | None = None,
    quarter: str | None = None,
) -> Job:
    job = Job(tenant_id=tenant_id, workspace_id=workspace_id, kind="report",
              status=JobStatus.queued,
              payload={
                  "company": company, "report_type": report_type, "year": year,
                  "quarter": quarter, "user_id": user_id,
              })
    db.add(job)
    db.commit()
    db.refresh(job)
    _enqueue(job.id)
    return job


def get_job(job_id: str, db: Session | None = None) -> Job:
    own = db is None
    if own:
        sm = get_sessionmaker()
        db = sm()
    try:
        job = db.get(Job, job_id)  # type: ignore[union-attr]
        if not job:
            raise NotFoundError("job not found")
        return job
    finally:
        if own:
            db.close()  # type: ignore[union-attr]


# ---- in-proc worker ----------------------------------------------------------

_queue: "queue.Queue[str]" = queue.Queue()
_worker_thread: threading.Thread | None = None
_lock = threading.Lock()


def _enqueue(job_id: str) -> None:
    settings = get_settings()
    if settings.jobs_mode != "inproc":
        return  # celery will pick it up
    _ensure_worker()
    _queue.put(job_id)


def _ensure_worker() -> None:
    global _worker_thread
    with _lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True,
                                          name="ira-jobs")
        _worker_thread.start()


def _worker_loop() -> None:  # pragma: no cover - background thread
    log.info("job worker started (inproc mode)")
    sm = get_sessionmaker()
    while True:
        try:
            job_id = _queue.get(timeout=5)
        except queue.Empty:
            continue
        try:
            _process_job(job_id, sm)
        except Exception as e:
            log.exception("job %s failed: %s", job_id, e)
        finally:
            _queue.task_done()


def _process_job(job_id: str, sm) -> None:
    """Process one job: ingest or report."""
    db = sm()
    try:
        job = db.get(Job, job_id)
        if not job:
            log.warning("unknown job %s", job_id)
            return
        job.status = JobStatus.running
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        try:
            if job.kind == "ingest":
                result = _do_ingest(job, db)
            elif job.kind == "report":
                result = _do_report(job, db)
            else:
                raise ValueError(f"unknown job kind: {job.kind}")
            job.result = result
            job.status = JobStatus.succeeded
            job.progress = 1.0
        except Exception as e:
            job.status = JobStatus.failed
            job.error = f"{type(e).__name__}: {e}"
            log.exception("job %s failed", job_id)
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def _do_ingest(job: Job, db: Session) -> dict:
    from app.db.models import DocumentType

    p = job.payload
    doc_type = None
    if p.get("doc_type"):
        try:
            doc_type = DocumentType(p["doc_type"])
        except ValueError:
            doc_type = None
    result = ingest_document(
        file_path=p["file_path"], workspace_id=job.workspace_id, db=db,
        title=p.get("title"), hint_company=p.get("company", ""),
        hint_ticker=p.get("ticker", ""), hint_year=p.get("year"),
        hint_quarter=p.get("quarter", ""), hint_doc_type=doc_type,
    )
    return {
        "document_id": result.document_id,
        "n_parents": result.n_parents,
        "n_children": result.n_children,
        "n_tables": result.n_tables,
        "n_financials": result.n_financials,
    }


def _do_report(job: Job, db: Session) -> dict:
    from app.reasoning.reports import build_report

    p = job.payload
    out = build_report(
        db=db, workspace_id=job.workspace_id, company=p["company"],
        report_type=p.get("report_type", "summary"),
        year=p.get("year"), quarter=p.get("quarter"),
    )
    return out


def wait_for_jobs(timeout: float = 30.0) -> None:
    """Test helper: block until the queue drains (inproc mode only)."""
    settings = get_settings()
    if settings.jobs_mode != "inproc":
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _queue.unfinished_tasks == 0:
            return
        time.sleep(0.1)

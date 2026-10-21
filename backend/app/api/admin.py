"""Admin/eval router: queue evaluation suites, list past runs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import EvalAccepted, EvalRunOut, EvalRunRequest, EvalSuiteOut
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.core.security import CurrentUser, can_admin
from app.db.base import get_db
from app.db.models import EvalItem, EvalRun, Workspace
from app.evaluation.harness import reconcile_run_metrics
from app.evaluation.suites import list_suites, resolve
from app.jobs.queue import enqueue_eval

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/eval/suites", response_model=list[EvalSuiteOut])
def eval_suites(user: CurrentUser):
    """The gold sets this deployment can run, addressed by name."""
    can_admin(user)
    return list_suites()


@router.post("/eval/run", response_model=EvalAccepted,
             status_code=status.HTTP_202_ACCEPTED)
def run_eval(
    payload: EvalRunRequest,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    """Queue a suite run and hand back the job to poll.

    Running the suite inside this request took minutes, which no client is
    willing to wait for: the Next.js proxy dropped it at 30s and the UI showed a
    500 while the run completed server-side anyway.
    """
    can_admin(user)
    ws = db.get(Workspace, payload.workspace_id)
    if not ws or ws.tenant_id != user.tenant_id:
        raise NotFoundError("workspace not found")
    # Resolved here rather than in the worker so an unknown suite is a 404 on
    # this call instead of a failed job the client finds out about by polling.
    _path, suite_id = resolve(payload.suite)
    job = enqueue_eval(
        db=db, workspace_id=ws.id, tenant_id=user.tenant_id, user_id=user.id,
        suite=suite_id,
    )
    return EvalAccepted(
        job_id=job.id, status=job.status.value, suite=suite_id,
        workspace_id=ws.id,
    )


@router.get("/eval/runs", response_model=list[EvalRunOut])
def list_eval_runs(db: Annotated[Session, Depends(get_db)], user: CurrentUser):
    can_admin(user)
    # Scoped by tenant, like every other list an admin can reach. Runs written
    # before eval_runs carried a tenant have no owner to match, so they stay
    # out of every list rather than being shown to everyone.
    runs = list(db.scalars(select(EvalRun)
                           .where(EvalRun.tenant_id == user.tenant_id)
                           .order_by(EvalRun.created_at.desc())
                           .limit(50)).all())
    if not runs:
        return []
    # One pass over the item rows: runs written by the older harness have to be
    # re-read before they can be shown, because their stored averages include
    # requests that failed instead of answering.
    item_rows: dict[str, list[tuple[str, dict]]] = {}
    for run_id, actual, metrics in db.execute(
        select(EvalItem.run_id, EvalItem.actual, EvalItem.metrics)
        .where(EvalItem.run_id.in_([r.id for r in runs]))
    ).all():
        item_rows.setdefault(run_id, []).append((actual, metrics))
    return [
        EvalRunOut(
            id=r.id, suite=r.suite, n_items=r.n_items,
            metrics=reconcile_run_metrics(r.metrics or {}, r.n_items,
                                          item_rows.get(r.id, [])),
            created_at=r.created_at,
        ) for r in runs
    ]

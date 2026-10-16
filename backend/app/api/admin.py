"""Admin/eval router: run evaluation suites, list past runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import workspace_dep
from app.api.schemas import EvalRunOut, EvalRunRequest
from app.core.logging import get_logger
from app.core.security import CurrentUser, can_admin
from app.db.base import get_db
from app.db.models import EvalRun, Workspace
from app.evaluation.harness import EvalRunner, EvalSuite

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


DEFAULT_SUITE_PATH = str(Path(__file__).resolve().parent.parent / "seed" /
                         "gold_qa.jsonl")


@router.post("/eval/run", response_model=EvalRunOut)
def run_eval(
    payload: EvalRunRequest,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    can_admin(user)
    ws = db.get(Workspace, payload.workspace_id)
    if not ws or ws.tenant_id != user.tenant_id:
        from app.core.errors import NotFoundError

        raise NotFoundError("workspace not found")
    suite_path = payload.suite_path or DEFAULT_SUITE_PATH
    if not Path(suite_path).exists():
        from app.core.errors import NotFoundError

        raise NotFoundError(f"eval suite not found: {suite_path}")
    suite = EvalSuite.from_jsonl(suite_path, workspace_id=ws.id, name="default")
    runner = EvalRunner(db)
    result = runner.run(suite)
    run = db.scalar(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1))
    return EvalRunOut(
        id=run.id if run else "", suite=result.suite, n_items=result.n_items,
        metrics=result.metrics, created_at=run.created_at if run else None,
    )


@router.get("/eval/runs", response_model=list[EvalRunOut])
def list_eval_runs(db: Annotated[Session, Depends(get_db)], user: CurrentUser):
    can_admin(user)
    return list(db.scalars(select(EvalRun).order_by(EvalRun.created_at.desc())
                           .limit(50)).all())

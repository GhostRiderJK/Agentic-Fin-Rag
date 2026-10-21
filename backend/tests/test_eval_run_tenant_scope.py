"""The eval-run rows are tenant-owned.

``GET /admin/eval/runs`` used to answer with every tenant's runs because the
table had no owner column at all. Two things had to be true to close that, and
both are easy to lose again: the writer has to record the owner, and a database
that already has the table has to gain the columns (``create_all`` skips
existing tables and this project has no Alembic).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from app.db.base import _ADDED_COLUMNS, _add_missing_columns
from app.db.models import Base, EvalItem, EvalRun
from app.db.mongodb.tenant_context import TenantContext
from app.evaluation.harness import EvalResult, EvalRunner, EvalSuite, GoldItem

TENANT_A = "a" * 32
TENANT_B = "b" * 32
WS_A = "w" * 32


@pytest.fixture()
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'scope.db'}", future=True)
    Base.metadata.create_all(bind=eng, tables=[EvalRun.__table__, EvalItem.__table__])
    yield eng
    eng.dispose()


def test_run_rows_record_who_ordered_them(engine):
    """The route can only filter on an owner the writer actually stored."""
    ctx = TenantContext(tenant_id=TENANT_A, user_id="u" * 32, workspace_id=WS_A)
    suite = EvalSuite(name="pif_gold_qa_v2", workspace_id=WS_A,
                      items=[GoldItem(question="q", expected="e")])
    result = EvalResult(suite=suite.name, n_items=1,
                        metrics={"n_scored": 1.0}, items=[])
    with Session(engine) as db:
        EvalRunner(db, ctx)._persist(suite, result)
        run = db.scalar(select(EvalRun))
        assert run.tenant_id == TENANT_A
        assert run.workspace_id == WS_A


def test_a_run_ordered_without_a_workspace_keeps_its_tenant(engine):
    ctx = TenantContext(tenant_id=TENANT_A, user_id="u" * 32, workspace_id=None)
    suite = EvalSuite(name="packaged", workspace_id="", items=[])
    with Session(engine) as db:
        EvalRunner(db, ctx)._persist(suite, EvalResult(suite=suite.name, n_items=0,
                                                       metrics={}, items=[]))
        run = db.scalar(select(EvalRun))
        assert run.tenant_id == TENANT_A
        assert run.workspace_id is None


def test_scope_columns_are_added_to_a_table_that_predates_them(tmp_path):
    """The deployed database already has eval_runs, without an owner column.

    ``create_all`` will not touch it, so the additive ALTER in init_db is the
    only thing standing between a fresh image and "no such column:
    eval_runs.tenant_id" on the first admin request.
    """
    eng = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}", future=True)
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE eval_runs (id VARCHAR(32) PRIMARY KEY, suite VARCHAR(128),"
            " created_at DATETIME, n_items INTEGER, metrics JSON, notes TEXT)"))
    assert {"tenant_id", "workspace_id"}.isdisjoint(
        {c["name"] for c in inspect(eng).get_columns("eval_runs")})

    _add_missing_columns(eng)

    cols = {c["name"] for c in inspect(eng).get_columns("eval_runs")}
    assert {"tenant_id", "workspace_id"} <= cols
    indexes = {i["name"] for i in inspect(eng).get_indexes("eval_runs")}
    assert "ix_eval_runs_tenant_id" in indexes
    _add_missing_columns(eng)  # idempotent: nothing to add a second time
    assert len(_ADDED_COLUMNS["eval_runs"]) == 2
    eng.dispose()


def test_legacy_rows_belong_to_nobody_and_are_not_shown_to_everybody(engine):
    """A row with no tenant of record is unattributable, not public.

    This is the consequence of scoping the route that the deployment has to
    accept: run history written before the column existed disappears from the
    History table until someone re-runs the suite.
    """
    with Session(engine) as db:
        db.add(EvalRun(suite="packaged", n_items=7, tenant_id=None))
        db.add(EvalRun(suite="pif_gold_qa_v2", n_items=10, tenant_id=TENANT_B))
        db.commit()
        mine = db.scalars(select(EvalRun).where(EvalRun.tenant_id == TENANT_A)).all()
        assert [r.suite for r in mine] == []
        theirs = db.scalars(select(EvalRun).where(EvalRun.tenant_id == TENANT_B)).all()
        assert [r.suite for r in theirs] == ["pif_gold_qa_v2"]

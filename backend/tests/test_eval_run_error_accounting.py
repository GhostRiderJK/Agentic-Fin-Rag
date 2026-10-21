"""Regression tests for eval-run error accounting.

A suite run is the number the operator trusts, so a run in which the pipeline
raised must not be reported as a measurement. Live case (2026-10-20, isolated
stack, run `d6a171b47c3a495f884fc50bb99005af`): every one of the 7 items failed
with a gateway 401, yet the job said `succeeded` and the table showed
`mean_citation_precision 1.0` — because the exception text was scored as if it
were an answer, and that stub metric is "1.0 if the text contains a number".
"""
from __future__ import annotations

import types
import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.base import db_scope, init_db
from app.db.models import EvalItem, EvalRun
from app.evaluation.harness import (
    EvalItemResult,
    EvalResult,
    EvalRunner,
    EvalSuite,
    GoldItem,
    aggregate_metrics,
    reconcile_run_metrics,
)


class _RaisingOrchestrator:
    """Stands in for the pipeline when no model can answer."""

    def __init__(self, error: str = "ProviderError: 401 No active credentials"):
        self.error = error

    def answer(self, request):  # noqa: ARG002 - signature only
        raise RuntimeError(self.error)


class _AnsweringOrchestrator:
    def answer(self, request):
        return types.SimpleNamespace(
            answer=f"Net income was 28,000 million in {request.question[-4:]}",
            citations=[{"chunk_id": "c1", "page": 3}],
            retrieval={"n_chunks": 6},
            verification={"outcome": "supported"},
        )


def _suite(*questions: str) -> EvalSuite:
    return EvalSuite(
        name="t",
        workspace_id="ws",
        items=[GoldItem(question=q, expected="Net income was 28,000 million",
                        expected_numbers=["28000"], expected_keywords=["net"])
               for q in questions],
    )


def _run(orch) -> EvalResult:
    return EvalRunner(db=None, ctx=None, orchestrator=orch).run(  # type: ignore[arg-type]
        _suite("What was net income in 2025?", "And the gross margin in 2025?"),
        persist=False,
    )


def test_a_run_where_nothing_answered_has_no_metric_table():
    result = _run(_RaisingOrchestrator())
    assert result.metrics == {"n_scored": 0.0, "n_errored": 2.0, "error_rate": 1.0}
    # The shape of the old bug: averages exist even though nothing was measured.
    assert not [k for k in result.metrics if k.startswith("mean_")]
    assert all(it.metrics == {} for it in result.items)
    assert all("401 No active credentials" in it.error for it in result.items)


def test_answered_items_are_scored_and_errored_ones_are_not():
    mixed = _RaisingOrchestrator()
    good = _AnsweringOrchestrator()

    class _HalfBroken:
        def __init__(self):
            self.calls = 0

        def answer(self, request):
            self.calls += 1
            if self.calls == 1:
                return good.answer(request)
            return mixed.answer(request)

    result = _run(_HalfBroken())
    assert result.metrics["n_scored"] == 1.0
    assert result.metrics["n_errored"] == 1.0
    assert result.metrics["error_rate"] == 0.5
    # The average is taken over the one real answer, not diluted by the failure.
    scored = result.items[0].metrics
    assert result.metrics["mean_numeric_accuracy"] == scored["numeric_accuracy"]
    assert result.items[1].metrics == {}


def test_errored_item_still_keeps_its_text_for_forensics():
    result = _run(_RaisingOrchestrator())
    assert result.items[0].actual.startswith("<error: ")
    assert "401 No active credentials" in result.items[0].actual


def test_aggregate_counts_and_averages_separately():
    one = {"faithfulness": 1.0, "numeric_accuracy": 0.0}
    assert aggregate_metrics([one, {}]) == {
        "n_scored": 1.0, "n_errored": 1.0, "error_rate": 0.5,
        "mean_faithfulness": 1.0, "pass_faithfulness": 1.0,
        "mean_numeric_accuracy": 0.0, "pass_numeric_accuracy": 0.0,
    }


def test_aggregate_of_an_empty_suite_is_counted_not_gone():
    assert aggregate_metrics([]) == {
        "n_scored": 0.0, "n_errored": 0.0, "error_rate": 0.0}


def _gold_file(tmp_path) -> str:
    """A real gold set on disk, so only the runner is faked in these two tests."""
    lines = [
        '{"question": "What was net income in 2025?", '
        '"expected": "28,000 million", "expected_numbers": ["28000"], '
        '"expected_keywords": ["net"]}',
        '{"question": "And the gross margin in 2025?", "expected": "46%", '
        '"expected_numbers": ["46"], "expected_keywords": ["margin"]}',
    ]
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_eval_job_fails_when_no_item_answered(monkeypatch, tmp_path):
    from app.evaluation import harness
    from app.jobs import queue as queue_mod

    monkeypatch.setattr("app.evaluation.suites.resolve",
                        lambda _s: (_gold_file(tmp_path), "t"))

    class _AllFailRunner(EvalRunner):
        def __init__(self, db, ctx, orchestrator=None):  # noqa: D107
            pass

        def run(self, suite, *, persist=True):  # noqa: ARG002
            return EvalResult(
                suite=suite.name, n_items=len(suite.items),
                metrics={"n_scored": 0.0, "n_errored": 2.0, "error_rate": 1.0},
                items=[EvalItemResult(question="q", expected="", actual="<error: x>",
                                      citations=[], metrics={}, latency_ms=1,
                                      error="RuntimeError: 401")],
            )

    monkeypatch.setattr(harness, "EvalRunner", _AllFailRunner)
    job = types.SimpleNamespace(payload={"suite": "t"}, workspace_id="ws",
                                tenant_id="tn", user_id="u")

    with pytest.raises(RuntimeError, match="all 2 eval items failed to answer"):
        queue_mod._do_eval(job, db=None)  # type: ignore[arg-type]


def test_eval_job_reports_partial_failures_without_faking_success(monkeypatch,
                                                                   tmp_path):
    from app.evaluation import harness
    from app.jobs import queue as queue_mod

    monkeypatch.setattr("app.evaluation.suites.resolve",
                        lambda _s: (_gold_file(tmp_path), "t"))

    class _HalfRunner(EvalRunner):
        def __init__(self, db, ctx, orchestrator=None):  # noqa: D107
            pass

        def run(self, suite, *, persist=True):  # noqa: ARG002
            return EvalResult(
                suite=suite.name, n_items=2,
                metrics={"n_scored": 1.0, "n_errored": 1.0, "error_rate": 0.5,
                         "mean_faithfulness": 1.0},
                items=[],
            )

    monkeypatch.setattr(harness, "EvalRunner", _HalfRunner)
    job = types.SimpleNamespace(payload={"suite": "t"}, workspace_id="ws",
                                tenant_id="tn", user_id="u")
    out = queue_mod._do_eval(job, db=None)  # type: ignore[arg-type]
    assert out["n_errored"] == 1
    assert out["metrics"]["error_rate"] == 0.5


def test_reconcile_leaves_current_rows_alone():
    stored = {"n_scored": 1.0, "n_errored": 1.0, "error_rate": 0.5}
    assert reconcile_run_metrics(stored, 2, [("<error: x>", {})]) == stored
    # No item rows to work from: the stored numbers are all there is.
    assert reconcile_run_metrics({"mean_faithfulness": 0.4}, 3, []) == \
        {"mean_faithfulness": 0.4}


def test_reconcile_rebuilds_a_legacy_row_over_answered_items_only():
    legacy = {"mean_faithfulness": 0.333, "mean_citation_precision": 1.0}
    items = [
        ("<error: 401 No active credentials>",
         {"faithfulness": 0.0, "citation_precision": 1.0}),
        ("<error: 401 No active credentials>",
         {"faithfulness": 0.0, "citation_precision": 1.0}),
        ("Net income was 28,000 million.",
         {"faithfulness": 1.0, "citation_precision": 1.0}),
    ]
    out = reconcile_run_metrics(legacy, 3, items)
    assert out["n_scored"] == 1.0
    assert out["n_errored"] == 2.0
    assert out["error_rate"] == round(2 / 3, 3)
    # The answered item is what the average now describes.
    assert out["mean_faithfulness"] == 1.0
    assert out["pass_faithfulness"] == 1.0


def test_reconcile_strips_fake_averages_when_nothing_answered():
    out = reconcile_run_metrics(
        {"mean_citation_precision": 1.0, "mean_numeric_accuracy": 0.143}, 2,
        [("<error: 401>", {"citation_precision": 1.0, "numeric_accuracy": 0.0}),
         ("<error: 401>", {"citation_precision": 1.0, "numeric_accuracy": 0.2})])
    assert out == {"n_scored": 0.0, "n_errored": 2.0, "error_rate": 1.0}


def test_reconcile_flags_item_rows_that_are_missing():
    out = reconcile_run_metrics({}, 5, [("<error: 401>", {})])
    assert out["n_items_unmatched"] == 4.0


# ---------------------------------------------------------------------------
# An error row is a stored document, and the dashboard reads it back out. The
# outbound HTTP handlers scrub credentials, but these two rows never pass
# through one: ``GET /admin/eval/runs`` returns ``EvalItem.actual`` and
# ``GET /jobs`` returns ``Job.error``, so the provider text has to be cleaned at
# the moment it is written.

_LEAKY = ("Error code: 401 - unauthorized: api_key=sk-abcdefghijklmnopqrstuvwxyz "
          "for provider: nvidia")


def test_a_provider_error_is_scrubbed_before_the_item_row_is_written():
    result = _run(_RaisingOrchestrator(_LEAKY))
    for item in result.items:
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in item.error
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in item.actual
        assert "api_key=[redacted]" in item.error
        # The part the operator needs is untouched.
        assert "Error code: 401" in item.actual


def test_a_failed_job_stores_a_scrubbed_error(monkeypatch):
    from app.db.base import get_sessionmaker, init_db
    from app.db.models import Job, JobStatus, Tenant, Workspace
    from app.jobs import queue as queue_mod

    init_db()

    def _boom(job, db):  # noqa: ARG001 - stand-in for the real runner
        raise RuntimeError(_LEAKY)

    monkeypatch.setattr(queue_mod, "_do_eval", _boom)
    sm = get_sessionmaker()
    with sm() as db:
        # jobs.tenant_id/workspace_id are enforced foreign keys on SQLite.
        tenant = Tenant(name=f"Scrub {uuid.uuid4().hex[:8]}")
        db.add(tenant)
        db.flush()
        ws = Workspace(tenant_id=tenant.id, name="w")
        db.add(ws)
        db.flush()
        job = Job(tenant_id=tenant.id, workspace_id=ws.id, kind="eval",
                  status=JobStatus.queued, payload={"suite": "t"})
        db.add(job)
        db.commit()
        job_id, tenant_id, ws_id = job.id, tenant.id, ws.id

    queue_mod._process_job(job_id, sm)

    with sm() as db:
        stored = db.get(Job, job_id)
        assert stored.status == JobStatus.failed
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in stored.error
        assert "api_key=[redacted]" in stored.error
        # The rows this test created are its own to remove.
        db.delete(stored)
        db.delete(db.get(Workspace, ws_id))
        db.delete(db.get(Tenant, tenant_id))
        db.commit()
def test_listing_recomputes_a_row_written_before_the_fix():
    """A pre-fix run row must not keep presenting a failed suite as a pass.

    The row below is the shape the old harness persisted: exception text scored
    as answers, so `mean_faithfulness` was 0.333 across three items of which two
    never answered. The listing rebuilds the aggregate over the answered item and
    reports the split; the stored row itself is not touched.
    """
    init_db()
    from app.main import app

    tag = uuid.uuid4().hex[:8]
    client = TestClient(app)
    tenant = client.post("/auth/tenants",
                         json={"name": f"EvalCounts {tag}"}).json()
    user = client.post("/auth/users", json={
        "email": f"counts-{tag}@x.com", "name": "C",
        "tenant_id": tenant["id"], "role": "admin"}).json()
    headers = {"X-API-Key": user["api_key"]}

    with db_scope() as db:
        # Owned by the listing admin's tenant: eval_runs is tenant-scoped, so a
        # row with no tenant would be invisible here rather than recomputed.
        run = EvalRun(suite="packaged", n_items=3, metrics={
            "mean_faithfulness": 0.333, "mean_citation_precision": 1.0},
            tenant_id=tenant["id"], workspace_id=None)
        db.add(run)
        db.flush()
        db.add_all([
            EvalItem(run_id=run.id, question="q1", expected="",
                     actual="<error: 401 No active credentials>", citations=[],
                     metrics={}, latency_ms=1),
            EvalItem(run_id=run.id, question="q2", expected="",
                     actual="<error: 401 No active credentials>", citations=[],
                     metrics={}, latency_ms=1),
            EvalItem(run_id=run.id, question="q3", expected="28,000",
                     actual="Net income was 28,000 million.", citations=[],
                     metrics={"faithfulness": 1.0, "citation_precision": 1.0},
                     latency_ms=1),
        ])
        db.commit()
        run_id = run.id

    rows = client.get("/admin/eval/runs", headers=headers).json()
    mine = next(r for r in rows if r["id"] == run_id)
    assert mine["metrics"]["n_errored"] == 2.0
    assert mine["metrics"]["n_scored"] == 1.0
    assert mine["metrics"]["error_rate"] == round(2 / 3, 3)
    # Rebuilt over the one item that answered, replacing the stored 0.333.
    assert mine["metrics"]["mean_faithfulness"] == 1.0
    assert mine["metrics"]["pass_faithfulness"] == 1.0

    with db_scope() as db:
        assert db.get(EvalRun, run_id).metrics["mean_faithfulness"] == 0.333

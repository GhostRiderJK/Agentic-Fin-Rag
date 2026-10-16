"""Tenant-isolation matrix: every tenant-owned repository, every hostile path.

test_mongo_isolation.py proves the rules on a handful of repositories. This
file applies the same battery mechanically to *all* of them, including the
domain methods each class adds on top of ``BaseRepository`` — because a
hand-written query in a subclass is exactly where scoping gets forgotten.

Two meta-tests keep the matrix honest as the schema grows:

* every collection listed in ``collections.TENANT_OWNED`` must have a case
  here, so a new tenant-owned collection cannot ship without isolation coverage;
* every ``BaseRepository`` subclass must target a tenant-owned collection, so a
  repository cannot be pointed at an unscoped one.

Each test runs twice: on mongomock-motor and on the real server at
``MONGODB_TEST_URL`` (see ``mongo_support``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import pytest

from app.db.mongodb import collections as C
from app.db.mongodb.base_repository import BaseRepository
from app.db.mongodb.repositories import (
    AuditLogRepository,
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    EvalItemRepository,
    EvalRunRepository,
    FeedbackRepository,
    JobRepository,
    MessageRepository,
    StructuredFinancialRepository,
    TenantRepository,
    UserRepository,
    WorkspaceRepository,
)
from app.db.mongodb.tenant_context import TenantContext, TenantContextError

if not os.path.exists(os.path.join(os.path.dirname(__file__), '__init__.py')):
    from mongo_support import ctx_a, ctx_b, db, run  # noqa: F401
else:
    from tests.mongo_support import ctx_a, ctx_b, db, run  # noqa: F401


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

Probe = Callable[..., Any]
"""``fn(repo, own_ctx, foreign_ctx, own_doc, foreign_doc)`` -> docs or ids.

The contract is that no probe may surface anything belonging to a tenant other
than ``own_ctx.tenant_id``. Returning nothing is always acceptable; returning
foreign data is a release blocker.
"""


@dataclass(frozen=True)
class Case:
    collection: str
    repo_cls: type
    doc: Callable[[str], Dict[str, Any]]
    probes: List[Probe] = field(default_factory=list)
    # Extra documents seeded only for tenant B, to give probes something
    # foreign worth reaching for.
    b_only: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.collection


def _as_records(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


CASES: List[Case] = [
    Case(
        collection=C.USERS,
        repo_cls=UserRepository,
        doc=lambda tag: {
            "email": f"matrix-{tag}@example.test",
            "name": f"User {tag}",
            "api_key": f"k-{tag}",
            "is_active": True,
        },
        probes=[
            lambda r, own, other, a, b: r.get_by_email(own, b["email"]),
            lambda r, own, other, a, b: r.get_by_api_key(own, b["api_key"]),
            lambda r, own, other, a, b: r.list_active(own),
            lambda r, own, other, a, b: r.get_by_email(own, ""),
        ],
    ),
    Case(
        collection=C.WORKSPACES,
        repo_cls=WorkspaceRepository,
        doc=lambda tag: {"name": f"ws-{tag}", "description": f"Workspace {tag}"},
        probes=[lambda r, own, other, a, b: r.list_for_tenant(own)],
    ),
    Case(
        collection=C.DOCUMENTS,
        repo_cls=DocumentRepository,
        doc=lambda tag: {
            "title": f"doc-{tag}",
            "workspace_id": f"ws-{tag}",
            "company": "SHARED",
            "year": 2025,
            "status": "ready",
        },
        b_only=[{
            "title": "b-only-doc",
            "workspace_id": "ws-B",
            "company": "ONLY-B",
            "year": 2024,
        }],
        probes=[
            lambda r, own, other, a, b: r.list_for_workspace(own, "ws-B"),
            lambda r, own, other, a, b: r.find_by_metadata(own, company="ONLY-B"),
            lambda r, own, other, a, b: r.find_by_metadata(own, company="SHARED"),
            lambda r, own, other, a, b: r.set_status(own, b["id"], "archived"),
            lambda r, own, other, a, b: r.list_for_workspace(own, b["workspace_id"]),
        ],
    ),
    Case(
        collection=C.CHUNKS,
        repo_cls=ChunkRepository,
        doc=lambda tag: {
            "document_id": f"doc-{tag}",
            "parent_id": f"parent-{tag}",
            "text": f"chunk text {tag}",
            "order": 1,
        },
        probes=[
            lambda r, own, other, a, b: r.list_for_document(own, b["document_id"]),
            lambda r, own, other, a, b: r.list_children(own, b["parent_id"]),
            lambda r, own, other, a, b: r.get_many_by_ids(own, [a["id"], b["id"]]),
            lambda r, own, other, a, b: r.get_many_by_ids(own, [b["id"]]),
            lambda r, own, other, a, b: r.get_many_by_ids(own, []),
        ],
    ),
    Case(
        collection=C.STRUCTURED_FINANCIALS,
        repo_cls=StructuredFinancialRepository,
        doc=lambda tag: {
            "company": "SHARED",
            "metric": "revenue",
            "fiscal_year": 2025,
            "quarter": "Q1",
            "value": Decimal("100.00"),
        },
        b_only=[{
            "company": "ONLY-B",
            "metric": "secret_margin",
            "fiscal_year": 2025,
            "quarter": "Q2",
            "value": Decimal("77.00"),
        }],
        probes=[
            lambda r, own, other, a, b: r.get_metric(
                own, company="ONLY-B", metric="secret_margin"),
            lambda r, own, other, a, b: r.get_metric(own, company="SHARED",
                                                    metric="revenue",
                                                    fiscal_year=2025),
            lambda r, own, other, a, b: r.compare_metric(
                own, companies=["SHARED", "ONLY-B"], metric="revenue"),
            lambda r, own, other, a, b: r.list_metrics(own, "ONLY-B"),
            lambda r, own, other, a, b: r.list_metrics(own, "SHARED"),
        ],
    ),
    Case(
        collection=C.CONVERSATIONS,
        repo_cls=ConversationRepository,
        doc=lambda tag: {"workspace_id": f"ws-{tag}", "title": f"conv-{tag}"},
        probes=[
            lambda r, own, other, a, b: r.list_for_workspace(own, b["workspace_id"]),
            lambda r, own, other, a, b: r.list_for_workspace(own, a["workspace_id"]),
        ],
    ),
    Case(
        collection=C.MESSAGES,
        repo_cls=MessageRepository,
        doc=lambda tag: {
            "conversation_id": f"conv-{tag}",
            "role": "user",
            "content": f"message {tag}",
        },
        probes=[
            lambda r, own, other, a, b: r.list_for_conversation(own, b["conversation_id"]),
            lambda r, own, other, a, b: r.list_for_conversation(own, a["conversation_id"]),
            lambda r, own, other, a, b: r.set_feedback(own, b["id"], 5),
        ],
    ),
    Case(
        collection=C.JOBS,
        repo_cls=JobRepository,
        doc=lambda tag: {
            "kind": "ingest",
            "status": "queued",
            "progress": Decimal("0.25"),
            "payload": {f"k{tag}": 1},
        },
        probes=[lambda r, own, other, a, b: r.list_recent(own, limit=10)],
    ),
    Case(
        collection=C.EVAL_RUNS,
        repo_cls=EvalRunRepository,
        doc=lambda tag: {"name": f"run-{tag}", "suite": "finqa", "status": "done"},
        probes=[lambda r, own, other, a, b: r.list_recent(own, limit=10)],
    ),
    Case(
        collection=C.EVAL_ITEMS,
        repo_cls=EvalItemRepository,
        doc=lambda tag: {"run_id": f"run-{tag}", "question": f"q-{tag}", "pass": False},
        probes=[
            lambda r, own, other, a, b: r.list_for_run(own, b["run_id"]),
            lambda r, own, other, a, b: r.list_for_run(own, a["run_id"]),
        ],
    ),
    Case(
        collection=C.AUDIT_LOG,
        repo_cls=AuditLogRepository,
        doc=lambda tag: {
            "action": f"action-{tag}",
            "resource": "document",
            "resource_id": f"doc-{tag}",
            "user_id": f"user-{tag}",
            "detail": {},
        },
        probes=[
            lambda r, own, other, a, b: r.list_recent(own, limit=10),
            lambda r, own, other, a, b: r.record(own, action="read",
                                                 resource="document",
                                                 resource_id=b["resource_id"]),
        ],
    ),
    Case(
        collection=C.FEEDBACK,
        repo_cls=FeedbackRepository,
        doc=lambda tag: {"message_id": f"msg-{tag}", "rating": 1, "note": f"n-{tag}"},
        probes=[
            lambda r, own, other, a, b: r.list_for_message(own, b["message_id"]),
            lambda r, own, other, a, b: r.list_for_message(own, a["message_id"]),
        ],
    ),
]

CASE_IDS = [c.collection for c in CASES]
case = pytest.param


def seed(case_obj: Case, ctx_a: TenantContext, ctx_b: TenantContext, run):
    """Insert one document per tenant (plus B's extras); return (doc_a, doc_b)."""
    repo = case_obj.repo_cls()
    a = run(repo.insert_one(ctx_a, case_obj.doc("A")))
    b = run(repo.insert_one(ctx_b, case_obj.doc("B")))
    for extra in case_obj.b_only:
        run(repo.insert_one(ctx_b, extra))
    return a, b


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_obj", CASES, ids=CASE_IDS)
class TestReadMatrix:
    """No read path on any tenant-owned collection may return foreign rows."""

    def test_foreign_id_is_invisible(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        assert run(repo.get_by_id(ctx_a, b["id"])) is None
        assert run(repo.exists(ctx_a, {"id": b["id"]})) is False
        assert run(repo.find_one(ctx_a, {"id": b["id"]})) is None

    def test_unfiltered_reads_are_scoped(self, case_obj, db, run, ctx_a, ctx_b):
        seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        many = run(repo.find_many(ctx_a, {}))
        assert len(many) == 1
        assert many[0]["tenant_id"] == ctx_a.tenant_id
        assert run(repo.count(ctx_a, {})) == 1

        async def _stream():
            return [doc async for doc in repo.stream(ctx_a, {})]

        streamed = run(_stream())
        assert [d["tenant_id"] for d in streamed] == [ctx_a.tenant_id]

    def test_caller_supplied_tenant_filter_cannot_widen(
        self, case_obj, db, run, ctx_a, ctx_b
    ):
        """A query that names another tenant is overridden by the context."""
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        found = run(repo.find_one(ctx_a, {"tenant_id": ctx_b.tenant_id}))
        assert found is not None
        assert found["tenant_id"] == ctx_a.tenant_id

        negated = run(repo.find_many(ctx_a, {"tenant_id": {"$ne": ctx_a.tenant_id}}))
        assert [d["tenant_id"] for d in negated] == [ctx_a.tenant_id]

    def test_id_or_injection_returns_only_own(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        rows = run(repo.find_many(
            ctx_a, {"$or": [{"id": a["id"]}, {"id": b["id"]}]}))
        assert {d["id"] for d in rows} == {a["id"]}

    def test_projection_read_is_scoped(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        assert run(repo.find_one(ctx_a, {"id": b["id"]},
                                 projection={"id": 1})) is None

    def test_aggregate_cannot_span_tenants(self, case_obj, db, run, ctx_a, ctx_b):
        seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        grouped = run(repo.aggregate(
            ctx_a, [{"$group": {"_id": "$tenant_id", "n": {"$sum": 1}}}]))
        assert {g["_id"] for g in grouped} == {ctx_a.tenant_id}

    def test_domain_methods_never_return_foreign_rows(
        self, case_obj, db, run, ctx_a, ctx_b
    ):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        for probe in case_obj.probes:
            result = run(probe(repo, ctx_a, ctx_b, a, b))
            if isinstance(result, int):  # a mutation probe; write matrix covers it
                continue
            for record in _as_records(result):
                assert record.get("tenant_id") == ctx_a.tenant_id, (
                    f"{case_obj.collection}: probe leaked tenant "
                    f"{record.get('tenant_id')!r} into {ctx_a.tenant_id!r}"
                )


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_obj", CASES, ids=CASE_IDS)
class TestWriteMatrix:
    def test_foreign_id_cannot_be_updated(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        assert run(repo.update_by_id(ctx_a, b["id"], {"hardened": True})) == 0
        assert run(repo.get_by_id(ctx_b, b["id"])).get("hardened") is None

    def test_foreign_id_cannot_be_deleted(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        assert run(repo.delete_by_id(ctx_a, b["id"])) == 0
        assert run(repo.get_by_id(ctx_b, b["id"])) is not None

    def test_bulk_writes_touch_only_own_rows(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        assert run(repo.update_many(ctx_a, {}, {"hardened": True})) >= 1
        assert run(repo.get_by_id(ctx_b, b["id"])).get("hardened") is None
        assert run(repo.delete_many(ctx_a, {})) == 1
        assert run(repo.get_by_id(ctx_b, b["id"])) is not None

    def test_insert_cannot_claim_another_tenant(self, case_obj, db, run, ctx_a, ctx_b):
        repo = case_obj.repo_cls()
        doc = dict(case_obj.doc("SNEAK"))
        doc["tenant_id"] = ctx_b.tenant_id
        inserted = run(repo.insert_one(ctx_a, doc))
        assert inserted["tenant_id"] == ctx_a.tenant_id
        assert run(repo.find_one(ctx_b, {"id": inserted["id"]})) is None

    def test_insert_many_stamps_every_row(self, case_obj, db, run, ctx_a, ctx_b):
        repo = case_obj.repo_cls()
        docs = run(repo.insert_many(ctx_a, [case_obj.doc("M1"),
                                           case_obj.doc("M2")]))
        assert {d["tenant_id"] for d in docs} == {ctx_a.tenant_id}
        assert run(repo.count(ctx_b, {})) == 0

    def test_ownership_cannot_be_transferred(self, case_obj, db, run, ctx_a, ctx_b):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        with pytest.raises(TenantContextError):
            run(repo.update_by_id(ctx_a, a["id"], {"tenant_id": ctx_b.tenant_id}))
        with pytest.raises(TenantContextError):
            run(repo.update_by_id(ctx_a, a["id"],
                                  {"$set": {"tenant_id": ctx_b.tenant_id}}))

    def test_update_with_foreign_query_filter_is_a_noop(
        self, case_obj, db, run, ctx_a, ctx_b
    ):
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        modified = run(repo.update_one(ctx_a, {"id": b["id"]}, {"hardened": True}))
        assert modified == 0

    def test_domain_mutations_cannot_reach_foreign_rows(
        self, case_obj, db, run, ctx_a, ctx_b
    ):
        """set_status / set_feedback / record against a foreign id."""
        a, b = seed(case_obj, ctx_a, ctx_b, run)
        repo = case_obj.repo_cls()
        for probe in case_obj.probes:
            result = run(probe(repo, ctx_a, ctx_b, a, b))
            if isinstance(result, int):
                assert result == 0, (
                    f"{case_obj.collection}: mutation probe changed "
                    f"{result} row(s) as {ctx_a.tenant_id}"
                )
        b_after = run(repo.get_by_id(ctx_b, b["id"]))
        assert b_after.get("status") == b.get("status")
        assert b_after.get("feedback") is None
        assert b_after.get("hardened") is None


# ---------------------------------------------------------------------------
# Context enforcement, uniformly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_obj", CASES, ids=CASE_IDS)
class TestContextMatrix:
    def test_every_operation_requires_a_context(self, case_obj, db, run):
        repo = case_obj.repo_cls()
        cases = [
            ("insert_one", repo.insert_one(None, case_obj.doc("X"))),
            ("find_one", repo.find_one(None)),
            ("find_many", repo.find_many(None, {})),
            ("count", repo.count(None, {})),
            ("aggregate", repo.aggregate(None, [{"$match": {}}])),
            ("update_by_id", repo.update_by_id(None, "x", {"a": 1})),
            ("delete_by_id", repo.delete_by_id(None, "x")),
            ("get_by_id", repo.get_by_id(None, "x")),
        ]
        for label, coro in cases:
            with pytest.raises(TenantContextError):
                run(coro)

    def test_stream_rejects_a_missing_context_before_iterating(
        self, case_obj, db, run
    ):
        repo = case_obj.repo_cls()
        with pytest.raises(TenantContextError):
            repo.stream(None, {})


# ---------------------------------------------------------------------------
# Meta: coverage itself is under test
# ---------------------------------------------------------------------------


def test_every_tenant_owned_collection_has_a_matrix_case():
    """Adding a tenant-owned collection without isolation coverage fails here."""
    covered = {c.collection for c in CASES}
    assert covered == set(C.TENANT_OWNED), (
        f"missing={sorted(set(C.TENANT_OWNED) - covered)} "
        f"unknown={sorted(covered - set(C.TENANT_OWNED))}"
    )


def test_every_repository_subclass_targets_a_tenant_owned_collection():
    import app.db.mongodb.repositories as repos

    concrete = {
        cls for cls in vars(repos).values()
        if isinstance(cls, type) and cls.__name__.endswith("Repository")
        and cls.__name__ not in {"BaseRepository", "_UnscopedRepository"}
    }
    scoped = {cls.collection_name for cls in concrete
              if issubclass(cls, BaseRepository)}
    unscoped = {cls.collection_name for cls in concrete
                if not issubclass(cls, BaseRepository)}
    assert scoped == set(C.TENANT_OWNED)
    # tenants is the one sanctioned exception: it *is* the tenancy root.
    assert unscoped == {C.TENANTS}


def test_matrix_cases_use_the_matching_repository_class():
    import app.db.mongodb.repositories as repos

    by_collection = {
        cls.collection_name: cls
        for cls in vars(repos).values()
        if isinstance(cls, type) and issubclass(cls, BaseRepository)
        and cls is not BaseRepository
    }
    for case_obj in CASES:
        assert by_collection[case_obj.collection] is case_obj.repo_cls, (
            f"{case_obj.collection} is covered by the wrong repository"
        )


class TestTenantRepositoryIsTheOneUnscopedException:
    """`tenants` has no tenant_id, so it cannot be repository-guarded.

    The isolation model therefore rests on it never being reachable from a
    tenant-facing route. These tests pin that shape rather than pretend to
    enforce it.
    """

    def test_it_is_not_a_base_repository(self):
        assert not issubclass(TenantRepository, BaseRepository)
        assert TenantRepository.collection_name == C.TENANTS

    def test_it_accepts_no_tenant_context(self, db, run):
        repo = TenantRepository()
        with pytest.raises(TypeError):
            repo.insert_one(TenantContext("tenant-AAA"), {"name": "x"})  # type: ignore[call-arg]

    def test_tenant_documents_have_no_tenant_id(self, db, run, ctx_a):
        repo = TenantRepository()
        created = run(repo.insert_one({"name": "tenant-AAA", "plan": "core"}))
        assert "tenant_id" not in created
        found = run(repo.get_by_name("tenant-AAA"))
        assert found["name"] == "tenant-AAA"

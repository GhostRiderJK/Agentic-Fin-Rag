"""Tenant isolation through the real HTTP surface.

``test_mongo_isolation_matrix.py`` proves the repository layer scopes every
query. This file proves the *routes* do, because a router that looks a row up
by id without joining it back to the caller's tenant leaks even when the
repository underneath is correct.

Two tenants are built through the public API (tenant -> admin user ->
workspace), each seeded with its own document, conversation, message and job,
and then every route that takes an id is driven with the *other* tenant's
credentials.

Routes that fail closed are asserted directly. Routes that leak are marked
``xfail(strict=True)``: the assertion states the required behaviour, the marker
records that the route does not meet it yet, and strictness makes the suite red
the moment someone fixes it, so the list of known holes cannot drift out of
date. Each marker names the item that closes it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.base import db_scope, get_sessionmaker, init_db
from app.db.models import (
    Conversation,
    EvalRun,
    Job,
    JobStatus,
    Message,
    MessageRole,
    Tenant,
    User,
)
from app.db.mongodb.repositories import DocumentRepository
from app.db.mongodb.runtime import run_sync
from app.db.mongodb.tenant_context import TenantContext
from mongo_support import app_db  # noqa: F401  (fixture used by name)


@pytest.fixture()
def client(app_db):  # noqa: F811 - documents live in the throwaway database
    init_db()
    from app.main import app

    return TestClient(app)


@dataclass(frozen=True)
class Actor:
    tenant_id: str
    user_id: str
    api_key: str
    workspace_id: str

    @property
    def headers(self) -> dict:
        return {"X-API-Key": self.api_key}


def _tag() -> str:
    """Everything is suffixed so one session-wide SQLite file can hold the
    rows of every test without them looking like each other."""
    return uuid.uuid4().hex[:10]


def _make_actor(client, label: str) -> Actor:
    suffix = _tag()
    tenant = client.post("/auth/tenants", json={"name": f"Firm {label} {suffix}"})
    assert tenant.status_code == 200, tenant.text
    tenant_id = tenant.json()["id"]

    user = client.post(
        "/auth/users",
        json={
            "email": f"admin-{label}-{suffix}@x.test",
            "name": f"Admin {label}",
            "tenant_id": tenant_id,
            "role": "admin",
        },
    )
    assert user.status_code == 200, user.text
    ws = client.post(
        "/workspaces",
        json={"name": f"WS {label} {suffix}"},
        headers={"X-API-Key": user.json()["api_key"]},
    )
    assert ws.status_code == 200, ws.text
    return Actor(
        tenant_id=tenant_id,
        user_id=user.json()["id"],
        api_key=user.json()["api_key"],
        workspace_id=ws.json()["id"],
    )


@pytest.fixture()
def actors(client):
    return _make_actor(client, "A"), _make_actor(client, "B")


@pytest.fixture()
def seeded(actors):
    """Give each tenant a document, conversation, message and job.

    Seeded through the ORM and the repository, not the API: the point is to
    have foreign rows to reach for, and doing it through routes would test the
    same code path twice.
    """
    out: dict[str, dict[str, str]] = {}
    for label, actor in (("A", actors[0]), ("B", actors[1])):
        suffix = _tag()
        doc = run_sync(DocumentRepository().insert_one(
            TenantContext(tenant_id=actor.tenant_id),
            {
                "workspace_id": actor.workspace_id,
                "title": f"Annual Report {label}",
                "filename": f"ar-{label}-{suffix}.txt",
                "doc_type": "annual_report",
                "company": f"COMPANY{label}",
                "ticker": f"CO{label}",
                "year": 2025,
                "quarter": "",
                "section": "",
                "status": "active",
                "parse_meta": {},
            },
        ))
        with db_scope() as db:
            conv = Conversation(
                workspace_id=actor.workspace_id,
                user_id=actor.user_id,
                title=f"conv {label}",
            )
            db.add(conv)
            db.flush()
            msg = Message(
                conversation_id=conv.id,
                role=MessageRole.assistant,
                content=f"answer for {label}",
            )
            db.add(msg)
            db.flush()
            job = Job(
                tenant_id=actor.tenant_id,
                workspace_id=actor.workspace_id,
                kind="report",
                status=JobStatus.queued,
            )
            db.add(job)
            db.flush()
            out[label] = {
                "document_id": doc["id"],
                "tenant_id": actor.tenant_id,
                "conversation_id": conv.id,
                "message_id": msg.id,
                "job_id": job.id,
                "suffix": suffix,
            }
    return out


def _snapshot(obj_type, obj_id: str) -> dict | None:
    """Read a row back in a fresh session, detached from any identity map.

    The route under test has its own session, so this proves what actually
    landed in the database rather than what an ORM object still remembers.
    """
    db = get_sessionmaker()()
    try:
        row = db.get(obj_type, obj_id)
        if row is None:
            return None
        db.expunge(row)
        return {c.name: getattr(row, c.name) for c in row.__table__.columns}
    finally:
        db.close()


def _document_snapshot(seeded: dict, label: str) -> dict | None:
    """The Mongo equivalent of :func:`_snapshot` for a document row.

    Read with the owning tenant's own context, so a leak would show up as the
    row changing rather than as a scoping accident in the reader.
    """
    return run_sync(DocumentRepository().get_by_id(
        TenantContext(tenant_id=seeded[label]["tenant_id"]),
        seeded[label]["document_id"],
    ))


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


class TestDocumentRoutes:
    def test_list_documents_returns_only_the_callers(self, client, actors, seeded):
        r = client.get(
            f"/workspaces/{actors[0].workspace_id}/documents",
            headers=actors[0].headers,
        )
        assert r.status_code == 200
        assert [d["id"] for d in r.json()] == [seeded["A"]["document_id"]]

    def test_foreign_workspace_path_is_not_found(self, client, actors, seeded):
        """Tenant A asking under tenant B's workspace id never learns it exists."""
        r = client.get(
            f"/workspaces/{actors[1].workspace_id}/documents",
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_foreign_document_id_under_own_workspace_is_not_found(
        self, client, actors, seeded
    ):
        r = client.get(
            f"/workspaces/{actors[0].workspace_id}/documents/"
            f"{seeded['B']['document_id']}",
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_patch_cannot_reach_a_foreign_document(self, client, actors, seeded):
        r = client.patch(
            f"/workspaces/{actors[0].workspace_id}/documents/"
            f"{seeded['B']['document_id']}",
            json={"title": "stolen"},
            headers=actors[0].headers,
        )
        assert r.status_code == 404
        assert _document_snapshot(seeded, "B")["title"] == "Annual Report B"

    def test_delete_cannot_reach_a_foreign_document(self, client, actors, seeded):
        r = client.delete(
            f"/workspaces/{actors[0].workspace_id}/documents/"
            f"{seeded['B']['document_id']}",
            headers=actors[0].headers,
        )
        assert r.status_code == 404
        assert _document_snapshot(seeded, "B") is not None

    def test_upload_to_a_foreign_workspace_is_rejected_before_writing(
        self, client, actors
    ):
        r = client.post(
            f"/workspaces/{actors[1].workspace_id}/documents",
            headers=actors[0].headers,
            data={"company": "ACME", "async_ingest": "false"},
            files={"file": ("x.txt", b"ACME revenue was $1 in 2025.", "text/plain")},
        )
        assert r.status_code == 404
        # Read with tenant B's own context: the workspace is empty because
        # nothing was written, not because the reader was scoped away.
        rows = run_sync(DocumentRepository().list_for_workspace(
            TenantContext(tenant_id=actors[1].tenant_id),
            actors[1].workspace_id,
        ))
        assert rows == []

    def test_own_document_is_patchable(self, client, actors, seeded):
        r = client.patch(
            f"/workspaces/{actors[0].workspace_id}/documents/"
            f"{seeded['A']['document_id']}",
            json={"status": "archived"},
            headers=actors[0].headers,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "archived"


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


class TestChatRoutes:
    def test_chat_in_a_foreign_workspace_is_not_found(self, client, actors, seeded):
        r = client.post(
            "/chat",
            json={"question": "revenue?", "workspace_id": actors[1].workspace_id},
            headers=actors[0].headers,
        )
        assert r.status_code == 404
        assert _snapshot(Message, seeded["B"]["message_id"])["content"] == "answer for B"

    def test_stream_in_a_foreign_workspace_is_not_found(self, client, actors):
        r = client.post(
            "/chat/stream",
            json={"question": "revenue?", "workspace_id": actors[1].workspace_id},
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_conversations_of_a_foreign_workspace_are_not_found(
        self, client, actors, seeded
    ):
        r = client.get(
            f"/workspaces/{actors[1].workspace_id}/conversations",
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_foreign_conversation_under_own_workspace_is_not_found(
        self, client, actors, seeded
    ):
        r = client.get(
            f"/workspaces/{actors[0].workspace_id}/conversations/"
            f"{seeded['B']['conversation_id']}/messages",
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_own_conversation_history_is_scoped_to_own_messages(
        self, client, actors, seeded
    ):
        r = client.get(
            f"/workspaces/{actors[0].workspace_id}/conversations/"
            f"{seeded['A']['conversation_id']}/messages",
            headers=actors[0].headers,
        )
        assert r.status_code == 200
        assert [m["id"] for m in r.json()] == [seeded["A"]["message_id"]]

    @pytest.mark.xfail(
        reason=(
            "POST /messages/{message_id}/feedback loads the message by id with "
            "no workspace or tenant check (app/api/chat.py), so any authenticated "
            "user can rate another tenant's answer. Closed by the chat migration "
            "in item (f), where MessageRepository scopes the lookup."
        ),
        strict=True,
    )
    def test_feedback_cannot_reach_a_foreign_message(self, client, actors, seeded):
        r = client.post(
            f"/messages/{seeded['B']['message_id']}/feedback",
            json={"rating": 1, "note": "from tenant A"},
            headers=actors[0].headers,
        )
        assert r.status_code == 404
        assert _snapshot(Message, seeded["B"]["message_id"])["feedback"] is None


# ---------------------------------------------------------------------------
# jobs and reports
# ---------------------------------------------------------------------------


class TestJobRoutes:
    def test_report_in_a_foreign_workspace_is_not_found(self, client, actors):
        r = client.post(
            "/reports",
            json={"workspace_id": actors[1].workspace_id, "company": "ACME"},
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_foreign_job_detail_is_not_found(self, client, actors, seeded):
        r = client.get(
            f"/jobs/{seeded['B']['job_id']}", headers=actors[0].headers
        )
        assert r.status_code == 404

    def test_job_list_excludes_foreign_jobs(self, client, actors, seeded):
        r = client.get("/jobs", headers=actors[0].headers)
        assert r.status_code == 200
        assert [j["id"] for j in r.json()] == [seeded["A"]["job_id"]]

    def test_job_list_filter_cannot_widen_scope(self, client, actors, seeded):
        r = client.get(
            "/jobs",
            params={"workspace_id": actors[1].workspace_id},
            headers=actors[0].headers,
        )
        assert r.status_code == 200
        assert r.json() == []


# ---------------------------------------------------------------------------
# admin / eval
# ---------------------------------------------------------------------------


class TestAdminRoutes:
    def test_eval_in_a_foreign_workspace_is_not_found(self, client, actors):
        r = client.post(
            "/admin/eval/run",
            json={"workspace_id": actors[1].workspace_id},
            headers=actors[0].headers,
        )
        assert r.status_code == 404

    def test_eval_suites_requires_credentials(self, client):
        assert client.get("/admin/eval/suites").status_code == 401

    def test_eval_suites_are_addressed_by_name(self, client, actors):
        r = client.get("/admin/eval/suites", headers=actors[0].headers)
        assert r.status_code == 200
        assert {"id": "packaged", "packaged": True, "n_items": 7} in r.json()

    def test_eval_suite_name_cannot_address_a_file_outside_its_directory(
        self, client, actors
    ):
        """The loader reads whatever path it is given, so the route must not
        accept one — a suite is a name inside the suites directory or nothing."""
        for hostile in ["../../app/.env", "a/b", "C:/Windows/win.ini"]:
            r = client.post(
                "/admin/eval/run",
                json={"workspace_id": actors[0].workspace_id, "suite": hostile},
                headers=actors[0].headers,
            )
            assert r.status_code == 422, (hostile, r.status_code, r.text)

    @pytest.mark.xfail(
        reason=(
            "eval_runs has no tenant column in the SQL schema, so "
            "GET /admin/eval/runs returns every tenant's runs to any admin "
            "(app/api/admin.py). Closed by item (f): eval_runs is tenant-owned "
            "in MongoDB and the route reads through EvalRunRepository."
        ),
        strict=True,
    )
    def test_eval_run_list_excludes_foreign_runs(self, client, actors, seeded):
        suite = f"tenant-B-{seeded['B']['suffix']}"
        with db_scope() as db:
            db.add(EvalRun(suite=suite, n_items=1))
        r = client.get("/admin/eval/runs", headers=actors[0].headers)
        assert r.status_code == 200
        assert suite not in {row["suite"] for row in r.json()}

    def test_eval_run_is_queued_and_names_no_run_row(self, client, actors, seeded):
        """POST /admin/eval/run used to answer with the newest eval_runs row
        anywhere in the database, which handed tenant A another tenant's run id.

        The route now only queues a job, so there is no run row in its response
        to leak. The queue is drained here because the worker writes to this
        test's database and must not outlive it.
        """
        from app.jobs.queue import wait_for_jobs

        with db_scope() as db:
            foreign = EvalRun(
                suite=f"tenant-B-{seeded['B']['suffix']}",
                n_items=0,
                created_at=datetime.now(timezone.utc) + timedelta(days=365),
            )
            db.add(foreign)
            db.flush()
            foreign_id = foreign.id

        r = client.post(
            "/admin/eval/run",
            json={"workspace_id": actors[0].workspace_id},
            headers=actors[0].headers,
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert {"job_id", "status", "suite", "workspace_id"} == set(body)
        assert body["job_id"] != foreign_id

        wait_for_jobs(timeout=180)
        with db_scope() as db:
            job = db.get(Job, body["job_id"])
            assert job.kind == "eval"
            assert job.status == JobStatus.succeeded, job.error
            assert job.result["metrics"], job.result


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


class TestAuthRoutes:
    def test_admin_cannot_add_a_user_to_another_tenant(self, client, actors):
        r = client.post(
            "/auth/users",
            json={
                "email": f"squatter-{_tag()}@x.test",
                "tenant_id": actors[1].tenant_id,
                "role": "admin",
            },
            headers=actors[0].headers,
        )
        assert r.status_code == 403

    def test_a_user_cannot_be_created_without_credentials_or_a_tenant(
        self, client, actors
    ):
        r = client.post("/auth/users", json={"email": f"anon-{_tag()}@x.test"})
        assert r.status_code == 401

    @pytest.mark.xfail(
        reason=(
            "POST /auth/tenants has no authentication at all (app/api/auth.py:36): "
            "anyone can create tenants. Closed by item (h) with a documented "
            "first-tenant bootstrap path."
        ),
        strict=True,
    )
    def test_tenant_creation_requires_credentials(self, client, actors):
        r = client.post("/auth/tenants", json={"name": f"Nobody's firm {_tag()}"})
        assert r.status_code == 401

    @pytest.mark.xfail(
        reason=(
            "POST /auth/users trusts the caller-supplied tenant_id whenever the "
            "tenant has no users yet (app/api/auth.py:58), so an anonymous "
            "request can make itself admin of any tenant that has not been "
            "claimed. Closed by item (h)."
        ),
        strict=True,
    )
    def test_anonymous_user_cannot_claim_an_unclaimed_tenant(self, client, actors):
        # The tenant is seeded through the ORM so this test does not also
        # depend on POST /auth/tenants, which item (h) is going to protect.
        with db_scope() as db:
            unclaimed = Tenant(name=f"Unclaimed {_tag()}")
            db.add(unclaimed)
            db.flush()
            unclaimed_id = unclaimed.id
        r = client.post(
            "/auth/users",
            json={
                "email": f"claimed-{_tag()}@x.test",
                "tenant_id": unclaimed_id,
                "role": "admin",
            },
        )
        assert r.status_code == 401
        with db_scope() as db:
            assert db.scalar(select(User).where(User.tenant_id == unclaimed_id)) is None

    def test_a_user_can_be_created_with_a_password_and_logs_in(self, client, actors):
        """Password auth was dead before passlib was dropped: ``hash_password``
        raised on its first call, so no user could hold a password and
        ``/auth/login`` could not succeed for anybody. API keys were the only
        working credential, which is what kept this invisible."""
        email = f"pw-{_tag()}@x.test"
        created = client.post(
            "/auth/users",
            json={
                "email": email,
                "tenant_id": actors[0].tenant_id,
                "role": "analyst",
                "password": "correct horse battery staple",
            },
            headers=actors[0].headers,
        )
        assert created.status_code == 200, created.text

        logged_in = client.post(
            "/auth/login",
            json={"email": email, "password": "correct horse battery staple"},
        )
        assert logged_in.status_code == 200, logged_in.text
        token = logged_in.json()["access_token"]

        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["tenant_id"] == actors[0].tenant_id

        wrong = client.post("/auth/login", json={"email": email, "password": "nope"})
        assert wrong.status_code == 401

    @pytest.mark.xfail(
        reason=(
            "Login answers with the full UserOut, which carries the API key "
            "(app/api/schemas.py:39). Checked on the schema because the password "
            "path above is currently unusable. Item (h) removes the field."
        ),
        strict=True,
    )
    def test_the_user_schema_does_not_expose_the_api_key(self):
        from app.api.schemas import TokenOut, UserOut

        assert "api_key" not in UserOut.model_fields
        assert TokenOut.model_fields["user"].annotation is UserOut

    @pytest.mark.xfail(
        reason=(
            "API keys are stored in plaintext and returned by every user "
            "response, including GET /auth/me (app/api/auth.py:100). Closed by "
            "item (h): hash at rest, reveal once."
        ),
        strict=True,
    )
    def test_me_does_not_reveal_the_api_key(self, client, actors):
        r = client.get("/auth/me", headers=actors[0].headers)
        assert r.status_code == 200
        assert "api_key" not in r.json()


# ---------------------------------------------------------------------------
# every route fails closed without credentials
# ---------------------------------------------------------------------------


def test_every_data_route_requires_credentials(client, actors, seeded):
    aw = actors[0].workspace_id
    doc, conv, msg, job = (
        seeded["A"]["document_id"],
        seeded["A"]["conversation_id"],
        seeded["A"]["message_id"],
        seeded["A"]["job_id"],
    )
    cases = [
        ("GET", "/workspaces", None),
        ("GET", f"/workspaces/{aw}/documents", None),
        ("GET", f"/workspaces/{aw}/documents/{doc}", None),
        ("PATCH", f"/workspaces/{aw}/documents/{doc}", {"title": "x"}),
        ("DELETE", f"/workspaces/{aw}/documents/{doc}", None),
        ("POST", "/chat", {"question": "q", "workspace_id": aw}),
        ("POST", "/chat/stream", {"question": "q", "workspace_id": aw}),
        ("GET", f"/workspaces/{aw}/conversations", None),
        ("GET", f"/workspaces/{aw}/conversations/{conv}/messages", None),
        ("POST", f"/messages/{msg}/feedback", {"rating": 1}),
        ("POST", "/reports", {"workspace_id": aw, "company": "ACME"}),
        ("GET", "/jobs", None),
        ("GET", f"/jobs/{job}", None),
        ("POST", "/admin/eval/run", {"workspace_id": aw}),
        ("GET", "/admin/eval/runs", None),
    ]
    leaks = []
    for method, path, body in cases:
        r = client.request(method, path, json=body)
        if r.status_code != 401:
            leaks.append(f"{method} {path} -> {r.status_code}")
    assert not leaks, "routes that answer without credentials: " + "; ".join(leaks)


def test_credentials_resolve_to_exactly_one_tenant(client, actors):
    """A key identifies one user in one tenant, and nothing else authenticates.

    If this ever maps a key to a tenant other than its owner's, every scoping
    test above becomes meaningless, so it is asserted directly.
    """
    for actor in actors:
        me = client.get("/auth/me", headers=actor.headers)
        assert me.status_code == 200
        assert me.json()["tenant_id"] == actor.tenant_id

    bogus = client.get("/workspaces", headers={"X-API-Key": "not-a-real-key"})
    assert bogus.status_code == 401
    assert client.get("/workspaces").status_code == 401


# ---------------------------------------------------------------------------
# coverage: no route may go unexamined
# ---------------------------------------------------------------------------

#: Every tenant-facing route, mapped to the case that checks its scoping.
#: A new route without a row here fails the meta-test below, so the battery
#: cannot silently fall behind the API.
COVERAGE: dict = {
    ("POST", "/auth/tenants"): "test_tenant_creation_requires_credentials",
    ("POST", "/auth/users"): "test_admin_cannot_add_a_user_to_another_tenant / "
                             "test_anonymous_user_cannot_claim_an_unclaimed_tenant",
    ("POST", "/auth/login"): "test_a_user_can_be_created_with_a_password_and_logs_in"
                             " / test_the_user_schema_does_not_expose_the_api_key",
    ("GET", "/auth/me"): "test_me_does_not_reveal_the_api_key",
    ("POST", "/workspaces"): "every actor is created through it; rows are stamped "
                             "with the caller's tenant",
    ("GET", "/workspaces"): "test_credentials_resolve_to_exactly_one_tenant",
    ("POST", "/workspaces/{workspace_id}/documents"):
        "test_upload_to_a_foreign_workspace_is_rejected_before_writing",
    ("GET", "/workspaces/{workspace_id}/documents"):
        "test_foreign_workspace_path_is_not_found / test_list_documents_returns_only_the_callers",
    ("GET", "/workspaces/{workspace_id}/documents/{document_id}"):
        "test_foreign_document_id_under_own_workspace_is_not_found",
    ("PATCH", "/workspaces/{workspace_id}/documents/{document_id}"):
        "test_patch_cannot_reach_a_foreign_document",
    ("DELETE", "/workspaces/{workspace_id}/documents/{document_id}"):
        "test_delete_cannot_reach_a_foreign_document",
    ("POST", "/chat"): "test_chat_in_a_foreign_workspace_is_not_found",
    ("POST", "/chat/stream"): "test_stream_in_a_foreign_workspace_is_not_found",
    ("GET", "/workspaces/{workspace_id}/conversations"):
        "test_conversations_of_a_foreign_workspace_are_not_found",
    ("GET", "/workspaces/{workspace_id}/conversations/{conversation_id}/messages"):
        "test_foreign_conversation_under_own_workspace_is_not_found",
    ("POST", "/messages/{message_id}/feedback"):
        "test_feedback_cannot_reach_a_foreign_message",
    ("POST", "/reports"): "test_report_in_a_foreign_workspace_is_not_found",
    ("GET", "/jobs"): "test_job_list_excludes_foreign_jobs / "
                      "test_job_list_filter_cannot_widen_scope",
    ("GET", "/jobs/{job_id}"): "test_foreign_job_detail_is_not_found",
    ("POST", "/admin/eval/run"): "test_eval_in_a_foreign_workspace_is_not_found / "
                                 "test_eval_suite_name_cannot_address_a_file_outside_its_directory / "
                                 "test_eval_run_is_queued_and_names_no_run_row",
    ("GET", "/admin/eval/runs"): "test_eval_run_list_excludes_foreign_runs",
    ("GET", "/admin/eval/suites"): "test_eval_suites_requires_credentials / "
                                   "test_eval_suites_are_addressed_by_name",
}

#: Routes that carry no tenant at all.
PUBLIC_ROUTES = {("GET", "/"), ("GET", "/health")}


def test_every_route_is_accounted_for():
    """The advertised API surface must equal the set of routes examined here.

    Read from the OpenAPI document rather than FastAPI's internal route list:
    this FastAPI keeps included routers behind lazy wrappers, and the spec is
    what a client can actually reach.
    """
    from app.main import app

    spec = app.openapi()
    live = {
        (method.upper(), path)
        for path, operations in spec["paths"].items()
        for method in operations
        if method != "head"
    }
    missing = sorted(live - set(COVERAGE) - PUBLIC_ROUTES)
    stale = sorted(set(COVERAGE) - live)
    assert not missing, f"routes with no isolation case: {missing}"
    assert not stale, f"COVERAGE rows for routes that no longer exist: {stale}"

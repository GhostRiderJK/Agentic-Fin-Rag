"""End-to-end smoke test exercising the full HTTP API.

Uses TestClient against the live FastAPI app with stub providers, exercising:
  - tenant/user creation + API-key auth
  - workspace creation
  - document upload + async ingest (waits for job)
  - chat Q&A with citations
  - eval run
"""

from __future__ import annotations

import time
import uuid

from fastapi.testclient import TestClient

from app.db.base import init_db
from mongo_support import app_db  # noqa: F401  (fixture used by name)


def test_full_flow(tmp_path, app_db):
    init_db()
    from app.main import app

    # The users table outlives one run (the SQLite file is session-wide, and
    # this test runs once per database backend), so the identity is unique.
    tag = uuid.uuid4().hex[:8]
    client = TestClient(app)

    # Tenant + user
    t = client.post("/auth/tenants", json={"name": f"Test Firm {tag}"})
    assert t.status_code == 200, t.text
    tenant_id = t.json()["id"]
    # No creator header → only allowed into a tenant with no users yet
    u = client.post("/auth/users", json={
        "email": f"a-{tag}@x.com", "name": "A", "tenant_id": tenant_id,
        # The eval routes below are admin-only, and the user route honours the
        # requested role instead of promoting the first user in a tenant.
        "role": "admin",
    })
    assert u.status_code == 200, u.text
    api_key = u.json()["api_key"]
    headers = {"X-API-Key": api_key}

    # Workspace
    ws = client.post("/workspaces", json={"name": f"WS {tag}"}, headers=headers)
    assert ws.status_code == 200, ws.text
    ws_id = ws.json()["id"]

    # Upload a TXT document (sync ingest)
    sample = (
        "ACME Annual Report 2025\n\n"
        "1. BUSINESS OVERVIEW\n\n"
        "ACME had revenue of $120 billion in 2025.\n\n"
        "2. RISK FACTORS\n\n"
        "Competition and supply chain risk.\n"
    )
    r = client.post(
        f"/workspaces/{ws_id}/documents",
        headers=headers,
        data={"company": "ACME", "year": "2025", "async_ingest": "false"},
        files={"file": ("acme.txt", sample.encode(), "text/plain")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["n_parents"] > 0
    document_id = r.json()["document_id"]

    # List documents
    docs = client.get(f"/workspaces/{ws_id}/documents", headers=headers)
    assert docs.status_code == 200
    assert len(docs.json()) >= 1
    listed = [d for d in docs.json() if d["id"] == document_id]
    assert len(listed) == 1, "the ingested document did not come back"
    assert listed[0]["company"] == "ACME"
    assert listed[0]["year"] == 2025
    assert listed[0]["status"] == "active"

    # The chunk rows the retriever will read
    from app.db.mongodb.repositories import ChunkRepository
    from app.db.mongodb.runtime import run_sync
    from app.db.mongodb.tenant_context import TenantContext

    ctx = TenantContext(tenant_id=tenant_id, workspace_id=ws_id)
    chunks = run_sync(ChunkRepository().find_searchable(ctx, [document_id]))
    assert chunks, "ingest wrote no searchable chunks"
    parent_ids = {c["parent_id"] for c in chunks}
    parents = run_sync(ChunkRepository().get_many_by_ids(ctx, sorted(parent_ids)))
    assert len(parents) == len(parent_ids)

    # Chat
    chat = client.post("/chat", json={
        "question": "What was ACME's revenue in 2025?",
        "workspace_id": ws_id,
    }, headers=headers)
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert "answer" in body
    assert "citations" in body
    assert "plan" in body
    assert body["plan"]["companies"] == ["ACME"]
    assert body["plan"]["years"] == [2025]
    # Retrieval found the Mongo-stored chunks.
    assert body["retrieval"]["n_chunks"] > 0
    # The stub generator makes no claims, so it cites nothing: the API only
    # returns sources the answer actually references with an [n] marker.
    assert body["citations"] == [], body
    # Scope the retrieved evidence to the ingested document through the
    # admin-only trace instead of loosening the citation policy.
    traced = client.post("/chat", json={
        "question": "What was ACME's revenue in 2025?",
        "workspace_id": ws_id,
        "include_evidence_trace": True,
    }, headers=headers)
    assert traced.status_code == 200, traced.text
    trace = traced.json()["retrieval"]["candidates"]
    assert trace, traced.json()["retrieval"]
    assert {c["document_id"] for c in trace} == {document_id}

    # Conversation appears in history
    convs = client.get(f"/workspaces/{ws_id}/conversations", headers=headers)
    assert convs.status_code == 200
    assert len(convs.json()) >= 1

    # Eval: the suite is queued, so the request returns immediately and the
    # caller polls the job it was handed.
    suites = client.get("/admin/eval/suites", headers=headers)
    assert suites.status_code == 200, suites.text
    assert {"id": "packaged", "packaged": True, "n_items": 7} in suites.json()

    queued = client.post("/admin/eval/run",
                         json={"workspace_id": ws_id}, headers=headers)
    assert queued.status_code == 202, queued.text
    job_id = queued.json()["job_id"]
    assert queued.json()["suite"] == "packaged"

    deadline = time.time() + 120
    job: dict = {}
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    assert job["status"] == "succeeded", job
    assert job["kind"] == "eval"
    assert job["result"]["n_items"] == 7
    assert "mean_faithfulness" in job["result"]["metrics"]

    runs = client.get("/admin/eval/runs", headers=headers)
    assert runs.status_code == 200
    assert any(r["suite"] == "packaged" for r in runs.json()), runs.json()

    # Deleting the document takes its searchable chunks with it
    dele = client.delete(
        f"/workspaces/{ws_id}/documents/{document_id}", headers=headers)
    assert dele.status_code == 200, dele.text
    assert run_sync(ChunkRepository().find_searchable(ctx, [document_id])) == []
    assert client.get(
        f"/workspaces/{ws_id}/documents/{document_id}",
        headers=headers).status_code == 404


def test_health():
    from app.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

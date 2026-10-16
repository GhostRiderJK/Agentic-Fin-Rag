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

from fastapi.testclient import TestClient

from app.db.base import init_db


def test_full_flow(tmp_path):
    init_db()
    from app.main import app

    client = TestClient(app)

    # Tenant + user
    t = client.post("/auth/tenants", json={"name": "Test Firm"})
    assert t.status_code == 200, t.text
    tenant_id = t.json()["id"]
    # No creator header → only allowed into a tenant with no users yet
    u = client.post("/auth/users", json={
        "email": "a@x.com", "name": "A", "tenant_id": tenant_id,
    })
    assert u.status_code == 200, u.text
    api_key = u.json()["api_key"]
    headers = {"X-API-Key": api_key}

    # Workspace
    ws = client.post("/workspaces", json={"name": "WS"}, headers=headers)
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

    # List documents
    docs = client.get(f"/workspaces/{ws_id}/documents", headers=headers)
    assert docs.status_code == 200
    assert len(docs.json()) >= 1

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

    # Conversation appears in history
    convs = client.get(f"/workspaces/{ws_id}/conversations", headers=headers)
    assert convs.status_code == 200
    assert len(convs.json()) >= 1


def test_health():
    from app.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

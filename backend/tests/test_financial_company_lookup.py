"""Reaching extracted figures by company name.

Two gaps that made the structured-data tools answer "no figures" for a corpus
that has them:

* ``find_by_company`` only matched rows whose name *contains* the typed term,
  so a shorter stored name ("Apple") could not answer a longer typed one
  ("Apple Inc.") — the case its own docstring described;
* ``PATCH /workspaces/{ws}/documents/{id}`` rewrote the document but left its
  extracted figures carrying the old (usually empty) company, so correcting
  metadata changed nothing the tools could find.
"""

from __future__ import annotations

import os
import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.db.mongodb.repositories import (
    DocumentRepository,
    StructuredFinancialRepository,
)
from app.db.mongodb.runtime import run_sync
from app.db.mongodb.tenant_context import TenantContext

if not os.path.exists(os.path.join(os.path.dirname(__file__), "__init__.py")):
    from mongo_support import app_db, ctx_a, db, run  # noqa: F401
else:
    from tests.mongo_support import app_db, ctx_a, db, run  # noqa: F401


def _row(company: str, *, ws: str = "ws-A", year: int = 2024) -> dict:
    return {
        "workspace_id": ws, "document_id": f"doc-{uuid.uuid4().hex[:6]}",
        "company": company, "ticker": "", "metric": "revenue",
        "fiscal_year": year, "quarter": "", "value": Decimal("1000.0"),
        "unit": "million", "currency": "SAR", "confidence": Decimal("0.8"),
    }


def test_company_lookup_matches_in_both_directions(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    run(repo.insert_many(ctx_a, [
        _row("Apple", ws="ws-1"),
        _row("Stellante Holdings", ws="ws-1"),
        _row("", ws="ws-1"),
    ]))

    # the typed term is longer than the stored name
    by_short = run(repo.find_by_company(ctx_a, workspace_id="ws-1",
                                        company="Apple Inc."))
    assert [r["company"] for r in by_short] == ["Apple"]

    # the stored name is longer than the typed term
    by_long = run(repo.find_by_company(ctx_a, workspace_id="ws-1",
                                       company="Stellante"))
    assert [r["company"] for r in by_long] == ["Stellante Holdings"]

    # an unnamed row is a substring of every term; it must never be returned
    assert run(repo.find_by_company(ctx_a, workspace_id="ws-1",
                                    company="PIF")) == []
    # partial names still match the forward way, which is the old behaviour
    assert [r["company"] for r in run(
        repo.find_by_company(ctx_a, workspace_id="ws-1", company="pple"))] == ["Apple"]


def test_company_lookup_stays_inside_the_workspace(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    run(repo.insert_many(ctx_a, [_row("Apple", ws="ws-1"),
                                 _row("Apple", ws="ws-2")]))

    rows = run(repo.find_by_company(ctx_a, workspace_id="ws-1",
                                    company="Apple Inc."))
    assert [r["workspace_id"] for r in rows] == ["ws-1"]


def _actor(client: TestClient, label: str) -> dict:
    suffix = uuid.uuid4().hex[:8]
    tenant = client.post("/auth/tenants", json={"name": f"Firm {label} {suffix}"})
    assert tenant.status_code == 200, tenant.text
    user = client.post(
        "/auth/users",
        json={"email": f"admin-{label}-{suffix}@x.test", "name": f"A {label}",
              "tenant_id": tenant.json()["id"], "role": "admin"})
    assert user.status_code == 200, user.text
    ws = client.post("/workspaces", json={"name": f"WS {label} {suffix}"},
                     headers={"X-API-Key": user.json()["api_key"]})
    assert ws.status_code == 200, ws.text
    return {"api_key": user.json()["api_key"], "user_id": user.json()["id"],
            "tenant_id": tenant.json()["id"], "workspace_id": ws.json()["id"]}


@pytest.fixture()
def client(app_db):  # noqa: F811 - documents live in the throwaway database
    from app.db.base import init_db
    init_db()
    from app.main import app
    return TestClient(app)


def _seed(client: TestClient, label: str, company: str) -> tuple[dict, str]:
    """One document plus two of its extracted figures, in a workspace of its own."""
    actor = _actor(client, label)
    ctx = TenantContext(tenant_id=actor["tenant_id"], user_id=actor["user_id"])
    doc = run_sync(DocumentRepository().insert_one(ctx, {
        "workspace_id": actor["workspace_id"], "title": "Annual report",
        "filename": "ar-2024.txt", "doc_type": "annual_report",
        "company": company, "ticker": "", "year": 2024, "quarter": "",
        "section": "", "status": "active", "parse_meta": {},
    }))
    run_sync(StructuredFinancialRepository().insert_many(ctx, [
        {**_row(company, ws=actor["workspace_id"]), "document_id": doc["id"],
         "fiscal_year": year} for year in (2023, 2024)
    ]))
    return actor, doc["id"]


def test_patch_company_carries_over_to_the_extracted_figures(client):
    actor, document_id = _seed(client, "A", "")
    ctx = TenantContext(tenant_id=actor["tenant_id"], user_id=actor["user_id"])
    repo = StructuredFinancialRepository()

    res = client.patch(
        f"/workspaces/{actor['workspace_id']}/documents/{document_id}",
        headers={"X-API-Key": actor["api_key"]},
        json={"company": "Public Investment Fund"})
    assert res.status_code == 200, res.text
    assert res.json()["company"] == "Public Investment Fund"

    rows = run_sync(repo.find_by_company(
        ctx, workspace_id=actor["workspace_id"],
        company="Public Investment Fund"))
    assert {r["company"] for r in rows} == {"Public Investment Fund"}
    # A figure keeps its own period: a 2023 comparative inside the 2024
    # filing must not be relabelled by the document's metadata.
    assert {r["fiscal_year"] for r in rows} == {2023, 2024}


def test_patch_without_company_leaves_figures_untouched(client):
    actor, document_id = _seed(client, "B", "Legacy Name")
    ctx = TenantContext(tenant_id=actor["tenant_id"], user_id=actor["user_id"])

    res = client.patch(
        f"/workspaces/{actor['workspace_id']}/documents/{document_id}",
        headers={"X-API-Key": actor["api_key"]}, json={"title": "Renamed filing"})
    assert res.status_code == 200, res.text
    rows = run_sync(StructuredFinancialRepository().find_by_company(
        ctx, workspace_id=actor["workspace_id"], company="Legacy Name"))
    assert [r["company"] for r in rows] == ["Legacy Name", "Legacy Name"]


def test_a_blank_company_is_never_copied_onto_the_figures(client):
    """Clearing the document's name must not orphan its extracted figures.

    A row with ``company: ""`` answers no lookup: the reverse branch treats the
    empty string as contained in every term, so it is excluded, and the forward
    branch would match the whole workspace if it were ever used as a term.
    """
    actor, document_id = _seed(client, "C", "Legacy Name")
    ctx = TenantContext(tenant_id=actor["tenant_id"], user_id=actor["user_id"])

    res = client.patch(
        f"/workspaces/{actor['workspace_id']}/documents/{document_id}",
        headers={"X-API-Key": actor["api_key"]}, json={"company": "   "})
    assert res.status_code == 200, res.text
    rows = run_sync(StructuredFinancialRepository().find_by_company(
        ctx, workspace_id=actor["workspace_id"], company="Legacy Name"))
    assert [r["company"] for r in rows] == ["Legacy Name", "Legacy Name"]


def test_a_lookup_without_a_term_returns_nothing(db, run, ctx_a):
    repo = StructuredFinancialRepository()
    run_sync(repo.insert_one(ctx_a, _row("Apple", ws="ws-9")))
    for blank in ("", "   "):
        assert run_sync(repo.find_by_company(
            ctx_a, workspace_id="ws-9", company=blank)) == []

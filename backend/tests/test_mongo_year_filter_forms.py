"""Fiscal years are stored as both numbers and text, and filters must see both.

``documents.year`` written by the current upload route is an int; rows created by
earlier code (measured in the deployed corpus: 2 of 5 documents) are text. Mongo
compares strictly across BSON types, so a single-typed filter drops whichever
shape it does not name — silently, with no error and no empty-workspace warning.
Retrieval then reads the year back into an int-valued query plan, so the mismatch
also disables the year filter without any signal.

Every case here runs on mongomock-motor and against the real test server (see
``mongo_support``), because type comparison is exactly the kind of behaviour a
mock gets wrong.
"""

from __future__ import annotations

import os
from decimal import Decimal

if not os.path.exists(os.path.join(os.path.dirname(__file__), '__init__.py')):
    from mongo_support import ctx_a, db, run  # noqa: F401
else:
    from tests.mongo_support import ctx_a, db, run  # noqa: F401

from app.db.mongodb.repositories import (  # noqa: E402
    DocumentRepository,
    StructuredFinancialRepository,
)


def _doc(repo, ctx, run_, year, doc_id_title):
    return run_(repo.insert_one(ctx, {
        "workspace_id": "ws-years", "title": doc_id_title, "filename": "f.pdf",
        "doc_type": "annual_report", "company": "PIF", "ticker": "",
        "year": year, "quarter": "", "section": "", "status": "active",
        "parse_meta": {},
    }))


def test_an_int_year_filter_also_finds_the_text_year_documents(db, run, ctx_a):
    docs = DocumentRepository()
    _doc(docs, ctx_a, run, 2024, "written as a number")
    _doc(docs, ctx_a, run, "2024", "written as text")

    rows = run(docs.find_matching(ctx_a, workspace_id="ws-years", year=2024))
    assert sorted(r["title"] for r in rows) == [
        "written as a number", "written as text",
    ]


def test_a_text_year_filter_also_finds_the_numeric_documents(db, run, ctx_a):
    docs = DocumentRepository()
    _doc(docs, ctx_a, run, 2023, "numeric")
    _doc(docs, ctx_a, run, "2023", "text")

    # The LLM tool loop and query parsers hand years over as whatever the model
    # wrote, so both spellings arrive here.
    rows = run(docs.find_matching(ctx_a, workspace_id="ws-years", year="2023"))
    assert len(rows) == 2


def test_a_year_filter_still_excludes_other_years(db, run, ctx_a):
    docs = DocumentRepository()
    for year in (2022, "2023", 2024):
        _doc(docs, ctx_a, run, year, f"y{year}")

    rows = run(docs.find_matching(ctx_a, workspace_id="ws-years", year=[2023, 2024]))
    assert sorted(str(r["year"]) for r in rows) == ["2023", "2024"]
    assert [r["title"] for r in rows] == ["y2023", "y2024"]


def test_find_by_metadata_matches_both_year_forms(db, run, ctx_a):
    docs = DocumentRepository()
    _doc(docs, ctx_a, run, 2024, "numeric")
    _doc(docs, ctx_a, run, "2024", "text")

    assert len(run(docs.find_by_metadata(ctx_a, workspace_id="ws-years", year=2024))) == 2


def test_distinct_years_returns_ints_and_survives_a_mixed_workspace(db, run, ctx_a):
    docs = DocumentRepository()
    _doc(docs, ctx_a, run, "2023", "text")
    _doc(docs, ctx_a, run, 2024, "numeric")
    _doc(docs, ctx_a, run, "2024", "same year, other shape")

    years = run(docs.distinct_years(ctx_a, workspace_id="ws-years", status="active"))
    # Retrieval tests ``plan_year in available_years`` with an int plan year:
    # a text value in this list can never match, which switches the filter off.
    assert years == [2023, 2024]
    assert all(type(y) is int for y in years)


def test_a_metric_lookup_by_text_year_finds_rows_stored_as_numbers(db, run, ctx_a):
    fin = StructuredFinancialRepository()
    run(fin.insert_many(ctx_a, [
        {"workspace_id": "ws-years", "document_id": "doc-1", "company": "PIF",
         "metric": "revenue", "fiscal_year": 2024, "quarter": "",
         "value": Decimal("1000.0")},
    ]))

    assert len(run(fin.get_metric(ctx_a, company="PIF", metric="revenue",
                                  fiscal_year="2024"))) == 1
    assert len(run(fin.compare_metric(ctx_a, companies=["PIF"], metric="revenue",
                                      fiscal_year="2024"))) == 1
    assert len(run(fin.find_by_company(ctx_a, workspace_id="ws-years",
                                       company="PIF", fiscal_year="2024"))) == 1

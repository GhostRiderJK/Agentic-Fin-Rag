"""Unit tests for individual modules (no API)."""

from __future__ import annotations

import pytest


def test_query_understanding_basic():
    from app.retrieval.query_understanding import understand_query

    plan = understand_query("Compare Apple and Microsoft revenue in 2025")
    assert "Apple" in plan.companies
    assert "Microsoft" in plan.companies
    assert 2025 in plan.years
    assert plan.is_comparison is True
    assert plan.is_metric_lookup is True

    plan2 = understand_query("What are Tesla's risk factors?")
    assert "Tesla" in plan2.companies
    assert plan2.is_metric_lookup is False


def test_query_understanding_metadata_filter():
    from app.retrieval.query_understanding import understand_query

    plan = understand_query("What was NVIDIA's Q3 2024 revenue?")
    flt = plan.to_metadata_filter()
    assert flt["company"] == "NVIDIA"
    assert flt["year"] == 2024
    assert flt["quarter"] == "Q3"


def test_query_rewrite_with_history():
    from app.retrieval.query_understanding import rewrite_query_with_history

    out = rewrite_query_with_history(
        "What about its EPS?", [("Tell me about Apple revenue.", "")]
    )
    assert "Apple" in out


def test_chunking_parent_child():
    from app.ingestion.chunking import chunk_document

    text = (
        "1. BUSINESS OVERVIEW\n\n"
        + ("ACME makes widgets. " * 50) + "\n\n"
        "2. RISK FACTORS\n\n"
        + ("The company faces competition. " * 50)
    )
    result = chunk_document(text)
    assert len(result.parents) >= 2
    for p in result.parents:
        assert p.children
        assert all(len(c.text) > 0 for c in p.children)


def test_chunking_keeps_every_chunk_on_the_page_it_came_from():
    """The parser knows page numbers; chunking used to discard them and stamp
    page=1 on all ~1,000 chunks of a real 80-page report, which made every PDF
    citation point at the cover."""
    from app.ingestion.chunking import chunk_document

    body = "ACME makes widgets and the segment grew 12 percent year over year. "
    pages = [
        (1, "1. BUSINESS OVERVIEW\n\n" + body * 8),
        # same section, continues here with no repeated header
        (2, body * 8),
        (3, "2. RISK FACTORS\n\n" + "The company faces competition. " * 8),
    ]
    result = chunk_document("\n\n".join(t for _, t in pages), pages=pages,
                            tables_by_page={3: ["| Revenue | 120,000 |"]})

    assert {p.page for p in result.parents} == {1, 2, 3}
    for parent in result.parents:
        assert parent.page in {c.page for c in parent.children} or not parent.children
        assert body.strip() in parent.text or "competition" in parent.text
    # The section title survives the page break it spans.
    spanning = [p for p in result.parents if p.section == "BUSINESS OVERVIEW"]
    assert {p.page for p in spanning} == {1, 2}
    # Tables attach by page, so only the page-3 parent carries one.
    assert [p.tables_md for p in result.parents if p.page == 3]
    assert not any(p.tables_md for p in result.parents if p.page != 3)


def test_table_extractor_extracts_financials():
    from app.ingestion.parser import TableExtract
    from app.ingestion.tables import extract_financials

    tbl = TableExtract(
        markdown="",
        rows=[
            ["Metric", "2025", "2024"],
            ["Revenue", "120,000", "110,000"],
            ["Net income", "28,000", "27,500"],
            ["EPS", "5.20", "4.95"],
            ["Not a metric", "abc", "def"],
        ],
        page=4,
    )
    rows = extract_financials([tbl], company="ACME", ticker="ACME")
    metrics = {r.metric for r in rows}
    assert "revenue" in metrics
    assert "net_income" in metrics
    assert "eps" in metrics
    rev_2025 = next(r for r in rows if r.metric == "revenue" and r.fiscal_year == 2025)
    assert rev_2025.value == 120_000.0


def test_cleaner_glossary_mapping():
    from app.ingestion.cleaner import clean_text

    out = clean_text("Laba Bersih naik. Pendapatan naik. Arus Kas Operasi naik.")
    assert "Net Income" in out
    assert "Revenue" in out
    assert "Operating Cash Flow" in out


def test_citation_verifier_supports_and_rejects():
    from app.citation import CitationVerifier, VerificationOutcome
    from app.retrieval.hybrid import RetrievedChunk

    chunks = [RetrievedChunk(
        chunk_id="c1", parent_id=None, document_id="d1",
        text="", parent_text="Apple revenue in 2025 was 120 billion dollars.",
        tables_md="", company="Apple", year=2025, quarter="",
        doc_type="annual_report", section="", page=4,
    )]
    v = CitationVerifier()
    supported = v.verify(
        "Apple revenue in 2025 was 120 billion dollars.", chunks)
    assert supported.outcome == VerificationOutcome.SUPPORTED

    rejected = v.verify(
        "Apple revenue in 2025 was 999 trillion dollars and they went bankrupt.",
        chunks,
    )
    assert rejected.outcome in (VerificationOutcome.PARTIAL,
                                 VerificationOutcome.REJECTED)


def test_eval_metrics_numeric_accuracy():
    from app.evaluation.harness import score_item

    m = score_item(
        question="What was ACME EPS in 2025?",
        expected="EPS was 5.20",
        actual="ACME's EPS in 2025 was 5.20.",
        expected_numbers=["5.20"],
        expected_keywords=["eps"],
        n_retrieved=5,
        verification_outcome="supported",
    )
    assert m["numeric_accuracy"] == 1.0
    assert m["faithfulness"] == 1.0


def test_stub_llm_deterministic():
    from app.providers.llm import StubLLM, Message

    llm = StubLLM()
    msgs = [Message(role="user", content="What was ACME revenue in 2025?")]
    a = llm.chat(msgs).content
    b = llm.chat(msgs).content
    assert a == b  # deterministic


def test_stub_embeddings_similarity():
    from app.providers.embeddings import StubEmbeddings
    import math

    emb = StubEmbeddings(dim=128)
    [v1, v2, v3] = emb.embed([
        "Apple revenue 2025",
        "Apple revenue 2025 fiscal year",
        "totally unrelated topic about weather",
    ])

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cos(v1, v2) > cos(v1, v3)


def test_financial_tools_with_seed():
    """End-to-end check of the SQL tool-calling target against a seeded DB."""
    from sqlalchemy import select
    from app.db.base import db_scope, init_db
    from app.db.models import (
        StructuredFinancial, Tenant, User, Workspace,
    )
    from app.reasoning.tools import get_financial_metric

    init_db()
    with db_scope() as db:
        t = db.scalar(select(Tenant)) or Tenant(name="T")
        if t.id is None:
            db.add(t)
            db.flush()
        ws = db.scalar(select(Workspace)) or Workspace(tenant_id=t.id, name="W")
        if ws.id is None:
            db.add(ws)
            db.flush()
        # clean + insert
        db.query(StructuredFinancial).delete()  # type: ignore[attr-defined]
        db.add(StructuredFinancial(
            workspace_id=ws.id, company="ACME", ticker="ACME",
            metric="eps", fiscal_year=2025, quarter="",
            value=5.20, unit="USD", currency="USD", page=4, confidence=1.0,
        ))
        db.flush()
        out = get_financial_metric(db, workspace_id=ws.id, company="ACME",
                                   metric="eps", fiscal_year=2025)
        assert out["results"]
        assert out["results"][0]["value"] == 5.20

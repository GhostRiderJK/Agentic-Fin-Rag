"""Unit tests for individual modules (no API)."""

from __future__ import annotations

import pytest
from mongo_support import app_db  # noqa: F401  (fixture used by name)


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


def test_lexical_query_expands_financial_abbreviations():
    from app.retrieval.hybrid import _lexical_query

    assert "assets under management" in _lexical_query("PIF AuM in 2024")
    assert "total shareholder return" in _lexical_query("annualized TSR")


def test_hybrid_bm25_scores_never_penalize_common_query_terms():
    from app.retrieval.hybrid import HybridRetriever, _query_term_coverage

    retriever = object.__new__(HybridRetriever)
    chunks = [
        {"id": "a", "text": "PIF annual report 2023 assets under management"},
        {"id": "b", "text": "PIF annual report 2023 assets under management"},
        {"id": "c", "text": "PIF annual report 2023 assets under management"},
        {"id": "d", "text": "unrelated investment narrative"},
    ]
    scores = retriever._bm25_scores("PIF AuM 2023", chunks)

    assert all(score >= 0 for score in scores.values())
    assert _query_term_coverage(
        "What was PIF's AuM in 2023?",
        "PIF Assets Under Management AuM 2023 SAR 2,871 billion",
    ) == 1.0


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


def test_pdf_printed_page_footer_is_distinct_from_physical_index():
    from app.ingestion.parser import (
        _printed_page_number,
        _printed_page_number_from_page,
    )

    assert _printed_page_number(
        "At a Glance\nCompanies established in 2024\n16\n10 11"
    ) == 10
    assert _printed_page_number(
        "Performance Review\nAuM reached SAR 3,424 BN\n18 19"
    ) == 18
    assert _printed_page_number("Projects funded 91") is None

    class Page:
        width = 1190
        height = 842
        chars = [
            {"text": "1", "x0": 51, "top": 801},
            {"text": "1", "x0": 51, "top": 801},
            {"text": "3", "x0": 55, "top": 801},
            {"text": "3", "x0": 55, "top": 801},
        ]

        def extract_words(self):
            raise AssertionError("character positions should be preferred")

    assert _printed_page_number_from_page(Page(), 7) == 13


def test_table_content_is_embeddable_alongside_page_text():
    """Table labels and values must share searchable child text, not just the
    parent context that is fetched after a different passage matched."""
    from app.ingestion.chunking import chunk_document

    table = "| Assets under Management | SAR 3,424 billion |"
    chunks = chunk_document(
        "PIF reported strong growth in 2024.",
        pages=[(6, "PIF reported strong growth in 2024.\n\n" + table)],
    )
    searchable = " ".join(child.text for parent in chunks.parents
                           for child in parent.children)
    assert "Assets under Management" in searchable
    assert "3,424" in searchable


def test_child_chunk_overlap_keeps_a_metric_label_with_its_value():
    from app.ingestion.chunking import _split_children

    children = _split_children(
        ("context " * 41) + "\nAssets Under Management\nSAR 3,424 billion"
    )
    assert any("Assets Under Management" in child and "3,424" in child
               for child in children)


def test_multi_year_retrieval_fans_out_to_each_requested_year():
    from app.reasoning.orchestrator import _multi_year_retrieve
    from app.retrieval.hybrid import RetrievalResult, RetrievedChunk
    from app.retrieval.query_understanding import understand_query

    class FakeRetriever:
        def __init__(self):
            self.years = []
            self.questions = []

        def retrieve(self, question, *, plan, rerank_top_k):
            self.years.extend(plan.years)
            self.questions.append(question)
            year = plan.years[0]
            chunk = RetrievedChunk(
                chunk_id=f"chunk-{year}", parent_id=None,
                document_id=f"doc-{year}", text=f"AuM {year}",
                parent_text=f"AuM in {year}", tables_md="", company="PIF",
                year=year, quarter="", doc_type="annual_report",
                section="", page=10,
            )
            return RetrievalResult(question, plan, [chunk])

    plan = understand_query("Compare assets under management in 2023 and 2024")
    fake = FakeRetriever()
    result = _multi_year_retrieve(fake, plan.question, plan)

    assert fake.years == [2023, 2024]
    assert ["2023" in q and "2024" not in q for q in fake.questions] == [True, False]
    assert {chunk.year for chunk in result.chunks} == {2023, 2024}


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
        "Apple revenue in 2025 was 120 billion dollars [1].", chunks)
    assert supported.outcome == VerificationOutcome.SUPPORTED

    rejected = v.verify(
        "Apple revenue in 2025 was 999 trillion dollars and they went bankrupt [1].",
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


def test_eval_numeric_accuracy_accepts_unit_qualified_gold():
    """PIF gold entries are phrases like "SAR 3,424 billion", not bare numbers.

    Comparing the phrase against the answer's numeric tokens scored a verbatim
    correct answer as 0.0, so the metric silently understated every run.
    """
    from app.evaluation.harness import score_item

    def score(actual: str, expected_numbers: list[str]) -> float:
        return score_item(
            question="What were PIF's total assets under management?",
            expected="AUM reached SAR 3,424 billion, up from SAR 2,871 billion.",
            actual=actual, expected_numbers=expected_numbers,
            expected_keywords=["aum"], n_retrieved=5,
            verification_outcome="supported")["numeric_accuracy"]

    answer = ("AuM was SAR 3,424 billion at year-end 2024, up from "
              "SAR 2,871 billion in 2023 [1][3].")
    phrases = ["SAR 3,424 billion", "SAR 2,871 billion"]
    assert score(answer, phrases) == 1.0
    # Bare-number gold keeps working exactly as before.
    assert score(answer, ["3,424", "2,871"]) == 1.0
    # One of two figures missing -> half credit, not zero and not full marks.
    assert score("AuM reached SAR 3,424 billion in 2024 [1].", phrases) == 0.5
    # A wrong figure scores zero. Units are deliberately not checked: "3,424
    # million" satisfies a "3,424 billion" gold entry, which is a documented
    # limit of this proxy rather than a pass.
    assert score("AuM was SAR 4,931 billion.", phrases) == 0.0
    # Both numbers inside one phrase must both appear.
    assert score("Split 87% internal.", ["87% internally, 13% externally"]) == 0.0
    assert score("Split 87% internal and 13% external.",
                 ["87% internally, 13% externally"]) == 1.0


def test_stub_llm_deterministic():
    from app.providers.llm import Message, StubLLM

    llm = StubLLM()
    msgs = [Message(role="user", content="What was ACME revenue in 2025?")]
    a = llm.chat(msgs).content
    b = llm.chat(msgs).content
    assert a == b  # deterministic


def test_stub_embeddings_similarity():

    from app.providers.embeddings import StubEmbeddings

    emb = StubEmbeddings(dim=128)
    [v1, v2, v3] = emb.embed([
        "Apple revenue 2025",
        "Apple revenue 2025 fiscal year",
        "totally unrelated topic about weather",
    ])

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cos(v1, v2) > cos(v1, v3)


def test_stub_reranker_does_not_penalize_long_evidence_chunks():
    from app.providers.reranker import StubReranker

    docs = [
        "PIF annual report " + ("general investment context " * 80),
        "PIF reported AuM in 2023 was SAR 2,871 billion. "
        + ("supporting context " * 35),
    ]
    ranked = StubReranker().rerank("PIF AuM 2023", docs)
    assert ranked[0][0] == 1


def test_financial_tools_with_seed(app_db):  # noqa: F811
    """End-to-end check of the metric tool-calling route against a seeded DB."""
    from decimal import Decimal

    from app.db.mongodb.repositories import StructuredFinancialRepository
    from app.db.mongodb.runtime import run_sync
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.tools import get_financial_metric

    ctx = TenantContext(tenant_id="tenant-seed", workspace_id="ws-seed")
    run_sync(StructuredFinancialRepository().insert_one(
        ctx,
        {
            "workspace_id": "ws-seed", "company": "ACME", "ticker": "ACME",
            "metric": "eps", "fiscal_year": 2025, "quarter": "",
            "value": Decimal("5.20"), "unit": "USD", "currency": "USD",
            "page": 4, "confidence": Decimal("1"),
        },
    ))

    out = get_financial_metric(ctx, company="ACME", metric="eps",
                               fiscal_year=2025)
    assert out["results"]
    # Money leaves the tool as a string: a float would round 5.20 through
    # binary arithmetic on the way to the LLM.
    assert out["results"][0]["value"] == "5.20"

    # Another workspace of the same tenant sees nothing.
    other = get_financial_metric(
        ctx.with_workspace("ws-other"), company="ACME", metric="eps",
        fiscal_year=2025)
    assert other["results"] == []

    # Nor does another tenant, even naming the same workspace.
    foreign = get_financial_metric(
        TenantContext(tenant_id="tenant-other", workspace_id="ws-seed"),
        company="ACME", metric="eps", fiscal_year=2025)
    assert foreign["results"] == []


def test_eval_suite_names_resolve_inside_the_suites_dir(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.core.errors import NotFoundError, ValidationError
    from app.evaluation import suites

    d = tmp_path / "eval_suites"
    d.mkdir()
    (d / "pif_gold.jsonl").write_text(
        '{"question": "q1", "expected": "a1"}\n\n{"question": "q2"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(get_settings(), "eval_suites_dir", str(d), raising=False)

    assert [s["id"] for s in suites.list_suites()] == ["packaged", "pif_gold"]
    assert suites.list_suites()[1]["n_items"] == 2

    path, name = suites.resolve("pif_gold")
    assert name == "pif_gold" and path.read_text(encoding="utf-8").startswith("{")

    # The packaged suite is what an empty name means.
    assert suites.resolve("")[1] == "packaged"

    with pytest.raises(ValidationError):
        suites.resolve("../etc/passwd")
    with pytest.raises(NotFoundError):
        suites.resolve("not_a_suite")

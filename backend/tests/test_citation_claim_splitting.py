"""Regressions for claim splitting in citation verification.

Every case here came from a live answer over the PIF 2024 annual report that was
correct, cited, and then abstained on.  The model writes a bullet's citation at
the end of the line, so "…SAR 284,005 million in 2024 vs. SAR 237,575 million in
2023 (+20%) [1]" was being cut at the period in "vs." into one fragment that
carried the marker and one that did not, and the marker-less halves counted as
unsupported claims until the ratio forced a rejection of the whole answer.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.citation import CitationVerifier, VerificationOutcome
from app.retrieval.hybrid import RetrievedChunk

INCOME_PASSAGE = (
    "Consolidated Statement of Profit or Loss\n"
    "Amounts in SAR million\n"
    "2024 2023 Change % Change\n"
    "Total revenue 413,423 331,360 82,063 25%\n"
    "Revenue 284,005 237,575 46,430 20%\n"
    "Income from investment activities 129,418 93,785 35,633 38%\n"
    "\n"
    "The 25% total revenue growth was driven by both core revenue, which rose "
    "20%, and a faster rise in income from investment activities, up 38%."
)

MEASURED_DRAFT = (
    "**Total revenue (FY2024): SAR 413,423 million**, up **SAR 82,063 million "
    "(+25%)** from SAR 331,360 million in FY2023 [1][4][5].\n"
    "\n"
    "Breakdown of the change:\n"
    "- **Revenue:** SAR 284,005 million in 2024 vs. SAR 237,575 million in "
    "2023 — up SAR 46,430 million (+20%) [1][4][5]\n"
    "- **Income from investment activities:** SAR 129,418 million in 2024 vs. "
    "SAR 93,785 million in 2023 — up SAR 35,633 million (+38%) [1][4][5]\n"
    "\n"
    "So the 25% total revenue growth was driven by both core revenue (+20%) and "
    "a faster rise in investment income (+38%) [1][4][5]."
)


def _chunk(text: str, chunk_id: str = "c1", page: int = 75) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, parent_id=None, document_id="doc", text=text,
        parent_text="", tables_md="", company="PIF", year=2024, quarter="",
        doc_type="annual_report", section="income statement", page=page,
    )


def _split(draft: str) -> list:
    return CitationVerifier()._split_claims(draft)


def test_measured_bullet_answer_is_fully_supported():
    chunks = [_chunk(INCOME_PASSAGE, "1"), _chunk(INCOME_PASSAGE, "2"),
              _chunk(INCOME_PASSAGE, "3"), _chunk(INCOME_PASSAGE, "4"),
              _chunk(INCOME_PASSAGE, "5")]

    result = CitationVerifier().verify(MEASURED_DRAFT, chunks)

    assert result.outcome == VerificationOutcome.SUPPORTED
    assert "no citation marker" not in result.note
    assert len(result.supported) == 4


def test_abbreviation_period_keeps_the_bullet_and_its_citation_together():
    claims = _split(
        "Revenue was SAR 284,005 million in 2024 vs. SAR 237,575 million in "
        "2023 [1]."
    )

    assert len(claims) == 1
    assert claims[0].refs == [1]
    assert "vs." in claims[0].text  # punctuation restored, not eaten


def test_unmarked_fragment_of_a_cited_line_shares_the_citation():
    claims = _split(
        "Revenue was SAR 284,005 million in 2024 [1]. Cost of revenue rose to\n"
        "SAR 189,164 million.\n"
    )
    # The marker travels inside its own line, and stops at the line break.
    assert [c.refs for c in claims] == [[1], [1], []]
    assert claims[1].text.endswith("[1]")


def test_a_line_with_no_citation_anywhere_stays_uncited():
    claims = _split("Revenue was SAR 284,005 million in 2024 and grew 20%.")

    assert len(claims) == 1
    assert claims[0].refs == []
    result = CitationVerifier().verify(claims[0].text, [_chunk(INCOME_PASSAGE)])
    assert "claim has no citation marker" in result.note


def test_digit_free_heading_line_is_not_a_claim():
    claims = _split("Breakdown of the change:\n- Revenue rose 20% [1].\n")

    assert [c.text for c in claims] == ["- Revenue rose 20% [1]."]


def test_list_markers_survive_so_the_trimmed_answer_stays_markdown():
    draft = ("Total revenue rose 25% [1].\n"
             "- Revenue rose 20% [1].\n"
             "- Investment income rose 38% [1].\n")

    claims = _split(draft)

    assert [c.text for c in claims] == [
        "Total revenue rose 25% [1].",
        "- Revenue rose 20% [1].",
        "- Investment income rose 38% [1].",
    ]
    assert claims[1].judged_text == "- Revenue rose 20%."


def test_heading_that_asserts_a_figure_is_still_a_claim():
    claims = _split("FY2024 revenue:\n")

    assert len(claims) == 1
    assert claims[0].refs == []


def test_sentences_with_their_own_markers_are_not_merged():
    chunks = [
        _chunk("PIF assets under management in 2024 were SAR 3,424 billion.", "1"),
        _chunk("PIF assets under management in 2023 were SAR 2,871 billion.", "2"),
    ]
    draft = ("PIF AuM in 2024 was SAR 3,424 billion [1]. "
             "PIF AuM in 2023 was SAR 2,871 billion [2].")

    claims = _split(draft)
    result = CitationVerifier().verify(draft, chunks)

    assert [c.refs for c in claims] == [[1], [2]]
    assert result.outcome == VerificationOutcome.SUPPORTED


def test_wrong_marker_is_not_rescued_by_the_cited_sibling_sentence():
    chunks = [
        _chunk("PIF assets under management in 2024 were SAR 3,424 billion.", "1"),
        _chunk("PIF had 40 employees at the end of 2024.", "2"),
    ]

    result = CitationVerifier().verify(
        "PIF AuM in 2024 was SAR 3,424 billion [1]. "
        "PIF had 40 employees at the end of 2024 [2]. "
        "PIF AuM in 2024 was SAR 9,999 billion [2].",
        chunks,
    )

    assert result.outcome == VerificationOutcome.PARTIAL
    assert any("9,999" in claim for claim in result.unsupported)
    assert not any("9,999" in claim for claim in result.supported)


def test_orchestrator_returns_supported_claims_on_a_partial_verdict(monkeypatch):
    import app.reasoning.orchestrator as orchestrator_module
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator
    from app.retrieval.hybrid import RetrievalResult

    chunks = [
        _chunk("PIF assets under management in 2024 were SAR 3,424 billion.", "1"),
        _chunk("PIF assets under management in 2023 were SAR 2,871 billion.", "2"),
    ]

    class FakeRetriever:
        def __init__(self, _ctx): pass

        def retrieve(self, query, *, plan, **_kwargs):
            return RetrievalResult(query=query, plan=plan, chunks=chunks)

    class PartialLLM:
        name = "test-responder"

        def chat(self, _messages, **_kwargs):
            return SimpleNamespace(content=(
                "PIF assets under management in 2024 were SAR 3,424 billion [1].\n"
                "- PIF assets under management in 2023 were SAR 2,871 billion [2].\n"
                "- The fund focuses on emerging markets worldwide [1]."
            ), tool_calls=[])

    monkeypatch.setattr(orchestrator_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        orchestrator_module, "build_dispatcher", lambda _ctx: (lambda *a: {}, [])
    )
    result = Orchestrator(
        TenantContext(tenant_id="tenant", workspace_id="workspace"),
        llm=PartialLLM(), verifier=CitationVerifier(),
    ).answer(AnswerRequest(
        question="What was PIF AuM in 2024 and 2023?", workspace_id="workspace",
    ))

    assert result.verification["outcome"] == "partial"
    assert "citation_verification_trimmed" in result.notes
    assert "3,424" in result.answer and "2,871" in result.answer
    assert "emerging markets" not in result.answer
    # The answer is rendered as markdown, so a trimmed bullet list must stay a
    # list instead of collapsing into one paragraph.
    assert result.answer.splitlines() == [
        "PIF assets under management in 2024 were SAR 3,424 billion [1].",
        "- PIF assets under management in 2023 were SAR 2,871 billion [2].",
    ]
    assert [c["chunk_id"] for c in result.citations] == ["1", "2"]


def test_orchestrator_still_abstains_when_nothing_is_supported(monkeypatch):
    import app.reasoning.orchestrator as orchestrator_module
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator
    from app.retrieval.hybrid import RetrievalResult

    chunk = _chunk("PIF assets under management in 2024 were SAR 3,424 billion.", "1")

    class FakeRetriever:
        def __init__(self, _ctx): pass

        def retrieve(self, query, *, plan):
            return RetrievalResult(query=query, plan=plan, chunks=[chunk])

    class HallucinatingLLM:
        name = "test-responder"

        def chat(self, _messages, **_kwargs):
            return SimpleNamespace(content=(
                "PIF paid a dividend of SAR 55 billion to shareholders [1]. "
                "PIF opened a branch in NEOM [1]."
            ), tool_calls=[])

    monkeypatch.setattr(orchestrator_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        orchestrator_module, "build_dispatcher", lambda _ctx: (lambda *a: {}, [])
    )
    result = Orchestrator(
        TenantContext(tenant_id="tenant", workspace_id="workspace"),
        llm=HallucinatingLLM(), verifier=CitationVerifier(),
    ).answer(AnswerRequest(
        question="Did PIF pay a dividend?", workspace_id="workspace",
    ))

    assert result.verification["outcome"] == "rejected"
    assert "citation_verification_abstained" in result.notes
    assert "55" not in result.answer
    assert result.citations == []

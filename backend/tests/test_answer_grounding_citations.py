"""Focused regressions for answer grounding and citation attribution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.citation import CitationVerifier, VerificationOutcome
from app.providers.llm import Message, StubLLM
from app.retrieval.hybrid import RetrievedChunk


def test_prompt_allows_only_numeric_bracket_citation_syntax():
    from app.reasoning.agents import SYSTEM_PROMPT

    assert "only the exact ASCII form [n]" in SYSTEM_PROMPT
    assert "Never use" in SYSTEM_PROMPT
    assert "【5†L1-L3】" in SYSTEM_PROMPT


def _chunk(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, parent_id=None, document_id=f"doc-{chunk_id}",
        text=text, parent_text=text, tables_md="", company="PIF",
        year=2024, quarter="", doc_type="annual_report", section="",
        page=page, source_filename="2024-en.pdf", physical_page=page,
    )


def test_claim_is_checked_only_against_the_chunk_it_cites():
    chunks = [
        _chunk("unrelated", "PIF employee count was 225 employees.", 6),
        _chunk("support", "PIF assets under management in 2024 were SAR 3,424 billion.", 6),
    ]
    verifier = CitationVerifier()

    wrong = verifier.verify(
        "PIF assets under management in 2024 were SAR 3,424 billion [1].",
        chunks,
    )
    correct = verifier.verify(
        "PIF assets under management in 2024 were SAR 3,424 billion [2].",
        chunks,
    )

    assert wrong.outcome in {VerificationOutcome.PARTIAL, VerificationOutcome.REJECTED}
    assert correct.outcome == VerificationOutcome.SUPPORTED


def test_multi_page_claim_can_use_multiple_explicit_citations():
    chunks = [
        _chunk("2024", "PIF assets under management in 2024 were SAR 3,424 billion.", 6),
        _chunk("2023", "PIF assets under management in 2023 were SAR 2,871 billion.", 5),
    ]

    result = CitationVerifier().verify(
        "PIF assets under management were SAR 3,424 billion in 2024 and "
        "SAR 2,871 billion in 2023 [1] [2].",
        chunks,
    )

    assert result.outcome == VerificationOutcome.SUPPORTED


@pytest.mark.parametrize("markers", ["[1]", "【1】", "[1] 【1】"])
def test_ascii_unicode_and_mixed_numeric_markers_resolve_same_context_item(markers):
    evidence = _chunk("aum", "PIF AuM was SAR 3,424 billion in 2024.", 9)
    result = CitationVerifier().verify(
        f"PIF AuM was SAR 3,424 billion in 2024 {markers}.", [evidence]
    )
    assert result.outcome == VerificationOutcome.SUPPORTED


@pytest.mark.parametrize("marker", ["【one】", "【0】", "[x]"])
def test_malformed_or_unavailable_citation_marker_fails_closed(marker):
    evidence = _chunk("aum", "PIF AuM was SAR 3,424 billion in 2024.", 9)
    result = CitationVerifier().verify(
        f"PIF AuM was SAR 3,424 billion in 2024 {marker}.", [evidence]
    )
    assert result.outcome == VerificationOutcome.REJECTED
    assert result.unsupported


def test_duplicate_citation_indexes_do_not_change_evidence_resolution():
    evidence = _chunk("aum", "PIF AuM was SAR 3,424 billion in 2024.", 9)
    result = CitationVerifier().verify(
        "PIF AuM was SAR 3,424 billion in 2024 [1] [1].", [evidence]
    )
    assert result.outcome == VerificationOutcome.SUPPORTED


def test_multiple_claims_are_verified_independently():
    chunks = [
        _chunk("supported", "PIF reported SAR 3,424 billion in 2024.", 9),
        _chunk("other", "PIF workforce was 2,500 employees.", 10),
    ]
    result = CitationVerifier().verify(
        "PIF reported SAR 3,424 billion in 2024 [1]. "
        "PIF reported SAR 9,999 billion in 2024 [2].", chunks
    )
    assert result.outcome == VerificationOutcome.REJECTED
    assert len(result.supported) == 1
    assert len(result.unsupported) == 1


def test_currency_cannot_be_borrowed_from_a_different_years_citation():
    chunks = [
        _chunk("2023", "PIF assets under management in 2023 were SAR 2,871 billion.", 22),
        _chunk("2024", "PIF assets under management in 2024 were 3,424 billion.", 9),
    ]

    result = CitationVerifier().verify(
        "PIF assets under management in 2024 were SAR 3,424 billion [1] [2].",
        chunks,
    )

    assert result.outcome == VerificationOutcome.REJECTED


def test_source_riyals_glyph_and_bn_abbreviation_support_sar_amount():
    evidence = _chunk("2024", "PIF assets under management were ª 3,424 BN.", 22)

    result = CitationVerifier().verify(
        "PIF assets under management were SAR 3,424 billion [1].", [evidence],
    )

    assert result.outcome == VerificationOutcome.SUPPORTED


def test_missing_or_invalid_citation_is_not_supported_by_other_chunks():
    chunks = [_chunk("support", "PIF reported SAR 3,424 billion in 2024.", 6)]
    verifier = CitationVerifier()

    missing = verifier.verify("PIF reported SAR 3,424 billion in 2024.", chunks)
    invalid = verifier.verify("PIF reported SAR 3,424 billion in 2024 [2].", chunks)

    assert missing.outcome == VerificationOutcome.REJECTED
    assert invalid.outcome == VerificationOutcome.REJECTED
    assert "citation" in missing.note
    assert "citation" in invalid.note


@pytest.mark.parametrize(
    "claim",
    [
        "PIF AuM was SAR 99 billion in 2024 [1].",
        "PIF AuM was USD 3,424 billion in 2024 [1].",
        "PIF AuM was SAR 3,424 million in 2024 [1].",
    ],
)
def test_verifier_rejects_financial_value_currency_and_scale_conflicts(claim):
    evidence = _chunk(
        "aumu-2024", "PIF assets under management in 2024 were SAR 3,424 billion.", 6,
    )

    result = CitationVerifier().verify(claim, [evidence])

    assert result.outcome == VerificationOutcome.REJECTED
    assert result.unsupported


def test_verifier_does_not_skip_short_numeric_claims():
    evidence = _chunk(
        "aumu-2024", "PIF assets under management in 2024 were SAR 3,424 billion.", 6,
    )

    result = CitationVerifier().verify("SAR 99 [1]", [evidence])

    assert result.outcome == VerificationOutcome.REJECTED


def test_verifier_rejects_short_uncited_answer():
    evidence = _chunk("fact", "PIF reported SAR 3,424 billion in 2024.", 6)

    result = CitationVerifier().verify("Yes.", [evidence])

    assert result.outcome == VerificationOutcome.REJECTED


def test_malformed_llm_judge_boolean_cannot_support_claim():
    class MalformedJudge:
        name = "test-model"

        def chat(self, _messages, **_kwargs):
            return SimpleNamespace(
                content='[{"index": 1, "supported": "false"}]',
            )

    evidence = _chunk("strategy", "The fund focuses on investment markets globally.", 9)
    result = CitationVerifier(MalformedJudge()).verify(
        "The fund focuses on global investment markets [1].", [evidence],
    )

    assert result.outcome == VerificationOutcome.REJECTED
    assert result.unsupported


def test_stub_is_an_explicit_non_factual_abstention():
    answer = StubLLM().chat([
        Message(role="system", content="Context has 3,424 SAR billion [1]."),
        Message(role="user", content="What was PIF AuM?"),
    ]).content

    assert "offline test stub" in answer.casefold()
    assert "3,424" not in answer
    assert "[1]" not in answer


def test_query_understanding_scopes_pif_and_bank_issuers():
    from app.retrieval.query_understanding import understand_query

    assert understand_query("What was PIF's AuM in 2024?").companies == ["PIF"]
    assert understand_query("What was BMO's return in Q4 2024?").companies == ["BMO"]
    assert understand_query("What did TD report in Q2 2025?").companies == ["TD"]
    assert understand_query("Red Sea Global's sustainable tourism results").companies == ["PIF"]


def test_pif_metrics_separate_stub_from_retrieval_evidence():
    from app.evaluation.pif_chat_check import _evaluate_item

    result = _evaluate_item(
        {
            "question": "What was PIF AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "Offline test stub: no financial answer was generated.",
            "generation": {"provider": "stub", "is_stub": True},
            "retrieval": {
                "n_chunks": 1,
                "candidates": [{"source_filename": "2024-en.pdf", "physical_page": 6}],
            },
            "citations": [],
            "verification": {"outcome": "stub_not_evaluated"},
        },
        index=1,
    )

    assert result["criteria"]["retrieval_evidence_success"] is True
    assert result["criteria"]["correct_source_document_retrieved"] is True
    assert result["criteria"]["supporting_physical_page_retrieved"] is True
    assert result["criteria"]["numeric_value_correct"] is None
    assert result["answer_quality_status"] == "not_evaluated_stub"
    assert result["end_to_end_pass"] is None


def test_page_match_from_wrong_document_does_not_count_as_support():
    from app.evaluation.pif_chat_check import _evaluate_item

    result = _evaluate_item(
        {
            "question": "What was PIF AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "SAR 3,424 billion in 2024 [1].",
            "generation": {"provider": "openai", "is_stub": False},
            "retrieval": {"n_chunks": 1, "candidates": [{
                "source_filename": "BMO_2024q4.pdf", "physical_page": 6,
            }]},
            "citations": [{
                "source_filename": "BMO_2024q4.pdf", "physical_page": 6,
                "index": 1,
            }],
            "verification": {"outcome": "supported"},
        },
        index=1,
    )

    assert result["criteria"]["correct_source_document_retrieved"] is False
    assert result["criteria"]["supporting_physical_page_retrieved"] is False
    assert result["criteria"]["citation_correctness"] is False
    assert result["end_to_end_pass"] is False


def test_printed_and_physical_page_metadata_remain_separate():
    from app.retrieval.hybrid import RetrievalResult
    from app.retrieval.query_understanding import understand_query

    chunk = _chunk("page", "Annual report evidence", 9)
    chunk.page = 0
    chunk.printed_page = 0
    chunk.page_number_type = "printed"
    citations = RetrievalResult(
        query="question", plan=understand_query("question"), chunks=[chunk],
    ).citation_list()

    assert citations[0]["page"] == 0
    assert citations[0]["printed_page"] == 0
    assert citations[0]["physical_page"] == 9
    assert citations[0]["page_number_type"] == "printed"


def test_api_citation_schema_serializes_both_page_number_systems():
    from app.api.schemas import CitationOut

    citation = CitationOut(
        index=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        source_filename="2024-en.pdf",
        printed_page=6,
        physical_page=9,
        page_number_type="printed",
        company="PIF",
        year=2024,
        quarter=None,
        doc_type="annual_report",
        section="overview",
        page=6,
        snippet="SAR 3,424 billion",
    )

    payload = citation.model_dump()

    assert payload["source_filename"] == "2024-en.pdf"
    assert payload["printed_page"] == 6
    assert payload["physical_page"] == 9
    assert payload["page_number_type"] == "printed"


def test_rendered_context_chunk_list_matches_citation_indexes():
    from app.retrieval.hybrid import RetrievalResult
    from app.retrieval.query_understanding import understand_query

    chunks = [_chunk(str(i), f"Evidence passage {i} " + "text " * 180, i)
              for i in range(1, 5)]
    retrieval = RetrievalResult(
        query="question", plan=understand_query("question"), chunks=chunks,
    )

    context, included = retrieval.context_with_chunks(max_chunks=4, max_chars=1700)
    citations = retrieval.citation_list(chunks=included)

    assert len(included) < len(chunks)
    assert len(citations) == len(included)
    assert len(context) <= 1700
    assert f"[{len(included)}]" in context
    assert f"[{len(included) + 1}]" not in context


def test_matching_child_passage_survives_truncated_parent_for_generation_and_verifier():
    from app.retrieval.hybrid import RetrievalResult
    from app.retrieval.query_understanding import understand_query

    parent = "Annual report overview. " + ("General strategy text. " * 120)
    child = "Red Sea National Academy provides hospitality training."
    chunk = _chunk("tail-fact", child, 35)
    chunk.parent_text = parent[:2400]
    result = RetrievalResult(
        query="What did the Red Sea National Academy provide?",
        plan=understand_query("What did the Red Sea National Academy provide?"),
        chunks=[chunk],
    )

    context, included = result.context_with_chunks(max_chunks=1, max_chars=3000)
    verification = CitationVerifier().verify(
        "The Red Sea National Academy provides hospitality training [1].",
        included,
    )

    assert child in context
    assert verification.outcome == VerificationOutcome.SUPPORTED


def test_verifier_cannot_use_evidence_truncated_out_of_generation_context():
    from app.retrieval.hybrid import RetrievalResult
    from app.retrieval.query_understanding import understand_query

    tail_claim = "PIF reported SAR 3,424 billion in 2024."
    chunk = _chunk(
        "tail-truncated", "Introductory report text. " * 80 + tail_claim, 9,
    )
    result = RetrievalResult(
        query="What was PIF AuM?", plan=understand_query("What was PIF AuM?"),
        chunks=[chunk],
    )

    context, included = result.context_with_chunks(max_chunks=1, max_chars=800)
    verification = CitationVerifier().verify(f"{tail_claim} [1]", included)

    assert tail_claim not in context
    assert verification.outcome == VerificationOutcome.REJECTED


def test_evidence_trace_is_admin_only():
    import pytest

    from app.api.chat import chat
    from app.api.schemas import ChatRequest
    from app.core.errors import ForbiddenError
    from app.db.models import Role

    with pytest.raises(ForbiddenError, match="evidence trace requires admin role"):
        chat(
            ChatRequest(question="question", workspace_id="workspace",
                        include_evidence_trace=True),
            db=None,
            user=SimpleNamespace(role=Role.analyst),
        )


def test_orchestrator_records_exact_context_and_returns_only_cited_sources(monkeypatch):
    import app.reasoning.orchestrator as orchestrator_module
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator
    from app.retrieval.hybrid import RetrievalResult
    from app.retrieval.query_understanding import understand_query

    chunks = [
        _chunk("2024", "PIF assets under management in 2024 were SAR 3,424 billion.", 6),
        _chunk("2023", "PIF assets under management in 2023 were SAR 2,871 billion.", 5),
    ]
    plan = understand_query("What were PIF assets under management in 2024 and 2023?")

    class FakeRetriever:
        def __init__(self, ctx):
            self.ctx = ctx

        def retrieve(self, question, *, plan):
            return RetrievalResult(query=question, plan=plan, chunks=chunks)

    class AnswerLLM:
        name = "test-responder"

        def chat(self, messages, **kwargs):
            return SimpleNamespace(
                content=("PIF assets under management in 2024 were SAR 3,424 "
                         "billion [1]. PIF assets under management in 2023 "
                         "were SAR 2,871 billion [2]."),
                tool_calls=[],
            )

    monkeypatch.setattr(orchestrator_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        orchestrator_module, "_multi_year_retrieve",
        lambda retriever, question, plan, **kwargs: RetrievalResult(
            query=question, plan=plan, chunks=chunks,
        ),
    )
    monkeypatch.setattr(orchestrator_module, "build_dispatcher", lambda ctx: (lambda *a: {}, []))
    result = Orchestrator(
        TenantContext(tenant_id="tenant", workspace_id="workspace"),
        llm=AnswerLLM(), verifier=CitationVerifier(),
    ).answer(AnswerRequest(
        question=plan.question, workspace_id="workspace",
        include_evidence_trace=True,
    ))

    assert result.generation == {
        "provider": "test-responder", "model": None, "is_stub": False,
    }
    assert [c["chunk_id"] for c in result.citations] == ["2024", "2023"]
    assert "SAR 3,424 billion" in result.retrieval["generation_context"]
    assert "SAR 2,871 billion" in result.retrieval["generation_context"]
    assert [c["chunk_id"] for c in result.retrieval["candidates"]] == [
        "2024", "2023",
    ]
    assert result.verification["outcome"] == "supported"


def test_orchestrator_normalizes_unicode_markers_and_reverifies(monkeypatch):
    import app.reasoning.orchestrator as orchestrator_module
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator
    from app.retrieval.hybrid import RetrievalResult

    chunk = _chunk(
        "aum-2024", "PIF assets under management in 2024 were SAR 3,424 billion.", 6,
    )
    question = "What was PIF AuM in 2024?"

    class FakeRetriever:
        def __init__(self, _ctx):
            pass

        def retrieve(self, query, *, plan):
            return RetrievalResult(query=query, plan=plan, chunks=[chunk])

    class FormatRetryLLM:
        name = "test-responder"

        def __init__(self):
            self.responses = iter([
                "PIF's 2024 AuM was SAR 3,424 billion【1】.",
            ])
            self.calls = 0

        def chat(self, _messages, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=next(self.responses), tool_calls=[])

    llm = FormatRetryLLM()
    monkeypatch.setattr(orchestrator_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        orchestrator_module, "build_dispatcher", lambda _ctx: (lambda *a: {}, [])
    )
    result = Orchestrator(
        TenantContext(tenant_id="tenant", workspace_id="workspace"),
        llm=llm, verifier=CitationVerifier(),
    ).answer(AnswerRequest(
        question=question, workspace_id="workspace",
        include_evidence_trace=True,
    ))

    assert llm.calls == 1
    assert result.verification["outcome"] == "supported"
    assert "[1]" in result.answer
    assert result.citations[0]["chunk_id"] == "aum-2024"
    assert "citation_format_retry" not in result.notes
    assert "【1】" in result.retrieval["generated_draft"]
    assert "[1]" in result.retrieval["final_draft"]
    assert result.retrieval["parsed_citation_indexes"] == [1]


def test_orchestrator_retries_malformed_markers_then_reverifies(monkeypatch):
    import app.reasoning.orchestrator as orchestrator_module
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator
    from app.retrieval.hybrid import RetrievalResult

    chunk = _chunk("aum-retry", "PIF AuM was SAR 3,424 billion in 2024.", 9)
    plan = orchestrator_module.understand_query("What was PIF AuM in 2024?")

    class FakeRetriever:
        def __init__(self, _ctx): pass
        def retrieve(self, query, *, plan):
            return RetrievalResult(query=query, plan=plan, chunks=[chunk])

    class RetryLLM:
        name = "test-responder"
        def __init__(self):
            self.responses = iter([
                "PIF AuM was SAR 3,424 billion in 2024 【page 9】.",
                "PIF AuM was SAR 3,424 billion in 2024 [1].",
            ])
            self.calls = 0
        def chat(self, _messages, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=next(self.responses), tool_calls=[])

    llm = RetryLLM()
    monkeypatch.setattr(orchestrator_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(orchestrator_module, "build_dispatcher", lambda _ctx: (lambda *a: {}, []))
    result = Orchestrator(
        TenantContext(tenant_id="tenant", workspace_id="workspace"),
        llm=llm, verifier=CitationVerifier(),
    ).answer(AnswerRequest(question=plan.question, workspace_id="workspace"))
    assert llm.calls == 2
    assert result.verification["outcome"] == "supported"
    assert "citation_format_retry" in result.notes
    assert result.citations[0]["physical_page"] == 9


@pytest.mark.parametrize(
    "outcome", [VerificationOutcome.PARTIAL, VerificationOutcome.REJECTED]
)
def test_orchestrator_abstains_when_verification_is_not_fully_supported(
    monkeypatch, outcome
):
    import app.reasoning.orchestrator as orchestrator_module
    from app.db.mongodb.tenant_context import TenantContext
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator
    from app.retrieval.hybrid import RetrievalResult
    from app.retrieval.query_understanding import understand_query

    chunk = _chunk("unsupported", "PIF reported SAR 3,424 billion in 2024.", 6)
    plan = understand_query("What was PIF AuM in 2024?")

    class FakeRetriever:
        def __init__(self, _ctx):
            pass

        def retrieve(self, query, *, plan):
            return RetrievalResult(query=query, plan=plan, chunks=[chunk])

    class DraftLLM:
        name = "test-responder"

        def chat(self, _messages, **_kwargs):
            return SimpleNamespace(
                content="PIF AuM was SAR 99 billion in 2024 [1].",
                tool_calls=[],
            )

    class IncompleteVerifier:
        def verify(self, _draft, _chunks):
            return SimpleNamespace(
                outcome=outcome,
                supported=[],
                unsupported=["PIF AuM was SAR 99 billion in 2024 [1]."],
                note="unsupported numeric claim",
            )

    monkeypatch.setattr(orchestrator_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        orchestrator_module, "build_dispatcher", lambda _ctx: (lambda *a: {}, [])
    )
    result = Orchestrator(
        TenantContext(tenant_id="tenant", workspace_id="workspace"),
        llm=DraftLLM(),
        verifier=IncompleteVerifier(),
    ).answer(AnswerRequest(question=plan.question, workspace_id="workspace"))

    assert "99" not in result.answer
    assert "can't verify" in result.answer
    assert result.citations == []
    assert result.verification["outcome"] == outcome.value
    assert "citation_verification_abstained" in result.notes

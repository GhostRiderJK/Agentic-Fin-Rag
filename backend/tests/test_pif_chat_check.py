"""Regression tests for the maintained PIF API checker."""

from __future__ import annotations

from app.evaluation import pif_chat_check as checker


def test_gold_source_page_matches_physical_page_not_printed_page():
    citations = [{"page": 16, "physical_page": 9, "page_number_type": "printed"}]

    assert checker.page_match(citations, 9) is True
    assert checker.page_match(citations, 16) is False


def test_filename_normalization_only_removes_an_exact_duplicate_stem():
    assert checker.normalize_duplicate_source_filename(
        "2024-en_2024-en.pdf"
    ) == "2024-en.pdf"
    assert checker.normalize_duplicate_source_filename(
        "2024-en_final.pdf"
    ) == "2024-en_final.pdf"


def test_evaluator_reports_exact_and_normalized_source_matching_separately():
    item = {
        "id": "pif-005", "question": "What was PIF AuM?",
        "expected": "SAR 3,424 billion", "expected_numbers": [],
        "source_page": 9,
    }
    response = {
        "answer": "Not evaluated.",
        "generation": {"provider": "stub", "is_stub": True},
        "retrieval": {"n_chunks": 10, "candidates": [
            {"source_filename": "2024-en_2024-en.pdf", "physical_page": 9}
            for _ in range(10)
        ]},
        "citations": [],
    }
    result = checker._evaluate_item(item, response, index=5)
    assert result["top_k_retrieval"]["5"]["source_document_retrieved"] is True
    assert result["top_k_retrieval"]["5"]["source_document_retrieved_exact_string"] is False
    assert result["criteria"]["supporting_physical_page_retrieved"] is True


def test_unknown_gold_page_is_not_scored():
    assert checker.page_match([], None) is None
    assert checker.page_match([{"page": 16}], 9) is None


def test_numeric_value_and_currency_marker_are_scored_separately():
    expected = ["SAR 3,424 billion"]
    answer = "PIF reported SAR 3,424 billion."

    assert checker.numeric_match(expected, answer) is True
    assert checker.unit_markers_match(expected, answer) is True
    assert checker.unit_markers_match(expected, "PIF reported $3,424 billion.") is False


def test_numeric_check_does_not_treat_printed_page_as_physical_page():
    citations = [{"page": 9, "physical_page": 6}]

    assert checker.page_match(citations, 9) is False
    assert checker.page_match(citations, 6) is True


def test_bool_is_not_accepted_as_a_physical_page_number():
    assert checker.cited_physical_pages([{"physical_page": True}]) == set()


def test_unit_marker_check_includes_financial_scale():
    expected = ["SAR 3,424 billion"]

    assert checker.unit_markers_match(expected, "SAR 3,424 billion") is True
    assert checker.unit_markers_match(expected, "SAR 3,424 million") is False


def test_currency_and_scale_must_be_attached_to_the_expected_value():
    expected = ["SAR 3,424 billion"]

    assert checker.unit_markers_match(
        expected, "PIF reported SAR 3,424 billion."
    ) is True
    assert checker.unit_markers_match(
        expected, "PIF reported SAR 3,424 million and USD 2 billion."
    ) is False


def test_question_pass_requires_retrieval_support_values_units_and_page():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "In 2024, AuM was SAR 3,424 billion [1].",
            "generation": {"provider": "omniroute", "is_stub": False},
            "retrieval": {
                "n_chunks": 1,
                "candidates": [{
                    "source_filename": "2024-en.pdf", "physical_page": 6,
                    "context_supplied": True,
                    "generation_context_text": "PIF reported SAR 3,424 billion.",
                }],
                "generation_context": "PIF reported SAR 3,424 billion.",
            },
            "verification": {"outcome": "supported", "supported": ["Claim"]},
            "citations": [{
                "document_id": "doc-1", "source_filename": "2024-en.pdf",
                "index": 1, "page": 11, "physical_page": 6,
                "snippet": "SAR 3,424 billion",
            }],
        },
        index=1,
    )

    assert result["question_id"] == "pif-001"
    assert result["passed"] is True
    assert result["sources"][0]["physical_page"] == 6
    assert result["sources"][0]["source_filename"] == "2024-en.pdf"
    assert result["criteria"]["expected_page_candidate_in_generation_context"] is True


def test_question_pass_fails_unverified_answer_and_missing_citation():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "In 2024, AuM was SAR 3,424 million.",
            "generation": {"provider": "omniroute", "is_stub": False},
            "retrieval": {"n_chunks": 1},
            "verification": {"outcome": "partial", "unsupported": ["Claim"]},
            "citations": [],
        },
        index=1,
    )

    assert result["passed"] is False
    assert "currency_and_unit_correct" in result["failure_reasons"]
    assert "answer_support_by_cited_evidence" in result["failure_reasons"]
    assert "citation_correctness" in result["failure_reasons"]


def test_answer_metrics_are_not_scored_without_generation_metadata():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "In 2024, AuM was SAR 3,424 billion [1].",
            "retrieval": {"n_chunks": 1},
            "verification": {"outcome": "supported"},
            "citations": [{
                "index": 1, "source_filename": "2024-en.pdf",
                "physical_page": 6,
            }],
        },
        index=1,
    )

    assert result["criteria"]["numeric_value_correct"] is None
    assert result["criteria"]["citation_correctness"] is None
    assert result["passed"] is None
    assert result["answer_quality_status"] == "not_evaluated_unknown_generation"


def test_stub_provider_without_flag_is_still_not_evaluated():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "In 2024, AuM was SAR 3,424 billion [1].",
            "generation": {"provider": "stub"},
            "retrieval": {"n_chunks": 1},
            "citations": [{
                "index": 1, "source_filename": "2024-en.pdf",
                "physical_page": 6,
            }],
        },
        index=1,
    )

    assert result["criteria"]["numeric_value_correct"] is None
    assert result["passed"] is None
    assert result["answer_quality_status"] == "not_evaluated_stub"


def test_provider_without_stub_flag_is_not_assumed_production():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "In 2024, AuM was SAR 3,424 billion [1].",
            "generation": {"provider": "omniroute"},
            "retrieval": {"n_chunks": 1},
            "citations": [{
                "index": 1, "source_filename": "2024-en.pdf",
                "physical_page": 6,
            }],
        },
        index=1,
    )

    assert result["criteria"]["numeric_value_correct"] is None
    assert result["passed"] is None
    assert result["answer_quality_status"] == "not_evaluated_unknown_generation"


def test_citation_page_match_requires_a_citation_reference_in_answer():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?",
            "expected": "SAR 3,424 billion",
            "expected_numbers": ["2024", "SAR 3,424 billion"],
            "source_page": 6,
        },
        {
            "answer": "In 2024, AuM was SAR 3,424 billion [2].",
            "generation": {"provider": "omniroute", "is_stub": False},
            "retrieval": {"n_chunks": 1},
            "verification": {"outcome": "supported"},
            "citations": [{
                "index": 1, "source_filename": "2024-en.pdf",
                "physical_page": 6,
            }],
        },
        index=1,
    )

    assert result["criteria"]["citation_correctness"] is False
    assert result["passed"] is False


def test_page_candidate_is_not_claimed_as_context_evidence_without_exact_text():
    base_item = {
        "question": "What was AuM in 2024?",
        "expected": "SAR 3,424 billion",
        "expected_numbers": ["2024", "SAR 3,424 billion"],
        "source_page": 6,
    }
    legacy = checker._evaluate_item(
        base_item,
        {
            "answer": "Offline test stub: no answer.",
            "generation": {"provider": "stub", "is_stub": True},
            "retrieval": {"n_chunks": 1, "candidates": [{
                "source_filename": "2024-en.pdf", "physical_page": 6,
                "context_supplied": True,
            }]},
        },
        index=1,
    )
    truncated = checker._evaluate_item(
        base_item,
        {
            "answer": "Offline test stub: no answer.",
            "generation": {"provider": "stub", "is_stub": True},
            "retrieval": {"n_chunks": 1, "candidates": [{
                "source_filename": "2024-en.pdf", "physical_page": 6,
                "context_supplied": True, "generation_context_text": "",
            }]},
        },
        index=1,
    )

    assert legacy["criteria"]["expected_page_candidate_in_generation_context"] is None
    assert truncated["criteria"]["expected_page_candidate_in_generation_context"] is False


def test_evaluation_reports_top_five_and_top_ten_with_rank_and_page_details():
    candidates = [
        {
            "source_filename": "other.pdf" if index == 1 else "2024-en.pdf",
            "document_id": f"doc-{index}", "chunk_id": f"chunk-{index}",
            "physical_page": 99 if index == 1 else (6 if index == 8 else index),
            "page": 12, "dense_score": 0.9 / index,
            "bm25_score": 0.5 / index, "rerank_score": 1 / index,
        }
        for index in range(1, 11)
    ]
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?", "expected": "SAR 3,424 billion",
            "expected_numbers": ["SAR 3,424 billion"], "source_page": 6,
        },
        {
            "answer": "SAR 3,424 billion [1].",
            "generation": {"provider": "stub", "is_stub": True},
            "retrieval": {"n_chunks": 10, "candidates": candidates},
            "citations": [{"index": 1, "source_filename": "2024-en.pdf",
                           "physical_page": 6}],
        },
        index=1,
    )

    assert result["top_k_retrieval"]["5"]["source_document_retrieved"] is True
    assert result["top_k_retrieval"]["5"]["exact_source_physical_page_retrieved"] is False
    assert result["top_k_retrieval"]["10"]["exact_source_physical_page_retrieved"] is True
    assert result["top_k_retrieval"]["10"]["results"][7]["position"] == 8
    assert result["top_k_retrieval"]["10"]["results"][7]["rerank_score"] == 0.125
    assert result["citation_markers"]["extracted_indexes"] == [1]


def test_evaluation_marks_k_ten_unavailable_when_api_returns_eight_candidates():
    candidates = [
        {"source_filename": "2024-en.pdf", "physical_page": 6}
        for _ in range(8)
    ]
    result = checker._evaluate_item(
        {"question": "What was AuM?", "expected": "", "source_page": 6},
        {
            "answer": "", "generation": {"provider": "stub", "is_stub": True},
            "retrieval": {"n_chunks": 8, "candidates": candidates},
            "citations": [],
        },
        index=1,
    )
    assert result["top_k_retrieval"]["5"]["k_complete"] is True
    assert result["top_k_retrieval"]["10"]["k_complete"] is False
    assert result["top_k_retrieval"]["10"]["source_document_retrieved"] is None
    assert result["top_k_retrieval"]["10"]["exact_source_physical_page_retrieved"] is None


def test_evaluation_reports_expected_page_through_retrieval_stages():
    candidates = [
        {
            "source_filename": "2024-en_2024-en.pdf", "physical_page": 9,
            "chunk_id": "hit", "context_supplied": False,
        }
    ]
    diagnostics = [
        {
            "fused_position": 4, "rerank_input_position": 4,
            "rerank_output_position": 2, "returned_position": 6,
            "source_filename": "2024-en_2024-en.pdf", "physical_page": 9,
            "reranked": True,
        }
    ]
    result = checker._evaluate_item(
        {
            "question": "What was AuM?", "expected": "", "source_page": 9,
            "expected_source_filenames": ["2024-en.pdf"],
        },
        {
            "answer": "", "generation": {"provider": "stub", "is_stub": True},
            "citations": [],
            "retrieval": {
                "n_chunks": 8, "candidates": candidates,
                "candidate_diagnostics": diagnostics,
                "filter_diagnostics": {
                    "applied_filter": {"workspace_id": "ws", "status": "active"},
                    "metadata_page_inventory": [{
                        "source_filename": "2024-en_2024-en.pdf",
                        "document_id": "doc", "physical_page": 9,
                        "chunk_count": 3,
                    }],
                },
            },
        },
        index=1,
    )
    stages = result["retrieval_stages"]
    assert stages["expected_page_in_metadata_filtered_inventory"] is True
    assert stages["expected_page_fused_candidate_positions"] == [4]
    assert stages["expected_page_reranker_output_positions"] == [2]
    assert stages["expected_page_api_returned_positions"] == [6]


def test_malformed_answer_citation_marker_is_reported_without_being_resolved():
    result = checker._evaluate_item(
        {
            "question": "Question", "expected": "Fact",
            "expected_numbers": [], "source_page": 6,
        },
        {
            "answer": "A fact [page 6].",
            "generation": {"provider": "omniroute", "is_stub": False},
            "retrieval": {"n_chunks": 1, "candidates": []},
            "citations": [],
        },
        index=1,
    )

    assert result["citation_markers"]["malformed_markers"] == ["[page 6]"]
    assert result["citation_markers"]["referenced_indexes_resolved"] == []
    assert result["criteria"]["citation_correctness"] is False


def test_reported_metrics_map_to_real_criteria_keys():
    # A metric that reads a criterion name the evaluator never writes is
    # silently reported as 0/0, hiding a stage that was actually measured.
    criteria_keys = set(checker._evaluate_item(
        {
            "question": "What was AuM in 2024?", "expected": "SAR 3,424 billion",
            "expected_numbers": ["SAR 3,424 billion"], "source_page": 6,
            "expected_source_filenames": ["2024-en.pdf"],
        },
        {
            "answer": "PIF reported SAR 3,424 billion [1].",
            "generation": {"provider": "omniroute", "is_stub": False},
            "verification": {"outcome": "supported"},
            "citations": [{"index": 1, "source_filename": "2024-en.pdf",
                           "physical_page": 6}],
            "retrieval": {
                "n_chunks": 1,
                "candidates": [{
                    "source_filename": "2024-en.pdf", "physical_page": 6,
                    "context_supplied": True,
                    "generation_context_text": "PIF reported SAR 3,424 billion",
                }],
            },
        },
        index=1,
    )["criteria"])

    unmapped = {
        name: criterion
        for name, criterion in checker.METRIC_FIELDS.items()
        if criterion not in criteria_keys
    }
    assert unmapped == {}


def test_generation_context_metric_is_counted_not_dropped():
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?", "expected": "SAR 3,424 billion",
            "expected_numbers": ["SAR 3,424 billion"], "source_page": 6,
            "expected_source_filenames": ["2024-en.pdf"],
        },
        {
            "answer": "PIF reported SAR 3,424 billion [1].",
            "generation": {"provider": "omniroute", "is_stub": False},
            "verification": {"outcome": "supported"},
            "citations": [{"index": 1, "source_filename": "2024-en.pdf",
                           "physical_page": 6}],
            "retrieval": {
                "n_chunks": 1,
                "candidates": [{
                    "source_filename": "2024-en.pdf", "physical_page": 6,
                    "context_supplied": True,
                    "generation_context_text": "PIF reported SAR 3,424 billion",
                }],
            },
        },
        index=1,
    )

    assert (result["criteria"]
            ["expected_page_candidate_in_generation_context"] is True)


def test_page_diagnostics_never_report_the_ambiguous_page_field_as_printed():
    printed = 16
    physical = 9
    candidate = {
        "source_filename": "2024-en.pdf", "document_id": "doc", "chunk_id": "chunk",
        # The API's "page" carries the printed page when one exists, otherwise
        # the physical page, so diagnostics must read the dedicated fields.
        "page": printed, "printed_page": printed, "physical_page": physical,
    }
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?", "expected": "", "source_page": physical,
            "expected_source_filenames": ["2024-en.pdf"],
        },
        {
            "answer": "", "generation": {"provider": "stub", "is_stub": True},
            "citations": [dict(candidate, index=1)],
            "retrieval": {"n_chunks": 1, "candidates": [candidate]},
        },
        index=1,
    )

    row = result["top_k_retrieval"]["5"]["results"][0]
    assert row["printed_page"] == printed
    assert row["physical_page"] == physical
    assert result["sources"][0]["printed_page"] == printed
    assert result["sources"][0]["physical_page"] == physical

    physical_only = dict(candidate, page=physical, printed_page=None)
    result = checker._evaluate_item(
        {
            "question": "What was AuM in 2024?", "expected": "",
            "source_page": physical, "expected_source_filenames": ["2024-en.pdf"],
        },
        {
            "answer": "", "generation": {"provider": "stub", "is_stub": True},
            "citations": [dict(physical_only, index=1)],
            "retrieval": {"n_chunks": 1, "candidates": [physical_only]},
        },
        index=1,
    )

    assert result["top_k_retrieval"]["5"]["results"][0]["printed_page"] is None
    assert result["sources"][0]["printed_page"] is None


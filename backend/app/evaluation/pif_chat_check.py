"""Scoring helpers and CLI for the local PIF chat evaluation.

The source corpus and credentials remain under the ignored ``data/`` directory;
this maintained implementation lives with the backend evaluation code.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIF_DIR = REPO_ROOT / "data" / "kaggle" / "pif-annual-reports"
BASE = "http://127.0.0.1:8010"
EXPECTED_SOURCE_FILES = {
    "pif-001": ["2024-en.pdf"], "pif-002": ["2024-en.pdf"],
    "pif-003": ["2024-en.pdf"], "pif-004": ["2024-en.pdf"],
    "pif-005": ["2024-en.pdf"], "pif-006": ["2024-en.pdf"],
    "pif-007": ["2024-en.pdf"],
    "pif-008": ["2023-en.pdf", "2024-en.pdf"],
    "pif-009": ["2024-en.pdf"], "pif-010": ["2024-en.pdf"],
}


GOLD_V2_FILENAME = "pif_gold_qa_v2.jsonl"
GOLD_V1_FILENAME = "pif_gold_qa.jsonl"
# A question is only scoreable if the grader can name the document it expects.
REQUIRED_GOLD_FIELDS = ("id", "question", "expected", "source_page",
                        "expected_source_filenames")


class GoldFormatError(ValueError):
    """The gold file cannot be scored as written."""


def gold_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_gold_fields(item: dict[str, Any], *, filename: str,
                         line_number: int) -> None:
    where = f"{filename}:{line_number}"
    missing = [field for field in REQUIRED_GOLD_FIELDS
               if item.get(field) in (None, "", [])]
    if missing:
        raise GoldFormatError(f"{where} is missing {missing}")
    page = item["source_page"]
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise GoldFormatError(f"{where} has source_page {page!r}; expected an "
                              "integer physical page >= 1")
    names = item["expected_source_filenames"]
    if not isinstance(names, list) or not all(
            isinstance(name, str) and name.strip() for name in names):
        raise GoldFormatError(f"{where} has expected_source_filenames {names!r}; "
                              "expected a list of non-empty filenames")


def load_gold(
    pif_dir: Path = DEFAULT_PIF_DIR,
    *,
    filename: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the gold set and say, in provenance, how attribution was decided.

    The versioned file carries stable IDs and explicit expected source filenames,
    so nothing here depends on line order. The legacy file depends on both, so it
    is only accepted when the versioned file is absent, and provenance records
    which fields were derived rather than stated.
    """
    chosen = Path(filename) if filename else None
    if chosen is None:
        candidate = pif_dir / GOLD_V2_FILENAME
        chosen = candidate if candidate.exists() else pif_dir / GOLD_V1_FILENAME
    path = chosen if chosen.is_absolute() else pif_dir / chosen.name
    if not path.exists():
        raise GoldFormatError(f"gold file not found: {path}")

    versioned = path.name == GOLD_V2_FILENAME
    items: list[dict[str, Any]] = []
    derived: list[str] = []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        item = dict(raw)
        if not item.get("id"):
            if versioned:
                raise GoldFormatError(f"{path.name}:{line_number} has no stable id")
            item["id"] = f"pif-{len(items) + 1:03d}"
            derived.append(f"{item['id']}: id from line number")
        if not item.get("expected_source_filenames"):
            if versioned:
                raise GoldFormatError(
                    f"{path.name}:{line_number} ({item['id']}) declares no "
                    "expected source filenames")
            from_table = EXPECTED_SOURCE_FILES.get(item["id"])
            if not from_table:
                raise GoldFormatError(
                    f"{path.name}:{line_number} ({item['id']}) has no filenames and "
                    "no EXPECTED_SOURCE_FILES entry; it cannot be scored")
            item["expected_source_filenames"] = list(from_table)
            derived.append(f"{item['id']}: filenames from EXPECTED_SOURCE_FILES")
        _require_gold_fields(item, filename=path.name, line_number=line_number)
        item.setdefault("expected_numbers", [])
        item.setdefault("expected_keywords", [])
        items.append(item)

    if limit is not None:
        items = items[:limit]
    provenance = {
        "gold_file": str(path),
        "gold_sha256": gold_sha256(path),
        "gold_version": "v2" if versioned else "v1",
        "attribution": "explicit in gold file" if not derived
                       else "derived (line order + hardcoded table)",
        "questions": len(items),
        "derived_fields": derived,
    }
    return items, provenance


def normalize_duplicate_source_filename(value: Any) -> Any:
    """Remove only an exactly repeated stem suffix before the extension.

    Example: ``2024-en_2024-en.pdf`` -> ``2024-en.pdf``. The original value
    remains in each diagnostic so normalization never hides source metadata.
    """
    if not isinstance(value, str):
        return value
    path = Path(value)
    stem, suffix = path.stem, path.suffix
    left, separator, right = stem.rpartition("_")
    if separator and left and left.casefold() == right.casefold():
        return str(path.with_name(left + suffix))
    return value


def chat(question: str, *, api_key: str, workspace_id: str,
         include_evidence_trace: bool = True) -> dict[str, Any]:
    req = urllib.request.Request(
        BASE + "/chat",
        data=json.dumps({
            "question": question, "workspace_id": workspace_id,
            "include_evidence_trace": include_evidence_trace,
        }).encode(),
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.loads(response.read())


def cited_physical_pages(citations: list[dict[str, Any]]) -> set[int]:
    """Extract physical PDF indices; PIF gold ``source_page`` uses this index."""
    return {
        citation["physical_page"]
        for citation in citations
        if isinstance(citation.get("physical_page"), int)
        and not isinstance(citation.get("physical_page"), bool)
    }


def page_match(citations: list[dict[str, Any]], source_page: int | None) -> bool | None:
    if source_page is None:
        return None
    pages = cited_physical_pages(citations)
    return source_page in pages if pages else None


def numeric_match(expected_numbers: list[str], answer: str) -> bool | None:
    """Check numeric tokens independently from currency/unit markers."""
    if not expected_numbers:
        return None
    actual = {
        value.replace(",", "")
        for value in re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer)
    }
    expected_values = [
        number.replace(",", "")
        for expected in expected_numbers
        for number in re.findall(r"-?\d[\d,]*(?:\.\d+)?", expected)
    ]
    return bool(expected_values) and all(number in actual for number in expected_values)


def unit_markers_match(expected_numbers: list[str], answer: str) -> bool | None:
    """Check currency, scale, and percentage markers separately from values."""
    def markers(text: str) -> set[str]:
        found = re.findall(
            r"\b(?:SAR|USD|EUR|GBP|billion|bn|million|mn|trillion|"
            r"percent|percentage points|pp)\b|[$€£%]",
            text,
            re.I,
        )
        normalized = set()
        for marker in found:
            value = marker.casefold()
            normalized.add({
                "$": "USD", "€": "EUR", "£": "GBP", "%": "%",
                "billion": "BN", "bn": "BN", "million": "MN", "mn": "MN",
                "trillion": "TN", "percent": "%", "percentage points": "PP",
                "pp": "PP",
            }.get(value, marker.upper()))
        return normalized

    expected_markers = set().union(*(markers(x) for x in expected_numbers))
    if not expected_markers:
        return None
    # Keep currency/scale checks attached to the expected number. Merely
    # mentioning SAR in one clause and the right scale in another is not a
    # correct unit for the value being scored.
    for expected in expected_numbers:
        expected_value_markers = markers(expected)
        if not expected_value_markers:
            continue
        expected_values = {
            value.replace(",", "")
            for value in re.findall(r"-?\d[\d,]*(?:\.\d+)?", expected)
        }
        found = False
        answer_matches = list(re.finditer(r"-?\d[\d,]*(?:\.\d+)?", answer))
        for index, match in enumerate(answer_matches):
            value = match.group(0).replace(",", "")
            if value not in expected_values:
                continue
            start = max(0, match.start() - 16)
            end = min(len(answer), match.end() + 16)
            if index:
                start = max(start, answer_matches[index - 1].end())
            if index + 1 < len(answer_matches):
                end = min(end, answer_matches[index + 1].start())
            nearby = answer[start:end]
            if expected_value_markers <= markers(nearby):
                found = True
                break
        if not found:
            return False
    return True


METRIC_FIELDS = {
    "retrieval_succeeded": "retrieval_succeeded",
    "retrieval_evidence_success": "retrieval_evidence_success",
    "correct_source_document_retrieved": "correct_source_document_retrieved",
    "supporting_physical_page_retrieved": "supporting_physical_page_retrieved",
    "supporting_page_in_generation_context":
        "expected_page_candidate_in_generation_context",
    "answer_support_by_cited_evidence": "answer_support_by_cited_evidence",
    "numeric_value_correct": "numeric_value_correct",
    "currency_and_unit_correct": "currency_and_unit_correct",
    "citation_correctness": "citation_correctness",
}


def _verification_supported(verification: dict[str, Any] | None) -> bool | None:
    if not verification:
        return None
    outcome = str(verification.get("outcome", "")).casefold()
    if outcome == "supported":
        return True
    if outcome in {"partial", "rejected"}:
        return False
    return None


def _flatten_diagnostic_rows(value: Any) -> list[dict[str, Any]]:
    """Flatten the single-query and fan-out diagnostic shapes."""
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_flatten_diagnostic_rows(item))
        return rows
    if isinstance(value, dict):
        if "subqueries" in value:
            return _flatten_diagnostic_rows(value["subqueries"])
        return [value]
    return []


def _evaluate_item(
    item: dict[str, Any], response: dict[str, Any], *, index: int,
) -> dict[str, Any]:
    answer = response.get("answer") or ""
    expected_numbers = item.get("expected_numbers", [])
    citations = response.get("citations") or []
    retrieval = response.get("retrieval") or {}
    candidate_diagnostics = retrieval.get("candidate_diagnostics") or []
    filter_diagnostics = retrieval.get("filter_diagnostics") or {}
    verification = response.get("verification")
    generation = response.get("generation") or {}
    provider = generation.get("provider")
    is_stub = generation.get("is_stub") is True or (
        isinstance(provider, str) and provider.strip().casefold() == "stub"
    )
    generation_known = (
        isinstance(provider, str) and bool(provider.strip())
        and type(generation.get("is_stub")) is bool
    )
    answer_evaluable = generation_known and not is_stub
    trace = retrieval.get("candidates") or citations
    trace_count = len(trace)
    question_id = item.get("id") or f"pif-{index:03d}"
    # load_gold guarantees the explicit field; the legacy table is only the
    # fallback for callers that pass a raw pre-v2 item.
    expected_sources = [
        normalize_duplicate_source_filename(source)
        for source in item.get("expected_source_filenames")
        or EXPECTED_SOURCE_FILES.get(question_id, [])]
    source_ok = (
        any(normalize_duplicate_source_filename(c.get("source_filename"))
            in expected_sources for c in trace)
        if expected_sources else None
    )
    page = item.get("source_page")
    def in_top_k(predicate, k: int) -> bool | None:
        # A short response is not silently presented as a complete @k list.
        if len(trace) < k:
            return None
        return any(predicate(candidate) for candidate in trace[:k])

    def expected_source(candidate: dict[str, Any]) -> bool:
        return bool(expected_sources) and normalize_duplicate_source_filename(
            candidate.get("source_filename")
        ) in [normalize_duplicate_source_filename(source)
              for source in expected_sources]

    def expected_source_exact(candidate: dict[str, Any]) -> bool:
        return bool(expected_sources) and candidate.get("source_filename") in expected_sources

    def expected_page(candidate: dict[str, Any]) -> bool:
        return (
            page is not None
            and candidate.get("physical_page") == page
            and (not expected_sources or expected_source(candidate))
        )

    top_k_diagnostics = {}
    for k in (5, 10):
        top_k_diagnostics[str(k)] = {
            "observed_result_count": min(len(trace), k),
            "returned_candidate_count": trace_count,
            "k_complete": trace_count >= k,
            "source_document_retrieved_exact_string": (
                in_top_k(expected_source_exact, k) if expected_sources else None
            ),
            "source_document_retrieved": (
                in_top_k(expected_source, k) if expected_sources else None
            ),
            "exact_source_physical_page_retrieved": (
                in_top_k(expected_page, k) if page is not None else None
            ),
            "results": [
                {
                    "position": position,
                    "source_filename": candidate.get("source_filename"),
                    "normalized_source_filename": normalize_duplicate_source_filename(
                        candidate.get("source_filename")
                    ),
                    "document_id": candidate.get("document_id"),
                    "chunk_id": candidate.get("chunk_id"),
                    "physical_page": candidate.get("physical_page"),
                    "printed_page": candidate.get("printed_page"),
                    "section": candidate.get("section"),
                    "dense_score": candidate.get("dense_score"),
                    "bm25_score": candidate.get("bm25_score"),
                    "rerank_score": candidate.get("rerank_score"),
                    "context_supplied": candidate.get("context_supplied"),
                }
                for position, candidate in enumerate(trace[:k], start=1)
            ],
        }
    page_retrieved = (
        any(c.get("physical_page") == page
            and (not expected_sources or
                 expected_source(c))
            for c in trace)
        if page is not None else None
    )
    page_candidate_in_context = (
        any(c.get("physical_page") == page
            and c.get("context_supplied") is True
            and bool(c.get("generation_context_text"))
            and (not expected_sources or
                 expected_source(c))
            for c in trace)
        if page is not None and any("generation_context_text" in c for c in trace)
        else None
    )
    metadata_inventory = _flatten_diagnostic_rows(
        [d.get("metadata_page_inventory", [])
         for d in _flatten_diagnostic_rows(filter_diagnostics)]
    )
    metadata_page_matches = [
        row for row in metadata_inventory
        if row.get("physical_page") == page
        and (not expected_sources or expected_source(row))
    ]
    retrieved_page_candidates = [
        row for row in candidate_diagnostics
        if row.get("physical_page") == page
        and (not expected_sources or expected_source(row))
    ]
    reranker_page_candidates = [
        row for row in retrieved_page_candidates if row.get("reranked") is True
    ]
    returned_page_candidates = [
        row for row in retrieved_page_candidates
        if isinstance(row.get("returned_position"), int)
    ]
    numbers_ok = numeric_match(expected_numbers, answer)
    units_ok = unit_markers_match(expected_numbers, answer)
    canonical_answer = re.sub(r"【(\d+)】", r"[\1]", answer)
    referenced_indexes = {
        int(index) for index in re.findall(r"\[(\d+)\]", canonical_answer)
    }
    malformed_markers = re.findall(
        r"【[^】]*】|\[(?!\d+\])[^\]]*\]", answer
    )
    answer_citations = [
        citation for citation in citations
        if citation.get("index") in referenced_indexes
    ]
    citation_ok = (
        any(c.get("physical_page") == page
            and (not expected_sources or
                 normalize_duplicate_source_filename(c.get("source_filename"))
                 in [normalize_duplicate_source_filename(source)
                     for source in expected_sources])
            for c in answer_citations)
        if page is not None and answer_evaluable else None
    )
    supported = _verification_supported(verification)
    criteria = {
        "retrieval_succeeded": bool(retrieval.get("n_chunks", 0)),
        "retrieval_evidence_success": bool(source_ok and page_retrieved),
        "correct_source_document_retrieved": source_ok,
        "supporting_physical_page_retrieved": page_retrieved,
        # This is candidate-page inclusion only. Passage-level support is
        # assessed separately from the exact context and answer verification.
        "expected_page_candidate_in_generation_context": page_candidate_in_context,
        "answer_present": bool(answer.strip()) if answer_evaluable else None,
        "answer_support_by_cited_evidence": supported if answer_evaluable else None,
        "numeric_value_correct": numbers_ok if answer_evaluable else None,
        "currency_and_unit_correct": units_ok if answer_evaluable else None,
        "citation_correctness": citation_ok,
    }
    failures = [name for name, value in criteria.items() if value is False]
    failures.extend(
        f"{name}_not_verified"
        for name, value in criteria.items()
        if value is None and name in {
            "correct_source_document_retrieved",
            "supporting_physical_page_retrieved", "citation_correctness",
        }
    )
    e2e_checks = [
        criteria[name] for name in (
            "correct_source_document_retrieved",
            "supporting_physical_page_retrieved", "answer_present",
            "expected_page_candidate_in_generation_context",
            "answer_support_by_cited_evidence", "numeric_value_correct",
            "currency_and_unit_correct", "citation_correctness",
        )
    ]
    passed = None if is_stub or not generation_known else (
        all(value is True for value in e2e_checks)
        if all(value is not None for value in e2e_checks) else False
    )
    return {
        "question_id": question_id,
        "question": item["question"],
        "expected": item.get("expected", ""),
        "expected_numbers": expected_numbers,
        "expected_physical_page": item.get("source_page"),
        "actual": answer,
        "generation": generation or {"provider": "unknown", "is_stub": None},
        "query": retrieval.get("query"),
        "query_plan": retrieval.get("plan") or response.get("plan"),
        "retrieved_evidence": trace,
        "candidate_diagnostics": candidate_diagnostics,
        "retrieval_stages": {
            "expected_page_in_metadata_filtered_inventory": (
                bool(metadata_page_matches) if metadata_inventory else None
            ),
            "expected_page_metadata_candidates": [
                {key: row.get(key) for key in (
                    "source_filename", "document_id", "physical_page", "chunk_count"
                )}
                for row in metadata_page_matches
            ],
            "expected_page_fused_candidate_positions": [
                row.get("fused_position") for row in retrieved_page_candidates
            ],
            "expected_page_reranker_input_positions": [
                row.get("rerank_input_position") for row in reranker_page_candidates
            ],
            "expected_page_reranker_output_positions": [
                row.get("rerank_output_position") for row in reranker_page_candidates
            ],
            "expected_page_api_returned_positions": [
                row.get("returned_position") for row in returned_page_candidates
            ],
            "filter_diagnostics": filter_diagnostics,
        },
        "generation_context": retrieval.get("generation_context"),
        "generation_messages": retrieval.get("generation_messages"),
        "expected_source_filenames": expected_sources,
        "candidate_counts": {
            "returned_by_api": retrieval.get("candidate_counts", {}).get(
                "returned_by_api", trace_count
            ),
            "retrieved": retrieval.get("candidate_counts", {}).get("retrieved"),
            "reranked": retrieval.get("candidate_counts", {}).get("reranked"),
            "included_in_final_context": sum(
                candidate.get("context_supplied") is True for candidate in trace
            ),
        },
        "top_k_retrieval": top_k_diagnostics,
        "citation_markers": {
            "extracted_indexes": sorted(referenced_indexes),
            "malformed_markers": malformed_markers,
            "referenced_indexes_resolved": sorted(
                citation.get("index") for citation in citations
                if citation.get("index") in referenced_indexes
            ),
        },
        "sources": [
            {
                "document_id": citation.get("document_id"),
                "source_filename": citation.get("source_filename"),
                "printed_page": citation.get("printed_page"),
                "physical_page": citation.get("physical_page"),
                "section": citation.get("section"),
                "snippet": citation.get("snippet", ""),
            }
            for citation in citations
        ],
        "citations": citations,
        "retrieval": retrieval,
        "retrieval_outcome": "retrieved" if retrieval.get("n_chunks", 0) else "no_context",
        "generation_verification": verification,
        "criteria": criteria,
        "passed": passed,
        "end_to_end_pass": passed,
        "answer_quality_status": (
            "not_evaluated_stub" if is_stub else
            "not_evaluated_unknown_generation" if not generation_known else
            "evaluated"
        ),
        "failure_reasons": failures,
    }


def run(limit: int = 3, pif_dir: Path = DEFAULT_PIF_DIR) -> int:
    """Run against an explicitly configured local backend and save raw evidence."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Do not print state contents; it contains the local API key.
    state = json.loads((pif_dir / "pif_state.json").read_text(encoding="utf-8"))
    api_key, workspace_id = state["api_key"], state["workspace_id"]
    items, gold_provenance = load_gold(pif_dir, limit=limit)
    print(f"gold={gold_provenance['gold_file']} "
          f"sha256={gold_provenance['gold_sha256']} "
          f"version={gold_provenance['gold_version']} "
          f"attribution={gold_provenance['attribution']} "
          f"questions={gold_provenance['questions']}")
    if gold_provenance["derived_fields"]:
        print("derived attribution: " + "; ".join(gold_provenance["derived_fields"]))
    request_failures = retrieval_failures = generation_failures = 0
    numeric_hits = numeric_checks = unit_hits = unit_checks = 0
    cited = page_hits = page_checks = 0
    results: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        started = time.time()
        try:
            response = chat(item["question"], api_key=api_key,
                            workspace_id=workspace_id)
        except Exception as exc:  # noqa: BLE001 - keep per-item failures in the report
            request_failures += 1
            results.append({
                "question_id": item.get("id") or f"pif-{index:03d}", "question": item["question"],
                "expected": item.get("expected", ""),
                "expected_numbers": item.get("expected_numbers", []),
                "expected_physical_page": item.get("source_page"), "actual": None,
                "sources": [], "citations": [], "retrieval": None,
                "retrieval_outcome": "not_observed", "criteria": {},
                "passed": False, "failure_reasons": ["request_failed"],
                "failure_stage": "request",
                "error": str(exc), "latency_seconds": round(time.time() - started, 3),
            })
            continue

        answer = response.get("answer") or ""
        if not answer.strip():
            generation_failures += 1
        result = _evaluate_item(item, response, index=index)
        result["failure_stage"] = "generation" if not answer.strip() else None
        result["error"] = None
        result["latency_seconds"] = round(time.time() - started, 3)
        results.append(result)
        retrieval_failures += result["retrieval_outcome"] == "no_context"
        numeric_value = result["criteria"]["numeric_value_correct"]
        unit_value = result["criteria"]["currency_and_unit_correct"]
        page_value = result["criteria"]["citation_correctness"]
        numeric_hits += bool(numeric_value)
        numeric_checks += numeric_value is not None
        if unit_value is not None:
            unit_hits += bool(unit_value)
            unit_checks += 1
        cited += bool(result["citations"])
        page_hits += bool(page_value)
        page_checks += page_value is not None
        print(f"{item['question'][:90]} | passed={result['passed']} "
              f"failures={result['failure_reasons']}")

    report_path = pif_dir / "pif_eval_results.json"
    if report_path.exists():
        report_path = pif_dir / f"pif_eval_results_{time.time_ns()}.json"
    metric_counts = {}
    for metric_name, criterion_name in METRIC_FIELDS.items():
        values = [
            result.get("criteria", {}).get(criterion_name)
            for result in results
        ]
        checked = [value for value in values if value is not None]
        metric_counts[metric_name] = {
            "passed": sum(value is True for value in checked),
            "checked": len(checked),
        }
    report = {
        "gold": gold_provenance,
        "asked": len(items), "successful_responses": len(results) - request_failures,
        "retrieval_failures": retrieval_failures, "request_failures": request_failures,
        "generation_failures": generation_failures, "numeric_matches": numeric_hits,
        "numeric_checks": numeric_checks, "unit_marker_matches": unit_hits,
        "unit_marker_checks": unit_checks, "responses_with_citations": cited,
        "citation_page_matches": page_hits, "citation_page_checks": page_checks,
        "passed": sum(result.get("passed") is True for result in results),
        "failed": sum(result.get("passed") is False for result in results),
        "not_evaluated_stub": sum(
            result.get("answer_quality_status") == "not_evaluated_stub"
            for result in results
        ),
        "not_evaluated_unknown_generation": sum(
            result.get("answer_quality_status") ==
            "not_evaluated_unknown_generation"
            for result in results
        ),
        "retrieval_evidence_successes": sum(
            bool(result.get("criteria", {}).get("retrieval_evidence_success"))
            for result in results
        ),
        "source_document_matches": sum(
            result.get("criteria", {}).get("correct_source_document_retrieved") is True
            for result in results
        ),
        "supporting_page_retrievals": sum(
            result.get("criteria", {}).get("supporting_physical_page_retrieved") is True
            for result in results
        ),
        "retrieval_at_k": {
            str(k): {
                "source_document_recall": {
                    "hits": sum(
                        result.get("top_k_retrieval", {}).get(str(k), {}).get(
                            "source_document_retrieved"
                        ) is True
                        for result in results
                    ),
                    "checked": sum(
                        result.get("top_k_retrieval", {}).get(str(k), {}).get(
                            "source_document_retrieved"
                        ) is not None
                        for result in results
                    ),
                },
                "source_document_recall_exact_string": {
                    "hits": sum(
                        result.get("top_k_retrieval", {}).get(str(k), {}).get(
                            "source_document_retrieved_exact_string"
                        ) is True
                        for result in results
                    ),
                    "checked": sum(
                        result.get("top_k_retrieval", {}).get(str(k), {}).get(
                            "source_document_retrieved_exact_string"
                        ) is not None
                        for result in results
                    ),
                },
                "exact_source_physical_page_recall": {
                    "hits": sum(
                        result.get("top_k_retrieval", {}).get(str(k), {}).get(
                            "exact_source_physical_page_retrieved"
                        ) is True
                        for result in results
                    ),
                    "checked": sum(
                        result.get("top_k_retrieval", {}).get(str(k), {}).get(
                            "exact_source_physical_page_retrieved"
                        ) is not None
                        for result in results
                    ),
                },
            }
            for k in (5, 10)
        },
        "metrics": metric_counts,
        "items": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"asked={len(items)} request_failures={request_failures} "
          f"retrieval_failures={retrieval_failures} generation_failures={generation_failures} "
          f"retrieval_evidence={report['retrieval_evidence_successes']}/{len(items)} "
          f"source_docs={report['source_document_matches']}/{len(items)} "
          f"support_pages={report['supporting_page_retrievals']}/{len(items)} "
          f"numeric={numeric_hits}/{numeric_checks} units={unit_hits}/{unit_checks} "
          f"citations={page_hits}/{page_checks} e2e_pass={report['passed']}/{len(items)} "
          f"stub_not_evaluated={report['not_evaluated_stub']} "
          f"unknown_generation={report['not_evaluated_unknown_generation']}; "
          f"evidence={report_path}")
    return 0


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    return run(limit)


if __name__ == "__main__":
    sys.exit(main())

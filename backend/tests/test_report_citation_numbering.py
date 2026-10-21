"""Regressions for multi-section report citation bookkeeping.

Each report section runs its own retrieval, so the model's ``[1]`` is local to
that section. Sources collected from the whole retrieval — including chunks
the char budget never put in the prompt — made the Sources block disagree with
the markers above it.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.reasoning import reports
from app.retrieval.hybrid import RetrievalResult, RetrievedChunk

SECTIONS = ["Revenue and growth", "Earnings per share", "Cash flow highlights",
            "Key risks", "Outlook and guidance"]


def _chunk(chunk_id: str, text: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, parent_id=None, document_id=f"doc-{chunk_id}",
        text=text, parent_text=text, tables_md="", company="PIF",
        year=2024, quarter="", doc_type="annual_report", section="",
        page=page, source_filename=f"{page}-en.pdf", physical_page=page,
    )


class _FakeRetriever:
    def __init__(self, per_section: list[list[RetrievedChunk]]):
        self._per_section = per_section
        self._i = 0

    def retrieve(self, query, plan=None):
        chunks = self._per_section[self._i]
        self._i += 1
        return RetrievalResult(query=query, plan=None, chunks=chunks, note="")


def _build(monkeypatch, per_section, section_text) -> dict:
    """Run one report against a fake retriever and canned section prose."""
    answers = iter(section_text)
    monkeypatch.setattr(reports, "get_llm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(reports, "HybridRetriever",
                        lambda ctx: _FakeRetriever(per_section))
    monkeypatch.setattr(reports, "build_dispatcher", lambda ctx: (None, []))
    monkeypatch.setattr(reports, "list_metrics",
                        lambda ctx, company=None, fiscal_year=None: {})
    monkeypatch.setattr(
        reports, "run_tool_loop",
        lambda llm, msgs, tools=None, dispatch=None, max_iterations=1:
        (next(answers), ""))
    monkeypatch.setattr(reports, "SUMMARY_SECTIONS", SECTIONS[:len(per_section)])
    return reports.build_report(SimpleNamespace(), company="PIF")


def test_shared_source_keeps_one_report_wide_number(monkeypatch):
    a, b, c = _chunk("a", "Revenue grew.", 10), _chunk("b", "EPS data.", 20), \
        _chunk("c", "Cash flow data.", 30)
    result = _build(
        monkeypatch,
        per_section=[[a, b], [b, c]],
        # Section 1 cites its local 1 and 2. Section 2 cites its local 1 (=b)
        # and 2 (=c), so [1] must become [2] and [2] must become [3].
        section_text=["Revenue grew [1] and margins [2].", "EPS [1], cash [2]."],
    )

    assert "grew [1] and margins [2]" in result["report"]
    assert "EPS [2], cash [3]" in result["report"]
    assert [cit["chunk_id"] for cit in result["citations"]] == ["a", "b", "c"]
    assert [cit["index"] for cit in result["citations"]] == [1, 2, 3]


def test_sources_list_only_the_chunks_that_reached_the_prompt(monkeypatch):
    many = [_chunk(f"c{i}", f"Text {i}.", i) for i in range(1, 9)]
    result = _build(
        monkeypatch,
        per_section=[many],
        section_text=["Cited [1] and [5], nothing else."],
    )

    # context_with_chunks(max_chunks=5) supplies five; the other three were
    # retrieved but never shown to the model.
    assert [cit["chunk_id"] for cit in result["citations"]] == \
        ["c1", "c2", "c3", "c4", "c5"]
    sources = result["report"].split("## Sources\n")[1].splitlines()
    assert len(sources) == 5
    assert [line[:4] for line in sources] == \
        ["[1] ", "[2] ", "[3] ", "[4] ", "[5] "]


def test_renumber_drops_markers_outside_the_supplied_context():
    # This section was given two chunks, so [9] is the model's own invention.
    # Keeping it would point the reader at report-wide source 9 — a section it
    # was never part of.
    assert reports._renumber("See [9].", {1: 4, 2: 5}) == "See."
    assert reports._renumber("Revenue rose [9] in [1].", {1: 4}) == \
        "Revenue rose in [4]."


def test_renumber_does_not_touch_bracketed_years():
    assert reports._renumber("FY [2024] grew [1].", {1: 7}) == "FY [2024] grew [7]."


def test_source_line_falls_back_to_filename_and_keeps_one_space():
    line = reports._format_sources([{
        "company": "", "source_filename": "2024-en.pdf", "year": "2024",
        "quarter": "", "doc_type": "annual_report", "page": 18,
    }])
    assert line == "[1] 2024-en.pdf 2024 annual_report p.18"
    assert "  " not in line

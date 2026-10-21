"""Regressions for the numeric/unit veto in citation verification.

These cases came out of an isolated end-to-end run: the same question over the
same document was answered and cited on one attempt and abstained on the next,
purely because the model wrote "SAR 413,423 million" instead of
"413,423 SAR million", or "(+25%)" instead of "25%".  A veto that fires on
typography is not a safety check, and it was also failing to fire on genuinely
wrong units.
"""

from __future__ import annotations

import pytest

from app.citation import (
    CitationVerifier,
    VerificationOutcome,
    _governing_units,
    _has_numeric_or_unit_mismatch,
    _normalized_numbers,
)
from app.retrieval.hybrid import RetrievedChunk

# Shape of a real income-statement passage: the unit is declared once, on a
# digit-free header line, and every figure below it is bare.
TABLE_PASSAGE = (
    "Consolidated Statement of Profit or Loss for the year ended "
    "31 December 2024\n"
    "Amounts in SAR million\n"
    "2024 2023 Change % Change\n"
    "Total Revenue 413,423 331,360 82,063 25%\n"
    "Cost of revenue (189,164) (152,598) 36,566 24%\n"
)


def _mismatch(claim: str, context: str = TABLE_PASSAGE) -> bool:
    return _has_numeric_or_unit_mismatch(claim, context, [context])


def _chunk(text: str, page: int = 75) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="revenue", parent_id=None, document_id="doc-1",
        text=text, parent_text=text, tables_md="", company="PIF",
        year=2024, quarter="", doc_type="annual_report", section="",
        page=page, source_filename="2024-en.pdf", physical_page=page,
    )


@pytest.mark.parametrize("claim", [
    "Total revenue was **413,423 SAR million** in 2024, up from **331,360 SAR "
    "million** in 2023 - an increase of **82,063 SAR million** (a **25%** rise)",
    "Total revenue for the year ended 31 December 2024 was **SAR 413,423 "
    "million**, compared with **SAR 331,360 million** in 2023.",
    "This represents an increase of **SAR 82,063 million**, or **25 %** "
    "year-on-year",
    "Total revenue was **413,423 SAR million** in 2024, an increase of "
    "**82,063 SAR million** (+25%) over 331,360 SAR million in 2023",
    "Total revenue was **413,423 million** in 2024",
])
def test_correctly_quoted_table_figures_are_not_a_mismatch(claim: str):
    assert not _mismatch(claim)


@pytest.mark.parametrize("claim", [
    "Total revenue was **413,423 SAR billion** in 2024",
    "Total revenue was **413,423 USD million** in 2024",
    "Total revenue was **413,423 trillion** in 2024",
    "Total revenue was **413,424 SAR million** in 2024",
])
def test_wrong_unit_or_value_is_still_vetoed(claim: str):
    assert _mismatch(claim)


def test_units_are_read_from_either_side_of_the_amount():
    text = "was SAR 413,423 million"
    start, end = text.index("413,423"), text.index("413,423") + 7
    assert _governing_units(text, start, end) == ("SAR", "million")
    flipped = "was 413,423 SAR million"
    start, end = flipped.index("413,423"), flipped.index("413,423") + 7
    assert _governing_units(flipped, start, end) == ("SAR", "million")


def test_sign_is_typography_not_magnitude():
    assert _normalized_numbers("(+25%) fell 5, while 1,234 held") == {
        "25", "5", "1234"}


def test_bn_abbreviation_and_riyal_glyph_still_bind():
    bn_context = (
        "Balance sheet summary\n"
        "Amounts in SAR bn\n"
        "Assets under Management 3,424 2,871\n"
    )
    assert not _has_numeric_or_unit_mismatch(
        "AuM reached **SAR 3,424 billion**.", bn_context, [bn_context])

    glyph_context = (
        "Consolidated figures\n"
        "Amounts in million\n"
        "Total Revenue ª 413,423 ª 331,360\n"
    )
    assert not _has_numeric_or_unit_mismatch(
        "Total revenue was **SAR 413,423 million**.", glyph_context,
        [glyph_context])


def test_units_bind_per_cited_chunk_not_across_the_whole_context():
    other = "Amounts in USD million\nRevenue 413,423 331,360\n"
    assert not _has_numeric_or_unit_mismatch(
        "Total revenue was **USD 413,423 million**.", other, [other])
    assert _has_numeric_or_unit_mismatch(
        "Total revenue was **SAR 413,423 million**.", other, [other])


def test_verifier_supports_the_answer_the_pipeline_actually_produced():
    draft = (
        "Total revenue for the year ended 31 December 2024 was "
        "**SAR 413,423 million**, compared with **SAR 331,360 million** in "
        "2023 [1]. This represents an increase of **SAR 82,063 million**, "
        "or **25 %** year-on-year [1]."
    )
    result = CitationVerifier().verify(draft, [_chunk(TABLE_PASSAGE)])
    assert result.outcome == VerificationOutcome.SUPPORTED, result.note

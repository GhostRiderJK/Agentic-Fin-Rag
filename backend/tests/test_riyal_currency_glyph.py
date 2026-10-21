"""Regression tests for the mis-decoded Riyal sign in PIF PDFs.

PIF's embedded font draws the Riyal sign where pdfplumber reads U+00AA
(feminine ordinal indicator), so stored chunk text said "ª 3,424 billion":
a stray glyph in every answer and report that quotes it, and a figure with no
currency for retrieval, fusion or the citation verifier to bind. 126 of the
2,136 chunks in the served corpus carry it (measured 2026-10-20).

The fix normalizes the glyph to "SAR" only in the shape that means currency,
so Spanish/Portuguese ordinals ("1ª edición") keep their text. Two call sites
are needed, because searchable chunk text is assembled from more than page text:
the parser normalizes page text and table cells, and the rendered table markdown
is normalized too — a sign in one cell with its figure in the next only becomes
adjacent in the joined line.
"""
from __future__ import annotations

import hashlib

import app.ingestion.parser as parser_mod
from app.ingestion.cleaner import clean_text, normalize_currency_glyphs
from tests.test_parser_dedupe import FakePage, install_pdf, single


def test_glyph_before_a_number_becomes_the_iso_code():
    assert normalize_currency_glyphs("reached ª 3,424 billion") == \
        "reached SAR 3,424 billion"
    assert normalize_currency_glyphs("ª3,424") == "SAR 3,424"
    assert normalize_currency_glyphs("ª\u00a0413,423") == "SAR 413,423"


def test_arabic_rial_sign_codepoints_are_covered():
    assert normalize_currency_glyphs("\ufdfc 12 m") == "SAR 12 m"
    assert normalize_currency_glyphs("\ufefc 12 m") == "SAR 12 m"


def test_scale_word_counts_as_the_figure_its_unit():
    # A table header column says "ª BN" with no number in the cell at all.
    assert normalize_currency_glyphs("ª BN Percentage") == "SAR BN Percentage"
    assert normalize_currency_glyphs("ª million") == "SAR million"


def test_ordinary_usage_of_the_same_glyph_is_left_alone():
    # "1ª" is an ordinal, not money: the glyph follows a digit.
    assert normalize_currency_glyphs("1ª edición") == "1ª edición"
    # A glyph with no figure after it asserts no currency.
    assert normalize_currency_glyphs("Treasury ª") == "Treasury ª"
    assert normalize_currency_glyphs("") == ""
    # A sign at the end of one table row must not borrow the figure that opens
    # the next row: the newline is not part of the separator run.
    assert normalize_currency_glyphs("Ratio ª\n5.3 percent") == \
        "Ratio ª\n5.3 percent"


def test_clean_text_normalizes_before_the_glossary():
    assert clean_text("Total  ª 2,871 billion") == "Total SAR 2,871 billion"


def test_page_text_is_normalized_at_parse_time(tmp_path, monkeypatch):
    pdf = install_pdf(tmp_path,
                      [FakePage(single("ª 3,424 BN", step=4.0))],
                      monkeypatch, dedupe=False)
    parsed = parser_mod.parse_file(pdf)
    assert "SAR 3,424 BN" in parsed.pages[0].text
    assert "ª" not in parsed.pages[0].text


def test_table_cells_and_markdown_are_normalized(tmp_path, monkeypatch):
    page = FakePage(single("Assets", top=100.0),
                    tables=[[["Item", "Amount"], ["AuM", "ª 3,424 BN"],
                             ["Employees", "2,962"]]])
    pdf = install_pdf(tmp_path, [page], monkeypatch, dedupe=False)
    parsed = parser_mod.parse_file(pdf)
    table = parsed.tables[0]
    assert table.rows[1][1] == "SAR 3,424 BN"
    assert "SAR 3,424 BN" in table.markdown
    assert "ª" not in table.markdown


def test_sign_and_figure_in_adjacent_cells_are_still_rewritten(tmp_path, monkeypatch):
    # What the served corpus actually stored after the first fix: the pipeline
    # splices rendered table markdown into searchable chunk text, so a chunk
    # reads "unsecured (ª | 56.25 b | illion)". Cell-level normalization cannot
    # see that shape — the adjacency only exists in the joined line.
    page = FakePage(single("Financing", top=100.0),
                    tables=[[["Item", "Facility", "Size"],
                             ["Revolving", "(ª", "56.25 billion)"],
                             ["Syndicated", "ª", "9 billion"]]])
    pdf = install_pdf(tmp_path, [page], monkeypatch, dedupe=False)
    parsed = parser_mod.parse_file(pdf)
    md = parsed.tables[0].markdown
    assert "ª" not in md
    assert "(SAR | 56.25 billion)" in md
    assert "| SAR | 9 billion |" in md
    # The cell delimiter survives: collapsing it would merge two table cells.
    assert "| (SAR | 56.25 billion) |" in md


def test_extractor_fingerprint_reports_the_new_contract():
    assert parser_mod.PARSER_CONTRACT_VERSION == 4
    assert parser_mod.extractor_fingerprint("pdf") == \
        hashlib.sha256(b"pdf|v4").hexdigest()[:16]

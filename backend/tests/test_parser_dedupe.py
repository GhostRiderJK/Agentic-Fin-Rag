"""Regression tests for the doubled-glyph char-layer defect (PIF corpus) and the
`pdfplumber_dedupe` extraction switch. The PDF library is faked at the char
level so the parser wiring is tested deterministically; real-pdfplumber
behaviour on the actual corpus is covered by
backend/scripts/exp_parser_compare_20261020.py artifacts under reports/.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import app.core.config as config_mod
import app.ingestion.parser as parser_mod


def char(text, x0, top):
    return {"text": text, "x0": x0, "x1": x0 + 4.0, "top": top,
            "bottom": top + 8.0}


class FakePage:
    width = 600.0
    height = 800.0

    def __init__(self, chars, tables=None):
        self.chars = list(chars)
        self._tables = tables or []

    def extract_text(self):
        by_row: dict[float, list[dict]] = {}
        for c in self.chars:
            by_row.setdefault(c["top"], []).append(c)
        lines = []
        for top in sorted(by_row):
            cs = sorted(by_row[top], key=lambda c: c["x0"])
            line = ""
            prev_x1 = None
            for c in cs:
                if prev_x1 is not None and c["x0"] - prev_x1 > 1.0:
                    line += " "
                line += c["text"]
                prev_x1 = c["x1"]
            lines.append(line)
        return "\n".join(lines)

    def extract_tables(self):
        return self._tables

    def extract_words(self):
        return []

    def dedupe_chars(self, tolerance=1):
        kept, seen = [], set()
        for c in self.chars:
            key = (c["text"], round(c["x0"] / tolerance),
                   round(c["top"] / tolerance))
            if key in seen:
                continue
            seen.add(key)
            kept.append(c)
        return FakePage(kept, self._tables)


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install_pdf(tmp_path, pages, monkeypatch, dedupe):
    monkeypatch.setitem(
        sys.modules, "pdfplumber",
        types.SimpleNamespace(open=lambda path: FakePDF(pages)))
    monkeypatch.setattr(
        config_mod, "get_settings",
        lambda: types.SimpleNamespace(pdfplumber_dedupe=dedupe))
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")  # only existence is checked; open is faked
    monkeypatch.setattr(
        config_mod, "get_settings",
        lambda: types.SimpleNamespace(pdfplumber_dedupe=dedupe))
    return pdf


def doubled(word, x0=50.0, top=100.0, step=5.0):
    out = []
    for i, ch in enumerate(word):
        out.append(char(ch, x0 + i * step, top))
        out.append(char(ch, x0 + i * step, top))
    return out


def single(word, x0=50.0, top=100.0, step=5.0):
    return [char(ch, x0 + i * step, top) for i, ch in enumerate(word)]


def test_dedupe_disabled_by_default_keeps_doubled_text(tmp_path, monkeypatch):
    pdf = install_pdf(tmp_path, [FakePage(doubled("About"))], monkeypatch,
                      dedupe=False)
    parsed = parser_mod.parse_file(pdf)
    assert parsed.pages[0].text == "AAbboouutt"
    assert parsed.deduped_pages == 0


def test_dedupe_enabled_collapses_doubled_glyphs(tmp_path, monkeypatch):
    page = doubled("About") + single("PIF", x0=50.0, top=200.0)
    pdf = install_pdf(tmp_path, [FakePage(page)], monkeypatch, dedupe=True)
    parsed = parser_mod.parse_file(pdf)
    assert "About" in parsed.pages[0].text
    assert "AAbboouutt" not in parsed.pages[0].text
    assert parsed.deduped_pages == 1


def test_dedupe_preserves_legitimate_repeats(tmp_path, monkeypatch):
    # "Committee" repeats tt/ee, "2023" repeats nothing, "11" is a real digit
    # pair — all single copies in the char layer, none may collapse.
    text_page = (single("Committee") + single("2023", top=200.0)
                 + single("11", x0=50.0, top=300.0))
    pdf = install_pdf(tmp_path, [FakePage(text_page)], monkeypatch, dedupe=True)
    parsed = parser_mod.parse_file(pdf)
    assert parsed.deduped_pages == 0  # gate: no position duplicates at all
    assert parsed.pages[0].text.splitlines() == ["Committee", "2023", "11"]


def test_dedupe_noop_on_clean_page_even_when_enabled(tmp_path, monkeypatch):
    clean = single("Revenue 2024", top=100.0)
    doubled_page = FakePage(doubled("Assets", top=100.0))
    pdf = install_pdf(tmp_path, [FakePage(clean), doubled_page], monkeypatch,
                      dedupe=True)
    parsed = parser_mod.parse_file(pdf)
    assert parsed.pages[0].text == "Revenue 2024"
    assert parsed.deduped_pages == 1


def test_dedupe_keeps_table_contract(tmp_path, monkeypatch):
    rows = [["Total Revenue", "1,234"], ["Net Income", "2,023"]]
    page = FakePage(doubled("Assets", top=100.0), tables=[rows])
    pdf = install_pdf(tmp_path, [page], monkeypatch, dedupe=True)
    parsed = parser_mod.parse_file(pdf)
    assert len(parsed.tables) == 1
    assert parsed.tables[0].rows == rows
    assert parsed.tables[0].page == 1
    assert "| Total Revenue | 1,234 |" in parsed.tables[0].markdown


def test_printed_footer_survives_offset_duplicate_glyphs(tmp_path, monkeypatch):
    # Footer "13" at bottom-left with each digit duplicated 0.4pt to the right
    # (the exact PIF failure mode). The duplicate is NOT at an identical
    # rounded position — the page must still be deduped because
    # dedupe_chars() actually collapses a glyph (audit finding: an
    # approximate position gate missed these collapses on 7 real pages).
    footer = [char("1", 30.0, 760.0), char("1", 30.4, 760.0),
              char("3", 35.0, 760.0), char("3", 35.4, 760.0)]
    body = single("Annual Report", top=100.0)
    pdf = install_pdf(tmp_path, [FakePage(footer + body)], monkeypatch,
                      dedupe=True)
    parsed = parser_mod.parse_file(pdf)
    assert parsed.pages[0].printed_page == 13
    assert parsed.deduped_pages == 1


def test_settings_failure_falls_back_to_baseline(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pdfplumber",
        types.SimpleNamespace(open=lambda path: FakePDF(
            [FakePage(doubled("About"))])))

    def boom():
        raise RuntimeError("no settings in this deployment")

    monkeypatch.setattr(config_mod, "get_settings", boom)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    parsed = parser_mod.parse_file(pdf)
    assert parsed.pages[0].text == "AAbboouutt"
    assert parsed.deduped_pages == 0


def test_fingerprint_tracks_the_dedupe_flag(tmp_path, monkeypatch):
    pages = lambda: [FakePage(doubled("About"))]  # noqa: E731
    pdf = install_pdf(tmp_path, pages(), monkeypatch, dedupe=False)
    off = parser_mod.parse_file(pdf).extractor_fingerprint
    install_pdf(tmp_path, pages(), monkeypatch, dedupe=True)
    on = parser_mod.parse_file(pdf).extractor_fingerprint
    assert off and on and off != on
    assert off == parser_mod.extractor_fingerprint("pdf", False)
    # A corpus fingerprint must be stable for the same configuration.
    install_pdf(tmp_path, pages(), monkeypatch, dedupe=True)
    assert parser_mod.parse_file(pdf).extractor_fingerprint == on

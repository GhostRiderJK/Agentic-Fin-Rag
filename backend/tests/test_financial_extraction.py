"""Regression tests for structured-financial extraction.

Two measured defects are pinned here.

1. pdfplumber's default table settings return degenerate grids on design-tool
   PDFs (the PIF annual reports yielded 1 usable table per document), and the
   text strategy answers with grids cut from prose. ``auto`` keeps the line
   result when it is usable, otherwise takes the text result when it looks like
   a table.
2. A number's reporting period is a *position* on the page, not a column index.
   The word-geometry reader assigns values to the year column they sit under,
   drops the "Change" columns, and refuses a value that could belong to two
   different periods.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import app.core.config as config_mod
import app.ingestion.parser as parser_mod
from app.ingestion.parser import LayoutWord, TableExtract
from app.ingestion.tables import (
    detect_unit_and_currency,
    extract_financials,
    extract_financials_from_pages,
    match_metric,
)

# --------------------------------------------------------------------------
# label matching
# --------------------------------------------------------------------------


def test_fragmented_label_still_matches():
    # A text-strategy grid cuts words apart; spaces are what the cut inserts.
    assert match_metric("Total Asset s") == "total_assets"
    # A real label for a metric this system does not store stays unmapped.
    assert match_metric("Investment sec urities an d derivatives") is None


def test_composite_label_is_not_mistaken_for_a_part():
    # "total liabilities and equity" is the balance-sheet total, not a liability
    assert match_metric("Total Liabilities and Equity") is None
    assert match_metric("Total Liabilities") == "total_liabilities"


def test_short_alias_does_not_swallow_a_longer_line():
    # "cash" is an alias, but a cash-flow line is not a cash balance.
    assert match_metric("Net cash used in investing activities") is None
    assert match_metric("Cash and cash equivalents") == "cash_and_equivalents"


def test_qualifiers_on_a_short_alias_still_name_the_metric():
    # The length floor must not cost these: a filing writes the metric with a
    # qualifier in front of, or around, the short alias.
    assert match_metric("Adjusted EBITDA") == "ebitda"
    assert match_metric("Revenue (net)") == "revenue"
    assert match_metric("Revenue from contracts with customers") == "revenue"
    assert match_metric("Net Sales") == "revenue"
    assert match_metric("Basic EPS") == "eps"
    assert match_metric("Total COGS") == "cost_of_goods_sold"
    assert match_metric("Capex") == "capex"
    # Still refused: "cash", "aset" and "utang" are only ever admitted as the
    # whole label, so a qualified line cannot be filed as the balance metric.
    assert match_metric("Aset Tetap") is None
    assert match_metric("Utang Bank Jangka Pendek") is None
    assert match_metric("Cash and cash equivalents") == "cash_and_equivalents"


def test_opening_balance_is_not_the_cash_position():
    assert match_metric(
        "Cash and cash equivalents at the beginning of the year") is None


def test_operating_cash_flow_needs_both_ideas():
    assert match_metric(
        "Net cash generated from operating activities") == "operating_cash_flow"
    assert match_metric("Net cash used in financing activities") is None


def test_statement_year_labels_match():
    assert match_metric("Profit for the year") == "net_income"
    assert match_metric("Profit/(Loss) for the year") == "net_income"
    assert match_metric("Cost of revenue") == "cost_of_goods_sold"
    assert match_metric("Unrelated narrative sentence") is None


# --------------------------------------------------------------------------
# unit / currency context
# --------------------------------------------------------------------------


def test_scale_and_currency_come_from_the_statement_header():
    assert detect_unit_and_currency("AMOUNTS IN MILLION SAR)") == ("million", "SAR")
    assert detect_unit_and_currency("Amounts in US dollars") == (None, "USD")
    # No claim is made when the document makes none.
    assert detect_unit_and_currency("Revenue grew year on year") == (None, None)


# --------------------------------------------------------------------------
# cell-grid reader (DOCX/HTML and well-ruled PDFs)
# --------------------------------------------------------------------------


def test_header_row_is_found_below_a_title_and_unit_line():
    tbl = TableExtract(markdown="", rows=[
        ["Consolidated statement of financial position"],
        ["Amounts in SAR million"],
        ["", "2024", "2023"],
        ["Total Assets", "2,500,000", "2,100,000"],
    ], page=75)
    rows = extract_financials([tbl], company="PIF", ticker="")
    assert {(r.metric, r.fiscal_year, r.unit, r.currency) for r in rows} == {
        ("total_assets", 2024, "million", "SAR"),
        ("total_assets", 2023, "million", "SAR"),
    }


def test_period_label_may_be_a_full_date_phrase():
    tbl = TableExtract(markdown="", rows=[
        ["Metric", "For the year ended 31 December 2024"],
        ["Revenue", "140,000"],
    ], page=3)
    rows = extract_financials([tbl], company="ACME", ticker="")
    assert [(r.fiscal_year, r.value) for r in rows] == [(2024, 140000.0)]


def test_a_figure_is_not_read_as_a_period():
    tbl = TableExtract(markdown="", rows=[
        ["Metric", "1,234,567", "3.4"],
        ["Revenue", "140,000", "130,000"],
    ], page=3)
    assert extract_financials([tbl], company="ACME", ticker="") == []


def test_quarter_label_is_kept():
    tbl = TableExtract(markdown="", rows=[
        ["Metric", "Q3 2024"],
        ["Revenue", "140,000"],
    ], page=3)
    rows = extract_financials([tbl], company="ACME", ticker="")
    assert [(r.quarter, r.fiscal_year) for r in rows] == [("Q3", 2024)]


# --------------------------------------------------------------------------
# word-geometry reader (PDFs)
# --------------------------------------------------------------------------


def word(text, x0, x1=None, top=0.0) -> LayoutWord:
    return LayoutWord(text=text, x0=x0, x1=x1 if x1 is not None else x0 + 30.0,
                      top=top)


def page(words, *, page_number=78, text="Amounts in SAR million"):
    return types.SimpleNamespace(page=page_number, text=text, words=words)


# year columns sit at x-centres 315 and 385; the two change columns at 450/520
HEADER = [word("2023", 300, 330, top=10), word("2024", 370, 400, top=10),
          word("Change", 420, 480, top=10), word("%", 515, 525, top=10)]


def row(label_words, values, top=40.0):
    return [word(t, x0, x1, top=top) for t, x0, x1 in label_words] + [
        word(t, x0, x1, top=top) for t, x0, x1 in values]


def test_values_are_assigned_by_position_not_order():
    # Only the right-hand column (2024) carries a figure on this row, plus a
    # "% change" value. Pairing values to years left to right would file 2,500
    # under 2023 and would also file the percentage as a measurement.
    words = HEADER + row([("Total", 50, 90), ("Assets", 95, 140)],
                         [("2,500", 358, 412), ("19%", 498, 542)])
    rows = extract_financials_from_pages([page(words)], company="PIF")
    assert {(r.metric, r.fiscal_year, r.value) for r in rows} == {
        ("total_assets", 2024, 2500.0)}
    assert {r.currency for r in rows} == {"SAR"}
    assert {r.unit for r in rows} == {"million"}


def test_each_value_lands_under_its_own_year():
    words = HEADER + row([("Total Assets", 50, 140)],
                         [("2,100", 288, 342), ("2,500", 358, 412),
                          ("400", 428, 472), ("19%", 498, 542)])
    rows = extract_financials_from_pages([page(words)], company="PIF")
    assert {(r.fiscal_year, r.value) for r in rows} == {
        (2023, 2100.0), (2024, 2500.0)}


def test_year_word_is_never_a_value():
    # A label that names a year must not file 2023 as a measurement, and the
    # value still belongs to the column it sits in (2024).
    words = HEADER + row([("Revenue", 50, 100), ("2023", 105, 140)],
                         [("1,000", 358, 412)])
    rows = extract_financials_from_pages([page(words)], company="PIF")
    assert [(r.fiscal_year, r.value) for r in rows] == [(2024, 1000.0)]


def test_ambiguous_value_is_dropped_not_guessed():
    near = [word("2023", 100, 130, top=10), word("2024", 125, 155, top=10)]
    words = near + row([("Total Assets", 400, 500)], [("1,000", 115, 141)])
    assert extract_financials_from_pages([page(words)], company="PIF") == []


def test_page_without_a_period_header_yields_nothing():
    words = row([("Revenue", 50, 100), ("grew", 105, 140), ("in", 145, 160),
                 ("2024", 165, 200)], [("1,000", 300, 340)])
    assert extract_financials_from_pages([page(words)], company="PIF") == []


def test_wrapped_label_loses_the_opening_balance():
    words = HEADER + [
        *row([("Cash", 50, 90), ("and", 95, 120),
              ("equivalents", 125, 200), ("at", 205, 220), ("the", 225, 245)],
             [("1,000", 288, 342), ("900", 358, 412)], top=40),
        *row([("beginning", 50, 110), ("of", 115, 130), ("the", 135, 155),
              ("year", 160, 190)], [], top=60),
    ]
    assert extract_financials_from_pages([page(words)], company="PIF") == []


def test_negative_in_split_parentheses_keeps_its_sign():
    words = HEADER + row([("Revenue", 50, 100)],
                         [("(", 288, 296), ("1,000", 300, 342), (")", 344, 352)])
    rows = extract_financials_from_pages([page(words)], company="PIF")
    assert [(r.fiscal_year, r.value) for r in rows] == [(2023, -1000.0)]


def test_pages_without_words_are_skipped():
    assert extract_financials_from_pages(
        [types.SimpleNamespace(page=1, text="x", words=[])], company="PIF") == []


# --------------------------------------------------------------------------
# parser wiring: table strategy
# --------------------------------------------------------------------------


class FakePage:
    """Answers only the calls the configured strategy is allowed to make."""

    width = 600.0
    height = 800.0
    page_number = 1

    def __init__(self, baseline, text_tables=None, words=None):
        self._baseline = baseline
        self._text_tables = text_tables
        self._words = words or []
        self.calls = []

    def extract_text(self):
        return "Total Assets 2,500"

    def extract_tables(self, table_settings=None):
        self.calls.append("text" if table_settings else "lines")
        if table_settings:
            if self._text_tables is None:
                raise AssertionError("unexpected text-strategy call")
            return self._text_tables
        return self._baseline

    def extract_words(self):
        return [{"text": w.text, "x0": w.x0, "x1": w.x1, "top": w.top}
                for w in self._words]

    def extract_table(self, *a, **k):  # pragma: no cover - unused
        return None


def install(tmp_path, pages, monkeypatch, strategy, dedupe=False):
    class FakePDF:
        def __init__(self, _path):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "pdfplumber",
                        types.SimpleNamespace(open=FakePDF))
    monkeypatch.setattr(config_mod, "get_settings", lambda: types.SimpleNamespace(
        pdfplumber_dedupe=dedupe, pdfplumber_table_strategy=strategy))
    pdf = Path(tmp_path) / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


PROSE_GRID = [[["our", "disciplined", "financial", "strategy",
                "and long-term investment approach continues to fuel value"]
               for _ in range(4)] for _ in range(4)]
TABLE_GRID = [[["Total Assets", "2,500", "2,100"],
               ["Total Equity", "1,800", "1,500"],
               ["Revenue", "900", "800"]]]
DEGENERATE = [[[""], ["CHANGE", "% CHANGE"]]]


def test_auto_keeps_the_line_result_when_it_is_usable(tmp_path, monkeypatch):
    p = FakePage(TABLE_GRID)
    pdf = install(tmp_path, [p], monkeypatch, strategy="auto")
    parsed = parser_mod.parse_file(pdf)
    assert p.calls == ["lines"]  # no second pass at all
    assert parsed.text_table_pages == 0
    assert parsed.tables[0].rows == TABLE_GRID[0]


def test_auto_takes_the_text_grid_when_lines_are_degenerate(tmp_path, monkeypatch):
    p = FakePage(DEGENERATE, text_tables=TABLE_GRID)
    pdf = install(tmp_path, [p], monkeypatch, strategy="auto")
    parsed = parser_mod.parse_file(pdf)
    assert p.calls == ["lines", "text"]
    assert parsed.text_table_pages == 1
    assert parsed.tables[0].rows == TABLE_GRID[0]


def test_auto_refuses_a_grid_cut_out_of_prose(tmp_path, monkeypatch):
    p = FakePage(DEGENERATE, text_tables=PROSE_GRID)
    pdf = install(tmp_path, [p], monkeypatch, strategy="auto")
    parsed = parser_mod.parse_file(pdf)
    assert parsed.text_table_pages == 0
    assert all(len(c) <= 60 for t in parsed.tables for r in t.rows for c in r)


def test_baseline_strategy_never_asks_for_the_text_grid(tmp_path, monkeypatch):
    p = FakePage(DEGENERATE, text_tables=TABLE_GRID)
    pdf = install(tmp_path, [p], monkeypatch, strategy="lines")
    parsed = parser_mod.parse_file(pdf)
    assert p.calls == ["lines"]
    assert parsed.text_table_pages == 0


def test_failing_text_strategy_keeps_baseline_tables(tmp_path, monkeypatch):
    class Exploding(FakePage):
        def extract_tables(self, table_settings=None):
            self.calls.append("text" if table_settings else "lines")
            if table_settings:
                raise RuntimeError("bad table settings on this page")
            return self._baseline

    p = Exploding(DEGENERATE)
    pdf = install(tmp_path, [p], monkeypatch, strategy="auto")
    parsed = parser_mod.parse_file(pdf)
    assert parsed.tables[0].rows == DEGENERATE[0]


def test_words_reach_the_parsed_page(tmp_path, monkeypatch):
    p = FakePage(TABLE_GRID, words=[word("Total", 50, 90)])
    pdf = install(tmp_path, [p], monkeypatch, strategy="auto")
    parsed = parser_mod.parse_file(pdf)
    assert parsed.pages[0].words[0].text == "Total"
    assert parsed.pages[0].words[0].x1 == 90


def test_fingerprint_records_the_table_strategy(tmp_path, monkeypatch):
    auto = parser_mod.parse_file(
        install(tmp_path, [FakePage(TABLE_GRID)], monkeypatch,
                strategy="auto")).extractor_fingerprint
    lines = parser_mod.parse_file(
        install(tmp_path, [FakePage(TABLE_GRID)], monkeypatch,
                strategy="lines")).extractor_fingerprint
    assert auto != lines
    # the historical baseline hash is unchanged, so an untouched corpus stays
    # comparable with the fingerprints already stored with it
    assert lines == parser_mod.extractor_fingerprint("pdf", False)

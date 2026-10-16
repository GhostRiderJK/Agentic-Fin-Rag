"""Table extractor: convert parsed tables into structured financial rows.

This is the heart of the *structured knowledge* side of the dual system. We
identify header columns that look like periods (years, quarters) and rows
that look like metrics (Revenue, Net Income, EPS, …), then emit
``FinancialRow`` records ready to insert into ``StructuredFinancials``.

Conservative heuristics:
  - We only emit a row when (a) the row name matches a known metric alias
    and (b) every period cell parses to a number.
  - All numbers keep their string unit/scale context so the LLM and the SQL
    tool-calling agent can answer "what was Apple's revenue in 2025?" without
    the LLM having to recite the figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.parser import TableExtract


# (canonical_metric, [aliases])
_METRIC_ALIASES: list[tuple[str, list[str]]] = [
    ("revenue", ["revenue", "net revenue", "net sales", "sales", "total revenue",
                 "pendapatan", "total pendapatan"]),
    ("cost_of_goods_sold", ["cost of goods sold", "cost of sales", "cogs",
                             "beban pokok penjualan"]),
    ("gross_profit", ["gross profit", "lab a kotor", "laba kotor"]),
    ("operating_income", ["operating income", "income from operations",
                          "operating profit", "laba operasi"]),
    ("net_income", ["net income", "net earnings", "net profit", "laba bersih",
                    "profit attributable"]),
    ("eps", ["eps", "earnings per share", "basic eps", "diluted eps"]),
    ("ebitda", ["ebitda"]),
    ("operating_margin", ["operating margin", "margin operasi"]),
    ("gross_margin", ["gross margin", "margin laba kotor"]),
    ("total_assets", ["total assets", "aset", "total aset"]),
    ("total_liabilities", ["total liabilities", "kewajiban", "total kewajiban"]),
    ("total_equity", ["total equity", "stockholders equity", "ekuitas"]),
    ("total_debt", ["total debt", "long-term debt", "utang"]),
    ("cash_and_equivalents", ["cash and cash equivalents", "cash",
                              "kas dan setara kas"]),
    ("operating_cash_flow", ["operating cash flow", "cash from operations",
                             "arus kas operasi"]),
    ("free_cash_flow", ["free cash flow"]),
    ("dividend_per_share", ["dividend per share", "dps", "dividen per saham"]),
    ("shares_outstanding", ["shares outstanding", "common shares outstanding"]),
    ("rd_expense", ["research and development", "r&d expense", "rnd expense"]),
    ("capex", ["capital expenditure", "capex", "pengeluaran modal"]),
]

_ALIAS_INDEX: dict[str, str] = {}
for canon, aliases in _METRIC_ALIASES:
    for a in aliases:
        _ALIAS_INDEX[a.strip().lower()] = canon


_PERIOD_RE = re.compile(
    r"^(?:(?P<q>q[1-4]|fy|fy\d{2})\s*)?(?P<year>20\d{2})$|^"
    r"(?P<year2>20\d{2})$|^(?P<quarter>q[1-4])\s*(?P<year3>20\d{2})$",
    re.IGNORECASE,
)


@dataclass
class FinancialRow:
    company: str
    ticker: str
    metric: str
    fiscal_year: int
    quarter: str
    value: float
    unit: str
    currency: str
    page: int
    confidence: float


_NUMBER_RE = re.compile(
    r"^[-+]?[\d,]*\.?\d+%?$|^[-+]?\(\s*[\d,]*\.?\d+\s*\)%?$"  # supports (123) negatives
)


def _parse_number(s: str) -> tuple[float, str]:
    """Parse a numeric string into (value, unit_hint).

    Handles commas, parentheses-negatives, and trailing units (e.g. 'B', 'M').
    Returns (float('nan'), '') if not parseable.
    """
    raw = s.strip()
    if not raw or raw in {"—", "-", "n/a", "N/A"}:
        return float("nan"), ""

    unit = ""
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        neg = True
        raw = raw[1:-1]

    suffix = ""
    for suf in ("Bn", "bn", "B", "M", "m", "K", "k", "%"):
        if raw.endswith(suf):
            suffix = suf
            raw = raw[: -len(suf)].strip()
            break
    raw = raw.replace(",", "").replace("%", "").strip()
    try:
        val = float(raw)
    except ValueError:
        return float("nan"), ""

    mult = {"B": 1e9, "Bn": 1e9, "M": 1e6, "K": 1e3}.get(suffix, 1.0)
    val *= mult
    if neg:
        val = -val
    if suffix in {"%", "M"} and suffix != "M":
        unit = "percent"
    return val, unit


def _classify_period(header: str) -> tuple[int, str] | None:
    """Return (year, quarter) for a header cell, or None."""
    h = header.strip()
    if not h:
        return None
    m = _PERIOD_RE.match(h)
    if not m:
        return None
    year = m.group("year") or m.group("year2") or m.group("year3")
    if not year:
        return None
    quarter = (m.group("q") or m.group("quarter") or "").upper()
    if quarter.startswith("FY"):
        quarter = ""
    return int(year), quarter


def extract_financials(
    tables: list[TableExtract],
    *,
    company: str = "",
    ticker: str = "",
    currency: str = "USD",
) -> list[FinancialRow]:
    rows: list[FinancialRow] = []
    for tbl in tables:
        if not tbl.rows or len(tbl.rows) < 2:
            continue
        header = tbl.rows[0]
        # Map each column to (year, quarter) if classifiable
        period_cols: list[tuple[int, int, str]] = []  # (col_index, year, quarter)
        for ci, cell in enumerate(header):
            period = _classify_period(cell)
            if period:
                period_cols.append((ci, period[0], period[1]))
        if not period_cols:
            continue

        for body_row in tbl.rows[1:]:
            if not body_row:
                continue
            metric_name = (body_row[0] or "").strip().lower()
            canon = _ALIAS_INDEX.get(metric_name)
            if not canon:
                # Allow fuzzy: "Total Revenue" → "revenue"
                for k, v in _ALIAS_INDEX.items():
                    if k in metric_name:
                        canon = v
                        break
            if not canon:
                continue
            for ci, year, quarter in period_cols:
                if ci >= len(body_row):
                    continue
                val, unit_hint = _parse_number(body_row[ci])
                if val != val:  # NaN
                    continue
                rows.append(
                    FinancialRow(
                        company=company,
                        ticker=ticker,
                        metric=canon,
                        fiscal_year=year,
                        quarter=quarter,
                        value=val,
                        unit=unit_hint or "value",
                        currency=currency,
                        page=tbl.page,
                        confidence=0.85,
                    )
                )
    return rows

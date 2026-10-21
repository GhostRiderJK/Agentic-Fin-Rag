"""Table extractor: convert parsed tables into structured financial rows.

This is the heart of the *structured knowledge* side of the dual system. We
identify header columns that look like periods (years, quarters) and rows
that look like metrics (Revenue, Net Income, EPS, …), then emit
``FinancialRow`` records ready to insert into ``StructuredFinancials``.

Two readers, because PDF tables come in two shapes:

  - ``extract_financials`` reads a cell grid. Sound when the grid has real
    columns — DOCX/HTML tables, and PDFs whose drawn rules survive intact.
  - ``extract_financials_from_pages`` reads word *positions*. Design-tool PDFs
    (PIF annual reports) give pdfplumber no column geometry to work with: its
    text grid splits labels mid-word ("total asset s") and puts each row's
    numbers wherever they happen to sit, so a column *index* means nothing.
    A number's reporting period is decided by its horizontal position relative
    to the year words in the header line, which is the one signal that holds.

Conservative heuristics:
  - We only emit a row when (a) the row name matches a known metric alias
    and (b) the number sits under a period column.
  - All numbers keep their string unit/scale context so the LLM and the SQL
    tool-calling agent can answer "what was Apple's revenue in 2025?" without
    the LLM having to recite the figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.parser import LayoutWord, TableExtract


# (canonical_metric, [aliases])
_METRIC_ALIASES: list[tuple[str, list[str]]] = [
    ("revenue", ["revenue", "net revenue", "net sales", "sales", "total revenue",
                 "pendapatan", "total pendapatan"]),
    ("cost_of_goods_sold", ["cost of goods sold", "cost of sales", "cogs",
                            "cost of revenue", "beban pokok penjualan"]),
    ("gross_profit", ["gross profit", "laba kotor"]),
    ("operating_income", ["operating income", "income from operations",
                          "operating profit", "laba operasi",
                          "operating profit loss"]),
    ("net_income", ["net income", "net earnings", "net profit", "laba bersih",
                    "profit attributable", "profit for the year",
                    "net profit for the year", "profit loss for the year",
                    "profit or loss for the year", "total profit"]),
    ("eps", ["eps", "earnings per share", "basic eps", "diluted eps"]),
    ("ebitda", ["ebitda"]),
    ("operating_margin", ["operating margin", "margin operasi"]),
    ("gross_margin", ["gross margin", "margin laba kotor"]),
    ("total_assets", ["total assets", "aset", "total aset"]),
    ("total_liabilities", ["total liabilities", "kewajiban", "total kewajiban"]),
    ("total_equity", ["total equity", "stockholders equity", "ekuitas",
                      "shareholders equity", "owners equity"]),
    ("total_debt", ["total debt", "long-term debt", "utang"]),
    ("cash_and_equivalents", ["cash and cash equivalents", "cash",
                              "kas dan setara kas"]),
    ("operating_cash_flow", ["operating cash flow", "cash from operations",
                             "arus kas operasi",
                             "cash generated from operating activities",
                             "cash used in operating activities",
                             "net cash from operating activities"]),
    ("free_cash_flow", ["free cash flow"]),
    ("dividend_per_share", ["dividend per share", "dps", "dividen per saham"]),
    ("shares_outstanding", ["shares outstanding", "common shares outstanding"]),
    ("rd_expense", ["research and development", "r&d expense", "rnd expense"]),
    ("capex", ["capital expenditure", "capex", "pengeluaran modal"]),
]

# Labels that read like a metric but are not one. "TOTAL LIABILITIES AND EQUITY"
# contains "total liabilities" and would otherwise store the balance-sheet
# total as the liabilities figure; the opening cash balance contains "cash and
# cash equivalents" and would overwrite the closing balance.
_NEGATIVE_FRAGMENTS = (
    "totalliabilitiesandequity",
    "totalliabilitiesandshareholders",
    "totalliabilitiesandowners",
    "beginningoftheyear",
    "beginningofperiod",
    "netincrease",
    "netdecrease",
    "decreaseincash",
    "increaseincash",
)

def _normalize_label(text: str) -> str:
    """Lower-case, letters+digits only.

    A PDF word grid cuts labels mid-word ("total asset s", "Investment sec
    urities an d derivatives"). Spaces are what the cut inserts, so removing
    every non-alphanumeric character makes a fragmented label equal to the
    intact one.
    """
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# Minimum alias length allowed to match *inside* a longer label. "cash" is a
# real alias, and every cash-flow line in a statement contains it; without this
# floor "net cash used in investing activities" is filed as a cash balance.
_MIN_CONTAINED_ALIAS = 8

# Short aliases the floor would otherwise block, admitted one at a time because
# real statements qualify them: "Adjusted EBITDA", "Revenue (net)", "Revenue
# from contracts with customers", "Net sales", "Basic EPS", "Total COGS".
# Deliberately absent: "cash" (the line that names it is the flow line the floor
# exists for), "aset" ("aset tetap" is a fixed-asset line, not total assets) and
# "utang" ("utang bank jangka pendek" is a bank loan, not total debt).
_SHORT_CONTAINED_ALIASES = frozenset({
    "revenue", "sales", "ebitda", "capex", "cogs", "eps", "dps", "ekuitas",
})

# Aliases that only mean the metric when a second idea is present. Cash-flow
# statements read "Net cash generated from / (used in) operating activities"
# and the same shape for investing and financing; only the operating one is a
# metric this system stores, so the qualifier has to be part of the match.
_COMPOSITE_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("operating_cash_flow", ("cash", "operating")),
    ("operating_income", ("profit", "operating")),
    ("operating_margin", ("margin", "operating")),
]

_NORM_ALIAS: dict[str, str] = {}
for _canon, _aliases in _METRIC_ALIASES:
    for _a in _aliases:
        _NORM_ALIAS[_normalize_label(_a)] = _canon


def match_metric(label: str) -> str | None:
    """Canonical metric for a row label, or None. Longest alias wins."""
    norm = _normalize_label(label)
    if not norm or any(neg in norm for neg in _NEGATIVE_FRAGMENTS):
        return None
    exact = _NORM_ALIAS.get(norm)
    if exact:
        return exact
    best: tuple[str, str] | None = None
    for alias, canon in _NORM_ALIAS.items():
        if len(alias) < _MIN_CONTAINED_ALIAS and alias not in _SHORT_CONTAINED_ALIASES:
            continue
        if alias not in norm:
            continue
        if best is None or len(alias) > len(best[0]):
            best = (alias, canon)
    composite: tuple[int, str] | None = None
    for canon, parts in _COMPOSITE_ALIASES:
        if all(p in norm for p in parts):
            width = sum(len(p) for p in parts)
            if composite is None or width > composite[0]:
                composite = (width, canon)
    if composite and (best is None or composite[0] > len(best[0])):
        return composite[1]
    return best[1] if best else None


# --------------------------------------------------------------------------
# Unit and currency context
# --------------------------------------------------------------------------

_SCALE_WORDS: tuple[tuple[str, str], ...] = (
    ("billions", "billion"), ("billion", "billion"), ("bn", "billion"),
    ("millions", "million"), ("million", "million"),
    ("thousands", "thousand"), ("thousand", "thousand"),
)

_CURRENCY_WORDS: tuple[tuple[str, str], ...] = (
    ("saudi arabian riyal", "SAR"), ("saudi riyal", "SAR"), ("sars", "SAR"),
    ("sar", "SAR"), ("riyals", "SAR"), ("riyal", "SAR"),
    ("us dollars", "USD"), ("us dollar", "USD"), ("usd", "USD"),
    ("u.s. dollar", "USD"), ("dollars", "USD"),
    ("euros", "EUR"), ("euro", "EUR"),
    ("pounds sterling", "GBP"), ("sterling", "GBP"),
    ("uae dirham", "AED"), ("dirham", "AED"),
    ("indian rupee", "INR"), ("rupee", "INR"), ("rupiah", "IDR"),
)


def detect_unit_and_currency(text: str) -> tuple[str | None, str | None]:
    """Read the scale and currency a table's numbers are stated in.

    Statements print it once, above the grid ("AMOUNTS IN SAR MILLION"). If it
    is missing we report None and let the caller's default stand rather than
    inventing a unit — a number filed as USD when it is SAR million is worse
    for a citation than a number that was never filed at all.
    """
    flat = " ".join((text or "").lower().split())
    if not flat:
        return None, None
    scale = next((unit for word, unit in _SCALE_WORDS
                  if re.search(rf"\b{re.escape(word)}\b", flat)), None)
    currency = next((code for word, code in _CURRENCY_WORDS
                     if re.search(rf"\b{re.escape(word)}\b", flat)), None)
    return scale, currency


_PERIOD_RE = re.compile(
    r"^(?:(?P<q>q[1-4]|fy|fy\d{2})\s*)?(?P<year>20\d{2})$|^"
    r"(?P<year2>20\d{2})$|^(?P<quarter>q[1-4])\s*(?P<year3>20\d{2})$",
    re.IGNORECASE,
)

_ANY_YEAR_RE = re.compile(r"(?<!\d)20\d{2}(?!\d)")
_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
# A grouped number or a decimal is a measurement, not a period label.
_FIGURE_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\.\d")
_MAX_PERIOD_LABEL_WORDS = 8
_MAX_HEADER_ROWS = 4


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
    """Return (year, quarter) for a header cell, or None.

    A cell qualifies when it names one reporting period and nothing else. The
    anchored form ("2024", "Q1 2024", "FY24") is checked first; then a short
    label that carries exactly one year and no figures, because statements
    head their columns with the period *ending* date ("31 December 2024",
    "For the year ended 31 December 2024"). A cell with a grouped number
    ("1,234,567") or a decimal is a value that happens to contain digits, not
    a period, and is refused.
    """
    h = " ".join((header or "").split())
    if not h:
        return None
    m = _PERIOD_RE.match(h)
    if m:
        year = m.group("year") or m.group("year2") or m.group("year3")
        if year:
            quarter = (m.group("q") or m.group("quarter") or "").upper()
            if quarter.startswith("FY"):
                quarter = ""
            return int(year), quarter
    if len(h.split()) > _MAX_PERIOD_LABEL_WORDS:
        return None
    if _FIGURE_RE.search(h):
        return None
    years = set(_ANY_YEAR_RE.findall(h))
    if len(years) != 1:
        return None
    q = _QUARTER_RE.search(h)
    return int(years.pop()), (q.group(1).upper() if q else "")


def _period_columns(row: list[str]) -> list[tuple[int, int, str]]:
    """(column index, year, quarter) for every period header cell in a row."""
    out: list[tuple[int, int, str]] = []
    for ci, cell in enumerate(row):
        period = _classify_period(cell)
        if period:
            out.append((ci, period[0], period[1]))
    return out


def _find_header(tbl_rows: list[list[str]]) -> tuple[int, list[tuple[int, int, str]]] | None:
    """First row that names reporting periods, within the leading rows.

    A grid's title, its unit line ("Amounts in SAR million") and a blank
    spacer routinely occupy row 0, so reading only row 0 as the header misses
    periods that sit one or two rows lower.
    """
    for ri in range(min(_MAX_HEADER_ROWS, len(tbl_rows))):
        cols = _period_columns(tbl_rows[ri])
        if cols and ri + 1 < len(tbl_rows):
            return ri, cols
    return None


def extract_financials(
    tables: list[TableExtract],
    *,
    company: str = "",
    ticker: str = "",
    currency: str = "USD",
) -> list[FinancialRow]:
    """Read metric rows from cell grids (DOCX/HTML tables, well-ruled PDFs).

    Column *index* is trusted here: in a real grid the value under a period
    header is in that header's column. PDFs built in a design tool do not
    offer that guarantee — see ``extract_financials_from_pages``.
    """
    rows: list[FinancialRow] = []
    for tbl in tables:
        if not tbl.rows or len(tbl.rows) < 2:
            continue
        found = _find_header(tbl.rows)
        if not found:
            continue
        header_index, period_cols = found

        flat = " ".join(" ".join(r) for r in tbl.rows)
        scale, table_currency = detect_unit_and_currency(flat)
        unit_default = scale or "value"
        use_currency = table_currency or currency

        for body_row in tbl.rows[header_index + 1:]:
            if not body_row:
                continue
            canon = match_metric(body_row[0])
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
                        unit=unit_hint or unit_default,
                        currency=use_currency,
                        page=tbl.page,
                        confidence=0.85,
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# PDF word geometry
# ---------------------------------------------------------------------------

_LINE_TOLERANCE = 3.0     # points: words this close vertically are one line
_COLUMN_TOLERANCE = 24.0  # points: a value this far from a year is not in it
_MAX_LABEL_TOKENS = 12    # beyond this, the "label" is a sentence
_MIN_LABEL_CHARS = 4

_YEAR_WORD_RE = re.compile(r"^20\d{2}$")
_PAREN_RE = re.compile(r"^[([{)\]}]+$")


@dataclass
class _Token:
    text: str
    center: float
    is_number: bool
    is_year: bool

    @property
    def label_part(self) -> str:
        return "" if self.is_number else self.text


def _group_lines(words: list[LayoutWord]) -> list[list[LayoutWord]]:
    """Cluster words into printed lines by vertical position, left to right."""
    lines: dict[int, list[LayoutWord]] = {}
    for w in words:
        lines.setdefault(round(w.top / _LINE_TOLERANCE), []).append(w)
    return [sorted(lines[k], key=lambda w: w.x0) for k in sorted(lines)]


def _tokenize(line: list[LayoutWord]) -> list[_Token]:
    """Turn one line of word boxes into tokens, gluing split parentheses.

    A negative in accountants' notation often arrives as three words — "(",
    "1,234", ")" — and the number alone parses as positive. Merge the brackets
    into the number they enclose before anything reads the value.
    """
    raw = [(w.text.strip(), (w.x0 + w.x1) / 2) for w in line if w.text.strip()]
    merged: list[tuple[str, float, float]] = [
        (text, center, center) for text, center in raw]

    changed = True
    while changed:
        changed = False
        out: list[tuple[str, float, float]] = []
        i = 0
        while i < len(merged):
            text, lo, hi = merged[i]
            nxt = merged[i + 1] if i + 1 < len(merged) else None
            if nxt and _PAREN_RE.match(text) and _numeric(nxt[0]):
                # opening bracket belongs to the number on its right
                out.append((text + nxt[0], lo, nxt[2]))
                i += 2
                changed = True
                continue
            if out and _PAREN_RE.match(text) and _numeric(out[-1][0]):
                out[-1] = (out[-1][0] + text, out[-1][1], hi)
                i += 1
                changed = True
                continue
            out.append((text, lo, hi))
            i += 1
        merged = out

    tokens: list[_Token] = []
    for text, center, _hi in merged:
        is_year = bool(_YEAR_WORD_RE.match(text))
        tokens.append(_Token(text=text, center=center, is_number=_numeric(text),
                             is_year=is_year))
    return tokens


def _numeric(text: str) -> bool:
    return bool(text) and any(ch.isdigit() for ch in text)


def _year_markers(lines: list[list[_Token]]) -> list[tuple[float, int, str]]:
    """(x center, year, quarter) for period columns.

    A year is a column header when its line carries another year beside it —
    that pairing is what a header row of a statement looks like, and it is what
    keeps a year mentioned in running prose from becoming a column.
    """
    markers: list[tuple[float, int, str]] = []
    for tokens in lines:
        years = [t for t in tokens if t.is_year]
        if len(years) < 2:
            continue
        for t in years:
            period = _classify_period(t.text)
            if period:
                markers.append((t.center, period[0], period[1]))
    return markers


def _assign_column(center: float,
                   markers: list[tuple[float, int, str]]) -> tuple[int, str] | None:
    """The period column a value sits in, or None if it sits in none.

    Ambiguous by design is not accepted: if two *different* periods are both
    within tolerance, the value could be filed under either year, so it is
    dropped rather than guessed.
    """
    near = [(abs(center - x), year, quarter) for x, year, quarter in markers]
    near = [n for n in near if n[0] <= _COLUMN_TOLERANCE]
    if not near:
        return None
    near.sort(key=lambda n: n[0])
    if any(n[1] != near[0][1] for n in near[1:]):
        return None
    return near[0][1], near[0][2]


def extract_financials_from_pages(
    pages,
    *,
    company: str = "",
    ticker: str = "",
    currency: str = "USD",
) -> list[FinancialRow]:
    """Read metric rows from a PDF's word positions.

    Every value is placed under a period by x-position, which is the only
    signal that survives a design-tool PDF: pdfplumber's text grid cuts labels
    mid-word and leaves each row's numbers in whatever column their words
    landed in, so a column index says nothing about the period.
    """
    rows: list[FinancialRow] = []
    for page in pages:
        words = getattr(page, "words", None)
        if not words:
            continue
        token_lines = [_tokenize(line) for line in _group_lines(words)]
        markers = _year_markers(token_lines)
        if not markers:
            continue
        scale, page_currency = detect_unit_and_currency(page.text or "")
        unit_default = scale or "value"
        use_currency = page_currency or currency

        for li, tokens in enumerate(token_lines):
            values = [t for t in tokens if t.is_number and not t.is_year]
            if not values:
                continue
            label = " ".join(t.label_part for t in tokens if t.label_part)
            # Statement labels wrap. "Cash and cash equivalents at the" on the
            # numbers' line and "beginning of the year" on the next is one row,
            # and read as two it loses the half that says which balance it is.
            if li + 1 < len(token_lines):
                tail = token_lines[li + 1]
                carry = " ".join(t.label_part for t in tail if t.label_part)
                if carry and not any(t.is_number for t in tail) and (
                        len(label.split()) + len(carry.split())
                        <= _MAX_LABEL_TOKENS):
                    label = f"{label} {carry}".strip()
            if len(label) < _MIN_LABEL_CHARS or len(label.split()) > _MAX_LABEL_TOKENS:
                continue
            canon = match_metric(label)
            if not canon:
                continue
            for v in values:
                assigned = _assign_column(v.center, markers)
                if not assigned:
                    continue
                val, unit_hint = _parse_number(v.text)
                if val != val:  # NaN
                    continue
                rows.append(
                    FinancialRow(
                        company=company,
                        ticker=ticker,
                        metric=canon,
                        fiscal_year=assigned[0],
                        quarter=assigned[1],
                        value=val,
                        unit=unit_hint or unit_default,
                        currency=use_currency,
                        page=page.page,
                        confidence=0.8,
                    )
                )
    return rows

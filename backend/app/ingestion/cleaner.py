"""Text cleaning + glossary mapping (PRD §local-context: cross-lingual
alignment between Indonesian and English financial terms).

Cleaning:
  * normalize mis-decoded currency glyphs to their ISO code,
  * normalize whitespace,
  * drop page-header/footer heuristics,
  * map Indonesian ↔ English financial glossary so embeddings/LLM see
    consistent terminology (e.g. "Laba Bersih" → "Net Income").

Glossary is intentionally small and additive; extend via ``add_terms``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class _Term:
    src: re.Pattern[str]
    dst: str


# Order matters: longer/more specific patterns first.
_DEFAULT_TERMS: list[tuple[str, str]] = [
    (r"\bLaba Bersih\b", "Net Income"),
    (r"\bLaba Rugi\b", "Income Statement"),
    (r"\bLaba Komprehensif\b", "Comprehensive Income"),
    (r"\bPendapatan\b", "Revenue"),
    # Multi-word Kas patterns BEFORE the generic 'Arus Kas' so they win.
    (r"\bArus Kas Operasi\b", "Operating Cash Flow"),
    (r"\bArus Kas Investasi\b", "Investing Cash Flow"),
    (r"\bArus Kas Pendanaan\b", "Financing Cash Flow"),
    (r"\bArus Kas\b", "Cash Flow"),
    (r"\bBeban Operasional\b", "Operating Expenses"),
    (r"\bBeban Pokok Penjualan\b", "Cost of Goods Sold"),
    (r"\bLaba Operasi\b", "Operating Income"),
    (r"\bHarga Per Saham\b", "Earnings Per Share"),
    (r"\bEkuitas\b", "Equity"),
    (r"\bAset\b", "Assets"),
    (r"\bKewajiban\b", "Liabilities"),
    (r"\bUtang\b", "Debt"),
    (r"\bDividen\b", "Dividend"),
    (r"\bMargin Laba Kotor\b", "Gross Margin"),
    (r"\bMargin Operasi\b", "Operating Margin"),
]


# pdfplumber decodes the Riyal sign that PIF's embedded font draws as U+00AA
# (feminine ordinal indicator), which both prints a stray glyph in answers and
# strips the currency off the figure. Only rewrite the standalone-glyph-before-
# a-number-or-scale-word shape so Spanish/Portuguese ordinals ("1ª edición")
# stay untouched. The table pipe is part of the allowed separator run because a
# rendered table row splits a figure across cells ("| ª | 56.25 |"): the sign
# lands in one cell, the number in the next, and the adjacency only exists in
# the joined markdown line. A newline is deliberately not in the class, so a
# sign at the end of one row cannot pair with a number at the start of the next.
_CURRENCY_GLYPH_RE = re.compile(
    r"(?<![0-9A-Za-z])[\xaa\ufefc\ufdfc]([ \t\u00a0|]*)"
    r"(?=[0-9\u0660-\u0669]|(?:BN|bn|Bn|billion|million|mn|trillion)\b)"
)


def _spell_out_currency(match: re.Match[str]) -> str:
    sep = match.group(1)
    # A pipe is a rendered table cell delimiter: collapsing it would merge two
    # cells and break the markdown. Any other run of blanks collapses to one
    # space, including a no-break space, so stored text stays plain.
    return "SAR" + sep if "|" in sep else "SAR "


def normalize_currency_glyphs(text: str) -> str:
    """Spell out the currency a mis-decoded Riyal sign was standing for."""
    if not text:
        return text
    return _CURRENCY_GLYPH_RE.sub(_spell_out_currency, text)


class TextCleaner:
    def __init__(self, terms: list[tuple[str, str]] | None = None) -> None:
        self._terms: list[_Term] = [
            _Term(re.compile(pat, re.IGNORECASE), dst)
            for pat, dst in (terms or _DEFAULT_TERMS)
        ]

    def add_terms(self, terms: list[tuple[str, str]]) -> None:
        for pat, dst in terms:
            self._terms.append(_Term(re.compile(pat, re.IGNORECASE), dst))

    def clean(self, text: str) -> str:
        if not text:
            return ""
        text = normalize_currency_glyphs(text)
        # Glossary mapping first (so downstream cleaning doesn't break patterns)
        for term in self._terms:
            text = term.src.sub(term.dst, text)
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Drop likely page-number-only lines (e.g. "42", "Page 42 of 100")
        text = re.sub(r"(?m)^\s*(page\s+)?\d{1,4}\s*(of\s+\d+)?\s*$", "", text)
        return text.strip()


_default_cleaner = TextCleaner()


def clean_text(text: str) -> str:
    return _default_cleaner.clean(text)

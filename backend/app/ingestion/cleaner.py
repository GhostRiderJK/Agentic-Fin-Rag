"""Text cleaning + glossary mapping (PRD §local-context: cross-lingual
alignment between Indonesian and English financial terms).

Cleaning:
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

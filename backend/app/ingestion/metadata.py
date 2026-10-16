"""Metadata extractor: derive (company, year, quarter, doc_type, section)
from a parsed document + the user-provided hints.

Used as the payload for vector points and as Document fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.db.models import DocumentType
from app.ingestion.parser import ParsedDocument

log = get_logger(__name__)


_COMPANIES = (
    "Apple", "Microsoft", "NVIDIA", "Tesla", "Alphabet", "Google", "Meta",
    "Amazon", "Amazon.com", "JPMorgan", "Johnson & Johnson", "Berkshire",
    "Adobe", "Salesforce", "Oracle", "Bank of America", "Visa", "Mastercard",
    "Bank Central Asia", "BCA", "Telkom", "Astra", "BRI", "ACME",
    "Walmart", "Procter & Gamble", "Coca-Cola", "Disney", "Pfizer",
    "Intel", "Cisco",
)

_TICKER_HINT = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "AMZN": "Amazon",
}


@dataclass
class DocMetadata:
    company: str
    ticker: str
    year: int | None
    quarter: str
    doc_type: DocumentType
    section: str


def infer_metadata(
    parsed: ParsedDocument,
    *,
    filename: str = "",
    hint_company: str = "",
    hint_ticker: str = "",
    hint_year: int | None = None,
    hint_quarter: str = "",
    hint_doc_type: DocumentType | None = None,
) -> DocMetadata:
    text = parsed.raw_text or filename
    company = hint_company or _guess_company(text, filename)
    ticker = hint_ticker or _guess_ticker(text)
    year = hint_year or _guess_year(text)
    quarter = hint_quarter or _guess_quarter(text, filename)
    doc_type = hint_doc_type or _guess_doc_type(filename, text, parsed)
    section = ""
    return DocMetadata(company=company, ticker=ticker, year=year, quarter=quarter,
                       doc_type=doc_type, section=section)


def _guess_company(text: str, filename: str) -> str:
    blob = f"{filename} {text[:4000]}"
    for c in _COMPANIES:
        if re.search(rf"\b{re.escape(c)}(?:['\u2019]?s)?\b", blob, re.IGNORECASE):
            return c
    # Try ticker pattern
    for tk, name in _TICKER_HINT.items():
        if re.search(rf"\b{tk}\b", blob):
            return name
    return ""


def _guess_ticker(text: str) -> str:
    for tk in _TICKER_HINT:
        if re.search(rf"\b{tk}\b", text[:4000]):
            return tk
    return ""


def _guess_year(text: str) -> int | None:
    years = re.findall(r"\b(20\d{2})\b", text[:8000])
    if not years:
        return None
    # Most frequent year
    from collections import Counter

    return Counter(years).most_common(1)[0][0]  # type: ignore[return-value]


def _guess_quarter(text: str, filename: str) -> str:
    blob = f"{filename} {text[:4000]}".lower()
    m = re.search(r"\bq([1-4])\b", blob)
    if m:
        return f"Q{m.group(1)}"
    if "first quarter" in blob:
        return "Q1"
    if "second quarter" in blob:
        return "Q2"
    if "third quarter" in blob:
        return "Q3"
    if "fourth quarter" in blob:
        return "Q4"
    return ""


def _guess_doc_type(
    filename: str, text: str, parsed: ParsedDocument
) -> DocumentType:
    blob = f"{filename} {text[:6000]}".lower()
    if "10-k" in blob or "annual report" in blob:
        return DocumentType.annual_report
    if "10-q" in blob or "quarterly report" in blob:
        return DocumentType.quarterly_report
    if "earnings call" in blob or "transcript" in blob or "prepared remarks" in blob:
        return DocumentType.earnings_call
    if "investor presentation" in blob or "investor day" in blob:
        return DocumentType.investor_presentation
    if "sustainability" in blob or "esg" in blob:
        return DocumentType.sustainability_report
    if "news" in blob or "press release" in blob:
        return DocumentType.news
    return DocumentType.other

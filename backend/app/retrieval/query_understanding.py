"""Query Understanding Agent (PRD v2 §3, "Metadata Filtering" section).

Extracts structured intent from a natural-language question:
  - entities (company + ticker)
  - year / quarter / time range
  - document type
  - section hints
  - a flag if this looks like a *metric lookup* (best served by the SQL
    tool-calling route rather than the document route)

Output is a ``QueryPlan`` consumed by the retrieval layer (to build a
metadata filter) and by the orchestrator (to decide routes & sub-queries).

The LLM can be used for richer extraction, but the deterministic rule-based
extractor is the default — it's fast, free, and predictable for tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.db.models import DocumentType


_COMPANIES = {
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Tesla": "TSLA",
    "Alphabet": "GOOGL", "Google": "GOOGL", "Meta": "META", "Amazon": "AMZN",
    "Amazon.com": "AMZN",
    "JPMorgan": "JPM", "Johnson & Johnson": "JNJ", "Visa": "V",
    "Walmart": "WMT", "Procter & Gamble": "PG", "Coca-Cola": "KO",
    "Disney": "DIS", "Pfizer": "PFE", "Intel": "INTC", "Cisco": "CSCO",
    "Oracle": "ORCL", "Adobe": "ADBE", "Salesforce": "CRM",
    "Bank Central Asia": "BBCA", "BCA": "BBCA", "Telkom": "TLKM",
    "Astra": "ASII", "BRI": "BBRI",
    "ACME": "ACME",
}

_DOC_TYPE_HINTS = {
    DocumentType.annual_report: [r"\b10-?k\b", r"annual report"],
    DocumentType.quarterly_report: [r"\b10-?q\b", r"quarterly report"],
    DocumentType.earnings_call: [r"earnings call", r"transcript", r"prepared remarks"],
    DocumentType.investor_presentation: [r"investor (?:day|presentation|deck)"],
    DocumentType.sustainability_report: [r"sustainability", r"\besg\b"],
    DocumentType.news: [r"\bnews\b", r"press release"],
    DocumentType.research_note: [r"research note"],
}

_METRIC_HINTS = [
    "revenue", "eps", "earnings per share", "net income", "ebitda",
    "operating margin", "gross margin", "cash flow", "debt", "dividend",
    "pe ratio", "price to earnings", "free cash flow", "capex",
]


@dataclass
class QueryPlan:
    question: str
    companies: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    quarters: list[str] = field(default_factory=list)
    doc_types: list[DocumentType] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    is_metric_lookup: bool = False
    is_comparison: bool = False
    sub_queries: list[str] = field(default_factory=list)

    def to_metadata_filter(self, *, available_years: list[int] | None = None) -> dict:
        """Build the metadata prefilter dict consumed by the vector store.

        Years are filtered *only* when they match an existing document year.
        This avoids a common failure: "What's the 2026 outlook?" mentions 2026
        but the relevant document is the 2025 annual report that contains the
        outlook. If ``available_years`` is provided, we keep only the query
        years that actually exist in the corpus; if none survive we drop the
        year filter entirely (fall back to all years).
        """
        flt: dict = {}
        if self.companies:
            # Multi-company comparison → list match
            flt["company"] = self.companies if len(self.companies) > 1 else self.companies[0]
        if self.years:
            years = self.years
            if available_years is not None:
                surviving = [y for y in years if y in available_years]
                # Only apply the year filter if at least one mentioned year
                # actually exists in the corpus. Otherwise, treat the year as
                # a content reference (e.g. outlook about 2026) not a filter.
                if surviving:
                    years = surviving
                else:
                    years = []
            if years:
                flt["year"] = years if len(years) > 1 else years[0]
        if self.quarters:
            flt["quarter"] = self.quarters if len(self.quarters) > 1 else self.quarters[0]
        if self.doc_types:
            flt["doc_type"] = (
                [d.value for d in self.doc_types]
                if len(self.doc_types) > 1 else self.doc_types[0].value
            )
        return flt


def understand_query(question: str) -> QueryPlan:
    plan = QueryPlan(question=question.strip())
    q = question.strip()
    ql = q.lower()

    # Companies — allow possessive/plural forms (Apple's, ACMEs, Teslas, etc.)
    for name, ticker in _COMPANIES.items():
        # Match "Apple", "Apple's", "Apples" — i.e. name followed by optional 's or s
        if re.search(rf"\b{re.escape(name)}(?:['\u2019]?s)?\b", q, re.IGNORECASE):
            if name not in plan.companies:
                plan.companies.append(name)
                plan.tickers.append(ticker)
    # Direct ticker matches (e.g. "NVDA")
    for name, ticker in _COMPANIES.items():
        if re.search(rf"\b{ticker}\b", q) and ticker not in plan.tickers:
            plan.tickers.append(ticker)
            if name not in plan.companies:
                plan.companies.append(name)

    # Years (take all 4-digit years found)
    plan.years = sorted({int(m) for m in re.findall(r"\b(20\d{2})\b", q)})
    # Quarters
    for m in re.findall(r"\bQ([1-4])\b", q, re.IGNORECASE):
        qn = f"Q{m}"
        if qn not in plan.quarters:
            plan.quarters.append(qn)
    for word, qn in [("first", "Q1"), ("second", "Q2"),
                     ("third", "Q3"), ("fourth", "Q4")]:
        if re.search(rf"\b{word} quarter\b", ql):
            if qn not in plan.quarters:
                plan.quarters.append(qn)

    # Document types
    for dt, patterns in _DOC_TYPE_HINTS.items():
        for pat in patterns:
            if re.search(pat, ql):
                if dt not in plan.doc_types:
                    plan.doc_types.append(dt)
                break

    # Sections (free-form)
    for sec in ("Risk Factors", "Management's Discussion", "Business Overview",
                "Outlook", "Corporate Governance", "Income Statement",
                "Balance Sheet", "Cash Flow"):
        if sec.lower() in ql:
            plan.sections.append(sec)

    # Comparison?
    if any(kw in ql for kw in ("compare", "vs", "versus", " versus ", "difference")):
        plan.is_comparison = True
    if len(plan.companies) >= 2:
        plan.is_comparison = True

    # Metric lookup? (best served by SQL tool-calling). Fires when a known
    # metric keyword appears alongside a company/period — i.e. the question
    # is asking for a numeric fact rather than a narrative.
    has_metric = any(re.search(rf"\b{re.escape(m)}\b", ql) for m in _METRIC_HINTS)
    has_entity = bool(plan.companies) or bool(plan.years)
    is_question = any(kw in ql for kw in (
        "what", "how much", "how many", "what's", "whats", "compare",
        "vs", "versus", "show", "get", "tell me", "=", "did",
    ))
    plan.is_metric_lookup = has_metric and has_entity and is_question

    # Sub-query decomposition for comparisons / multi-year questions
    if plan.is_comparison and plan.companies:
        for c in plan.companies:
            plan.sub_queries.append(f"{c} {q.replace('compare', '').strip()}")
    elif plan.years and len(plan.years) > 1:
        for y in plan.years:
            plan.sub_queries.append(q.replace(str(plan.years[0]), str(y)))

    return plan


def rewrite_query_with_history(question: str, history: list[tuple[str, str]]) -> str:
    """Light query rewriting: if the question uses pronouns / is elliptical,
    prepend the last user question's company for context.

    A full LLM-based rewriter lives in ``conversation.py``; this is the
    deterministic fallback used in tests / offline mode.
    """
    if not history:
        return question
    last_user, _ = history[-1]
    # If current question names no company but the previous did, carry it.
    current_plan = understand_query(question)
    if current_plan.companies:
        return question
    prev_plan = understand_query(last_user)
    if prev_plan.companies:
        return f"{prev_plan.companies[0]} — {question}"
    return question

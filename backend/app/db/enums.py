"""Document lifecycle enums.

These live outside ``app.db.models`` because document data is stored in
MongoDB: the values describe the domain, not a table column, and both the Mongo
documents and the request schemas use them.
"""

from __future__ import annotations

import enum


class DocumentStatus(str, enum.Enum):
    active = "active"
    superseded = "superseded"
    archived = "archived"


class DocumentType(str, enum.Enum):
    annual_report = "annual_report"
    quarterly_report = "quarterly_report"
    earnings_call = "earnings_call"
    investor_presentation = "investor_presentation"
    sustainability_report = "sustainability_report"
    news = "news"
    research_note = "research_note"
    other = "other"

"""Structured-data tools the Analyst agent can call (PRD v2 §8).

These expose the relational ``StructuredFinancials`` store to the LLM via
function-calling. Numeric facts live here — the LLM never has to recite them.

Tools implemented:
  - get_financial_metric(company, metric, fiscal_year, quarter)
  - list_metrics(company, fiscal_year)
  - compare_metrics(companies, metric, fiscal_year)
  - list_periods(company)

Each tool is exposed both as a callable (for the orchestrator) and as a
``ToolSpec`` for the LLM (function-calling schema).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import StructuredFinancial
from app.providers.llm import ToolSpec


def _normalize_company(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "")).strip()


def get_financial_metric(
    db: Session,
    *,
    workspace_id: str,
    company: str,
    metric: str,
    fiscal_year: int | None = None,
    quarter: str = "",
) -> dict:
    company = _normalize_company(company)
    q = select(StructuredFinancial).where(
        StructuredFinancial.workspace_id == workspace_id,
        StructuredFinancial.company.ilike(f"%{company}%"),
        StructuredFinancial.metric == metric,
    )
    if fiscal_year is not None:
        q = q.where(StructuredFinancial.fiscal_year == fiscal_year)
    if quarter:
        q = q.where(StructuredFinancial.quarter == quarter.upper())
    q = q.order_by(StructuredFinancial.fiscal_year.desc(),
                   StructuredFinancial.quarter.desc())
    rows = list(db.scalars(q).all())
    return {"metric": metric, "company": company, "results": [_row_dict(r) for r in rows]}


def list_metrics(
    db: Session, *, workspace_id: str, company: str, fiscal_year: int | None = None,
) -> dict:
    company = _normalize_company(company)
    q = select(StructuredFinancial).where(
        StructuredFinancial.workspace_id == workspace_id,
        StructuredFinancial.company.ilike(f"%{company}%"),
    )
    if fiscal_year is not None:
        q = q.where(StructuredFinancial.fiscal_year == fiscal_year)
    rows = list(db.scalars(q).all())
    metrics = sorted({r.metric for r in rows})
    return {"company": company, "fiscal_year": fiscal_year, "metrics": metrics,
            "count": len(metrics)}


def compare_metrics(
    db: Session, *, workspace_id: str, companies: list[str],
    metric: str, fiscal_year: int | None = None,
) -> dict:
    out: dict[str, Any] = {"metric": metric, "fiscal_year": fiscal_year, "results": {}}
    for c in companies:
        r = get_financial_metric(db, workspace_id=workspace_id, company=c,
                                 metric=metric, fiscal_year=fiscal_year)
        out["results"][c] = r["results"]
    return out


def list_periods(db: Session, *, workspace_id: str, company: str) -> dict:
    company = _normalize_company(company)
    q = select(StructuredFinancial).where(
        StructuredFinancial.workspace_id == workspace_id,
        StructuredFinancial.company.ilike(f"%{company}%"),
    )
    rows = list(db.scalars(q).all())
    periods = sorted({(r.fiscal_year, r.quarter) for r in rows}, reverse=True)
    return {"company": company,
            "periods": [{"year": y, "quarter": q or "FY"} for y, q in periods]}


def _row_dict(r: StructuredFinancial) -> dict:
    return {
        "company": r.company,
        "ticker": r.ticker,
        "metric": r.metric,
        "fiscal_year": r.fiscal_year,
        "quarter": r.quarter or "FY",
        "value": r.value,
        "unit": r.unit,
        "currency": r.currency,
        "page": r.page,
    }


# ---- Tool registry: maps tool name → (callable, ToolSpec) ------------------

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_financial_metric",
        description=(
            "Look up a NUMERIC financial metric (revenue, eps, net_income, "
            "ebitda, operating_margin, total_debt, etc.) for a company, "
            "optionally for a specific fiscal year and quarter. Use this "
            "instead of reciting numbers from memory. Returns exact figures "
            "with units and source page.\n\n"
            "IMPORTANT: This tool only returns NUMERIC data. Do NOT call it "
            "for textual content like 'risk_factors', 'strategy', 'business "
            "overview', 'outlook', 'competition' — those are in the retrieved "
            "document context, not in this tool. Valid metric keys: revenue, "
            "eps, net_income, gross_profit, operating_income, ebitda, "
            "operating_margin, gross_margin, total_assets, total_liabilities, "
            "total_equity, total_debt, cash_and_equivalents, "
            "operating_cash_flow, free_cash_flow, dividend_per_share, "
            "shares_outstanding, rd_expense, capex, cost_of_goods_sold."
        ),
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name, e.g. Apple"},
                "metric": {
                    "type": "string",
                    "description": "Metric key (numeric only): revenue, eps, net_income, ebitda, operating_margin, total_debt, etc.",
                },
                "fiscal_year": {"type": "integer", "description": "e.g. 2025"},
                "quarter": {"type": "string", "description": "e.g. Q1 or empty for FY"},
            },
            "required": ["company", "metric"],
        },
    ),
    ToolSpec(
        name="compare_metrics",
        description="Compare the same metric across multiple companies.",
        parameters={
            "type": "object",
            "properties": {
                "companies": {"type": "array", "items": {"type": "string"}},
                "metric": {"type": "string"},
                "fiscal_year": {"type": "integer"},
            },
            "required": ["companies", "metric"],
        },
    ),
    ToolSpec(
        name="list_metrics",
        description="List all available structured metrics for a company.",
        parameters={
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "fiscal_year": {"type": "integer"},
            },
            "required": ["company"],
        },
    ),
    ToolSpec(
        name="list_periods",
        description="List available fiscal periods for a company.",
        parameters={
            "type": "object",
            "properties": {"company": {"type": "string"}},
            "required": ["company"],
        },
    ),
]


def build_dispatcher(
    db: Session, *, workspace_id: str
) -> tuple[Callable[[str, dict], dict], list[ToolSpec]]:
    """Return (dispatch_fn, tool_specs). The orchestrator calls
    dispatch_fn(name, args) to execute a tool call."""

    def dispatch(name: str, args: dict) -> dict:
        if name == "get_financial_metric":
            return get_financial_metric(db, workspace_id=workspace_id, **args)
        if name == "compare_metrics":
            return compare_metrics(db, workspace_id=workspace_id, **args)
        if name == "list_metrics":
            return list_metrics(db, workspace_id=workspace_id, **args)
        if name == "list_periods":
            return list_periods(db, workspace_id=workspace_id, **args)
        return {"error": f"unknown tool: {name}"}

    return dispatch, TOOL_SPECS

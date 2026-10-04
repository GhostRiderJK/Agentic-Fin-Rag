"""Seed the structured_financials table from the ACME sample data so the SQL
tool-calling route is demonstrable even without PDF tables.

Run:  python -m app.seed.seed_financials
"""

from __future__ import annotations

import typer
from rich.console import Console
from sqlalchemy import select

from app.db.base import db_scope, init_db
from app.db.models import StructuredFinancial, Workspace

app = typer.Typer(add_completion=False)
console = Console()


# Values from the ACME sample annual report's financial highlights section.
_ROWS = [
    # (metric, year, quarter, value, unit, page)
    ("revenue", 2025, "", 120_000.0, "million USD", 4),
    ("revenue", 2024, "", 110_000.0, "million USD", 4),
    ("cost_of_goods_sold", 2025, "", 68_400.0, "million USD", 4),
    ("cost_of_goods_sold", 2024, "", 59_400.0, "million USD", 4),
    ("gross_profit", 2025, "", 51_600.0, "million USD", 4),
    ("gross_profit", 2024, "", 50_600.0, "million USD", 4),
    ("operating_income", 2025, "", 36_000.0, "million USD", 4),
    ("operating_income", 2024, "", 35_200.0, "million USD", 4),
    ("net_income", 2025, "", 28_000.0, "million USD", 4),
    ("net_income", 2024, "", 27_500.0, "million USD", 4),
    ("eps", 2025, "", 5.20, "USD", 4),
    ("eps", 2024, "", 4.95, "USD", 4),
    ("operating_cash_flow", 2025, "", 32_000.0, "million USD", 4),
    ("operating_cash_flow", 2024, "", 30_000.0, "million USD", 4),
    ("capex", 2025, "", 4_000.0, "million USD", 4),
    ("capex", 2024, "", 3_500.0, "million USD", 4),
    ("free_cash_flow", 2025, "", 28_000.0, "million USD", 4),
    ("free_cash_flow", 2024, "", 26_500.0, "million USD", 4),
    ("total_debt", 2025, "", 12_000.0, "million USD", 4),
    ("total_debt", 2024, "", 13_000.0, "million USD", 4),
    ("dividend_per_share", 2025, "", 1.40, "USD", 4),
    ("dividend_per_share", 2024, "", 1.20, "USD", 4),
    # Q4 quarterly from earnings call
    ("revenue", 2025, "Q4", 32_000.0, "million USD", 2),
    ("eps", 2025, "Q4", 1.45, "USD", 2),
    ("operating_cash_flow", 2025, "Q4", 9_000.0, "million USD", 2),
]


@app.callback(invoke_without_command=True)
def main():
    init_db()
    with db_scope() as db:
        ws = db.scalar(select(Workspace).order_by(Workspace.created_at.desc()))
        if not ws:
            console.print("[red]✗[/red] no workspace. Run `ira seed` first.")
            raise typer.Exit(1)
        added = 0
        for metric, year, quarter, value, unit, page in _ROWS:
            existing = db.scalar(select(StructuredFinancial).where(
                StructuredFinancial.workspace_id == ws.id,
                StructuredFinancial.company == "ACME",
                StructuredFinancial.metric == metric,
                StructuredFinancial.fiscal_year == year,
                StructuredFinancial.quarter == quarter,
                StructuredFinancial.unit == unit,
            ))
            if existing:
                existing.value = value
                continue
            db.add(StructuredFinancial(
                workspace_id=ws.id, company="ACME", ticker="ACME",
                metric=metric, fiscal_year=year, quarter=quarter,
                value=value, unit=unit, currency="USD", page=page,
                confidence=1.0,
            ))
            added += 1
        console.print(f"[green]✓[/green] seeded {added} financial rows "
                      f"(workspace={ws.id})")


if __name__ == "__main__":
    app()

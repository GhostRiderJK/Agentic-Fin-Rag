"""Command-line interface.

Commands:
  ira serve            — run the FastAPI app with uvicorn
  ira init-db          — create database tables
  ira seed             — create a demo tenant + user + workspace + sample doc
  ira ingest <file>    — ingest a file into a workspace (sync)
  ira eval <suite>     — run an evaluation suite
  ira ask <question>   — quick one-shot Q&A (uses default workspace)

Usage:
  python -m app.cli serve
  ira seed
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app import __version__
from app.core.config import get_settings

app = typer.Typer(add_completion=False, help="Investment Research Assistant CLI")
console = Console()


@app.command()
def serve(host: str = "", port: int = 0, reload: bool = False):
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=host or settings.app_host,
        port=port or settings.app_port,
        reload=reload,
    )


@app.command("init-db")
def init_db_cmd():
    """Create all database tables."""
    from app.db.base import init_db

    get_settings().ensure_dirs()
    init_db()
    console.print("[green]✓[/green] database initialized")


@app.command()
def seed():
    """Create a demo tenant, user, workspace, and a sample document."""
    from datetime import datetime, timezone

    from sqlalchemy import select

    from app.db.base import db_scope, init_db
    from app.db.models import Role, Tenant, User, Workspace
    from app.core.security import new_api_key

    init_db()
    with db_scope() as db:
        existing = db.scalar(select(Tenant).where(Tenant.name == "Demo Firm"))
        if existing:
            console.print(
                f"[yellow]![/yellow] demo tenant exists: {existing.id}"
            )
            tenant = existing
        else:
            tenant = Tenant(name="Demo Firm")
            db.add(tenant)
            db.flush()
            console.print(f"[green]✓[/green] created tenant: {tenant.id}")

        user = db.scalar(select(User).where(User.email == "analyst@demo.local"))
        if not user:
            user = User(
                tenant_id=tenant.id, email="analyst@demo.local",
                name="Demo Analyst", role=Role.admin, api_key=new_api_key(),
            )
            db.add(user)
            db.flush()
            console.print(f"[green]✓[/green] created user: {user.email}")

        # ALWAYS print the api_key so re-running seed is sufficient to recover
        # the credentials (previously it only printed on first creation).
        console.print(f"    email   : {user.email}")
        console.print(f"    api_key : [bold]{user.api_key}[/bold]")

        ws = db.scalar(select(Workspace).where(
            Workspace.tenant_id == tenant.id, Workspace.name == "Demo Workspace"))
        if not ws:
            ws = Workspace(tenant_id=tenant.id, name="Demo Workspace",
                           description="Seed workspace for demo")
            db.add(ws)
            db.flush()
            console.print(f"[green]✓[/green] created workspace: {ws.id}")
        console.print(f"    workspace: {ws.id}  ({ws.name})")

    console.print(
        "\n[dim]Paste the api_key above into the UI's 'Connect with API "
        "key' field.[/dim]"
    )


@app.command()
def ingest(
    file: str,
    workspace_id: str = typer.Option("", help="Workspace ID"),
    company: str = "",
    ticker: str = "",
    year: int = 0,
    quarter: str = "",
):
    """Ingest a file synchronously into the given workspace."""
    from sqlalchemy import select

    from app.db.base import db_scope, init_db
    from app.db.models import Workspace
    from app.ingestion.pipeline import ingest_document

    init_db()
    with db_scope() as db:
        ws = None
        if workspace_id:
            ws = db.get(Workspace, workspace_id)
        else:
            ws = db.scalar(select(Workspace).order_by(Workspace.created_at.desc()))
        if not ws:
            console.print("[red]✗[/red] no workspace found. Run `ira seed` first.")
            raise typer.Exit(1)
        console.print(f"Ingesting [bold]{file}[/bold] into workspace {ws.id}…")
        result = ingest_document(
            file_path=file, workspace_id=ws.id, db=db,
            hint_company=company, hint_ticker=ticker,
            hint_year=year or None, hint_quarter=quarter,
        )
        console.print(
            f"[green]✓[/green] ingested doc {result.document_id} "
            f"(parents={result.n_parents} children={result.n_children} "
            f"tables={result.n_tables} financials={result.n_financials})"
        )


@app.command()
def eval_cmd(
    suite: str = typer.Option("", help="Path to JSONL suite"),
    workspace_id: str = typer.Option("", help="Workspace ID"),
):
    """Run an evaluation suite."""
    from sqlalchemy import select

    from app.db.base import db_scope, init_db
    from app.db.models import Workspace
    from app.evaluation.harness import EvalRunner, EvalSuite

    init_db()
    suite_path = suite or str(Path(__file__).resolve().parent / "seed" /
                               "gold_qa.jsonl")
    with db_scope() as db:
        ws = None
        if workspace_id:
            ws = db.get(Workspace, workspace_id)
        else:
            ws = db.scalar(select(Workspace).order_by(Workspace.created_at.desc()))
        if not ws:
            console.print("[red]✗[/red] no workspace found.")
            raise typer.Exit(1)
        s = EvalSuite.from_jsonl(suite_path, workspace_id=ws.id)
        runner = EvalRunner(db)
        result = runner.run(s, persist=True)
        table = Table(title=f"Eval: {result.suite} ({result.n_items} items)")
        table.add_column("Metric")
        table.add_column("Mean", justify="right")
        table.add_column("Pass-rate", justify="right")
        keys = {k[len("mean_"):] for k in result.metrics if k.startswith("mean_")}
        for k in sorted(keys):
            table.add_row(
                k,
                f"{result.metrics.get(f'mean_{k}', 0):.3f}",
                f"{result.metrics.get(f'pass_{k}', 0):.2%}",
            )
        console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(...),
    workspace_id: str = typer.Option("", help="Workspace ID"),
):
    """One-shot Q&A against a workspace."""
    from sqlalchemy import select

    from app.db.base import db_scope, init_db
    from app.db.models import Workspace
    from app.reasoning.orchestrator import AnswerRequest, Orchestrator

    init_db()
    with db_scope() as db:
        ws = None
        if workspace_id:
            ws = db.get(Workspace, workspace_id)
        else:
            ws = db.scalar(select(Workspace).order_by(Workspace.created_at.desc()))
        if not ws:
            console.print("[red]✗[/red] no workspace found.")
            raise typer.Exit(1)
        orch = Orchestrator(db)
        resp = orch.answer(AnswerRequest(question=question, workspace_id=ws.id))
        console.print(resp.answer)
        console.print(f"\n[dim]latency={resp.latency_ms}ms "
                      f"chunks={resp.retrieval.get('n_chunks', 0)} "
                      f"verification={resp.verification.get('outcome')}[/dim]")


# Rename eval command to avoid shadowing builtin
app.command(name="eval")(eval_cmd)


if __name__ == "__main__":
    app()

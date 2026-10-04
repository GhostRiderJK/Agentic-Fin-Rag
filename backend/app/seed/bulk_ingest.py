"""Bulk-ingest all downloaded EDGAR filings into a workspace via the running
server's HTTP API.

WHY via HTTP (not CLI): Qdrant local mode is single-process. If you ingest
from a separate CLI process while the server runs, the server can't see
those vectors. This script uploads each file to /workspaces/{id}/documents
so vectors land in the server's process.

Usage:
    # 1. Start the server first:
    #    python -m app.cli serve
    #
    # 2. In another terminal:
    python -m app.seed.bulk_ingest \\
        --api-key 090e602f... \\
        --workspace-id e15ca286... \\
        --dir ./data/edgar_pdfs

If --api-key / --workspace-id are omitted, it tries to read them from the
demo DB (the user created by `ira seed` and its most recent workspace).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

app = typer.Typer(add_completion=False)
console = Console()

BASE = "http://127.0.0.1:8000"

# Form-type → document_type
DOC_TYPE_MAP = {
    "10-K": "annual_report",
    "10-Q": "quarterly_report",
}

# Ticker → company name (matches metadata.py known companies)
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla",
    "META": "Meta", "AMZN": "Amazon", "GOOGL": "Alphabet", "JPM": "JPMorgan",
    "JNJ": "Johnson & Johnson", "V": "Visa", "WMT": "Walmart", "PG": "Procter & Gamble",
    "KO": "Coca-Cola", "DIS": "Disney", "PFE": "Pfizer", "INTC": "Intel",
    "CSCO": "Cisco", "ORCL": "Oracle", "ADBE": "Adobe", "CRM": "Salesforce",
}

FILENAME_RE = re.compile(r"^(?P<ticker>[A-Z]+)_(?P<form>10-[KQ])_(?P<date>\d{4}-\d{2}-\d{2})")


def parse_filename(path: Path) -> dict:
    m = FILENAME_RE.match(path.name)
    if not m:
        return {}
    ticker = m.group("ticker")
    form = m.group("form")
    date = m.group("date")
    year = int(date[:4])
    month = int(date[5:7])
    # quarter from filing month
    quarter = f"Q{((month - 1) // 3) + 1}"
    return {
        "ticker": ticker,
        "company": COMPANY_NAMES.get(ticker, ticker),
        "doc_type": DOC_TYPE_MAP.get(form, "other"),
        "year": str(year),
        "quarter": quarter if form == "10-Q" else "",
    }


def healthcheck() -> bool:
    try:
        r = httpx.get(f"{BASE}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


@app.callback(invoke_without_command=True)
def main(
    api_key: str = typer.Option("", help="API key (omitted = read from demo DB)"),
    workspace_id: str = typer.Option("", help="Workspace ID (omitted = newest demo ws)"),
    dir: str = typer.Option("./data/edgar_pdfs", help="Directory of filings"),
    limit: int = typer.Option(0, help="Max files to ingest (0 = all)"),
    sleep_s: float = typer.Option(2.0, help="Sleep between uploads (give worker breathing room)"),
    sync: bool = typer.Option(True, help="Wait for each ingest to finish before the next (avoids SQLite write lock contention with the worker thread)"),
    skip_existing: bool = typer.Option(True, help="Skip files whose filename already exists as a document in the workspace"),
):
    """Bulk-ingest EDGAR filings via the running server's HTTP API."""
    if not healthcheck():
        console.print(
            f"[red]✗[/red] server not reachable at {BASE}. "
            "Start it first:  python -m app.cli serve"
        )
        raise typer.Exit(1)

    # Resolve credentials from demo DB if not provided
    if not api_key or not workspace_id:
        from sqlalchemy import select

        from app.db.base import db_scope
        from app.db.models import User, Workspace

        with db_scope() as db:
            if not api_key:
                u = db.scalar(select(User).where(User.email == "analyst@demo.local"))
                if u:
                    api_key = u.api_key
            if not workspace_id:
                # Pick the newest workspace belonging to the api_key's tenant,
                # NOT just the newest workspace globally (which may belong to
                # a different tenant and would 404).
                user = db.scalar(select(User).where(User.api_key == api_key))
                if user:
                    ws = db.scalar(
                        select(Workspace)
                        .where(Workspace.tenant_id == user.tenant_id)
                        .order_by(Workspace.created_at.desc())
                    )
                    if ws:
                        workspace_id = ws.id
    if not api_key or not workspace_id:
        console.print(
            "[red]✗[/red] could not resolve api_key / workspace_id.\n"
            "  - Make sure `python -m app.cli seed` was run (creates demo user+ws).\n"
            "  - Or pass them explicitly: --api-key ... --workspace-id ..."
        )
        raise typer.Exit(1)
    console.print(f"using api_key={api_key[:12]}…  workspace_id={workspace_id}")

    # Pre-flight: verify the workspace is accessible to this api_key before
    # attempting 99 uploads that would all 404.
    try:
        check = httpx.get(
            f"{BASE}/workspaces/{workspace_id}/documents",
            headers={"X-API-Key": api_key},
            timeout=10.0,
        )
        if check.status_code == 404:
            console.print(
                f"[red]✗[/red] workspace {workspace_id} not found (or belongs to "
                "a different tenant than this API key).\n"
                "  Fix: create a workspace via the UI, or run `python -m app.cli seed` "
                "to use the demo workspace, then re-run this script.\n"
                "  Or pass an explicit workspace that you own via --workspace-id."
            )
            raise typer.Exit(1)
        if check.status_code != 200:
            console.print(
                f"[red]✗[/red] pre-flight check failed: HTTP {check.status_code} {check.text[:200]}"
            )
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗[/red] pre-flight check error: {e}")
        raise typer.Exit(1)
    console.print("[green]✓[/green] workspace accessible\n")

    src = Path(dir)
    if not src.exists():
        console.print(f"[red]✗[/red] directory not found: {src}")
        raise typer.Exit(1)
    files = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".htm", ".html"}
    )

    # Optionally skip files whose filename already exists in the workspace.
    existing_filenames: set[str] = set()
    if skip_existing:
        try:
            resp = httpx.get(
                f"{BASE}/workspaces/{workspace_id}/documents",
                headers={"X-API-Key": api_key},
                timeout=30.0,
            )
            if resp.status_code == 200:
                existing_filenames = {d.get("filename", "") for d in resp.json()}
        except Exception:
            pass
        before = len(files)
        files = [p for p in files if p.name not in existing_filenames]
        skipped = before - len(files)
        if skipped:
            console.print(f"[dim]Skipping {skipped} already-ingested files[/dim]")

    if limit > 0:
        files = files[:limit]
    console.print(f"[bold]Ingesting {len(files)} filings via HTTP…[/bold]")

    headers = {"X-API-Key": api_key}
    url = f"{BASE}/workspaces/{workspace_id}/documents"

    succeeded = 0
    failed = 0
    # Use a long timeout because sync ingest of a large 10-K can take
    # minutes (parsing + embedding hundreds of chunks).
    with httpx.Client(timeout=900.0) as client, Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("ingest", total=len(files))
        for path in files:
            progress.advance(task)
            meta = parse_filename(path)
            if not meta:
                console.print(f"[yellow]skip[/yellow] (bad filename): {path.name}")
                failed += 1
                continue
            progress.update(task, description=f"{meta['ticker']} {meta.get('year')}")
            # sync=True (default): wait for each ingest to finish before the
            # next upload. This serializes DB writes and avoids SQLite
            # "database is locked" contention with the async worker thread.
            # sync=False: queue the job and move on (much faster uploads, but
            # needs Postgres or a remote DB to handle concurrent writes well).
            try:
                with path.open("rb") as f:
                    r = client.post(
                        url,
                        headers=headers,
                        files={"file": (path.name, f, "application/octet-stream")},
                        data={
                            "company": meta["company"],
                            "ticker": meta["ticker"],
                            "year": meta["year"],
                            "quarter": meta["quarter"],
                            "doc_type": meta["doc_type"],
                            "async_ingest": "false" if sync else "true",
                        },
                    )
                if r.status_code == 200:
                    succeeded += 1
                    if sync:
                        body = r.json()
                        console.print(
                            f"  [green]✓[/green] {path.name}: "
                            f"{body.get('n_parents',0)} parents / "
                            f"{body.get('n_children',0)} children / "
                            f"{body.get('n_financials',0)} financials"
                        )
                else:
                    console.print(
                        f"[red]✗[/red] {path.name}: HTTP {r.status_code} {r.text[:200]}"
                    )
                    failed += 1
            except Exception as e:
                console.print(f"[red]✗[/red] {path.name}: {e}")
                failed += 1
            time.sleep(sleep_s if not sync else 0.2)

    console.print(
        f"\n[green]✓[/green] queued {succeeded}/{len(files)} for ingestion "
        f"({failed} failed)"
    )
    if sync:
        console.print(
            f"\n[green]✓[/green] {succeeded} documents ingested and immediately "
            "available for chat (sync mode)."
        )
    else:
        console.print(
            "\n[dim]Ingestion queued — vectors appear gradually as the worker "
            "processes each job. Check the Documents tab in the UI; status "
            "flips to 'active' as each finishes.[/dim]"
        )
        console.print(
            f"[dim]Monitor: GET {BASE}/jobs?workspace_id={workspace_id}[/dim]"
        )


if __name__ == "__main__":
    app()

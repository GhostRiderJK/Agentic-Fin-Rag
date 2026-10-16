"""Download 100+ financial PDFs from SEC EDGAR (free, public domain).

Strategy:
  - For each company, fetch its submissions JSON.
  - Pick recent 10-K and 10-Q filings.
  - For each filing, open the index.json, find the primary .pdf (or .htm
    primary doc — saved as .htm and ingested as HTML).
  - Download into ./data/edgar_pdfs/<ticker>/<form>_<date>.<ext>

Usage:
    python -m app.seed.download_pdfs --out ./data/edgar_pdfs --target 100

Notes:
  - SEC requires a descriptive User-Agent header. We set one.
  - SEC rate-limit: ~10 req/sec. We sleep 0.15s between requests.
  - All filings on EDGAR are public domain (filed by public companies).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

app = typer.Typer(add_completion=False)
console = Console()

# CIK map for a set of large, well-known filers with consistent PDF/HTML
# filings. CIKs are zero-padded to 10 digits as EDGAR requires.
COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "META": "0001326801",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "JPM": "0000019617",
    "JNJ": "0000200406",
    "V": "0001403161",
    "WMT": "0000104169",
    "PG": "0000080424",
    "KO": "0000021344",
    "DIS": "0001744489",  # older Disney CIK; updated below if needed
    "PFE": "0000078003",
    "INTC": "0000050863",
    "CSCO": "0000858877",
    "ORCL": "0001341439",
    "ADBE": "0000796343",
    "CRM": "0001108524",
}

USER_AGENT = "Investment Research research@example.com"
# submissions JSON lives on data.sec.gov; the archive tree lives on www.sec.gov
DATA_BASE = "https://data.sec.gov"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def fetch(url: str, client: httpx.Client) -> dict | str:
    r = client.get(url, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "json" in ctype:
        return r.json()
    return r.text


def list_filings(ticker: str, cik: str, client: httpx.Client,
                 *, want_forms: list[str], limit: int) -> list[dict]:
    """Return list of recent filings of the given forms."""
    data = fetch(f"{DATA_BASE}/submissions/CIK{cik}.json", client)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary = recent.get("primaryDocument", [])
    primary_desc = recent.get("primaryDocDescription", [])

    out: list[dict] = []
    for i, form in enumerate(forms):
        if form in want_forms:
            # build accession folder path: dashes removed
            acc_nodash = accessions[i].replace("-", "")
            out.append({
                "ticker": ticker,
                "cik": cik,
                "form": form,
                "date": dates[i],
                "accession": accessions[i],
                "acc_folder": f"{ARCHIVE_BASE}/{int(cik)}/{acc_nodash}",
                "primary_doc": primary[i],
                "primary_desc": primary_desc[i] if i < len(primary_desc) else "",
            })
            if len(out) >= limit:
                break
    return out


def download_filing(filing: dict, out_dir: Path, client: httpx.Client) -> Path | None:
    """Download the primary document of a filing. Returns local path or None."""
    primary = filing["primary_doc"]
    url = f"{filing['acc_folder']}/{primary}"
    ext = Path(primary).suffix.lower()
    if ext not in {".pdf", ".htm", ".html"}:
        # Skip exotic formats
        return None
    ticker = filing["ticker"]
    fname = f"{ticker}_{filing['form']}_{filing['date']}{ext}"
    dest_dir = out_dir / ticker
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    if dest.exists() and dest.stat().st_size > 1024:
        return dest  # already downloaded
    try:
        with client.stream("GET", url, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except Exception as e:
        console.print(f"[red]✗[/red] {filing['ticker']} {filing['date']}: {e}")
        return None
    return dest


@app.callback(invoke_without_command=True)
def main(
    out: str = typer.Option("./data/edgar_pdfs", help="Output directory"),
    target: int = typer.Option(100, help="Target number of filings"),
    forms: str = typer.Option("10-K,10-Q", help="Comma-separated form types"),
):
    """Download financial PDFs/HTMLs from SEC EDGAR."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    want_forms = [f.strip() for f in forms.split(",") if f.strip()]
    # Allocate target across companies
    per_company = max(1, target // len(COMPANIES))

    all_filings: list[dict] = []
    console.print(f"[bold]Fetching filings index for {len(COMPANIES)} companies…[/bold]")
    with httpx.Client(timeout=30.0) as client:
        for ticker, cik in COMPANIES.items():
            try:
                fl = list_filings(ticker, cik, client, want_forms=want_forms,
                                  limit=per_company)
                all_filings.extend(fl)
                console.print(f"  {ticker}: {len(fl)} filings queued")
                time.sleep(0.15)  # be polite
            except Exception as e:
                console.print(f"  [red]{ticker}[/red]: {e}")
        all_filings = all_filings[:target]
        console.print(f"\n[bold]Downloading {len(all_filings)} filings…[/bold]")
        downloaded = 0
        skipped = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("download", total=len(all_filings))
            for f in all_filings:
                progress.advance(task)
                p = download_filing(f, out_dir, client)
                if p:
                    downloaded += 1
                else:
                    skipped += 1
                time.sleep(0.15)
    console.print(
        f"\n[green]✓[/green] downloaded {downloaded} filings "
        f"(skipped {skipped}) into {out_dir}"
    )
    # write index json for downstream ingestion
    index = []
    for p in Path(out_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".htm", ".html"}:
            index.append({"path": str(p), "size": p.stat().st_size})
    (out_dir / "_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    console.print(f"[green]✓[/green] index: {out_dir / '_index.json'} ({len(index)} files)")


if __name__ == "__main__":
    app()

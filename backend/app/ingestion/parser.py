"""Document parser: PDF / DOCX / HTML / TXT → text + tables + per-page info.

We use pdfplumber for PDF text + tables (layout-aware), python-docx for DOCX,
BeautifulSoup for HTML, and a plain read for TXT. Scanned PDFs (little text)
are routed through ``ocr.py`` automatically.

The output is a ``ParsedDocument`` — a list of pages, each with text + tables
(Markdown). This is a single-purpose function that's easy to unit test.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import ValidationError
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class TableExtract:
    markdown: str
    rows: list[list[str]] = field(default_factory=list)
    page: int = 0


@dataclass
class ParsedPage:
    page: int
    text: str
    tables: list[TableExtract] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    raw_text: str
    tables: list[TableExtract]
    needs_ocr: bool
    source_format: str

    def page_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def parse_file(path: str | Path, *, mime_type: str | None = None) -> ParsedDocument:
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"file not found: {path}")
    ext = p.suffix.lower()
    if ext == ".pdf" or (mime_type == "application/pdf"):
        return _parse_pdf(p)
    if ext == ".docx":
        return _parse_docx(p)
    if ext in (".html", ".htm"):
        return _parse_html(p.read_text(encoding="utf-8", errors="ignore"))
    if ext == ".txt":
        return _parse_text(p.read_text(encoding="utf-8", errors="ignore"))
    if ext == ".md":
        return _parse_text(p.read_text(encoding="utf-8", errors="ignore"))
    raise ValidationError(f"unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# PDF (pdfplumber — layout-aware, table extraction included)
# ---------------------------------------------------------------------------


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    clean = [[" ".join((c or "").split()) for c in row] for row in rows]
    header = clean[0]
    body = clean[1:] if len(clean) > 1 else []
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in body:
        # pad row to header length
        row = (row + [""] * len(header))[: len(header)]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _parse_pdf(p: Path) -> ParsedDocument:
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover
        raise ValidationError(
            "pdfplumber is required to parse PDFs. Install with: "
            "pip install pdfplumber"
        ) from e

    pages: list[ParsedPage] = []
    all_tables: list[TableExtract] = []
    total_text = ""
    with pdfplumber.open(str(p)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables_md: list[TableExtract] = []
            try:
                for t in page.extract_tables() or []:
                    rows = [[(c or "") for c in row] for row in t]
                    md = _table_to_markdown(rows)
                    if md:
                        tables_md.append(TableExtract(markdown=md, rows=rows, page=i))
            except Exception as e:  # pragma: no cover
                log.debug("table extraction failed on page %d: %s", i, e)
            pages.append(ParsedPage(page=i, text=text, tables=tables_md))
            all_tables.extend(tables_md)
            total_text += text + "\n"

    # OCR fallback: if the PDF yielded almost no text per page, mark for OCR.
    n_pages = max(len(pages), 1)
    avg_chars = len(total_text.strip()) / n_pages
    needs_ocr = avg_chars < 50

    return ParsedDocument(
        pages=pages,
        raw_text=total_text,
        tables=all_tables,
        needs_ocr=needs_ocr,
        source_format="pdf",
    )


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def _parse_docx(p: Path) -> ParsedDocument:
    try:
        from docx import Document
    except ImportError as e:  # pragma: no cover
        raise ValidationError("python-docx is required to parse DOCX") from e

    doc = Document(str(p))
    paragraphs = [pg.text for pg in doc.paragraphs if pg.text.strip()]
    tables: list[TableExtract] = []
    for i, t in enumerate(doc.tables):
        rows = [[c.text for c in row.cells] for row in t.rows]
        md = _table_to_markdown(rows)
        if md:
            tables.append(TableExtract(markdown=md, rows=rows, page=1))
    text = "\n\n".join(paragraphs)
    page = ParsedPage(page=1, text=text, tables=tables)
    return ParsedDocument(
        pages=[page],
        raw_text=text,
        tables=tables,
        needs_ocr=False,
        source_format="docx",
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _parse_html(content: str) -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    tables: list[TableExtract] = []
    for t in soup.find_all("table"):
        rows = []
        for tr in t.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            rows.append([c.get_text(" ", strip=True) for c in cells])
        md = _table_to_markdown(rows)
        if md:
            tables.append(TableExtract(markdown=md, rows=rows, page=1))
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    page = ParsedPage(page=1, text=text, tables=tables)
    return ParsedDocument(
        pages=[page],
        raw_text=text,
        tables=tables,
        needs_ocr=False,
        source_format="html",
    )


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def _parse_text(content: str) -> ParsedDocument:
    page = ParsedPage(page=1, text=content)
    return ParsedDocument(
        pages=[page],
        raw_text=content,
        tables=[],
        needs_ocr=False,
        source_format="txt",
    )


# ---------------------------------------------------------------------------
# Bytes helper (used by the API upload path)
# ---------------------------------------------------------------------------


def parse_bytes(data: bytes, *, filename: str, mime_type: str | None = None) -> ParsedDocument:
    """Parse content from in-memory bytes by writing to a temp file."""
    import tempfile

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return parse_file(tmp_path, mime_type=mime_type)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


__all__ = [
    "parse_file",
    "parse_bytes",
    "ParsedDocument",
    "ParsedPage",
    "TableExtract",
]

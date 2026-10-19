"""Ingestion pipeline orchestrator.

Stitching order (per PRD §7 + v2 §1, §2, §5):

  parse → (OCR if needed) → clean → extract tables → extract metadata →
  parent/child chunk → embed children → upsert vectors + structured rows +
  persist Document/Chunk records.

Returns a ``ChunkRecord`` summary the API can serialize.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.db.models import (
    Chunk,
    Document,
    DocumentStatus,
    DocumentType,
    StructuredFinancial,
)
from app.ingestion.chunking import ChunkingResult, chunk_document
from app.ingestion.cleaner import clean_text
from app.ingestion.metadata import DocMetadata, infer_metadata
from app.ingestion.ocr import ocr_pdf
from app.ingestion.parser import ParsedDocument, parse_file
from app.ingestion.tables import extract_financials
from app.providers import get_embeddings, get_vectordb

log = get_logger(__name__)


@dataclass
class ChunkRecord:
    chunk_id: str
    parent_id: str | None
    is_parent: bool
    text: str
    section: str
    page: int


@dataclass
class IngestResult:
    document_id: str
    n_parents: int
    n_children: int
    n_tables: int
    n_financials: int
    needs_ocr: bool
    metadata: DocMetadata
    chunks: list[ChunkRecord]


def ingest_document(
    *,
    file_path: str,
    workspace_id: str,
    db: Session,
    title: str | None = None,
    hint_company: str = "",
    hint_ticker: str = "",
    hint_year: int | None = None,
    hint_quarter: str = "",
    hint_doc_type: DocumentType | None = None,
    section_filter: str = "",
) -> IngestResult:
    """Run the full pipeline for one document. Returns the IngestResult."""
    p = Path(file_path)
    if not p.exists():
        raise ValidationError(f"file not found: {file_path}")

    # 1. Parse (+ OCR fallback if scanned)
    parsed = parse_file(p)
    if parsed.needs_ocr and parsed.source_format == "pdf":
        log.info("document appears scanned; running OCR")
        ocr = ocr_pdf(p)
        if ocr.raw_text.strip():
            parsed = ocr

    # 2. Clean text
    cleaned_pages = [clean_text(pg.text) for pg in parsed.pages]
    cleaned_text = "\n\n".join(cleaned_pages)

    # 3. Extract structured financials from tables
    financials = extract_financials(
        parsed.tables,
        company=hint_company,
        ticker=hint_ticker,
    )

    # 4. Metadata
    meta = infer_metadata(
        parsed,
        filename=p.name,
        hint_company=hint_company,
        hint_ticker=hint_ticker,
        hint_year=hint_year,
        hint_quarter=hint_quarter,
        hint_doc_type=hint_doc_type,
    )
    if section_filter:
        meta.section = section_filter

    # 5. Chunk (parent/child) — attach tables by page
    tables_by_page: dict[int, list[str]] = {}
    for tbl in parsed.tables:
        tables_by_page.setdefault(tbl.page, []).append(tbl.markdown)
    chunk_result: ChunkingResult = chunk_document(
        cleaned_text,
        pages=[(pg.page, txt) for pg, txt in zip(parsed.pages, cleaned_pages)],
        tables_by_page=tables_by_page,
    )

    # 6. Persist Document
    doc = Document(
        workspace_id=workspace_id,
        title=title or p.stem,
        filename=p.name,
        file_path=str(p),
        mime_type=_guess_mime(p),
        doc_type=meta.doc_type,
        company=meta.company,
        ticker=meta.ticker,
        year=meta.year,
        quarter=meta.quarter,
        section=meta.section,
        status=DocumentStatus.active,
        parse_meta={
            "source_format": parsed.source_format,
            "needs_ocr": parsed.needs_ocr,
            "n_pages": len(parsed.pages),
            "n_tables": len(parsed.tables),
        },
    )
    db.add(doc)
    db.flush()  # get doc.id

    # 7. Persist structured financials (dedup by unique constraint)
    _persist_financials(db, financials, workspace_id=workspace_id,
                        document_id=doc.id, meta=meta)

    # 8. Embed child chunks and upsert vectors
    chunk_records = _embed_and_persist(
        db=db,
        doc=doc,
        chunk_result=chunk_result,
        meta=meta,
        workspace_id=workspace_id,
    )

    db.commit()
    log.info("ingested doc=%s parents=%d children=%d financials=%d tables=%d",
             doc.id, sum(1 for c in chunk_records if c.is_parent),
             sum(1 for c in chunk_records if not c.is_parent),
             len(financials), len(parsed.tables))

    return IngestResult(
        document_id=doc.id,
        n_parents=sum(1 for c in chunk_records if c.is_parent),
        n_children=sum(1 for c in chunk_records if not c.is_parent),
        n_tables=len(parsed.tables),
        n_financials=len(financials),
        needs_ocr=parsed.needs_ocr,
        metadata=meta,
        chunks=chunk_records,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _persist_financials(
    db: Session,
    financials,
    *,
    workspace_id: str,
    document_id: str,
    meta: DocMetadata,
) -> None:
    for row in financials:
        existing = db.scalar(
            select(StructuredFinancial).where(
                StructuredFinancial.workspace_id == workspace_id,
                StructuredFinancial.company == (row.company or meta.company),
                StructuredFinancial.ticker == (row.ticker or meta.ticker),
                StructuredFinancial.metric == row.metric,
                StructuredFinancial.fiscal_year == row.fiscal_year,
                StructuredFinancial.quarter == row.quarter,
                StructuredFinancial.unit == row.unit,
            )
        )
        if existing:
            # newest wins
            existing.value = row.value
            existing.document_id = document_id
            existing.page = row.page
            continue
        db.add(
            StructuredFinancial(
                workspace_id=workspace_id,
                document_id=document_id,
                company=row.company or meta.company,
                ticker=row.ticker or meta.ticker,
                metric=row.metric,
                fiscal_year=row.fiscal_year,
                quarter=row.quarter,
                value=row.value,
                unit=row.unit,
                currency=row.currency,
                page=row.page,
                confidence=row.confidence,
            )
        )


def _embed_and_persist(
    *, db: Session, doc: Document, chunk_result: ChunkingResult,
    meta: DocMetadata, workspace_id: str,
) -> list[ChunkRecord]:
    embeddings = get_embeddings()
    vdb = get_vectordb()
    vdb.ensure_collection(getattr(embeddings, "dim", 1536))

    points = []
    out: list[ChunkRecord] = []

    for parent in chunk_result.parents:
        parent_id = uuid.uuid4().hex
        parent_chunk = Chunk(
            id=parent_id,
            document_id=doc.id,
            parent_id=None,
            is_parent=True,
            text=parent.text,
            section=parent.section,
            page=parent.page,
            tables_md=parent.tables_md,
            token_count=max(1, len(parent.text) // 4),
        )
        db.add(parent_chunk)
        out.append(ChunkRecord(chunk_id=parent_id, parent_id=None, is_parent=True,
                               text=parent.text, section=parent.section,
                               page=parent.page))

        # Embed the children
        child_texts = [c.text for c in parent.children]
        if not child_texts:
            continue
        vectors = embeddings.embed(child_texts)
        for child, vector in zip(parent.children, vectors):
            child_id = uuid.uuid4().hex
            point_id = uuid.uuid4().hex
            payload = {
                "workspace_id": workspace_id,
                "document_id": doc.id,
                "chunk_id": child_id,
                "parent_id": parent_id,
                "company": meta.company,
                "ticker": meta.ticker,
                "year": meta.year,
                "quarter": meta.quarter,
                "doc_type": meta.doc_type.value,
                "section": child.section or "",
                "page": child.page,
                "text": child.text,
                "tables_md": "",
                "status": DocumentStatus.active.value,
            }
            points.append({"id": point_id, "vector": vector, "payload": payload})
            child_chunk = Chunk(
                id=child_id,
                document_id=doc.id,
                parent_id=parent_id,
                is_parent=False,
                text=child.text,
                section=child.section,
                page=child.page,
                vector_point_id=point_id,
                token_count=max(1, len(child.text) // 4),
            )
            db.add(child_chunk)
            out.append(ChunkRecord(chunk_id=child_id, parent_id=parent_id,
                                   is_parent=False, text=child.text,
                                   section=child.section, page=child.page))

    # Upsert in batches (Qdrant handles large batches fine; chunk for memory)
    BATCH = 200
    for i in range(0, len(points), BATCH):
        vdb.upsert(points[i: i + BATCH])
    return out


def _guess_mime(p: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".html": "text/html",
        ".htm": "text/html",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }.get(p.suffix.lower(), "application/octet-stream")

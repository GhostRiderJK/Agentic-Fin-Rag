"""Ingestion pipeline orchestrator.

Stitching order (per PRD §7 + v2 §1, §2, §5):

  parse → (OCR if needed) → clean → extract tables → extract metadata →
  parent/child chunk → embed children → vectors + MongoDB records.

Everything lands in MongoDB through the tenant-scoped repositories, and the
document row is written last so a half-ingested filing is never searchable.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.db.enums import DocumentStatus, DocumentType
from app.db.mongodb.base_repository import new_id
from app.db.mongodb.repositories import (
    ChunkRepository,
    DocumentRepository,
    StructuredFinancialRepository,
)
from app.db.mongodb.runtime import run_sync
from app.db.mongodb.tenant_context import TenantContext
from app.ingestion.chunking import ChunkingResult, chunk_document
from app.ingestion.cleaner import clean_text
from app.ingestion.metadata import DocMetadata, infer_metadata
from app.ingestion.ocr import ocr_pdf
from app.ingestion.parser import ParsedDocument as ParsedDocument
from app.ingestion.parser import parse_file
from app.ingestion.tables import extract_financials
from app.providers import get_embeddings, get_vectordb

log = get_logger(__name__)

_documents = DocumentRepository()
_chunks = ChunkRepository()
_financials = StructuredFinancialRepository()


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
    ctx: TenantContext,
    *,
    file_path: str,
    source_name: str | None = None,
    title: str | None = None,
    hint_company: str = "",
    hint_ticker: str = "",
    hint_year: int | None = None,
    hint_quarter: str = "",
    hint_doc_type: DocumentType | None = None,
    section_filter: str = "",
) -> IngestResult:
    """Run the full pipeline for one document. Returns the IngestResult."""
    ws = ctx.workspace_id or ""
    if not ws:
        raise ValidationError("ingestion needs a workspace to file the document in")

    p = Path(file_path)
    if not p.exists():
        raise ValidationError(f"file not found: {file_path}")
    source_hash = _sha256_file(p)
    display_name = Path(source_name).name if source_name else p.name

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
        filename=display_name,
        hint_company=hint_company,
        hint_ticker=hint_ticker,
        hint_year=hint_year,
        hint_quarter=hint_quarter,
        hint_doc_type=hint_doc_type,
    )
    if section_filter:
        meta.section = section_filter

    # Same-name reports from different companies or reporting periods remain
    # independent, while a changed version of one logical report replaces it.
    source_identity = _logical_source_identity(
        display_name, meta.company, meta.ticker, meta.year, meta.quarter,
        meta.doc_type.value,
    )
    document_id = _stable_document_id(
        ctx.tenant_id, ws, source_identity, source_hash
    )

    existing = run_sync(_documents.get_by_id(ctx, document_id))
    if existing and existing.get("source_hash") == source_hash:
        # A published generation is complete because document publication is
        # the final write. Retire any older generations left by an interrupted
        # cleanup, then return the existing indexed counts.
        _retire_older_generations(
            ctx, ws, source_identity, display_name, str(p.resolve()), meta,
            document_id,
        )
        stored_chunks = run_sync(_chunks.list_for_document(ctx, document_id))
        records = [
            ChunkRecord(chunk_id=c["id"], parent_id=c.get("parent_id"),
                        is_parent=bool(c.get("is_parent")), text=c.get("text", ""),
                        section=c.get("section", ""), page=c.get("page", 0))
            for c in stored_chunks
        ]
        parse_meta = existing.get("parse_meta") or {}
        return IngestResult(
            document_id=document_id,
            n_parents=sum(c.is_parent for c in records),
            n_children=sum(not c.is_parent for c in records),
            n_tables=int(parse_meta.get("n_tables", len(parsed.tables))),
            n_financials=run_sync(_financials.count(ctx, {"document_id": document_id})),
            needs_ocr=bool(parse_meta.get("needs_ocr", parsed.needs_ocr)),
            metadata=meta,
            chunks=records,
        )

    # If an earlier attempt failed before publishing the document, remove
    # only its unreachable staging data before retrying this same generation.
    if existing is None:
        partial_chunks = run_sync(_chunks.list_for_document(ctx, document_id))
        # Vectors are upserted only after Mongo chunk records. This also avoids
        # deleting from a Qdrant collection that has not yet been created on a
        # document's first ingestion.
        if partial_chunks:
            get_vectordb().delete_by_document(document_id)
        run_sync(_chunks.delete_for_document(ctx, document_id))
        run_sync(_financials.detach_document(ctx, document_id))

    # 5. Chunk (parent/child) — attach tables by page
    tables_by_page: dict[int, list[str]] = {}
    for tbl in parsed.tables:
        tables_by_page.setdefault(tbl.page, []).append(tbl.markdown)
    # Put table rows into the searchable child text as well as retaining their
    # Markdown on the parent. Previously only paragraph text was embedded, so
    # a table-only metric could not enter either dense or BM25 retrieval.
    searchable_pages = []
    for pg, txt in zip(parsed.pages, cleaned_pages, strict=True):
        page_tables = tables_by_page.get(pg.page, [])
        enriched = txt
        if page_tables:
            enriched += "\n\nExtracted tables:\n" + "\n\n".join(page_tables)
        searchable_pages.append((pg.page, enriched))
    chunk_result: ChunkingResult = chunk_document(
        cleaned_text,
        pages=searchable_pages,
        tables_by_page=tables_by_page,
    )
    printed_pages = {pg.page: pg.printed_page for pg in parsed.pages}

    # 6. Persist. MongoDB has no cross-collection transaction on a standalone
    # server, so the order is what makes a partial failure safe: chunks and
    # vectors first (unreachable until the document exists), then the numbers,
    # then the document itself as the publication marker. Retrieval only ever
    # sees documents that were written all the way through.
    chunk_records = _embed_and_persist(
        ctx=ctx,
        document_id=document_id,
        workspace_id=ws,
        chunk_result=chunk_result,
        meta=meta,
        printed_pages=printed_pages,
    )

    _persist_financials(
        ctx, financials, workspace_id=ws, document_id=document_id, meta=meta
    )

    run_sync(_documents.insert_one(
        ctx,
        {
            "id": document_id,
            "workspace_id": ws,
            "source_identity": source_identity,
            "source_hash": source_hash,
            "source_name": display_name,
            "title": title or Path(display_name).stem,
            "filename": display_name,
            "file_path": str(p),
            "mime_type": _guess_mime(p),
            "doc_type": meta.doc_type.value,
            "company": meta.company,
            "ticker": meta.ticker,
            "year": meta.year,
            "quarter": meta.quarter,
            "section": meta.section,
            "status": DocumentStatus.active.value,
            "parse_meta": {
                "source_format": parsed.source_format,
                "needs_ocr": parsed.needs_ocr,
                "n_pages": len(parsed.pages),
                "n_tables": len(parsed.tables),
            },
        },
        doc_id=document_id,
    ))

    _retire_older_generations(
        ctx, ws, source_identity, display_name, str(p.resolve()), meta,
        document_id,
    )

    log.info("ingested doc=%s parents=%d children=%d financials=%d tables=%d",
             document_id, sum(1 for c in chunk_records if c.is_parent),
             sum(1 for c in chunk_records if not c.is_parent),
             len(financials), len(parsed.tables))

    return IngestResult(
        document_id=document_id,
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


def _dec(value: float) -> Decimal:
    """Money/ratio as an exact decimal.

    The extractor returns floats (it multiplies by a Bn/M/K suffix), so the
    conversion goes through ``str`` to get the shortest repr rather than the
    binary expansion. The repository rejects a raw float at the boundary.
    """
    return Decimal(str(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_document_id(
    tenant_id: str, workspace_id: str, source_identity: str, source_hash: str
) -> str:
    identity = f"finrag:{tenant_id}:{workspace_id}:{source_identity}:{source_hash}"
    return uuid.uuid5(uuid.NAMESPACE_URL, identity).hex


def _logical_source_identity(
    filename: str, company: str, ticker: str, year: int | None,
    quarter: str, doc_type: str,
) -> str:
    """Stable source key that separates same-name reports by report scope."""
    parts = (
        filename.casefold().strip(), company.casefold().strip(),
        ticker.casefold().strip(), str(year or ""), quarter.casefold().strip(),
        doc_type.casefold().strip(),
    )
    return "report:" + hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _retire_older_generations(
    ctx: TenantContext, workspace_id: str, source_identity: str,
    source_filename: str, source_path: str, meta: DocMetadata,
    current_id: str,
) -> None:
    """Keep only the published generation searchable for this logical source.

    Publish the new generation first. Older records are retained as
    ``superseded`` for audit, while their chunks and vectors are removed so
    stale text cannot enter retrieval. Re-running ingestion retries this
    cleanup if a prior cleanup stopped partway through.
    """
    candidates = run_sync(_documents.find_many(ctx, {
        "workspace_id": workspace_id,
        "$or": [
            {"source_identity": source_identity},
            # Older records used a basename identity. Constrain migration by
            # the report dimensions so unrelated same-name files survive.
            {
                "source_identity": source_filename.casefold(),
                "company": meta.company,
                "ticker": meta.ticker,
                "year": meta.year,
                "quarter": meta.quarter,
                "doc_type": meta.doc_type.value,
            },
            # Older local ingestions may have no source_identity field.
            {"file_path": source_path},
        ],
    }))
    vdb = get_vectordb()
    for previous in candidates:
        previous_id = previous.get("id")
        if not previous_id or previous_id == current_id:
            continue
        run_sync(_documents.update_by_id(ctx, previous_id, {
            "status": DocumentStatus.superseded.value,
        }))
        vdb.delete_by_document(previous_id)
        run_sync(_chunks.delete_for_document(ctx, previous_id))
        run_sync(_financials.detach_document(ctx, previous_id))


def _persist_financials(
    ctx: TenantContext,
    financials,
    *,
    workspace_id: str,
    document_id: str,
    meta: DocMetadata,
) -> None:
    """Upsert extracted numbers, newest document wins.

    SQL had a unique constraint to lean on; MongoDB does not, so the same key
    (workspace, company, ticker, metric, year, quarter, unit) is looked up
    before writing. Rows that repeat inside one document collapse here first,
    which is what SQLAlchemy's autoflush used to do.
    """
    dedup: dict[tuple, dict] = {}
    for row in financials:
        company = row.company or meta.company
        ticker = row.ticker or meta.ticker
        key = (workspace_id, company, ticker, row.metric, row.fiscal_year,
               row.quarter, row.unit)
        dedup[key] = {
            "workspace_id": workspace_id,
            "document_id": document_id,
            "company": company,
            "ticker": ticker,
            "metric": row.metric,
            "fiscal_year": row.fiscal_year,
            "quarter": row.quarter,
            "value": _dec(row.value),
            "unit": row.unit,
            "currency": row.currency,
            "page": row.page,
            "confidence": _dec(row.confidence),
        }

    to_insert: list[dict] = []
    for row in dedup.values():
        existing = run_sync(_financials.find_one(ctx, {
            "workspace_id": row["workspace_id"],
            "company": row["company"],
            "ticker": row["ticker"],
            "metric": row["metric"],
            "fiscal_year": row["fiscal_year"],
            "quarter": row["quarter"],
            "unit": row["unit"],
        }))
        if existing is None:
            to_insert.append(row)
            continue
        run_sync(_financials.update_by_id(ctx, existing["id"], {
            "value": row["value"],
            "document_id": row["document_id"],
            "page": row["page"],
        }))
    if to_insert:
        run_sync(_financials.insert_many(ctx, to_insert))


def _embed_and_persist(
    *, ctx: TenantContext, document_id: str, workspace_id: str,
    chunk_result: ChunkingResult, meta: DocMetadata,
    printed_pages: dict[int, int | None],
) -> list[ChunkRecord]:
    embeddings = get_embeddings()
    vdb = get_vectordb()
    vdb.ensure_collection(getattr(embeddings, "dim", 1536))

    points = []
    chunk_docs: list[dict] = []
    out: list[ChunkRecord] = []

    for parent in chunk_result.parents:
        parent_id = new_id()
        chunk_docs.append({
            "id": parent_id,
            "workspace_id": workspace_id,
            "document_id": document_id,
            "parent_id": None,
            "is_parent": True,
            "text": parent.text,
            "section": parent.section,
            "page": parent.page,
            "printed_page": printed_pages.get(parent.page),
            "tables_md": parent.tables_md,
            "token_count": max(1, len(parent.text) // 4),
        })
        out.append(ChunkRecord(chunk_id=parent_id, parent_id=None, is_parent=True,
                               text=parent.text, section=parent.section,
                               page=parent.page))

        # Embed the children
        child_texts = [c.text for c in parent.children]
        if not child_texts:
            continue
        vectors = embeddings.embed(child_texts)
        for child, vector in zip(parent.children, vectors, strict=True):
            child_id = new_id()
            point_id = new_id()
            payload = {
                "workspace_id": workspace_id,
                "document_id": document_id,
                "chunk_id": child_id,
                "parent_id": parent_id,
                "company": meta.company,
                "ticker": meta.ticker,
                "year": meta.year,
                "quarter": meta.quarter,
                "doc_type": meta.doc_type.value,
                "section": child.section or "",
                "page": child.page,
                "printed_page": printed_pages.get(child.page),
                "text": child.text,
                "tables_md": "",
                "status": DocumentStatus.active.value,
            }
            points.append({"id": point_id, "vector": vector, "payload": payload})
            chunk_docs.append({
                "id": child_id,
                "workspace_id": workspace_id,
                "document_id": document_id,
                "parent_id": parent_id,
                "is_parent": False,
                "text": child.text,
                "section": child.section,
                "page": child.page,
                "printed_page": printed_pages.get(child.page),
                "vector_point_id": point_id,
                "token_count": max(1, len(child.text) // 4),
            })
            out.append(ChunkRecord(chunk_id=child_id, parent_id=parent_id,
                                   is_parent=False, text=child.text,
                                   section=child.section, page=child.page))

    run_sync(_chunks.insert_many(ctx, chunk_docs))

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

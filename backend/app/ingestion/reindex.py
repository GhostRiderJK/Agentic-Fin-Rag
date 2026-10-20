"""Re-embed published MongoDB chunks into the isolated Nomic test collection.

This path deliberately avoids normal ingestion and its global vector-store
singleton. It only reads MongoDB and writes the fixed ``chunks_nomic768_test``
Qdrant collection; it never modifies source records or the original collection.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.errors import ProviderError
from app.db.enums import DocumentStatus
from app.db.mongodb.repositories import ChunkRepository, DocumentRepository
from app.db.mongodb.runtime import run_sync
from app.db.mongodb.tenant_context import TenantContext
from app.providers.embeddings import BaseEmbeddings, get_embeddings
from app.providers.vectordb import QdrantVectorDB

TARGET_COLLECTION = "chunks_nomic768_test"
_POINT_NAMESPACE = uuid.UUID("d1ed4f3a-d978-4ac3-9c52-afe1c6bbff89")


@dataclass
class ReindexReport:
    tenant_id: str
    workspace_id: str
    collection: str = TARGET_COLLECTION
    dry_run: bool = False
    documents_seen: int = 0
    chunks_seen: int = 0
    chunks_ready: int = 0
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_point_id(tenant_id: str, workspace_id: str, document_id: str,
                     chunk_id: str) -> str:
    key = f"{tenant_id}:{workspace_id}:{document_id}:{chunk_id}"
    return str(uuid.uuid5(_POINT_NAMESPACE, key))


def reindex_existing_chunks(
    *,
    tenant_id: str,
    workspace_id: str,
    batch_size: int = 64,
    embedder: BaseEmbeddings | None = None,
    vector_db: QdrantVectorDB | None = None,
    documents: DocumentRepository | None = None,
    chunks: ChunkRepository | None = None,
    dry_run: bool = False,
    document_limit: int | None = None,
) -> ReindexReport:
    """Re-embed active, already-published child chunks in a strict scope.

    Upserts are idempotent: existing stored Qdrant IDs are reused, and a
    deterministic UUID5 is used only when the Mongo child lacks that ID.
    Failed batches are reported and can safely be retried by rerunning.
    """
    if not tenant_id or not workspace_id:
        raise ValueError("tenant_id and workspace_id are required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if document_limit is not None and document_limit <= 0:
        raise ValueError("document_limit must be positive when provided")

    ctx = TenantContext(tenant_id=tenant_id, workspace_id=workspace_id)
    doc_repo = documents or DocumentRepository()
    chunk_repo = chunks or ChunkRepository()
    docs = run_sync(doc_repo.find_matching(
        ctx, workspace_id=workspace_id, status=DocumentStatus.active.value
    ))
    docs = sorted(docs, key=lambda doc: str(doc.get("id") or ""))
    report = ReindexReport(
        tenant_id=tenant_id, workspace_id=workspace_id,
        dry_run=dry_run, documents_seen=len(docs),
    )
    if document_limit is not None:
        docs = docs[:document_limit]
    report.documents_seen = len(docs)
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for doc in docs:
        document_id = str(doc.get("id") or "")
        if (not document_id or doc.get("workspace_id") != workspace_id
                or doc.get("tenant_id") not in (None, tenant_id)):
            report.chunks_failed += 1
            report.failures.append({"document_id": document_id, "error": "document identity/scope mismatch"})
            continue
        child_rows = run_sync(chunk_repo.list_for_document(ctx, document_id))
        report.chunks_seen += len(child_rows)
        report.chunks_skipped += sum(1 for row in child_rows if row.get("is_parent"))
        children = [row for row in child_rows if not row.get("is_parent")]
        for chunk in children:
            if (chunk.get("document_id") != document_id
                    or chunk.get("workspace_id") != workspace_id
                    or chunk.get("tenant_id") not in (None, tenant_id)
                    or not chunk.get("id") or not isinstance(chunk.get("text"), str)
                    or not chunk.get("text", "").strip()):
                report.chunks_failed += 1
                report.failures.append({
                    "document_id": document_id,
                    "chunk_id": str(chunk.get("id") or ""),
                    "error": "chunk text or document/workspace association is invalid",
                })
                continue
            pending.append((doc, chunk))

    report.chunks_ready = len(pending)
    if dry_run:
        return report

    embedder = embedder or get_embeddings()
    if embedder.name != "ollama":
        raise ProviderError("reindex requires the configured local Ollama embeddings provider")
    if embedder.dim != 768:
        raise ProviderError(
            f"reindex target requires 768-dimensional vectors; provider reports {embedder.dim}"
        )
    vdb = vector_db or QdrantVectorDB(collection_name=TARGET_COLLECTION)
    if getattr(vdb, "_collection", TARGET_COLLECTION) != TARGET_COLLECTION:
        raise ProviderError("reindex refuses to write outside chunks_nomic768_test")
    vdb.ensure_collection(768)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        try:
            vectors = embedder.embed([chunk["text"] for _, chunk in batch])
            if len(vectors) != len(batch):
                raise ProviderError("embedding response count does not match chunk batch")
            points = []
            for (doc, chunk), vector in zip(batch, vectors, strict=True):
                if len(vector) != 768:
                    raise ProviderError("embedding response contains a non-768 vector")
                document_id = str(doc["id"])
                chunk_id = str(chunk["id"])
                point_id = chunk.get("vector_point_id") or _stable_point_id(
                    tenant_id, workspace_id, document_id, chunk_id
                )
                payload = {
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "parent_id": chunk.get("parent_id"),
                    "company": doc.get("company"),
                    "ticker": doc.get("ticker"),
                    "year": doc.get("year"),
                    "quarter": doc.get("quarter"),
                    "doc_type": doc.get("doc_type"),
                    "section": chunk.get("section") or "",
                    "page": chunk.get("page"),
                    "printed_page": chunk.get("printed_page"),
                    "text": chunk["text"],
                    "tables_md": "",
                    "status": doc.get("status"),
                    "source_name": doc.get("source_name") or doc.get("filename"),
                    "source_hash": doc.get("source_hash"),
                    "title": doc.get("title"),
                }
                points.append({"id": point_id, "vector": vector, "payload": payload})
            vdb.upsert(points)
            report.chunks_processed += len(points)
        except Exception as exc:
            report.chunks_failed += len(batch)
            report.failures.extend({
                "document_id": str(doc.get("id") or ""),
                "chunk_id": str(chunk.get("id") or ""),
                "error": f"{type(exc).__name__}: {exc}",
            } for doc, chunk in batch)

    return report

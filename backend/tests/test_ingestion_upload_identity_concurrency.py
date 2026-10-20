"""Focused regression tests for immutable uploads and scoped source identity."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from starlette.datastructures import UploadFile

from app.api import documents as document_api
from app.db.enums import DocumentType
from app.db.mongodb.tenant_context import TenantContext
from app.ingestion import pipeline
from app.ingestion.chunking import ChildChunk, ChunkingResult, ParentChunk
from app.ingestion.metadata import DocMetadata


def test_same_filename_isolated_by_company_and_reporting_period():
    first = pipeline._logical_source_identity(
        "annual-report.pdf", "Company A", "AAA", 2023, "", "annual_report"
    )
    other_company = pipeline._logical_source_identity(
        "annual-report.pdf", "Company B", "BBB", 2023, "", "annual_report"
    )
    other_year = pipeline._logical_source_identity(
        "annual-report.pdf", "Company A", "AAA", 2024, "", "annual_report"
    )
    same_report_revision = pipeline._logical_source_identity(
        "ANNUAL-REPORT.pdf", "company a", "aaa", 2023, "", "annual_report"
    )

    assert len({first, other_company, other_year}) == 3
    assert first == same_report_revision


def test_uploads_with_same_name_use_distinct_immutable_job_paths(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        document_api,
        "get_settings",
        lambda: SimpleNamespace(storage_path=str(tmp_path), upload_max_size_mb=5),
    )
    monkeypatch.setattr(document_api, "can_write", lambda _user: None)
    queued: list[dict] = []

    def enqueue(**kwargs):
        queued.append(kwargs)
        return SimpleNamespace(id=f"job-{len(queued)}")

    monkeypatch.setattr(document_api, "enqueue_ingest", enqueue)
    ctx = TenantContext(tenant_id="tenant", workspace_id="workspace")
    user = SimpleNamespace(tenant_id="tenant")

    for _ in range(2):
        upload = UploadFile(
            filename="annual-report.pdf", file=BytesIO(b"identical report bytes")
        )
        document_api.upload_document(
            ctx, user, None, upload, None, "", "", None, "", None, True
        )

    paths = [Path(job["file_path"]) for job in queued]
    assert paths[0] != paths[1]
    assert all(path.read_bytes() == b"identical report bytes" for path in paths)
    assert [job["source_name"] for job in queued] == [
        "annual-report.pdf", "annual-report.pdf"
    ]


def test_changed_content_keeps_same_logical_report_identity(tmp_path: Path):
    # Identity intentionally excludes the content hash; the pipeline combines
    # this key with the hash to create a new generation and retire the prior one.
    first = pipeline._logical_source_identity(
        "annual-report.pdf", "PIF", "PIF", 2023, "", "annual_report"
    )
    source = tmp_path / "source.pdf"
    source.write_text("original report", encoding="utf-8")
    content_hash_before = pipeline._sha256_file(source)
    source.write_text("revised report", encoding="utf-8")
    content_hash_after = pipeline._sha256_file(source)
    second = pipeline._logical_source_identity(
        "annual-report.pdf", "PIF", "PIF", 2023, "", "annual_report"
    )
    assert first == second
    assert content_hash_before != content_hash_after


def test_printed_page_is_preserved_on_searchable_chunks_and_vectors(monkeypatch):
    class Chunks:
        def __init__(self):
            self.rows = []

        async def insert_many(self, _ctx, rows):
            self.rows.extend(rows)

    class Embeddings:
        dim = 1

        def embed(self, texts):
            return [[0.1] for _ in texts]

    class VectorDB:
        def __init__(self):
            self.points = []

        def ensure_collection(self, _dim):
            pass

        def upsert(self, points):
            self.points.extend(points)

    chunks = Chunks()
    vectors = VectorDB()
    monkeypatch.setattr(pipeline, "_chunks", chunks)
    monkeypatch.setattr(pipeline, "run_sync", lambda coro: _run(coro))
    monkeypatch.setattr(pipeline, "get_embeddings", Embeddings)
    monkeypatch.setattr(pipeline, "get_vectordb", lambda: vectors)
    result = ChunkingResult(parents=[ParentChunk(
        text="A sufficiently long parent passage for metadata testing.",
        section="Performance", page=9, char_start=0, char_end=55,
        children=[ChildChunk(
            text="A sufficiently long child passage for metadata testing.",
            section="Performance", page=9, char_start=0, char_end=54,
        )],
    )])

    pipeline._embed_and_persist(
        ctx=TenantContext(tenant_id="tenant", workspace_id="workspace"),
        document_id="doc-1", workspace_id="workspace", chunk_result=result,
        meta=DocMetadata("PIF", "PIF", 2024, "", DocumentType.annual_report, ""),
        printed_pages={9: 16},
    )

    child = next(row for row in chunks.rows if not row["is_parent"])
    assert child["page"] == 9
    assert child["printed_page"] == 16
    assert vectors.points[0]["payload"]["page"] == 9
    assert vectors.points[0]["payload"]["printed_page"] == 16


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("test repository coroutine unexpectedly awaited I/O")

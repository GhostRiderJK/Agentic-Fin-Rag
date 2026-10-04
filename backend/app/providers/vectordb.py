"""Vector DB provider abstraction over Qdrant.

Uses Qdrant local mode by default (in-process, no server). Can switch to a
remote Qdrant cluster via QDRANT_MODE=remote.

The schema stores:
  - vector: embedding
  - payload: {
      workspace_id, document_id, chunk_id, parent_id, company, year,
      quarter, doc_type, section, page, text, tables_md, status
    }
"""

from __future__ import annotations

from typing import Any, Protocol

from app.core.config import get_settings
from app.core.errors import ProviderError
from app.core.logging import get_logger

log = get_logger(__name__)


class VectorPoint(dict):
    """A point to upsert: {id, vector, payload}."""


class SearchHit(dict):
    """A search hit: {id, score, payload}."""


class BaseVectorDB(Protocol):
    def ensure_collection(self, dim: int) -> None: ...
    def upsert(self, points: list[VectorPoint]) -> None: ...
    def search(
        self, vector: list[float], *, top_k: int,
        filter_: dict | None = None,
    ) -> list[SearchHit]: ...
    def delete_by_document(self, document_id: str) -> None: ...
    def count(self, filter_: dict | None = None) -> int: ...


# ---------------------------------------------------------------------------
# Qdrant (local or remote)
# ---------------------------------------------------------------------------


class QdrantVectorDB:
    name = "qdrant"

    def __init__(self) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:  # pragma: no cover
            raise ProviderError("qdrant-client not installed") from e
        settings = get_settings()
        self._collection = settings.qdrant_collection
        if settings.qdrant_mode == "remote":
            self._client = QdrantClient(
                url=settings.qdrant_url or None,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            from pathlib import Path

            Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=settings.qdrant_path)
        self._dim: int | None = None

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.http.exceptions import UnexpectedResponse

        self._dim = dim
        try:
            self._client.get_collection(self._collection)
            return
        except (UnexpectedResponse, ValueError):
            pass
        from qdrant_client.http.models import (
            Distance,
            VectorParams,
        )

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info("created qdrant collection %s (dim=%d)", self._collection, dim)

    def upsert(self, points: list[VectorPoint]) -> None:
        from qdrant_client.http.models import PointStruct

        ps = [
            PointStruct(id=str(p["id"]), vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        if not ps:
            return
        self._client.upsert(collection_name=self._collection, points=ps)

    def search(self, vector, *, top_k, filter_=None):
        from qdrant_client.http.models import Filter

        flt = Filter(must=_qdrant_filter(filter_)) if filter_ else None
        try:
            # Newer qdrant-client deprecates .search in favor of query_points;
            # use whichever is available.
            if hasattr(self._client, "query_points"):
                resp = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    limit=top_k,
                    query_filter=flt,
                )
                results = resp.points
            else:  # pragma: no cover
                results = self._client.search(
                    collection_name=self._collection,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=flt,
                )
        except Exception as e:
            raise ProviderError(f"qdrant search error: {e}") from e
        return [
            SearchHit(
                id=str(r.id),
                score=float(r.score),
                payload=dict(r.payload or {}),
            )
            for r in results
        ]

    def delete_by_document(self, document_id: str) -> None:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id",
                                      match=MatchValue(value=document_id))]
            ),
        )

    def count(self, filter_=None) -> int:
        from qdrant_client.http.models import Filter

        flt = Filter(must=_qdrant_filter(filter_)) if filter_ else None
        try:
            r = self._client.count(collection_name=self._collection,
                                    count_filter=flt, exact=True)
            return int(r.count)
        except Exception:
            return 0


def _qdrant_filter(filter_: dict | None) -> list:
    """Convert our simple ``{"key": value, "key": [v1,v2]}`` filter into
    Qdrant ``FieldCondition``s."""
    from qdrant_client.http.models import FieldCondition, MatchAny, MatchValue

    if not filter_:
        return []
    conds = []
    for k, v in filter_.items():
        if isinstance(v, list):
            conds.append(FieldCondition(key=k, match=MatchAny(any=v)))
        else:
            conds.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return conds


# ---------------------------------------------------------------------------
# In-memory (used as a fallback / in tests where Qdrant isn't available)
# ---------------------------------------------------------------------------


class InMemoryVectorDB:
    name = "memory"

    def __init__(self) -> None:
        self._points: list[VectorPoint] = []

    def ensure_collection(self, dim: int) -> None:
        self._dim = dim

    def upsert(self, points: list[VectorPoint]) -> None:
        for p in points:
            self._points.append(p)

    def search(self, vector, *, top_k, filter_=None) -> list[SearchHit]:
        import math

        scored = []
        for p in self._points:
            if filter_ and not _matches(p["payload"], filter_):
                continue
            v = p["vector"]
            dot = sum(a * b for a, b in zip(vector, v))
            na = math.sqrt(sum(x * x for x in vector)) or 1.0
            nb = math.sqrt(sum(x * x for x in v)) or 1.0
            scored.append((p, dot / (na * nb)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            SearchHit(id=str(p["id"]), score=float(s), payload=dict(p["payload"]))
            for p, s in scored[:top_k]
        ]

    def delete_by_document(self, document_id: str) -> None:
        self._points = [
            p for p in self._points
            if p["payload"].get("document_id") != document_id
        ]

    def count(self, filter_=None) -> int:
        if not filter_:
            return len(self._points)
        return sum(1 for p in self._points if _matches(p["payload"], filter_))


def _matches(payload: dict, filter_: dict) -> bool:
    for k, v in filter_.items():
        if isinstance(v, list):
            if payload.get(k) not in v:
                return False
        else:
            if payload.get(k) != v:
                return False
    return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_vdb_singleton: BaseVectorDB | None = None


def get_vectordb() -> BaseVectorDB:
    global _vdb_singleton
    if _vdb_singleton is not None:
        return _vdb_singleton

    settings = get_settings()
    try:
        _vdb_singleton = QdrantVectorDB()
        log.info("vector store ready: qdrant %s (path=%s)",
                 settings.qdrant_mode, settings.qdrant_path)
    except Exception as e:
        # CRITICAL: falling back to in-memory silently would cause queries to
        # return zero hits when ingestion happened in another process. Surface
        # this loudly and let the caller decide.
        log.error(
            "qdrant unavailable (%s). In local mode this usually means another "
            "process (e.g. a running server) holds the storage lock. Either: "
            "(1) ingest via the running server's HTTP API so vectors live in "
            "the same process, or (2) set QDRANT_MODE=remote and run a qdrant "
            "server. Falling back to in-memory (vectors will NOT be shared "
            "across processes).",
            e,
        )
        _vdb_singleton = InMemoryVectorDB()
    return _vdb_singleton


def reset_vectordb() -> None:
    """Test helper."""
    global _vdb_singleton
    _vdb_singleton = None

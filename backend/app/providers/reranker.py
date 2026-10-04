"""Reranker provider abstraction.

Implementations:
  - StubReranker      (lexical-overlap based; deterministic)
  - CohereReranker    (when COHERE_API_KEY set)
  - CrossEncoderReranker (optional, lazy)
"""

from __future__ import annotations

import re
from typing import Protocol

from app.core.config import get_settings
from app.core.errors import ProviderError
from app.core.logging import get_logger

log = get_logger(__name__)


class BaseReranker(Protocol):
    name: str

    def rerank(self, query: str, documents: list[str],
               top_k: int | None = None) -> list[tuple[int, float]]:
        """Return list of (doc_index, score) sorted desc, top_k entries."""


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------


class StubReranker:
    """Jaccard-overlap-based lexical reranker.

    Deterministic, no network. A reasonable proxy that surfaces documents
    sharing the most distinctive query terms.
    """

    name = "stub"

    def rerank(self, query: str, documents: list[str], top_k=None):
        qtoks = _tok(query)
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            dtoks = _tok(doc)
            if not qtoks or not dtoks:
                scored.append((i, 0.0))
                continue
            inter = len(qtoks & dtoks)
            union = len(qtoks | dtoks) or 1
            scored.append((i, inter / union))
        scored.sort(key=lambda t: t[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


class CohereReranker:
    name = "cohere"

    def __init__(self) -> None:
        try:
            import cohere
        except ImportError as e:  # pragma: no cover
            raise ProviderError("cohere package not installed") from e
        settings = get_settings()
        if not settings.cohere_api_key:
            raise ProviderError("COHERE_API_KEY is not set")
        self._client = cohere.Client(settings.cohere_api_key)

    def rerank(self, query: str, documents: list[str], top_k=None):
        from app.core.errors import ProviderError

        if not documents:
            return []
        try:
            r = self._client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_k or len(documents),
            )
        except Exception as e:
            raise ProviderError(f"Cohere rerank error: {e}") from e
        return [(res.index, res.relevance_score) for res in r.results]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_reranker_singleton: BaseReranker | None = None


def get_reranker() -> BaseReranker:
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton

    settings = get_settings()
    provider = settings.reranker_provider
    if provider == "cohere":
        if not settings.cohere_api_key:
            log.warning("RERANKER_PROVIDER=cohere but key unset; using stub")
            _reranker_singleton = StubReranker()
        else:
            _reranker_singleton = CohereReranker()
    elif provider == "cross_encoder":
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401

            _reranker_singleton = _CrossEncoderReranker()
        except Exception as e:  # pragma: no cover
            log.warning(f"cross-encoder unavailable ({e}); using stub")
            _reranker_singleton = StubReranker()
    else:
        _reranker_singleton = StubReranker()
    return _reranker_singleton


class _CrossEncoderReranker:  # pragma: no cover - optional path
    name = "cross_encoder"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, documents, top_k=None):
        pairs = [(query, d) for d in documents]
        scores = self._m.predict(pairs)
        ranked = sorted(enumerate(float(s) for s in scores),
                        key=lambda t: t[1], reverse=True)
        return ranked[: (top_k or len(ranked))]


def reset_reranker() -> None:
    """Test helper."""
    global _reranker_singleton
    _reranker_singleton = None

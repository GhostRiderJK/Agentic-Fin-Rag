"""Embeddings provider abstraction.

Implementations:
  - OpenAIEmbeddings   (default when key set)
  - StubEmbeddings     (deterministic hash-based vectors, dim configurable)
  - SentenceTransformerEmbeddings (optional, local model — lazy import)

Stub vectors are deterministic per-token hash projections so that:
  * similar texts produce vectors with high cosine similarity,
  * tests are reproducible without any model download or API spend.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.core.config import get_settings
from app.core.errors import ProviderError
from app.core.logging import get_logger

log = get_logger(__name__)


class BaseEmbeddings(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class StubEmbeddings:
    """Deterministic hash-projection embeddings.

    Each token is hashed to seed positions; a vector is the averaged
    contribution across all tokens. Two texts sharing many tokens will land
    close together in cosine distance — enough for retrieval to work in
    tests.
    """

    name = "stub"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            out.append(self._embed_one(text))
        return out

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokenize(text)
        if not toks:
            # deterministic non-zero vector for empty input
            toks = ["<empty>"]
        for tok in toks:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            for i in range(0, min(8, self.dim)):
                idx = (h >> (i * 4)) % self.dim
                sign = 1.0 if ((h >> (i + 8)) & 1) == 0 else -1.0
                vec[idx] += sign
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbeddings:
    name = "openai"

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ProviderError("openai package not installed") from e
        settings = get_settings()
        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self.dim = settings.embeddings_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        from app.core.errors import ProviderError

        # OpenAI recommends replacing newlines for embedding quality
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        try:
            resp = self._client.embeddings.create(model=self._model, input=cleaned)
        except Exception as e:
            raise ProviderError(f"OpenAI embeddings error: {e}") from e
        return [list(d.embedding) for d in resp.data]


# ---------------------------------------------------------------------------
# Sentence-transformers (lazy)
# ---------------------------------------------------------------------------


class SentenceTransformerEmbeddings:
    name = "sentence_transformers"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ProviderError("sentence-transformers not installed") from e
        settings = get_settings()
        self._model = SentenceTransformer(settings.sentence_transformer_model)
        self.dim = settings.embeddings_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in np.asarray(vecs)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_emb_singleton: BaseEmbeddings | None = None


def get_embeddings() -> BaseEmbeddings:
    global _emb_singleton
    if _emb_singleton is not None:
        return _emb_singleton

    settings = get_settings()
    provider = settings.embeddings_provider
    if provider == "openai":
        if not settings.openai_api_key:
            log.warning("EMBEDDINGS_PROVIDER=openai but key unset; using stub")
            _emb_singleton = StubEmbeddings(dim=settings.embeddings_dim)
        else:
            _emb_singleton = OpenAIEmbeddings()
    elif provider == "sentence_transformers":
        try:
            _emb_singleton = SentenceTransformerEmbeddings()
        except ProviderError as e:
            log.warning(f"{e}; using stub")
            _emb_singleton = StubEmbeddings(dim=settings.embeddings_dim)
    else:
        _emb_singleton = StubEmbeddings(dim=settings.embeddings_dim)
    return _emb_singleton


def reset_embeddings() -> None:
    """Test helper."""
    global _emb_singleton
    _emb_singleton = None

"""Embeddings provider abstraction.

Implementations:
  - OllamaEmbeddings   (local /api/embed; the project's embedding source)
  - OpenAIEmbeddings   (used when a hosted key is configured)
  - StubEmbeddings     (deterministic hash-based vectors, dim configurable)

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
        api_key = settings.openai_api_key.get_secret_value()
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self._model = settings.openai_embedding_model
        self.dim = settings.embeddings_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        from app.core.errors import ProviderError

        if not texts:
            return []
        # OpenAI recommends replacing newlines for embedding quality
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        try:
            # The v3 API supports caller-selected dimensions. Other models do
            # not, so their returned width is checked against the configured
            # collection width below.
            params = {"dimensions": self.dim} if self._model.startswith(
                "text-embedding-3-"
            ) else {}
            resp = self._client.embeddings.create(
                model=self._model, input=cleaned, **params
            )
        except Exception as e:
            raise ProviderError(f"OpenAI embeddings error: {e}") from e
        vectors = [list(d.embedding) for d in resp.data]
        if len(vectors) != len(texts):
            raise ProviderError(
                f"OpenAI returned {len(vectors)} vectors for {len(texts)} inputs",
                provider="openai",
                model=self._model,
            )
        for index, vector in enumerate(vectors):
            if len(vector) != self.dim:
                raise ProviderError(
                    f"EMBEDDING_DIM={self.dim} but model {self._model!r} "
                    f"returned {len(vector)} dimensions (input {index})",
                    provider="openai",
                    model=self._model,
                )
        try:
            return [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as e:
            raise ProviderError(
                f"OpenAI embeddings returned a non-numeric vector: {e}",
                provider="openai",
                model=self._model,
            ) from e


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaEmbeddings:
    """Local Ollama embeddings through the native ``/api/embed`` route.

    ``EMBEDDING_MODEL`` may carry the ``ollama/`` role prefix used by
    FALLBACK_CHAIN; Ollama itself wants the bare tag.
    """

    name = "ollama"

    def __init__(self, *, client=None) -> None:
        import httpx

        settings = get_settings()
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.embedding_model.split("/", 1)[-1]
        self.dim = settings.embeddings_dim
        self._http = client or httpx.Client(
            timeout=settings.embedding_request_timeout_seconds
        )
        self._healthcheck()

    def _healthcheck(self) -> None:
        from app.core.errors import ProviderError

        try:
            r = self._http.get(f"{self._base}/api/tags")
            r.raise_for_status()
            available = {t.get("name") or "" for t in r.json().get("models", [])}
        except Exception as e:
            raise ProviderError(
                f"Ollama not reachable at {self._base}: {e}. Start it with "
                f"'ollama serve' and pull the model with "
                f"'ollama pull {self._model}'.",
                provider="ollama",
            ) from e
        # Ollama reports tags ("nomic-embed-text:latest") for unqualified asks.
        if not any(
            name == self._model or name.startswith(f"{self._model}:")
            for name in available
        ):
            raise ProviderError(
                f"Ollama model {self._model!r} is not installed. "
                f"Run 'ollama pull {self._model}'. "
                f"Available: {sorted(n for n in available if n)}",
                provider="ollama",
                model=self._model,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        from app.core.errors import ProviderError

        if not texts:
            return []
        payload = {"model": self._model, "input": [t or " " for t in texts]}
        try:
            r = self._http.post(f"{self._base}/api/embed", json=payload)
            r.raise_for_status()
            vectors = r.json().get("embeddings") or []
        except Exception as e:
            raise ProviderError(f"Ollama embeddings error: {e}", provider="ollama",
                                model=self._model) from e
        if len(vectors) != len(texts):
            raise ProviderError(
                f"Ollama returned {len(vectors)} vectors for {len(texts)} inputs",
                provider="ollama",
                model=self._model,
            )
        for i, vec in enumerate(vectors):
            if len(vec) != self.dim:
                raise ProviderError(
                    f"EMBEDDING_DIM={self.dim} but model {self._model!r} "
                    f"returned {len(vec)} dimensions (input {i}). "
                    f"Set EMBEDDING_DIM={len(vec)} and rebuild the vector index.",
                    provider="ollama",
                    model=self._model,
                )
        return [[float(x) for x in vec] for vec in vectors]


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
        if not settings.openai_api_key.get_secret_value():
            raise ProviderError(
                "OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai",
                provider="openai",
            )
        _emb_singleton = OpenAIEmbeddings()
    elif provider == "ollama":
        # Fail loudly: silently substituting stub vectors would embed tenant
        # documents in a space that no real model can query.
        _emb_singleton = OllamaEmbeddings()
    else:
        _emb_singleton = StubEmbeddings(dim=settings.embeddings_dim)
    return _emb_singleton


def reset_embeddings() -> None:
    """Test helper."""
    global _emb_singleton
    _emb_singleton = None

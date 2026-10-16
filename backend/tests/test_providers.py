"""Embedding/LLM provider wiring tests.

The Ollama provider is exercised through an httpx MockTransport so the HTTP
contract is checked without a running daemon. Nothing here produces fake
vectors for a result — ``StubEmbeddings`` remains the only such path.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import reset_settings_cache
from app.core.errors import ProviderError
from app.providers.embeddings import (
    OllamaEmbeddings,
    get_embeddings,
    reset_embeddings,
)


@pytest.fixture
def ollama_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setenv("EMBEDDING_MODEL", "ollama/nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "ollama")
    reset_settings_cache()
    reset_embeddings()
    yield
    reset_settings_cache()
    reset_embeddings()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _tags(models):
    return httpx.Response(200, json={"models": [{"name": m} for m in models]})


def test_ollama_embeddings_call_the_native_embed_route(ollama_env):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags(["nomic-embed-text:latest"])
        assert request.url.path == "/api/embed"
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"embeddings": [[1.0] * 768, [0.5] * 768]}
        )

    emb = OllamaEmbeddings(client=_client(handler))
    out = emb.embed(["revenue grew", "margin fell"])

    assert emb.name == "ollama"
    assert emb.dim == 768
    assert [len(v) for v in out] == [768, 768]
    assert out[1][0] == 0.5
    # The role prefix is Ollama's own namespace, not part of the model tag.
    assert seen == [{"model": "nomic-embed-text", "input": ["revenue grew", "margin fell"]}]


def test_ollama_embeddings_reject_a_dimension_mismatch(ollama_env):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags(["nomic-embed-text:latest"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    emb = OllamaEmbeddings(client=_client(handler))
    with pytest.raises(ProviderError, match="EMBEDDING_DIM=768"):
        emb.embed(["text"])


def test_ollama_embeddings_reject_a_short_answer(ollama_env):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return _tags(["nomic-embed-text:latest"])
        return httpx.Response(200, json={"embeddings": [[0.0] * 768]})

    emb = OllamaEmbeddings(client=_client(handler))
    with pytest.raises(ProviderError, match="1 vectors for 2 inputs"):
        emb.embed(["a", "b"])


def test_a_missing_local_model_is_an_error_not_a_silent_downgrade(ollama_env):
    def handler(request: httpx.Request) -> httpx.Response:
        return _tags(["llama3.1:latest"])

    with pytest.raises(ProviderError, match="not installed"):
        OllamaEmbeddings(client=_client(handler))


def test_unreachable_ollama_names_the_commands_to_run(ollama_env):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderError, match="ollama serve"):
        OllamaEmbeddings(client=_client(handler))


def test_ollama_provider_does_not_fall_back_to_stub(ollama_env):
    """Embedding documents with stub vectors would silently poison the index."""
    with pytest.raises(ProviderError, match="Ollama not reachable"):
        get_embeddings()


def test_stub_is_still_the_default_and_needs_no_daemon(ollama_env, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "stub")
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    reset_settings_cache()
    reset_embeddings()
    emb = get_embeddings()
    assert emb.name == "stub"
    assert len(emb.embed(["anything"])[0]) == 256

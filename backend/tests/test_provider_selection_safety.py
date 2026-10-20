"""Provider selection must not disguise a configured real provider as a stub."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.errors import ProviderError
from app.providers import embeddings, reranker


@pytest.fixture(autouse=True)
def reset_provider_singletons():
    embeddings.reset_embeddings()
    reranker.reset_reranker()
    yield
    embeddings.reset_embeddings()
    reranker.reset_reranker()


def test_openai_embedding_selection_fails_without_key(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(
            embeddings_provider="openai",
            openai_api_key=SimpleNamespace(get_secret_value=lambda: ""),
            embeddings_dim=8,
        ),
    )

    with pytest.raises(ProviderError, match="OPENAI_API_KEY is required"):
        embeddings.get_embeddings()


def test_openai_embedding_provider_is_selected_when_key_is_configured(monkeypatch):
    class FakeOpenAIEmbeddings:
        name = "openai"

    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(
            embeddings_provider="openai",
            openai_api_key=SimpleNamespace(get_secret_value=lambda: "configured"),
            embeddings_dim=8,
        ),
    )
    monkeypatch.setattr(embeddings, "OpenAIEmbeddings", FakeOpenAIEmbeddings)

    assert embeddings.get_embeddings().name == "openai"


def test_openai_v3_requests_configured_dimension(monkeypatch):
    calls = []

    class EmbeddingEndpoint:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.25] * 8)])

    instance = object.__new__(embeddings.OpenAIEmbeddings)
    instance._client = SimpleNamespace(embeddings=EmbeddingEndpoint())
    instance._model = "text-embedding-3-small"
    instance.dim = 8

    assert instance.embed(["financial facts"])[0] == [0.25] * 8
    assert calls == [{
        "model": "text-embedding-3-small",
        "input": ["financial facts"],
        "dimensions": 8,
    }]


@pytest.mark.parametrize(
    ("model", "returned_width"),
    [("text-embedding-ada-002", 8), ("text-embedding-3-small", 7)],
)
def test_openai_embedding_validates_vector_width(model, returned_width):
    class EmbeddingEndpoint:
        def create(self, **kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1] * returned_width)]
            )

    instance = object.__new__(embeddings.OpenAIEmbeddings)
    instance._client = SimpleNamespace(embeddings=EmbeddingEndpoint())
    instance._model = model
    instance.dim = 8

    if returned_width == 8:
        assert len(instance.embed(["text"])[0]) == 8
    else:
        with pytest.raises(ProviderError, match="returned 7 dimensions"):
            instance.embed(["text"])


def test_openai_embedding_validates_response_count():
    class EmbeddingEndpoint:
        def create(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 8)])

    instance = object.__new__(embeddings.OpenAIEmbeddings)
    instance._client = SimpleNamespace(embeddings=EmbeddingEndpoint())
    instance._model = "text-embedding-3-small"
    instance.dim = 8

    with pytest.raises(ProviderError, match="1 vectors for 2 inputs"):
        instance.embed(["one", "two"])


def test_openai_embedding_provider_errors_are_not_swallowed():
    class EmbeddingEndpoint:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    instance = object.__new__(embeddings.OpenAIEmbeddings)
    instance._client = SimpleNamespace(embeddings=EmbeddingEndpoint())
    instance._model = "text-embedding-3-small"
    instance.dim = 8

    with pytest.raises(ProviderError, match="OpenAI embeddings error"):
        instance.embed(["text"])


def test_cohere_selection_fails_without_key(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "get_settings",
        lambda: SimpleNamespace(reranker_provider="cohere", cohere_api_key=""),
    )

    with pytest.raises(ProviderError, match="COHERE_API_KEY is required"):
        reranker.get_reranker()


def test_cross_encoder_failure_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "get_settings",
        lambda: SimpleNamespace(reranker_provider="cross_encoder"),
    )

    def unavailable():
        raise RuntimeError("dependency unavailable")

    monkeypatch.setattr(reranker, "_CrossEncoderReranker", unavailable)
    with pytest.raises(ProviderError, match="Cross-encoder reranker is unavailable"):
        reranker.get_reranker()


def test_explicit_stub_providers_remain_available(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(embeddings_provider="stub", embeddings_dim=8),
    )
    monkeypatch.setattr(
        reranker,
        "get_settings",
        lambda: SimpleNamespace(reranker_provider="stub"),
    )

    assert embeddings.get_embeddings().name == "stub"
    assert reranker.get_reranker().name == "stub"


def test_cohere_provider_is_selected_when_explicit_and_configured(monkeypatch):
    class FakeCohereReranker:
        name = "cohere"

    monkeypatch.setattr(
        reranker,
        "get_settings",
        lambda: SimpleNamespace(reranker_provider="cohere", cohere_api_key="configured"),
    )
    monkeypatch.setattr(reranker, "CohereReranker", FakeCohereReranker)

    assert reranker.get_reranker().name == "cohere"

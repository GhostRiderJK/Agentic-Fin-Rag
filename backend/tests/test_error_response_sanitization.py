"""Provider error text reaches the caller, so credential shapes are scrubbed.

A gateway that echoes the request it received can put the caller's API key into
an error body, and the app quotes provider errors verbatim to whoever sees them.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import ProviderError, ProviderRateLimitError, install_exception_handlers


def _app_that_raises(exc: Exception) -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    def _boom():
        raise exc

    return TestClient(app)


def test_key_shaped_tokens_never_reach_the_response():
    leaked = (
        "OpenAI error: Error code: 401 - {'error': {'message': "
        "\"No active credentials for provider: ollama. "
        "Authorization: Bearer sk-proj-abcdefghijklmnopqrst\"}}"
    )
    client = _app_that_raises(ProviderError(leaked))

    body = client.get("/boom").json()

    assert "sk-proj-abcdefghijklmnopqrst" not in body["message"]
    assert "[redacted]" in body["message"]
    # The diagnosis itself stays readable: only the secret is removed.
    assert "No active credentials for provider: ollama" in body["message"]


def test_non_provider_messages_are_left_alone():
    client = _app_that_raises(ProviderError("EMBEDDING_DIM=768 does not match the store"))
    assert client.get("/boom").json()["message"] == (
        "EMBEDDING_DIM=768 does not match the store")


def test_rate_limit_status_is_preserved():
    client = _app_that_raises(ProviderRateLimitError("429 Too Many Requests"))
    response = client.get("/boom")
    assert response.status_code == 429
    assert response.json()["error"] == "provider_rate_limit"

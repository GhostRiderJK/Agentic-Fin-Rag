"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--allow-mongo-skip",
        action="store_true",
        default=False,
        help=(
            "Skip instead of fail when the real test MongoDB at MONGODB_TEST_URL "
            "is unreachable. Equivalent to REQUIRE_MONGO=0."
        ),
    )


def pytest_configure(config):
    # Honour the CLI opt-out before any fixture reads the environment.
    if config.getoption("--allow-mongo-skip", default=False):
        os.environ["REQUIRE_MONGO"] = "0"


@pytest.fixture(scope="session", autouse=True)
def _isolated_env(tmp_path_factory):
    """Run all tests against a temp data dir + in-memory vector store +
    stub LLM so they never touch real keys or persistent data."""
    tmp = tmp_path_factory.mktemp("ira")
    os.environ["APP_ENV"] = "test"
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["EMBEDDINGS_PROVIDER"] = "stub"
    os.environ["RERANKER_PROVIDER"] = "stub"
    os.environ["QDRANT_MODE"] = "local"
    os.environ["QDRANT_PATH"] = str(tmp / "qdrant")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp / 'test.db'}"
    os.environ["STORAGE_PATH"] = str(tmp / "storage")
    os.environ["JOBS_MODE"] = "inproc"
    os.environ["OPENAI_API_KEY"] = ""  # force stub providers in tests

    # Provide test defaults for the gateway config (unused when LLM_PROVIDER=stub)
    os.environ.setdefault("LLM_API_KEY", "test-key")
    os.environ.setdefault("LLM_BASE_URL", "http://test")
    os.environ.setdefault("CHAT_MODEL", "nvidia/nvidia/nemotron-3-super-120b-a12b")
    os.environ.setdefault("REASONING_MODEL", "nvidia/nvidia/nemotron-3-super-120b-a12b")
    os.environ.setdefault(
        "VISION_MODEL", "nvidia/meta/llama-3.2-11b-vision-instruct"
    )
    os.environ.setdefault(
        "FALLBACK_CHAIN",
        "nvidia/nvidia/nemotron-3-super-120b-a12b,"
        "nvidia/poolside/laguna-xs-2.1,"
        "nvidia/stepfun-ai/step-5-preview,"
        "ollama/llama3.1",
    )
    os.environ.setdefault("ALLOWED_PROVIDERS", "nvidia,ollama")
    # Clear caches so new env is picked up
    from app.core.config import reset_settings_cache
    from app.db.base import reset_engine
    from app.providers.llm import reset_llm
    from app.providers.embeddings import reset_embeddings
    from app.providers.reranker import reset_reranker
    from app.providers.vectordb import reset_vectordb

    reset_settings_cache()
    reset_engine()
    reset_llm()
    reset_embeddings()
    reset_reranker()
    reset_vectordb()
    yield
    # Close the file-backed local Qdrant client here rather than letting its
    # destructor run during interpreter shutdown, when the import system is
    # already torn down and the close prints an unrelated ImportError.
    from app.providers import vectordb as _vdb

    client = getattr(_vdb._vdb_singleton, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()
    _vdb.reset_vectordb()

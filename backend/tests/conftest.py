"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


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

    # Provide test defaults for OmniRoute config (won't be used when LLM_PROVIDER=stub)
    os.environ.setdefault("OMNIROUTE_API_KEY", "test-key")
    os.environ.setdefault("OMNIROUTE_BASE_URL", "http://test")
    os.environ.setdefault("CHAT_MODEL", "nvidia/nvidia/nemotron-3-super-120b-a12b")
    os.environ.setdefault("REASONING_MODEL", "nvidia/nvidia/nemotron-3-ultra-550b-a55b")
    os.environ.setdefault("FALLBACK_CHAIN", "nvidia/nvidia/nemotron-3-super-120b-a12b,ollama/llama3.1")
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

"""Provider abstraction layer.

Every external capability (LLM chat, embeddings, reranker, vector store) is
defined as a Protocol/base class with multiple implementations selected via
``Settings``. This makes "OpenAI now, swap later" trivial and lets tests run
with deterministic stub providers (no API keys, no spend).
"""

from app.providers.llm import BaseLLM, get_llm
from app.providers.embeddings import BaseEmbeddings, get_embeddings
from app.providers.reranker import BaseReranker, get_reranker
from app.providers.vectordb import BaseVectorDB, get_vectordb

__all__ = [
    "BaseLLM",
    "get_llm",
    "BaseEmbeddings",
    "get_embeddings",
    "BaseReranker",
    "get_reranker",
    "BaseVectorDB",
    "get_vectordb",
]

"""Application configuration, env-driven via pydantic-settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Search for .env in CWD, then the backend/ dir, then the repo root, so the
# app works whether launched from backend/ or the repo root.
_HERE = Path(__file__).resolve().parent                 # .../backend/app/core
_REPO = _HERE.parents[2]                                  # repo root
_ENV_FILES = [
    str(Path.cwd() / ".env"),                             # current working dir
    str(_HERE.parents[1] / ".env"),                       # backend/.env
    str(_REPO / ".env"),                                  # repo root .env
]


def _parse_env_file(path: str) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE per line, supports quotes + # comments."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        # strip inline trailing comment if not quoted
        if v and v[0] in "\"'" and v[-1] == v[0]:
            v = v[1:-1]
        else:
            v = v.split(" #", 0)[0].strip()
        out[k] = v
    return out


def _is_valid_field(k: str) -> bool:
    """True if `k` matches a Settings field name (case-insensitive)."""
    if not k or k.startswith("_"):
        return False
    fields = Settings.model_fields  # type: ignore[attr-defined]
    return k.lower() in {f.lower() for f in fields}


class Settings(BaseSettings):
    # We handle .env file parsing manually in get_settings() to ensure
    # .env file values take priority over OS environment variables.
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Investment Research Assistant"
    app_env: Literal["dev", "test", "prod"] = "dev"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_log_level: str = "info"

    # MongoDB (primary relational store for FinRAG 2.0)
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "finrag"
    mongodb_server_selection_timeout_ms: int = 5000

    # Legacy SQLite URL — retained only for the one-time import script
    database_url: str = "sqlite:///./data/app.db"

    # Vector DB
    qdrant_mode: Literal["local", "remote"] = "local"
    qdrant_path: str = "./data/qdrant"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "chunks"

    # Storage
    storage_path: str = "./data/storage"
    upload_max_size_mb: int = 100

    # LLM Providers Configuration
    llm_provider: Literal["openai", "anthropic", "ollama", "stub", "omnirroute"] = "stub"
    
    # OmniRoute Configuration
    omnirroute_base_url: str = "http://localhost:20128/v1"
    omnirroute_api_key: str = ""  # Will be set via environment variable
    
    # Model Configuration (pinned IDs as required)
    chat_model: str = "nvidia/nvidia/nemotron-3-super-120b-a12b"
    reasoning_model: str = "nvidia/nvidia/nemotron-3-ultra-550b-a55b"
    vision_model: str = ""  # To be configured
    embedding_model: str = ""  # To be configured (may use local if no Nemotron embedding)
    rerank_model: str = ""  # To be configured
    
    # Fallback chain (explicit pinned IDs, comma-separated)
    fallback_chain: str = "nvidia/nvidia/nemotron-3-super-120b-a12b,ollama/llama3.1"

    # Provider allow-list (security requirement, comma-separated)
    allowed_providers: str = "nvidia,ollama"

    @property
    def fallback_chain_list(self) -> List[str]:
        """Parse fallback_chain string into list."""
        return [m.strip() for m in self.fallback_chain.split(",") if m.strip()]

    @property
    def allowed_providers_list(self) -> List[str]:
        """Parse allowed_providers string into list."""
        return [p.strip() for p in self.allowed_providers.split(",") if p.strip()]

    # Embeddings
    embeddings_provider: Literal["openai", "stub", "sentence_transformers"] = "stub"
    embeddings_dim: int = 1536
    sentence_transformer_model: str = "BAAI/bge-small-en-v1.5"

    # Reranker
    reranker_provider: Literal["stub", "cohere", "cross_encoder"] = "stub"
    reranker_top_k: int = 5
    cohere_api_key: str = ""

    # Retrieval
    retrieval_top_k: int = 10
    hybrid_dense_weight: float = 0.6
    hybrid_bm25_weight: float = 0.4

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_alg: str = "HS256"
    jwt_ttl_minutes: int = 720
    api_key_header: str = "X-API-Key"

    # Jobs
    jobs_mode: Literal["inproc", "celery"] = "inproc"
    redis_url: str = ""

    # CORS
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @field_validator("cors_origins")
    @classmethod
    def _strip_cors(cls, v: str) -> str:
        return ",".join(part.strip() for part in v.split(",") if part.strip())

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _check_omnirroute_config(self) -> "Settings":
        """Validate OmniRoute configuration when selected as LLM provider."""
        if self.llm_provider == "omnirroute":
            if not self.omnirroute_api_key:
                raise ValueError("OMNIRROUTE_API_KEY must be set when LLM_PROVIDER=omnirroute")
            if not self.chat_model:
                raise ValueError("CHAT_MODEL must be set when using OmniRoute")
            if not self.reasoning_model:
                raise ValueError("REASONING_MODEL must be set when using OmniRoute")
            # Check that models are in allowed providers
            for model in [self.chat_model, self.reasoning_model]:
                if model and not any(model.startswith(provider + "/") for provider in self.allowed_providers_list):
                    raise ValueError(f"Model {model} uses a provider not in ALLOWED_PROVIDERS: {self.allowed_providers_list}")
        return self

    def ensure_dirs(self) -> None:
        db_dir = self._database_dir()
        for p in (self.storage_path, self.qdrant_path, db_dir):
            if p:
                Path(p).mkdir(parents=True, exist_ok=True)

    def _database_dir(self) -> str:
        """Return the directory holding the SQLite file, or '' for non-file URLs."""
        url = self.database_url
        # SQLite URLs look like: sqlite:///./data/app.db or sqlite:////abs/path.db
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            # Handle 4-slash absolute paths on Unix
            if path.startswith("/"):
                path = path[1:]
            return str(Path(path).resolve().parent)
        return ""
    

@lru_cache
def get_settings() -> Settings:
    # In test mode, never load .env — tests configure the environment
    # explicitly via os.environ in conftest.py.
    if os.environ.get("APP_ENV") == "test":
        # Provide minimal defaults for test mode
        os.environ.setdefault("LLM_PROVIDER", "stub")
        os.environ.setdefault("OMNIROUTE_API_KEY", "test-key")
        os.environ.setdefault("OMNIROUTE_BASE_URL", "http://test")
        os.environ.setdefault("CHAT_MODEL", "nvidia/nvidia/nemotron-3-super-120b-a12b")
        os.environ.setdefault("REASONING_MODEL", "nvidia/nvidia/nemotron-3-ultra-550b-a55b")
        os.environ.setdefault("FALLBACK_CHAIN", "nvidia/nvidia/nemotron-3-super-120b-a12b,ollama/llama3.1")
        os.environ.setdefault("ALLOWED_PROVIDERS", "nvidia,ollama")
        return Settings()  # type: ignore[call-arg]

    # Parse .env files first to get values
    env_values: dict[str, str] = {}
    for path in reversed(_ENV_FILES):  # apply root .env last so it wins
        if Path(path).exists():
            try:
                env_values.update(_parse_env_file(path))
            except Exception:
                pass

    # Push .env values into os.environ so pydantic can read them
    # This ensures .env values take precedence over existing env vars
    for k, v in env_values.items():
        if _is_valid_field(k):
            os.environ[k] = v

    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Test helper: clear the cached Settings (after monkeypatching env)."""
    get_settings.cache_clear()

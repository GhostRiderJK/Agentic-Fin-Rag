"""Application configuration, env-driven via pydantic-settings.

Precedence, highest first:

1. keyword arguments passed to ``Settings(...)``
2. real process environment variables
3. ``.env`` files (see ``_ENV_FILES``; a later file overrides an earlier one)
4. the defaults declared below

Real environment variables win over ``.env`` on purpose: that is how a
container orchestrator or CI injects production secrets. Nothing in this
module writes to ``os.environ``.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent                 # .../backend/app/core
_REPO = _HERE.parents[2]                                  # repo root

# Later entries override earlier ones, so the most specific file goes last:
# repo-root .env < backend/.env < .env in the current working directory.
# Files that do not exist are skipped.
_ENV_FILES = (
    str(_REPO / ".env"),
    str(_HERE.parents[1] / ".env"),
    str(Path.cwd() / ".env"),
)

_DEFAULT_JWT_SECRET = "change-me-in-production"
_MIN_PROD_JWT_SECRET_LEN = 32

def _provider_allowed(model: str, allowed: List[str]) -> bool:
    """True when ``model`` is ``<provider>/...`` and the provider is allowed."""
    provider, sep, _ = model.partition("/")
    return bool(sep) and provider in allowed


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
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

    # MongoDB (primary store for FinRAG 2.0). Use 127.0.0.1, not "localhost",
    # which can resolve to ::1 and reach a different server on Windows.
    mongodb_url: str = "mongodb://127.0.0.1:27017"
    mongodb_test_url: str = "mongodb://127.0.0.1:27018/?directConnection=true"
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

    # LLM provider. The misspelling "omnirroute" is mapped to "omniroute"
    # (with a warning) by _normalise_llm_provider before validation.
    llm_provider: Literal["openai", "anthropic", "ollama", "stub", "omniroute"] = "stub"

    # OmniRoute gateway.
    omniroute_base_url: str = Field(
        default="http://127.0.0.1:20128/v1",
        validation_alias="OMNIROUTE_BASE_URL",
    )
    omniroute_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="OMNIROUTE_API_KEY",
    )

    # Deprecated misspellings from an earlier revision, kept so existing .env
    # files keep working. They are separate fields, not extra aliases on the
    # fields above, because pydantic-settings cannot tell us which alias
    # matched; a populated deprecated field is an unambiguous signal.
    # _resolve_deprecated_names folds them into the real fields.
    omnirroute_base_url: str | None = Field(
        default=None, validation_alias="OMNIRROUTE_BASE_URL", exclude=True, repr=False
    )
    omnirroute_api_key: SecretStr | None = Field(
        default=None, validation_alias="OMNIRROUTE_API_KEY", exclude=True, repr=False
    )

    # Model configuration: pinned IDs, never "auto" routing
    chat_model: str = "nvidia/nvidia/nemotron-3-super-120b-a12b"
    reasoning_model: str = "nvidia/nvidia/nemotron-3-ultra-550b-a55b"
    vision_model: str = ""      # to be configured
    embedding_model: str = ""   # to be configured (may be local)
    rerank_model: str = ""      # to be configured

    # Explicit pinned fallback chain, comma-separated
    fallback_chain: str = "nvidia/nvidia/nemotron-3-super-120b-a12b,ollama/llama3.1"

    # Providers allowed to receive tenant data, comma-separated
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
    jwt_secret: str = _DEFAULT_JWT_SECRET
    jwt_alg: str = "HS256"
    jwt_ttl_minutes: int = 720
    api_key_header: str = "X-API-Key"

    # Jobs
    jobs_mode: Literal["inproc", "celery"] = "inproc"
    redis_url: str = ""

    # CORS
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # -- validators ---------------------------------------------------------

    @field_validator("cors_origins")
    @classmethod
    def _strip_cors(cls, v: str) -> str:
        return ",".join(part.strip() for part in v.split(",") if part.strip())

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_llm_provider(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        v = v.strip().lower()
        if v == "omnirroute":
            log.warning(
                "LLM_PROVIDER='omnirroute' is deprecated (misspelling); "
                "use 'omniroute'."
            )
            return "omniroute"
        return v

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        self._resolve_deprecated_names()
        self._validate_omniroute()
        self._validate_jwt_secret()
        return self

    def _resolve_deprecated_names(self) -> None:
        """Honour the misspelled OMNIRROUTE_* names, with a warning.

        The correct name wins when both are set.
        """
        for old, new, replacement in (
            ("omnirroute_base_url", "omniroute_base_url", "OMNIROUTE_BASE_URL"),
            ("omnirroute_api_key", "omniroute_api_key", "OMNIROUTE_API_KEY"),
        ):
            value = getattr(self, old)
            if value is None:
                continue
            log.warning(
                "%s is deprecated (misspelling); use %s instead.",
                old.upper(),
                replacement,
            )
            if new not in self.model_fields_set:
                setattr(self, new, value)

    def _validate_omniroute(self) -> None:
        if self.llm_provider != "omniroute":
            return
        problems: List[str] = []
        if not self.omniroute_api_key.get_secret_value():
            problems.append("OMNIROUTE_API_KEY must be set when LLM_PROVIDER=omniroute")
        if not self.omniroute_base_url:
            problems.append("OMNIROUTE_BASE_URL must be set when LLM_PROVIDER=omniroute")
        if not self.chat_model:
            problems.append("CHAT_MODEL must be set when LLM_PROVIDER=omniroute")
        if not self.reasoning_model:
            problems.append("REASONING_MODEL must be set when LLM_PROVIDER=omniroute")

        allowed = self.allowed_providers_list
        if not allowed:
            problems.append(
                "ALLOWED_PROVIDERS must list at least one provider "
                "when LLM_PROVIDER=omniroute"
            )
        else:
            pinned = {
                "CHAT_MODEL": self.chat_model,
                "REASONING_MODEL": self.reasoning_model,
                "VISION_MODEL": self.vision_model,
            }
            for i, model in enumerate(self.fallback_chain_list):
                pinned[f"FALLBACK_CHAIN[{i}]"] = model
            for name, model in pinned.items():
                if model and not _provider_allowed(model, allowed):
                    problems.append(
                        f"{name}={model!r} uses a provider not in "
                        f"ALLOWED_PROVIDERS {allowed}"
                    )
        if problems:
            raise ValueError("; ".join(problems))

    def _validate_jwt_secret(self) -> None:
        if self.app_env != "prod":
            return
        if (
            self.jwt_secret == _DEFAULT_JWT_SECRET
            or len(self.jwt_secret) < _MIN_PROD_JWT_SECRET_LEN
        ):
            raise ValueError(
                "JWT_SECRET must be set to a random value of at least "
                f"{_MIN_PROD_JWT_SECRET_LEN} characters when APP_ENV=prod"
            )

    # -- derived values -----------------------------------------------------

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

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
    # Tests configure the environment explicitly (tests/conftest.py) and must
    # never pick up a developer's .env file.
    if os.environ.get("APP_ENV", "").lower() == "test":
        return Settings(_env_file=None)  # type: ignore[call-arg]
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Test helper: clear the cached Settings (after monkeypatching env)."""
    get_settings.cache_clear()

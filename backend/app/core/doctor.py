"""Read-only configuration diagnostics for ``ira doctor``.

Nothing here writes, creates a directory, opens a database cursor or calls a
model. The outbound traffic is a TCP connect to each endpoint the app itself is
configured to use, plus one ``GET /collections/{name}`` against the configured
Qdrant server: an answer a resolved config string cannot give is whether that
collection actually stores the dimension the embedder produces.

Secrets are never included in a finding — only whether one is present.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.core.config import _DEFAULT_JWT_SECRET, _ENV_FILES, Settings, get_settings

#: (host, port) -> reachable?
Probe = Callable[[str, int], bool]

#: (url, headers) -> (HTTP status or None when the call never completed, body)
HttpProbe = Callable[[str, dict], tuple[int | None, dict]]

#: DNS names docker-compose creates; resolvable only inside that network.
_COMPOSE_SERVICE_NAMES = frozenset(
    {"mongo", "mongo-test", "qdrant", "backend", "frontend", "redis"})

#: How a container addresses the machine running Docker.
_CONTAINER_ONLY_ALIASES = frozenset({"host.docker.internal"})


@dataclass(frozen=True, slots=True)
class Finding:
    level: str          # ok | info | warn | error
    area: str
    setting: str
    message: str
    fix: str = ""


def tcp_probe(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """True when something is listening. Connect only, no protocol traffic."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def http_get_json(url: str, headers: dict, timeout_s: float = 3.0):
    """One read-only GET, never retried, never raising.

    Returns ``(None, {})`` when the request never got an answer — the caller
    distinguishes "cannot tell" from "the server said no".
    """
    try:
        import httpx

        resp = httpx.get(url, headers=headers, timeout=timeout_s)
        body = resp.json() if resp.content else {}
        return resp.status_code, body if isinstance(body, dict) else {}
    except Exception:
        return None, {}


def _mask(url: str) -> str:
    """Strip any password from a URI before it is shown."""
    parts = urlsplit(url)
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, "", ""))


def _host_and_port(url: str, default_port: int) -> tuple[str, int] | None:
    parts = urlsplit(url)
    if not parts.hostname:
        return None
    return parts.hostname, parts.port or default_port


def _probe_endpoint(endpoint, label, setting, url, probe):
    if endpoint is None:
        return Finding("error", label, setting, f"{url!r} has no host to reach",
                       fix="Set the URL to a reachable host.")
    host, port = endpoint
    if probe is None:
        return Finding("info", label, setting, f"{host}:{port} (not probed)")
    if probe(host, port):
        return Finding("ok", label, setting, f"{host}:{port} is accepting connections")
    if host in _COMPOSE_SERVICE_NAMES or host in _CONTAINER_ONLY_ALIASES:
        kind = ("compose network name" if host in _COMPOSE_SERVICE_NAMES
                else "container alias for the host machine")
        return Finding("error", label, setting,
                       f"{host}:{port} is a {kind}, unreachable from this process",
                       fix=f"{setting} is in the container form. On the host, use "
                           "127.0.0.1 with the published port, or run this check "
                           "inside the container.")
    return Finding("error", label, setting, f"nothing is listening on {host}:{port}",
                   fix=f"Start the service, or point {setting} at the right endpoint.")


# ---------------------------------------------------------------------------


def _env_files_findings(settings: Settings) -> list[Finding]:
    present = [str(p) for p in _ENV_FILES if Path(p).exists()]
    if settings.app_env == "test":
        return [Finding("info", "env", "APP_ENV",
                        "APP_ENV=test: .env files are ignored and tests configure "
                        "the environment explicitly")]
    if not present:
        return [Finding("info", "env", ".env",
                        "no .env file found; defaults plus real environment "
                        "variables are in force")]
    return [Finding("info", "env", ".env",
                    f"{len(present)} .env file(s) loaded, last wins: {present[-1]}",
                    fix="Process environment variables still override these files.")]


def _mongo_findings(settings: Settings, probe) -> list[Finding]:
    endpoint = _host_and_port(settings.mongodb_url, 27017)
    out = [
        _probe_endpoint(endpoint, "mongodb", "MONGODB_URL",
                        _mask(settings.mongodb_url), probe),
        Finding("info", "mongodb", "MONGODB_DB_NAME",
                f"database {settings.mongodb_db_name!r}"),
    ]
    default = _host_and_port(Settings.model_fields["mongodb_url"].default or "", 27017)
    if endpoint and default and endpoint != default:
        out.append(Finding(
            "warn", "mongodb", "MONGODB_URL",
            f"reaches {endpoint[0]}:{endpoint[1]} rather than the packaged "
            f"{default[0]}:{default[1]}",
            fix="Any mongod answers a ping. Confirm this one holds the corpus "
                "before trusting document counts or search results."))
    prod = endpoint
    test = _host_and_port(settings.mongodb_test_url, 27017)
    if prod and prod == test:
        out.insert(0, Finding(
            "error", "mongodb", "MONGODB_TEST_URL",
            "the isolation suite and the application point at the same mongod",
            fix="Give MONGODB_TEST_URL its own server (or port); the isolation "
                "tests write, and would write into the production store."))
    return out


def _qdrant_findings(settings: Settings, probe, exists) -> list[Finding]:
    if settings.qdrant_mode == "local":
        path = Path(settings.qdrant_path)
        if not exists(path):
            return [Finding("warn", "qdrant", "QDRANT_PATH",
                            f"local path {path} does not exist",
                            fix="It is created on first ingest; vectors stay "
                                "inside this process' storage.")]
        return [Finding("info", "qdrant", "QDRANT_MODE",
                        f"local embedded store at {path}, collection "
                        f"{settings.qdrant_collection!r}",
                        fix="Only one process can hold local-mode storage; "
                            "ingest through the running server, or set "
                            "QDRANT_MODE=remote.")]
    if not settings.qdrant_url:
        return [Finding("error", "qdrant", "QDRANT_URL",
                        "QDRANT_MODE=remote but QDRANT_URL is empty",
                        fix="Set QDRANT_URL to the server, e.g. http://127.0.0.1:6333.")]
    findings = [_probe_endpoint(_host_and_port(settings.qdrant_url, 6333),
                                "qdrant", "QDRANT_URL",
                                _mask(settings.qdrant_url), probe),
                Finding("ok" if settings.qdrant_api_key else "info",
                        "qdrant", "QDRANT_API_KEY",
                        "api key present" if settings.qdrant_api_key
                        else "no api key set",
                        fix="" if settings.qdrant_api_key
                        else "Set QDRANT_API_KEY when the server requires it.")]
    return findings


def _collection_dimension_findings(settings: Settings, get_json) -> list[Finding]:
    """Compare the live collection's stored dimension with the configured embedder.

    Ingestion and retrieval read the same ``QDRANT_COLLECTION``, and
    ``ensure_collection`` refuses to touch a collection whose vectors differ in
    width from the embedder's output. A mismatch therefore does not degrade
    search quality quietly — it stops every ingest and returns nothing useful
    from retrieval, which is why the answer belongs in the pre-flight report.
    """
    if get_json is None:
        return []
    if settings.qdrant_mode != "remote" or not settings.qdrant_url:
        return []
    if settings.embeddings_provider == "stub":
        # Already reported as an error by _embedding_findings; stub vectors never
        # reach a remote collection.
        return []
    name = settings.qdrant_collection
    url = f"{settings.qdrant_url.rstrip('/')}/collections/{name}"
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    status, body = get_json(url, headers)
    if status is None:
        return [Finding("warn", "qdrant", "QDRANT_COLLECTION",
                        f"{name!r} could not be read over REST on {settings.qdrant_url}",
                        fix="The port answered but the HTTP call did not; check the "
                            "scheme (REST is 6333, not the 6334 gRPC port) and "
                            "QDRANT_API_KEY.")]
    if status == 404:
        return [Finding("info", "qdrant", "QDRANT_COLLECTION",
                        f"collection {name!r} does not exist on this server",
                        fix=f"The first ingest creates it at {settings.embeddings_dim} "
                            "dimensions; until then every search returns no chunks.")]
    if status >= 400:
        return [Finding("error", "qdrant", "QDRANT_COLLECTION",
                        f"reading collection {name!r} returned HTTP {status}",
                        fix="Point QDRANT_URL at the server holding the corpus, or "
                            "set QDRANT_API_KEY if it requires one.")]
    result = body.get("result") or {}
    params = ((result.get("config") or {}).get("params") or {}).get("vectors")
    stored_dim = params.get("size") if isinstance(params, dict) else None
    points = result.get("points_count")
    if stored_dim is None:
        return [Finding("warn", "qdrant", "QDRANT_COLLECTION",
                        f"collection {name!r} has no single unnamed vector, so its "
                        "dimension cannot be read",
                        fix="This app writes one unnamed cosine vector per point; a "
                            "named-vector collection needs its own migration.")]
    if stored_dim != settings.embeddings_dim:
        return [Finding(
            "error", "qdrant", "QDRANT_COLLECTION",
            f"{name!r} stores {stored_dim}-dimensional vectors ({points} points) but "
            f"EMBEDDING_DIM={settings.embeddings_dim} for "
            f"{settings.embedding_model!r}",
            fix="Ingest refuses to widen or narrow an existing collection, so this "
                "pair stores nothing. Either match EMBEDDING_DIM/EMBEDDING_MODEL to "
                "the collection or point QDRANT_COLLECTION at one built by the "
                "configured model; existing vectors are never rewritten.")]
    return [Finding("ok", "qdrant", "QDRANT_COLLECTION",
                    f"{name!r} stores {stored_dim}-dimensional vectors "
                    f"({points} points), matching {settings.embedding_model!r}")]


def _embedding_findings(settings: Settings, probe) -> list[Finding]:
    provider = settings.embeddings_provider
    if provider == "stub":
        if settings.qdrant_mode == "remote":
            return [Finding("error", "embeddings", "EMBEDDINGS_PROVIDER",
                            "stub vectors cannot be written to a remote collection; "
                            "ingestion refuses this combination",
                            fix="Set EMBEDDINGS_PROVIDER=ollama (or openai), or run "
                                "against a local store.")]
        return [Finding("warn", "embeddings", "EMBEDDINGS_PROVIDER",
                        "stub embeddings: vectors are hash placeholders, so dense "
                        "retrieval returns noise and any quality metric is meaningless",
                        fix="Set EMBEDDINGS_PROVIDER=ollama for local nomic-embed-text.")]
    if provider == "ollama":
        # OllamaEmbeddings strips any "<provider>/" prefix, so both
        # "nomic-embed-text" and "ollama/nomic-embed-text" are valid here.
        model = settings.embedding_model.split("/", 1)[-1]
        out = [
            _probe_endpoint(_host_and_port(settings.ollama_base_url, 11434),
                           "embeddings", "OLLAMA_BASE_URL",
                           _mask(settings.ollama_base_url), probe),
            Finding("info", "embeddings", "EMBEDDING_MODEL",
                    f"{model} at {settings.embeddings_dim} dimensions"),
        ]
        expected = {"nomic-embed-text": 768}
        for name, dim in expected.items():
            if name in settings.embedding_model and settings.embeddings_dim != dim:
                out.append(Finding("error", "embeddings", "EMBEDDING_DIM",
                                   f"{name} emits {dim} dimensions but "
                                   f"EMBEDDING_DIM={settings.embeddings_dim}",
                                   fix=f"Set EMBEDDING_DIM={dim} and re-create the "
                                       "collection; dimension mismatch fails every upsert."))
        return out
    if not settings.openai_api_key.get_secret_value():
        return [Finding("error", "embeddings", "OPENAI_API_KEY",
                        "EMBEDDINGS_PROVIDER=openai but no API key is configured",
                        fix="Set OPENAI_API_KEY.")]
    return [Finding("ok", "embeddings", "EMBEDDINGS_PROVIDER",
                    f"{provider}, dim={settings.embeddings_dim}")]


def _llm_findings(settings: Settings, probe) -> list[Finding]:
    provider = settings.llm_provider
    if provider == "stub":
        return [Finding("warn", "llm", "LLM_PROVIDER",
                        "stub generator: answers are canned text, so /chat cannot "
                        "be evaluated for faithfulness",
                        fix="Set LLM_PROVIDER=omniroute (or openai/anthropic/ollama).")]
    if provider == "omniroute":
        # Settings already refuses to load an omniroute configuration with a
        # missing key or a pinned model outside ALLOWED_PROVIDERS, so the
        # doctor only reports what is in force and whether it is reachable.
        return [_probe_endpoint(_host_and_port(settings.omniroute_base_url, 80),
                               "llm", "LLM_BASE_URL",
                               _mask(settings.omniroute_base_url), probe),
                Finding("ok", "llm", "LLM_API_KEY", "gateway key configured"),
                Finding("ok", "llm", "CHAT_MODEL", settings.chat_model),
                Finding("ok", "llm", "REASONING_MODEL", settings.reasoning_model),
                Finding("info", "llm", "VISION_MODEL", settings.vision_model or "(unset)")]
    if provider == "ollama":
        return [_probe_endpoint(_host_and_port(settings.ollama_base_url, 11434),
                                "llm", "OLLAMA_BASE_URL",
                                _mask(settings.ollama_base_url), probe),
                Finding("ok", "llm", "OLLAMA_CHAT_MODEL", settings.ollama_chat_model)]
    key = (settings.openai_api_key if provider == "openai" else settings.anthropic_api_key
           ).get_secret_value()
    return [Finding("ok" if key else "error", "llm", f"{provider.upper()}_API_KEY",
                    "key present" if key else "no API key configured",
                    fix="" if key else f"Set {provider.upper()}_API_KEY.")]


def _reranker_findings(settings: Settings) -> list[Finding]:
    if settings.reranker_provider == "cohere" and not settings.cohere_api_key:
        return [Finding("error", "reranker", "COHERE_API_KEY",
                        "RERANKER_PROVIDER=cohere but no key is configured",
                        fix="Set COHERE_API_KEY or use the stub/cross-encoder.")]
    if settings.reranker_provider == "stub":
        return [Finding("info", "reranker", "RERANKER_PROVIDER",
                        "stub reranker keeps the fusion order; recall is measured "
                        "without reordering")]
    return [Finding("ok", "reranker", "RERANKER_PROVIDER", settings.reranker_provider)]


def _retrieval_findings(settings: Settings) -> list[Finding]:
    out = []
    total = settings.hybrid_dense_weight + settings.hybrid_bm25_weight
    out.append(Finding("ok" if abs(total - 1.0) < 1e-6 else "error", "retrieval",
                       "HYBRID_*_WEIGHT",
                       f"dense {settings.hybrid_dense_weight} + bm25 "
                       f"{settings.hybrid_bm25_weight} = {round(total, 4)}",
                       fix="" if abs(total - 1.0) < 1e-6 else "Weights should sum to 1."))
    if settings.retrieval_page_quota:
        out.append(Finding("info", "retrieval", "RETRIEVAL_PAGE_QUOTA",
                           f"keeping at most {settings.retrieval_page_quota} chunk(s) "
                           "per document page",
                           fix="0 restores the original ranking exactly."))
    return out


def _extraction_findings(settings: Settings) -> list[Finding]:
    return [
        Finding("info", "parser", "PDFPLUMBER_DEDUPE",
                f"{'on' if settings.pdfplumber_dedupe else 'off'} — changes the "
                "extracted text, so stored chunks are only consistent after a "
                "re-ingest"),
        Finding("info", "parser", "PDFPLUMBER_TABLE_STRATEGY",
                f"{settings.pdfplumber_table_strategy!r} — changes parent-chunk "
                "table text and the extractor fingerprint"),
    ]


def _auth_findings(settings: Settings) -> list[Finding]:
    if settings.jwt_secret == _DEFAULT_JWT_SECRET:
        # APP_ENV=prod cannot load with this value at all (Settings rejects
        # it), so anything reaching here is a development deployment.
        return [Finding("warn", "auth", "JWT_SECRET",
                        "the shipped development secret is in force",
                        fix="Set JWT_SECRET to a random value of at least 32 "
                            "characters before exposing the app.")]
    return [Finding("ok", "auth", "JWT_SECRET", "custom secret configured")]


def _jobs_findings(settings: Settings) -> list[Finding]:
    if settings.jobs_mode == "celery" and not settings.redis_url:
        return [Finding("error", "jobs", "REDIS_URL",
                        "JOBS_MODE=celery but no broker is configured",
                        fix="Set REDIS_URL, or use JOBS_MODE=inproc.")]
    return [Finding("info", "jobs", "JOBS_MODE", settings.jobs_mode)]


def _cors_findings(settings: Settings) -> list[Finding]:
    origins = settings.cors_origins_list
    if not origins:
        return [Finding("warn", "cors", "CORS_ORIGINS", "no allowed origins; "
                        "the browser app will be blocked")]
    if "*" in origins:
        return [Finding("error", "cors", "CORS_ORIGINS", "wildcard origin allows "
                        "any site to call the API with an API key",
                        fix="List the deployed frontend origins.")]
    return [Finding("ok", "cors", "CORS_ORIGINS", ", ".join(origins))]


def _storage_findings(settings: Settings, exists) -> list[Finding]:
    out = []
    for setting, value in (("STORAGE_PATH", settings.storage_path),
                           ("EVAL_SUITES_DIR", settings.eval_suites_dir),
                           ("DATABASE_URL", settings.database_url)):
        if value.startswith("sqlite:///"):
            target = Path(value[len("sqlite:///"):]).resolve().parent
        else:
            target = Path(value)
        if not exists(target):
            out.append(Finding("warn", "storage", setting,
                               f"{target} does not exist",
                               fix="Directories are created by `ira init-db`; this "
                                   "check never writes."))
        else:
            out.append(Finding("ok", "storage", setting, str(target)))
    return out


def run_diagnostics(settings: Settings | None = None, *,
                    probe: Probe | None = tcp_probe,
                    exists: Callable[[Path], bool] = Path.exists,
                    get_json: HttpProbe | None = None) -> list[Finding]:
    """Every finding for the current configuration, ordered by area.

    ``probe=None`` reports endpoints as unprobed instead of opening sockets.
    ``get_json`` defaults to no HTTP at all: only the CLI asks the vector store
    what dimension its collection actually holds.
    """
    s = settings if settings is not None else get_settings()
    findings: list[Finding] = []
    findings += _env_files_findings(s)
    findings += _mongo_findings(s, probe)
    findings += _qdrant_findings(s, probe, exists)
    findings += _collection_dimension_findings(s, get_json)
    findings += _embedding_findings(s, probe)
    findings += _llm_findings(s, probe)
    findings += _reranker_findings(s)
    findings += _retrieval_findings(s)
    findings += _extraction_findings(s)
    findings += _auth_findings(s)
    findings += _jobs_findings(s)
    findings += _cors_findings(s)
    findings += _storage_findings(s, exists)
    return findings

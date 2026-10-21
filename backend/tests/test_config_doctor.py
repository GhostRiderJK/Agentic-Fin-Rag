"""Offline tests for the read-only configuration doctor.

``Settings`` keyword arguments outrank the process environment and .env files,
so each test builds an exact configuration and asserts on the findings it
produces. Probing and directory existence are injected: the doctor never opens
a socket or touches the filesystem in these tests.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.doctor import Finding, _mask, run_diagnostics

NOTHING_EXISTS = lambda _p: False  # noqa: E731
ALL_REACHABLE = lambda _h, _p: True  # noqa: E731
NONE_REACHABLE = lambda _h, _p: False  # noqa: E731


def report(**overrides) -> list[Finding]:
    settings = Settings(_env_file=None, **overrides)
    return run_diagnostics(settings, probe=ALL_REACHABLE, exists=NOTHING_EXISTS)


def areas(findings, level: str) -> set[str]:
    return {f.setting for f in findings if f.level == level}


def message_for(findings, setting: str, level: str | None = None) -> str:
    matches = [f for f in findings
               if f.setting == setting and (level is None or f.level == level)]
    assert matches, f"no finding for {setting} ({level}) in {[f.setting for f in findings]}"
    return " | ".join(f.message + " " + f.fix for f in matches)


def test_defaults_are_only_warnings_not_errors():
    findings = report()
    assert not [f for f in findings if f.level == "error"], findings


def test_stub_embeddings_against_a_remote_store_is_an_error():
    findings = report(embeddings_provider="stub", qdrant_mode="remote",
                      qdrant_url="http://127.0.0.1:6333")
    assert "EMBEDDINGS_PROVIDER" in areas(findings, "error")
    assert "remote" in message_for(findings, "EMBEDDINGS_PROVIDER", "error")


def test_stub_embeddings_locally_is_a_warning_about_retrieval_quality():
    findings = report(embeddings_provider="stub")
    assert "EMBEDDINGS_PROVIDER" in areas(findings, "warn")
    assert "hash placeholders" in message_for(findings, "EMBEDDINGS_PROVIDER", "warn")


def test_ollama_dimension_mismatch_is_an_error():
    findings = report(embeddings_provider="ollama",
                      embedding_model="ollama/nomic-embed-text",
                      embeddings_dim=1536)
    assert "EMBEDDING_DIM" in areas(findings, "error")
    assert "768" in message_for(findings, "EMBEDDING_DIM", "error")


def test_unqualified_ollama_model_name_is_accepted():
    """OllamaEmbeddings strips any "<provider>/" prefix, so both forms work."""
    for model in ("nomic-embed-text", "ollama/nomic-embed-text"):
        findings = report(embeddings_provider="ollama",
                          embedding_model=model, embeddings_dim=768)
        assert not [f for f in findings if f.level == "error"
                    and f.setting == "EMBEDDING_MODEL"], findings


def test_remote_qdrant_without_a_url_is_an_error():
    findings = report(qdrant_mode="remote", qdrant_url="")
    assert "QDRANT_URL" in areas(findings, "error")


def test_test_database_pointing_at_production_is_an_error():
    findings = report(mongodb_url="mongodb://127.0.0.1:27019",
                      mongodb_test_url="mongodb://127.0.0.1:27019")
    assert "MONGODB_TEST_URL" in areas(findings, "error")
    assert "isolation" in message_for(findings, "MONGODB_TEST_URL", "error")


def test_compose_service_names_are_called_out_as_container_form():
    settings = Settings(_env_file=None, qdrant_mode="remote",
                        qdrant_url="http://qdrant:6333")
    findings = run_diagnostics(settings, probe=NONE_REACHABLE, exists=NOTHING_EXISTS)
    assert "compose network name" in message_for(findings, "QDRANT_URL", "error")


def test_host_docker_internal_is_called_out_as_container_form():
    settings = Settings(_env_file=None, embeddings_provider="ollama",
                        ollama_base_url="http://host.docker.internal:11434")
    findings = run_diagnostics(settings, probe=NONE_REACHABLE, exists=NOTHING_EXISTS)
    assert "container alias" in message_for(findings, "OLLAMA_BASE_URL", "error")


def test_a_non_default_mongo_endpoint_is_flagged_for_confirmation():
    findings = report(mongodb_url="mongodb://127.0.0.1:27017")
    assert "MONGODB_URL" in areas(findings, "warn")
    assert "before trusting document counts" in message_for(findings, "MONGODB_URL", "warn")


def test_unreachable_endpoints_are_reported_per_service():
    settings = Settings(_env_file=None, embeddings_provider="ollama",
                        llm_provider="ollama")
    findings = run_diagnostics(settings, probe=NONE_REACHABLE, exists=NOTHING_EXISTS)
    errors = [f for f in findings if f.level == "error"]
    assert {f.area for f in errors} >= {"mongodb", "llm", "embeddings"}
    assert all("nothing is listening" in f.message for f in errors)


def test_probe_none_reports_endpoints_instead_of_guessing():
    settings = Settings(_env_file=None)
    findings = run_diagnostics(settings, probe=None, exists=NOTHING_EXISTS)
    assert not [f for f in findings if f.level == "error"]
    assert "not probed" in message_for(findings, "MONGODB_URL")


def test_celery_without_a_broker_is_an_error():
    findings = report(jobs_mode="celery", redis_url="")
    assert "REDIS_URL" in areas(findings, "error")


def test_wildcard_cors_is_an_error():
    findings = report(cors_origins="*")
    assert "CORS_ORIGINS" in areas(findings, "error")


def test_hybrid_weights_that_do_not_sum_to_one_are_an_error():
    findings = report(hybrid_dense_weight=0.9, hybrid_bm25_weight=0.9)
    assert "HYBRID_*_WEIGHT" in areas(findings, "error")


def test_development_jwt_secret_is_flagged():
    assert "JWT_SECRET" in areas(report(jwt_secret="change-me-in-production"), "warn")
    assert "JWT_SECRET" not in areas(report(jwt_secret="x" * 40), "warn")


def test_storage_gaps_are_reported_without_being_created(tmp_path, monkeypatch):
    """The doctor warns about a missing directory; it must not mkdir one."""
    missing = tmp_path / "nope"
    calls: list[Path] = []

    def spy(path: Path) -> bool:
        calls.append(path)
        return path.exists()

    settings = Settings(_env_file=None, storage_path=str(missing),
                        eval_suites_dir=str(missing / "suites"))
    findings = run_diagnostics(settings, probe=ALL_REACHABLE, exists=spy)
    assert not missing.exists()
    assert any("does not exist" in f.message for f in findings)
    assert missing in calls


def test_credentials_never_appear_in_a_finding():
    settings = Settings(_env_file=None,
                        mongodb_url="mongodb://admin:s3cr3t-pass@127.0.0.1:27019",
                        qdrant_api_key="qdrant-secret")
    findings = run_diagnostics(settings, probe=NONE_REACHABLE, exists=NOTHING_EXISTS)
    blob = " ".join(f.message + f.fix for f in findings)
    assert "s3cr3t-pass" not in blob
    assert "admin" not in blob
    assert "qdrant-secret" not in blob


def test_mask_only_shows_that_a_password_exists():
    assert _mask("mongodb://u:p@h:27017/db") == "mongodb://***@h:27017/db"
    assert _mask("mongodb://127.0.0.1:27019") == "mongodb://127.0.0.1:27019"


def test_parser_flags_are_reported_as_reingest_boundaries():
    findings = report(pdfplumber_dedupe=True, pdfplumber_table_strategy="lines")
    text = message_for(findings, "PDFPLUMBER_DEDUPE") + message_for(
        findings, "PDFPLUMBER_TABLE_STRATEGY")
    assert "re-ingest" in text
    assert "fingerprint" in text


# ---------------------------------------------------------------------------
# live collection dimension vs the configured embedder
# ---------------------------------------------------------------------------


def collection_report(status, body, *, calls=None, **overrides) -> list[Finding]:
    def fake_get_json(url: str, headers: dict):
        if calls is not None:
            calls.append((url, headers))
        return status, body

    conf = dict(qdrant_mode="remote", qdrant_url="http://127.0.0.1:6333",
                qdrant_collection="chunks_nomic768", embeddings_provider="ollama",
                embedding_model="nomic-embed-text", embeddings_dim=768)
    conf.update(overrides)
    settings = Settings(_env_file=None, **conf)
    return run_diagnostics(settings, probe=ALL_REACHABLE, exists=NOTHING_EXISTS,
                           get_json=fake_get_json)


def collection_body(dim, points=1816) -> dict:
    return {"result": {"points_count": points,
                       "config": {"params": {"vectors": {"size": dim,
                                                         "distance": "Cosine"}}}}}


def test_matching_collection_dimension_is_reported_as_ok():
    findings = collection_report(200, collection_body(768))
    text = message_for(findings, "QDRANT_COLLECTION", "ok")
    assert "768-dimensional" in text and "1816 points" in text


def test_stored_dimension_that_differs_from_the_embedder_is_an_error():
    findings = collection_report(200, collection_body(1536))
    text = message_for(findings, "QDRANT_COLLECTION", "error")
    assert "1536-dimensional" in text and "EMBEDDING_DIM=768" in text
    assert "never rewritten" in text


def test_missing_collection_is_reported_as_the_empty_corpus_it_is():
    findings = collection_report(404, {"result": None, "status": "error"})
    text = message_for(findings, "QDRANT_COLLECTION", "info")
    assert "does not exist" in text and "creates it at 768" in text


def test_unreadable_collection_never_stands_as_a_clean_bill_of_health():
    findings = collection_report(None, {})
    assert "could not be read over REST" in message_for(findings, "QDRANT_COLLECTION", "warn")
    findings = collection_report(403, {"status": "error"})
    assert "HTTP 403" in message_for(findings, "QDRANT_COLLECTION", "error")


def test_named_vector_collection_is_flagged_rather_than_guessed():
    body = {"result": {"points_count": 5,
                       "config": {"params": {"vectors": {"dense": {"size": 768}}}}}}
    findings = collection_report(200, body)
    assert "no single unnamed vector" in message_for(findings, "QDRANT_COLLECTION", "warn")


def test_local_mode_and_stub_embedders_skip_the_http_read():
    calls: list = []
    findings = collection_report(200, collection_body(768), calls=calls,
                                 qdrant_mode="local", qdrant_path="/tmp/nope")
    assert calls == []
    assert not [f for f in findings if f.setting == "QDRANT_COLLECTION"]
    calls.clear()
    findings = collection_report(200, collection_body(768), calls=calls,
                                 embeddings_provider="stub")
    assert calls == []
    assert not [f for f in findings if f.setting == "QDRANT_COLLECTION"]


def test_doctor_without_a_fetcher_opens_no_http():
    """The offline default must stay offline: no get_json means no REST read."""
    findings = report(qdrant_mode="remote", qdrant_url="http://127.0.0.1:6333",
                      embeddings_provider="ollama", embeddings_dim=1536,
                      embedding_model="nomic-embed-text")
    assert not [f for f in findings if f.setting == "QDRANT_COLLECTION"]


def test_qdrant_api_key_goes_in_the_header_and_never_in_the_text():
    calls: list = []
    findings = collection_report(200, collection_body(768), calls=calls,
                                 qdrant_api_key="qdrant-rest-secret")
    assert calls[0][1] == {"api-key": "qdrant-rest-secret"}
    assert calls[0][0].endswith("/collections/chunks_nomic768")
    blob = " ".join(f.message + f.fix for f in findings)
    assert "qdrant-rest-secret" not in blob

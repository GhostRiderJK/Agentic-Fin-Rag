"""Configuration tests: provider validation, deprecated names, precedence.

These construct ``Settings`` directly (never ``get_settings``) and scrub every
config-related variable first, so a developer's shell or .env cannot change
the outcome.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from app.core.config import Settings

LOGGER = "app.core.config"

_CONFIG_ENV_VARS = (
    "APP_ENV",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "OMNIROUTE_API_KEY",
    "OMNIROUTE_BASE_URL",
    "OMNIRROUTE_API_KEY",
    "OMNIRROUTE_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CHAT_MODEL",
    "REASONING_MODEL",
    "VISION_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDINGS_DIM",
    "FALLBACK_CHAIN",
    "ALLOWED_PROVIDERS",
    "JWT_SECRET",
    "RETRIEVAL_PAGE_QUOTA",
)

STRONG_SECRET = "x" * 40


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_settings(monkeypatch):
    """Build Settings from the given env vars, ignoring any .env file."""

    def _make(env_file=None, **env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return Settings(_env_file=env_file)

    return _make


def _omniroute_env(**overrides):
    env = {"LLM_PROVIDER": "omniroute", "OMNIROUTE_API_KEY": "secret-value"}
    env.update(overrides)
    return env


# -- provider validation ----------------------------------------------------


def test_default_provider_is_stub_and_needs_no_credentials(make_settings):
    assert make_settings().llm_provider == "stub"


def test_unknown_llm_provider_raises(make_settings):
    with pytest.raises(ValidationError, match="llm_provider"):
        make_settings(LLM_PROVIDER="bogus")


def test_omniroute_requires_an_api_key(make_settings):
    with pytest.raises(ValidationError, match="OMNIROUTE_API_KEY"):
        make_settings(LLM_PROVIDER="omniroute")


def test_omniroute_requires_a_base_url(make_settings):
    with pytest.raises(ValidationError, match="OMNIROUTE_BASE_URL"):
        make_settings(**_omniroute_env(OMNIROUTE_BASE_URL=""))


def test_valid_omniroute_config_loads_and_hides_the_key(make_settings):
    s = make_settings(**_omniroute_env())
    assert s.llm_provider == "omniroute"
    assert s.omniroute_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(s)
    assert "secret-value" not in str(s)


def test_model_from_a_provider_outside_the_allow_list_is_rejected(make_settings):
    with pytest.raises(ValidationError, match="ALLOWED_PROVIDERS"):
        make_settings(**_omniroute_env(CHAT_MODEL="openrouter/nvidia/x:free"))


def test_fallback_chain_outside_the_allow_list_is_rejected(make_settings):
    with pytest.raises(ValidationError, match="FALLBACK_CHAIN"):
        make_settings(
            **_omniroute_env(FALLBACK_CHAIN="nvidia/a,gemini/gemini-3.6-flash")
        )


def test_model_without_a_provider_prefix_is_rejected(make_settings):
    with pytest.raises(ValidationError, match="ALLOWED_PROVIDERS"):
        make_settings(**_omniroute_env(CHAT_MODEL="nemotron-3-super"))


def test_empty_allow_list_is_rejected(make_settings):
    with pytest.raises(ValidationError, match="ALLOWED_PROVIDERS"):
        make_settings(**_omniroute_env(ALLOWED_PROVIDERS=""))


# -- deprecated misspelling -------------------------------------------------


def test_deprecated_provider_value_maps_and_warns(make_settings, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        s = make_settings(**_omniroute_env(LLM_PROVIDER="omnirroute"))
    assert s.llm_provider == "omniroute"
    assert any("deprecated" in r.getMessage() for r in caplog.records)


def test_deprecated_key_name_still_works_and_warns(make_settings, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        s = make_settings(LLM_PROVIDER="omniroute", OMNIRROUTE_API_KEY="old-key")
    assert s.omniroute_api_key.get_secret_value() == "old-key"
    assert any("OMNIRROUTE_API_KEY" in r.getMessage() for r in caplog.records)


def test_deprecated_base_url_name_still_works_and_warns(make_settings, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        s = make_settings(
            **_omniroute_env(OMNIRROUTE_BASE_URL="http://127.0.0.1:9999/v1")
        )
    assert s.omniroute_base_url == "http://127.0.0.1:9999/v1"
    assert any("OMNIRROUTE_BASE_URL" in r.getMessage() for r in caplog.records)


def test_correct_names_produce_no_deprecation_warning(make_settings, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        make_settings(
            **_omniroute_env(OMNIROUTE_BASE_URL="http://127.0.0.1:9999/v1")
        )
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]


def test_correct_name_wins_when_both_names_are_set(make_settings):
    s = make_settings(
        LLM_PROVIDER="omniroute",
        OMNIROUTE_API_KEY="new-key",
        OMNIRROUTE_API_KEY="old-key",
    )
    assert s.omniroute_api_key.get_secret_value() == "new-key"
    assert "old-key" not in repr(s)


# -- pinned model configuration ---------------------------------------------


def test_llm_names_are_accepted_as_gateway_aliases(make_settings):
    s = make_settings(
        **_omniroute_env(LLM_API_KEY="llm-name", LLM_BASE_URL="http://gw:1/v1")
    )
    assert s.omniroute_api_key.get_secret_value() == "llm-name"
    assert s.omniroute_base_url == "http://gw:1/v1"


def test_preferred_llm_name_wins_over_the_omniroute_name(make_settings):
    s = make_settings(
        LLM_PROVIDER="omniroute",
        LLM_API_KEY="preferred",
        OMNIROUTE_API_KEY="legacy",
    )
    assert s.omniroute_api_key.get_secret_value() == "preferred"
    assert "legacy" not in repr(s)


def test_defaults_pin_the_measured_models(make_settings):
    """nemotron-3-ultra-550b is advertised by the gateway but dead upstream.

    A default that names it costs every request its timeout budget before
    falling back, so the defaults are asserted here instead of left to drift.
    """
    s = make_settings()
    assert s.chat_model == "nvidia/nvidia/nemotron-3-super-120b-a12b"
    assert s.reasoning_model == s.chat_model
    assert "ultra" not in s.reasoning_model
    assert s.vision_model == "nvidia/meta/llama-3.2-11b-vision-instruct"
    assert s.fallback_chain_list == [
        "nvidia/nvidia/nemotron-3-super-120b-a12b",
        "nvidia/poolside/laguna-xs-2.1",
        "nvidia/stepfun-ai/step-5-preview",
        "ollama/llama3.1",
    ]
    assert s.embedding_model == "ollama/nomic-embed-text"
    assert s.embeddings_dim == 768


def test_embedding_dim_prefers_the_short_name(make_settings):
    assert make_settings(EMBEDDING_DIM="1024").embeddings_dim == 1024


def test_embedding_dim_accepts_the_plural_legacy_name(make_settings):
    assert make_settings(EMBEDDINGS_DIM="512").embeddings_dim == 512


# -- page diversity quota ---------------------------------------------------


def test_page_quota_is_disabled_by_default(make_settings):
    assert make_settings().retrieval_page_quota == 0


@pytest.mark.parametrize("value", ["0", "1", "2"])
def test_supported_page_quota_values_are_accepted(make_settings, value: str):
    assert make_settings(RETRIEVAL_PAGE_QUOTA=value).retrieval_page_quota == int(value)


@pytest.mark.parametrize("value", ["-1", "3", "10"])
def test_out_of_range_page_quota_is_rejected(make_settings, value: str):
    with pytest.raises(ValidationError, match="RETRIEVAL_PAGE_QUOTA"):
        make_settings(RETRIEVAL_PAGE_QUOTA=value)


def test_non_numeric_page_quota_is_rejected(make_settings):
    with pytest.raises(ValidationError, match="retrieval_page_quota"):
        make_settings(RETRIEVAL_PAGE_QUOTA="abc")


# -- precedence -------------------------------------------------------------


def test_real_env_overrides_dotenv(make_settings, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("CHAT_MODEL=nvidia/from-dotenv\n", encoding="utf-8")
    s = make_settings(env_file=str(dotenv), CHAT_MODEL="nvidia/from-env")
    assert s.chat_model == "nvidia/from-env"


def test_dotenv_is_used_when_the_env_var_is_absent(make_settings, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("CHAT_MODEL=nvidia/from-dotenv\n", encoding="utf-8")
    assert make_settings(env_file=str(dotenv)).chat_model == "nvidia/from-dotenv"


def test_unknown_keys_in_dotenv_are_ignored(make_settings, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("NOT_A_SETTING=1\nCHAT_MODEL=nvidia/x\n", encoding="utf-8")
    assert make_settings(env_file=str(dotenv)).chat_model == "nvidia/x"


# -- production JWT secret --------------------------------------------------


def test_default_jwt_secret_is_refused_in_prod(make_settings):
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        make_settings(APP_ENV="prod")


def test_short_jwt_secret_is_refused_in_prod(make_settings):
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        make_settings(APP_ENV="prod", JWT_SECRET="too-short")


def test_strong_jwt_secret_is_accepted_in_prod(make_settings):
    assert make_settings(APP_ENV="prod", JWT_SECRET=STRONG_SECRET).app_env == "prod"


def test_default_jwt_secret_is_allowed_in_dev(make_settings):
    assert make_settings(APP_ENV="dev").app_env == "dev"

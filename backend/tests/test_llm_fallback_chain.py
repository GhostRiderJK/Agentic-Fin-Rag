"""The gateway pinned fallback chain must be used at request time.

The catalogue lists model ids that are upstream-dead, and one failed call puts a
model on a ~2 minute credential cooldown, so a single pinned id is not a
reliable dependency. FALLBACK_CHAIN is the operator's statement of what else is
acceptable; these tests pin that it is actually consulted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.core.errors import ProviderError
from app.providers.llm import OmnirouteLLM

PRIMARY = "nvidia/nvidia/nemotron-3-super-120b-a12b"
SECONDARY = "nvidia/poolside/laguna-xs-2.1"
THIRD = "nvidia/stepfun-ai/step-5-preview"
FOURTH = "ollama/llama3.1"


class _FakeCompletions:
    def __init__(self, behaviour) -> None:
        self.behaviour = behaviour
        self.requested: list[str] = []

    def create(self, **kwargs):
        self.requested.append(kwargs["model"])
        outcome = self.behaviour(kwargs["model"])
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(content=outcome, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class _FakeClient:
    def __init__(self, behaviour) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(behaviour))


def _provider(monkeypatch, behaviour):
    """Return (provider, models_requested_in_order)."""
    llm = OmnirouteLLM()
    client = _FakeClient(behaviour)
    monkeypatch.setattr(llm, "_client", client)
    return llm, client.chat.completions.requested


def _chain(monkeypatch, models: list[str]) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "chat_model", models[0], raising=False)
    monkeypatch.setattr(settings, "fallback_chain", ",".join(models), raising=False)


def test_dead_primary_falls_to_the_next_pinned_model(monkeypatch):
    _chain(monkeypatch, [PRIMARY, SECONDARY, THIRD, FOURTH])

    def behaviour(model):
        if model == PRIMARY:
            return RuntimeError("[nvidia/x] [404]: upstream error (reset after 2m)")
        return f"answered by {model}"

    llm, requested = _provider(monkeypatch, behaviour)
    result = llm.chat([{"role": "user", "content": "hi"}])

    assert result.content == f"answered by {SECONDARY}"
    assert requested == [PRIMARY, SECONDARY]


def test_transient_primary_also_falls_through(monkeypatch):
    _chain(monkeypatch, [PRIMARY, SECONDARY])

    def behaviour(model):
        if model == PRIMARY:
            return RuntimeError("Rate limit exceeded, temporarily unavailable")
        return "ok"

    llm, requested = _provider(monkeypatch, behaviour)
    assert llm.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert requested == [PRIMARY, SECONDARY]


def test_every_model_failing_surfaces_one_error(monkeypatch):
    _chain(monkeypatch, [PRIMARY, SECONDARY, THIRD])

    def behaviour(model):
        return RuntimeError(f"{model} is not available")

    llm, requested = _provider(monkeypatch, behaviour)
    with pytest.raises(ProviderError) as exc:
        llm.chat([{"role": "user", "content": "hi"}])

    assert "All 3 pinned gateway models failed" in str(exc.value)
    assert requested == [PRIMARY, SECONDARY, THIRD]


def test_repeated_ids_are_attempted_once(monkeypatch):
    # The shipped default repeats the primary id inside FALLBACK_CHAIN.
    _chain(monkeypatch, [PRIMARY, PRIMARY, SECONDARY])

    def behaviour(model):
        if model == PRIMARY:
            return RuntimeError("404 upstream error")
        return "ok"

    llm, requested = _provider(monkeypatch, behaviour)
    assert llm.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert requested == [PRIMARY, SECONDARY]


def test_plain_openai_provider_does_not_use_the_gateway_chain(monkeypatch):
    """The chain holds gateway ids; sending them to api.openai.com is not a fallback."""
    from app.providers.llm import OpenAIChatLLM

    llm = OpenAIChatLLM(api_key="k", model="gpt-4o-mini")
    client = _FakeClient(
        lambda model: "ok" if model != "gpt-4o-mini" else RuntimeError("boom"))
    monkeypatch.setattr(llm, "_client", client)
    _chain(monkeypatch, [PRIMARY, SECONDARY])

    with pytest.raises(ProviderError):
        llm.chat([{"role": "user", "content": "hi"}])

    assert client.chat.completions.requested == ["gpt-4o-mini"]

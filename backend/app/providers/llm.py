"""LLM provider abstraction.

Implementations:
  - OpenAIChatLLM        (default when OPENAI_API_KEY is set)
  - AnthropicLLM
  - OllamaLLM
  - StubLLM              (deterministic, no network — used in tests and when
                          no key is configured)

A *stub* LLM produces rule-based responses derived from the prompt so the
system is fully demonstrable offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any, Literal, Protocol

from app.core.config import get_settings
from app.core.errors import ProviderError
from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


class Message(dict):
    """Light wrapper around a chat message dict: {role, content}."""


class ToolSpec(dict):
    """Light wrapper around a tool spec dict.

    Schema loosely follows OpenAI's function-calling shape so the same object
    works for OpenAI, Anthropic, and the stub:

      {"name": str, "description": str,
       "parameters": {JSON schema}}
    """


class ToolCall(dict):
    """A tool call requested by the model.

      {"id": str, "name": str, "arguments": dict}
    """


class ChatResult(Protocol):
    """The non-streaming return value from ``chat``."""

    content: str
    tool_calls: list[ToolCall]
    raw: Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseLLM:
    name: str = "base"

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> ChatResult:  # type: ignore[override]
        raise NotImplementedError

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        # Default fallback: generate then chunk the string.
        result = self.chat(messages, tools=tools, temperature=temperature,
                           max_tokens=max_tokens, stop=stop)
        yield result.content

    def count_tokens(self, text: str) -> int:
        """Rough token estimate (~4 chars/token)."""
        return max(1, len(text) // 4)


class _ChatResultImpl:
    __slots__ = ("content", "tool_calls", "raw")

    def __init__(self, content: str, tool_calls: list[ToolCall] | None = None,
                 raw: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover
        return f"ChatResult(content={self.content[:80]!r}, tool_calls={self.tool_calls})"


# ---------------------------------------------------------------------------
# Stub (deterministic, offline)
# ---------------------------------------------------------------------------


class StubLLM(BaseLLM):
    """Deterministic offline test double; it never produces factual answers.

    It inspects the latest user message and any provided ``tools``:

      * If tools are provided and the question looks like a metric lookup
        ("eps", "revenue", etc. with a company name), it emits a tool call.
      * Otherwise it returns an explicit abstention. It does not inspect the
        evidence to synthesize an answer, so its output is not an answer-quality
        or production-model evaluation.

    The abstention is deterministic so tests remain stable.
    """

    name = "stub"

    _METRIC_PATTERNS = {
        "revenue": r"revenue|sales|turnover",
        "eps": r"\beps\b|earnings per share",
        "net_income": r"net income|net profit|laba bersih",
        "operating_margin": r"operating margin",
        "pe_ratio": r"\bp/?e\b|price.to.earnings|pe ratio",
        "ebitda": r"ebitda",
        "cash_flow": r"cash flow|arus kas",
        "debt": r"\bdebt\b|utang|total debt",
        "gross_margin": r"gross margin",
        "dividend": r"dividend",
    }

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, stop=None):
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), {}
        )
        q = (last_user.get("content") or "").strip()
        # Tool-calling path for metric lookups
        if tools:
            tool = self._pick_metric_tool(q, tools)
            if tool:
                args = self._extract_tool_args(q)
                tc = ToolCall(
                    id=f"call_{_hash(q)[:8]}",
                    name=tool["name"],
                    arguments=args,
                )
                return _ChatResultImpl(content="", tool_calls=[tc])

        # This provider is a test double, not a financial answer generator.
        # Never extract numbers from prompt wrappers or emit placeholder
        # citations that could be mistaken for grounded output.
        answer = (
            "Offline test stub: no financial answer was generated. "
            "Use a configured production model to evaluate answer quality."
        )

        return _ChatResultImpl(content=answer, tool_calls=[])

    # -- helpers --

    @staticmethod
    def _pick_metric_tool(q: str, tools: list[ToolSpec]) -> ToolSpec | None:
        ql = q.lower()
        for tool in tools:
            name = tool.get("name", "").lower()
            if "metric" in name or "financial" in name:
                # If the question names any metric, fire the tool.
                if any(
                    re.search(pat, ql, re.IGNORECASE)
                    for pat in StubLLM._METRIC_PATTERNS.values()
                ):
                    return tool
        return None

    @staticmethod
    def _extract_tool_args(q: str) -> dict:
        ql = q.lower()
        company = "Unknown"
        # crude company extraction
        for c in ("Apple", "Microsoft", "NVIDIA", "Tesla", "Google", "Alphabet",
                  "Meta", "Amazon"):
            if c.lower() in ql:
                company = c
                break
        metric = "revenue"
        for key, pat in StubLLM._METRIC_PATTERNS.items():
            if re.search(pat, ql, re.IGNORECASE):
                metric = key
                break
        year = 2025
        m = re.search(r"\b(20\d{2})\b", q)
        if m:
            year = int(m.group(1))
        quarter = ""
        qm = re.search(r"\bq([1-4])\b", ql)
        if qm:
            quarter = f"Q{qm.group(1)}"
        return {"company": company, "metric": metric, "fiscal_year": year,
                "quarter": quarter}

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIChatLLM(BaseLLM):
    name = "openai"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None) -> None:
        from app.core.errors import ProviderError

        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ProviderError("openai package not installed") from e
        settings = get_settings()
        key = settings.openai_api_key.get_secret_value() if api_key is None else api_key
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key,
                              **({"base_url": base_url} if base_url else {}))
        self._model = model or settings.openai_chat_model

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, stop=None):
        return self._chat_with(self._model, messages, tools=tools,
                               temperature=temperature, max_tokens=max_tokens, stop=stop)

    def _chat_with(self, model: str, messages, *, tools=None, temperature=0.2,
                   max_tokens=None, stop=None):
        from app.core.errors import ProviderError, ProviderTransientError

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [_clean_msg(m) for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("rate_limit", "timeout", "temporarily")):
                raise ProviderTransientError(f"OpenAI transient error: {e}") from e
            raise ProviderError(f"OpenAI error: {e}") from e

        choice = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        if getattr(choice, "tool_calls", None):
            for tc in choice.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                            arguments=args))
        return _ChatResultImpl(content=choice.content or "", tool_calls=tool_calls,
                               raw=resp)

    def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None,
               stop=None):
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [_clean_msg(m) for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece


def _clean_msg(m: Message) -> dict:
    """Convert our Message dict to OpenAI's expected shape.

    Handles three special cases:
      - role="tool": needs tool_call_id + content
      - role="assistant" with _tool_calls: emit tool_calls array
      - role="assistant" plain: content only
    """
    role = m.get("role", "user")
    content = m.get("content", "")
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.get("tool_call_id", ""),
            "content": str(content),
        }
    if role == "assistant" and m.get("_tool_calls"):
        return {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _safe_json(tc["arguments"]),
                    },
                }
                for tc in m["_tool_calls"]
            ],
        }
    if isinstance(content, list):
        return {"role": role, "content": content}
    return {"role": role, "content": str(content)}


def _safe_json(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return "{}"


def _to_openai_tool(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
        },
    }


# ---------------------------------------------------------------------------
# Anthropic & Ollama (thin, lazy)
# ---------------------------------------------------------------------------


class OmnirouteLLM(OpenAIChatLLM):
    """The self-hosted OpenAI-compatible gateway.

    Same wire protocol, different endpoint, key and model id, so only the
    constructor differs.
    """

    name = "omniroute"

    def __init__(self) -> None:
        from app.core.errors import ProviderError

        settings = get_settings()
        key = settings.omniroute_api_key.get_secret_value()
        if not key:
            raise ProviderError("LLM_API_KEY / OMNIROUTE_API_KEY is not set")
        super().__init__(api_key=key, base_url=settings.omniroute_base_url,
                         model=settings.chat_model)

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, stop=None):
        from app.core.errors import ProviderError, ProviderTransientError

        settings = get_settings()
        # A gateway model id can be listed and still be upstream-dead, and one
        # failed call puts that model on a ~2 minute credential cooldown. The
        # pinned chain is the operator's statement of what else is acceptable,
        # so use it instead of failing the request on the first dead id.
        chain: list[str] = []
        for model in [settings.chat_model, *settings.fallback_chain_list]:
            if model and model not in chain:
                chain.append(model)

        last_error: Exception | None = None
        for model in chain:
            try:
                return self._chat_with(model, messages, tools=tools,
                                       temperature=temperature, max_tokens=max_tokens,
                                       stop=stop)
            except (ProviderError, ProviderTransientError) as e:
                last_error = e
                log.warning("gateway model %s failed: %s", model, str(e)[:200])
        raise ProviderError(
            f"All {len(chain)} pinned gateway models failed; last error: {last_error}"
        ) from last_error


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ProviderError("anthropic package not installed") from e
        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value()
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = settings.anthropic_chat_model

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=2048, stop=None):
        from app.core.errors import ProviderError

        sys_msgs = [m["content"] for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        try:
            resp = self._client.messages.create(
                model=self._model,
                system="\n\n".join(sys_msgs) if sys_msgs else
                "You are a helpful investment research analyst.",
                messages=[_clean_msg(m) for m in rest],
                temperature=temperature,
                max_tokens=max_tokens or 2048,
            )
        except Exception as e:
            raise ProviderError(f"Anthropic error: {e}") from e
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _ChatResultImpl(content=text, tool_calls=[], raw=resp)


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self) -> None:
        import httpx

        settings = get_settings()
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_chat_model
        self._num_ctx = settings.ollama_num_ctx
        self._http = httpx.Client(timeout=120.0)

    def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None, stop=None):
        from app.core.errors import ProviderError

        options: dict = {"temperature": temperature}
        if self._num_ctx > 0:
            # Without this Ollama allocates the model's default context and the
            # load fails outright on a small GPU (see Settings.ollama_num_ctx).
            options["num_ctx"] = self._num_ctx
        payload = {
            "model": self._model,
            "messages": [_clean_msg(m) for m in messages],
            "stream": False,
            "options": options,
        }
        try:
            r = self._http.post(f"{self._base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}") from e
        return _ChatResultImpl(content=data.get("message", {}).get("content", ""),
                               tool_calls=[], raw=data)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_llm_singleton: BaseLLM | None = None


def get_llm() -> BaseLLM:
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton

    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key.get_secret_value():
            log.warning("LLM_PROVIDER=openai but OPENAI_API_KEY unset; "
                        "falling back to stub LLM")
            _llm_singleton = StubLLM()
            return _llm_singleton
        _llm_singleton = OpenAIChatLLM()
    elif provider == "anthropic":
        if not settings.anthropic_api_key.get_secret_value():
            log.warning("LLM_PROVIDER=anthropic but key unset; using stub")
            _llm_singleton = StubLLM()
        else:
            _llm_singleton = AnthropicLLM()
    elif provider == "ollama":
        _llm_singleton = OllamaLLM()
    elif provider == "omniroute":
        if not settings.omniroute_api_key.get_secret_value():
            log.warning("LLM_PROVIDER=omniroute but LLM_API_KEY/OMNIROUTE_API_KEY "
                        "unset; falling back to stub LLM")
            _llm_singleton = StubLLM()
        else:
            _llm_singleton = OmnirouteLLM()
    else:
        _llm_singleton = StubLLM()
    return _llm_singleton


def reset_llm() -> None:
    """Test helper."""
    global _llm_singleton
    _llm_singleton = None


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

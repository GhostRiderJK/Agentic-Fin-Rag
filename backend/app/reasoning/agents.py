"""Agents: small, single-role classes used by the orchestrator.

Each agent is just a function with a clear contract, so they are easy to
reason about and easy to test in isolation. We don't ship a heavyweight
agent framework — the orchestrator wires them in sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.providers.llm import BaseLLM, Message
from app.retrieval.hybrid import RetrievalResult
from app.retrieval.query_understanding import QueryPlan

SYSTEM_PROMPT = """\
You are an expert equity research analyst assistant. You answer questions
strictly from the provided source context and structured financial data.

Rules:
1. Base every numeric claim on the provided retrieved context OR on tool-call
   results. Never invent numbers.
2. Quote source citations as [1], [2], ... mapping to the provided sources.
3. The retrieved context under "Retrieved context" below IS your primary
   source. If the context contains information relevant to the question,
   ANSWER from it directly — do NOT say "I don't have enough information"
   just because a tool returned no data. Tool calls are only for NUMERIC
   metrics (revenue, EPS, etc.); textual content (risk factors, strategy,
   business overview) comes from the retrieved context.
4. Only say "I don't have enough information" if the retrieved context
   genuinely does not contain anything relevant to the question.
5. For numeric facts (revenue, EPS, margins), call get_financial_metric.
   Do NOT call get_financial_metric for textual topics like risk_factors,
   strategy, outlook — those are not numeric metrics.
6. Be concise. Use bullet points when listing drivers, risks, or comparisons.
7. When comparing companies, organize the answer with one section per
   company so the comparison is clear.
"""


@dataclass
class DraftAnswer:
    text: str
    tool_results: list[dict]
    context_used: int


def build_messages(
    *,
    question: str,
    context: str,
    citations: list[dict],
    history: list[tuple[str, str]] | None = None,
    tool_hint: str = "",
) -> list[Message]:
    msgs: list[Message] = [Message(role="system", content=SYSTEM_PROMPT.strip())]
    if history:
        for u, a in history[-6:]:  # last 6 turns
            msgs.append(Message(role="user", content=u))
            msgs.append(Message(role="assistant", content=a))
    ctx_block = f"Retrieved context (with citation markers):\n\n{context}" if context else \
        "No retrieved context is available for this question."
    if citations:
        src = "\n".join(
            f"[{c['index']}] {c['company'] or 'Unknown'} {c['year'] or ''} "
            f"{c['quarter'] or ''} {c['doc_type']} p.{c['page']}"
            for c in citations
        )
        ctx_block += f"\n\nSources:\n{src}"
    if tool_hint:
        ctx_block += f"\n\n{tool_hint}"
    msgs.append(Message(role="system", content=ctx_block))
    msgs.append(Message(role="user", content=question))
    return msgs


def run_tool_loop(
    llm: BaseLLM,
    messages: list[Message],
    tools: list[Any],
    dispatch: Any,
    *,
    max_iterations: int = 3,
) -> tuple[str, list[dict]]:
    """Loop the LLM through tool-calling until it produces a final answer
    or hits the iteration cap. Returns (final_text, tool_results)."""
    tool_results: list[dict] = []
    msgs = list(messages)
    for _ in range(max_iterations):
        # Disable tools after the first call so the model must synthesize.
        result = llm.chat(msgs, tools=tools if not tool_results else None,
                          temperature=0.1)
        if not result.tool_calls:
            return result.content, tool_results
        # Execute each tool call and append a tool message
        for tc in result.tool_calls:
            try:
                out = dispatch(tc["name"], tc["arguments"])
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            tool_results.append({
                "name": tc["name"], "arguments": tc["arguments"], "result": out,
            })
        # IMPORTANT: OpenAI requires the assistant message that PRECEDES
        # role="tool" messages to carry its tool_calls array. We attach the
        # raw tool_calls so the LLM provider can render them correctly. We
        # pass it as a special "_tool_calls" key that the OpenAI provider
        # knows how to translate.
        msgs.append(Message(
            role="assistant",
            content=result.content or "",
            _tool_calls=result.tool_calls,  # type: ignore[call-arg]
        ))
        for tc, tr in zip(result.tool_calls, tool_results[-len(result.tool_calls):]):
            msgs.append(
                Message(role="tool",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        content=f"Tool {tc['name']} returned: "
                                + _truncate_json(tr["result"]))
            )
    # Final synthesis pass with no tools available
    final = llm.chat(msgs, tools=None, temperature=0.1)
    return final.content, tool_results


def _truncate_json(obj: Any, limit: int = 2000) -> str:
    """Render ``obj`` as JSON, truncating to ``limit`` chars so we don't blow
    the LLM context window when a SQL tool returns a huge table."""
    import json

    s = json.dumps(obj, default=str)
    if len(s) <= limit:
        return s
    # Try to truncate a 'results' list gracefully
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        head = obj["results"][:20]
        kept = {**obj, "results": head, "_truncated": True,
                "_total_results": len(obj["results"])}
        s2 = json.dumps(kept, default=str)
        if len(s2) <= limit:
            return s2
    return s[:limit] + "…(truncated)"

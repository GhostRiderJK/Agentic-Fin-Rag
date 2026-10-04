"""Conversation memory + query rewriting (PRD v2 §9).

Loads the last N turns from a conversation, exposes them to the
orchestrator, and provides an LLM-backed query-rewriter for pronoun
resolution ("How about Microsoft?" → "Microsoft — <previous question>").

Falls back to the deterministic rewriter in
``app.retrieval.query_understanding.rewrite_query_with_history`` when no LLM
is configured.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.db.models import Conversation, Message, MessageRole
from app.providers.llm import BaseLLM, Message as LLMMessage, get_llm
from app.retrieval.query_understanding import rewrite_query_with_history


def load_history(db: Session, conversation_id: str, *, limit: int = 12) -> list[tuple[str, str]]:
    """Return the last ``limit`` (user, assistant) turn pairs."""
    msgs = list(db.scalars(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc()).limit(limit * 2)
    ).all())[::-1]
    pairs: list[tuple[str, str]] = []
    last_user = None
    for m in msgs:
        if m.role == MessageRole.user:
            last_user = m.content
        elif m.role == MessageRole.assistant and last_user is not None:
            pairs.append((last_user, m.content))
            last_user = None
    return pairs


def rewrite(
    question: str,
    history: list[tuple[str, str]],
    *,
    llm: BaseLLM | None = None,
) -> str:
    """LLM-backed rewriter if available, deterministic fallback otherwise."""
    llm = llm or get_llm()
    if llm.name in {"stub", "base"}:
        return rewrite_query_with_history(question, history)

    if not history:
        return question
    last_user, _ = history[-1]
    prompt = (
        "Rewrite the user's latest message into a self-contained question "
        "that resolves pronouns and references using conversation history. "
        "Output ONLY the rewritten question, nothing else.\n\n"
        f"Previous question: {last_user}\n"
        f"Latest message: {question}\n"
        "Rewritten question:"
    )
    try:
        resp = llm.chat([LLMMessage(role="user", content=prompt)], temperature=0.0,
                        max_tokens=120)
        out = (resp.content or "").strip().strip('"').strip()
        return out or question
    except Exception:
        return rewrite_query_with_history(question, history)


def ensure_conversation(
    db: Session, *, conversation_id: str | None, workspace_id: str, user_id: str,
    title: str | None = None,
) -> Conversation:
    if conversation_id:
        c = db.get(Conversation, conversation_id)
        if not c or c.workspace_id != workspace_id:
            raise NotFoundError("conversation not found")
        return c
    c = Conversation(
        workspace_id=workspace_id, user_id=user_id,
        title=title or "New conversation",
    )
    db.add(c)
    db.flush()
    return c


def append_message(
    db: Session, *, conversation_id: str, role: MessageRole, content: str,
    citations: list | None = None, trace: dict | None = None,
) -> Message:
    m = Message(
        conversation_id=conversation_id, role=role, content=content,
        citations=citations or [], trace=trace or {},
    )
    db.add(m)
    db.flush()
    return m

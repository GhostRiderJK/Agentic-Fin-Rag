"""Chat router: synchronous + streaming answers, conversation history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import workspace_dep
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    ConversationOut,
    FeedbackIn,
    MessageOut,
)
from app.conversation import (
    append_message,
    ensure_conversation,
    load_history,
)
from app.core.security import CurrentUser
from app.db.base import get_db
from app.db.models import Conversation, FeedbackLog, Message, MessageRole, Role, Workspace
from app.db.mongodb.tenant_context import TenantContext
from app.reasoning.orchestrator import AnswerRequest, Orchestrator

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    if payload.include_evidence_trace and user.role != Role.admin:
        from app.core.errors import ForbiddenError

        raise ForbiddenError("evidence trace requires admin role")
    # Validate workspace access
    ws = db.get(Workspace, payload.workspace_id)
    if not ws or ws.tenant_id != user.tenant_id:
        from app.core.errors import NotFoundError

        raise NotFoundError("workspace not found")
    ctx = TenantContext(tenant_id=user.tenant_id, user_id=user.id,
                        workspace_id=ws.id)

    conv = ensure_conversation(
        db, conversation_id=payload.conversation_id,
        workspace_id=payload.workspace_id, user_id=user.id,
        title=payload.question[:80],
    )
    history = load_history(db, conv.id)
    append_message(db, conversation_id=conv.id, role=MessageRole.user,
                   content=payload.question)

    orch = Orchestrator(ctx)
    resp = orch.answer(AnswerRequest(
        question=payload.question, workspace_id=payload.workspace_id,
        conversation_id=conv.id, history=history, mode=payload.mode,
        include_evidence_trace=payload.include_evidence_trace,
    ))
    msg = append_message(
        db, conversation_id=conv.id, role=MessageRole.assistant,
        content=resp.answer, citations=resp.citations,
        trace={"plan": resp.plan, "tool_results": resp.tool_results,
               "verification": resp.verification,
               "generation": resp.generation},
    )
    db.commit()

    return ChatResponse(
        answer=resp.answer,
        citations=[CitationOut(**c) for c in resp.citations],
        plan=resp.plan, tool_results=resp.tool_results,
        verification=resp.verification, retrieval=resp.retrieval,
        generation=resp.generation,
        latency_ms=resp.latency_ms, notes=resp.notes,
        message_id=msg.id, conversation_id=conv.id,
    )


@router.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    """Token-streaming chat. Falls back to single-chunk if LLM doesn't stream."""
    if payload.include_evidence_trace and user.role != Role.admin:
        from app.core.errors import ForbiddenError

        raise ForbiddenError("evidence trace requires admin role")
    ws = db.get(Workspace, payload.workspace_id)
    if not ws or ws.tenant_id != user.tenant_id:
        from app.core.errors import NotFoundError

        raise NotFoundError("workspace not found")
    ctx = TenantContext(tenant_id=user.tenant_id, user_id=user.id,
                        workspace_id=ws.id)

    conv = ensure_conversation(
        db, conversation_id=payload.conversation_id,
        workspace_id=payload.workspace_id, user_id=user.id,
        title=payload.question[:80],
    )
    history = load_history(db, conv.id)
    append_message(db, conversation_id=conv.id, role=MessageRole.user,
                   content=payload.question)

    # We compute the full response once (retrieval + verification), then
    # stream the final text for the UX. This keeps citations/verification
    # accurate while still giving a streaming feel for the LLM output.
    orch = Orchestrator(ctx)
    resp = orch.answer(AnswerRequest(
        question=payload.question, workspace_id=payload.workspace_id,
        conversation_id=conv.id, history=history, mode=payload.mode,
        include_evidence_trace=payload.include_evidence_trace,
    ))
    append_message(
        db, conversation_id=conv.id, role=MessageRole.assistant,
        content=resp.answer, citations=resp.citations,
        trace={"plan": resp.plan, "verification": resp.verification,
               "generation": resp.generation},
    )
    db.commit()

    import json

    def gen():
        # First send metadata, then stream the answer
        meta = {
            "type": "meta",
            "citations": resp.citations,
            "plan": resp.plan,
            "verification": resp.verification,
            "generation": resp.generation,
            "retrieval": resp.retrieval,
            "conversation_id": conv.id,
            "latency_ms": resp.latency_ms,
        }
        yield f"data: {json.dumps(meta)}\n\n"
        for chunk in _chunks(resp.answer):
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _chunks(text: str, size: int = 8) -> list[str]:
    return [text[i: i + size] for i in range(0, len(text), size)] or [" "]


# ---- Conversations -----------------------------------------------------------


@router.get("/workspaces/{workspace_id}/conversations",
            response_model=list[ConversationOut])
def list_conversations(
    ws: Annotated[Workspace, Depends(workspace_dep)],
    db: Annotated[Session, Depends(get_db)],
):
    return list(db.scalars(
        select(Conversation).where(Conversation.workspace_id == ws.id)
        .order_by(Conversation.updated_at.desc())
    ).all())


@router.get("/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
            response_model=list[MessageOut])
def list_messages(
    ws: Annotated[Workspace, Depends(workspace_dep)],
    db: Annotated[Session, Depends(get_db)],
    conversation_id: str,
):
    conv = db.get(Conversation, conversation_id)
    if not conv or conv.workspace_id != ws.id:
        from app.core.errors import NotFoundError

        raise NotFoundError("conversation not found")
    return list(db.scalars(
        select(Message).where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
    ).all())


@router.post("/messages/{message_id}/feedback")
def feedback(
    message_id: str,
    payload: FeedbackIn,
    db: Annotated[Session, Depends(get_db)],
    user: CurrentUser,
):
    msg = db.get(Message, message_id)
    conv = db.get(Conversation, msg.conversation_id) if msg else None
    # A rating is a write on another tenant's answer unless the conversation is
    # scoped, so ownership is checked along the same chain every conversation
    # route uses: message -> conversation -> workspace -> tenant.
    ws = db.get(Workspace, conv.workspace_id) if conv else None
    if not msg or not ws or ws.tenant_id != user.tenant_id:
        from app.core.errors import NotFoundError

        raise NotFoundError("message not found")
    msg.feedback = payload.rating
    db.add(FeedbackLog(message_id=message_id, user_id=user.id,
                       rating=payload.rating, note=payload.note))
    db.commit()
    return {"ok": True, "rating": payload.rating}

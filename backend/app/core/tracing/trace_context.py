"""Request tracing context management."""

import uuid
from contextvars import ContextVar
from typing import Optional

# Context variables for tracing
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
tenant_id_ctx: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the request ID in context and return it."""
    if request_id is None:
        request_id = generate_request_id()
    request_id_ctx.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_ctx.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """Set the trace ID in context and return it."""
    if trace_id is None:
        trace_id = generate_trace_id()
    trace_id_ctx.set(trace_id)
    return trace_id


def get_trace_id() -> Optional[str]:
    """Get the current trace ID from context."""
    return trace_id_ctx.get()


def set_tenant_id(tenant_id: Optional[str]) -> None:
    """Set the tenant ID in context."""
    tenant_id_ctx.set(tenant_id)


def get_tenant_id() -> Optional[str]:
    """Get the current tenant ID from context."""
    return tenant_id_ctx.get()


def set_user_id(user_id: Optional[str]) -> None:
    """Set the user ID in context."""
    user_id_ctx.set(user_id)


def get_user_id() -> Optional[str]:
    """Get the current user ID from context."""
    return user_id_ctx.get()


def get_trace_context() -> dict:
    """Get all trace context as a dictionary."""
    return {
        "request_id": get_request_id(),
        "trace_id": get_trace_id(),
        "tenant_id": get_tenant_id(),
        "user_id": get_user_id(),
    }


def clear_trace_context() -> None:
    """Clear all trace context."""
    request_id_ctx.set(None)
    trace_id_ctx.set(None)
    tenant_id_ctx.set(None)
    user_id_ctx.set(None)
"""Typed domain exceptions and FastAPI exception handlers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base application error. status_code becomes the HTTP status."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class ProviderError(AppError):
    """Error calling an external provider (LLM/embeddings/etc)."""

    status_code = 502
    code = "provider_error"

    # ``provider`` is optional because every raise site in app/providers
    # currently omits it; as a required positional it turned any provider
    # failure into TypeError instead of a 502.
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, **details: Any) -> None:
        super().__init__(message, provider=provider, model=model, **details)


class ProviderTransientError(ProviderError):
    """Retryable provider error."""

    code = "provider_transient"
    status_code = 503  # Service Unavailable


class ProviderRateLimitError(ProviderError):
    """Provider rate limit error."""

    code = "provider_rate_limit"
    status_code = 429  # Too Many Requests


class ProviderAuthenticationError(ProviderError):
    """Provider authentication error."""

    code = "provider_auth_error"
    status_code = 401  # Unauthorized


class ConfigurationError(AppError):
    """Configuration validation error."""

    code = "configuration_error"
    status_code = 500  # Internal Server Error


class TenantIsolationError(AppError):
    """Tenant isolation violation error."""

    code = "tenant_isolation_violation"
    status_code = 403  # Forbidden


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[\s_-]?key|secret[\s_-]?key|access[\s_-]?token|refresh[\s_-]?token|"
    r"authorization|bearer|password|token|secret)\b(\s*[:=]\s*)(\S[^\n,;]*)")
_BEARING_TOKEN_RE = re.compile(r"\b(?:sk|pk|ghp|xox[baprs]-[a-z])[-_][A-Za-z0-9_-]{8,}")


def without_secrets(text: str) -> str:
    """Scrub credential material out of an error message before it is returned.

    Provider failures quote the upstream response, and a gateway that echoes the
    request it received can put the caller's key into that text. That text is
    then handed to whoever the error is shown to, so the outbound path scrubs it
    instead of relying on providers to have been written carefully. Anything that
    *stores* an error for later display (job rows, eval item rows) has to scrub
    at write time too — those readers do not go through the exception handlers.
    """
    scrubbed = _SECRET_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
    return _BEARING_TOKEN_RE.sub("[redacted]", scrubbed)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": without_secrets(exc.message),
                "details": {k: without_secrets(v) if isinstance(v, str) else v
                             for k, v in exc.details.items()},
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "details": {"exception_type": type(exc).__name__} if str(exc) else {},
            },
        )

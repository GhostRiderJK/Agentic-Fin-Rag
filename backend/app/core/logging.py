"""Structured logging setup with JSON formatting and tracing."""

import logging
import sys

from app.core.config import get_settings
from app.core.logging_structured.json_formatter import JSONFormatter, setup_json_logging
from app.core.tracing.trace_context import clear_trace_context, get_trace_context


def configure_logging() -> None:
    """Configure application logging with JSON formatting."""
    settings = get_settings()
    level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    # Set up JSON logging
    setup_json_logging()

    # Quiet noisy libs
    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class LoggingMiddleware:
    """Middleware to add request ID and trace ID to logs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request and trace IDs
        from app.core.tracing.trace_context import set_request_id, set_trace_id
        request_id = set_request_id()
        trace_id = set_trace_id()

        # Add context to all log records in this request
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            # Add tracing context to log record
            record.request_id = request_id
            record.trace_id = trace_id
            return record

        logging.setLogRecordFactory(record_factory)

        try:
            await self.app(scope, receive, send)
        finally:
            # Restore old factory and clear context
            logging.setLogRecordFactory(old_factory)
            from app.core.tracing.trace_context import clear_trace_context
            clear_trace_context()
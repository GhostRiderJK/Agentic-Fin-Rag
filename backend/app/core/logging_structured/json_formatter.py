"""JSON log formatter for structured logging."""

import json
import logging
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""

    def __init__(self, *, datetime_format: str = "%Y-%m-%dT%H:%M:%S"):
        super().__init__()
        self.datetime_format = datetime_format

    def format(self, record: logging.LogRecord) -> str:
        """Format the record as a JSON string."""
        log_record: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).strftime(
                self.datetime_format
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "trace_id"):
            log_record["trace_id"] = record.trace_id
        if hasattr(record, "tenant_id"):
            log_record["tenant_id"] = record.tenant_id
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id
        if hasattr(record, "operation"):
            log_record["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_record["duration_ms"] = record.duration_ms
        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code

        # Add any other extra attributes
        for key, value in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "timestamp",
                "level",
                "logger",
                "message",
                "module",
                "function",
                "line",
                "request_id",
                "trace_id",
                "tenant_id",
                "user_id",
                "operation",
                "duration_ms",
                "status_code",
            }:
                if not key.startswith("_") and key not in log_record:
                    log_record[key] = value

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def setup_json_logging() -> None:
    """Set up JSON logging for the application."""
    import logging
    import sys

    from app.core.config import get_settings

    settings = get_settings()
    level = getattr(logging, settings.app_log_level.upper(), logging.INFO)

    # Create JSON formatter
    json_formatter = JSONFormatter()

    # Configure root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(json_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]  # Replace existing handlers

    # Quiet noisy libraries
    for noisy in ("httpx", "httpcore", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
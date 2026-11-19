"""Process-wide logging setup (stdlib `logging`, configured once at the entry point).

Modules only do `logger = logging.getLogger(__name__)`; they never configure handlers.

Two formats:
- text (default): human-readable, with the request id of the question being handled.
- json: one JSON object per line (timestamp, severity, logger, message, request_id and the
  structured fields passed via `extra=`), ready for Cloud Logging / any log pipeline.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

from pool_assistant.observability import RequestIdFilter

TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(request_id)s | %(name)s | %(message)s"
_HANDLER_NAME = "pool_assistant"

# Third-party loggers that are too chatty at INFO (HTTP requests, SDK internals).
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "google_genai", "urllib3")

# Attributes every LogRecord has: anything else came from `extra=` and is structured data.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message", "asctime", "request_id", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,     # the key Cloud Logging maps to severity
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        payload |= {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", fmt: Literal["text", "json"] = "text") -> None:
    """Configure the root logger. Idempotent: safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level)

    handler = next((h for h in root.handlers if h.get_name() == _HANDLER_NAME), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

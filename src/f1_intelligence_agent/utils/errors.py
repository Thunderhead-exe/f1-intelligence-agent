"""Error-message sanitization helpers."""

from __future__ import annotations

import re


def sanitize_error_message(message: object) -> str:
    """Remove credential-looking fragments from provider error messages."""

    text = str(message)
    text = re.sub(
        r"Incorrect API key provided: [^'.\s]+",
        "Incorrect API key provided: [redacted]",
        text,
    )
    text = re.sub(
        r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)([^'\"\s,}]+)",
        r"\1[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_\-]{12,}\b", "[redacted-openai-key]", text)
    return text


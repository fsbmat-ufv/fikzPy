"""Small diagnostic helpers for pipeline tracing."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path


LOGGER = logging.getLogger("fikzpy.pipeline")


def log_event(section: str, message: str) -> None:
    """Emit one concise diagnostic line."""
    line = f"[{section}] {message}"
    LOGGER.info(line)
    print(line)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

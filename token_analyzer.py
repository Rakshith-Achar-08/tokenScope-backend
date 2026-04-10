"""Token analysis and ingestion utilities for TokenScope."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import tiktoken

# Set cache directory to writable location to avoid permission errors
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"

_ENCODING = tiktoken.get_encoding("cl100k_base")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TokenAnalysis:
    """Normalized text and local token count."""

    sanitized_text: str
    token_count: int


def sanitize_text(raw_text: str) -> str:
    """Normalize input text for deterministic counting and scoring."""
    if raw_text is None:
        return ""
    sanitized = raw_text.strip().replace("\x00", "")
    sanitized = _WHITESPACE_RE.sub(" ", sanitized)
    return sanitized


def analyze_text(raw_text: str) -> TokenAnalysis:
    """Sanitize input and count tokens locally with cl100k_base."""
    sanitized = sanitize_text(raw_text)
    token_count = len(_ENCODING.encode(sanitized))
    return TokenAnalysis(sanitized_text=sanitized, token_count=token_count)

"""Prompt trimming logic driven by word-level importance scores."""

from __future__ import annotations

import math
import re
from typing import Any

_TOKEN_RE = re.compile(r"\b[\w'-]+\b|[.,;:!?]")
_WORD_RE = re.compile(r"\b[\w'-]+\b")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile
    floor_idx = math.floor(k)
    ceil_idx = math.ceil(k)
    if floor_idx == ceil_idx:
        return sorted_values[floor_idx]
    lower = sorted_values[floor_idx] * (ceil_idx - k)
    upper = sorted_values[ceil_idx] * (k - floor_idx)
    return lower + upper


def generate_trimmed_prompt(text: str, word_scores: list[dict[str, Any]]) -> str:
    """Trim low-importance words while preserving useful punctuation structure."""
    if not text.strip() or not word_scores:
        return ""

    meaningful_scores = [s["score"] for s in word_scores if s["score"] > 0.0]
    # --- SAFEGUARD INJECTED HERE ---
    if len(word_scores) < 8:
        cutoff = 0.0
    else:
        cutoff = _percentile(meaningful_scores, 0.20) if meaningful_scores else 0.0
    # -------------------------------

    keep_word_by_index: dict[int, bool] = {}
    for entry in word_scores:
        keep_word_by_index[entry["index"]] = (
            not entry.get("is_noise", False) and entry["score"] > 0.0 and entry["score"] >= cutoff
        )

    tokens = _TOKEN_RE.findall(text)
    rendered_tokens: list[str] = []
    word_ptr = 0

    for i, token in enumerate(tokens):
        if _WORD_RE.fullmatch(token):
            if keep_word_by_index.get(word_ptr, False):
                rendered_tokens.append(token)
            word_ptr += 1
            continue

        # Keep punctuation only when it separates/ends kept words.
        prev_token_is_kept_word = bool(rendered_tokens and _WORD_RE.fullmatch(rendered_tokens[-1]))
        next_word_kept = False
        lookahead_ptr = word_ptr
        for future in tokens[i + 1 :]:
            if _WORD_RE.fullmatch(future):
                next_word_kept = keep_word_by_index.get(lookahead_ptr, False)
                break
            if future in ".!?":
                break
        if prev_token_is_kept_word or next_word_kept:
            rendered_tokens.append(token)

    # Clean up spaces around punctuation.
    compact = " ".join(rendered_tokens)
    compact = re.sub(r"\s+([.,;:!?])", r"\1", compact)
    compact = re.sub(r"([.,;:!?])(\w)", r"\1 \2", compact)
    return compact.strip()

"""Hybrid YAKE + fallback importance scoring engine."""

from __future__ import annotations

import math
import re
from typing import Any

import yake

NOISE_WORDS = {
    "please",
    "hey",
    "act",
    "expert",
    "detailed",
    "explanation",
}

_WORD_RE = re.compile(r"\b[\w'-]+\b")


def _extract_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _normalize_scores(raw_scores: list[float]) -> list[float]:
    if not raw_scores:
        return []
    max_score = max(raw_scores)
    min_score = min(raw_scores)
    if math.isclose(max_score, min_score):
        return [1.0 if score > 0 else 0.0 for score in raw_scores]
    return [(score - min_score) / (max_score - min_score) for score in raw_scores]


def calculate_importance_scores(text: str) -> list[dict[str, Any]]:
    """Return per-word importance scores using YAKE, with deterministic fallback."""
    words = _extract_words(text)
    if not words:
        return []

    keyword_extractor = yake.KeywordExtractor(
        lan="en", n=2, dedupLim=0.9, dedupFunc="seqm", windowsSize=1, top=60
    )
    yake_keywords = keyword_extractor.extract_keywords(text)

    # YAKE returns lower score for more importance. Invert that so higher is better.
    per_word_raw_scores: dict[str, float] = {}
    for keyphrase, yake_score in yake_keywords:
        inverted_score = 1.0 / (yake_score + 1e-9)
        for token in _extract_words(keyphrase):
            lowered = token.lower()
            per_word_raw_scores[lowered] = max(
                per_word_raw_scores.get(lowered, 0.0), inverted_score
            )

    has_meaningful_scores = any(score > 0 for score in per_word_raw_scores.values())
    if not has_meaningful_scores:
        # Simple TF-IDF-like fallback heuristic: longer words are treated as more specific.
        per_word_raw_scores = {
            word.lower(): min(len(word) / 12.0, 1.0)
            for word in words
            if word.lower() not in NOISE_WORDS
        }

    raw_by_position: list[float] = []
    for word in words:
        lowered = word.lower()
        if lowered in NOISE_WORDS:
            raw_by_position.append(0.0)
            continue
        raw_by_position.append(per_word_raw_scores.get(lowered, min(len(word) / 12.0, 1.0)))

    normalized_by_position = _normalize_scores(raw_by_position)

    return [
        {
            "index": idx,
            "word": word,
            "score": float(normalized_by_position[idx]),
            "is_noise": word.lower() in NOISE_WORDS,
        }
        for idx, word in enumerate(words)
    ]

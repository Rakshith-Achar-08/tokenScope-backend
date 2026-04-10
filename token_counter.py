"""Token cost calculations and session-level tracking."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_COST_TABLE_USD_PER_1K_TOKENS = {
    "gemini-1.5-flash": 0.00035,
    "gpt-4o": 0.00500,
    "claude-3-haiku": 0.00025,
}

_SESSION_RUNNING_TOTAL_USD = 0.0


@dataclass(frozen=True)
class SessionCostResult:
    model_name: str
    original_cost_usd: float
    trimmed_cost_usd: float
    savings_usd: float
    savings_percent: float
    session_running_total_usd: float


def _calculate_cost(tokens: int, model_name: str) -> float:
    per_1k = MODEL_COST_TABLE_USD_PER_1K_TOKENS.get(
        model_name, MODEL_COST_TABLE_USD_PER_1K_TOKENS["gemini-1.5-flash"]
    )
    return (tokens / 1000.0) * per_1k


def calculate_session_costs(
    original_tokens: int, trimmed_tokens: int, model_name: str
) -> SessionCostResult:
    """Calculate per-request and running total savings in USD."""
    global _SESSION_RUNNING_TOTAL_USD

    original_cost = _calculate_cost(original_tokens, model_name)
    trimmed_cost = _calculate_cost(trimmed_tokens, model_name)
    savings = max(original_cost - trimmed_cost, 0.0)
    savings_percent = (savings / original_cost * 100.0) if original_cost > 0 else 0.0

    _SESSION_RUNNING_TOTAL_USD += trimmed_cost

    return SessionCostResult(
        model_name=model_name,
        original_cost_usd=round(original_cost, 10),
        trimmed_cost_usd=round(trimmed_cost, 10),
        savings_usd=round(savings, 10),
        savings_percent=round(savings_percent, 4),
        session_running_total_usd=round(_SESSION_RUNNING_TOTAL_USD, 10),
    )

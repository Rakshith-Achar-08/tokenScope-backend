"""FastAPI service entrypoint for TokenScope backend."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from typing import Any

from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from importance_engine import calculate_importance_scores
from prompt_trimmer import generate_trimmed_prompt
from token_analyzer import analyze_text
from token_counter import MODEL_COST_TABLE_USD_PER_1K_TOKENS, calculate_session_costs

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Configure optional Gemini client once on app startup."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        # Initializes SDK-level defaults for future model calls.
        genai.Client(api_key=api_key)
    yield


app = FastAPI(title="TokenScope Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _build_diff_preview(word_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meaningful = [row["score"] for row in word_scores if row["score"] > 0.0]
    cutoff = _percentile(meaningful, 0.20) if meaningful else 0.0
    preview: list[dict[str, Any]] = []
    for row in word_scores:
        kept = (not row.get("is_noise", False)) and row["score"] > 0.0 and row["score"] >= cutoff
        preview.append(
            {
                "index": row["index"],
                "word": row["word"],
                "score": row["score"],
                "status": "kept" if kept else "removed",
            }
        )
    return preview


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw user prompt")
    model: str = Field(default="gemini-1.5-flash", description="Target LLM model")


class AnalyzeResponse(BaseModel):
    heatmap_data: list[dict[str, Any]]
    cost_card: dict[str, Any]
    trimmed_prompt: str
    diff_preview: list[dict[str, Any]]
    session_history: dict[str, Any]

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if payload.model not in MODEL_COST_TABLE_USD_PER_1K_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{payload.model}'. Supported: {list(MODEL_COST_TABLE_USD_PER_1K_TOKENS)}",
        )

    original = analyze_text(payload.text)
    word_scores = calculate_importance_scores(original.sanitized_text)

    trimmed_prompt = generate_trimmed_prompt(original.sanitized_text, word_scores)
    trimmed = analyze_text(trimmed_prompt)

    cost = calculate_session_costs(
        original_tokens=original.token_count,
        trimmed_tokens=trimmed.token_count,
        model_name=payload.model,
    )
    diff_preview = _build_diff_preview(word_scores)

    return AnalyzeResponse(
        heatmap_data=word_scores,
        cost_card={
            "model": cost.model_name,
            "original_tokens": original.token_count,
            "trimmed_tokens": trimmed.token_count,
            "original_cost_usd": cost.original_cost_usd,
            "trimmed_cost_usd": cost.trimmed_cost_usd,
            "savings_usd": cost.savings_usd,
            "savings_percent": cost.savings_percent,
        },
        trimmed_prompt=trimmed_prompt,
        diff_preview=diff_preview,
        session_history={
            "running_total_usd": cost.session_running_total_usd,
            "message": "In-memory running total since server start.",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

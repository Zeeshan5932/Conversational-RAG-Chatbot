"""Groq chat model factory."""

from __future__ import annotations

from langchain_groq import ChatGroq

from app.config import settings
from app.utils.logger import logger


def get_llm() -> ChatGroq:
    """Return a configured ChatGroq instance."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    logger.debug("Initializing ChatGroq with model %s", settings.LLM_MODEL)
    return ChatGroq(
        model=settings.LLM_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        max_tokens=settings.MAX_TOKENS,
        temperature=0.2,
        max_retries=3,
    )


def get_gemini_llm() -> ChatGroq:
    """Backward-compatible alias for the Groq LLM factory."""
    return get_llm()
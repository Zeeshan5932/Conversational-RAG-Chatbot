"""Gemini embeddings factory."""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.utils.logger import logger


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a configured GoogleGenerativeAIEmbeddings instance."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    logger.debug("Initializing Gemini embeddings with model %s", settings.EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )


def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Backward-compatible alias for the embeddings factory."""
    return get_embeddings()
"""Gemini embeddings factory."""

from __future__ import annotations

from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.utils.logger import logger


def _normalize_model_name(model_name: str) -> str:
    """Return a clean Google model identifier without redundant prefixes."""
    return model_name.replace("models/", "").strip()


def _is_model_unavailable_error(exception: Exception) -> bool:
    """Detect Google embedding model lookup failures that can be retried."""
    error_text = str(exception).lower()
    return (
        "not found" in error_text
        or "not supported for embedcontent" in error_text
        or "model_not_supported" in error_text
        or "404" in error_text
    )


class ResilientGoogleEmbeddings(Embeddings):
    """Embeddings wrapper that falls back across known Google model aliases."""

    def __init__(self, model_names: Sequence[str], google_api_key: str):
        cleaned_model_names: list[str] = []
        clients: list[GoogleGenerativeAIEmbeddings] = []
        seen_models: set[str] = set()

        for model_name in model_names:
            clean_model_name = _normalize_model_name(model_name)
            if not clean_model_name or clean_model_name in seen_models:
                continue

            seen_models.add(clean_model_name)
            cleaned_model_names.append(clean_model_name)
            clients.append(
                GoogleGenerativeAIEmbeddings(
                    model=clean_model_name,
                    google_api_key=google_api_key,
                )
            )

        if not clients:
            raise ValueError("No valid Google embedding model candidates were configured.")

        self._model_names = cleaned_model_names
        self._clients = clients

    def _embed_with_fallback(self, method_name: str, *args):
        last_error: Exception | None = None

        for model_name, client in zip(self._model_names, self._clients, strict=True):
            try:
                return getattr(client, method_name)(*args)
            except Exception as exception:
                last_error = exception
                if not _is_model_unavailable_error(exception):
                    raise

                logger.warning(
                    "Google embedding model %s is unavailable; trying the next candidate.",
                    model_name,
                )

        message = (
            "All configured Google embedding models failed. Tried: "
            + ", ".join(self._model_names)
        )
        raise ValueError(message) from last_error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_fallback("embed_documents", texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_fallback("embed_query", text)


def get_embeddings() -> Embeddings:
    """Return a configured embeddings instance with Google model fallback."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    candidate_models = [
        settings.EMBEDDING_MODEL,
        "gemini-embedding-001",
        "embedding-001",
        "text-embedding-004",
        "models/gemini-embedding-001",
        "models/embedding-001",
        "models/text-embedding-004",
    ]

    logger.debug("Initializing Gemini embeddings with candidates: %s", candidate_models)
    return ResilientGoogleEmbeddings(candidate_models, settings.GOOGLE_API_KEY)


def get_embedding_function() -> Embeddings:
    """Backward-compatible alias for the embeddings factory."""
    return get_embeddings()
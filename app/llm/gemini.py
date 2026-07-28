from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.config import settings
from app.utils.logger import logger


def get_gemini_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Factory creating configured ChatGoogleGenerativeAI instance."""
    logger.debug(f"Initializing Gemini LLM with model: {settings.GEMINI_MODEL}")
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
    )


def get_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Factory creating configured GoogleGenerativeAIEmbeddings instance."""
    logger.debug(f"Initializing Gemini Embeddings with model: {settings.EMBEDDING_MODEL}")
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY
    )
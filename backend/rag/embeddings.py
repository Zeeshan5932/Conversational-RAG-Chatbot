from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.llm.gemini import get_gemini_embeddings


def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Returns configured Gemini embeddings instance."""
    return get_gemini_embeddings()
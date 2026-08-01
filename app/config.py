from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralized application configuration strictly managed by Pydantic."""

    # API Keys
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    TAVILY_API_KEY: Optional[str] = None
    FIRECRAWL_API_KEY: Optional[str] = None

    # Model Specifications
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "text-embedding-004"
    MAX_TOKENS: int = 2048
    # Application Parameters
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    DOCUMENTS_DIR: str = str(BASE_DIR / "data" / "documents")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
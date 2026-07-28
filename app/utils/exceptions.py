class AppBaseException(Exception):
    """Base exception for application errors."""
    pass


class DocumentProcessingError(AppBaseException):
    """Raised when document ingestion or processing fails."""
    pass


class VectorStoreError(AppBaseException):
    """Raised when vector database operations fail."""
    pass
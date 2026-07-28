from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.config import settings
from app.rag.embeddings import get_embedding_function
from app.utils.exceptions import VectorStoreError
from app.utils.logger import logger


class VectorStoreManager:
    """Manages ChromaDB vector store initialization, persistence, and queries."""

    _instance = None

    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embeddings = get_embedding_function()
        
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
            logger.info(f"Initialized ChromaDB vector store at: {settings.CHROMA_PERSIST_DIR}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise VectorStoreError(f"ChromaDB initialization failed: {str(e)}") from e

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Adds chunked documents to the vector database."""
        if not documents:
            return []

        try:
            ids = [doc.metadata.get("chunk_id") for doc in documents]
            # Handle potential missing chunk IDs gracefully
            if any(doc_id is None for doc_id in ids):
                ids = None

            added_ids = self.vector_store.add_documents(documents=documents, ids=ids)
            logger.info(f"Successfully indexed {len(documents)} chunks in collection '{self.collection_name}'.")
            return added_ids
        except Exception as e:
            logger.error(f"Failed to index documents into vector store: {str(e)}")
            raise VectorStoreError(f"Indexing failed: {str(e)}") from e

    def get_vector_store(self) -> Chroma:
        """Returns underlying Chroma instance."""
        return self.vector_store
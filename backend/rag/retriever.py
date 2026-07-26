from typing import List, Tuple
from langchain_core.documents import Document
from app.rag.vectorstore import VectorStoreManager
from app.utils.logger import logger


class DocumentRetriever:
    """Handles semantic context retrieval from the vector store."""

    def __init__(self, vector_store_manager: VectorStoreManager, k: int = 4):
        self.vector_store = vector_store_manager.get_vector_store()
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieves top-k relevant document chunks for a query."""
        logger.info(f"Retrieving top {self.k} documents for query: '{query}'")
        results = self.vector_store.similarity_search(query, k=self.k)
        logger.info(f"Retrieved {len(results)} relevant chunks.")
        return results

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieves document chunks along with similarity distance scores."""
        logger.info(f"Retrieving top {self.k} documents with scores for query: '{query}'")
        return self.vector_store.similarity_search_with_score(query, k=self.k)
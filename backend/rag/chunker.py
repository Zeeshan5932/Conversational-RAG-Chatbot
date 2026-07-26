from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import logger


class DocumentChunker:
    """Splits documents into contextual chunks optimized for vector retrieval."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into smaller chunks, preserving and augmenting metadata."""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        
        # Inject chunk IDs for tracking and attribution
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", 1)
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"

        logger.info(f"Split {len(documents)} document sections into {len(chunks)} chunks.")
        return chunks
import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile

from app.config import settings
from app.rag.document_loader import DocumentLoader
from app.rag.chunker import DocumentChunker
from app.rag.vectorstore import VectorStoreManager
from app.utils.exceptions import DocumentProcessingError
from app.utils.logger import logger


class DocumentService:
    """Service orchestrating file persistence, loading, chunking, and embedding storage."""

    def __init__(self):
        self.upload_dir = Path(settings.DOCUMENTS_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = DocumentChunker()
        self.vector_store_manager = VectorStoreManager()

    async def save_and_ingest_file(self, file: UploadFile) -> Dict[str, Any]:
        """Saves uploaded file to disk and indexes its contents into ChromaDB."""
        file_path = self.upload_dir / file.filename

        try:
            # Save file locally
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)

            logger.info(f"Saved uploaded file to {file_path}")

            # Load document
            docs = DocumentLoader.load_file(str(file_path))

            # Chunk document
            chunks = self.chunker.split_documents(docs)

            # Store in ChromaDB
            indexed_ids = self.vector_store_manager.add_documents(chunks)

            return {
                "filename": file.filename,
                "status": "success",
                "pages": len(docs),
                "chunks": len(chunks),
                "indexed_records": len(indexed_ids)
            }

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            if file_path.exists():
                os.remove(file_path)
            raise DocumentProcessingError(f"Ingestion failed for {file.filename}: {str(e)}") from e
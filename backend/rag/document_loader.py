from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from app.utils.exceptions import DocumentProcessingError
from app.utils.logger import logger


class DocumentLoader:
    """Handles loading and parsing of supported document types (PDF, DOCX, TXT)."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """Loads a document from the filesystem and injects standard metadata.
        
        Args:
            file_path: Absolute or relative path to the file.
            
        Returns:
            List of LangChain Document objects with populated metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file format '{ext}'. Supported formats: {cls.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Loading document: {path.name} ({ext})")
        
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(path))
            elif ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            else:
                raise DocumentProcessingError(f"No loader configured for extension {ext}")

            documents = loader.load()
            
            # Standardize metadata across all loaded pages/documents
            for idx, doc in enumerate(documents):
                doc.metadata["source_file"] = path.name
                doc.metadata["file_type"] = ext.lstrip(".")
                if "page" not in doc.metadata:
                    doc.metadata["page"] = idx + 1

            logger.info(f"Successfully loaded {len(documents)} document pages/sections from {path.name}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load document {path.name}: {str(e)}")
            raise DocumentProcessingError(f"Error reading file {path.name}: {str(e)}") from e
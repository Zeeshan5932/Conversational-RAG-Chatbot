from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from app.config import settings
from app.services.document_service import DocumentService
from app.services.chat_service import ChatService
from app.tools.url_reader import URLReaderTool
from app.utils.exceptions import DocumentProcessingError

router = APIRouter()
document_service = DocumentService()
chat_service = ChatService()
url_reader = URLReaderTool()


class HealthCheckResponse(BaseModel):
    status: str
    environment: str
    model: str


class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    pages: int
    chunks: int


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    route_used: str
    docs_retrieved: Optional[List[Dict[str, Any]]] = []
    citations: Optional[List[Dict[str, Any]]] = []


class URLResearchRequest(BaseModel):
    url: str


class URLResearchResponse(BaseModel):
    url: str
    title: str
    content_preview: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        environment=settings.APP_ENV,
        model=settings.GEMINI_MODEL
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".txt"}
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed: {allowed_extensions}"
        )

    try:
        result = await document_service.save_and_ingest_file(file)
        return DocumentUploadResponse(
            filename=result["filename"],
            status=result["status"],
            pages=result["pages"],
            chunks=result["chunks"]
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    try:
        result = await chat_service.process_chat_message(
            message=request.message,
            thread_id=request.thread_id or "default"
        )
        return ChatResponse(
            answer=result["answer"],
            route_used=result["route_used"],
            docs_retrieved=result["docs_retrieved"],
            citations=result["citations"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/research/url", response_model=URLResearchResponse)
async def research_url_endpoint(request: URLResearchRequest):
    if not request.url.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL cannot be empty.")

    res = url_reader.read_url(request.url)
    return URLResearchResponse(
        url=res["url"],
        title=res["title"],
        content_preview=res["content"][:300] + "..." if len(res["content"]) > 300 else res["content"]
    )
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter()


class HealthCheckResponse(BaseModel):
    status: str
    environment: str
    model: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint validating application parameters."""
    return HealthCheckResponse(
        status="healthy",
        environment=settings.APP_ENV,
        model=settings.GEMINI_MODEL
    )
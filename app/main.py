from fastapi import FastAPI
from app.config import settings
from app.api.routes import router as api_router
from app.utils.logger import logger

app = FastAPI(
    title="AI Research Assistant API",
    version="1.0.0",
    description="Backend API powering modern AI Research Assistant"
)

app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting AI Research Assistant API service in [{settings.APP_ENV}] mode.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
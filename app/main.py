from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router as api_router
from app.services.ingestion import ingestion_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- SERVER STARTING UP ---")
    print("1. Initializing Ingestion Queue...")
    ingestion_service.initialize_queue()
    
    print("2. Spawning Background Worker Task...")
    worker_task = asyncio.create_task(ingestion_service.worker_loop())
    print("--- READY FOR TRAFFIC ---\n")
    
    yield 
    
    print("Shutting down background tasks...")
    worker_task.cancel()

# THE CRITICAL LINE: lifespan=lifespan must be here!
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "online"}
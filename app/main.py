
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat
from app.db.postgres import connect_to_postgres, close_postgres
from app.ai.rag.pipeline import rag_pipeline
from app.ai.rag.vector_store import vector_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")

UPLOADS_DIR = (
    Path(__file__).resolve().parents[1]
    / "uploads"
    / "datasheets"
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Connecting to Supabase PostgreSQL...")

    await connect_to_postgres()

    try:
        # Initialize database tables and vector index
        logger.info("Initializing pgvector database schema...")
        await vector_store.ensure_vector_index()

        logger.info("PostgreSQL database initialized.")

        # Ingest supported documents
        logger.info("Starting document ingestion...")

        summary = await rag_pipeline.data_ingestion(UPLOADS_DIR)

        logger.info(
            "Ingestion summary: %s ingested, %s skipped, %s failed.",
            summary["ingested"],
            summary["skipped"],
            summary["failed"]
        )

        # Start accepting API requests
        logger.info("EmbeddedSense API is ready.")

        yield

    finally:
        await close_postgres()
        logger.info("PostgreSQL connection closed.")


app = FastAPI(
    title="EmbeddedSense API",
    description="RAG-powered hardware failure diagnosis assistant",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://embeddedsense-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def landing_page():
    return {
        "message": "Welcome to EmbeddedSense API"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


app.include_router(chat.router)

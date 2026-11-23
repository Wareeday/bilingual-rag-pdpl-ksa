"""FastAPI application entrypoint serving both REST API and Bilingual Web UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import router as api_router
from app.core.config import Settings, get_settings
from app.services.rag_pipeline import BilingualRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def _init_pipeline(app: FastAPI, cfg: Settings) -> None:
    """Initialize the RAG pipeline once and auto-load bundled KSA sample docs."""
    if getattr(app.state, "pipeline", None) is not None:
        return
    logger.info("Initializing %s v%s", cfg.app_name, __version__)
    pipeline = BilingualRAGPipeline(settings=cfg)
    if not pipeline.retriever.list_documents():
        loaded = pipeline.load_sample_documents()
        if loaded:
            logger.info("Auto-indexed %d bundled KSA sample documents.", len(loaded))
    app.state.pipeline = pipeline


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for clean testing and production deployment."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _init_pipeline(app, cfg)
        yield
        logger.info("Shutting down %s", cfg.app_name)

    app = FastAPI(
        title=cfg.app_name,
        version=__version__,
        description=(
            "Production-grade Bilingual (Arabic + English) Enterprise RAG microservice "
            "with Hybrid Search (Vector + BM25 RRF), exact page citations, and Saudi PDPL "
            "(SDAIA) PII redaction guardrails."
        ),
        lifespan=lifespan,
    )

    # Ensure pipeline is ready even under ASGI test clients that skip lifespan
    _init_pipeline(app, cfg)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()

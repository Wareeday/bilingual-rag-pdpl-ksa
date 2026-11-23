"""REST API v1 endpoints for Bilingual Enterprise RAG with PDPL Guardrails."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app import __version__
from app.models.schemas import (
    AuditLogEntry,
    DocumentMetadata,
    DocumentUploadResponse,
    HealthResponse,
    PIIRedactRequest,
    PIIRedactionResult,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_pipeline import BilingualRAGPipeline

router = APIRouter()


def _get_pipeline(request: Request) -> BilingualRAGPipeline:
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """Return service readiness, active backends, and index statistics."""
    pipeline = _get_pipeline(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=pipeline.settings.llm_provider,
        llm_configured=pipeline.llm.is_configured,
        embedding_provider=pipeline.embeddings.active_provider,
        embedding_model=pipeline.settings.embedding_model,
        vector_backend=pipeline.retriever.vector_backend,
        indexed_documents=len(pipeline.retriever.list_documents()),
        indexed_chunks=pipeline.retriever.chunk_count,
        pdpl_guardrails_enabled=pipeline.settings.enable_pii_redaction,
    )


@router.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
)
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> DocumentUploadResponse:
    """Upload and index an Arabic or English PDF/TXT/MD enterprise document."""
    pipeline = _get_pipeline(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 15MB free-tier limit.")

    try:
        meta = pipeline.ingest_bytes(content=content, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentUploadResponse(status="indexed", document=meta)


@router.post(
    "/api/v1/documents/load-sample",
    response_model=list[DocumentMetadata],
    tags=["Documents"],
)
async def load_sample_documents(request: Request) -> list[DocumentMetadata]:
    """Load bundled Bilingual (Arabic + English) Saudi Labor Law & HR Policy samples."""
    pipeline = _get_pipeline(request)
    pipeline.load_sample_documents()
    return pipeline.retriever.list_documents()


@router.get(
    "/api/v1/documents",
    response_model=list[DocumentMetadata],
    tags=["Documents"],
)
async def list_documents(request: Request) -> list[DocumentMetadata]:
    """List all currently indexed enterprise documents."""
    pipeline = _get_pipeline(request)
    return pipeline.retriever.list_documents()


@router.delete("/api/v1/documents/{doc_id}", tags=["Documents"])
async def delete_document(request: Request, doc_id: str) -> dict[str, str]:
    """Remove a document and its chunks from Vector and BM25 indices."""
    pipeline = _get_pipeline(request)
    removed = pipeline.retriever.delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Q&A"])
async def query_documents(
    request: Request, payload: QueryRequest
) -> QueryResponse:
    """Run PDPL-protected Bilingual Hybrid Search RAG Q&A with exact page citations."""
    pipeline = _get_pipeline(request)
    return await pipeline.query(payload)


@router.post(
    "/api/v1/pii/redact",
    response_model=PIIRedactionResult,
    tags=["PDPL Guardrails"],
)
async def redact_pii(
    request: Request, payload: PIIRedactRequest
) -> PIIRedactionResult:
    """Test Saudi PDPL PII redaction on any Arabic or English text."""
    pipeline = _get_pipeline(request)
    return pipeline.redact_text(payload.text)


@router.get(
    "/api/v1/audit/logs",
    response_model=list[AuditLogEntry],
    tags=["PDPL Guardrails"],
)
async def get_audit_logs(request: Request, limit: int = 50) -> list[AuditLogEntry]:
    """Return recent PDPL compliance audit logs (zero raw PII stored)."""
    pipeline = _get_pipeline(request)
    return pipeline.get_audit_logs(limit=min(max(limit, 1), 200))

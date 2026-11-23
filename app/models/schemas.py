"""Pydantic data models for the Bilingual Enterprise RAG API (Pydantic v1 & v2 compatible)."""

from __future__ import annotations

from typing import Any, Literal, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model cleanly across v1 and v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # type: ignore[return-value]


def model_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Deserialize a dict into a Pydantic model across v1 and v2."""
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)  # type: ignore[return-value]


def model_from_json(cls: type[T], raw: str) -> T:
    """Deserialize a JSON string into a Pydantic model across v1 and v2."""
    if hasattr(cls, "model_validate_json"):
        return cls.model_validate_json(raw)
    return cls.parse_raw(raw)  # type: ignore[return-value]


class PIIEntity(BaseModel):
    """Single detected PII span."""

    entity_type: str = Field(..., description="PDPL PII category, e.g. SAUDI_NATIONAL_ID")
    placeholder: str = Field(..., description="Redaction token injected in text")
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["ksa_rule", "presidio"] = "ksa_rule"


class PIIRedactionResult(BaseModel):
    """Result of running Saudi PDPL PII redaction on text."""

    redacted_text: str
    pii_detected: bool
    entity_count: int
    entity_summary: dict[str, int] = Field(default_factory=dict)
    entities: list[PIIEntity] = Field(default_factory=list)


class PIIRedactRequest(BaseModel):
    """Request payload for the standalone PDPL PII redaction sandbox."""

    text: str = Field(..., min_length=1, max_length=20000)


class DocumentChunk(BaseModel):
    """Single page-aware text chunk stored in the hybrid index."""

    chunk_id: str
    doc_id: str
    filename: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    text: str
    language: Literal["ar", "en", "mixed"] = "en"


class DocumentMetadata(BaseModel):
    """Metadata for an indexed enterprise document."""

    doc_id: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int
    language: Literal["ar", "en", "mixed"]
    pii_redacted_count: int = 0
    uploaded_at: str


class DocumentUploadResponse(BaseModel):
    """Response returned after indexing a document."""

    status: str = "indexed"
    document: DocumentMetadata


class Citation(BaseModel):
    """Exact page-level citation returned with every RAG answer."""

    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    excerpt: str
    language: Literal["ar", "en", "mixed"]
    rrf_score: float
    vector_score: float
    bm25_score: float


class QueryRequest(BaseModel):
    """Query payload for Bilingual RAG Q&A."""

    query: str = Field(..., min_length=2, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=15)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_pii_redaction: bool = Field(default=True)
    doc_ids: list[str] | None = Field(
        default=None, description="Optional list of document IDs to restrict search"
    )


class LatencyBreakdown(BaseModel):
    """End-to-end latency breakdown in milliseconds."""

    pii_ms: float
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    """Complete grounded RAG answer with citations and PDPL redaction report."""

    answer: str
    detected_language: Literal["ar", "en", "mixed"]
    redacted_query: str
    pii_report: PIIRedactionResult
    context_pii_redacted_count: int
    citations: list[Citation]
    llm_provider_used: str
    model_used: str
    latency: LatencyBreakdown


class AuditLogEntry(BaseModel):
    """PDPL-compliant audit record (never stores raw PII)."""

    timestamp: str
    event_type: str
    query_sha256: str
    pii_detected: bool
    entity_count: int
    entity_summary: dict[str, int]
    action_taken: str


class HealthResponse(BaseModel):
    """Health and readiness telemetry."""

    status: str
    version: str
    llm_provider: str
    llm_configured: bool
    embedding_provider: str
    embedding_model: str
    vector_backend: str
    indexed_documents: int
    indexed_chunks: int
    pdpl_guardrails_enabled: bool

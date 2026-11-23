"""End-to-End Bilingual Enterprise RAG Pipeline with Saudi PDPL Guardrails."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.models.schemas import (
    AuditLogEntry,
    DocumentMetadata,
    LatencyBreakdown,
    PIIRedactionResult,
    QueryRequest,
    QueryResponse,
)
from app.services.arabic_utils import detect_language
from app.services.audit_logger import PDPLAuditLogger
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import BilingualEmbeddingService
from app.services.hybrid_retriever import HybridRetriever
from app.services.llm_client import LLMClient
from app.services.pii_guardrail import PDPLGuardrail


class BilingualRAGPipeline:
    """Coordinates ingestion, PDPL PII redaction, hybrid retrieval, and cited generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pii_guardrail = PDPLGuardrail(use_presidio=settings.use_presidio)
        self.audit_logger = PDPLAuditLogger(
            log_file=settings.data_dir / "pdpl_audit.jsonl"
        )
        self.processor = DocumentProcessor(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.embeddings = BilingualEmbeddingService(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            hf_api_token=settings.hf_api_token,
        )
        self.retriever = HybridRetriever(
            embedding_service=self.embeddings,
            data_dir=settings.data_dir,
            use_qdrant=settings.use_qdrant,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            qdrant_collection=settings.qdrant_collection,
        )
        self.llm = LLMClient(settings=settings)

    def ingest_bytes(
        self, content: bytes, filename: str, doc_id: str | None = None
    ) -> DocumentMetadata:
        """Parse, optionally redact PII, chunk, and index a document."""
        extracted = self.processor.process_bytes(
            content=content, filename=filename, doc_id=doc_id
        )

        pii_count = 0
        if self.settings.redact_on_ingest:
            for chunk in extracted.chunks:
                res = self.pii_guardrail.redact(chunk.text)
                if res.pii_detected:
                    chunk.text = res.redacted_text
                    pii_count += res.entity_count

        metadata = DocumentMetadata(
            doc_id=extracted.doc_id,
            filename=extracted.filename,
            file_type=extracted.file_type,
            page_count=extracted.page_count,
            chunk_count=len(extracted.chunks),
            language=extracted.language,  # type: ignore[arg-type]
            pii_redacted_count=pii_count,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self.retriever.add_document(metadata=metadata, chunks=extracted.chunks)
        return metadata

    def load_sample_documents(self) -> list[DocumentMetadata]:
        """Index all bundled Saudi Labor Law / HR Policy sample documents."""
        sample_dir = self.settings.sample_docs_dir
        if not sample_dir.exists():
            repo_sample_dir = Path(__file__).resolve().parents[2] / "sample_docs"
            if repo_sample_dir.exists():
                sample_dir = repo_sample_dir

        loaded: list[DocumentMetadata] = []
        if not sample_dir.exists():
            return loaded

        existing_names = {d.filename for d in self.retriever.list_documents()}
        for file_path in sorted(sample_dir.iterdir()):
            if file_path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                continue
            if file_path.name in existing_names:
                continue
            meta = self.ingest_bytes(
                content=file_path.read_bytes(),
                filename=file_path.name,
                doc_id=f"sample_{file_path.stem}",
            )
            loaded.append(meta)
        return loaded

    def redact_text(self, text: str) -> PIIRedactionResult:
        """Run standalone PDPL PII redaction and record audit entry."""
        result = self.pii_guardrail.redact(text)
        self.audit_logger.record(
            event_type="pii_sandbox",
            redacted_text=result.redacted_text,
            pii_result=result,
            action_taken="redacted_sandbox" if result.pii_detected else "clean_pass",
        )
        return result

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute full PDPL-protected Bilingual RAG Q&A."""
        t_start = time.perf_counter()
        detected_lang = detect_language(request.query)

        # Step 1: PDPL PII Redaction on User Query BEFORE embeddings or LLM
        t_pii_start = time.perf_counter()
        if request.enable_pii_redaction and self.settings.enable_pii_redaction:
            pii_report = self.pii_guardrail.redact(request.query)
            sanitized_query = pii_report.redacted_text
            self.audit_logger.record(
                event_type="rag_query",
                redacted_text=sanitized_query,
                pii_result=pii_report,
                action_taken=(
                    "redacted_before_llm" if pii_report.pii_detected else "clean_pass"
                ),
            )
        else:
            pii_report = PIIRedactionResult(
                redacted_text=request.query,
                pii_detected=False,
                entity_count=0,
                entity_summary={},
                entities=[],
            )
            sanitized_query = request.query
        pii_ms = (time.perf_counter() - t_pii_start) * 1000.0

        # Step 2: Hybrid Search (Dense Vector + BM25 + RRF)
        t_ret_start = time.perf_counter()
        citations = self.retriever.search(
            query=sanitized_query,
            top_k=request.top_k,
            vector_weight=request.vector_weight,
            doc_ids=request.doc_ids,
        )

        # Defense-in-depth: redact any PII inside retrieved chunks before LLM call
        context_pii_count = 0
        if request.enable_pii_redaction and self.settings.enable_pii_redaction:
            for cit in citations:
                c_res = self.pii_guardrail.redact(cit.excerpt)
                if c_res.pii_detected:
                    cit.excerpt = c_res.redacted_text
                    context_pii_count += c_res.entity_count
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        # Step 3: Cited Bilingual LLM Generation
        t_llm_start = time.perf_counter()
        answer, provider_used, model_used = await self.llm.generate_answer(
            redacted_query=sanitized_query,
            citations=citations,
            detected_language=detected_lang,
        )
        llm_ms = (time.perf_counter() - t_llm_start) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return QueryResponse(
            answer=answer,
            detected_language=detected_lang,
            redacted_query=sanitized_query,
            pii_report=pii_report,
            context_pii_redacted_count=context_pii_count,
            citations=citations,
            llm_provider_used=provider_used,
            model_used=model_used,
            latency=LatencyBreakdown(
                pii_ms=round(pii_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                llm_ms=round(llm_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )

    def get_audit_logs(self, limit: int = 50) -> list[AuditLogEntry]:
        return self.audit_logger.list_recent(limit=limit)

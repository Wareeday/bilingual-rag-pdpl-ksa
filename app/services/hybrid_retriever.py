"""Hybrid Search Engine: Dense Vector (Qdrant / NumPy) + Okapi BM25 + RRF Fusion.

Combines semantic multilingual vector search with Arabic-normalized BM25 lexical
matching using Reciprocal Rank Fusion (RRF) so both conceptual queries and exact
Saudi legal article numbers (e.g. "المادة 77" / "Article 77") rank accurately.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from app.models.schemas import (
    Citation,
    DocumentChunk,
    DocumentMetadata,
    model_from_dict,
    model_to_dict,
)
from app.services.arabic_utils import tokenize_bilingual
from app.services.embeddings import BilingualEmbeddingService

logger = logging.getLogger(__name__)


class _OkapiBM25:
    """Lightweight, zero-dependency Okapi BM25 scorer with bilingual tokenization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avg_dl: float = 0.0

    def fit(self, corpus_texts: list[str]) -> None:
        self.doc_tokens = [tokenize_bilingual(t) for t in corpus_texts]
        self.doc_freqs = Counter()
        total_len = 0
        for tokens in self.doc_tokens:
            total_len += len(tokens)
            for unique_tok in set(tokens):
                self.doc_freqs[unique_tok] += 1
        self.avg_dl = (total_len / len(self.doc_tokens)) if self.doc_tokens else 0.0

    def score(self, query: str) -> list[float]:
        if not self.doc_tokens:
            return []
        q_tokens = tokenize_bilingual(query)
        n_docs = len(self.doc_tokens)
        scores = [0.0] * n_docs

        for idx, d_tokens in enumerate(self.doc_tokens):
            if not d_tokens:
                continue
            tf_map = Counter(d_tokens)
            dl = len(d_tokens)
            score = 0.0
            for qt in q_tokens:
                df = self.doc_freqs.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (dl / max(self.avg_dl, 1e-9))
                )
                score += idf * (tf * (self.k1 + 1.0)) / denom
            scores[idx] = score
        return scores


class HybridRetriever:
    """Manages document persistence, Vector Index (Qdrant/NumPy), BM25, and RRF."""

    def __init__(
        self,
        embedding_service: BilingualEmbeddingService,
        data_dir: Path,
        use_qdrant: bool = True,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        qdrant_collection: str = "ksa_enterprise_rag",
    ) -> None:
        self.embeddings = embedding_service
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.data_dir / "index_store.json"
        self.qdrant_collection = qdrant_collection

        self._documents: dict[str, DocumentMetadata] = {}
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []
        self._bm25 = _OkapiBM25()

        self._qdrant_client = None
        self.vector_backend = "numpy_local"

        if use_qdrant:
            self._init_qdrant(qdrant_url, qdrant_api_key)

        self._load_store()

    def _init_qdrant(self, qdrant_url: str, qdrant_api_key: str) -> None:
        try:
            from qdrant_client import QdrantClient

            if qdrant_url:
                self._qdrant_client = QdrantClient(
                    url=qdrant_url, api_key=qdrant_api_key or None
                )
                self.vector_backend = "qdrant_cloud"
            else:
                self._qdrant_client = QdrantClient(":memory:")
                self.vector_backend = "qdrant_memory"
            logger.info("Qdrant vector backend initialized (%s).", self.vector_backend)
        except Exception as exc:
            logger.info(
                "qdrant-client not active (%s); using built-in NumPy cosine vector store.",
                exc,
            )
            self._qdrant_client = None
            self.vector_backend = "numpy_local"

    def _sync_qdrant(self) -> None:
        if self._qdrant_client is None or not self._chunks or not self._vectors:
            return
        from qdrant_client.models import Distance, PointStruct, VectorParams

        dim = len(self._vectors[0])
        self._qdrant_client.recreate_collection(
            collection_name=self.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=idx,
                vector=vec,
                payload={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id},
            )
            for idx, (chunk, vec) in enumerate(zip(self._chunks, self._vectors))
        ]
        self._qdrant_client.upsert(collection_name=self.qdrant_collection, points=points)

    def _rebuild_indices(self) -> None:
        self._bm25.fit([c.text for c in self._chunks])
        if self._qdrant_client is not None and self._chunks:
            try:
                self._sync_qdrant()
            except Exception as exc:
                logger.warning("Qdrant sync skipped: %s", exc)

    def _save_store(self) -> None:
        payload: dict[str, Any] = {
            "documents": {k: model_to_dict(v) for k, v in self._documents.items()},
            "chunks": [model_to_dict(c) for c in self._chunks],
            "vectors": self._vectors,
        }
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _load_store(self) -> None:
        if not self.store_path.exists():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            self._documents = {
                k: model_from_dict(DocumentMetadata, v)
                for k, v in raw.get("documents", {}).items()
            }
            self._chunks = [
                model_from_dict(DocumentChunk, c) for c in raw.get("chunks", [])
            ]
            self._vectors = raw.get("vectors", [])
            if self._chunks and (
                len(self._vectors) != len(self._chunks)
                or len(self._vectors[0]) != self.embeddings.dimension
            ):
                self._vectors = self.embeddings.embed_documents(
                    [c.text for c in self._chunks]
                )
            self._rebuild_indices()
        except Exception as exc:
            logger.warning("Failed to load existing index (%s); starting fresh.", exc)
            self._documents = {}
            self._chunks = []
            self._vectors = []

    def add_document(
        self, metadata: DocumentMetadata, chunks: list[DocumentChunk]
    ) -> None:
        """Index a new document into both Vector and BM25 stores."""
        if metadata.doc_id in self._documents:
            self.delete_document(metadata.doc_id)

        vectors = self.embeddings.embed_documents([c.text for c in chunks])
        self._documents[metadata.doc_id] = metadata
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)
        self._rebuild_indices()
        self._save_store()

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from all indices."""
        if doc_id not in self._documents:
            return False
        del self._documents[doc_id]
        kept_chunks: list[DocumentChunk] = []
        kept_vectors: list[list[float]] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            if chunk.doc_id != doc_id:
                kept_chunks.append(chunk)
                kept_vectors.append(vec)
        self._chunks = kept_chunks
        self._vectors = kept_vectors
        self._rebuild_indices()
        self._save_store()
        return True

    def list_documents(self) -> list[DocumentMetadata]:
        return list(self._documents.values())

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        doc_ids: list[str] | None = None,
        rrf_k: int = 60,
    ) -> list[Citation]:
        """Execute Hybrid Search (Vector + BM25) fused via Reciprocal Rank Fusion."""
        if not self._chunks:
            return []

        allowed_indices = [
            idx
            for idx, ch in enumerate(self._chunks)
            if not doc_ids or ch.doc_id in doc_ids
        ]
        if not allowed_indices:
            return []

        q_vec = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        vec_scores = self._compute_vector_scores(q_vec, allowed_indices)

        all_bm25 = self._bm25.score(query)
        bm25_scores = {idx: all_bm25[idx] for idx in allowed_indices}

        vec_ranked = sorted(
            allowed_indices, key=lambda i: vec_scores.get(i, 0.0), reverse=True
        )
        bm25_ranked = sorted(
            allowed_indices, key=lambda i: bm25_scores.get(i, 0.0), reverse=True
        )

        vec_rank_map = {idx: rank for rank, idx in enumerate(vec_ranked, start=1)}
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked, start=1)}

        bm25_weight = max(0.0, 1.0 - vector_weight)
        fused: list[tuple[int, float]] = []

        for idx in allowed_indices:
            v_contrib = vector_weight / (rrf_k + vec_rank_map[idx])
            b_contrib = (
                bm25_weight / (rrf_k + bm25_rank_map[idx])
                if bm25_scores[idx] > 0
                else 0.0
            )
            fused.append((idx, v_contrib + b_contrib))

        fused.sort(key=lambda item: item[1], reverse=True)
        top_hits = fused[:top_k]

        citations: list[Citation] = []
        for idx, rrf_score in top_hits:
            chunk = self._chunks[idx]
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    excerpt=chunk.text,
                    language=chunk.language,
                    rrf_score=round(float(rrf_score), 5),
                    vector_score=round(float(vec_scores.get(idx, 0.0)), 4),
                    bm25_score=round(float(bm25_scores.get(idx, 0.0)), 4),
                )
            )
        return citations

    def _compute_vector_scores(
        self, q_vec: np.ndarray, allowed_indices: list[int]
    ) -> dict[int, float]:
        if self._qdrant_client is not None:
            try:
                hits = self._qdrant_client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=q_vec.tolist(),
                    limit=len(self._chunks),
                )
                hit_map = {int(h.id): float(h.score) for h in hits}
                return {idx: hit_map.get(idx, 0.0) for idx in allowed_indices}
            except Exception as exc:
                logger.warning("Qdrant search failed (%s); falling back to NumPy.", exc)

        scores: dict[int, float] = {}
        q_norm = float(np.linalg.norm(q_vec)) + 1e-9
        for idx in allowed_indices:
            d_vec = np.asarray(self._vectors[idx], dtype=np.float32)
            d_norm = float(np.linalg.norm(d_vec)) + 1e-9
            scores[idx] = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
        return scores

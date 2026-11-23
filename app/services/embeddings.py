"""Multilingual Embedding Service (Arabic + English).

Supports 3 zero-cost backends:
1. sentence_transformers: Runs locally (e.g. BAAI/bge-m3 or intfloat/multilingual-e5-small)
   -> Recommended for Hugging Face Spaces (16GB RAM free) & Local Docker.
2. huggingface_api: Calls Hugging Face Inference API remotely (0 local RAM)
   -> Recommended for Render 512MB Free Tier.
3. local_hash: Zero-dependency deterministic bilingual dense embeddings using NumPy
   -> Used automatically as safe fallback in CI/unit tests or offline mode.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Sequence

import httpx
import numpy as np

from app.services.arabic_utils import normalize_arabic, tokenize_bilingual

logger = logging.getLogger(__name__)


class BilingualEmbeddingService:
    """Produces L2-normalized dense embeddings for Arabic and English text."""

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model_name: str = "intfloat/multilingual-e5-small",
        hf_api_token: str = "",
        fallback_dim: int = 384,
    ) -> None:
        self.requested_provider = provider
        self.model_name = model_name
        self.hf_api_token = hf_api_token
        self.fallback_dim = fallback_dim
        self.active_provider = provider
        self._st_model = None
        self.dimension = fallback_dim

        if provider == "sentence_transformers":
            self._init_sentence_transformer()
        elif provider == "huggingface_api":
            if not hf_api_token:
                logger.warning(
                    "HF_API_TOKEN not set for huggingface_api; falling back to local_hash."
                )
                self.active_provider = "local_hash"
        else:
            self.active_provider = "local_hash"

    def _init_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading multilingual embedding model: %s", self.model_name)
            self._st_model = SentenceTransformer(self.model_name)
            dim = self._st_model.get_sentence_embedding_dimension()
            if dim:
                self.dimension = int(dim)
            self.active_provider = "sentence_transformers"
        except Exception as exc:
            logger.warning(
                "SentenceTransformer unavailable (%s). Falling back to local_hash embeddings.",
                exc,
            )
            self._st_model = None
            self.active_provider = "local_hash"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks (adds 'passage: ' prefix for E5 models)."""
        if not texts:
            return []
        prepared = [self._format_for_model(t, is_query=False) for t in texts]
        return self._embed(prepared)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query (adds 'query: ' prefix for E5 models)."""
        prepared = self._format_for_model(query, is_query=True)
        return self._embed([prepared])[0]

    def _format_for_model(self, text: str, is_query: bool) -> str:
        if "e5" in self.model_name.lower() and self.active_provider != "local_hash":
            prefix = "query: " if is_query else "passage: "
            return prefix + text
        return text

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.active_provider == "sentence_transformers" and self._st_model is not None:
            vectors = self._st_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return [vec.tolist() for vec in vectors]

        if self.active_provider == "huggingface_api" and self.hf_api_token:
            try:
                return self._embed_via_hf_api(texts)
            except Exception as exc:
                logger.warning("HF Inference API failed (%s); falling back to local_hash.", exc)

        return [self._local_hash_embed(t) for t in texts]

    def _embed_via_hf_api(self, texts: list[str]) -> list[list[float]]:
        url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{self.model_name}/pipeline/feature-extraction"
        )
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json={"inputs": texts})
            resp.raise_for_status()
            data = resp.json()

        results: list[list[float]] = []
        for item in data:
            arr = np.asarray(item, dtype=np.float32)
            # Mean-pool token embeddings if 2D
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            self.dimension = int(arr.shape[0])
            results.append(arr.tolist())
        return results

    def _local_hash_embed(self, text: str) -> list[float]:
        """Deterministic bilingual word + character n-gram hashed dense vector."""
        vec = np.zeros(self.fallback_dim, dtype=np.float32)
        normalized = normalize_arabic(text).lower()
        tokens = tokenize_bilingual(normalized, remove_stopwords=True)

        features: list[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"w:{tok}", 2.0))
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    features.append((f"c3:{tok[i:i+3]}", 0.6))

        if not features:
            features.append(("empty", 1.0))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            idx = val % self.fallback_dim
            sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0:
            vec /= norm
        return vec.tolist()

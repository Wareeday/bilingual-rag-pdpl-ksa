"""Application configuration loaded from environment variables (.env).

Designed for 100% free-tier deployment (Hugging Face Spaces Docker, Render, or Local Docker).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings."""

    app_name: str = field(
        default_factory=lambda: os.getenv(
            "APP_NAME", "Bilingual Enterprise RAG (PDPL Guardrails)"
        )
    )
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "production"))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "7860")))

    # LLM settings (Groq free tier recommended; Ollama for local; extractive fallback if no key)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower()
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )

    # Embedding settings:
    # - "sentence_transformers": local multilingual model (best for HF Spaces 16GB RAM / Docker)
    # - "huggingface_api": free HF Inference API (0 local RAM, best for Render 512MB free tier)
    # - "local_hash": zero-dependency fast deterministic bilingual embeddings (tests/offline)
    embedding_provider: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_PROVIDER", "sentence_transformers"
        ).lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )
    )
    hf_api_token: str = field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))

    # Vector Store settings (Qdrant local/cloud or built-in numpy store)
    use_qdrant: bool = field(
        default_factory=lambda: _as_bool(os.getenv("USE_QDRANT"), default=True)
    )
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "ksa_enterprise_rag")
    )

    # RAG & Chunking parameters
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "650")))
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100"))
    )
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5"))
    )
    default_vector_weight: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_VECTOR_WEIGHT", "0.6"))
    )

    # PDPL Guardrails
    enable_pii_redaction: bool = field(
        default_factory=lambda: _as_bool(
            os.getenv("ENABLE_PII_REDACTION"), default=True
        )
    )
    use_presidio: bool = field(
        default_factory=lambda: _as_bool(os.getenv("USE_PRESIDIO"), default=False)
    )
    redact_on_ingest: bool = field(
        default_factory=lambda: _as_bool(os.getenv("REDACT_ON_INGEST"), default=False)
    )

    # Storage paths
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data_store"))
    )
    sample_docs_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SAMPLE_DOCS_DIR", "sample_docs"))
    )


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings

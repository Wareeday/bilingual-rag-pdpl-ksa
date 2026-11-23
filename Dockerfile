# Production Dockerfile — Ready for Hugging Face Spaces (Free 16GB RAM) & Local Docker
FROM python:3.11-slim

WORKDIR /app

# Create non-root user (required by Hugging Face Spaces security policy)
RUN useradd -m -u 1000 appuser

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create writable data store & HF cache directory
RUN mkdir -p /app/data_store /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app/data_store /home/appuser/.cache

USER appuser

ENV HOST=0.0.0.0
ENV PORT=7860
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV EMBEDDING_PROVIDER=sentence_transformers
ENV EMBEDDING_MODEL=intfloat/multilingual-e5-small
ENV USE_QDRANT=true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

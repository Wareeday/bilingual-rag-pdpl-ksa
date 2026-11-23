"""Free-Tier LLM Client supporting Groq API, Local Ollama, and Extractive Fallback.

Uses httpx directly (no heavy SDK dependency conflicts) and enforces strict
bilingual grounding with mandatory [Source: filename, Page X] citations.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx

from app.core.config import Settings
from app.models.schemas import Citation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise bilingual (Arabic & English) Enterprise Compliance & Policy AI Assistant for Saudi Arabia.

STRICT RULES:
1. Answer ONLY using the provided retrieved context chunks below. Do NOT invent articles, numbers, or policies not present in the context.
2. Language Matching: If the user's question is in Arabic, respond in clear Modern Standard Arabic. If in English, respond in clear professional English. If mixed, match the primary question language.
3. Mandatory Page Citations: Cite EVERY factual claim inline using exact tags like [Source: filename, Page X].
4. Privacy & PDPL: If the query or context contains redacted tokens like [REDACTED_SAUDI_NATIONAL_ID] or [REDACTED_KSA_PHONE], treat them as protected PII and NEVER attempt to guess or reconstruct personal information.
5. Insufficient Context: If the context does not contain the answer, state clearly that the indexed enterprise documents do not cover that detail."""


class LLMClient:
    """Unified LLM caller with graceful fallback when no API key is configured."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        if self.settings.llm_provider == "groq":
            return bool(self.settings.groq_api_key.strip())
        if self.settings.llm_provider == "ollama":
            return True
        return False

    async def generate_answer(
        self,
        redacted_query: str,
        citations: list[Citation],
        detected_language: Literal["ar", "en", "mixed"],
    ) -> tuple[str, str, str]:
        """Generate a cited bilingual answer.

        Returns:
            (answer_text, provider_used, model_used)
        """
        if not citations:
            if detected_language == "ar":
                return (
                    "لم يتم العثور على مستندات مفهرسة للإجابة على هذا السؤال. يرجى رفع مستند PDF أو تحميل مستندات نظام العمل السعودي التجريبية أولاً.",
                    "none",
                    "none",
                )
            return (
                "No indexed documents were found to answer this question. Please upload a PDF/TXT policy document or click 'Load Sample KSA Docs' first.",
                "none",
                "none",
            )

        context_block = self._build_context_block(citations)
        user_prompt = (
            f"RETRIEVED ENTERPRISE CONTEXT:\n{context_block}\n\n"
            f"USER QUESTION (PDPL-Sanitized):\n{redacted_query}\n\n"
            f"Provide a concise, well-structured answer with inline [Source: filename, Page X] citations."
        )

        if self.settings.llm_provider == "groq" and self.settings.groq_api_key.strip():
            try:
                answer = await self._call_groq(user_prompt)
                return answer, "groq", self.settings.groq_model
            except Exception as exc:
                logger.error("Groq API call failed: %s", exc)
                return (
                    self._extractive_fallback(citations, detected_language, error=str(exc)),
                    "extractive_fallback",
                    "hybrid-rrf-extractive",
                )

        if self.settings.llm_provider == "ollama":
            try:
                answer = await self._call_ollama(user_prompt)
                return answer, "ollama", self.settings.ollama_model
            except Exception as exc:
                logger.error("Ollama call failed: %s", exc)
                return (
                    self._extractive_fallback(citations, detected_language, error=str(exc)),
                    "extractive_fallback",
                    "hybrid-rrf-extractive",
                )

        return (
            self._extractive_fallback(citations, detected_language),
            "extractive_fallback",
            "hybrid-rrf-extractive",
        )

    @staticmethod
    def _build_context_block(citations: list[Citation]) -> str:
        parts: list[str] = []
        for idx, cit in enumerate(citations, start=1):
            parts.append(
                f"--- Chunk {idx} | Source: {cit.filename} | Page: {cit.page_number} ---\n"
                f"{cit.excerpt}"
            )
        return "\n\n".join(parts)

    async def _call_groq(self, user_prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.groq_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _call_ollama(self, user_prompt: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "options": {"temperature": self.settings.llm_temperature},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()

    @staticmethod
    def _extractive_fallback(
        citations: list[Citation],
        detected_language: Literal["ar", "en", "mixed"],
        error: str | None = None,
    ) -> str:
        """Deterministic cited response when running without an LLM API key."""
        top = citations[:3]
        if detected_language == "ar":
            header = (
                "وضع الاستخراج المباشر (أضف مفتاح GROQ_API_KEY المجاني في ملف .env لتفعيل التلخيص الذكي عبر Llama-3.3-70B):\n\n"
            )
            if error:
                header = f"تعذر الاتصال بمزود النموذج اللغوي ({error}). إليك أهم المقاطع الموثقة من المستندات:\n\n"
            bullets = [
                f"• {c.excerpt[:380].strip()} [Source: {c.filename}, Page {c.page_number}]"
                for c in top
            ]
            return header + "\n\n".join(bullets)

        header = (
            "Extractive Citation Mode (Add your free GROQ_API_KEY in .env to enable Llama-3.3-70B synthesis):\n\n"
        )
        if error:
            header = f"LLM provider unavailable ({error}). Returning top hybrid-retrieved passages:\n\n"
        bullets = [
            f"• {c.excerpt[:380].strip()} [Source: {c.filename}, Page {c.page_number}]"
            for c in top
        ]
        return header + "\n\n".join(bullets)

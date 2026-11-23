"""Page-aware Bilingual (Arabic + English) Document Ingestion & Chunking.

Supports PDF, TXT, and Markdown files while preserving exact 1-indexed page
numbers for enterprise citation traceability.
"""

from __future__ import annotations

import io
import re
import uuid
from dataclasses import dataclass

from pypdf import PdfReader

from app.models.schemas import DocumentChunk
from app.services.arabic_utils import detect_language


@dataclass
class ExtractedDocument:
    """Intermediate extracted document representation."""

    doc_id: str
    filename: str
    file_type: str
    page_count: int
    language: str
    chunks: list[DocumentChunk]


class DocumentProcessor:
    """Extracts text per page and produces overlapping, page-aware chunks."""

    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_bytes(
        self, content: bytes, filename: str, doc_id: str | None = None
    ) -> ExtractedDocument:
        """Parse raw file bytes into page-aware chunks."""
        lower_name = filename.lower()
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

        if lower_name.endswith(".pdf"):
            pages = self._extract_pdf_pages(content)
            file_type = "pdf"
        elif lower_name.endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="replace")
            pages = self._split_text_into_pages(text)
            file_type = "md" if lower_name.endswith(".md") else "txt"
        else:
            raise ValueError(
                "Unsupported file type. Please upload PDF (.pdf), Text (.txt), or Markdown (.md)."
            )

        all_text = "\n".join(text for _, text in pages)
        if not all_text.strip():
            raise ValueError(
                f"No extractable text found in '{filename}'. If it is a scanned PDF, run OCR first."
            )

        doc_lang = detect_language(all_text)
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page_num, page_text in pages:
            cleaned = self._clean_whitespace(page_text)
            if not cleaned:
                continue
            for piece in self._split_with_overlap(cleaned):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_c{chunk_index}",
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        text=piece,
                        language=detect_language(piece),
                    )
                )
                chunk_index += 1

        return ExtractedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            page_count=max(len(pages), 1),
            language=doc_lang,
            chunks=chunks,
        )

    @staticmethod
    def _extract_pdf_pages(content: bytes) -> list[tuple[int, str]]:
        reader = PdfReader(io.BytesIO(content))
        pages: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            pages.append((idx, extracted))
        return pages

    @staticmethod
    def _split_text_into_pages(text: str) -> list[tuple[int, str]]:
        # Support explicit form-feed or "--- PAGE X ---" markers in text files
        if "\f" in text:
            raw_pages = text.split("\f")
        elif re.search(r"^===\s*PAGE\s+\d+\s*===$", text, flags=re.MULTILINE | re.IGNORECASE):
            raw_pages = re.split(
                r"^===\s*PAGE\s+\d+\s*===$", text, flags=re.MULTILINE | re.IGNORECASE
            )
            raw_pages = [p for p in raw_pages if p.strip()]
        else:
            raw_pages = [text]
        return [(idx, p) for idx, p in enumerate(raw_pages, start=1)]

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_with_overlap(self, text: str) -> list[str]:
        """Split text along paragraph/sentence boundaries with controlled overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                # Prefer paragraph or sentence break inside the last 30% of the window
                search_start = start + int(self.chunk_size * 0.65)
                split_pos = -1
                for sep in ("\n\n", "\n", "。", ".", "؟", "!", ";", " "):
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1:
                        split_pos = pos + len(sep)
                        break
                if split_pos != -1 and split_pos > start:
                    end = split_pos

            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)

            if end >= length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

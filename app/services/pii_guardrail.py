"""Saudi PDPL / SDAIA-Aligned PII Detection & Redaction Guardrail.

Detects and redacts sensitive personal data BEFORE any prompt or document
chunk is transmitted to an external LLM API.

Covers Saudi-specific and global PII:
- Saudi National ID (10 digits starting with 1)
- Saudi Iqama / Resident ID (10 digits starting with 2)
- Saudi Mobile & Landline Numbers (+966 / 00966 / 05X, including Arabic-Indic digits)
- Saudi IBAN (SA + 22 alphanumeric/numeric characters)
- Email Addresses
- Payment Card Numbers (Luhn-validated to avoid false positives)
- Passport Numbers
- IPv4 Addresses
- Optional Microsoft Presidio integration when USE_PRESIDIO=true
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.models.schemas import PIIEntity, PIIRedactionResult
from app.services.arabic_utils import normalize_digits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PatternRule:
    entity_type: str
    placeholder: str
    pattern: re.Pattern[str]
    priority: int  # lower = higher priority when spans overlap
    validator: str | None = None


def _luhn_check(number_str: str) -> bool:
    """Validate credit card digits using the Luhn checksum algorithm."""
    digits = [int(ch) for ch in number_str if ch.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


_KSA_RULES: list[_PatternRule] = [
    _PatternRule(
        entity_type="SAUDI_IBAN",
        placeholder="[REDACTED_SAUDI_IBAN]",
        pattern=re.compile(r"\bSA\d{2}(?:[\s-]?\d{4}){5}\b", re.IGNORECASE),
        priority=1,
    ),
    _PatternRule(
        entity_type="EMAIL_ADDRESS",
        placeholder="[REDACTED_EMAIL]",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        priority=2,
    ),
    _PatternRule(
        entity_type="CREDIT_CARD",
        placeholder="[REDACTED_CREDIT_CARD]",
        pattern=re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
        priority=3,
        validator="luhn",
    ),
    _PatternRule(
        entity_type="KSA_PHONE_NUMBER",
        placeholder="[REDACTED_KSA_PHONE]",
        pattern=re.compile(
            r"(?<!\d)(?:(?:\+|00)966[\s-]?5\d(?:[\s-]?\d){7}|05\d(?:[\s-]?\d){7}|(?:\+|00)966[\s-]?1[1-7](?:[\s-]?\d){7})(?!\d)"
        ),
        priority=4,
    ),
    _PatternRule(
        entity_type="SAUDI_NATIONAL_ID",
        placeholder="[REDACTED_SAUDI_NATIONAL_ID]",
        pattern=re.compile(r"(?<!\d)1\d{9}(?!\d)"),
        priority=5,
    ),
    _PatternRule(
        entity_type="SAUDI_IQAMA_ID",
        placeholder="[REDACTED_IQAMA_ID]",
        pattern=re.compile(r"(?<!\d)2\d{9}(?!\d)"),
        priority=6,
    ),
    _PatternRule(
        entity_type="PASSPORT_NUMBER",
        placeholder="[REDACTED_PASSPORT]",
        pattern=re.compile(
            r"(?:passport(?:\s*(?:no\.?|number|#))?|جواز\s*السفر|رقم\s*الجواز)\s*[:#-]?\s*([A-Z]{1,2}\d{6,8})\b",
            re.IGNORECASE,
        ),
        priority=7,
    ),
    _PatternRule(
        entity_type="IP_ADDRESS",
        placeholder="[REDACTED_IP_ADDRESS]",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        priority=8,
    ),
]


class PDPLGuardrail:
    """Bilingual Saudi PDPL PII detector and redactor."""

    def __init__(self, use_presidio: bool = False) -> None:
        self.use_presidio = use_presidio
        self._presidio_analyzer = None
        if use_presidio:
            self._init_presidio()

    def _init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine

            self._presidio_analyzer = AnalyzerEngine()
            logger.info("Microsoft Presidio AnalyzerEngine initialized.")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Presidio requested but unavailable (%s). Using built-in KSA PDPL engine.",
                exc,
            )
            self._presidio_analyzer = None

    def redact(self, text: str) -> PIIRedactionResult:
        """Detect and mask all Saudi PDPL PII spans in Arabic or English text."""
        if not text:
            return PIIRedactionResult(
                redacted_text="",
                pii_detected=False,
                entity_count=0,
                entity_summary={},
                entities=[],
            )

        # Normalize Arabic-Indic numerals so Arabic phone/ID numbers are always caught
        scan_text = normalize_digits(text)
        candidates: list[tuple[int, int, int, PIIEntity]] = []

        for rule in _KSA_RULES:
            for match in rule.pattern.finditer(scan_text):
                if rule.entity_type == "PASSPORT_NUMBER" and match.lastindex:
                    start, end = match.start(1), match.end(1)
                    matched_str = match.group(1)
                else:
                    start, end = match.start(), match.end()
                    matched_str = match.group(0)

                if rule.validator == "luhn" and not _luhn_check(matched_str):
                    continue

                candidates.append(
                    (
                        rule.priority,
                        start,
                        end,
                        PIIEntity(
                            entity_type=rule.entity_type,
                            placeholder=rule.placeholder,
                            start=start,
                            end=end,
                            score=1.0,
                            source="ksa_rule",
                        ),
                    )
                )

        # Optional Presidio NER pass (catches person names / locations in English)
        if self._presidio_analyzer is not None:
            try:  # pragma: no cover
                results = self._presidio_analyzer.analyze(
                    text=scan_text,
                    entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "IBAN_CODE"],
                    language="en",
                )
                for res in results:
                    if res.score >= 0.7:
                        candidates.append(
                            (
                                10,
                                res.start,
                                res.end,
                                PIIEntity(
                                    entity_type=res.entity_type,
                                    placeholder=f"[REDACTED_{res.entity_type}]",
                                    start=res.start,
                                    end=res.end,
                                    score=round(float(res.score), 2),
                                    source="presidio",
                                ),
                            )
                        )
            except Exception as exc:  # pragma: no cover
                logger.warning("Presidio analysis skipped: %s", exc)

        # Resolve overlapping spans by priority, then longest span
        candidates.sort(key=lambda item: (item[0], -(item[2] - item[1]), item[1]))
        selected: list[PIIEntity] = []
        occupied: list[tuple[int, int]] = []

        for _, start, end, entity in candidates:
            if any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied):
                continue
            occupied.append((start, end))
            selected.append(entity)

        # Apply replacements from right to left so indices stay valid
        selected.sort(key=lambda e: e.start)
        redacted = text
        for entity in reversed(selected):
            redacted = redacted[: entity.start] + entity.placeholder + redacted[entity.end :]

        summary: dict[str, int] = {}
        for entity in selected:
            summary[entity.entity_type] = summary.get(entity.entity_type, 0) + 1

        return PIIRedactionResult(
            redacted_text=redacted,
            pii_detected=len(selected) > 0,
            entity_count=len(selected),
            entity_summary=summary,
            entities=selected,
        )

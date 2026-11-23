"""PDPL Compliance Audit Logger.

Logs every query and redaction event with SHA-256 anonymized fingerprints.
Raw PII is NEVER written to disk or memory logs.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from app.models.schemas import (
    AuditLogEntry,
    PIIRedactionResult,
    model_from_json,
    model_to_dict,
)


class PDPLAuditLogger:
    """In-memory + JSONL persistent audit log for PDPL governance."""

    def __init__(self, log_file: Path, max_memory_entries: int = 200) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._entries: deque[AuditLogEntry] = deque(maxlen=max_memory_entries)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.log_file.exists():
            return
        try:
            with self.log_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    self._entries.append(model_from_json(AuditLogEntry, line))
        except Exception:
            self._entries.clear()

    def record(
        self,
        event_type: str,
        redacted_text: str,
        pii_result: PIIRedactionResult,
        action_taken: str,
    ) -> AuditLogEntry:
        """Persist a PDPL-safe audit entry."""
        digest = hashlib.sha256(redacted_text.encode("utf-8")).hexdigest()[:16]
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            query_sha256=digest,
            pii_detected=pii_result.pii_detected,
            entity_count=pii_result.entity_count,
            entity_summary=pii_result.entity_summary,
            action_taken=action_taken,
        )
        self._entries.append(entry)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(model_to_dict(entry), ensure_ascii=False) + "\n")
        return entry

    def list_recent(self, limit: int = 50) -> list[AuditLogEntry]:
        """Return newest audit entries first."""
        entries = list(self._entries)
        entries.reverse()
        return entries[:limit]

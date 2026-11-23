"""Unit & Integration tests for Bilingual Enterprise RAG with Saudi PDPL Guardrails.

Compatible with both `pytest -v` and `python3 -m unittest discover -s tests -v`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.arabic_utils import detect_language, normalize_arabic, normalize_digits
from app.services.pii_guardrail import PDPLGuardrail


class TestArabicUtils(unittest.TestCase):
    def test_arabic_normalization_and_digits(self) -> None:
        raw = "أَحْمَد إِبْرَاهِيم — المادة ٧٧ في شَرِكَةً"
        norm = normalize_arabic(raw)
        self.assertIn("77", norm)
        self.assertIn("احمد", norm)
        self.assertIn("شركه", norm)

    def test_language_detection(self) -> None:
        self.assertEqual(detect_language("ما هي ساعات العمل في شهر رمضان؟"), "ar")
        self.assertEqual(detect_language("What is the probation period?"), "en")
        self.assertEqual(detect_language("Saudi Labor Law المادة 84"), "mixed")

    def test_arabic_indic_digit_conversion(self) -> None:
        self.assertEqual(normalize_digits("٠٥٥١٢٣٤٥٦٧"), "0551234567")


class TestPDPLGuardrail(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = PDPLGuardrail(use_presidio=False)

    def test_redacts_saudi_pii_in_english_and_arabic(self) -> None:
        text = (
            "Employee National ID 1098765432, Iqama 2456789123, "
            "Phone +966 55 123 4567, Arabic Phone ٠٥٠٩٨٧٦٥٤٣, "
            "IBAN SA0380000000608010167519, Email hr.user@company.sa"
        )
        res = self.guard.redact(text)
        self.assertTrue(res.pii_detected)
        self.assertEqual(res.entity_count, 6)
        self.assertIn("[REDACTED_SAUDI_NATIONAL_ID]", res.redacted_text)
        self.assertIn("[REDACTED_IQAMA_ID]", res.redacted_text)
        self.assertIn("[REDACTED_KSA_PHONE]", res.redacted_text)
        self.assertIn("[REDACTED_SAUDI_IBAN]", res.redacted_text)
        self.assertIn("[REDACTED_EMAIL]", res.redacted_text)
        self.assertNotIn("1098765432", res.redacted_text)
        self.assertNotIn("2456789123", res.redacted_text)
        self.assertNotIn("SA0380000000608010167519", res.redacted_text)

    def test_clean_text_passes_without_false_positives(self) -> None:
        text = "Under Article 84 of Saudi Labor Law, employees receive half a month wage for the first 5 years."
        res = self.guard.redact(text)
        self.assertFalse(res.pii_detected)
        self.assertEqual(res.entity_count, 0)
        self.assertEqual(res.redacted_text, text)


class TestEndToEndAPI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        repo_root = Path(__file__).resolve().parents[1]
        settings = Settings(
            llm_provider="groq",
            groq_api_key="",  # Force deterministic extractive fallback in tests
            embedding_provider="local_hash",
            use_qdrant=False,
            data_dir=Path(self._tmp.name),
            sample_docs_dir=repo_root / "sample_docs",
            enable_pii_redaction=True,
        )
        self.app = create_app(settings=settings)
        self.client_ctx = TestClient(self.app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        self._tmp.cleanup()

    def test_health_and_auto_loaded_samples(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertGreaterEqual(body["indexed_documents"], 3)
        self.assertGreater(body["indexed_chunks"], 0)

    def test_english_rag_query_with_pii_redaction_and_citations(self) -> None:
        payload = {
            "query": "My Iqama is 2456789123 and email is warda@company.sa. What is the End-of-Service Award under Article 84?",
            "top_k": 3,
            "vector_weight": 0.6,
            "enable_pii_redaction": True,
        }
        resp = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["detected_language"], "en")
        self.assertTrue(data["pii_report"]["pii_detected"])
        self.assertNotIn("2456789123", data["redacted_query"])
        self.assertNotIn("warda@company.sa", data["redacted_query"])
        self.assertGreater(len(data["citations"]), 0)
        self.assertIn("Page", data["answer"])

    def test_arabic_rag_query_with_arabic_id_redaction(self) -> None:
        payload = {
            "query": "ما هي ساعات العمل في شهر رمضان؟ رقم هويتي ١٠٩٨٧٦٥٤٣٢",
            "top_k": 3,
            "enable_pii_redaction": True,
        }
        resp = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["detected_language"], "ar")
        self.assertTrue(data["pii_report"]["pii_detected"])
        self.assertIn("[REDACTED_SAUDI_NATIONAL_ID]", data["redacted_query"])
        self.assertGreater(len(data["citations"]), 0)

    def test_pdf_sample_page_citations(self) -> None:
        payload = {
            "query": "What is the hotel accommodation cap in Riyadh and domestic KSA travel per diem?",
            "top_k": 3,
        }
        resp = self.client.post("/api/v1/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        pdf_hits = [
            c for c in data["citations"] if c["filename"] == "ksa_finance_expense_policy_en.pdf"
        ]
        self.assertTrue(len(pdf_hits) > 0)
        self.assertEqual(pdf_hits[0]["page_number"], 2)

    def test_pii_sandbox_and_audit_logs(self) -> None:
        resp = self.client.post(
            "/api/v1/pii/redact",
            json={"text": "Contact +966551234567 or IBAN SA0380000000608010167519"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["entity_count"], 2)

        audit_resp = self.client.get("/api/v1/audit/logs?limit=10")
        self.assertEqual(audit_resp.status_code, 200)
        logs = audit_resp.json()
        self.assertGreater(len(logs), 0)
        self.assertTrue(logs[0]["pii_detected"])

    def test_upload_and_delete_document(self) -> None:
        content = b"=== PAGE 1 ===\nCustom Riyadh AI Policy: All models require SDAIA review."
        upload_resp = self.client.post(
            "/api/v1/documents/upload",
            files={"file": ("custom_policy.txt", content, "text/plain")},
        )
        self.assertEqual(upload_resp.status_code, 200)
        doc_id = upload_resp.json()["document"]["doc_id"]

        del_resp = self.client.delete(f"/api/v1/documents/{doc_id}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json()["status"], "deleted")


if __name__ == "__main__":
    unittest.main()

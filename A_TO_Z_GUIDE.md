# A-to-Z Complete Teaching & Implementation Guide
## Bilingual (Arabic + English) Enterprise RAG with Saudi PDPL Guardrails

Yeh guide aapko **zero se live production deployment** tak har step sikhaati hai — aur saath hi har technical concept explain karti hai taake KSA AI Engineer interviews mein aap confidently har sawaal ka jawab de sakein.

---

## 1. Yeh Project KSA Companies Ko Kyun Pasand Aata Hai?

Saudi Arabia mein Vision 2030 aur **SDAIA (Saudi Data & AI Authority)** ke under har enterprise (banks, government, HR, FinTech, healthcare) ko 3 real challenges face karne padte hain:

1. **Bilingual Reality (Arabic + English)**: Policies, contracts, aur Saudi Labor Law documents aksar Arabic aur English dono mein hote hain. Normal English-only embeddings Arabic queries par poor results deti hain.
2. **Exact Article / Policy Retrieval**: Legal aur HR teams ko sirf "similar meaning" nahi chahiye — unhein exact `Article 77` ya `المادة 84` chahiye with **exact Page Number citation**. Sirf Vector search exact article numbers miss kar sakti hai; isliye hum **Hybrid Search (Dense Vector + Arabic-Normalized BM25 + Reciprocal Rank Fusion)** use karte hain.
3. **Saudi PDPL (Personal Data Protection Law) Compliance**: Agar koi employee prompt mein apna **Saudi National ID (`1xxxxxxxxx`)**, **Iqama (`2xxxxxxxxx`)**, **+966 mobile**, ya **Saudi IBAN (`SA...`)** likh de aur wo seedha external LLM API (OpenAI/Groq) ko chala jaye, toh yeh PDPL violation hai. Hamara **PDPL Guardrail** LLM call se pehle hi sab PII automatically mask kar deta hai aur zero-PII audit log save karta hai.

---

## 2. Har File Ka Role (Interview Walkthrough)

| File | Kya Karti Hai (Simple Explanation) |
| :--- | :--- |
| `app/main.py` | FastAPI server start karta hai, startup par sample KSA docs auto-load karta hai, aur `/` par frontend UI serve karta hai. |
| `app/core/config.py` | `.env` file se saari settings (API keys, model names, chunk size, PDPL toggles) read karta hai. |
| `app/models/schemas.py` | Strict Pydantic v2 data contracts (Request/Response validation) taake API kabhi malformed JSON return na kare. |
| `app/services/arabic_utils.py` | Arabic text normalize karta hai (Tashkeel hatana, `أ إ آ -> ا`, `ة -> ه`, `ى -> ي`, aur Arabic digits `٠١٢٣ -> 0123`) + language detect (`ar`/`en`/`mixed`) karta hai. |
| `app/services/pii_guardrail.py` | Saudi PDPL PII engine: National ID, Iqama, KSA Phone (including Arabic digits), Saudi IBAN, Email, Luhn-verified Credit Card, Passport, IP redact karta hai. Optional Microsoft Presidio support bhi included hai. |
| `app/services/audit_logger.py` | Har redaction event ko `data_store/pdpl_audit.jsonl` mein save karta hai using SHA-256 hash — raw PII kabhi disk par save nahi hota. |
| `app/services/document_processor.py` | PDF (`pypdf`), TXT, aur MD files ko **page-by-page** extract karke overlapping chunks (650 chars, 100 overlap) banata hai aur har chunk ke saath exact `page_number` attach karta hai. |
| `app/services/embeddings.py` | 3 free-tier modes support karta hai: (1) `sentence_transformers` (local `multilingual-e5-small` ya `BAAI/bge-m3`), (2) `huggingface_api` (0 MB local RAM for Render free tier), (3) `local_hash` (fast CI/testing fallback). |
| `app/services/hybrid_retriever.py` | **Dense Vector Search** (Qdrant ya built-in NumPy cosine) + **Okapi BM25** ko **Reciprocal Rank Fusion (RRF)** se combine karta hai. |
| `app/services/llm_client.py` | **Groq Free Tier** (`llama-3.3-70b-versatile`) ya local **Ollama** ko call karta hai with strict citation prompt. Agar API key na ho toh extractive citation fallback deta hai taake app kabhi crash na ho. |
| `app/services/rag_pipeline.py` | Poora end-to-end flow orchestrate karta hai: Query -> Language Detect -> PII Redact -> Audit Log -> Hybrid Search -> Chunk PII Check -> LLM -> Latency Metrics. |
| `frontend/index.html` | Single-file responsive Web UI (Tailwind CDN) with automatic Arabic RTL / English LTR switching, PII Sandbox, aur Audit Log table. |

---

## 3. Step-by-Step Local Setup & Testing (5 Minutes)

### Step 1: Free Groq API Key Hasil Karein (1 Minute)
1. Jayein: **https://console.groq.com/keys**
2. Google/GitHub se free sign in karein.
3. **"Create API Key"** click karein aur key copy kar lein (koi credit card nahi chahiye).

### Step 2: ZIP Extract Karein & Virtual Environment Banayein
```bash
unzip bilingual-rag-pdpl-ksa.zip
cd bilingual-rag-pdpl-ksa

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate
```

### Step 3: Dependencies Install Karein
- **Full Mode (Recommended — Local Multilingual Embeddings + Qdrant)**:
  ```bash
  pip install -r requirements.txt
  ```
- **Fast Lightweight Mode (Agar laptop RAM kam hai ya jaldi test karna hai)**:
  ```bash
  pip install -r requirements-lite.txt
  ```

### Step 4: `.env` File Configure Karein
```bash
cp .env.example .env
```
`.env` file open karein aur apni Groq key paste karein:
```env
GROQ_API_KEY=gsk_your_actual_free_key_here
```
*(Agar aapne `requirements-lite.txt` install kiya hai, toh `.env` mein `EMBEDDING_PROVIDER=local_hash` ya `huggingface_api` set kar dein.)*

### Step 5: App Run Karein!
```bash
uvicorn app.main:app --reload --port 7860
```
Browser mein open karein:
- **Web App UI**: `http://localhost:7860`
- **Swagger API Docs**: `http://localhost:7860/docs`

### Step 6: Automated Tests Run Karein
```bash
EMBEDDING_PROVIDER=local_hash USE_QDRANT=false pytest -v
```
Saare tests GREEN (PASS) hone chahiye!

---

## 4. FREE Live Hosting Guide (Best Method: Hugging Face Spaces)

**Hugging Face Spaces (Docker SDK)** AI Engineers ke liye sabse best free hosting hai kyunki yeh **2 vCPU + 16GB RAM bilkul FREE** deta hai (Render free tier sirf 512MB RAM deta hai).

### 5-Minute Live Deployment Steps:
1. Free account banayein: **https://huggingface.co**
2. Click **New Space** -> Space name dein: `ksa-bilingual-rag-pdpl`
3. **Space SDK** select karein: **Docker** -> Template: **Blank** -> Hardware: **CPU basic (2 vCPU · 16GB RAM · FREE)** -> **Create Space**.
4. Apne local project ko HF Space par push karein (ya Files tab se upload karein):
   ```bash
   git init
   git add .
   git commit -m "Initial production release: Bilingual RAG with Saudi PDPL Guardrails"
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ksa-bilingual-rag-pdpl
   git push hf main
   ```
5. HF Space ke **Settings** tab mein jayein -> **Variables and secrets** -> **New secret**:
   - Name: `GROQ_API_KEY`
   - Value: *(apni Groq key paste karein)*
6. Space automatically build hoga (~2–3 minutes). Build complete hote hi aapka **live public URL** ready ho jayega!
   - Example: `https://YOUR_USERNAME-ksa-bilingual-rag-pdpl.hf.space`

### Alternative Free Hosting: Render.com
Agar aap GitHub se direct Render par deploy karna chahti hain:
1. Project ko apne **GitHub repo** par push karein.
2. Render.com par **New -> Blueprint** select karein aur apna GitHub repo connect karein.
3. `render.yaml` automatically detect ho jayega. Sirf `GROQ_API_KEY` aur free `HF_API_TOKEN` (from `huggingface.co/settings/tokens`) add karein.

---

## 5. GitHub & Deployment Checklist

- [ ] Code ko public GitHub repo `bilingual-rag-pdpl-ksa` par push karein.
- [ ] Confirm karein `.github/workflows/ci.yml` par **Green Checkmark** aa raha hai.
- [ ] `README.md` ke top par apna **Live Hugging Face Space URL** add karein.
- [ ] Optional: **2-Minute Demo Video** record karein (script `README.md` ke bottom mein hai) aur README mein embed karein.
- [ ] Verify karein ke `docs/architecture_diagram.png` GitHub `README.md` mein cleanly render ho raha hai.

---

## 6. Top 5 Interview Questions (With Crisp Answers)

1. **Q: Why Hybrid Search (Vector + BM25) instead of pure Vector Search?**
   - *Answer*: Dense multilingual embeddings excel at semantic paraphrasing across Arabic and English, but often struggle with exact legal identifiers like `Article 77` or `المادة 109`. Combining Arabic-normalized Okapi BM25 with Dense Vectors via **Reciprocal Rank Fusion (RRF)** preserves both semantic recall and exact lexical precision without needing fragile score calibration.
2. **Q: How do you handle Arabic text quirks in RAG?**
   - *Answer*: Before BM25 tokenization and PII scanning, we normalize Arabic orthographic variations — stripping Tashkeel/diacritics and Tatweel, unifying Alef forms (`أ إ آ -> ا`), Teh Marbuta (`ة -> ه`), Alef Maksura (`ى -> ي`), and converting Arabic-Indic numerals (`٠-٩`) to ASCII digits so Arabic phone/ID numbers are never missed.
3. **Q: How does your system align with Saudi PDPL?**
   - *Answer*: We enforce a pre-LLM redaction boundary: both the user query AND retrieved context chunks pass through our Saudi PDPL guardrail (masking National IDs, Iqamas, KSA phones, Saudi IBANs, emails, and Luhn-checked cards) before hitting any external LLM API. Audit logs record only SHA-256 hashes and entity counts — zero raw PII is persisted.
4. **Q: How would you upgrade embeddings for maximum Arabic accuracy in production?**
   - *Answer*: Change one env var: `EMBEDDING_MODEL=BAAI/bge-m3` (supports 100+ languages, 8192-token context, and dense+sparse retrieval) and connect a managed Qdrant cluster via `QDRANT_URL`.
5. **Q: How do you prevent LLM hallucinations?**
   - *Answer*: Low temperature (`0.1`), strict grounding system prompt requiring inline `[Source: filename, Page X]` citations, and an explicit refusal instruction when retrieved chunks do not contain the answer. Grounding quality can also be audited continuously against retrieved chunks.

# 🧠 DocuMind

A production-grade **RAG (Retrieval Augmented Generation)** system that answers questions from your uploaded PDFs and images using fully local, free AI — no API keys, no cloud, no cost.

```
┌──────────────────────────────────────────────────────────────┐
│  PDF / Image  →  Parse / OCR  →  Chunk  →  Batch Embed      │
│  ChromaDB  →  BM25 Hybrid Rerank  →  llama3.2:3b  →  Answer │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Technology |
|---|---|
| **LLM** | Ollama + llama3.2:3b (fully local, auto-detects installed model) |
| **Embeddings** | nomic-embed-text via Ollama batch API |
| **Vector DB** | ChromaDB (persistent, local) |
| **PDF Parsing** | PyMuPDF — exact page numbers per chunk |
| **OCR** | Tesseract + pytesseract |
| **Reranking** | BM25 Hybrid (vector + lexical scores) |
| **Evidence** | Answer-aligned, shows Page No: reference |
| **Backend** | FastAPI + async Python |
| **Frontend** | Vanilla HTML/CSS/JS — dark glass theme |
| **Deployment** | Docker + docker-compose |

---

## 🏗️ Project Structure

```
knowledge-assistant/
├── backend/
│   ├── main.py                    # FastAPI app, clears session on startup
│   ├── api_routes.py              # /upload /ask /documents /reset endpoints
│   ├── services/
│   |   ├── pdf_parser.py          # PyMuPDF text extraction with page numbers
│   |   ├── ocr_parser.py          # Tesseract OCR for images
│   |   ├── text_chunker.py        # Per-page chunking — exact page numbers
│   |   ├── embeddings.py          # Batch embedding via /api/embed
│   |   ├── vector_store.py        # ChromaDB cosine similarity store
│   |   ├── reranker.py            # BM25 hybrid reranker
│   |   └── rag_pipeline.py        # RAG orchestration + evidence alignment
|   ├── database/
|   |   └──vector_db               # Vector database
|   └── uploads/
├── frontend/
│   ├── templates/
|   |   └──index.html              # Main UI
│   └── static/
│       ├── style.css              # Dark glass theme
│       └── script.js              # Split-panel animation + API calls
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start — Git Clone + Docker (Recommended)

This is the easiest way to run DocuMind. Docker handles everything — Python, Tesseract, Ollama, and the app itself.

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — make sure it is running
- At least **8 GB RAM** free (llama3.2:3b needs ~2 GB, leave room for the OS)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/{my_user_name}}/DocuMind.git
cd DocuMind
```

> Replace `yourusername` with my actual GitHub username.

---

### Step 2 — Start everything with Docker

```bash
docker compose up --build
```

This single command will:
1. Build the FastAPI application image
2. Pull and start the Ollama LLM server
3. Automatically download `llama3.2:3b` (~2 GB) and `nomic-embed-text` (~274 MB)
4. Start the app on **http://localhost:8000**

> ⏳ **First run only:** Model downloads take 5–15 minutes depending on your internet speed. You will see `Models ready!` in the logs when done.

---

### Step 3 — Open the app

```
http://localhost:8000
```

Upload a PDF or image, type a question, and get an answer with page-referenced evidence.

---

### Step 4 — Check logs (optional)

```bash
# All services
docker compose logs -f

# Just the app
docker compose logs -f app

# Just Ollama
docker compose logs -f ollama
```

---

### Step 5 — Stop the app

```bash
# Stop (keeps your models and data)
docker compose down

# Stop and wipe all data including downloaded models
docker compose down -v
```

---

### Updating to the latest version

```bash
git pull origin main
docker compose up --build
```

> No `-v` needed when updating code — your Ollama models are preserved.

---

## 💻 Local Setup (No Docker)

Use this if you prefer to run without Docker.

### Step 1 — Clone the repo

```bash
git clone https://github.com/{my_user_name}/DocuMind.git
cd DocuMind
```

### Step 2 — Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
```

### Step 3 — Pull models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Step 4 — Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Windows — installer at https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 5 — Install Python dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install
pip install -r requirements.txt
```

### Step 6 — Start Ollama and the app

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start the app
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7 — Open in browser

```
http://localhost:8000
```

---

## ⚙️ Configuration

All configuration is via environment variables (or edit the defaults directly).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. Set to `http://ollama:11434` in Docker automatically. |
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | LLM for answer generation. Falls back to any installed model automatically. |

**Change LLM** — set the env var or edit `backend/services/rag_pipeline.py`:
```python
LLM_MODEL = "mistral"   # or phi3, llama3, llama3.2, etc.
```

**Change embedding model** — edit `backend/services/embeddings.py`:
```python
EMBEDDING_MODEL = "nomic-embed-text"
```

**Change chunk size** — edit `backend/api_routes.py`:
```python
chunker = TextChunker(chunk_size=600, chunk_overlap=60)
```
Larger chunks = more context per card. Smaller = faster embedding and more granular evidence.

---

## 🔌 API Reference

### Upload a document
```http
POST /api/v1/upload
Content-Type: multipart/form-data

file: <PDF or image>
```
```json
{
  "message": "Document processed successfully",
  "filename": "report.pdf",
  "chunks_created": 87,
  "characters_extracted": 42310,
  "file_type": "pdf"
}
```

### Ask a question
```http
POST /api/v1/ask
Content-Type: application/json

{ "question": "What are the key findings?", "top_k": 5 }
```
```json
{
  "answer": "The key findings are...",
  "sources": [
    {
      "text": "...chunk text...",
      "source": "report.pdf",
      "page": "14",
      "score": 0.812
    }
  ]
}
```

### List documents
```http
GET /api/v1/documents
```

### Delete a document
```http
DELETE /api/v1/documents/{filename}
```

### Reset session
```http
POST /api/v1/reset
```

---

## 🛠️ Troubleshooting

**Ollama not connecting**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start it manually
ollama serve
```

**Models not downloading in Docker**
```bash
# Watch the init container
docker compose logs -f ollama-init

# Or pull manually inside the container
docker exec -it knowledge-ollama ollama pull llama3.2:3b
docker exec -it knowledge-ollama ollama pull nomic-embed-text
```

**No text extracted from PDF**
- The PDF may be scanned (image-based). Try uploading the pages as images instead — OCR will handle it.
- Password-protected PDFs are not supported.

**Slow responses on CPU**
- `llama3.2:3b` is the fastest supported model. Ensure `OLLAMA_NUM_PARALLEL=1` in `docker-compose.yml` so all RAM is dedicated to one request at a time.
- A 200-page PDF takes ~8 seconds to embed (batch API), and ~20–40 seconds to generate an answer on CPU.

**ChromaDB errors on restart**
```bash
rm -rf backend/database/vector_db/
# Then restart — the DB is rebuilt on next upload
```

**Port 8000 already in use**
```bash
# Change the port in docker-compose.yml
ports:
  - "8080:8000"   # access on http://localhost:8080
```

---

## 🐳 GPU Support (NVIDIA)

Uncomment the GPU section in `docker-compose.yml`:

```yaml
# Under the ollama service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then rebuild:
```bash
docker compose up --build
```

Generation speed improves from ~30s to ~2–3s per answer.

---

## 📦 Tech Stack

```
FastAPI  ──  uvicorn  ──  Python 3.11
  │
  ├── PyMuPDF ──────────── PDF parsing with page tracking
  ├── pytesseract ───────── OCR for images
  ├── chromadb ──────────── Vector database (cosine similarity)
  ├── httpx ─────────────── Async HTTP client (Ollama API)
  └── jinja2 ────────────── HTML templating

Ollama (local, no API key needed)
  ├── llama3.2:3b ───────── Answer generation
  └── nomic-embed-text ───── Batch vector embeddings
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
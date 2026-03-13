# 🧠 DocuMind

A production-grade **RAG (Retrieval Augmented Generation)** system that answers questions from your uploaded PDFs and images using fully local AI — no cloud, no API keys required for local use.

```
PDF / Image → Parse / OCR → Chunk → Embed
ChromaDB → BM25 Hybrid Rerank → LLM → Answer + Page Evidence
```

---

## Stack

| Component | Technology |
|---|---|
| **LLM** | Groq API (cloud) or Ollama (local) |
| **Embeddings** | fastembed in-process (cloud) or nomic-embed-text via Ollama (local) |
| **Vector DB** | ChromaDB |
| **PDF Parsing** | PyMuPDF with exact page number tracking |
| **OCR** | Tesseract |
| **Reranking** | BM25 Hybrid (vector + lexical) |
| **Backend** | FastAPI + async Python |
| **Frontend** | Vanilla HTML/CSS/JS |

---

## Project Structure

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
|   └── uploads/                   # Uploaded files will be here
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

## Deploying to HuggingFace Spaces

### Prerequisites

- A [HuggingFace account](https://huggingface.co/join)
- A [Groq API key](https://console.groq.com) — used as the LLM on Spaces
- Your project pushed to GitHub

---

### Step 1 — Get a Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Navigate to **API Keys** → **Create API Key**
3. Copy the key — you will need it in Step 4

---

### Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/knowledge-assistant.git
git push -u origin main
```

---

### Step 3 — Create a HuggingFace Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Set:
   - **Space name**: `documind`
   - **SDK**: Docker
   - **Visibility**: Public
3. Click **Create Space**

---

### Step 4 — Add your API key as a Secret

1. Inside your Space → **Settings** tab
2. Scroll to **Repository secrets** → **New secret**
3. Set **Name** to `GROQ_API_KEY` and paste your key as the value
4. Click **Add secret**

---

### Step 5 — Link your GitHub repository

1. Still in **Settings** → scroll to **Linked GitHub repository**
2. Click **Link repository** → authorise HuggingFace
3. Select your repo and the `main` branch → **Link**

The Space will start building immediately. Watch progress in the **Logs** tab.

---

### Step 6 — Open the app

Once status changes to **Running**, click the **App** tab:

```
https://huggingface.co/spaces/yourusername/documind
```

---

### Updating

Any push to `main` automatically triggers a rebuild:

```bash
git add .
git commit -m "update"
git push origin main
```

---

## Running Locally with Docker

Uses Ollama locally — no Groq key needed.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- 8 GB RAM available

### Start

```bash
git clone https://github.com/yourusername/knowledge-assistant.git
cd knowledge-assistant
docker compose up --build
```

On first run, Docker pulls `llama3.2:3b` (~2 GB) and `nomic-embed-text` (~274 MB). This takes several minutes depending on your connection. Subsequent starts are instant.

Open `http://localhost:8000`

### Stop

```bash
docker compose down        # preserves downloaded models
docker compose down -v     # removes everything including models
```

### Update

```bash
git pull origin main
docker compose up --build
```

---

## Running Locally Without Docker

### Step 1 — Clone and install

```bash
git clone https://github.com/yourusername/knowledge-assistant.git
cd knowledge-assistant

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Install Tesseract

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Windows — installer at https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 3 — Set your Groq key

```bash
export GROQ_API_KEY=your_key_here    # macOS / Linux
set GROQ_API_KEY=your_key_here       # Windows
```

Or create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### Step 4 — Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required for HF Spaces and local-no-Docker mode |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Set automatically by docker-compose |
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | Ollama model for local Docker mode |

---

## Troubleshooting

**"No LLM configured"**
`GROQ_API_KEY` is not set. Add it under Space → Settings → Repository secrets.

**"Invalid GROQ_API_KEY"**
The key was copied incorrectly. Regenerate it at [console.groq.com](https://console.groq.com).

**"Groq rate limit hit"**
The free tier allows approximately 30 requests per minute. Wait a moment and retry.

**Space stuck on Building**
Open the **Logs** tab inside your Space to see the error.

**No text extracted from PDF**
The PDF is likely scanned (image-based). Upload the pages as images instead — Tesseract OCR will handle them.

**Local Docker — models not downloading**
```bash
docker compose logs -f ollama-init
docker exec -it knowledge-ollama ollama list
```

**ChromaDB error on restart**
```bash
rm -rf backend/database/vector_db/
```
The database rebuilds automatically on next upload.

---

## API Reference

### Upload a document
```http
POST /api/v1/upload
Content-Type: multipart/form-data
```

### Ask a question
```http
POST /api/v1/ask
Content-Type: application/json

{ "question": "What are the key findings?", "top_k": 5 }
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

## License

Mad_titaN — free to use, modify, and distribute.
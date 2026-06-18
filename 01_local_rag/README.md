<div align="center">

# PDF RAG Analyzer

**Intelligent Document Understanding · Retrieval-Augmented Generation · Local-First RAG Pipeline**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-00d4aa?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.49-8b5cf6?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.9-3b82f6?style=flat-square&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-emerald?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-8b5cf6?style=flat-square)]()

**Upload a PDF. Get an instant AI summary, document type classification, and natural-language Q&A — powered by vector search and a cloud LLM, with zero training data required.**

---

[Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Tech Stack](#tech-stack) · [API](#api-endpoints) · [Why This Approach](#why-this-approach)

</div>

---

## Features

- **End-to-end RAG pipeline** — extract, chunk, embed, index, retrieve, and answer, all in < 2 seconds for typical documents
- **Document intelligence on upload** — automatically classifies document type (payslip, invoice, contract, book, report, etc.) and generates a 2–3 sentence summary using a cloud LLM
- **Natural-language Q&A** — ask questions in plain English; the system retrieves the most relevant chunks via cosine similarity and generates a coherent answer grounded in the document
- **One file, one vector store** — each PDF gets an isolated ChromaDB collection, enabling clean multi-document management without cross-contamination
- **Source transparency** — every answer includes expandable source chunks so you can verify and trace the reasoning
- **Duplicate-aware uploads** — content-addressed storage (SHA-256 fingerprint) prevents duplicate indexing
- **Glassmorphism UI** — dark-theme interface with animated gradient borders, smooth transitions, and a blue / emerald / purple palette; drag-and-drop or click to upload
- **Provider-agnostic LLM layer** — currently integrates with Bailian's Anthropic-compatible API, but swaps cleanly to OpenAI, Groq, Together, or any OpenAI-compatible endpoint (see [switching providers](#switching-llm-provider))

## Architecture

```
                         ┌─────────────────────────────────┐
                         │         Browser (HTML/JS)        │
                         │  Upload · Query · View Results   │
                         └──────────────┬──────────────────┘
                                        │ HTTP (REST)
                         ┌──────────────▼──────────────────┐
                         │       FastAPI + Uvicorn          │
                         │    pdf_analyzer.py (backend)     │
                         └──────┬──────────────┬───────────┘
                                │              │
                 ┌──────────────▼──┐    ┌──────▼───────────┐
                 │   PDF Pipeline  │    │   Query Pipeline  │
                 │  (on upload)    │    │   (on question)   │
                 └──────┬─────────┘    └──────┬────────────┘
                        │                     │
          ┌─────────────▼──────────┐  ┌───────▼──────────────┐
          │ 1. PyMuPDF (text ext)  │  │ 1. ChromaDB search   │
          │ 2. Text splitter       │  │    (cosine sim, k=4) │
          │ 3. all-MiniLM-L6-v2    │  │ 2. Concatenate top-k │
          │ 4. ChromaDB persist    │  │ 3. LLM generates     │
          │ 5. LLM → summary+type  │  │    answer from ctx   │
          └────────────────────────┘  └──────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **One collection per file** | Clean isolation; delete/update a single file without affecting others; simple metadata tracking |
| **all-MiniLM-L6-v2** | 80 MB model, runs on CPU in < 100 ms per query, 384-dim embeddings — optimal quality/speed trade-off for document retrieval |
| **Cloud LLM (not local)** | Avoids downloading multi-GB models; provides superior summarization and reasoning; the Bailian plan includes free models (Kimi K2.5, GLM-5) with $0 cost |
| **Anthropic SDK** | Bailian exposes an Anthropic-compatible API; using the official SDK ensures clean request/response handling and easy migration |
| **SHA-256 content addressing** | Prevents duplicate indexing at the file level — identical content always produces the same `file_id` |
| **Vanilla JS frontend** | Zero dependencies, no build step, instant load. The entire UI is a single HTML file served directly by FastAPI |

## Quick Start

### Prerequisites

- Python 3.11+
- A [Bailian](https://www.aliyun.com/product/bailian) API key (or any Anthropic/OpenAI-compatible LLM)

### Setup

```bash
# Clone / navigate to the project
cd GenAI-Journey/RAG/Project/01_local_rag

# Create & activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r ../requirments.txt
pip install anthropic python-dotenv
```

### Configure

Create `.env` in `01_local_rag/`:

```env
BAILIAN_API_KEY=sk-your-key-here
BAILIAN_BASE_URL=https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
LLM_MODEL=kimi-k2.5
```

Free models (pricing = **$0**):

| Model | Context Window | Best For |
|---|---|---|
| `kimi-k2.5` | 262K tokens | Complex documents, long-context reasoning |
| `glm-5` | 202K tokens | General-purpose Q&A and summarization |
| `glm-4.7` | 202K tokens | Lightweight, fast responses |

### Run

```bash
cd 01_local_rag
venv\Scripts\python.exe -m uvicorn pdf_analyzer:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**.

> **Note:** The first startup downloads the `all-MiniLM-L6-v2` embedding model (~80 MB). Subsequent starts are instant.

## Usage

### 1. Upload a Document

Drag any PDF onto the animated drop zone, or click the "Upload PDF" button. The system:

- Extracts all text via PyMuPDF
- Splits into 512-character chunks with 64-character overlap
- Embeds each chunk and persists to ChromaDB
- Calls the LLM to generate a **summary** and **document type**
- Returns the result within 1–3 seconds (depending on document size)

### 2. Review the Intelligence

The right panel shows:
- A **document type badge** (e.g., "Payslip", "Book", "Contract", "Report")
- An AI-generated **summary** of the entire document

### 3. Ask Questions

Type any question and press **Ask**. The pipeline:

1. Embeds your query and retrieves the top-4 most similar chunks (cosine similarity)
2. Concatenates them as context
3. Sends context + question to the LLM
4. Returns a natural-language **answer** with expandable source chunks

#### Example Queries

```text
"What is this document about?"
"How much is the late payment fee?"
"Summarize the key terms in simple English"
"Who are the parties involved?"
"What dates are mentioned?"
"List all monetary values in this document"
```

## Project Structure

```
01_local_rag/
├── pdf_analyzer.py          # FastAPI entry point · all routes + LLM + RAG logic
├── pdf_static/
│   └── index.html           # Single-page UI (no build tools, zero deps)
├── uploads/                 # Uploaded PDFs (content-addressed)
├── pdf_chroma/              # ChromaDB vector store (persisted)
├── pdf_metadata.json        # Maps file_id → filename, collection, summary, type
├── .env                     # API keys (gitignored)
├── README.md
└── .gitignore
```

Only **one Python file** powers the entire backend — `pdf_analyzer.py` (250 lines). The frontend is a single `index.html`.

## API Endpoints

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/` | Serves the frontend UI | — |
| `POST` | `/api/upload` | Upload + index a PDF | — |
| `POST` | `/api/query` | Ask a question about a document | — |
| `GET` | `/api/files` | List all indexed documents | — |
| `DELETE` | `/api/files/{file_id}` | Delete document + its vector index | — |

### Example: Upload

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@payslip.pdf"
```

Response:
```json
{
  "file_id": "a1b2c3d4e5f6",
  "filename": "payslip.pdf",
  "status": "uploaded",
  "chunks": 12,
  "summary": "This document is a monthly payslip issued to an employee, detailing salary, deductions, and net pay for June 2025.",
  "doc_type": "payslip"
}
```

### Example: Query

```bash
curl -X POST http://localhost:8000/api/query \
  -F "file_id=a1b2c3d4e5f6" \
  -F "query=What is the net salary?" \
  -F "k=4"
```

Response:
```json
{
  "file_id": "a1b2c3d4e5f6",
  "filename": "payslip.pdf",
  "query": "What is the net salary?",
  "answer": "The net salary for June 2025 is $4,250.00 after deductions of $750.00 for taxes and benefits.",
  "sources": [
    "Net Pay: $4,250.00\nGross Salary: $5,000.00\nDeductions: $750.00",
    "Pay Period: June 2025\nEmployee ID: EMP-1234"
  ]
}
```

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Backend framework** | [FastAPI](https://fastapi.tiangolo.com/) | Async, auto-docs, Pydantic validation, production-ready |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | ASGI server, hot-reload during development |
| **PDF parsing** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | Fastest pure-Python PDF text extraction, handles corrupted files gracefully |
| **Text splitting** | [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/) | Splits at natural boundaries (paragraphs → sentences → words) |
| **Embeddings** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384-dim, 80 MB, runs on CPU, top performer on MTEB |
| **Vector database** | [ChromaDB](https://www.trychroma.com/) | Embedded, zero-config, SQLite-backed, automatic persistence |
| **LLM client** | [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) → Bailian API | Clean messages API, streaming support, easy to swap providers |
| **Frontend** | Vanilla HTML/CSS/JS | Zero build step, no framework lock-in, single file |
| **Auth / secrets** | [python-dotenv](https://github.com/theskumar/python-dotenv) | 12-factor app config management |

## Why This Approach

**Why not just dump raw chunks?**  
Raw chunks are useful for debugging but useless for end users. Adding an LLM between retrieval and response transforms the system from "here's some text" to "here's your answer."

**Why use ChromaDB instead of FAISS or Pinecone?**  
ChromaDB requires zero configuration, persists to disk automatically, and integrates natively with LangChain. For a local, single-machine deployment, it's the simplest path to a production-quality vector store.

**Why not a local LLM (Llama, Mistral, etc.)?**  
Running a local LLM requires 4–16 GB of RAM/VRAM and minutes of download time. A cloud API call adds ~200 ms latency and delivers GPT-4-class quality with zero hardware cost. The Bailian free tier (Kimi K2.5, GLM-5) costs nothing and runs on Alibaba's infrastructure.

**Why Vanilla JS instead of React/Vue?**  
The frontend has exactly one stateful interaction (upload → query → display). A framework would add 50–200 KB of JavaScript and a build step for zero benefit. Vanilla JS keeps the UI instant-loading and trivially customizable.

## Switching LLM Provider

The LLM layer is a single function (`_call_llm`). To use a different provider:

### OpenAI

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...", base_url="https://api.openai.com/v1")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
)
```

### Groq

```python
client = OpenAI(api_key="gsk-...", base_url="https://api.groq.com/openai/v1")
```

### Ollama (local)

```python
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
```

## Contributing

Contributions are welcome! The codebase is intentionally small (~250 lines Python + 1 HTML file) — easy to understand, easy to extend.

**Ideas for contributions:**
- Add support for OCR-based PDFs (images) via Tesseract or DocTR
- Implement streaming responses for the LLM
- Add a comparison view (side-by-side chunks vs. LLM answer)
- Support batch upload of multiple PDFs
- Add a dark/light theme toggle
- Write unit tests for the RAG pipeline

## License

[MIT](LICENSE) — free for personal, educational, and commercial use.

---

<div align="center">

Built with Python, FastAPI, ChromaDB, and ❤️

</div>

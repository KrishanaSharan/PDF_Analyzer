import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

import fitz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


load_dotenv()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "pdf_chroma"
STATIC_DIR = BASE_DIR / "pdf_static"
METADATA_FILE = BASE_DIR / "pdf_metadata.json"
ALLOWED_EXT = {".pdf"}

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)

app = FastAPI(title="PDF RAG Analyzer with Ollama")

_llm = None


def _normalize_base_url(url: str) -> str:
    if not url:
        return url

    normalized = url.strip().rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def get_llm():
    global _llm

    if _llm is None:
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        if provider == "ollama":
            if OpenAI is None:
                return None

            api_key = os.getenv("OPENAI_API_KEY", "ollama").strip()
            base_url = _normalize_base_url(
                os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
            )

            _llm = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

        elif provider == "openai":
            if OpenAI is None:
                return None

            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            raw_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
            base_url = _normalize_base_url(raw_base_url) if raw_base_url else None

            if not api_key:
                return None

            if base_url:
                _llm = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                _llm = OpenAI(api_key=api_key)

        else:
            return None

    return _llm


def _call_llm(system: str, user: str, model: str | None = None, max_tokens: int = 300) -> str | None:
    client = get_llm()

    if not client:
        return None

    try:
        resp = client.chat.completions.create(
            model=model or os.getenv("LLM_MODEL", "mistral:latest"),
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user[:90000]
                }
            ],
            max_tokens=max_tokens,
            temperature=0.2
        )

        if resp.choices:
            content = resp.choices[0].message.content
            if content:
                return content.strip()

        return None

    except Exception as e:
        print(f"[LLM] {type(e).__name__}: {e}")
        return None


def _summarize(text: str) -> str | None:
    return _call_llm(
        "You are a document analyst. Summarize the document concisely in 2-3 sentences.",
        f"Summarize this document:\n\n{text[:6000]}",
        max_tokens=250
    )


def _classify(text: str) -> str:
    result = _call_llm(
        "You classify documents. Return ONLY one word from this list: payslip, invoice, academic_paper, book, report, contract, letter, manual, financial, legal, resume, presentation, or other.",
        f"Classify this document:\n\n{text[:3000]}",
        max_tokens=20
    )

    allowed = {
        "payslip", "invoice", "academic_paper", "book",
        "report", "contract", "letter", "manual",
        "financial", "legal", "resume", "presentation", "other"
    }

    if result:
        clean = result.lower().strip().replace(".", "")
        if clean in allowed:
            return clean

    return "other"


def _answer(question: str, context: str) -> str | None:
    return _call_llm(
        "Answer clearly and concisely based only on the provided PDF context. If the context does not contain the answer, say: Not found in the document.",
        f"Context:\n{context}\n\nQuestion: {question}",
        max_tokens=500
    )


def _load_meta() -> dict:
    if METADATA_FILE.exists():
        return json.loads(
            METADATA_FILE.read_text(encoding="utf-8")
        )
    return {}


def _save_meta(meta: dict) -> None:
    METADATA_FILE.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )


def _extract_text(path: Path) -> str:
    doc = fitz.open(str(path))
    text = "\n".join(
        page.get_text()
        for page in doc
    )
    doc.close()
    return text.strip()


def _file_id_for(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:12]


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():
        return """
        <h2>PDF RAG Analyzer</h2>
        <p>index.html not found inside pdf_static folder.</p>
        """

    return index_path.read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    content = await file.read()
    file_id = _file_id_for(content)

    meta = _load_meta()

    if file_id in meta:
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "already_exists",
            "chunks": meta[file_id]["chunks"],
            "summary": meta[file_id].get("summary", ""),
            "doc_type": meta[file_id].get("doc_type", "other")
        }

    dest = UPLOAD_DIR / f"{file_id}.pdf"
    dest.write_bytes(content)

    raw_text = _extract_text(dest)

    if not raw_text:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF"
        )

    chunks = text_splitter.split_text(raw_text)

    collection_name = f"pdf_{file_id}"

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name
    )

    doc_summary = _summarize(raw_text)
    doc_type = _classify(raw_text)

    meta[file_id] = {
        "filename": file.filename,
        "collection_name": collection_name,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "size": len(content),
        "summary": doc_summary or "Summary unavailable. Make sure Ollama is running and the model is installed.",
        "doc_type": doc_type
    }

    _save_meta(meta)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "status": "uploaded",
        "chunks": len(chunks),
        "summary": meta[file_id]["summary"],
        "doc_type": meta[file_id]["doc_type"]
    }


@app.post("/api/query")
def query_pdf(
    file_id: str = Form(...),
    query: str = Form(...),
    k: int = Form(4)
):
    meta = _load_meta()

    if file_id not in meta:
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=meta[file_id]["collection_name"]
    )

    results = db.similarity_search(query, k=k)

    sources = [
        r.page_content
        for r in results
    ]

    context = "\n\n".join(sources)

    answer = _answer(query, context)

    if not answer:
        answer = "LLM unavailable. Make sure Ollama is running and model is installed."

    return {
        "file_id": file_id,
        "filename": meta[file_id]["filename"],
        "query": query,
        "answer": answer,
        "sources": sources
    }


@app.get("/api/files")
def list_files():
    meta = _load_meta()

    return [
        {
            "file_id": fid,
            "filename": info["filename"],
            "uploaded_at": info["uploaded_at"],
            "chunks": info["chunks"],
            "size": info["size"],
            "summary": info.get("summary", ""),
            "doc_type": info.get("doc_type", "other")
        }
        for fid, info in meta.items()
    ]


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str):
    meta = _load_meta()

    if file_id not in meta:
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    collection_name = meta[file_id]["collection_name"]

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name
    )

    db.delete_collection()

    (UPLOAD_DIR / f"{file_id}.pdf").unlink(missing_ok=True)

    del meta[file_id]
    _save_meta(meta)

    return {
        "status": "deleted",
        "file_id": file_id
    }


if __name__ == "__main__":
    uvicorn.run(
        "pdf_analyzer:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
"""FastAPI server for NITRAG Medical RAG system.

Start with:
    uv run python scripts/start_server.py
    # or
    uv run uvicorn nitrag.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_STORE_ROOT = PROJECT_ROOT / "rag_store"
DATA_DIR = PROJECT_ROOT / "data"


def _load_dotenv() -> None:
    """Load .env from project root without requiring python-dotenv."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    import os as _os
    with open(env_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key and key not in _os.environ:
                _os.environ[key] = val.strip()


_load_dotenv()

app = FastAPI(title="NITRAG Medical RAG", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline / store cache
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_cache: Dict[str, Any] = {}       # "{doc_id}:{preset}" → RAGPipeline
_processing_status: Dict[str, Dict] = {}   # doc_id → {stage, error, done}

CONFIG_PRESETS = {
    "local_ollama": "Local Ollama (llama3.1:8b + nomic-embed)",
    "fast_local": "Fast Local (mistral:7b + bge-small)",
    "openai_cloud": "OpenAI Cloud (gpt-4o + text-embedding-3-large)",
    "medical_precise": "Medical Precise (HyDE + wide retrieval)",
}


def _load_store(doc_id: str):
    from nitrag.chunk_manager import PdfTokenStore
    store = PdfTokenStore(encoding_model_name="gpt-4o", root_dir=RAG_STORE_ROOT)
    store.load(doc_id)
    return store


def _get_pipeline(doc_id: str, config_preset: str = "openai_cloud"):
    cache_key = f"{doc_id}:{config_preset}"
    if cache_key not in _pipeline_cache:
        from nitrag.rag_pipeline import RAGPipeline
        from nitrag.config import RAGConfig
        store = _load_store(doc_id)
        factory = getattr(RAGConfig, config_preset, None)
        if factory is None:
            raise ValueError(f"Unknown config preset: {config_preset!r}")
        _pipeline_cache[cache_key] = RAGPipeline(store, factory())
    return _pipeline_cache[cache_key]


def _check_stages(doc_dir: Path) -> Dict[str, bool]:
    def has_parquets(d: Path) -> bool:
        return d.is_dir() and any(d.glob("*.parquet"))

    indexes = doc_dir / "indexes"
    return {
        "layout": (doc_dir / "layout_pages.parquet").exists(),
        "clinical": (doc_dir / "clinical_document_metadata.json").exists(),
        "chunks": has_parquets(doc_dir / "chunks"),
        "enriched": has_parquets(doc_dir / "chunks_enriched"),
        "lexical_index": indexes.is_dir() and any(indexes.rglob("docs.parquet")),
        "embeddings": (doc_dir / "embeddings").is_dir()
            and any((doc_dir / "embeddings").rglob("manifest.json")),
        "vector_index": indexes.is_dir() and any(indexes.rglob("faiss.index")),
    }


def _doc_info(doc_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = doc_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            mani = json.load(f)
    except Exception:
        return None
    stages = _check_stages(doc_dir)
    ready = stages.get("embeddings", False) and stages.get("vector_index", False)
    return {
        "doc_id": doc_dir.name,
        "source_name": mani.get("source_pdf_name", doc_dir.name),
        "total_pages": mani.get("total_pages", 0),
        "document_type": mani.get("document_type") or "",
        "stages": stages,
        "ready": ready,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if not ui_path.exists():
        raise HTTPException(500, "UI not found. Run the server from the project root.")
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/documents")
async def list_documents():
    if not RAG_STORE_ROOT.exists():
        return {"documents": []}
    docs = []
    for doc_dir in sorted(RAG_STORE_ROOT.iterdir()):
        if not doc_dir.is_dir():
            continue
        info = _doc_info(doc_dir)
        if info:
            docs.append(info)
    return {"documents": docs}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    doc_dir = RAG_STORE_ROOT / doc_id
    if not doc_dir.is_dir():
        raise HTTPException(404, f"Document {doc_id!r} not found")
    info = _doc_info(doc_dir)
    if not info:
        raise HTTPException(404, f"Document {doc_id!r} has no manifest")
    proc = _processing_status.get(doc_id, {})
    return {**info, "processing": proc}


@app.get("/api/config/presets")
async def list_presets():
    return {"presets": [{"id": k, "label": v} for k, v in CONFIG_PRESETS.items()]}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {
        "filename": file.filename,
        "path": str(dest),
        "message": f"Uploaded. POST /api/process to start processing.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document processing
# ─────────────────────────────────────────────────────────────────────────────

# Stage labels shown in the UI progress stream
_PIPELINE_STAGES: List[Dict[str, str]] = [
    {"id": "layout",        "label": "Extracting document layout"},
    {"id": "clinical",      "label": "Extracting clinical metadata"},
    {"id": "chunks",        "label": "Chunking document"},
    {"id": "enriched",      "label": "Enriching chunk metadata"},
    {"id": "lexical_index", "label": "Building lexical indexes"},
    {"id": "embeddings",    "label": "Generating embeddings"},
    {"id": "vector_index",  "label": "Building vector index"},
]


def _emit(doc_id: str, stage: str, status: str, message: str = "") -> None:
    _processing_status[doc_id] = {
        "current_stage": stage,
        "status": status,          # "running" | "done" | "error"
        "message": message,
        "ts": time.time(),
        "done": status in ("done", "error"),
    }


def _run_full_pipeline(pdf_path: str, doc_id_hint: Optional[str]) -> None:
    """Run all pipeline stages for a PDF. Updates _processing_status throughout."""
    from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
    from nitrag.clinical_metadata_extractor import ClinicalMetadataExtractor
    from nitrag.chunk_manager import PdfTokenStore, ChunkManager, register_default_chunkers
    from nitrag.chunk_metadata_enricher import ChunkMetadataEnricher
    from nitrag.index_manager import IndexManager, register_default_indexers
    from nitrag.config import RAGConfig
    from nitrag.embedding_manager import EmbeddingManager
    from nitrag.vector_index_manager import VectorIndexManager

    pdf = Path(pdf_path)
    _id = doc_id_hint or "pending"

    try:
        # Stage 1 — layout
        _emit(_id, "layout", "running")
        extractor = PyMuPDFLayoutExtractor(encoding_model_name="gpt-4o", root_dir=RAG_STORE_ROOT)
        manifest = extractor.extract(pdf, overwrite=True)
        doc_dir = Path(manifest["paths"]["document_dir"])
        _id = doc_dir.name
        _processing_status[_id] = _processing_status.pop(doc_id_hint or "pending", {})
        # Keep the placeholder key as a forward so the UI can follow it to the real doc_id
        if doc_id_hint and doc_id_hint != _id:
            _processing_status[doc_id_hint] = {"forwarded_to": _id}
        _emit(_id, "layout", "running")

        # Stage 2 — clinical
        _emit(_id, "clinical", "running")
        ClinicalMetadataExtractor(doc_dir).run()

        # Stage 3 — chunking
        # overwrite=True because stage 1 already created the doc directory;
        # ingest_pdf needs to write manifest.json into it.
        _emit(_id, "chunks", "running")
        store = PdfTokenStore(encoding_model_name="gpt-4o", root_dir=RAG_STORE_ROOT)
        store.ingest_pdf(pdf, overwrite=True)
        mgr = ChunkManager(store)
        register_default_chunkers(mgr)
        mgr.execute_all(continue_on_error=True)

        # Stage 4 — enrichment
        _emit(_id, "enriched", "running")
        ChunkMetadataEnricher(doc_dir).enrich_all(overwrite=True)

        # Stage 5 — lexical indexing
        _emit(_id, "lexical_index", "running")
        index_mgr = IndexManager(store=store, use_enriched_chunks=True)
        register_default_indexers(index_mgr)
        index_mgr.execute_all(continue_on_error=True, overwrite=True)

        # Stage 6 — embeddings
        _emit(_id, "embeddings", "running")
        rag_config = RAGConfig.openai_cloud()
        em = EmbeddingManager(store, rag_config.embedding)
        em.embed_all_strategies(use_enriched=True, overwrite=True)

        # Stage 6b — vector index
        _emit(_id, "vector_index", "running")
        vim = VectorIndexManager(store, em, rag_config.vector_index)
        vim.build_all(overwrite=True)

        _emit(_id, "vector_index", "done", f"Ready: {_id}")

    except Exception as exc:
        tb = traceback.format_exc()
        _emit(_id, _processing_status.get(_id, {}).get("current_stage", "unknown"), "error", str(exc))


class ProcessRequest(BaseModel):
    pdf_path: str
    config_preset: str = "openai_cloud"


@app.post("/api/process")
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    pdf = Path(request.pdf_path)
    if not pdf.exists():
        raise HTTPException(404, f"File not found: {request.pdf_path}")
    if not pdf.suffix.lower() == ".pdf":
        raise HTTPException(400, "Only PDF files are supported")

    # Give a stable placeholder ID while we don't yet know the real doc_id
    placeholder = f"pending_{pdf.stem}"
    _emit(placeholder, "layout", "running", "Starting…")

    background_tasks.add_task(_run_full_pipeline, str(pdf), placeholder)
    return {"status": "processing", "placeholder_id": placeholder, "pdf": pdf.name}


@app.get("/api/process/status/{doc_id}")
async def process_status(doc_id: str):
    status = _processing_status.get(doc_id)
    # Follow placeholder → real doc_id forward
    if status and "forwarded_to" in status:
        doc_id = status["forwarded_to"]
        status = _processing_status.get(doc_id)
    if status is None:
        # Also check real doc stages on disk
        doc_dir = RAG_STORE_ROOT / doc_id
        if doc_dir.is_dir():
            stages = _check_stages(doc_dir)
            all_done = all(stages.values())
            return {"doc_id": doc_id, "done": all_done, "stages": stages, "current_stage": None}
        raise HTTPException(404, f"No processing record for {doc_id!r}")
    return {"doc_id": doc_id, **status}


@app.get("/api/process/stream/{doc_id}")
async def process_stream(doc_id: str):
    """SSE stream for real-time processing progress."""
    async def generate() -> Iterator[str]:
        last_stage = None
        for _ in range(300):          # max 5 min at 1s intervals
            status = _processing_status.get(doc_id, {})
            stage = status.get("current_stage")
            state = status.get("status", "unknown")
            msg = status.get("message", "")
            if stage != last_stage or state in ("done", "error"):
                payload = json.dumps({"stage": stage, "status": state, "message": msg})
                yield f"data: {payload}\n\n"
                last_stage = stage
            if status.get("done"):
                yield "data: [DONE]\n\n"
                return
            await asyncio.sleep(1)
        yield "data: [TIMEOUT]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    doc_id: str
    config_preset: str = "local_ollama"
    evaluate: bool = True


def _serialize_response(response) -> Dict[str, Any]:
    return {
        "query": response.query,
        "answer": response.answer,
        "citations": [
            {
                "number": c.number,
                "chunk_id": c.chunk_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "quote": c.quote,
                "confidence": c.confidence,
                "source_label": c.source_label,
            }
            for c in response.citations
        ],
        "context": {
            "chunks": [
                {
                    "citation_number": ch.citation_number,
                    "chunk_id": ch.chunk_id,
                    "text": ch.text,
                    "page_start": ch.page_start,
                    "page_end": ch.page_end,
                    "section": ch.section,
                    "score": round(float(ch.score or 0), 4),
                    "retriever": ch.retriever,
                    "token_count": ch.token_count,
                    "source_label": ch.source_label,
                    "contains_medication": ch.contains_medication,
                    "contains_lab": ch.contains_lab,
                    "contains_diagnosis": ch.contains_diagnosis,
                    "contains_vital": ch.contains_vital,
                    "clinical_quality_score": round(float(ch.clinical_quality_score or 0), 3),
                }
                for ch in response.context.chunks
            ],
            "total_tokens": response.context.total_tokens,
            "truncated": response.context.truncated,
        },
        "evaluation": response.evaluation.to_dict() if response.evaluation else None,
        "latency": response.latency,
        "config_snapshot": response.config_snapshot,
    }


@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    doc_dir = RAG_STORE_ROOT / request.doc_id
    if not doc_dir.is_dir():
        raise HTTPException(404, f"Document {request.doc_id!r} not found")

    stages = _check_stages(doc_dir)
    if not (stages.get("embeddings") and stages.get("vector_index")):
        raise HTTPException(
            400,
            "Document is not ready for querying. "
            "Run scripts/run_pipeline.py to complete embedding and indexing.",
        )

    try:
        pipeline = _get_pipeline(request.doc_id, request.config_preset)
    except Exception as e:
        raise HTTPException(400, f"Pipeline init failed: {e}")

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: pipeline.answer(request.query, evaluate=request.evaluate),
        )
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            raise HTTPException(
                503,
                "LLM not reachable. Check that Ollama (or your configured LLM) is running. "
                f"Details: {err}",
            )
        raise HTTPException(500, err)

    return _serialize_response(response)

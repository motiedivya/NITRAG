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
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
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
    "openai_cloud": "OpenAI Cloud (gpt-4o + text-embedding-3-small)",
    "medical_precise": "Medical Precise (HyDE + wide retrieval)",
}

NARRATIVE_SUMMARY_PROMPT = """\
You are a senior clinical documentation specialist producing a Narrative Summary — \
a formal, defensible medico-legal artifact used for care coordination, insurance \
authorisation, specialist referral, and legal review.

Write a comprehensive, clinically precise Narrative Summary of the document. \
Include all applicable sections below; omit any section for which the document \
contains no relevant information.

## Patient & Encounter
Full name, date of birth, sex, MRN / accession number, visit or exam date and time, \
referring provider, facility, and document type.

## Clinical Context
Reason for study or visit. Presenting complaint. Relevant clinical history and indication.

## Findings
Organised by anatomical region or body system. Include all specific measurements, \
signal characteristics, laterality, severity qualifiers, and verbatim descriptors \
where clinically relevant. Quote exact numeric values.

## Impression / Diagnoses
Numbered list of primary and secondary diagnoses or radiological impressions. \
Use precise ICD-style language where possible.

## Medications & Treatments
All medications with dose, route, and frequency. Any procedures performed or \
treatments administered during this encounter.

## Plan & Follow-Up
Recommended next steps, follow-up imaging or testing, referrals, discharge instructions, \
and patient-facing guidance.

## Critical Flags
Any urgent, critical, or actionable findings requiring immediate clinical attention. \
Abnormal values outside reference ranges. Unexpected incidental findings.

CITATION RULES (strictly enforced):
1. Every factual claim MUST carry an inline [N] citation.
2. Never write a sentence without at least one citation.
3. All numeric values (measurements, doses, dates, lab results) MUST be cited.
4. Place [N] immediately after the claim, before the full stop.
5. Multiple citations per sentence are permitted and encouraged.

STYLE:
- Formal, professional clinical prose. Third person. Past tense for findings.
- Precise medical terminology. No colloquialisms or hedging language.
- Do not speculate or infer beyond the retrieved passages.
- Do not summarise what you cannot cite.
"""


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


def _get_pdf_path(doc_id: str) -> Optional[Path]:
    doc_dir = RAG_STORE_ROOT / doc_id
    manifest_path = doc_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            mani = json.load(f)
    except Exception:
        return None
    # Try absolute path stored in manifest
    p = Path(mani.get("source_pdf_path", "") or "")
    if p.is_absolute() and p.exists():
        return p
    # Try relative to project root
    if p.parts:
        pp = PROJECT_ROOT / p
        if pp.exists():
            return pp
    # Fall back: name in data dir
    name = mani.get("source_pdf_name", "")
    if name:
        pp = DATA_DIR / name
        if pp.exists():
            return pp
    return None


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
    has_summary = (doc_dir / "narrative_summary.json").exists()
    return {
        "doc_id": doc_dir.name,
        "source_name": mani.get("source_pdf_name", doc_dir.name),
        "total_pages": mani.get("total_pages", 0),
        "document_type": mani.get("document_type") or "",
        "stages": stages,
        "ready": ready,
        "has_summary": has_summary,
    }


def _generate_narrative_summary(doc_id: str) -> None:
    summary_path = RAG_STORE_ROOT / doc_id / "narrative_summary.json"
    if summary_path.exists():
        return
    from nitrag.rag_pipeline import RAGPipeline
    from nitrag.config import RAGConfig
    store = _load_store(doc_id)
    config = RAGConfig.openai_cloud()
    config.llm.system_prompt = NARRATIVE_SUMMARY_PROMPT
    config.generation.max_context_tokens = 8000
    config.retrieval.top_k_retrieve = 40
    config.retrieval.top_k_rerank = 20
    pipeline = RAGPipeline(store, config)
    response = pipeline.answer(
        "Provide a comprehensive narrative summary of this entire clinical document, "
        "covering all patients, findings, diagnoses, medications, treatments, and clinical plans.",
        evaluate=False,
    )
    result = _serialize_response(response)
    result["doc_id"] = doc_id
    result["query"] = "Narrative Summary"
    result["generated_at"] = time.time()
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)


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


@app.get("/api/documents/{doc_id}/pdf")
async def serve_pdf(doc_id: str):
    pdf_path = _get_pdf_path(doc_id)
    if pdf_path is None:
        raise HTTPException(404, "Original PDF not found on disk")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


def _split_into_sentences(text: str) -> list:
    """Split text into matchable sentences."""
    import re
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # If only one part (no sentence breaks), try splitting on clause separators
    if len(parts) == 1:
        parts = re.split(r'(?:[;:])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_fragments(sentence: str) -> list:
    """Return progressively shorter fragments of a sentence for fuzzy matching."""
    words = sentence.split()
    n = len(words)
    frags = [sentence]
    if n > 8:
        frags.append(" ".join(words[:max(6, n * 4 // 5)]))
    if n > 12:
        frags.append(" ".join(words[:max(5, n * 3 // 5)]))
    # Character-length fallbacks
    for length in (80, 55, 35):
        if len(sentence) > length + 10:
            frags.append(sentence[:length])
    return frags


def _find_quote_rects(page, quote: str) -> list:
    """Locate all rectangles on the page that correspond to the quote text.

    Strategy 1 — sentence-by-sentence exact search using PyMuPDF.
    Strategy 2 — word-overlap (Jaccard) on text blocks when exact search fails.
    Returns (matched_rects, used_fallback).
    """
    import re
    import fitz

    matched: list = []

    # ── Strategy 1: sentence-level exact search ──────────────────────────────
    sentences = _split_into_sentences(quote)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        for frag in _sentence_fragments(sent):
            frag = frag.strip()
            if not frag:
                continue
            rects = page.search_for(frag)
            if rects:
                for r in rects[:6]:
                    ann = page.add_highlight_annot(r)
                    ann.set_colors(stroke=[1.0, 0.85, 0.0])
                    ann.update()
                    matched.append(r)
                break  # found this sentence, move to next

    if matched:
        return matched

    # ── Strategy 2: word-overlap fallback on text blocks ────────────────────
    quote_words = set(re.findall(r"[a-z]+", quote.lower()))
    if not quote_words:
        return []

    best_score = 0.0
    best_rect = None
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:          # skip image blocks
            continue
        bx0, by0, bx1, by1, btext = block[0], block[1], block[2], block[3], block[4]
        if not btext or btext.isspace():
            continue
        bwords = set(re.findall(r"[a-z]+", btext.lower()))
        if not bwords:
            continue
        jaccard = len(quote_words & bwords) / max(len(quote_words | bwords), 1)
        if jaccard > best_score:
            best_score = jaccard
            best_rect = fitz.Rect(bx0, by0, bx1, by1)

    if best_rect and best_score > 0.18:
        ann = page.add_highlight_annot(best_rect)
        ann.set_colors(stroke=[1.0, 0.85, 0.0])
        ann.update()
        return [best_rect]

    return []


@app.get("/api/documents/{doc_id}/page/{page_num}")
async def render_page(doc_id: str, page_num: int, q: str = ""):
    """Render a tight cropped snippet of a PDF page with exact sentence highlights."""
    import fitz
    pdf_path = _get_pdf_path(doc_id)
    if pdf_path is None:
        raise HTTPException(404, "Original PDF not found on disk")
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise HTTPException(500, f"Cannot open PDF: {e}")
    if page_num < 0 or page_num >= len(doc):
        raise HTTPException(400, f"Page {page_num} out of range (0–{len(doc)-1})")

    page = doc[page_num]
    page_rect = page.rect
    matched_rects: list = []

    if q:
        matched_rects = _find_quote_rects(page, q)

    if matched_rects:
        # Tight bounding box over all matched rects
        x0 = min(r.x0 for r in matched_rects)
        y0 = min(r.y0 for r in matched_rects)
        x1 = max(r.x1 for r in matched_rects)
        y1 = max(r.y1 for r in matched_rects)

        # Estimate a line height and add ~2.5 lines of context padding
        line_h = max(12.0, (y1 - y0) / max(len(matched_rects), 1))
        pad_v = max(36.0, line_h * 2.5)

        clip = fitz.Rect(
            max(page_rect.x0, x0 - 16),
            max(page_rect.y0, y0 - pad_v),
            min(page_rect.x1, x1 + 16),
            min(page_rect.y1, y1 + pad_v),
        )
        # Ensure a minimum clip height so single-line hits are still readable
        if (clip.y1 - clip.y0) < 56:
            clip = fitz.Rect(clip.x0,
                             max(page_rect.y0, clip.y0 - 18),
                             clip.x1,
                             min(page_rect.y1, clip.y1 + 18))

        mat = fitz.Matrix(2.8, 2.8)
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    else:
        # No match — render full page at modest scale as fallback
        mat = fitz.Matrix(1.8, 1.8)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

    png = pix.tobytes("png")
    doc.close()
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/documents/{doc_id}/summary")
async def get_summary(doc_id: str):
    summary_path = RAG_STORE_ROOT / doc_id / "narrative_summary.json"
    if not summary_path.exists():
        raise HTTPException(404, "Narrative summary not yet generated")
    with open(summary_path) as f:
        return json.load(f)


@app.post("/api/documents/{doc_id}/summarize")
async def trigger_summary(doc_id: str, background_tasks: BackgroundTasks):
    doc_dir = RAG_STORE_ROOT / doc_id
    if not doc_dir.is_dir():
        raise HTTPException(404, f"Document {doc_id!r} not found")
    summary_path = doc_dir / "narrative_summary.json"
    if summary_path.exists():
        return {"status": "already_exists"}
    _processing_status[doc_id] = {
        "current_stage": "narrative_summary", "status": "running",
        "message": "Generating Narrative Summary…", "ts": time.time(), "done": False,
    }
    background_tasks.add_task(_run_summarize_only, doc_id)
    return {"status": "generating"}


def _run_summarize_only(doc_id: str) -> None:
    try:
        _generate_narrative_summary(doc_id)
        _emit(doc_id, "narrative_summary", "done", "Summary ready")
    except Exception as exc:
        _emit(doc_id, "narrative_summary", "error", str(exc))


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
        # Don't mark done yet — summary generation follows
        _processing_status[_id].update({"current_stage": "vector_index", "status": "complete"})

        # Stage 7 — narrative summary (LLM call, may take 30–60 s)
        _emit(_id, "narrative_summary", "running", "Generating with GPT-4o…")
        try:
            _generate_narrative_summary(_id)
            _emit(_id, "narrative_summary", "done", f"Ready: {_id}")
        except Exception as exc:
            _emit(_id, "narrative_summary", "error", f"Summary failed (doc is still queryable): {exc}")

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

    result = _serialize_response(response)
    result["doc_id"] = request.doc_id
    return result

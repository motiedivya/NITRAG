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

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
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


_NARRATIVE_QUERY = (
    "Provide a comprehensive narrative summary of this entire clinical document, "
    "covering all patients, findings, diagnoses, medications, treatments, and clinical plans."
)


def _generate_narrative_summary(doc_id: str) -> None:
    summary_path = RAG_STORE_ROOT / doc_id / "narrative_summary.json"
    if summary_path.exists():
        return

    from nitrag.rag_pipeline import RAGPipeline
    from nitrag.config import RAGConfig
    from nitrag.chunk_manager import ChunkManager, register_default_chunkers
    from nitrag.generation_manager import GenerationManager, resolve_citations

    store = _load_store(doc_id)

    # Ensure sentence_based chunks exist (on-demand for already-processed docs)
    sentence_chunk_path = store.paths.chunks_dir / "sentence_based.parquet"
    if not sentence_chunk_path.exists():
        print(f"[narrative] Building sentence_based chunks for {doc_id}…")
        cm = ChunkManager(store)
        register_default_chunkers(cm)
        cm.execute("sentence_based")

    config = RAGConfig.openai_cloud()
    config.llm.system_prompt = NARRATIVE_SUMMARY_PROMPT
    config.generation.max_context_tokens = 32000   # fit entire document
    config.generation.context_ordering = "page"    # chronological order

    pipeline = RAGPipeline(store, config)

    # Pass 1: full-document narrative with all sentence chunks in page order
    print(f"[narrative] Pass 1 — full-document generation for {doc_id}…")
    response = pipeline.answer_full_document(
        query=_NARRATIVE_QUERY,
        chunk_strategy="sentence_based",
    )

    # Pass 2: completeness check
    print(f"[narrative] Pass 2 — completeness check for {doc_id}…")
    completeness_prompt = (
        "You are a medical editor reviewing a clinical narrative summary for completeness.\n\n"
        "NARRATIVE SUMMARY:\n" + response.answer + "\n\n"
        "SOURCE EVIDENCE (all sentences from the document):\n"
        + response.context.formatted_text + "\n\n"
        "Identify any clinical visits, diagnoses, procedures, medications, imaging results, "
        "or dated events that appear in the SOURCE EVIDENCE but are NOT mentioned in the "
        "NARRATIVE SUMMARY.\n"
        "If the summary is complete, respond with exactly one word: COMPLETE\n"
        "Otherwise list each missing item on its own line."
    )
    gen = GenerationManager(config.llm)
    completeness_result = gen.generate_text(completeness_prompt)

    if completeness_result and "COMPLETE" not in completeness_result.strip().upper()[:20]:
        # Pass 3: fill gaps
        print(f"[narrative] Pass 3 — filling gaps for {doc_id}…")
        gap_prompt = (
            "Extend the following medical narrative summary to include these missing items:\n\n"
            "MISSING ITEMS:\n" + completeness_result + "\n\n"
            "CURRENT SUMMARY:\n" + response.answer + "\n\n"
            "Insert each missing item at the correct chronological location in the summary. "
            "Every new or modified sentence must carry an inline [N] citation referencing "
            "the evidence already numbered above."
        )
        refined = gen.generate_text_with_context(gap_prompt, response.context)
        if refined:
            response.answer = refined
            response.generation_result.citations = resolve_citations(refined, response.context)
            response.citations = response.generation_result.citations

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


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its pipeline artefacts from rag_store."""
    import shutil as _shutil
    doc_dir = RAG_STORE_ROOT / doc_id
    if not doc_dir.is_dir():
        raise HTTPException(404, f"Document {doc_id!r} not found")

    # Evict from pipeline cache
    to_drop = [k for k in _pipeline_cache if k.startswith(doc_id + ":")]
    for k in to_drop:
        del _pipeline_cache[k]

    # Remove any in-progress status entry
    _processing_status.pop(doc_id, None)

    # Remove the source PDF from data/ if it lives there
    pdf_path = _get_pdf_path(doc_id)
    if pdf_path and pdf_path.exists() and DATA_DIR in pdf_path.parents:
        try:
            pdf_path.unlink()
        except Exception:
            pass  # non-fatal — rag_store dir removal is the important part

    # Remove the entire rag_store directory
    try:
        _shutil.rmtree(doc_dir)
    except Exception as e:
        raise HTTPException(500, f"Could not delete document data: {e}")

    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/pdf")
async def serve_pdf(doc_id: str):
    pdf_path = _get_pdf_path(doc_id)
    if pdf_path is None:
        raise HTTPException(404, "Original PDF not found on disk")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


_HIGHLIGHT_STOP = frozenset([
    "the","a","an","is","was","were","are","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might","can",
    "to","of","in","on","at","by","for","with","and","or","but","from","that",
    "this","it","as","no","not","also","about","after","before","its","their",
    "they","he","she","we","you","i","was","all","any","both","each","more",
    "most","other","such","than","then","when","where","which","who","how",
])


def _split_into_sentences(text: str) -> list:
    """Split text into matchable sentences."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
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
    for length in (90, 65, 45, 30):
        if len(sentence) > length + 8:
            frags.append(sentence[:length])
    return frags


def _find_by_word_window(page, quote: str) -> list:
    """Word-level sliding-window match: robust fallback for OCR / whitespace gaps.

    Returns Rect objects only — caller is responsible for drawing.
    """
    import re
    import fitz

    q_words = [re.sub(r"[^a-z0-9]", "", w) for w in quote.lower().split()]
    q_words = [w for w in q_words if len(w) >= 4 and w not in _HIGHLIGHT_STOP]
    if len(q_words) < 3:
        return []
    q_set = set(q_words)

    page_words = page.get_text("words")   # (x0, y0, x1, y1, word, blk, ln, wn)
    if not page_words:
        return []
    page_norm = [re.sub(r"[^a-z0-9]", "", w[4].lower()) for w in page_words]
    n_p = len(page_words)

    anchors = [i for i, w in enumerate(page_norm) if w in q_set]
    if not anchors:
        return []

    win_radius = max(len(q_words) * 3, 20)
    best_score = 0.0
    best_start, best_end = 0, 0

    for anchor in anchors[:40]:
        s = max(0, anchor - win_radius // 2)
        e = min(n_p, anchor + win_radius)
        window_set = {w for w in page_norm[s:e] if w}
        inter = len(q_set & window_set)
        score = inter / len(q_set)
        if score > best_score:
            best_score = score
            best_start, best_end = s, e

    if best_score < 0.45:
        return []

    matched = []
    for i in range(best_start, best_end):
        if page_norm[i] in q_set:
            w = page_words[i]
            matched.append(fitz.Rect(w[0], w[1], w[2], w[3]))
    return matched


def _find_quote_rects(page, quote: str) -> list:
    """Three-strategy search for quote text.

    Returns Rect objects only — caller is responsible for drawing highlights.

    1. Exact sentence/fragment search via PyMuPDF search_for()
    2. Word-level sliding window (robust to OCR / whitespace differences)
    3. Block Jaccard fallback (content words only, threshold 0.32)
    """
    import re
    import fitz

    matched: list = []

    # ── Strategy 1: exact sentence/fragment search ───────────────────────────
    for sent in _split_into_sentences(quote):
        sent = sent.strip()
        if len(sent) < 10:
            continue
        for frag in _sentence_fragments(sent):
            frag = frag.strip()
            if not frag:
                continue
            rects = page.search_for(frag)
            if rects:
                matched.extend(rects[:8])
                break

    if matched:
        return matched

    # ── Strategy 2: word-level window ────────────────────────────────────────
    matched = _find_by_word_window(page, quote)
    if matched:
        return matched

    # ── Strategy 3: block Jaccard (content words, tighter threshold) ─────────
    q_words = {w for w in re.findall(r"[a-z]+", quote.lower())
               if w not in _HIGHLIGHT_STOP and len(w) >= 4}
    if not q_words:
        return []

    best_score = 0.0
    best_rect = None
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        btext = block[4]
        if not btext or btext.isspace():
            continue
        b_words = {w for w in re.findall(r"[a-z]+", btext.lower())
                   if w not in _HIGHLIGHT_STOP and len(w) >= 4}
        if not b_words:
            continue
        jaccard = len(q_words & b_words) / max(len(q_words | b_words), 1)
        if jaccard > best_score:
            best_score = jaccard
            best_rect = fitz.Rect(block[0], block[1], block[2], block[3])

    if best_rect and best_score >= 0.32:
        return [best_rect]

    return []


def _draw_highlights(page, rects: list) -> None:
    """Draw semi-transparent yellow highlight over rects using page shapes.

    Uses new_shape/commit so the overlay renders reliably in get_pixmap(),
    unlike highlight annotations which depend on PDF-level appearance streams.
    """
    if not rects:
        return
    import fitz
    shape = page.new_shape()
    for r in rects:
        shape.draw_rect(fitz.Rect(r.x0, r.y0 - 1, r.x1, r.y1 + 1))
    shape.finish(color=None, fill=(1.0, 0.88, 0.0), fill_opacity=0.45, width=0)
    shape.commit()


def _find_rects_from_layout(doc_id: str, page_num: int, quote: str) -> list:
    """Fallback for scanned PDFs: match quote words against OCR layout element bboxes.

    Uses layout_elements.parquet which stores pre-computed bounding boxes from
    the OCR pipeline. Returns Rect objects for elements with >35% word overlap.
    """
    import re
    import fitz

    el_path = RAG_STORE_ROOT / doc_id / "layout_elements.parquet"
    if not el_path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_parquet(el_path, columns=["page_number", "text_preview", "x0", "y0", "x1", "y1"])
    except Exception:
        return []

    page_df = df[df["page_number"] == page_num]
    if page_df.empty:
        return []

    q_words = {w for w in re.findall(r"[a-z]+", quote.lower())
               if len(w) >= 4 and w not in _HIGHLIGHT_STOP}
    if len(q_words) < 2:
        return []

    matched = []
    for _, row in page_df.iterrows():
        text = str(row.get("text_preview") or "")
        if len(text) < 4:
            continue
        t_words = set(re.findall(r"[a-z]+", text.lower()))
        overlap = len(q_words & t_words) / max(len(q_words), 1)
        if overlap >= 0.35:
            matched.append(fitz.Rect(float(row["x0"]), float(row["y0"]),
                                     float(row["x1"]), float(row["y1"])))
    return matched


def _find_quote_rects_from_words(
    doc_id: str, page_num: int, quote: str, page_end: int = -1
) -> tuple:
    """Primary highlight strategy: word-level bbox matching via layout_words.parquet.

    Searches across [page_num, page_end] (inclusive) when page_end > page_num and
    returns the page that best matches the quote.  Hyphen-split tokenisation handles
    compound drug names like 'Amoxicillin-Clavulanate'.

    Returns (best_page_num, rects).  rects is empty when no match is found.
    """
    import re
    import fitz

    words_path = RAG_STORE_ROOT / doc_id / "layout_words.parquet"
    if not words_path.exists():
        return page_num, []
    try:
        import pandas as pd
        df = pd.read_parquet(words_path, columns=["page_number", "text", "x0", "y0", "x1", "y1"])
    except Exception:
        return page_num, []

    def _tokens(w: str) -> list:
        """Split on hyphens first, then strip non-alphanumeric from each part."""
        parts = re.split(r"-", str(w))
        return [re.sub(r"[^a-z0-9]", "", p.lower()) for p in parts
                if re.sub(r"[^a-z0-9]", "", p.lower())]

    # Build query token set from content words only
    q_tokens = []
    for raw in quote.split():
        q_tokens.extend(_tokens(raw))
    q_content = [t for t in q_tokens if len(t) >= 3 and t not in _HIGHLIGHT_STOP]
    if len(q_content) < 2:
        q_content = [t for t in q_tokens if t]
    if not q_content:
        return page_num, []
    q_set = set(q_content)
    # Lower threshold for short quotes (allergies = 2-3 words)
    threshold = 0.30 if len(q_set) <= 4 else 0.40

    pages_to_search = (
        range(page_num, min(page_end, page_num + 6) + 1)
        if page_end > page_num else [page_num]
    )

    best_page  = page_num
    best_rects: list = []
    best_score = 0.0

    for pn in pages_to_search:
        page_df = df[df["page_number"] == pn].reset_index(drop=True)
        if page_df.empty:
            continue

        p_texts = page_df["text"].tolist()
        # Each page-word expands to a set of tokens (handles compound words)
        p_token_sets = [set(_tokens(w)) for w in p_texts]
        n_p = len(p_texts)
        if n_p < 1:
            continue

        n_q = len(q_tokens)
        win = min(n_q + max(4, n_q // 4), n_p)

        pg_best_score, pg_best_i = 0.0, 0
        for i in range(max(n_p - win + 1, 1)):
            window_tokens: set = set()
            for ts in p_token_sets[i:i + win]:
                window_tokens |= ts
            hits  = len(q_set & window_tokens)
            score = hits / len(q_set)
            if score > pg_best_score:
                pg_best_score = score
                pg_best_i     = i

        if pg_best_score > best_score:
            best_score = pg_best_score
            best_page  = pn
            if pg_best_score >= threshold:
                rects = []
                for k in range(pg_best_i, min(pg_best_i + win, n_p)):
                    if q_set & p_token_sets[k]:
                        row = page_df.iloc[k]
                        rects.append(fitz.Rect(float(row["x0"]), float(row["y0"]),
                                               float(row["x1"]), float(row["y1"])))
                best_rects = rects

    if best_score < threshold:
        return page_num, []

    return best_page, best_rects


def _quote_match_count(page, quote: str) -> int:
    """Cheap probe: how many fragment hits does this page have? No annotations."""
    count = 0
    for sent in _split_into_sentences(quote):
        sent = sent.strip()
        if len(sent) < 10:
            continue
        for frag in _sentence_fragments(sent):
            if page.search_for(frag.strip()):
                count += 1
                break
    return count


@app.get("/api/documents/{doc_id}/page/{page_num}")
async def render_page(doc_id: str, page_num: int, q: str = "", page_end: int = -1, crop: int = 0):
    """Render a PDF page with highlighted quote.

    When page_end > page_num and q is supplied, searches the page range for
    the best match. Highlights are drawn as filled shapes (reliable rendering).

    crop=1: crops the pixmap to the highlighted region ± 80pt padding (for
    citation card thumbnails). crop=0: returns the full page at 1.8× (for
    the lightbox / PDF viewer).
    """
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

    # Primary page selection + highlight: word-level bbox search across chunk range
    target_page_num = page_num
    highlight_rects: list = []
    if q:
        target_page_num, highlight_rects = _find_quote_rects_from_words(
            doc_id, page_num, q, page_end=page_end
        )
        page_candidate = doc[target_page_num]
        # Secondary: exact/fragment search via PyMuPDF search_for()
        if not highlight_rects:
            # Re-run page selection with search_for() when word-level fails
            if page_end > page_num:
                end = min(page_end, page_num + 5, len(doc) - 1)
                best_count = _quote_match_count(doc[page_num], q)
                for pn in range(page_num + 1, end + 1):
                    c = _quote_match_count(doc[pn], q)
                    if c > best_count:
                        best_count = c
                        target_page_num = pn
                    if best_count > 0 and c == 0:
                        break
                page_candidate = doc[target_page_num]
            highlight_rects = _find_quote_rects(page_candidate, q)
        # Tertiary: OCR layout element bboxes (block-level fallback)
        if not highlight_rects:
            highlight_rects = _find_rects_from_layout(doc_id, target_page_num, q)

    page = doc[target_page_num]
    if highlight_rects:
        _draw_highlights(page, highlight_rects)

    mat = fitz.Matrix(1.8, 1.8)

    if crop and highlight_rects:
        # Compute bounding box of all highlighted rects + 80pt vertical padding
        bbox = highlight_rects[0]
        for r in highlight_rects[1:]:
            bbox = bbox | r
        pad = 80
        clip = fitz.Rect(
            0,
            max(0.0, bbox.y0 - pad),
            page.rect.width,
            min(page.rect.height, bbox.y1 + pad),
        )
        pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
    else:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

    png = pix.tobytes("png")
    doc.close()
    # No cache for cropped thumbnails (quote-dependent); full page can cache
    cache = "no-store" if (crop and highlight_rects) else "public, max-age=3600"
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": cache})


@app.get("/api/documents/{doc_id}/page/{page_num}/pdf")
async def render_page_pdf(doc_id: str, page_num: int, q: str = ""):
    """Return a single-page searchable PDF with highlight annotations.

    The original text layer is preserved so the viewer can select and copy text.
    Highlight annotations are added using add_highlight_annot() so they render
    in any PDF viewer (browser-native or embedded iframe).
    """
    import fitz
    pdf_path = _get_pdf_path(doc_id)
    if pdf_path is None:
        raise HTTPException(404, "Original PDF not found on disk")
    try:
        src = fitz.open(str(pdf_path))
    except Exception as e:
        raise HTTPException(500, f"Cannot open PDF: {e}")
    if page_num < 0 or page_num >= len(src):
        src.close()
        raise HTTPException(400, f"Page {page_num} out of range (0–{len(src)-1})")

    # Extract just this page into a fresh single-page document
    out = fitz.open()
    out.insert_pdf(src, from_page=page_num, to_page=page_num)
    src.close()

    page = out[0]

    if q:
        _, rects = _find_quote_rects_from_words(doc_id, page_num, q)
        if not rects:
            rects = _find_quote_rects(page, q)
        if not rects:
            rects = _find_rects_from_layout(doc_id, page_num, q)
        if rects:
            annot = page.add_highlight_annot(rects)
            annot.set_colors(stroke=(1.0, 0.85, 0.0))  # yellow
            annot.update()

    pdf_bytes = out.tobytes()
    out.close()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Cache-Control": "no-store"},
    )


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
async def upload_pdf(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    background_tasks: BackgroundTasks = None,
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Auto-start processing (same as POST /api/process)
    placeholder = f"pending_{dest.stem}"
    _emit(placeholder, "layout", "running", "Starting…")
    if background_tasks is not None:
        background_tasks.add_task(_run_full_pipeline, str(dest), placeholder, use_ocr)

    return {
        "filename": file.filename,
        "path": str(dest),
        "placeholder_id": placeholder,
        "use_ocr": use_ocr,
        "message": "Uploaded and processing started.",
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


def _run_full_pipeline(pdf_path: str, doc_id_hint: Optional[str], use_ocr: bool = False) -> None:
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
        # Stage 1 — layout (with optional Google Vision OCR)
        _emit(_id, "layout", "running")
        ocr_engine = None
        if use_ocr:
            try:
                from nitrag.google_vision_ocr import GoogleVisionOCR
                _ocr = GoogleVisionOCR()
                if _ocr.is_available():
                    ocr_engine = _ocr
                    print(f"[pipeline] Google Vision OCR enabled for {pdf.name}")
                else:
                    print("[pipeline] Google Vision OCR requested but credentials unavailable — skipping")
            except Exception as _oe:
                print(f"[pipeline] Could not initialise Google Vision OCR: {_oe}")
        extractor = PyMuPDFLayoutExtractor(
            encoding_model_name="gpt-4o", root_dir=RAG_STORE_ROOT, ocr_engine=ocr_engine
        )
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
        # Stage 1 (layout extractor) already wrote tokens.i32, pages.parquet,
        # elements.parquet, and manifest.json — including any OCR text.
        # Loading from those files preserves OCR output; re-ingesting via
        # ingest_pdf() would overwrite tokens with PyMuPDF-only output and
        # discard all OCR text, producing zero chunks for scanned documents.
        _emit(_id, "chunks", "running")
        store = PdfTokenStore(encoding_model_name="gpt-4o", root_dir=RAG_STORE_ROOT)
        store.load(_id)
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
    use_ocr: bool = False


@app.post("/api/process")
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    pdf = Path(request.pdf_path)
    if not pdf.exists():
        raise HTTPException(404, f"File not found: {request.pdf_path}")
    if not pdf.suffix.lower() == ".pdf":
        raise HTTPException(400, "Only PDF files are supported")

    placeholder = f"pending_{pdf.stem}"
    _emit(placeholder, "layout", "running", "Starting…")

    background_tasks.add_task(_run_full_pipeline, str(pdf), placeholder, request.use_ocr)
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


def _serialize_response(response, sentence_citations=None) -> Dict[str, Any]:
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
        "sentence_citations": sentence_citations or [],
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

    # Post-generation: map each answer sentence to exact source sentences
    try:
        from nitrag.generation_manager import build_sentence_citations
        sent_cites = build_sentence_citations(response.answer, response.context)
    except Exception:
        sent_cites = []

    result = _serialize_response(response, sentence_citations=sent_cites)
    result["doc_id"] = request.doc_id
    return result

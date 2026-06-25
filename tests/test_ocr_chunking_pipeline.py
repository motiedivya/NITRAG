"""
Tests that the OCR → chunking pipeline produces chunks.

Before the fix, PyMuPDFLayoutExtractor.extract() wrote OCR tokens to
tokens.i32 and OCR pages/elements to layout_*.parquet, but the server
then called store.ingest_pdf() which overwrote tokens.i32 without OCR
and wrote empty pages.parquet / elements.parquet — producing zero chunks
for scanned documents.

The fix: the extractor now also writes pages.parquet, elements.parquet,
and manifest.json so that store.load() can be used instead of ingest_pdf().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Sample_ocr.pdf has native text — used for native-text and file-writing tests.
OCR_PDF = DATA_DIR / "Sample_ocr.pdf"


def _make_image_only_pdf(path: Path) -> None:
    """Create a minimal 1-page PDF whose content is a rasterised image (no native text)."""
    import fitz

    # Build a valid PNG via PyMuPDF's own Pixmap — avoids manual PNG encoding.
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4))
    pix.set_rect(pix.irect, (255, 255, 255))
    png_bytes = pix.tobytes("png")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(50, 50, 545, 792), stream=png_bytes)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Minimal fake OCR engine (no Google credentials needed)
# ---------------------------------------------------------------------------

class FakeOCREngine:
    """Returns hard-coded OCR paragraphs instead of calling Google Vision."""

    def is_available(self) -> bool:
        return True

    def ocr_page_image(
        self,
        img_bytes: bytes,
        page_number: int,
        page_width: float,
        page_height: float,
        render_scale: float,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "element_type": "block",
                "page_number": page_number,
                "block_number": 0,
                "line_number": 0,
                "text": "Patient has hypertension and diabetes mellitus.",
                "x0": 50.0, "y0": 50.0,
                "x1": page_width - 50.0, "y1": 150.0,
                "source": "google_vision",
                "confidence": 0.99,
                "word_dicts": [
                    {"text": "Patient", "x0": 50.0, "y0": 50.0, "x1": 110.0, "y1": 80.0,
                     "line_number": 0, "confidence": 0.99},
                ],
            },
            {
                "element_type": "block",
                "page_number": page_number,
                "block_number": 1,
                "line_number": 0,
                "text": "Blood pressure 145/90 mmHg. Heart rate 88 bpm.",
                "x0": 50.0, "y0": 160.0,
                "x1": page_width - 50.0, "y1": 220.0,
                "source": "google_vision",
                "confidence": 0.97,
                "word_dicts": [],
            },
        ]


# ---------------------------------------------------------------------------
# Helper: run the full extract → load → chunk cycle
# ---------------------------------------------------------------------------

def _run_extract_and_load(pdf_path: Path, tmp_path: Path, ocr_engine=None):
    from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
    from nitrag.chunk_manager import PdfTokenStore, ChunkManager, register_default_chunkers

    extractor = PyMuPDFLayoutExtractor(
        encoding_model_name="gpt-4o",
        root_dir=tmp_path,
        ocr_engine=ocr_engine,
    )
    manifest = extractor.extract(pdf_path, overwrite=True)
    doc_id = manifest["document_id"]

    store = PdfTokenStore(encoding_model_name="gpt-4o", root_dir=tmp_path)
    store.load(doc_id)

    mgr = ChunkManager(store)
    register_default_chunkers(mgr)
    mgr.execute_all(continue_on_error=False)

    return doc_id, store, mgr


# ---------------------------------------------------------------------------
# Tests: extractor writes store-compatible files
# ---------------------------------------------------------------------------

class TestExtractorWritesStoreFiles:
    def test_manifest_json_written(self, tmp_path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        ex = PyMuPDFLayoutExtractor(root_dir=tmp_path)
        manifest = ex.extract(OCR_PDF, overwrite=True)
        doc_id = manifest["document_id"]
        assert (tmp_path / doc_id / "manifest.json").exists()

    def test_pages_parquet_written(self, tmp_path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        ex = PyMuPDFLayoutExtractor(root_dir=tmp_path)
        manifest = ex.extract(OCR_PDF, overwrite=True)
        doc_id = manifest["document_id"]
        assert (tmp_path / doc_id / "pages.parquet").exists()

    def test_elements_parquet_written(self, tmp_path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        ex = PyMuPDFLayoutExtractor(root_dir=tmp_path)
        manifest = ex.extract(OCR_PDF, overwrite=True)
        doc_id = manifest["document_id"]
        assert (tmp_path / doc_id / "elements.parquet").exists()

    def test_manifest_json_has_required_keys(self, tmp_path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        ex = PyMuPDFLayoutExtractor(root_dir=tmp_path)
        manifest = ex.extract(OCR_PDF, overwrite=True)
        doc_id = manifest["document_id"]
        store_manifest = json.loads((tmp_path / doc_id / "manifest.json").read_text())
        for key in ("document_id", "tokens_path", "pages_path", "elements_path",
                    "chunks_dir", "total_tokens"):
            assert key in store_manifest, f"manifest.json missing key: {key}"

    def test_store_load_succeeds_after_extraction(self, tmp_path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        from nitrag.chunk_manager import PdfTokenStore
        ex = PyMuPDFLayoutExtractor(root_dir=tmp_path)
        manifest = ex.extract(OCR_PDF, overwrite=True)
        store = PdfTokenStore(root_dir=tmp_path)
        store.load(manifest["document_id"])  # must not raise
        assert store.document_id == manifest["document_id"]


# ---------------------------------------------------------------------------
# Tests: native-text PDF produces chunks via load() (regression guard)
# Note: Sample_ocr.pdf has native text on most pages, so no OCR engine needed.
# ---------------------------------------------------------------------------

class TestNativePdfChunking:
    def test_fixed_512_chunks_produced(self, tmp_path):
        doc_id, store, mgr = _run_extract_and_load(OCR_PDF, tmp_path)
        chunks = mgr.chunks("fixed_512")
        assert len(chunks) > 0, "fixed_512 chunker produced no chunks"

    def test_page_based_chunks_produced(self, tmp_path):
        doc_id, store, mgr = _run_extract_and_load(OCR_PDF, tmp_path)
        chunks = mgr.chunks("page_based")
        assert len(chunks) > 0, "page_based chunker produced no chunks"

    def test_block_based_chunks_produced(self, tmp_path):
        doc_id, store, mgr = _run_extract_and_load(OCR_PDF, tmp_path)
        chunks = mgr.chunks("block_based")
        assert len(chunks) > 0, "block_based chunker produced no chunks"

    def test_total_tokens_positive(self, tmp_path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        from nitrag.chunk_manager import PdfTokenStore
        ex = PyMuPDFLayoutExtractor(root_dir=tmp_path)
        manifest = ex.extract(OCR_PDF, overwrite=True)
        store = PdfTokenStore(root_dir=tmp_path)
        store.load(manifest["document_id"])
        assert store.total_tokens > 0


# ---------------------------------------------------------------------------
# Tests: OCR-injected pages produce chunks  ← the core regression being fixed
#
# Strategy: pass FakeOCREngine to force injection on image-only pages, then
# confirm chunks come out.  The key assertion is that tokens written during
# OCR injection survive into the PdfTokenStore — i.e. store.ingest_pdf() is
# no longer called after extract().
# ---------------------------------------------------------------------------

class TestOCRChunkingPipeline:
    """
    Uses a synthetic image-only PDF so FakeOCREngine actually fires on every
    page (OCR only triggers when a page has zero native text tokens).
    """

    @pytest.fixture
    def image_pdf(self, tmp_path) -> Path:
        p = tmp_path / "image_only.pdf"
        _make_image_only_pdf(p)
        return p

    def _make_extractor(self, tmp_path: Path):
        from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
        return PyMuPDFLayoutExtractor(
            root_dir=tmp_path,
            ocr_engine=FakeOCREngine(),
        )

    def test_ocr_manifest_json_written(self, tmp_path, image_pdf):
        ex = self._make_extractor(tmp_path)
        manifest = ex.extract(image_pdf, overwrite=True)
        assert (tmp_path / manifest["document_id"] / "manifest.json").exists()

    def test_ocr_pages_parquet_written(self, tmp_path, image_pdf):
        ex = self._make_extractor(tmp_path)
        manifest = ex.extract(image_pdf, overwrite=True)
        assert (tmp_path / manifest["document_id"] / "pages.parquet").exists()

    def test_ocr_elements_parquet_written(self, tmp_path, image_pdf):
        ex = self._make_extractor(tmp_path)
        manifest = ex.extract(image_pdf, overwrite=True)
        assert (tmp_path / manifest["document_id"] / "elements.parquet").exists()

    def test_ocr_store_loads_with_positive_tokens(self, tmp_path, image_pdf):
        from nitrag.chunk_manager import PdfTokenStore
        ex = self._make_extractor(tmp_path)
        manifest = ex.extract(image_pdf, overwrite=True)
        store = PdfTokenStore(root_dir=tmp_path)
        store.load(manifest["document_id"])
        assert store.total_tokens > 0, "No tokens found after OCR injection"

    def test_ocr_fixed_512_chunks_produced(self, tmp_path, image_pdf):
        """Core regression: fixed-token chunker must yield chunks after OCR."""
        doc_id, store, mgr = _run_extract_and_load(image_pdf, tmp_path, FakeOCREngine())
        chunks = mgr.chunks("fixed_512")
        assert len(chunks) > 0, (
            "fixed_512 chunker produced 0 chunks after OCR injection — "
            "store.ingest_pdf() may have overwritten OCR tokens"
        )

    def test_ocr_page_based_chunks_produced(self, tmp_path, image_pdf):
        doc_id, store, mgr = _run_extract_and_load(image_pdf, tmp_path, FakeOCREngine())
        chunks = mgr.chunks("page_based")
        assert len(chunks) > 0, "page_based chunker produced 0 chunks after OCR injection"

    def test_ocr_block_based_chunks_produced(self, tmp_path, image_pdf):
        doc_id, store, mgr = _run_extract_and_load(image_pdf, tmp_path, FakeOCREngine())
        chunks = mgr.chunks("block_based")
        assert len(chunks) > 0, "block_based chunker produced 0 chunks after OCR injection"

    def test_ocr_chunks_decode_to_ocr_text(self, tmp_path, image_pdf):
        """Decoded chunk text must include content injected by the fake OCR engine."""
        doc_id, store, mgr = _run_extract_and_load(image_pdf, tmp_path, FakeOCREngine())
        chunks = mgr.chunks("fixed_512")
        assert chunks, "no chunks to decode"
        all_text = "".join(
            store.decode_span(c["start_index"], c["end_index"]) for c in chunks
        )
        assert "hypertension" in all_text.lower(), (
            "OCR-injected text not found in decoded chunks; "
            "tokens were likely overwritten by a stale ingest_pdf() call"
        )

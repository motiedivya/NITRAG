"""
google_vision_ocr.py — Google Cloud Vision OCR integration for NIT-RAG.

Renders scanned/image-only PDF pages to PNG via PyMuPDF and sends them to
the Vision API DOCUMENT_TEXT_DETECTION endpoint.  Returns per-page lists of
word/line dicts that are structurally compatible with the element rows produced
by PyMuPDFLayoutExtractor so they can be injected into the same parquet tables.

Usage (from within the pipeline):
    ocr = GoogleVisionOCR(credentials_path="ocr-neuralit-4e01e06ccf84.json")
    # ocr.is_available() → True when credentials load successfully
    pages = ocr.ocr_pdf(pdf_path, page_numbers=[0, 3, 7])
    # pages[0] → list of element-like dicts for page 0
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Render resolution for OCR — 200 DPI equivalent in fitz units (72 pt/inch)
_OCR_SCALE = 200 / 72  # ≈ 2.78×


_CREDENTIALS_PATH_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
_DEFAULT_KEY_PATH = Path(__file__).parent.parent / "ocr-neuralit-4e01e06ccf84.json"


class GoogleVisionOCR:
    """
    Wraps the Google Cloud Vision DOCUMENT_TEXT_DETECTION API.

    Parameters
    ----------
    credentials_path : path to service-account JSON key (falls back to
        ``GOOGLE_APPLICATION_CREDENTIALS`` env var, then the repo-root default).
    """

    def __init__(self, credentials_path: Optional[str | Path] = None):
        self._cred_path: Optional[Path] = None
        self._client = None  # lazy

        # Resolve credentials
        if credentials_path:
            self._cred_path = Path(credentials_path)
        elif os.environ.get(_CREDENTIALS_PATH_ENV):
            self._cred_path = Path(os.environ[_CREDENTIALS_PATH_ENV])
        elif _DEFAULT_KEY_PATH.exists():
            self._cred_path = _DEFAULT_KEY_PATH

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if credentials exist and the Vision client can be built."""
        try:
            self._ensure_client()
            return self._client is not None
        except Exception:
            return False

    def ocr_pdf(
        self,
        pdf_path: str | Path,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        OCR the requested pages of a PDF.

        Parameters
        ----------
        pdf_path     : path to the PDF file.
        page_numbers : 0-based page indices to process.  None → all pages.

        Returns
        -------
        dict mapping page_number → list of element-like dicts (see _vision_to_elements).
        """
        import fitz  # PyMuPDF

        self._ensure_client()
        pdf_path = Path(pdf_path)
        result: Dict[int, List[Dict[str, Any]]] = {}

        doc = fitz.open(str(pdf_path))
        try:
            targets: List[int] = list(page_numbers) if page_numbers is not None else list(range(len(doc)))
            for pg_num in targets:
                if pg_num < 0 or pg_num >= len(doc):
                    continue
                page = doc[pg_num]
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
                mat = fitz.Matrix(_OCR_SCALE, _OCR_SCALE)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img_bytes = pix.tobytes("png")
                elems = self._ocr_image_bytes(img_bytes, pg_num, page_width, page_height, _OCR_SCALE)
                result[pg_num] = elems
                print(f"[vision_ocr] page {pg_num}: {len(elems)} elements")
        finally:
            doc.close()

        return result

    def ocr_page_image(
        self,
        img_bytes: bytes,
        page_number: int,
        page_width: float,
        page_height: float,
        render_scale: float = _OCR_SCALE,
    ) -> List[Dict[str, Any]]:
        """OCR an already-rendered page image (PNG bytes)."""
        self._ensure_client()
        return self._ocr_image_bytes(img_bytes, page_number, page_width, page_height, render_scale)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self._cred_path is None or not self._cred_path.exists():
            raise RuntimeError(
                "Google Vision credentials not found.  Set GOOGLE_APPLICATION_CREDENTIALS "
                "or pass credentials_path= to GoogleVisionOCR()."
            )
        from google.cloud import vision
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            str(self._cred_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._client = vision.ImageAnnotatorClient(credentials=creds)

    def _ocr_image_bytes(
        self,
        img_bytes: bytes,
        page_number: int,
        page_width: float,
        page_height: float,
        render_scale: float,
    ) -> List[Dict[str, Any]]:
        from google.cloud import vision

        image = vision.Image(content=img_bytes)
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Vision API error on page {page_number}: {response.error.message}")

        return _vision_response_to_elements(
            response, page_number, page_width, page_height, render_scale
        )


# ── Response → element conversion ─────────────────────────────────────────────

def _poly_to_bbox(vertices, page_width: float, page_height: float, scale: float):
    """Convert Vision API bounding poly vertices to (x0, y0, x1, y1) in PDF units."""
    xs = [v.x / scale for v in vertices]
    ys = [v.y / scale for v in vertices]
    return (
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(page_width, max(xs)),
        min(page_height, max(ys)),
    )


def _vision_response_to_elements(
    response,
    page_number: int,
    page_width: float,
    page_height: float,
    scale: float,
) -> List[Dict[str, Any]]:
    """
    Convert a Vision DOCUMENT_TEXT_DETECTION response to a list of element-like
    dicts keyed the same way as PyMuPDFLayoutExtractor elements.

    Granularity: one dict per PARAGRAPH (= rough block), with text being the
    full paragraph text.  We also emit one dict per WORD for the word table.
    """
    elements: List[Dict[str, Any]] = []
    full_text = response.full_text_annotation

    if not full_text or not full_text.pages:
        return elements

    vpage = full_text.pages[0]
    block_number = 0

    for block in vpage.blocks:
        for para in block.paragraphs:
            # Collect words
            words_text: List[str] = []
            word_dicts: List[Dict[str, Any]] = []

            for wi, word in enumerate(para.words):
                word_text = "".join(
                    symbol.text for symbol in word.symbols
                )
                words_text.append(word_text)
                wx0, wy0, wx1, wy1 = _poly_to_bbox(
                    word.bounding_box.vertices, page_width, page_height, scale
                )
                word_dicts.append({
                    "element_type": "word",
                    "page_number": page_number,
                    "block_number": block_number,
                    "line_number": wi,
                    "text": word_text,
                    "x0": wx0, "y0": wy0, "x1": wx1, "y1": wy1,
                    "source": "google_vision",
                    "confidence": getattr(word, "confidence", None),
                })

            para_text = " ".join(words_text).strip()
            if not para_text:
                continue

            px0, py0, px1, py1 = _poly_to_bbox(
                para.bounding_box.vertices, page_width, page_height, scale
            )

            elements.append({
                "element_type": "block",  # maps to a layout block
                "page_number": page_number,
                "block_number": block_number,
                "line_number": 0,
                "text": para_text,
                "x0": px0, "y0": py0, "x1": px1, "y1": py1,
                "source": "google_vision",
                "confidence": getattr(para, "confidence", None),
                "word_dicts": word_dicts,
            })
            block_number += 1

    return elements

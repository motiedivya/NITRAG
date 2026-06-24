"""
pdf_ingestion.py — PDF Ingestion Pipeline for NIT-RAG

Sits on top of PyMuPDFLayoutExtractor and adds:
  - Text normalization  (ligatures, Unicode NFC, zero-width chars, control chars)
  - Page type detection (native | scanned_ocr | mixed | image_only)
  - Multi-column layout detection + reading-order correction
  - OCR-aware heading detection (position/shape-based; works without font cues)
  - Per-page quality metrics

Drop-in replacement for PyMuPDFLayoutExtractor.extract():

    pipeline = PDFIngestionPipeline(root_dir="rag_store")
    result   = pipeline.ingest("path/to/doc.pdf")
    # result has the same shape as PyMuPDFLayoutExtractor.extract() output
    # but the stored parquets carry additional augmentation columns.

New columns added per table
───────────────────────────
layout_spans.parquet
    normalized_text             str   text after normalization

layout_elements.parquet
    normalized_text             str   text after normalization
    reading_order_index         int   position in correct reading order on the page
    heading_score_final         float best heading score (font-based OR position-based)
    is_heading_candidate_final  bool  final heading flag

layout_pages.parquet
    page_type                   str   "native" | "scanned_ocr" | "mixed" | "image_only"
    column_count                int   1 or 2 (multi-column detection)
    font_variety_score          float distinct (font, size) pairs / total spans
    font_size_cv                float coefficient of variation of font sizes on the page
    image_area_ratio            float image area / page area
    has_ocr_font_indicator      bool  any span uses a known OCR font name
    distinct_font_count         int   unique font names on the page
    ocr_quality_score           float proxy quality score (1.0 for native pages)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq

from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor, write_parquet


# ---------------------------------------------------------------------------
# Page type enum
# ---------------------------------------------------------------------------

class PageType(str, Enum):
    NATIVE     = "native"       # Regular PDF with rich font / layout metadata
    SCANNED_OCR = "scanned_ocr" # Scanned image + OCR text layer
    MIXED      = "mixed"        # Some pages native, some scanned (classified per-page)
    IMAGE_ONLY = "image_only"   # No extractable text at all


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IngestionConfig:
    encoding_model_name: str = "gpt-4o"

    # ---- text normalization ----
    normalize_text: bool = True
    # Attempt to join words broken across lines by a trailing hyphen.
    # Off by default — risky on medical abbreviations (e.g. "anti-\ncoagulant").
    join_broken_hyphens: bool = False

    # ---- column detection ----
    detect_columns: bool = True
    # Normalised x range (0-1) in which to look for a column gap.
    column_center_zone: Tuple[float, float] = (0.30, 0.70)
    # Minimum blocks in each half to declare a 2-column layout.
    min_blocks_per_column: int = 3

    # ---- OCR-aware heading detection ----
    ocr_aware_headings: bool = True
    # Gap above the line must be >= this multiple of the page median line-gap.
    heading_gap_ratio: float = 1.4
    heading_max_words: int = 15
    # Same threshold used by the font-based detector in document_metadata_extractor.
    heading_min_score: float = 0.45

    # ---- page-type thresholds ----
    # Pages with fewer unique (font, size) pairs than this fraction of total spans
    # are flagged as possibly OCR'd.
    scanned_font_variety_threshold: float = 0.06
    # Pages where images cover more than this fraction of the area → OCR indicator.
    scanned_image_area_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_LIGATURE_MAP: Dict[str, str] = {
    "ﬀ": "ff",   # ﬀ
    "ﬁ": "fi",   # ﬁ
    "ﬂ": "fl",   # ﬂ
    "ﬃ": "ffi",  # ﬃ
    "ﬄ": "ffl",  # ﬄ
    "ﬅ": "st",   # ﬅ
    "ﬆ": "st",   # ﬆ
    "æ": "ae",   # æ — OCR sometimes emits this in clinical text
    "œ": "oe",   # œ
    "’": "'",    # right single quotation mark → apostrophe
    "‘": "'",    # left  single quotation mark
    "“": '"',    # left  double quotation mark
    "”": '"',    # right double quotation mark
    "–": "-",    # en dash
    "—": "-",    # em dash
    "·": ".",    # middle dot (common OCR noise)
}

# Zero-width space / joiner / non-joiner / BOM / soft-hyphen
_ZERO_WIDTH = re.compile(r"[​‌‍﻿­]")
# Control characters (keep \t = 0x09, \n = 0x0a)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r" {2,}")


class TextNormalizer:
    """
    Lightweight, deterministic text normaliser for OCR and native PDF text.

    Applies, in order:
      1. Unicode NFC (canonical composition)
      2. Ligature expansion via a translation table
      3. Zero-width / soft-hyphen removal
      4. Non-printable control character removal (preserves \\t and \\n)
      5. Intra-line multi-space collapse
      6. Optional broken-hyphen joining across lines
    """

    def __init__(self, join_broken_hyphens: bool = False):
        self._join_hyphens = join_broken_hyphens
        self._table = str.maketrans(_LIGATURE_MAP)

    def normalize(self, text: str) -> str:
        if not text:
            return text
        text = unicodedata.normalize("NFC", text)
        text = text.translate(self._table)
        text = _ZERO_WIDTH.sub("", text)
        text = _CONTROL.sub("", text)
        # Collapse runs of spaces within each line only.
        lines = text.split("\n")
        lines = [_MULTI_SPACE.sub(" ", ln) for ln in lines]
        return "\n".join(lines)

    def normalize_block(self, lines: List[str]) -> List[str]:
        """
        Normalise a list of line strings.
        With join_broken_hyphens=True, merges lines split by a trailing hyphen.
        """
        normed = [self.normalize(ln) for ln in lines]

        if not self._join_hyphens:
            return normed

        result: List[str] = []
        skip_next = False
        for i, ln in enumerate(normed):
            if skip_next:
                skip_next = False
                continue
            stripped = ln.rstrip()
            if stripped.endswith("-") and i + 1 < len(normed):
                result.append(stripped[:-1] + normed[i + 1].lstrip())
                skip_next = True
            else:
                result.append(ln)
        return result


# ---------------------------------------------------------------------------
# Page type detection
# ---------------------------------------------------------------------------

_OCR_FONT_RE = re.compile(
    r"(glyphless|glyph.less|ocr|tesseract|abbyy|recognita|acrobat.capture)",
    re.IGNORECASE,
)


class PageTypeDetector:
    """
    Classifies each page as NATIVE, SCANNED_OCR, MIXED, or IMAGE_ONLY.

    Heuristics (in order of confidence):
      1. Any span uses a known OCR font name → SCANNED_OCR
      2. No text at all, but images present → IMAGE_ONLY
      3. Very low font variety AND images above threshold → SCANNED_OCR
      4. Very low font variety AND size CV near zero → SCANNED_OCR
         (all-same-font, all-same-size — classic single-font OCR output)
      5. Images present but font variety is healthy → MIXED
      6. Otherwise → NATIVE
    """

    def __init__(self, config: IngestionConfig):
        self.cfg = config

    def detect(
        self,
        page_spans: List[Dict[str, Any]],
        page_images: List[Dict[str, Any]],
        page_width: float,
        page_height: float,
    ) -> Tuple[PageType, Dict[str, Any]]:
        page_area = max(1.0, page_width * page_height)

        image_area = sum(
            max(0.0, float(img.get("x1") or 0) - float(img.get("x0") or 0)) *
            max(0.0, float(img.get("y1") or 0) - float(img.get("y0") or 0))
            for img in page_images
        )
        image_area_ratio = image_area / page_area

        metrics: Dict[str, Any] = {
            "font_variety_score": 0.0,
            "image_area_ratio": round(image_area_ratio, 4),
            "size_cv": 0.0,
            "has_ocr_font": False,
            "distinct_fonts": 0,
            "span_count": len(page_spans),
        }

        if not page_spans:
            ptype = PageType.IMAGE_ONLY if image_area_ratio > 0.1 else PageType.NATIVE
            return ptype, metrics

        fonts = [s.get("font_name") or "" for s in page_spans]
        sizes = [float(s.get("font_size") or 0.0) for s in page_spans]

        distinct_fonts = len({f for f in fonts if f})

        # Unique (font_name, size_bucket) pairs — bucket font size to nearest 0.5 pt.
        font_size_pairs = {
            (s.get("font_name") or "", round(float(s.get("font_size") or 0.0) * 2) / 2)
            for s in page_spans
        }
        font_variety_score = len(font_size_pairs) / max(1, len(page_spans))

        valid_sizes = [sz for sz in sizes if sz > 0]
        mean_size = float(np.mean(valid_sizes)) if valid_sizes else 0.0
        size_cv = (
            float(np.std(valid_sizes)) / mean_size
            if len(valid_sizes) > 1 and mean_size > 0
            else 0.0
        )

        has_ocr_font = any(_OCR_FONT_RE.search(f) for f in fonts if f)

        metrics.update({
            "font_variety_score": round(font_variety_score, 4),
            "size_cv": round(size_cv, 4),
            "has_ocr_font": has_ocr_font,
            "distinct_fonts": distinct_fonts,
        })

        low_variety = font_variety_score < self.cfg.scanned_font_variety_threshold
        has_images = image_area_ratio > self.cfg.scanned_image_area_threshold

        if has_ocr_font:
            return PageType.SCANNED_OCR, metrics
        if not page_spans:
            return PageType.IMAGE_ONLY, metrics
        if low_variety and has_images:
            return PageType.SCANNED_OCR, metrics
        if low_variety and size_cv < 0.05 and distinct_fonts <= 2:
            # Extremely uniform text — hallmark of single-font OCR output.
            return PageType.SCANNED_OCR, metrics
        if has_images and not low_variety:
            return PageType.MIXED, metrics
        return PageType.NATIVE, metrics


# ---------------------------------------------------------------------------
# Column detection + reading order
# ---------------------------------------------------------------------------

class ColumnDetector:
    """
    Detects 2-column page layouts and assigns a reading_order_index to each
    element so downstream consumers can iterate in correct left→right order.
    """

    def __init__(self, config: IngestionConfig):
        self.cfg = config

    def detect_column_count(
        self,
        block_elements: List[Dict[str, Any]],
        page_width: float,
    ) -> int:
        """Returns 1 (single-column) or 2 (double-column)."""
        min_blocks = self.cfg.min_blocks_per_column * 2
        if page_width <= 0 or len(block_elements) < min_blocks:
            return 1

        centers_norm = [
            (float(e.get("x0") or 0) + float(e.get("x1") or 0)) / 2.0 / page_width
            for e in block_elements
            if e.get("x0") is not None and e.get("x1") is not None
        ]
        if not centers_norm:
            return 1

        lo, hi = self.cfg.column_center_zone
        center_zone = [c for c in centers_norm if lo <= c <= hi]
        left_blocks  = [c for c in centers_norm if c < lo]
        right_blocks = [c for c in centers_norm if c > hi]

        both_sides = (
            len(left_blocks)  >= self.cfg.min_blocks_per_column and
            len(right_blocks) >= self.cfg.min_blocks_per_column
        )
        sparse_center = len(center_zone) <= max(2, int(0.15 * len(centers_norm)))

        return 2 if (both_sides and sparse_center) else 1

    def build_reading_order(
        self,
        elements: List[Dict[str, Any]],
        column_count: int,
        page_number: int,
    ) -> Dict[int, int]:
        """
        Returns {element_id: reading_order_index} for all elements on page_number.
        For single-column pages: top-to-bottom, left-to-right.
        For two-column pages: left column (top-to-bottom) then right column.
        """
        page_els = [e for e in elements if int(e.get("page_number") or -1) == page_number]

        def _sort_key(e: Dict[str, Any]) -> Tuple[float, float]:
            return (float(e.get("y0") or 0), float(e.get("x0") or 0))

        if column_count == 1 or not page_els:
            ordered = sorted(page_els, key=_sort_key)
            return {e["element_id"]: i for i, e in enumerate(ordered)}

        # Determine split x from the median of block x-centres.
        x_centres = [
            (float(e.get("x0") or 0) + float(e.get("x1") or 0)) / 2.0
            for e in page_els
            if e.get("x0") is not None
        ]
        split_x = float(np.median(x_centres)) if x_centres else 0.0

        left  = sorted([e for e in page_els if (float(e.get("x0") or 0) + float(e.get("x1") or 0)) / 2 <= split_x], key=_sort_key)
        right = sorted([e for e in page_els if (float(e.get("x0") or 0) + float(e.get("x1") or 0)) / 2 >  split_x], key=_sort_key)

        ordered = left + right
        return {e["element_id"]: i for i, e in enumerate(ordered)}


# ---------------------------------------------------------------------------
# OCR-aware heading detection
# ---------------------------------------------------------------------------

_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*|[A-Z])[\).\s\-]+")


class OCRAwareHeadingDetector:
    """
    Heading detector for pages where font metadata is unreliable (scanned/OCR'd).

    Scoring signal                        Weight
    ──────────────────────────────────────────────
    Large vertical gap above the line     up to 0.35
    Short word count (≤ max_words)              0.15
    Very short word count (≤ 5 words)           0.10
    ALL CAPS with ≤ 10 words                    0.20
    Title Case with ≤ 10 words                  0.10
    Ends with ":"  and ≤ 12 words               0.15
    Doesn't end with "." or ","                 0.05
    Numbered heading pattern                    0.10
    Over-long line (> max_words)               -0.30
    Very long text (> 160 chars)               -0.20

    For NATIVE pages the existing font-based score is used unchanged.
    """

    def __init__(self, config: IngestionConfig):
        self.cfg = config

    def annotate(
        self,
        elements: List[Dict[str, Any]],
        page_type_map: Dict[int, PageType],
    ) -> List[Dict[str, Any]]:
        # Build a map of line elements per page.
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for e in elements:
            if e.get("element_type") == "line":
                by_page.setdefault(int(e["page_number"]), []).append(e)

        scores: Dict[int, Tuple[float, bool]] = {}  # element_id → (score, is_heading)

        for page_num, page_lines in by_page.items():
            ptype = page_type_map.get(page_num, PageType.NATIVE)
            sorted_lines = sorted(page_lines, key=lambda e: (float(e.get("y0") or 0), float(e.get("x0") or 0)))

            if ptype == PageType.NATIVE:
                for e in sorted_lines:
                    scores[e["element_id"]] = (
                        float(e.get("heading_score") or 0.0),
                        bool(e.get("is_heading_candidate") or False),
                    )
                continue

            # Compute typical vertical gap between consecutive lines on this page.
            gaps = [
                float(sorted_lines[i].get("y0") or 0) - float(sorted_lines[i - 1].get("y1") or 0)
                for i in range(1, len(sorted_lines))
                if (float(sorted_lines[i].get("y0") or 0) - float(sorted_lines[i - 1].get("y1") or 0)) > 0
            ]
            median_gap = float(np.median(gaps)) if gaps else 2.0

            for i, e in enumerate(sorted_lines):
                text = (e.get("normalized_text") or e.get("text") or "").strip()
                words = text.split()
                word_count = len(words)

                if not text or word_count == 0:
                    scores[e["element_id"]] = (0.0, False)
                    continue

                if e.get("is_repeated_header_candidate") or e.get("is_repeated_footer_candidate"):
                    scores[e["element_id"]] = (0.0, False)
                    continue

                gap_above = (
                    float(e.get("y0") or 0) - float(sorted_lines[i - 1].get("y1") or 0)
                    if i > 0
                    else float(e.get("y0") or 0)  # distance from top of page
                )
                gap_ratio = gap_above / max(0.5, median_gap)

                score = 0.0

                # Gap signal
                if gap_ratio >= self.cfg.heading_gap_ratio:
                    score += min(0.35, 0.08 * gap_ratio)

                # Length signals
                if 1 <= word_count <= self.cfg.heading_max_words:
                    score += 0.15
                if word_count <= 5:
                    score += 0.10

                # Case signals
                letters       = [c for c in text if c.isalpha()]
                upper_letters = [c for c in text if c.isupper()]

                if letters and len(upper_letters) / max(1, len(letters)) > 0.85 and word_count <= 10:
                    score += 0.20   # ALL CAPS
                elif text.istitle() and word_count <= 10:
                    score += 0.10

                # Colon ending
                if text.endswith(":") and word_count <= 12:
                    score += 0.15

                # Doesn't end with sentence-final punctuation
                if not text.endswith(".") and not text.endswith(","):
                    score += 0.05

                # Numbered heading
                if _NUMBERED_HEADING_RE.match(text):
                    score += 0.10

                # Penalties
                if word_count > self.cfg.heading_max_words:
                    score -= 0.30
                if len(text) > 160:
                    score -= 0.20

                score = max(0.0, min(1.0, score))
                scores[e["element_id"]] = (score, score >= self.cfg.heading_min_score)

        # Write scores back onto every element.
        result: List[Dict[str, Any]] = []
        for e in elements:
            e = dict(e)
            eid = e["element_id"]
            if eid in scores:
                sc, is_h = scores[eid]
                e["heading_score_final"]        = sc
                e["is_heading_candidate_final"] = is_h
            else:
                # Blocks and page-level elements: pass through existing or default.
                e["heading_score_final"]        = float(e.get("heading_score") or 0.0)
                e["is_heading_candidate_final"] = bool(e.get("is_heading_candidate") or False)
            result.append(e)
        return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class PDFIngestionPipeline:
    """
    Complete PDF ingestion pipeline for NIT-RAG.

    Wraps PyMuPDFLayoutExtractor, then augments its parquet outputs in-place
    with normalisation, page typing, column detection, reading order, and
    OCR-aware heading detection.

    The returned manifest is identical in structure to what
    PyMuPDFLayoutExtractor.extract() returns, plus an
    'ingestion_augmentation' key with per-run statistics.

    Typical usage
    ─────────────
    # In the main pipeline:
    from nitrag.pdf_ingestion import PDFIngestionPipeline, IngestionConfig
    pipeline = PDFIngestionPipeline(root_dir=RAG_STORE)
    manifest = pipeline.ingest(PDF_PATH, overwrite=True)

    # Custom config:
    cfg = IngestionConfig(detect_columns=False, join_broken_hyphens=True)
    pipeline = PDFIngestionPipeline(config=cfg, root_dir=RAG_STORE)
    manifest = pipeline.ingest(PDF_PATH)
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        root_dir: Union[str, Path] = "rag_store",
    ):
        self.config   = config or IngestionConfig()
        self.root_dir = Path(root_dir)

        self._extractor       = PyMuPDFLayoutExtractor(
            encoding_model_name=self.config.encoding_model_name,
            root_dir=self.root_dir,
        )
        self._normalizer      = TextNormalizer(join_broken_hyphens=self.config.join_broken_hyphens)
        self._type_detector   = PageTypeDetector(self.config)
        self._col_detector    = ColumnDetector(self.config)
        self._heading_detector = OCRAwareHeadingDetector(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline on a PDF and return the augmented manifest.

        Steps
        ─────
        1. Raw layout extraction (PyMuPDFLayoutExtractor)
        2. Text normalisation  → adds 'normalized_text' to spans + elements
        3. Page type detection → adds 'page_type', quality metrics to pages
        4. Column detection    → adds 'column_count' to pages
        5. Reading order       → adds 'reading_order_index' to elements
        6. OCR-aware headings  → adds 'heading_score_final', 'is_heading_candidate_final'
        7. Re-write augmented parquets
        8. Update manifest on disk
        """
        pdf_path = Path(pdf_path)
        print(f"[Ingestion] Extracting layout: {pdf_path.name}")
        manifest = self._extractor.extract(pdf_path, document_id=document_id, overwrite=overwrite)

        doc_dir = Path(manifest["paths"]["document_dir"])

        # ---- load tables ----
        pages    = pq.read_table(doc_dir / "layout_pages.parquet").to_pylist()
        elements = pq.read_table(doc_dir / "layout_elements.parquet").to_pylist()
        spans    = pq.read_table(doc_dir / "layout_spans.parquet").to_pylist()

        images_path = doc_dir / "layout_images.parquet"
        images = pq.read_table(images_path).to_pylist() if images_path.exists() else []

        print(f"[Ingestion] Augmenting {len(pages)} pages · {len(elements)} elements · {len(spans)} spans")

        # Index by page for O(1) lookups.
        spans_by_page:  Dict[int, List] = {}
        images_by_page: Dict[int, List] = {}
        for s in spans:
            spans_by_page.setdefault(int(s["page_number"]), []).append(s)
        for img in images:
            images_by_page.setdefault(int(img["page_number"]), []).append(img)

        # ---- step 2: text normalisation ----
        if self.config.normalize_text:
            for s in spans:
                s["normalized_text"] = self._normalizer.normalize(s.get("text") or "")
            for e in elements:
                e["normalized_text"] = self._normalizer.normalize(e.get("text") or "")
        else:
            for s in spans:
                s["normalized_text"] = s.get("text") or ""
            for e in elements:
                e["normalized_text"] = e.get("text") or ""

        # ---- step 3: page type detection ----
        page_type_map: Dict[int, PageType] = {}

        for page in pages:
            pnum = int(page["page_number"])
            ptype, metrics = self._type_detector.detect(
                spans_by_page.get(pnum, []),
                images_by_page.get(pnum, []),
                float(page["page_width"]),
                float(page["page_height"]),
            )
            page_type_map[pnum] = ptype

            page["page_type"]               = ptype.value
            page["font_variety_score"]      = metrics["font_variety_score"]
            page["font_size_cv"]            = metrics["size_cv"]
            page["image_area_ratio"]        = metrics["image_area_ratio"]
            page["has_ocr_font_indicator"]  = metrics["has_ocr_font"]
            page["distinct_font_count"]     = metrics["distinct_fonts"]
            # Quality proxy: for scanned pages, higher font variety = more structure recovered.
            if ptype == PageType.SCANNED_OCR:
                page["ocr_quality_score"] = round(min(1.0, metrics["font_variety_score"] * 10), 3)
            else:
                page["ocr_quality_score"] = 1.0

        # ---- step 4: column detection ----
        col_count_by_page: Dict[int, int] = {}

        for page in pages:
            pnum = int(page["page_number"])
            if self.config.detect_columns:
                block_els = [
                    e for e in elements
                    if int(e.get("page_number") or -1) == pnum and e.get("element_type") == "block"
                ]
                col_count = self._col_detector.detect_column_count(
                    block_els, float(page["page_width"])
                )
            else:
                col_count = 1
            col_count_by_page[pnum] = col_count
            page["column_count"] = col_count

        # ---- step 5: reading order ----
        reading_order: Dict[int, int] = {}
        for pnum, col_count in col_count_by_page.items():
            order = self._col_detector.build_reading_order(elements, col_count, pnum)
            reading_order.update(order)

        for e in elements:
            e["reading_order_index"] = reading_order.get(int(e["element_id"]), -1)

        # ---- step 6: OCR-aware heading detection ----
        if self.config.ocr_aware_headings:
            elements = self._heading_detector.annotate(elements, page_type_map)
        else:
            for e in elements:
                e["heading_score_final"]        = float(e.get("heading_score") or 0.0)
                e["is_heading_candidate_final"] = bool(e.get("is_heading_candidate") or False)

        # ---- step 7: write augmented parquets ----
        write_parquet(pages,    doc_dir / "layout_pages.parquet")
        write_parquet(elements, doc_dir / "layout_elements.parquet")
        write_parquet(spans,    doc_dir / "layout_spans.parquet")

        # ---- step 8: update manifest ----
        native_ct  = sum(1 for p in pages if p.get("page_type") == PageType.NATIVE.value)
        scanned_ct = sum(1 for p in pages if p.get("page_type") == PageType.SCANNED_OCR.value)
        two_col_ct = sum(1 for p in pages if int(p.get("column_count") or 1) > 1)

        manifest["ingestion_augmentation"] = {
            "total_pages":        len(pages),
            "native_pages":       native_ct,
            "scanned_ocr_pages":  scanned_ct,
            "two_column_pages":   two_col_ct,
            "text_normalized":    self.config.normalize_text,
            "ocr_aware_headings": self.config.ocr_aware_headings,
            "pipeline_version":   "pdf_ingestion_v1",
        }

        (doc_dir / "layout_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[Ingestion] Complete.")
        print(f"  Native pages:      {native_ct}")
        print(f"  Scanned/OCR pages: {scanned_ct}")
        print(f"  2-column pages:    {two_col_ct}")

        return manifest

from __future__ import annotations

import json
import re
import uuid
import hashlib
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def normalize_text_for_repetition(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bpage\s+\d+\s*(of\s*\d+)?\b", "page <n>", text)
    text = re.sub(r"\b\d+\s*/\s*\d+\b", "<n>/<n>", text)
    text = re.sub(r"\b\d{1,4}\b", "<n>", text)
    return text


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        table = pa.Table.from_pylist([])
    else:
        table = pa.Table.from_pylist(records)

    pq.write_table(table, path, compression="zstd")


def rect_to_tuple(rect_like) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if rect_like is None:
        return None, None, None, None
    try:
        return (
            float(rect_like[0]),
            float(rect_like[1]),
            float(rect_like[2]),
            float(rect_like[3]),
        )
    except Exception:
        return None, None, None, None


def bbox_features(
    bbox: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    page_width: float,
    page_height: float,
) -> Dict[str, Optional[float]]:
    x0, y0, x1, y1 = bbox

    if None in bbox or page_width <= 0 or page_height <= 0:
        return {
            "width": None,
            "height": None,
            "area": None,
            "x0_norm": None,
            "y0_norm": None,
            "x1_norm": None,
            "y1_norm": None,
            "center_x_norm": None,
            "center_y_norm": None,
        }

    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    area = width * height

    return {
        "width": width,
        "height": height,
        "area": area,
        "x0_norm": x0 / page_width,
        "y0_norm": y0 / page_height,
        "x1_norm": x1 / page_width,
        "y1_norm": y1 / page_height,
        "center_x_norm": ((x0 + x1) / 2.0) / page_width,
        "center_y_norm": ((y0 + y1) / 2.0) / page_height,
    }


def is_bold_font(font_name: Optional[str]) -> bool:
    if not font_name:
        return False
    f = font_name.lower()
    return any(x in f for x in ["bold", "black", "heavy", "semibold", "demibold"])


def is_italic_font(font_name: Optional[str]) -> bool:
    if not font_name:
        return False
    f = font_name.lower()
    return any(x in f for x in ["italic", "oblique"])


def text_shape_features(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    letters = re.findall(r"[A-Za-z]", stripped)
    upper_letters = re.findall(r"[A-Z]", stripped)

    return {
        "text_len": len(stripped),
        "word_count": len(stripped.split()),
        "has_digit": bool(re.search(r"\d", stripped)),
        "ends_with_colon": stripped.endswith(":"),
        "is_all_caps": bool(letters) and len(upper_letters) / max(1, len(letters)) > 0.85,
        "looks_numbered_heading": bool(re.match(r"^\s*(\d+(\.\d+)*|[A-Z])[\).\s-]+", stripped)),
    }


class PyMuPDFLayoutExtractor:
    """
    Layout extraction layer for RAG metadata extraction.

    Outputs:
      - tokens.i32
      - layout_pages.parquet
      - layout_elements.parquet
      - layout_spans.parquet
      - layout_words.parquet
      - layout_images.parquet
      - layout_drawings.parquet
      - layout_manifest.json

    Element hierarchy:
      page
        block
          line
            span

    Important:
      Token spans are based on the canonical text stream written by this extractor.
    """

    def __init__(
        self,
        encoding_model_name: str = "gpt-4o",
        root_dir: Union[str, Path] = "rag_store",
        ocr_engine=None,  # GoogleVisionOCR instance or None
    ):
        self.encoding_model_name = encoding_model_name
        self.encoding = tiktoken.encoding_for_model(encoding_model_name)
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_engine = ocr_engine  # injected by caller when use_ocr=True

    def extract(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        overwrite: bool = True,
        store_tokens: bool = True,
    ) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        file_hash = sha256_file(pdf_path)
        document_id = document_id or f"doc_{file_hash[:16]}"

        document_dir = self.root_dir / document_id
        if document_dir.exists() and not overwrite:
            raise FileExistsError(f"Document directory already exists: {document_dir}")

        document_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "document_dir": document_dir,
            "tokens": document_dir / "tokens.i32",
            "pages": document_dir / "layout_pages.parquet",
            "elements": document_dir / "layout_elements.parquet",
            "spans": document_dir / "layout_spans.parquet",
            "words": document_dir / "layout_words.parquet",
            "images": document_dir / "layout_images.parquet",
            "drawings": document_dir / "layout_drawings.parquet",
            "manifest": document_dir / "layout_manifest.json",
        }

        started_at = utc_now_iso()

        pages: List[Dict[str, Any]] = []
        elements: List[Dict[str, Any]] = []
        spans_out: List[Dict[str, Any]] = []
        words_out: List[Dict[str, Any]] = []
        images_out: List[Dict[str, Any]] = []
        drawings_out: List[Dict[str, Any]] = []

        global_token_index = 0
        element_id = 0
        span_id = 0
        word_id = 0
        image_id = 0
        drawing_id = 0

        doc = fitz.open(str(pdf_path))

        pdf_metadata = dict(doc.metadata or {})
        is_encrypted = bool(getattr(doc, "is_encrypted", False))
        needs_pass = bool(getattr(doc, "needs_pass", False))

        token_file = paths["tokens"].open("wb") if store_tokens else None

        try:
            for page_number, page in enumerate(doc):
                page_start = global_token_index

                page_rect = page.rect
                page_width = float(page_rect.width)
                page_height = float(page_rect.height)
                rotation = int(getattr(page, "rotation", 0) or 0)

                # -------------------------
                # Text layout extraction
                # -------------------------
                try:
                    text_dict = page.get_text("dict", sort=True)
                except TypeError:
                    text_dict = page.get_text("dict")

                page_text_chars = 0
                page_line_count = 0
                page_block_count = 0
                page_span_count = 0
                page_image_block_count = 0

                blocks = text_dict.get("blocks", [])

                for block_number, block in enumerate(blocks):
                    block_type_raw = block.get("type")
                    block_bbox = rect_to_tuple(block.get("bbox"))
                    block_bbox_f = bbox_features(block_bbox, page_width, page_height)

                    # type 0 = text block, type 1 = image block in PyMuPDF dict output.
                    if block_type_raw == 1:
                        x0, y0, x1, y1 = block_bbox
                        images_out.append({
                            "image_id": image_id,
                            "document_id": document_id,
                            "page_number": page_number,
                            "block_number": block_number,
                            "source": "get_text_dict_image_block",
                            "xref": block.get("xref"),
                            "ext": block.get("ext"),
                            "width_px": block.get("width"),
                            "height_px": block.get("height"),
                            "colorspace": block.get("colorspace"),
                            "xres": block.get("xres"),
                            "yres": block.get("yres"),
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            **block_bbox_f,
                            "metadata_json": safe_json_dumps({
                                "raw_keys": list(block.keys()),
                            }),
                        })
                        image_id += 1
                        page_image_block_count += 1
                        continue

                    if block_type_raw != 0:
                        continue

                    block_start: Optional[int] = None
                    block_end: Optional[int] = None
                    block_text_parts = []
                    block_line_ids = []

                    for line_number, line in enumerate(block.get("lines", [])):
                        line_bbox = rect_to_tuple(line.get("bbox"))
                        line_bbox_f = bbox_features(line_bbox, page_width, page_height)

                        line_start: Optional[int] = None
                        line_end: Optional[int] = None
                        line_text_parts = []
                        line_span_ids = []

                        for span_number, span in enumerate(line.get("spans", [])):
                            raw_text = span.get("text", "")

                            if raw_text == "":
                                continue

                            span_text = raw_text
                            token_ids = self.encoding.encode(span_text)

                            if token_ids:
                                start = global_token_index
                                end = global_token_index + len(token_ids)

                                if token_file is not None:
                                    np.asarray(token_ids, dtype=np.int32).tofile(token_file)

                                global_token_index = end
                            else:
                                start = global_token_index
                                end = global_token_index

                            if line_start is None:
                                line_start = start
                            line_end = end

                            if block_start is None:
                                block_start = start
                            block_end = end

                            span_bbox = rect_to_tuple(span.get("bbox"))
                            span_bbox_f = bbox_features(span_bbox, page_width, page_height)

                            font_name = span.get("font")
                            font_size = float(span.get("size") or 0.0)
                            flags = int(span.get("flags") or 0)

                            span_row = {
                                "span_id": span_id,
                                "document_id": document_id,
                                "page_number": page_number,
                                "block_number": block_number,
                                "line_number": line_number,
                                "span_number": span_number,
                                "start_index": start,
                                "end_index": end,
                                "token_length": end - start,
                                "text": span_text,
                                "text_preview": span_text[:300],
                                "font_name": font_name,
                                "font_size": font_size,
                                "flags": flags,
                                "color": span.get("color"),
                                "ascender": span.get("ascender"),
                                "descender": span.get("descender"),
                                "origin_x": float(span.get("origin", [None, None])[0]) if span.get("origin") else None,
                                "origin_y": float(span.get("origin", [None, None])[1]) if span.get("origin") else None,
                                "is_bold_fontname": is_bold_font(font_name),
                                "is_italic_fontname": is_italic_font(font_name),
                                "x0": span_bbox[0],
                                "y0": span_bbox[1],
                                "x1": span_bbox[2],
                                "y1": span_bbox[3],
                                **span_bbox_f,
                                "metadata_json": safe_json_dumps({
                                    "source": "pymupdf.get_text_dict",
                                    "exact_token_span": True,
                                    "raw_span_keys": list(span.keys()),
                                }),
                            }

                            spans_out.append(span_row)

                            line_span_ids.append(span_id)
                            span_id += 1
                            page_span_count += 1

                            line_text_parts.append(span_text)
                            block_text_parts.append(span_text)
                            page_text_chars += len(span_text)

                        if line_start is not None and line_end is not None:
                            # Add one newline after each line to canonical token stream.
                            newline_tokens = self.encoding.encode("\n")
                            if newline_tokens:
                                if token_file is not None:
                                    np.asarray(newline_tokens, dtype=np.int32).tofile(token_file)
                                global_token_index += len(newline_tokens)
                                line_end = global_token_index
                                block_end = global_token_index

                            line_text = "".join(line_text_parts).rstrip() + "\n"

                            line_element_id = element_id
                            block_line_ids.append(line_element_id)

                            shape = text_shape_features(line_text)

                            x0, y0, x1, y1 = line_bbox

                            elements.append({
                                "element_id": line_element_id,
                                "document_id": document_id,
                                "element_type": "line",
                                "page_number": page_number,
                                "block_number": block_number,
                                "line_number": line_number,
                                "start_index": line_start,
                                "end_index": line_end,
                                "token_length": line_end - line_start,
                                "text": line_text,
                                "text_preview": line_text[:300],
                                "span_ids_json": safe_json_dumps(line_span_ids),
                                "child_element_ids_json": safe_json_dumps([]),
                                "parent_element_id": None,  # filled later if needed
                                "x0": x0,
                                "y0": y0,
                                "x1": x1,
                                "y1": y1,
                                **line_bbox_f,
                                **shape,
                                "avg_font_size": self._avg_font_size_for_spans(spans_out, line_span_ids),
                                "max_font_size": self._max_font_size_for_spans(spans_out, line_span_ids),
                                "dominant_font_name": self._dominant_font_for_spans(spans_out, line_span_ids),
                                "contains_bold": self._any_bold_for_spans(spans_out, line_span_ids),
                                "contains_italic": self._any_italic_for_spans(spans_out, line_span_ids),
                                "is_top_zone": bool(line_bbox_f["y0_norm"] is not None and line_bbox_f["y0_norm"] <= 0.12),
                                "is_bottom_zone": bool(line_bbox_f["y1_norm"] is not None and line_bbox_f["y1_norm"] >= 0.88),
                                "metadata_json": safe_json_dumps({
                                    "source": "pymupdf.get_text_dict",
                                    "exact_token_span": True,
                                }),
                            })
                            element_id += 1
                            page_line_count += 1

                    if block_start is not None and block_end is not None:
                        block_text = "".join(block_text_parts).strip()
                        x0, y0, x1, y1 = block_bbox
                        shape = text_shape_features(block_text)

                        elements.append({
                            "element_id": element_id,
                            "document_id": document_id,
                            "element_type": "block",
                            "page_number": page_number,
                            "block_number": block_number,
                            "line_number": None,
                            "start_index": block_start,
                            "end_index": block_end,
                            "token_length": block_end - block_start,
                            "text": block_text,
                            "text_preview": block_text[:500],
                            "span_ids_json": safe_json_dumps([]),
                            "child_element_ids_json": safe_json_dumps(block_line_ids),
                            "parent_element_id": None,
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            **block_bbox_f,
                            **shape,
                            "avg_font_size": None,
                            "max_font_size": None,
                            "dominant_font_name": None,
                            "contains_bold": None,
                            "contains_italic": None,
                            "is_top_zone": bool(block_bbox_f["y0_norm"] is not None and block_bbox_f["y0_norm"] <= 0.12),
                            "is_bottom_zone": bool(block_bbox_f["y1_norm"] is not None and block_bbox_f["y1_norm"] >= 0.88),
                            "metadata_json": safe_json_dumps({
                                "source": "pymupdf.get_text_dict",
                                "exact_token_span": True,
                                "derived_from": "line/span elements",
                            }),
                        })
                        element_id += 1
                        page_block_count += 1

                # Add page separator.
                if global_token_index > page_start:
                    sep_tokens = self.encoding.encode("\n")
                    if sep_tokens:
                        if token_file is not None:
                            np.asarray(sep_tokens, dtype=np.int32).tofile(token_file)
                        global_token_index += len(sep_tokens)

                # ── Google Vision OCR injection ────────────────────────────────
                # If this page has no native text and an OCR engine is provided,
                # render the page, call Vision API, and inject the resulting text
                # as block elements in the same token stream.
                _ocr_injected = False
                if self.ocr_engine is not None and global_token_index == page_start:
                    try:
                        import fitz as _fitz_local
                        _ocr_mat = _fitz_local.Matrix(200 / 72, 200 / 72)
                        _ocr_pix = page.get_pixmap(matrix=_ocr_mat, colorspace=_fitz_local.csRGB)
                        _ocr_img = _ocr_pix.tobytes("png")
                        _ocr_elems = self.ocr_engine.ocr_page_image(
                            _ocr_img, page_number, page_width, page_height, 200 / 72
                        )
                        for _oe in _ocr_elems:
                            _oe_text = (_oe.get("text") or "").strip()
                            if not _oe_text:
                                continue
                            _oe_tokens = self.encoding.encode(_oe_text + "\n")
                            if not _oe_tokens:
                                continue
                            _oe_start = global_token_index
                            _oe_end = global_token_index + len(_oe_tokens)
                            if token_file is not None:
                                np.asarray(_oe_tokens, dtype=np.int32).tofile(token_file)
                            global_token_index = _oe_end
                            page_text_chars += len(_oe_text)
                            page_block_count += 1
                            _oe_bbox = (
                                float(_oe.get("x0", 0)), float(_oe.get("y0", 0)),
                                float(_oe.get("x1", page_width)), float(_oe.get("y1", page_height)),
                            )
                            elements.append({
                                "element_id": element_id,
                                "document_id": document_id,
                                "element_type": "block",
                                "page_number": page_number,
                                "block_number": _oe.get("block_number", page_block_count - 1),
                                "line_number": None,
                                "start_index": _oe_start,
                                "end_index": _oe_end,
                                "token_length": _oe_end - _oe_start,
                                "text_chars": len(_oe_text),
                                "text_preview": _oe_text[:300],
                                "x0": _oe_bbox[0], "y0": _oe_bbox[1],
                                "x1": _oe_bbox[2], "y1": _oe_bbox[3],
                                **bbox_features(_oe_bbox, page_width, page_height),
                                **text_shape_features(_oe_text),
                                "avg_font_size": None,
                                "max_font_size": None,
                                "dominant_font_name": "google_vision",
                                "contains_bold": None,
                                "contains_italic": None,
                                "is_top_zone": _oe_bbox[1] / max(page_height, 1) < 0.15,
                                "is_bottom_zone": _oe_bbox[3] / max(page_height, 1) > 0.85,
                                "span_ids_json": safe_json_dumps([]),
                                "child_element_ids_json": safe_json_dumps([]),
                                "parent_element_id": None,
                                "reading_order_index": page_block_count - 1,
                                "heading_score_final": 0.0,
                                "is_heading_candidate_final": False,
                                "metadata_json": safe_json_dumps({
                                    "source": "google_vision",
                                    "confidence": _oe.get("confidence"),
                                }),
                            })
                            element_id += 1
                            # Also inject word-level rows for word table
                            for _wdict in (_oe.get("word_dicts") or []):
                                _wtext = (_wdict.get("text") or "").strip()
                                if _wtext:
                                    words_out.append({
                                        "word_id": word_id,
                                        "document_id": document_id,
                                        "page_number": page_number,
                                        "block_number": _oe.get("block_number", 0),
                                        "line_number": _wdict.get("line_number", 0),
                                        "word_number": 0,
                                        "text": _wtext,
                                        "x0": float(_wdict.get("x0", 0)),
                                        "y0": float(_wdict.get("y0", 0)),
                                        "x1": float(_wdict.get("x1", page_width)),
                                        "y1": float(_wdict.get("y1", page_height)),
                                        "metadata_json": safe_json_dumps({"source": "google_vision"}),
                                    })
                                    word_id += 1
                        _ocr_injected = bool(_ocr_elems)
                    except Exception as _ocr_exc:
                        print(f"[vision_ocr] page {page_number} OCR failed: {_ocr_exc}")

                page_end = global_token_index

                # -------------------------
                # Word extraction
                # -------------------------
                try:
                    words = page.get_text("words", sort=True)
                except TypeError:
                    words = page.get_text("words")

                for w in words:
                    # PyMuPDF words format commonly:
                    # x0, y0, x1, y1, word, block_no, line_no, word_no
                    if len(w) < 5:
                        continue

                    x0, y0, x1, y1 = map(float, w[:4])
                    word_text = str(w[4])
                    block_no = int(w[5]) if len(w) > 5 else None
                    line_no = int(w[6]) if len(w) > 6 else None
                    word_no = int(w[7]) if len(w) > 7 else None

                    bbox = (x0, y0, x1, y1)

                    words_out.append({
                        "word_id": word_id,
                        "document_id": document_id,
                        "page_number": page_number,
                        "block_number": block_no,
                        "line_number": line_no,
                        "word_number": word_no,
                        "text": word_text,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        **bbox_features(bbox, page_width, page_height),
                    })
                    word_id += 1

                # -------------------------
                # Images from get_image_info if available
                # -------------------------
                if hasattr(page, "get_image_info"):
                    try:
                        image_infos = page.get_image_info(xrefs=True)
                    except TypeError:
                        image_infos = page.get_image_info()

                    for info in image_infos or []:
                        bbox = rect_to_tuple(info.get("bbox"))
                        x0, y0, x1, y1 = bbox
                        images_out.append({
                            "image_id": image_id,
                            "document_id": document_id,
                            "page_number": page_number,
                            "block_number": None,
                            "source": "page.get_image_info",
                            "xref": info.get("xref"),
                            "ext": info.get("ext"),
                            "width_px": info.get("width"),
                            "height_px": info.get("height"),
                            "colorspace": info.get("colorspace"),
                            "xres": info.get("xres"),
                            "yres": info.get("yres"),
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            **bbox_features(bbox, page_width, page_height),
                            "metadata_json": safe_json_dumps(info),
                        })
                        image_id += 1

                # -------------------------
                # Vector drawings
                # -------------------------
                try:
                    drawing_paths = page.get_drawings()
                except Exception:
                    drawing_paths = []

                for d in drawing_paths or []:
                    bbox = rect_to_tuple(d.get("rect"))
                    x0, y0, x1, y1 = bbox
                    drawings_out.append({
                        "drawing_id": drawing_id,
                        "document_id": document_id,
                        "page_number": page_number,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        **bbox_features(bbox, page_width, page_height),
                        "items_count": len(d.get("items") or []),
                        "type": d.get("type"),
                        "stroke_opacity": d.get("stroke_opacity"),
                        "fill_opacity": d.get("fill_opacity"),
                        "width_line": d.get("width"),
                        "color": str(d.get("color")),
                        "fill": str(d.get("fill")),
                        "metadata_json": safe_json_dumps({
                            k: v for k, v in d.items()
                            if k not in {"items"}
                        }),
                    })
                    drawing_id += 1

                # -------------------------
                # Page row
                # -------------------------
                has_text = page_end > page_start
                page_area = page_width * page_height
                text_density = page_text_chars / max(1.0, page_area)

                page_row = {
                    "document_id": document_id,
                    "page_number": page_number,
                    "start_index": page_start,
                    "end_index": page_end,
                    "token_length": page_end - page_start,
                    "page_width": page_width,
                    "page_height": page_height,
                    "rotation": rotation,
                    "has_text": bool(has_text),
                    "text_chars": page_text_chars,
                    "text_density": text_density,
                    "line_count": page_line_count,
                    "block_count": page_block_count,
                    "span_count": page_span_count,
                    "word_count": len(words),
                    "image_block_count": page_image_block_count,
                    "drawing_count": len(drawing_paths or []),
                    "is_probably_scanned": bool((not (page_end > page_start and not _ocr_injected)) and (page_image_block_count > 0 or len(image_infos or []) > 0 if 'image_infos' in locals() else False)),
                    "metadata_json": safe_json_dumps({
                        "source": "google_vision" if _ocr_injected else "pymupdf",
                        "ocr_injected": _ocr_injected,
                    }),
                }

                pages.append(page_row)

                # Page element.
                elements.append({
                    "element_id": element_id,
                    "document_id": document_id,
                    "element_type": "page",
                    "page_number": page_number,
                    "block_number": None,
                    "line_number": None,
                    "start_index": page_start,
                    "end_index": page_end,
                    "token_length": page_end - page_start,
                    "text": "",
                    "text_preview": "",
                    "span_ids_json": safe_json_dumps([]),
                    "child_element_ids_json": safe_json_dumps([]),
                    "parent_element_id": None,
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": page_width,
                    "y1": page_height,
                    **bbox_features((0.0, 0.0, page_width, page_height), page_width, page_height),
                    "text_len": page_text_chars,
                    "word_count": len(words),
                    "has_digit": None,
                    "ends_with_colon": None,
                    "is_all_caps": None,
                    "looks_numbered_heading": None,
                    "avg_font_size": None,
                    "max_font_size": None,
                    "dominant_font_name": None,
                    "contains_bold": None,
                    "contains_italic": None,
                    "is_top_zone": None,
                    "is_bottom_zone": None,
                    "metadata_json": safe_json_dumps({
                        "source": "pymupdf",
                        "exact_token_span": True,
                    }),
                })
                element_id += 1

                if page_number % 50 == 0:
                    print(f"Extracted page {page_number} | tokens={global_token_index} | elements={len(elements)}")

        finally:
            if token_file is not None:
                token_file.close()
            doc.close()

        # Add derived metadata flags after seeing whole document.
        elements = self._annotate_repeated_headers_footers(elements, total_pages=len(pages))
        elements = self._annotate_heading_candidates(elements)

        finished_at = utc_now_iso()

        manifest = {
            "document_id": document_id,
            "source_pdf_path": str(pdf_path),
            "source_pdf_name": pdf_path.name,
            "source_sha256": file_hash,
            "encoding_model_name": self.encoding_model_name,
            "total_pages": len(pages),
            "total_tokens": global_token_index,
            "total_elements": len(elements),
            "total_spans": len(spans_out),
            "total_words": len(words_out),
            "total_images": len(images_out),
            "total_drawings": len(drawings_out),
            "pdf_metadata": pdf_metadata,
            "is_encrypted": is_encrypted,
            "needs_pass": needs_pass,
            "created_at": started_at,
            "finished_at": finished_at,
            "layout_extractor_version": "pymupdf_layout_v1",
            "paths": {k: str(v) for k, v in paths.items()},
        }

        write_parquet(pages, paths["pages"])
        write_parquet(elements, paths["elements"])
        write_parquet(spans_out, paths["spans"])
        write_parquet(words_out, paths["words"])
        write_parquet(images_out, paths["images"])
        write_parquet(drawings_out, paths["drawings"])

        paths["manifest"].write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write PdfTokenStore-compatible aliases so ChunkManager can load
        # directly from this extractor's output without re-ingesting via
        # PdfTokenStore.ingest_pdf() (which would lose OCR data).
        import shutil as _shutil
        _store_pages = document_dir / "pages.parquet"
        _store_elements = document_dir / "elements.parquet"
        _shutil.copy2(str(paths["pages"]), str(_store_pages))
        _shutil.copy2(str(paths["elements"]), str(_store_elements))
        _chunks_dir = document_dir / "chunks"
        _chunks_dir.mkdir(parents=True, exist_ok=True)
        (document_dir / "manifest.json").write_text(
            json.dumps({
                "document_id": document_id,
                "source_pdf_path": str(pdf_path),
                "source_pdf_name": pdf_path.name,
                "source_sha256": file_hash,
                "encoding_model_name": self.encoding_model_name,
                "total_tokens": global_token_index,
                "total_pages": len(pages),
                "total_elements": len(elements),
                "tokens_path": str(paths["tokens"]),
                "pages_path": str(_store_pages),
                "elements_path": str(_store_elements),
                "chunks_dir": str(_chunks_dir),
                "created_at": started_at,
                "finished_at": finished_at,
                "storage_version": "v1_memmap_tokens_parquet_metadata",
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Layout extraction complete.")
        print(f"Document ID: {document_id}")
        print(f"Pages: {len(pages)}")
        print(f"Elements: {len(elements)}")
        print(f"Spans: {len(spans_out)}")
        print(f"Words: {len(words_out)}")
        print(f"Images: {len(images_out)}")
        print(f"Drawings: {len(drawings_out)}")
        print(f"Output dir: {document_dir}")

        return manifest

    # -------------------------
    # Span aggregation helpers
    # -------------------------

    def _get_spans_by_ids(self, spans: List[Dict[str, Any]], ids: List[int]) -> List[Dict[str, Any]]:
        wanted = set(ids)
        return [s for s in spans if int(s["span_id"]) in wanted]

    def _avg_font_size_for_spans(self, spans: List[Dict[str, Any]], ids: List[int]) -> Optional[float]:
        ss = self._get_spans_by_ids(spans, ids)
        vals = [float(s["font_size"]) for s in ss if s.get("font_size") is not None]
        return float(np.mean(vals)) if vals else None

    def _max_font_size_for_spans(self, spans: List[Dict[str, Any]], ids: List[int]) -> Optional[float]:
        ss = self._get_spans_by_ids(spans, ids)
        vals = [float(s["font_size"]) for s in ss if s.get("font_size") is not None]
        return float(np.max(vals)) if vals else None

    def _dominant_font_for_spans(self, spans: List[Dict[str, Any]], ids: List[int]) -> Optional[str]:
        ss = self._get_spans_by_ids(spans, ids)
        fonts = [s.get("font_name") for s in ss if s.get("font_name")]
        if not fonts:
            return None
        return max(set(fonts), key=fonts.count)

    def _any_bold_for_spans(self, spans: List[Dict[str, Any]], ids: List[int]) -> bool:
        ss = self._get_spans_by_ids(spans, ids)
        return any(bool(s.get("is_bold_fontname")) for s in ss)

    def _any_italic_for_spans(self, spans: List[Dict[str, Any]], ids: List[int]) -> bool:
        ss = self._get_spans_by_ids(spans, ids)
        return any(bool(s.get("is_italic_fontname")) for s in ss)

    # -------------------------
    # Derived annotations
    # -------------------------

    def _annotate_repeated_headers_footers(
        self,
        elements: List[Dict[str, Any]],
        total_pages: int,
        min_repetition_ratio: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Marks repeated top/bottom lines as header/footer candidates.
        """
        if total_pages <= 2:
            for e in elements:
                e["is_repeated_header_candidate"] = False
                e["is_repeated_footer_candidate"] = False
            return elements

        top_counts: Dict[str, int] = {}
        bottom_counts: Dict[str, int] = {}

        for e in elements:
            if e.get("element_type") != "line":
                continue

            text = e.get("text") or ""
            norm = normalize_text_for_repetition(text)

            if len(norm) < 3:
                continue

            if e.get("is_top_zone"):
                top_counts[norm] = top_counts.get(norm, 0) + 1

            if e.get("is_bottom_zone"):
                bottom_counts[norm] = bottom_counts.get(norm, 0) + 1

        min_count = max(2, int(np.ceil(total_pages * min_repetition_ratio)))

        repeated_top = {k for k, v in top_counts.items() if v >= min_count}
        repeated_bottom = {k for k, v in bottom_counts.items() if v >= min_count}

        for e in elements:
            if e.get("element_type") != "line":
                e["is_repeated_header_candidate"] = False
                e["is_repeated_footer_candidate"] = False
                continue

            norm = normalize_text_for_repetition(e.get("text") or "")

            e["is_repeated_header_candidate"] = bool(e.get("is_top_zone") and norm in repeated_top)
            e["is_repeated_footer_candidate"] = bool(e.get("is_bottom_zone") and norm in repeated_bottom)

        return elements

    def _annotate_heading_candidates(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Marks line-level heading candidates using font size, boldness, shape, and length.
        This is intentionally heuristic. Later your MetadataExtractor can improve it.
        """
        line_sizes = [
            float(e["max_font_size"])
            for e in elements
            if e.get("element_type") == "line" and e.get("max_font_size") is not None
        ]

        if not line_sizes:
            for e in elements:
                e["is_heading_candidate"] = False
                e["heading_score"] = 0.0
            return elements

        median_size = float(np.median(line_sizes))
        p80_size = float(np.percentile(line_sizes, 80))
        p90_size = float(np.percentile(line_sizes, 90))

        for e in elements:
            if e.get("element_type") != "line":
                e["is_heading_candidate"] = False
                e["heading_score"] = 0.0
                continue

            text = (e.get("text") or "").strip()
            word_count = int(e.get("word_count") or 0)
            max_size = float(e.get("max_font_size") or 0.0)
            contains_bold = bool(e.get("contains_bold"))
            is_all_caps = bool(e.get("is_all_caps"))
            ends_with_colon = bool(e.get("ends_with_colon"))
            looks_numbered = bool(e.get("looks_numbered_heading"))
            is_header = bool(e.get("is_repeated_header_candidate"))
            is_footer = bool(e.get("is_repeated_footer_candidate"))

            score = 0.0

            if not text or is_header or is_footer:
                e["is_heading_candidate"] = False
                e["heading_score"] = 0.0
                continue

            if max_size >= p90_size:
                score += 0.35
            elif max_size >= p80_size:
                score += 0.25
            elif max_size > median_size:
                score += 0.15

            if contains_bold:
                score += 0.20

            if is_all_caps and 1 <= word_count <= 12:
                score += 0.15

            if ends_with_colon and 1 <= word_count <= 14:
                score += 0.15

            if looks_numbered:
                score += 0.10

            if 1 <= word_count <= 12:
                score += 0.10

            if word_count > 20:
                score -= 0.20

            if len(text) > 160:
                score -= 0.20

            score = max(0.0, min(1.0, score))

            e["heading_score"] = float(score)
            e["is_heading_candidate"] = bool(score >= 0.45)

        return elements


def layout_to_markdown(
    elements: List[Dict[str, Any]],
    remove_headers_footers: bool = True,
) -> str:
    lines = []

    line_elements = [
        e for e in elements
        if e.get("element_type") == "line"
    ]

    line_elements.sort(key=lambda e: (
        int(e["page_number"]),
        float(e["y0"] or 0),
        float(e["x0"] or 0),
    ))

    current_page = None

    for e in line_elements:
        page = int(e["page_number"])

        if page != current_page:
            current_page = page
            lines.append(f"\n<!-- page_number: {page} -->\n")

        if remove_headers_footers and (
            e.get("is_repeated_header_candidate") or e.get("is_repeated_footer_candidate")
        ):
            continue

        text = (e.get("text") or "").strip()
        if not text:
            continue

        if e.get("is_heading_candidate"):
            # Use heading score to choose heading depth later if you want.
            lines.append(f"\n## {text}\n")
        else:
            lines.append(text)

    return "\n".join(lines).strip()


def attach_simple_heading_paths(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current_heading_path = []

    ordered = sorted(
        elements,
        key=lambda e: (
            int(e.get("page_number") or -1),
            float(e.get("y0") or 0),
            float(e.get("x0") or 0),
            int(e.get("element_id") or 0),
        )
    )

    output = []

    for e in ordered:
        e = dict(e)

        if e.get("element_type") == "line":
            text = (e.get("text") or "").strip()

            if e.get("is_heading_candidate") and text:
                current_heading_path = [text]
                e["section_name"] = text
                e["heading_path_json"] = safe_json_dumps(current_heading_path)
            else:
                e["section_name"] = current_heading_path[-1] if current_heading_path else None
                e["heading_path_json"] = safe_json_dumps(current_heading_path)

        else:
            e["section_name"] = None
            e["heading_path_json"] = safe_json_dumps(current_heading_path)

        output.append(e)

    return output

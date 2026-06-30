"""
paddleocr_consumer.py — PaddleOCR consumer for 6kpro (GPU).

Generic: accepts base64 images or local file paths.
Reusable across projects that need GPU-accelerated OCR.

Topic   : nitrag.ocr.request
Payload : {
    image_b64:  str | null,   # base64-encoded PNG/JPEG
    image_path: str | null,   # absolute path on the 6kpro filesystem
    lang:       str,          # "en" (default) | "ch" | "fr" | etc.
    return_word_box: bool     # include per-word bounding boxes (default false)
  }
Result  : {
    lines: [{text, confidence, bbox: [x0,y0,x1,y1]}, ...],
    full_text: str,
    word_count: int
  }

Env vars
--------
NSQ_NSQD_TCP       comma-separated nsqd TCP addresses (default: 10.9.0.36:4150)
NSQ_LOOKUPD_HTTP   comma-separated nsqlookupd HTTP addresses
PADDLE_LANG        default OCR language (default: en)
PADDLE_USE_GPU     1 or 0 (default: 1)
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from .base import BaseConsumer

log = logging.getLogger(__name__)

TOPIC = "nitrag.ocr.request"
DEFAULT_LANG = os.environ.get("PADDLE_LANG", "en")
USE_GPU = os.environ.get("PADDLE_USE_GPU", "1") == "1"


class PaddleOCRConsumer(BaseConsumer):
    def __init__(self, **kwargs) -> None:
        super().__init__(topic=TOPIC, channel="nitrag", max_in_flight=1, **kwargs)
        self._ocr: Optional[Any] = None

    def _get_ocr(self, lang: str):
        # Re-use the same instance if same lang; OCR init is expensive (~5s on GPU).
        key = lang
        if self._ocr is None or self._ocr_lang != key:
            log.info("Initialising PaddleOCR lang=%s use_gpu=%s", lang, USE_GPU)
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=lang)
            self._ocr_lang = key
            log.info("PaddleOCR ready")
        return self._ocr

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        lang: str = payload.get("lang") or DEFAULT_LANG
        image_b64: Optional[str] = payload.get("image_b64")
        image_path: Optional[str] = payload.get("image_path")
        return_word_box: bool = bool(payload.get("return_word_box", False))

        if not image_b64 and not image_path:
            raise ValueError("payload must contain 'image_b64' or 'image_path'")

        ocr = self._get_ocr(lang)
        tmp_path = None

        try:
            if image_b64:
                img_bytes = base64.b64decode(image_b64)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(img_bytes)
                    tmp_path = f.name
                src = tmp_path
            else:
                src = image_path

            results = ocr.predict(src, return_word_box=return_word_box)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        lines = _parse_results(results)
        full_text = "\n".join(ln["text"] for ln in lines)

        return {
            "lines": lines,
            "full_text": full_text,
            "word_count": len(full_text.split()),
        }


def _parse_results(results) -> List[Dict[str, Any]]:
    """Normalise PaddleOCR 3.x predict() output → list of line dicts."""
    lines: List[Dict[str, Any]] = []
    if not results:
        return lines

    # results is a list of per-image result objects
    for item in results:
        if item is None:
            continue
        # PaddleOCR 3.x: item.rec_texts / item.rec_scores / item.dt_boxes
        texts = getattr(item, "rec_texts", None)
        scores = getattr(item, "rec_scores", None)
        boxes = getattr(item, "dt_boxes", None)

        if texts is None:
            # Fallback: older dict-based output
            if isinstance(item, list):
                for det in item:
                    if det is None:
                        continue
                    box, (text, conf) = det[0], det[1]
                    lines.append(_bbox_line(box, text, conf))
            continue

        for i, text in enumerate(texts):
            conf = float(scores[i]) if scores and i < len(scores) else 1.0
            box = boxes[i].tolist() if boxes is not None and i < len(boxes) else None
            lines.append(_bbox_line(box, text, conf))

    return lines


def _bbox_line(box, text: str, conf: float) -> Dict[str, Any]:
    bbox = None
    if box is not None:
        pts = box if isinstance(box[0], (list, tuple)) else [[p, p] for p in box]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    return {"text": text, "confidence": round(conf, 4), "bbox": bbox}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("paddleocr_consumer starting — topic=%s", TOPIC)
    PaddleOCRConsumer().run()


if __name__ == "__main__":
    main()

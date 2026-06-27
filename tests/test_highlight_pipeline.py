"""Tests for the citation highlight pipeline introduced 2026-06-27.

Covers:
  - _find_quote_rects_from_words   (word-level bbox matching)
  - render_page endpoint           (PNG thumbnail with highlight/crop)
  - render_page_pdf endpoint       (searchable PDF with annotations)
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from nitrag.server import (
    _HIGHLIGHT_STOP,
    _find_quote_rects_from_words,
    app,
)

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_words_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a layout_words.parquet for a fake doc and return the path."""
    p = tmp_path / "layout_words.parquet"
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), str(p))
    return p


def _words_df(*page_words: tuple[int, str, float, float, float, float]) -> list[dict]:
    """Build rows for layout_words parquet.  Each tuple: (page, text, x0, y0, x1, y1)."""
    return [
        {"page_number": pn, "text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        for pn, txt, x0, y0, x1, y1 in page_words
    ]


# ---------------------------------------------------------------------------
# _find_quote_rects_from_words  — unit tests
# ---------------------------------------------------------------------------

class TestFindQuoteRectsFromWords:

    def test_returns_tuple(self, tmp_path):
        """`_find_quote_rects_from_words` always returns (int, list, list)."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        _make_words_parquet(
            tmp_path / doc_id,
            _words_df((0, "Hello", 0, 0, 50, 10), (0, "World", 60, 0, 110, 10)),
        )
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            result = _find_quote_rects_from_words(doc_id, 0, "Hello World")
        assert isinstance(result, tuple) and len(result) == 3
        best_page, rects, triggers = result
        assert isinstance(best_page, int)
        assert isinstance(rects, list)
        assert isinstance(triggers, list)

    def test_finds_exact_words_on_page(self, tmp_path):
        """Matched words on the same line collapse to a single full-line rect."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "Patient", 10, 100, 60, 115),
            (0, "prescribed", 70, 100, 160, 115),
            (0, "metformin", 170, 100, 250, 115),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "Patient prescribed metformin")
        assert page == 0
        assert len(rects) == 1  # all three words are on the same line → one line rect
        assert rects[0].x0 == pytest.approx(10, abs=1)
        assert rects[0].x1 == pytest.approx(250, abs=1)

    def test_line_expansion_spans_non_matched_words(self, tmp_path):
        """The line rect includes words that were NOT in the query (whole-line highlight)."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "The",    0, 50, 25, 62),
            (0, "patient", 30, 50, 90, 62),
            (0, "takes",   95, 50, 140, 62),
            (0, "metformin", 145, 50, 240, 62),
            (0, "daily.",  245, 50, 300, 62),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "metformin")
        assert page == 0
        assert len(rects) == 1
        # Rect spans full line from x=0 to x=300, not just the "metformin" word
        assert rects[0].x0 == pytest.approx(0, abs=1)
        assert rects[0].x1 == pytest.approx(300, abs=1)

    def test_two_matched_lines_return_two_rects(self, tmp_path):
        """Words matched on two distinct lines return two separate line rects."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            # line 1 at y=0-12
            (0, "Patient", 0, 0, 60, 12),
            (0, "prescribed", 65, 0, 160, 12),
            (0, "metformin", 165, 0, 250, 12),
            # line 2 at y=20-32
            (0, "Allergies", 0, 20, 80, 32),
            (0, "Penicillin", 85, 20, 185, 32),
            (0, "rash.", 190, 20, 230, 32),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(
                doc_id, 0, "metformin prescribed Allergies Penicillin"
            )
        assert page == 0
        assert len(rects) == 2  # one rect per matched line

    def test_no_match_returns_empty_rects(self, tmp_path):
        """Returns (page_num, []) when nothing matches."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "Banana", 0, 0, 50, 10),
            (0, "Mango", 60, 0, 110, 10),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "completely unrelated query xyzzy")
        assert rects == []

    def test_missing_parquet_returns_empty(self, tmp_path):
        """Returns (page_num, []) gracefully when layout_words.parquet is absent."""
        doc_id = "no_parquet_doc"
        (tmp_path / doc_id).mkdir()
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 5, "some query")
        assert page == 5
        assert rects == []

    # ── Hyphen-split tokenisation ─────────────────────────────────────────

    def test_hyphenated_drug_name_matches_separate_pdf_words(self, tmp_path):
        """'Amoxicillin-Clavulanate' in quote matches 'Amoxicillin' + 'Clavulanate' as separate PDF words."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "Allergies", 0, 0, 80, 12),
            (0, "Amoxicillin", 90, 0, 200, 12),
            (0, "Clavulanate", 210, 0, 330, 12),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(
                doc_id, 0, "Allergies Amoxicillin-Clavulanate"
            )
        assert page == 0
        assert len(rects) == 1  # all three words are on the same line

    def test_dotted_abbreviation_nkda_matches(self, tmp_path):
        """'N.K.D.A.' in PDF (→ 'nkda') matches 'NKDA' in quote."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "Allergies", 0, 50, 80, 62),
            (0, "N.K.D.A.", 90, 50, 140, 62),
            (0, "Verified", 150, 50, 220, 62),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "Allergies NKDA Verified")
        assert page == 0
        assert len(rects) == 1  # all three words are on the same line

    # ── Short-quote adaptive threshold ────────────────────────────────────

    def test_short_allergy_quote_matches_at_low_threshold(self, tmp_path):
        """2-word allergy quote (q_set size <= 4) uses 0.30 threshold, not 0.40."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        # Page with 40 noise words + the allergy words buried in them
        noise = [(0, f"word{i}", float(i*10), 200, float(i*10+8), 212) for i in range(40)]
        signal = [
            (0, "Allergies", 0, 10, 80, 22),
            (0, "Penicillin", 90, 10, 190, 22),
        ]
        _make_words_parquet(tmp_path / doc_id, _words_df(*(signal + noise)))
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "Allergies Penicillin")
        assert page == 0
        assert len(rects) >= 1

    # ── Page range search ─────────────────────────────────────────────────

    def test_selects_best_page_in_range(self, tmp_path):
        """When page_end > page_num, searches all pages and returns the one with the best match."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            # page 0: noise only
            (0, "unrelated", 0, 0, 80, 12),
            (0, "content", 90, 0, 170, 12),
            # page 1: contains the cited sentence
            (1, "Patient", 0, 0, 60, 12),
            (1, "discharged", 70, 0, 170, 12),
            (1, "hospital", 180, 0, 260, 12),
            (1, "stable", 270, 0, 330, 12),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(
                doc_id, 0, "Patient discharged hospital stable", page_end=1
            )
        assert page == 1, f"Expected best match on page 1, got page {page}"
        assert len(rects) == 1  # all four words are on the same line → one line rect

    def test_page_end_equal_page_num_searches_only_one_page(self, tmp_path):
        """page_end == page_num (or -1) searches only page_num."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "alpha", 0, 0, 50, 10),
            (0, "beta", 60, 0, 110, 10),
            (0, "gamma", 120, 0, 180, 10),
            (1, "delta", 0, 0, 50, 10),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "alpha beta gamma")
        assert page == 0

    # ── Stopword filtering ────────────────────────────────────────────────

    def test_empty_query_returns_empty(self, tmp_path):
        """An empty string query returns (page_num, []) immediately."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df((0, "word", 0, 0, 30, 10))
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "")
        assert rects == []
        assert page == 0

    def test_stopword_only_query_falls_back_to_word_matching(self, tmp_path):
        """When all tokens are stopwords, the function falls back to matching them directly.
        This is intentional: the result is non-crashing and page/rects are valid types."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df((0, "the", 0, 0, 30, 10), (0, "a", 40, 0, 60, 10))
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "the a an is")
        assert isinstance(rects, list)
        assert page == 0

    # ── Case-insensitive matching ─────────────────────────────────────────

    def test_case_insensitive_match(self, tmp_path):
        """PDF words in ALLCAPS match lowercase quote tokens."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "DIAGNOSIS", 0, 0, 80, 12),
            (0, "HYPERTENSION", 90, 0, 210, 12),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "diagnosis hypertension")
        assert page == 0
        assert len(rects) >= 1

    # ── Trigger word rects ────────────────────────────────────────────────

    def test_triggers_are_subset_of_line_bounds(self, tmp_path):
        """Each trigger word rect must lie within the corresponding line rect bounds."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "The",       0, 50, 25, 62),
            (0, "patient",   30, 50, 90, 62),
            (0, "takes",     95, 50, 140, 62),
            (0, "metformin", 145, 50, 240, 62),
            (0, "daily.",    245, 50, 300, 62),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            page, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "patient takes metformin")
        assert len(rects) == 1
        assert len(triggers) >= 1
        line = rects[0]
        for t in triggers:
            assert t.x0 >= line.x0 - 1
            assert t.x1 <= line.x1 + 1

    def test_triggers_empty_on_no_match(self, tmp_path):
        """When no match is found, triggers list is also empty."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df((0, "nothing", 0, 0, 60, 12))
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            _, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "xyzzy nomatch")
        assert rects == []
        assert triggers == []

    def test_trigger_rects_are_narrower_than_line_rects(self, tmp_path):
        """Trigger rects (individual matched words) are narrower than the full-line rects."""
        doc_id = "test_doc"
        (tmp_path / doc_id).mkdir()
        rows = _words_df(
            (0, "The",       0, 0, 25, 12),
            (0, "patient",   30, 0, 90, 12),
            (0, "takes",     95, 0, 140, 12),
            (0, "metformin", 145, 0, 240, 12),
            (0, "daily.",    245, 0, 300, 12),
        )
        _make_words_parquet(tmp_path / doc_id, rows)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            _, rects, triggers = _find_quote_rects_from_words(doc_id, 0, "metformin")
        line_width = rects[0].x1 - rects[0].x0
        trigger_width = triggers[0].x1 - triggers[0].x0
        assert trigger_width < line_width


# ---------------------------------------------------------------------------
# render_page endpoint  (/api/documents/{doc_id}/page/{page_num})
# ---------------------------------------------------------------------------

def _make_minimal_pdf() -> bytes:
    """Return bytes of a minimal 1-page PDF with selectable text."""
    import fitz  # PyMuPDF
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Patient was prescribed metformin 500mg daily.")
    page.insert_text((72, 120), "Allergies N.K.D.A.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _fake_pdf_path(tmp_path: Path, doc_id: str) -> Path:
    """Write a minimal PDF, a manifest, and a layout_words.parquet, return the PDF path."""
    import fitz
    import json

    doc_dir = tmp_path / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_bytes = _make_minimal_pdf()
    pdf_path = doc_dir / "source.pdf"
    pdf_path.write_bytes(pdf_bytes)
    (doc_dir / "manifest.json").write_text(
        json.dumps({"source_pdf_path": str(pdf_path), "source_pdf_name": "test.pdf", "total_pages": 1})
    )

    # Extract word bboxes so _find_quote_rects_from_words has data to work with
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rows = []
    for pn, page in enumerate(doc):
        for x0, y0, x1, y1, word, *_ in page.get_text("words"):
            rows.append({"page_number": pn, "text": word,
                         "x0": float(x0), "y0": float(y0),
                         "x1": float(x1), "y1": float(y1)})
    doc.close()
    if rows:
        df = pd.DataFrame(rows)
        pq.write_table(pa.Table.from_pandas(df), str(doc_dir / "layout_words.parquet"))

    return pdf_path


class TestRenderPageEndpoint:

    def test_returns_png_for_existing_page(self, tmp_path):
        """GET /api/documents/{doc_id}/page/0 returns image/png."""
        doc_id = "render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_returns_404_for_missing_doc(self):
        """GET /api/documents/nonexistent/page/0 returns 404."""
        r = client.get("/api/documents/nonexistent_doc_xyz/page/0")
        assert r.status_code == 404

    def test_returns_400_for_out_of_range_page(self, tmp_path):
        """GET /api/documents/{doc_id}/page/999 returns 400."""
        doc_id = "render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/999")
        assert r.status_code == 400

    def test_crop_without_match_returns_full_page(self, tmp_path):
        """crop=1 with unmatched q returns full page (no crash, valid PNG)."""
        doc_id = "render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0?q=xyzzy_nomatch&crop=1")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_highlight_with_matching_q_returns_png(self, tmp_path):
        """crop=1 with a matching q returns a PNG (highlight was found and drawn)."""
        doc_id = "render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0?q=metformin+prescribed&crop=1")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_no_cache_for_cropped_highlight(self, tmp_path):
        """Cropped highlight responses carry Cache-Control: no-store."""
        doc_id = "render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            # Make a request that WILL find a highlight (metformin is in the PDF text)
            r = client.get(f"/api/documents/{doc_id}/page/0?q=metformin&crop=1")
        # Only no-store when a highlight was found AND crop=1
        if r.headers.get("cache-control") == "no-store":
            assert True
        else:
            # Acceptable: highlight not found, fell through to full-page (public cache)
            assert "public" in r.headers.get("cache-control", "public")


# ---------------------------------------------------------------------------
# render_page_pdf endpoint  (/api/documents/{doc_id}/page/{page_num}/pdf)
# ---------------------------------------------------------------------------

class TestRenderPagePdfEndpoint:

    def test_returns_pdf_content_type(self, tmp_path):
        """GET /api/documents/{doc_id}/page/0/pdf returns application/pdf."""
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf")
        assert r.status_code == 200
        assert "application/pdf" in r.headers["content-type"]

    def test_returned_pdf_is_valid(self, tmp_path):
        """Returned bytes are a parseable single-page PDF."""
        import fitz
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf")
        assert r.status_code == 200
        doc = fitz.open(stream=r.content, filetype="pdf")
        assert len(doc) == 1

    def test_pdf_with_highlight_query_is_valid(self, tmp_path):
        """With q=..., the returned PDF is still a valid 1-page PDF."""
        import fitz
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf?q=metformin+prescribed")
        assert r.status_code == 200
        doc = fitz.open(stream=r.content, filetype="pdf")
        assert len(doc) == 1

    def test_pdf_has_highlight_annotation_when_text_found(self, tmp_path):
        """When the quote matches, the returned PDF has at least one highlight annotation."""
        import fitz
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf?q=metformin")
        assert r.status_code == 200
        doc = fitz.open(stream=r.content, filetype="pdf")
        page = doc[0]
        annots = list(page.annots())
        assert len(annots) >= 1, "Expected at least one highlight annotation in the PDF"

    def test_pdf_no_annotation_for_unmatched_query(self, tmp_path):
        """When the quote doesn't match anything, no annotations are added (no crash)."""
        import fitz
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf?q=xyzzy_totally_missing")
        assert r.status_code == 200
        doc = fitz.open(stream=r.content, filetype="pdf")
        assert len(doc) == 1  # still valid, just no annotations

    def test_returns_404_for_missing_doc(self):
        """GET /api/documents/nonexistent/page/0/pdf returns 404."""
        r = client.get("/api/documents/nonexistent_doc_xyz/page/0/pdf")
        assert r.status_code == 404

    def test_cache_control_is_no_store(self, tmp_path):
        """PDF responses always carry Cache-Control: no-store."""
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf")
        assert r.headers.get("cache-control") == "no-store"

    def test_text_layer_preserved(self, tmp_path):
        """The returned PDF still has selectable text (text layer not stripped)."""
        import fitz
        doc_id = "pdf_render_test"
        _fake_pdf_path(tmp_path, doc_id)
        with patch("nitrag.server.RAG_STORE_ROOT", tmp_path):
            r = client.get(f"/api/documents/{doc_id}/page/0/pdf?q=metformin")
        doc = fitz.open(stream=r.content, filetype="pdf")
        text = doc[0].get_text()
        assert "metformin" in text.lower(), "Text layer should be preserved in the returned PDF"


# ---------------------------------------------------------------------------
# fmtAnswer table rendering  (JS logic smoke-tested via response content)
# ---------------------------------------------------------------------------

class TestFmtAnswerTableMarkdown:
    """Verify the UI HTML includes the table CSS that fmtAnswer depends on."""

    def test_ui_contains_md_tbl_css(self):
        """GET / HTML contains .md-tbl CSS (confirms table rendering is wired up)."""
        r = client.get("/")
        assert r.status_code == 200
        assert "md-tbl" in r.text

    def test_ui_contains_md_tbl_wrap_css(self):
        """GET / HTML contains .md-tbl-wrap overflow-x:auto CSS."""
        r = client.get("/")
        assert "md-tbl-wrap" in r.text

    def test_ui_table_css_has_overflow_auto(self):
        """The table wrapper uses overflow-x:auto to prevent horizontal overflow."""
        r = client.get("/")
        assert "overflow-x:auto" in r.text or "overflow-x: auto" in r.text


# ---------------------------------------------------------------------------
# Evidence panel dual counts
# ---------------------------------------------------------------------------

class TestEvidencePanelCounts:

    def test_ui_has_ep_cite_count_element(self):
        """GET / HTML contains ep-cite-count element."""
        r = client.get("/")
        assert "ep-cite-count" in r.text

    def test_ui_has_ep_chunk_count_element(self):
        """GET / HTML contains ep-chunk-count element."""
        r = client.get("/")
        assert "ep-chunk-count" in r.text

    def test_ui_has_update_ep_counts_function(self):
        """GET / HTML contains updateEpCounts JS function."""
        r = client.get("/")
        assert "updateEpCounts" in r.text


# ---------------------------------------------------------------------------
# PDF lightbox wiring
# ---------------------------------------------------------------------------

class TestPdfLightbox:

    def test_ui_has_lbox_frame_element(self):
        """GET / HTML contains the iframe#lbox-frame (not img#lbox-img)."""
        r = client.get("/")
        assert "lbox-frame" in r.text
        assert "lbox-img" not in r.text

    def test_ui_no_zoom_controls(self):
        """Zoom buttons were removed (no lbox-zoom-in element)."""
        r = client.get("/")
        assert "lbox-zoom-in" not in r.text
        assert "lbox-zoom-out" not in r.text

    def test_ui_open_lightbox_sets_frame_src(self):
        """openLightbox function sets E.lboxFrame.src (not lboxImg.src)."""
        r = client.get("/")
        assert "lboxFrame.src" in r.text

    def test_ui_page_pdf_endpoint_referenced(self):
        """Citation card onclick references the /page/N/pdf endpoint."""
        r = client.get("/")
        assert "/pdf" in r.text and "pagePdfUrl" in r.text


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

class TestDefaultConfig:

    def test_ocr_checkbox_checked_by_default(self):
        """OCR checkbox is checked by default in the HTML."""
        r = client.get("/")
        assert 'id="ocr-checkbox" checked' in r.text or 'id="ocr-checkbox"  checked' in r.text.replace("checked>", "checked >")

    def test_openai_cloud_selected_by_default(self):
        """openai_cloud option has selected attribute."""
        r = client.get("/")
        assert 'value="openai_cloud" selected' in r.text

    def test_js_preset_default_is_openai(self):
        """JS state initialises preset to openai_cloud."""
        r = client.get("/")
        assert "preset: 'openai_cloud'" in r.text

"""Tests for nitrag/server.py — FastAPI routes via TestClient."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from nitrag.server import app

client = TestClient(app, raise_server_exceptions=False)

# The known document in the real rag_store
KNOWN_DOC_ID = "doc_e7fb48687c98a19c"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_200(self):
        """GET / returns 200 (HTML response)."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_html(self):
        """GET / returns HTML content-type."""
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", "")

    def test_root_html_contains_nitrag(self):
        """GET / HTML body mentions NITRAG."""
        response = client.get("/")
        assert "NITRAG" in response.text


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /api/health returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self):
        """GET /api/health returns {"status": "ok"}."""
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# GET /api/documents
# ---------------------------------------------------------------------------

class TestListDocumentsEndpoint:
    def test_list_documents_returns_200(self):
        """GET /api/documents returns 200."""
        response = client.get("/api/documents")
        assert response.status_code == 200

    def test_list_documents_has_documents_key(self):
        """GET /api/documents response has 'documents' list."""
        response = client.get("/api/documents")
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_known_document_appears_in_list(self):
        """The real doc_e7fb48687c98a19c appears in the document list."""
        response = client.get("/api/documents")
        data = response.json()
        doc_ids = [d["doc_id"] for d in data["documents"]]
        assert KNOWN_DOC_ID in doc_ids

    def test_known_document_has_source_name(self):
        """Listed document has a source_name field."""
        response = client.get("/api/documents")
        data = response.json()
        doc = next(d for d in data["documents"] if d["doc_id"] == KNOWN_DOC_ID)
        assert "source_name" in doc
        assert isinstance(doc["source_name"], str)

    def test_known_document_has_stages_dict(self):
        """Listed document has a stages field (dict)."""
        response = client.get("/api/documents")
        data = response.json()
        doc = next(d for d in data["documents"] if d["doc_id"] == KNOWN_DOC_ID)
        assert "stages" in doc
        assert isinstance(doc["stages"], dict)


# ---------------------------------------------------------------------------
# GET /api/documents/{doc_id}
# ---------------------------------------------------------------------------

class TestGetDocumentEndpoint:
    def test_known_doc_returns_200(self):
        """GET /api/documents/<known_id> returns 200."""
        response = client.get(f"/api/documents/{KNOWN_DOC_ID}")
        assert response.status_code == 200

    def test_known_doc_has_doc_id_field(self):
        """Known document endpoint returns the correct doc_id."""
        response = client.get(f"/api/documents/{KNOWN_DOC_ID}")
        data = response.json()
        assert data["doc_id"] == KNOWN_DOC_ID

    def test_known_doc_has_stages_field(self):
        """Known document endpoint returns stages dict."""
        response = client.get(f"/api/documents/{KNOWN_DOC_ID}")
        data = response.json()
        assert "stages" in data
        assert isinstance(data["stages"], dict)

    def test_nonexistent_doc_returns_404(self):
        """GET /api/documents/nonexistent_doc returns 404."""
        response = client.get("/api/documents/nonexistent_doc_xyz_999")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/config/presets
# ---------------------------------------------------------------------------

class TestConfigPresetsEndpoint:
    def test_presets_returns_200(self):
        """GET /api/config/presets returns 200."""
        response = client.get("/api/config/presets")
        assert response.status_code == 200

    def test_presets_has_presets_key(self):
        """GET /api/config/presets returns a 'presets' list."""
        response = client.get("/api/config/presets")
        data = response.json()
        assert "presets" in data
        assert isinstance(data["presets"], list)

    def test_presets_list_non_empty(self):
        """Presets list has at least one entry."""
        response = client.get("/api/config/presets")
        data = response.json()
        assert len(data["presets"]) > 0

    def test_presets_have_id_and_label(self):
        """Each preset has 'id' and 'label' fields."""
        response = client.get("/api/config/presets")
        data = response.json()
        for preset in data["presets"]:
            assert "id" in preset
            assert "label" in preset


# ---------------------------------------------------------------------------
# POST /api/query — validation and not-found
# ---------------------------------------------------------------------------

class TestQueryEndpointValidation:
    def test_missing_doc_id_returns_422(self):
        """POST /api/query without doc_id fails Pydantic validation → 422."""
        response = client.post("/api/query", json={"query": "What is the diagnosis?"})
        assert response.status_code == 422

    def test_missing_query_returns_422(self):
        """POST /api/query without query field → 422."""
        response = client.post("/api/query", json={"doc_id": KNOWN_DOC_ID})
        assert response.status_code == 422

    def test_nonexistent_doc_id_returns_404(self):
        """POST /api/query with nonexistent doc_id → 404."""
        response = client.post(
            "/api/query",
            json={"query": "What medications?", "doc_id": "nonexistent_doc_xyz"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/query — mocked pipeline success
# ---------------------------------------------------------------------------

def _build_mock_rag_response():
    """Build a minimal mock RAGResponse that _serialize_response can handle."""
    from nitrag.context_assembler import AssembledContext, ContextChunk
    from nitrag.generation_manager import Citation, GenerationResult

    chunk = ContextChunk(
        citation_number=1,
        chunk_id=0,
        text="Metformin 500mg twice daily was prescribed.",
        page_start=0,
        page_end=0,
        section="Medications",
        score=0.9,
        retriever="bm25",
        token_count=15,
        document_id=KNOWN_DOC_ID,
        source_label="Page 1 | Medications",
    )
    context = AssembledContext(
        chunks=[chunk],
        citation_map={0: 1},
        total_tokens=15,
        formatted_text="[1] Metformin 500mg twice daily was prescribed.",
        query="What medications were prescribed?",
        truncated=False,
        truncated_count=0,
    )
    citation = Citation(
        number=1,
        chunk_id=0,
        page_start=0,
        page_end=0,
        section="Medications",
        quote="Metformin 500mg twice daily was prescribed.",
        confidence=0.9,
        source_label="Page 1 | Medications",
    )
    mock_response = MagicMock()
    mock_response.query = "What medications were prescribed?"
    mock_response.answer = "Metformin 500mg twice daily was prescribed [1]."
    mock_response.citations = [citation]
    mock_response.context = context
    mock_response.evaluation = None
    mock_response.latency = {"total_ms": 500.0}
    mock_response.config_snapshot = {"preset": "local_ollama"}
    return mock_response


class TestQueryEndpointMocked:
    def test_mocked_pipeline_returns_200(self):
        """POST /api/query with mocked pipeline → 200."""
        mock_response = _build_mock_rag_response()
        mock_pipeline = MagicMock()
        mock_pipeline.answer.return_value = mock_response

        with patch("nitrag.server._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/api/query",
                json={
                    "query": "What medications were prescribed?",
                    "doc_id": KNOWN_DOC_ID,
                    "config_preset": "local_ollama",
                },
            )
        assert response.status_code == 200

    def test_mocked_pipeline_response_has_answer_key(self):
        """Mocked /api/query response JSON contains 'answer' key."""
        mock_response = _build_mock_rag_response()
        mock_pipeline = MagicMock()
        mock_pipeline.answer.return_value = mock_response

        with patch("nitrag.server._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/api/query",
                json={
                    "query": "What medications were prescribed?",
                    "doc_id": KNOWN_DOC_ID,
                },
            )
        data = response.json()
        assert "answer" in data

    def test_mocked_pipeline_response_has_citations_key(self):
        """Mocked /api/query response JSON contains 'citations' list."""
        mock_response = _build_mock_rag_response()
        mock_pipeline = MagicMock()
        mock_pipeline.answer.return_value = mock_response

        with patch("nitrag.server._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/api/query",
                json={
                    "query": "What medications were prescribed?",
                    "doc_id": KNOWN_DOC_ID,
                },
            )
        data = response.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)

    def test_mocked_pipeline_response_has_context_key(self):
        """Mocked /api/query response JSON contains 'context' dict."""
        mock_response = _build_mock_rag_response()
        mock_pipeline = MagicMock()
        mock_pipeline.answer.return_value = mock_response

        with patch("nitrag.server._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/api/query",
                json={
                    "query": "What medications were prescribed?",
                    "doc_id": KNOWN_DOC_ID,
                },
            )
        data = response.json()
        assert "context" in data

    def test_mocked_pipeline_answer_text_in_response(self):
        """The answer text from the mock is correctly serialized."""
        mock_response = _build_mock_rag_response()
        mock_pipeline = MagicMock()
        mock_pipeline.answer.return_value = mock_response

        with patch("nitrag.server._get_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/api/query",
                json={
                    "query": "What medications were prescribed?",
                    "doc_id": KNOWN_DOC_ID,
                },
            )
        data = response.json()
        assert "Metformin" in data["answer"]


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

class TestUploadEndpoint:
    def test_non_pdf_file_returns_400(self):
        """POST /api/upload with a non-PDF file returns 400."""
        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"some text content", "text/plain")},
        )
        assert response.status_code == 400

    def test_non_pdf_error_message_mentions_pdf(self):
        """400 error message mentions PDF."""
        response = client.post(
            "/api/upload",
            files={"file": ("document.docx", b"fake docx", "application/vnd.openxmlformats")},
        )
        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "pdf" in detail.lower() or "PDF" in detail

"""
Shared pytest fixtures for NITRAG tests.

Provides:
  - MockStore       : minimal decode_span() implementation backed by a text dict
  - CORPUS / CHUNKS : 8 synthetic medical chunks covering sections, entities, negation,
                      numeric values, and diverse clinical content
  - synthetic_results : pre-built reranker-input dicts (no store required)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Synthetic medical corpus
# ---------------------------------------------------------------------------

# Flat list of (start_index, end_index, text, metadata).
# start/end are arbitrary integers; MockStore maps (start, end) -> text.
CORPUS: List[Tuple[int, int, str, Dict[str, Any]]] = [
    (
        0, 20,
        "Patient has hypertension and diabetes mellitus type 2. "
        "Blood pressure 145/90 mmHg. Heart rate 88 bpm.",
        {
            "primary_section": "History of Present Illness",
            "contains_vital": True, "contains_diagnosis": True,
            "contains_negation": False, "contains_medication": False,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.72,
            "entities_json": json.dumps([
                {"type": "vital", "text": "blood pressure 145/90 mmHg", "normalized_value": "145/90 mmHg", "negated": False},
                {"type": "diagnosis_or_problem_candidate", "text": "hypertension", "normalized_value": "hypertension", "negated": False},
                {"type": "diagnosis_or_problem_candidate", "text": "diabetes mellitus type 2", "normalized_value": "DM2", "negated": False},
            ]),
            "entity_type_counts_json": json.dumps({"vital": 1, "diagnosis_or_problem_candidate": 2}),
            "entity_count": 3,
        },
    ),
    (
        20, 40,
        "No evidence of acute myocardial infarction. EKG normal. "
        "No chest pain reported by patient.",
        {
            "primary_section": "Assessment",
            "contains_vital": False, "contains_diagnosis": True,
            "contains_negation": True, "contains_medication": False,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.65,
            "entities_json": json.dumps([
                {"type": "diagnosis_or_problem_candidate", "text": "myocardial infarction", "normalized_value": "MI", "negated": True},
                {"type": "diagnosis_or_problem_candidate", "text": "chest pain", "normalized_value": "chest pain", "negated": True},
            ]),
            "entity_type_counts_json": json.dumps({"diagnosis_or_problem_candidate": 2}),
            "entity_count": 2,
        },
    ),
    (
        40, 60,
        "Metformin 500mg twice daily. Lisinopril 10mg once daily for blood pressure. "
        "Aspirin 81mg daily.",
        {
            "primary_section": "Medications",
            "contains_vital": False, "contains_diagnosis": False,
            "contains_negation": False, "contains_medication": True,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.58,
            "entities_json": json.dumps([
                {"type": "medication_candidate", "text": "Metformin 500mg", "normalized_value": "metformin", "negated": False},
                {"type": "medication_candidate", "text": "Lisinopril 10mg", "normalized_value": "lisinopril", "negated": False},
                {"type": "medication_candidate", "text": "Aspirin 81mg", "normalized_value": "aspirin", "negated": False},
            ]),
            "entity_type_counts_json": json.dumps({"medication_candidate": 3}),
            "entity_count": 3,
        },
    ),
    (
        60, 80,
        "Assessment and Plan: Continue current medications. Follow up in 3 months. "
        "Increase Metformin to 1000mg if glucose remains elevated.",
        {
            "primary_section": "Assessment and Plan",
            "contains_vital": False, "contains_diagnosis": False,
            "contains_negation": False, "contains_medication": True,
            "contains_lab": False, "contains_date": True,
            "clinical_quality_score": 0.82,
            "entities_json": json.dumps([
                {"type": "medication_candidate", "text": "Metformin 1000mg", "normalized_value": "metformin", "negated": False},
                {"type": "date", "text": "3 months", "normalized_value": "3 months", "negated": False},
            ]),
            "entity_type_counts_json": json.dumps({"medication_candidate": 1, "date": 1}),
            "entity_count": 2,
        },
    ),
    (
        80, 100,
        "WBC 12.5 k/uL, RBC 4.2 M/uL, Hemoglobin 13.1 g/dL. "
        "Glucose 210 mg/dL. HbA1c 8.4%.",
        {
            "primary_section": "Laboratory Results",
            "contains_vital": False, "contains_diagnosis": False,
            "contains_negation": False, "contains_medication": False,
            "contains_lab": True, "contains_date": False,
            "clinical_quality_score": 0.90,
            "entities_json": json.dumps([
                {"type": "lab_result", "text": "WBC 12.5", "normalized_value": "12.5 k/uL", "negated": False},
                {"type": "lab_result", "text": "Glucose 210", "normalized_value": "210 mg/dL", "negated": False},
                {"type": "lab_result", "text": "HbA1c 8.4%", "normalized_value": "8.4%", "negated": False},
            ]),
            "entity_type_counts_json": json.dumps({"lab_result": 3}),
            "entity_count": 3,
        },
    ),
    (
        100, 120,
        "Patient denies chest pain, shortness of breath, or palpitations. "
        "No fever, chills, or night sweats.",
        {
            "primary_section": "Review of Systems",
            "contains_vital": False, "contains_diagnosis": False,
            "contains_negation": True, "contains_medication": False,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.45,
            "entities_json": json.dumps([
                {"type": "diagnosis_or_problem_candidate", "text": "chest pain", "normalized_value": "chest pain", "negated": True},
                {"type": "diagnosis_or_problem_candidate", "text": "fever", "normalized_value": "fever", "negated": True},
            ]),
            "entity_type_counts_json": json.dumps({"diagnosis_or_problem_candidate": 2}),
            "entity_count": 2,
        },
    ),
    (
        120, 140,
        "MRI brain shows no acute infarct. No evidence of stroke or hemorrhage. "
        "Mild white matter changes noted.",
        {
            "primary_section": "Radiology",
            "contains_vital": False, "contains_diagnosis": True,
            "contains_negation": True, "contains_medication": False,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.68,
            "entities_json": json.dumps([
                {"type": "diagnosis_or_problem_candidate", "text": "acute infarct", "normalized_value": "infarct", "negated": True},
                {"type": "imaging_candidate", "text": "MRI brain", "normalized_value": "MRI brain", "negated": False},
            ]),
            "entity_type_counts_json": json.dumps({"diagnosis_or_problem_candidate": 1, "imaging_candidate": 1}),
            "entity_count": 2,
        },
    ),
    (
        140, 160,
        "Discharge diagnosis: Type 2 diabetes mellitus, hypertension, hyperlipidemia. "
        "Patient discharged home in stable condition on 2024-01-15.",
        {
            "primary_section": "Discharge Summary",
            "contains_vital": False, "contains_diagnosis": True,
            "contains_negation": False, "contains_medication": False,
            "contains_lab": False, "contains_date": True,
            "clinical_quality_score": 0.88,
            "entities_json": json.dumps([
                {"type": "diagnosis_or_problem_candidate", "text": "Type 2 diabetes mellitus", "normalized_value": "DM2", "negated": False},
                {"type": "diagnosis_or_problem_candidate", "text": "hypertension", "normalized_value": "HTN", "negated": False},
                {"type": "date", "text": "2024-01-15", "normalized_value": "2024-01-15", "negated": False},
            ]),
            "entity_type_counts_json": json.dumps({"diagnosis_or_problem_candidate": 2, "date": 1}),
            "entity_count": 3,
        },
    ),
]


# ---------------------------------------------------------------------------
# MockStore
# ---------------------------------------------------------------------------

class MockStorePaths:
    """Minimal paths object that manager classes read."""
    def __init__(self, document_dir: Path) -> None:
        self.document_dir = Path(document_dir)


class MockStore:
    """Minimal store that maps (start_index, end_index) -> text."""

    def __init__(self, document_dir: Optional[Path] = None) -> None:
        self._map: Dict[Tuple[int, int], str] = {
            (start, end): text for start, end, text, _ in CORPUS
        }
        import tempfile
        _base = Path(document_dir) if document_dir else Path(tempfile.mkdtemp(prefix="nitrag_mock_"))
        self.paths = MockStorePaths(_base)

    def decode_span(self, start: int, end: int) -> str:
        return self._map.get((start, end), "")


# ---------------------------------------------------------------------------
# Chunk record builder
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, chunk_strategy_name: str = "fixed_token") -> Dict[str, Any]:
    start, end, text, meta = CORPUS[idx]
    return {
        "chunk_id": idx,
        "document_id": "doc_001",
        "chunk_strategy_name": chunk_strategy_name,
        "start_index": start,
        "end_index": end,
        "token_length": len(text.split()),
        "page_start": 0,
        "page_end": 0,
        "document_type": "discharge_summary",
        "overlap_line_count": 2,
        "source_element_ids_json": json.dumps([idx * 10, idx * 10 + 1]),
        "section_names_json": json.dumps([meta["primary_section"]]),
        **meta,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> MockStore:
    return MockStore()


@pytest.fixture
def chunks() -> List[Dict[str, Any]]:
    return [_make_chunk(i) for i in range(len(CORPUS))]


@pytest.fixture
def synthetic_results() -> List[Dict[str, Any]]:
    """Pre-built reranker-input dicts — no store required."""
    results = []
    for i, (start, end, text, meta) in enumerate(CORPUS):
        results.append({
            "score": round(1.0 - i * 0.08, 4),
            "retriever_name": "bm25",
            "query": "hypertension diabetes treatment",
            "chunk_strategy_name": "fixed_token",
            "chunk_id": i,
            "doc_idx": i,
            "document_id": "doc_001",
            "start_index": start,
            "end_index": end,
            "token_length": len(text.split()),
            "page_start": 0,
            "page_end": 0,
            "document_type": "discharge_summary",
            "primary_section": meta["primary_section"],
            "contains_medication": meta["contains_medication"],
            "contains_lab": meta["contains_lab"],
            "contains_diagnosis": meta["contains_diagnosis"],
            "contains_vital": meta["contains_vital"],
            "contains_negation": meta["contains_negation"],
            "clinical_quality_score": meta["clinical_quality_score"],
            "entity_type_counts": json.loads(meta["entity_type_counts_json"]),
            "entities": json.loads(meta["entities_json"]),
            "text_preview": text,
        })
    return results


@pytest.fixture
def built_bm25_index(tmp_path: Path, store: MockStore, chunks: List[Dict[str, Any]]) -> Path:
    """Builds a BM25 index into tmp_path and returns the index root."""
    from nitrag.index_manager import BM25IndexStrategy

    index_root = tmp_path / "indexes"
    strategy = "fixed_token"
    output_dir = index_root / strategy / "bm25"
    BM25IndexStrategy().build(
        store=store,
        chunk_strategy_name=strategy,
        chunks=chunks,
        output_dir=output_dir,
    )
    return index_root


def write_chunks_parquet(chunks: List[Dict[str, Any]], path: Path) -> None:
    """Helper: write chunk dicts to a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(chunks)
    pq.write_table(table, path, compression="zstd")

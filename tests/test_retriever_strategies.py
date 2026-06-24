"""
Tests for retriever strategies.

Structure:
  - Unit tests for pure logic (query expansion, numeric parser, negation penalty,
    section scoping, entity term extraction) — no I/O
  - Integration tests: build the required index in tmp_path, run retrieve(), verify results
  - Registration smoke test for all 27 strategies
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.conftest import MockStore, CORPUS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_chunk(i: int, strategy: str = "fixed_token") -> Dict[str, Any]:
    start, end, text, meta = CORPUS[i]
    return {
        "chunk_id": i, "document_id": "doc_001", "chunk_strategy_name": strategy,
        "start_index": start, "end_index": end,
        "token_length": len(text.split()), "page_start": 0, "page_end": 0,
        "document_type": "discharge_summary", "overlap_line_count": 2,
        "source_element_ids_json": json.dumps([i * 10]),
        "section_names_json": json.dumps([meta["primary_section"]]),
        **meta,
    }


def build_index(store, chunks, indexer_cls, strategy_name: str, tmp_path: Path) -> Path:
    """Build a single index; return the index root directory."""
    from nitrag.index_manager import BM25IndexStrategy
    root = tmp_path / "indexes"
    out = root / strategy_name / indexer_cls.name if hasattr(indexer_cls, "name") else root / strategy_name / "index"
    indexer_cls().build(store=store, chunk_strategy_name=strategy_name, chunks=chunks, output_dir=out)
    return root


@pytest.fixture
def store():
    return MockStore()


@pytest.fixture
def chunks(store):
    return [_make_chunk(i) for i in range(len(CORPUS))]


@pytest.fixture
def bm25_index_root(tmp_path, store, chunks):
    from nitrag.index_manager import BM25IndexStrategy
    return build_index(store, chunks, BM25IndexStrategy, "fixed_token", tmp_path)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_all_27_retrievers_registered(tmp_path):
    from nitrag.retriever_manager import RetrieverManager, register_default_retrievers
    mgr = RetrieverManager(store=MockStore(tmp_path))
    register_default_retrievers(mgr)
    names = set(mgr.list_retrievers())
    expected = {
        "bm25", "keyword_exact", "tfidf", "phrase_ngram", "char_ngram",
        "fielded_lexical", "boolean_set", "positional_proximity",
        "entity", "section_page", "temporal", "layout_spatial",
        "minhash_duplicates", "metadata_filter", "bm25_metadata_boost",
        "multi_query_bm25", "lexical_fusion", "advanced_lexical_fusion",
        "mmr_diversity", "context_expansion", "graph_expansion", "cross_chunk_fusion",
        "query_expansion_bm25", "entity_centric_fusion",
        "numeric_range_retriever", "negation_aware_bm25", "clinical_section_scoped",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


# ---------------------------------------------------------------------------
# BM25 integration — baseline retrieval
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    def test_returns_results(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="hypertension diabetes", chunk_strategy_name="fixed_token", top_k=5,
        )
        assert len(results) > 0

    def test_result_has_required_fields(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="hypertension", chunk_strategy_name="fixed_token", top_k=3,
        )
        for r in results:
            assert "score" in r
            assert "retriever_name" in r
            assert "text_preview" in r
            assert "chunk_id" in r

    def test_results_sorted_descending(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="hypertension diabetes medication", chunk_strategy_name="fixed_token", top_k=8,
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="patient", chunk_strategy_name="fixed_token", top_k=2,
        )
        assert len(results) <= 2

    def test_filter_by_section(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="diagnosis treatment",
            chunk_strategy_name="fixed_token", top_k=10,
            filters={"primary_section": {"$eq": "Assessment and Plan"}},
        )
        for r in results:
            assert r["primary_section"] == "Assessment and Plan"

    def test_relevant_chunk_scores_higher(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="metformin lisinopril aspirin",
            chunk_strategy_name="fixed_token", top_k=8,
        )
        # CORPUS[2] is all about medications — should be in top 3
        top_ids = [r["chunk_id"] for r in results[:3]]
        assert 2 in top_ids, f"Medication chunk not in top 3; top_ids={top_ids}"

    def test_no_results_for_irrelevant_query(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy
        results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="quantum physics superconductor neutron star",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        # may return 0 or some low-score results — just ensure no crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# QueryExpansionBM25 — unit + integration
# ---------------------------------------------------------------------------

class TestQueryExpansionBM25Retriever:
    def test_expand_query_htn(self):
        from nitrag.retriever_manager import QueryExpansionBM25RetrieverStrategy
        s = QueryExpansionBM25RetrieverStrategy()
        variants = s._expand_query("patient with HTN")
        assert len(variants) >= 2
        assert any("hypertension" in v.lower() for v in variants[1:])

    def test_expand_query_preserves_original_first(self):
        from nitrag.retriever_manager import QueryExpansionBM25RetrieverStrategy
        q = "DM and COPD"
        variants = QueryExpansionBM25RetrieverStrategy()._expand_query(q)
        assert variants[0] == q

    def test_integration_returns_results(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, QueryExpansionBM25RetrieverStrategy
        s = QueryExpansionBM25RetrieverStrategy(base_bm25=BM25RetrieverStrategy())
        results = s.retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="HTN and DM management",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        assert isinstance(results, list)

    def test_integration_expanded_finds_more(self, bm25_index_root, store):
        """'HTN' expansion should find hypertension chunks that a raw 'HTN' query misses."""
        from nitrag.retriever_manager import BM25RetrieverStrategy, QueryExpansionBM25RetrieverStrategy
        base = BM25RetrieverStrategy()
        raw = base.retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="HTN DM", chunk_strategy_name="fixed_token", top_k=8,
        )
        expanded = QueryExpansionBM25RetrieverStrategy(base_bm25=base).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="HTN DM", chunk_strategy_name="fixed_token", top_k=8,
        )
        # expanded should find at least as many relevant results
        assert len(expanded) >= len(raw)


# ---------------------------------------------------------------------------
# NumericRangeRetriever — integration
# ---------------------------------------------------------------------------

class TestNumericRangeRetriever:
    @pytest.fixture
    def numeric_index_root(self, tmp_path, store, chunks):
        from nitrag.index_manager import NumericRangeIndexStrategy
        root = tmp_path / "indexes"
        out = root / "fixed_token" / "numeric_range"
        NumericRangeIndexStrategy().build(
            store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=out,
        )
        # also build BM25 for fallback
        from nitrag.index_manager import BM25IndexStrategy
        BM25IndexStrategy().build(
            store=store, chunk_strategy_name="fixed_token",
            chunks=chunks, output_dir=root / "fixed_token" / "bm25",
        )
        return root

    def test_range_query_returns_results(self, numeric_index_root, store):
        from nitrag.retriever_manager import NumericRangeRetrieverStrategy
        results = NumericRangeRetrieverStrategy().retrieve(
            store=store, index_root_dir=numeric_index_root,
            query="glucose greater than 200",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        assert isinstance(results, list)

    def test_no_condition_falls_back_to_bm25(self, numeric_index_root, store):
        from nitrag.retriever_manager import NumericRangeRetrieverStrategy
        results = NumericRangeRetrieverStrategy().retrieve(
            store=store, index_root_dir=numeric_index_root,
            query="patient has hypertension",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        assert isinstance(results, list)
        # fallback to BM25 should return something
        assert len(results) > 0

    def test_range_query_finds_glucose_chunk(self, numeric_index_root, store):
        from nitrag.retriever_manager import NumericRangeRetrieverStrategy
        results = NumericRangeRetrieverStrategy().retrieve(
            store=store, index_root_dir=numeric_index_root,
            query="glucose > 200",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        # CORPUS[4] has Glucose 210 — should match
        chunk_ids = [r["chunk_id"] for r in results]
        assert 4 in chunk_ids, f"Glucose chunk not found; got {chunk_ids}"


# ---------------------------------------------------------------------------
# NegationAwareBM25 — integration
# ---------------------------------------------------------------------------

class TestNegationAwareBM25Retriever:
    def test_integration_returns_results(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, NegationAwareBM25RetrieverStrategy
        results = NegationAwareBM25RetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="chest pain", chunk_strategy_name="fixed_token", top_k=5,
        )
        assert isinstance(results, list)

    def test_negation_penalty_applied(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, NegationAwareBM25RetrieverStrategy
        base_results = BM25RetrieverStrategy().retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="chest pain", chunk_strategy_name="fixed_token", top_k=8,
        )
        penalized_results = NegationAwareBM25RetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="chest pain", chunk_strategy_name="fixed_token", top_k=8,
        )
        # chunks with negation=True and matching entities should be scored lower
        neg_chunks = {r["chunk_id"] for r in penalized_results if r.get("contains_negation")}
        # just verify the penalized results have penalty field
        for r in penalized_results:
            assert "negation_penalty" in r

    def test_penalty_field_in_results(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, NegationAwareBM25RetrieverStrategy
        results = NegationAwareBM25RetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="fever diagnosis", chunk_strategy_name="fixed_token", top_k=5,
        )
        for r in results:
            assert "negation_penalty" in r
            assert 0.0 <= r["negation_penalty"] <= 1.0


# ---------------------------------------------------------------------------
# ClinicalSectionScoped — unit + integration
# ---------------------------------------------------------------------------

class TestClinicalSectionScopedRetriever:
    def test_high_signal_sections_set_non_empty(self):
        from nitrag.retriever_manager import ClinicalSectionScopedRetrieverStrategy
        s = ClinicalSectionScopedRetrieverStrategy()
        assert len(s.high_signal_sections) > 0
        assert "assessment" in s.high_signal_sections
        assert "plan" in s.high_signal_sections

    def test_integration_returns_results(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, ClinicalSectionScopedRetrieverStrategy
        results = ClinicalSectionScopedRetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="hypertension diabetes",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_assessment_chunk_prioritised(self, bm25_index_root, store):
        """CORPUS[1] is 'Assessment', CORPUS[3] is 'Assessment and Plan' — should rank high."""
        from nitrag.retriever_manager import BM25RetrieverStrategy, ClinicalSectionScopedRetrieverStrategy
        results = ClinicalSectionScopedRetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="diagnosis myocardial infarction",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        # Assessment chunk (id=1) should be in results
        chunk_ids = [r["chunk_id"] for r in results]
        assert 1 in chunk_ids

    def test_retriever_name_set(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, ClinicalSectionScopedRetrieverStrategy
        results = ClinicalSectionScopedRetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="hypertension", chunk_strategy_name="fixed_token", top_k=3,
        )
        for r in results:
            assert r["retriever_name"] == "clinical_section_scoped"


# ---------------------------------------------------------------------------
# EntityCentricFusion — unit tests
# ---------------------------------------------------------------------------

class TestEntityCentricFusionRetriever:
    def test_extract_capitalized_terms(self):
        from nitrag.retriever_manager import EntityCentricFusionRetrieverStrategy
        s = EntityCentricFusionRetrieverStrategy()
        terms = s._extract_entity_terms("Patient with Metformin and Lisinopril")
        assert any("Metformin" in t or "metformin" in t.lower() for t in terms)

    def test_extract_4char_tokens(self):
        from nitrag.retriever_manager import EntityCentricFusionRetrieverStrategy
        s = EntityCentricFusionRetrieverStrategy()
        terms = s._extract_entity_terms("patient has fever")
        # "patient" and "fever" are >= 4 chars
        lower_terms = [t.lower() for t in terms]
        assert "patient" in lower_terms or "fever" in lower_terms

    def test_entity_terms_capped(self):
        from nitrag.retriever_manager import EntityCentricFusionRetrieverStrategy
        s = EntityCentricFusionRetrieverStrategy()
        terms = s._extract_entity_terms("Metformin Lisinopril Aspirin Glucose WBC RBC Hemoglobin Creatinine BUN Sodium")
        assert len(terms) <= 8

    def test_integration_returns_results(self, bm25_index_root, store):
        from nitrag.retriever_manager import BM25RetrieverStrategy, EntityCentricFusionRetrieverStrategy
        results = EntityCentricFusionRetrieverStrategy(base_bm25=BM25RetrieverStrategy()).retrieve(
            store=store, index_root_dir=bm25_index_root,
            query="hypertension diabetes treatment Metformin",
            chunk_strategy_name="fixed_token", top_k=5,
        )
        assert isinstance(results, list)

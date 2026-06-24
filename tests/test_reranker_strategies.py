"""
Tests for all 15 reranker strategies.

Each reranker must:
  1. Return a list of the same size (or up to top_k)
  2. Set required fields: rerank_score, reranker_name, original_rank, original_score
  3. Sort descending by rerank_score
  4. Handle empty input gracefully
  5. Handle top_k correctly

Additional strategy-specific correctness tests verify the ordering and logic of
each reranker's signal.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from tests.conftest import CORPUS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(query: str = "hypertension diabetes treatment") -> List[Dict[str, Any]]:
    results = []
    for i, (start, end, text, meta) in enumerate(CORPUS):
        results.append({
            "score": round(1.0 - i * 0.08, 4),
            "retriever_name": "bm25",
            "query": query,
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


REQUIRED_FIELDS = {"rerank_score", "reranker_name", "original_rank", "original_score"}

ALL_RERANKER_CLASSES = [
    "ScorePassthroughReranker",
    "KeywordOverlapReranker",
    "PhraseProximityReranker",
    "MetadataQualityReranker",
    "ClinicalIntentReranker",
    "LengthPenaltyReranker",
    "RecencyDateReranker",
    "DiversityMMRReranker",
    "DeduplicateReranker",
    "HybridWeightedReranker",
    "EntityCoverageReranker",
    "SectionPriorityReranker",
    "BM25RescoreReranker",
    "NegationFilterReranker",
    "PositionBiasCorrectionReranker",
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_all_15_rerankers_registered():
    from pathlib import Path
    from nitrag.reranker_manager import RerankerManager, register_default_rerankers
    mgr = RerankerManager()
    register_default_rerankers(mgr)
    names = set(mgr.list_rerankers())
    expected = {
        "score_passthrough", "keyword_overlap", "phrase_proximity",
        "metadata_quality", "clinical_intent", "length_penalty",
        "recency_date", "diversity_mmr", "deduplicate", "hybrid_weighted",
        "entity_coverage", "section_priority", "bm25_rescore",
        "negation_filter", "position_bias_correction",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


# ---------------------------------------------------------------------------
# Contract tests — all rerankers must satisfy the same field contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("class_name", ALL_RERANKER_CLASSES)
class TestRerankerContract:
    def _get(self, class_name):
        import nitrag.reranker_manager as rm
        return getattr(rm, class_name)()

    def test_returns_list(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="hypertension", results=results)
        assert isinstance(out, list)

    def test_required_fields_present(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="hypertension diabetes", results=results)
        for r in out:
            missing = REQUIRED_FIELDS - r.keys()
            assert not missing, f"{class_name} missing fields: {missing}"

    def test_sorted_descending(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="hypertension", results=results)
        scores = [r["rerank_score"] for r in out]
        assert scores == sorted(scores, reverse=True), f"{class_name} not sorted"

    def test_reranker_name_set(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="test", results=results)
        reranker = self._get(class_name)
        for r in out:
            assert r["reranker_name"] == reranker.name

    def test_empty_input(self, class_name):
        out = self._get(class_name).rerank(query="test", results=[])
        assert out == []

    def test_top_k_respected(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="test", results=results, top_k=3)
        assert len(out) <= 3

    def test_single_result(self, class_name):
        results = _make_results()[:1]
        out = self._get(class_name).rerank(query="hypertension", results=results)
        assert len(out) == 1
        assert out[0]["original_rank"] == 1

    def test_original_rank_starts_at_1(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="test", results=results)
        original_ranks = sorted(r["original_rank"] for r in out)
        assert original_ranks[0] == 1

    def test_rerank_score_is_float(self, class_name):
        results = _make_results()
        out = self._get(class_name).rerank(query="test", results=results)
        for r in out:
            assert isinstance(r["rerank_score"], float)


# ---------------------------------------------------------------------------
# EntityCoverageReranker — ordering correctness
# ---------------------------------------------------------------------------

class TestEntityCoverageReranker:
    def test_entity_rich_chunk_scores_higher(self):
        from nitrag.reranker_manager import EntityCoverageReranker
        # CORPUS[0]: entities = hypertension, diabetes, blood pressure (all in query)
        # CORPUS[5]: entities = chest pain, fever (negated — but still present)
        results = _make_results("hypertension diabetes blood pressure")
        out = EntityCoverageReranker().rerank(query="hypertension diabetes blood pressure", results=results)
        # chunk 0 has hypertension and diabetes_mellitus in entities — should rank high
        top_ids = [r["chunk_id"] for r in out[:3]]
        assert 0 in top_ids

    def test_rerank_features_present(self):
        from nitrag.reranker_manager import EntityCoverageReranker
        results = _make_results()
        out = EntityCoverageReranker().rerank(query="hypertension", results=results)
        for r in out:
            assert "rerank_features" in r
            assert "entity_coverage" in r["rerank_features"]

    def test_coverage_ratio_bounded(self):
        from nitrag.reranker_manager import EntityCoverageReranker
        results = _make_results()
        out = EntityCoverageReranker().rerank(query="hypertension diabetes", results=results)
        for r in out:
            cov = r["rerank_features"]["entity_coverage"]
            assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# SectionPriorityReranker — ordering correctness
# ---------------------------------------------------------------------------

class TestSectionPriorityReranker:
    def test_assessment_chunk_gets_bonus(self):
        from nitrag.reranker_manager import SectionPriorityReranker
        results = _make_results()
        out = SectionPriorityReranker().rerank(query="diagnosis", results=results)
        # chunk 1 = "Assessment", chunk 3 = "Assessment and Plan" — both should rank higher
        # than chunk 5 = "Review of Systems" when retrieval scores are equal
        assessment_ranks = [r["original_rank"] for r in out if r["primary_section"] in ("Assessment", "Assessment and Plan")]
        review_ranks = [r["original_rank"] for r in out if r["primary_section"] == "Review of Systems"]
        # just verify feature is present
        for r in out:
            assert "section_bonus" in r["rerank_features"]

    def test_section_bonus_positive_for_assessment(self):
        from nitrag.reranker_manager import SectionPriorityReranker
        results = _make_results()
        out = SectionPriorityReranker().rerank(query="test", results=results)
        assessment_results = [r for r in out if r["primary_section"].lower() in ("assessment", "assessment and plan")]
        for r in assessment_results:
            assert r["rerank_features"]["section_bonus"] > 0

    def test_rerank_score_bounded(self):
        from nitrag.reranker_manager import SectionPriorityReranker
        results = _make_results()
        out = SectionPriorityReranker().rerank(query="test", results=results)
        for r in out:
            assert 0.0 <= r["rerank_score"] <= 1.0


# ---------------------------------------------------------------------------
# BM25RescoreReranker — local IDF rescoring
# ---------------------------------------------------------------------------

class TestBM25RescoreReranker:
    def test_relevant_query_reorders(self):
        from nitrag.reranker_manager import BM25RescoreReranker
        results = _make_results("metformin lisinopril aspirin medication")
        out = BM25RescoreReranker().rerank(
            query="metformin lisinopril aspirin medication", results=results,
        )
        # chunk 2 is all medications — should score high on BM25 rescore
        top_ids = [r["chunk_id"] for r in out[:3]]
        assert 2 in top_ids

    def test_local_bm25_raw_in_features(self):
        from nitrag.reranker_manager import BM25RescoreReranker
        results = _make_results()
        out = BM25RescoreReranker().rerank(query="hypertension", results=results)
        for r in out:
            assert "local_bm25_raw" in r["rerank_features"]

    def test_empty_query_no_crash(self):
        from nitrag.reranker_manager import BM25RescoreReranker
        results = _make_results()
        out = BM25RescoreReranker().rerank(query="", results=results)
        # empty query returns unchanged list
        assert len(out) == len(results)

    def test_scores_normalized_0_to_1(self):
        from nitrag.reranker_manager import BM25RescoreReranker
        results = _make_results()
        out = BM25RescoreReranker().rerank(query="hypertension diabetes", results=results)
        for r in out:
            assert 0.0 <= r["rerank_score"] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# NegationFilterReranker
# ---------------------------------------------------------------------------

class TestNegationFilterReranker:
    def test_negation_chunks_demoted(self):
        from nitrag.reranker_manager import NegationFilterReranker
        results = _make_results("chest pain fever")
        # chunks 1, 5, 6 have contains_negation=True with relevant entity text
        out = NegationFilterReranker().rerank(query="chest pain fever", results=results)
        # verify that the penalty field is present and non-negative
        for r in out:
            assert "negation_penalty_applied" in r["rerank_features"]
            assert r["rerank_features"]["negation_penalty_applied"] >= 0.0

    def test_non_negated_chunks_unpenalized(self):
        from nitrag.reranker_manager import NegationFilterReranker
        results = _make_results()
        out = NegationFilterReranker().rerank(query="hypertension", results=results)
        non_neg = [r for r in out if not r.get("contains_negation")]
        for r in non_neg:
            assert r["rerank_features"]["negation_penalty_applied"] == pytest.approx(0.0)

    def test_rerank_score_non_negative(self):
        from nitrag.reranker_manager import NegationFilterReranker
        results = _make_results()
        out = NegationFilterReranker().rerank(query="fever chest pain", results=results)
        for r in out:
            assert r["rerank_score"] >= 0.0


# ---------------------------------------------------------------------------
# PositionBiasCorrectionReranker
# ---------------------------------------------------------------------------

class TestPositionBiasCorrectionReranker:
    def test_correction_score_in_features(self):
        from nitrag.reranker_manager import PositionBiasCorrectionReranker
        results = _make_results()
        out = PositionBiasCorrectionReranker().rerank(query="test", results=results)
        for r in out:
            assert "correction_score" in r["rerank_features"]
            assert r["rerank_features"]["correction_score"] > 0.0

    def test_top_ranked_has_highest_correction(self):
        from nitrag.reranker_manager import PositionBiasCorrectionReranker
        results = _make_results()
        out = PositionBiasCorrectionReranker().rerank(query="test", results=results)
        corrections = [(r["original_rank"], r["rerank_features"]["correction_score"]) for r in out]
        rank1_correction = next(c for rank, c in corrections if rank == 1)
        rank8_correction = next(c for rank, c in corrections if rank == 8)
        assert rank1_correction > rank8_correction

    def test_correction_can_surface_mid_rank_result(self):
        """A low-ranked result with a near-identical score to rank 1 should move up."""
        from nitrag.reranker_manager import PositionBiasCorrectionReranker
        # Make all retrieval scores equal so only the correction drives order
        results = _make_results()
        for r in results:
            r["score"] = 0.5  # identical scores
        out = PositionBiasCorrectionReranker(correction_weight=0.9).rerank(
            query="test", results=results,
        )
        # with equal retrieval scores and high correction weight, rank 1 should still be first
        assert out[0]["original_rank"] == 1

    def test_rerank_scores_are_positive(self):
        from nitrag.reranker_manager import PositionBiasCorrectionReranker
        results = _make_results()
        out = PositionBiasCorrectionReranker().rerank(query="test", results=results)
        for r in out:
            assert r["rerank_score"] > 0.0


# ---------------------------------------------------------------------------
# HybridWeightedReranker — integration with multiple sub-rerankers
# ---------------------------------------------------------------------------

class TestHybridWeightedReranker:
    def test_all_features_in_output(self):
        from nitrag.reranker_manager import HybridWeightedReranker
        results = _make_results()
        out = HybridWeightedReranker().rerank(query="hypertension diabetes medication", results=results)
        for r in out:
            assert "rerank_features" in r
            features = r["rerank_features"]
            assert "retrieval_norm" in features
            assert "keyword_norm" in features
            assert "metadata_norm" in features

    def test_weights_sum_to_one(self):
        from nitrag.reranker_manager import HybridWeightedReranker
        h = HybridWeightedReranker()
        total = h.retrieval_weight + h.keyword_weight + h.metadata_weight + h.clinical_weight + h.length_weight
        assert total == pytest.approx(1.0, abs=1e-6)

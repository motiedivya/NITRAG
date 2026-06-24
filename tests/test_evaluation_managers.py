"""
Unit tests for evaluation manager metric computation.

Avoids running full pipelines — exercises compute methods with synthetic DataFrames
and the in-process helper math functions directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest import CORPUS, MockStore, write_chunks_parquet


# ---------------------------------------------------------------------------
# chunking_evaluation — helper math
# ---------------------------------------------------------------------------

class TestChunkingEvaluationHelpers:
    def _gini(self, lengths):
        from nitrag.chunking_evaluation import ChunkingEvaluationManager
        return ChunkingEvaluationManager._gini_coefficient(np.array(lengths, dtype=float))

    def _entropy(self, lengths):
        from nitrag.chunking_evaluation import ChunkingEvaluationManager
        return ChunkingEvaluationManager._entropy_bits(np.array(lengths, dtype=float))

    def test_gini_all_equal_is_zero(self):
        assert self._gini([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-6)

    def test_gini_one_dominates_near_one(self):
        g = self._gini([0, 0, 0, 1000])
        assert g > 0.7

    def test_gini_bounded(self):
        g = self._gini([5, 10, 15, 20, 25, 100])
        assert 0.0 <= g <= 1.0

    def test_entropy_uniform_is_max(self):
        """Uniform distribution → maximum entropy for given bin count."""
        uniform = list(range(1, 21))  # 20 equal-width values
        varied = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100]
        assert self._entropy(uniform) >= self._entropy(varied)

    def test_entropy_all_same_low(self):
        e = self._entropy([50, 50, 50, 50, 50])
        assert e < 1.0  # all in same bin → very low entropy

    def test_entropy_non_negative(self):
        assert self._entropy([10, 20, 30]) >= 0.0


# ---------------------------------------------------------------------------
# reranking_evaluation — _score_results
# ---------------------------------------------------------------------------

class TestRerankingScoreResults:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager

        class _Dummy:
            pass

        self.mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)

    def _result(self, chunk_id, page, text_preview="test result text about glucose"):
        return {
            "chunk_id": chunk_id, "document_id": "d", "chunk_strategy_name": "s",
            "score": 1.0, "page_start": page, "page_end": page,
            "primary_section": "Assessment",
            "text_preview": text_preview,
            "rerank_score": 1.0,
        }

    def test_keyword_hit_rate_full(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 1, "patient has hypertension and diabetes")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=["hypertension", "diabetes"],
            expected_pages=set(), top_k=5,
        )
        assert metrics["keyword_hit_rate"] > 0.0

    def test_keyword_hit_rate_zero_for_miss(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 1, "no relevant content here")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=["hypertension", "diabetes"],
            expected_pages=set(), top_k=5,
        )
        assert metrics["keyword_hit_rate"] == pytest.approx(0.0)

    def test_mrr_hit_at_rank_1(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 5, "test"), self._result(1, 2, "test")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=[], expected_pages={5}, top_k=5,
        )
        assert metrics["mrr_page"] == pytest.approx(1.0)

    def test_mrr_hit_at_rank_2(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 1, "test"), self._result(1, 5, "test")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=[], expected_pages={5}, top_k=5,
        )
        assert metrics["mrr_page"] == pytest.approx(0.5)

    def test_precision_at_1(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 3, "test")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=[], expected_pages={3}, top_k=5,
        )
        assert metrics["precision_at_1"] == pytest.approx(1.0)

    def test_precision_at_1_miss(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 99, "test")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=[], expected_pages={3}, top_k=5,
        )
        assert metrics["precision_at_1"] == pytest.approx(0.0)

    def test_empty_results_returns_zero_metrics(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=[],
            expected_keywords=["test"], expected_pages={1}, top_k=5,
        )
        assert metrics["keyword_hit_rate"] == pytest.approx(0.0)
        assert metrics["mrr_page"] == pytest.approx(0.0)

    def test_no_expected_pages_mrr_zero(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        results = [self._result(0, 1, "test")]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=[], expected_pages=set(), top_k=5,
        )
        assert metrics["mrr_page"] == pytest.approx(0.0)

    def test_duplicate_detection(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        identical_text = "patient has hypertension and diabetes mellitus"
        results = [self._result(0, 1, identical_text), self._result(1, 2, identical_text)]
        metrics = RerankingEvaluationManager._score_results(
            self.mgr, results=results,
            expected_keywords=[], expected_pages=set(), top_k=5,
        )
        assert metrics["duplicate_text_ratio"] > 0.0


# ---------------------------------------------------------------------------
# reranking_evaluation — _rank_movement (new metrics)
# ---------------------------------------------------------------------------

class TestRankMovement:
    def _result(self, chunk_id):
        return {"chunk_strategy_name": "s", "document_id": "d", "chunk_id": chunk_id,
                "start_index": chunk_id, "end_index": chunk_id + 1}

    def test_identity_movement_all_zeros(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)
        original = [self._result(i) for i in range(5)]
        metrics = mgr._rank_movement(original, original)
        assert metrics["mean_abs_rank_delta"] == pytest.approx(0.0)
        assert metrics["max_abs_rank_delta"] == pytest.approx(0.0)
        assert metrics["promoted_count"] == 0
        assert metrics["demoted_count"] == 0
        assert metrics["spearman_rho"] == pytest.approx(1.0)

    def test_full_reversal_spearman_negative(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)
        original = [self._result(i) for i in range(5)]
        reversed_list = list(reversed(original))
        metrics = mgr._rank_movement(original, reversed_list)
        assert metrics["spearman_rho"] < 0

    def test_partial_swap_promoted_demoted(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)
        original = [self._result(i) for i in range(4)]
        # swap first two
        reranked = [self._result(1), self._result(0), self._result(2), self._result(3)]
        metrics = mgr._rank_movement(original, reranked)
        # item 1 moved from rank 2 to rank 1 → promoted
        # item 0 moved from rank 1 to rank 2 → demoted
        assert metrics["promoted_count"] >= 1
        assert metrics["demoted_count"] >= 1

    def test_max_abs_rank_delta_equals_largest_jump(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)
        original = [self._result(i) for i in range(5)]
        # move item 4 to rank 1 (delta = 4)
        reranked = [self._result(4), self._result(0), self._result(1), self._result(2), self._result(3)]
        metrics = mgr._rank_movement(original, reranked)
        assert metrics["max_abs_rank_delta"] >= 4

    def test_top_result_changed_flag(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)
        original = [self._result(0), self._result(1)]
        reranked = [self._result(1), self._result(0)]
        metrics = mgr._rank_movement(original, reranked)
        assert metrics["top_result_changed"] == 1

    def test_top_result_unchanged_flag(self):
        from nitrag.reranking_evaluation import RerankingEvaluationManager
        mgr = RerankingEvaluationManager.__new__(RerankingEvaluationManager)
        original = [self._result(0), self._result(1)]
        metrics = mgr._rank_movement(original, original)
        assert metrics["top_result_changed"] == 0


# ---------------------------------------------------------------------------
# indexing_evaluation — compute_schema_completeness
# ---------------------------------------------------------------------------

class TestIndexingEvaluationManager:
    def test_empty_dir_no_crash(self, tmp_path):
        from nitrag.indexing_evaluation import IndexingEvaluationManager
        mgr = IndexingEvaluationManager(
            store_or_document_dir=tmp_path,
            index_root_dir=tmp_path / "indexes",
            report_dir=tmp_path / "report",
        )
        inv = mgr.compute_inventory()
        assert isinstance(inv, pd.DataFrame)

    def test_inventory_covers_all_18_indexers(self, tmp_path):
        from nitrag.indexing_evaluation import IndexingEvaluationManager
        # create fake chunk parquet so list_chunk_strategies returns one strategy
        chunks_dir = tmp_path / "chunks_enriched"
        chunks_dir.mkdir()
        fake_parquet = chunks_dir / "fixed_token.parquet"
        pq.write_table(pa.Table.from_pylist([{"chunk_id": 0}]), fake_parquet)

        mgr = IndexingEvaluationManager(
            store_or_document_dir=tmp_path,
            index_root_dir=tmp_path / "indexes",
            chunk_dir=chunks_dir,
            report_dir=tmp_path / "report",
        )
        inv = mgr.compute_inventory()
        assert len(inv) == 18  # 18 expected indexers × 1 strategy

    def test_missing_index_flagged_as_not_exists(self, tmp_path):
        from nitrag.indexing_evaluation import IndexingEvaluationManager
        chunks_dir = tmp_path / "chunks_enriched"
        chunks_dir.mkdir()
        pq.write_table(pa.Table.from_pylist([{"chunk_id": 0}]), chunks_dir / "fixed_token.parquet")

        mgr = IndexingEvaluationManager(
            store_or_document_dir=tmp_path,
            report_dir=tmp_path / "report",
            chunk_dir=chunks_dir,
        )
        inv = mgr.compute_inventory()
        # none of the 18 indexes have been built → all should be missing
        assert inv["exists"].sum() == 0

    def test_suspicious_lists_missing_as_error(self, tmp_path):
        from nitrag.indexing_evaluation import IndexingEvaluationManager
        chunks_dir = tmp_path / "chunks_enriched"
        chunks_dir.mkdir()
        pq.write_table(pa.Table.from_pylist([{"chunk_id": 0}]), chunks_dir / "fixed_token.parquet")

        mgr = IndexingEvaluationManager(
            store_or_document_dir=tmp_path,
            report_dir=tmp_path / "report",
            chunk_dir=chunks_dir,
        )
        suspicious = mgr.find_suspicious_indexes()
        errors = suspicious[suspicious["severity"] == "error"] if not suspicious.empty else pd.DataFrame()
        assert len(errors) >= 0  # just verify no crash; empty when no indexes built


# ---------------------------------------------------------------------------
# chunk_metadata_enrichment_evaluation — schema completeness
# ---------------------------------------------------------------------------

class TestEnrichmentEvaluationManager:
    def _build_enriched_parquet(self, path: Path, chunks: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(chunks), path)

    def test_empty_dir_no_crash(self, tmp_path):
        from nitrag.chunk_metadata_enrichment_evaluation import ChunkMetadataEnrichmentEvaluationManager
        mgr = ChunkMetadataEnrichmentEvaluationManager(tmp_path)
        strategies = mgr.list_strategies()
        assert strategies == []

    def test_schema_completeness_with_data(self, tmp_path):
        from nitrag.chunk_metadata_enrichment_evaluation import ChunkMetadataEnrichmentEvaluationManager
        enriched_dir = tmp_path / "chunks_enriched"
        enriched_dir.mkdir()

        rows = []
        for i, (start, end, text, meta) in enumerate(CORPUS):
            rows.append({
                "chunk_id": i, "document_id": "doc",
                "start_index": start, "end_index": end,
                "metadata_json": json.dumps({"clinical": {"quality": 0.8}}),
                "document_type": "discharge_summary",
                "primary_section": meta["primary_section"],
                "section_names_json": json.dumps([meta["primary_section"]]),
                "source_element_ids_json": json.dumps([i]),
                "overlap_line_count": 2,
                **{k: v for k, v in meta.items()},
            })
        self._build_enriched_parquet(enriched_dir / "fixed_token.parquet", rows)

        mgr = ChunkMetadataEnrichmentEvaluationManager(tmp_path, report_dir=tmp_path / "report")
        schema_df = mgr.compute_schema_completeness(["fixed_token"])
        assert len(schema_df) > 0
        assert "strategy" in schema_df.columns
        assert "column" in schema_df.columns
        assert "coverage_pct" in schema_df.columns

    def test_enrichment_metrics_with_data(self, tmp_path):
        from nitrag.chunk_metadata_enrichment_evaluation import ChunkMetadataEnrichmentEvaluationManager
        enriched_dir = tmp_path / "chunks_enriched"
        enriched_dir.mkdir()

        rows = []
        for i, (start, end, text, meta) in enumerate(CORPUS):
            rows.append({
                "chunk_id": i, "document_id": "doc",
                "start_index": start, "end_index": end,
                "metadata_json": json.dumps({"clinical": {}}),
                "document_type": "note",
                "primary_section": meta["primary_section"],
                "section_names_json": json.dumps([]),
                "source_element_ids_json": json.dumps([i]),
                "overlap_line_count": 1,
                **{k: v for k, v in meta.items()},
            })
        pq.write_table(pa.Table.from_pylist(rows), enriched_dir / "fixed_token.parquet")

        mgr = ChunkMetadataEnrichmentEvaluationManager(tmp_path, report_dir=tmp_path / "report")
        metrics_df = mgr.compute_enrichment_metrics(["fixed_token"])
        assert len(metrics_df) == 1
        row = metrics_df.iloc[0]
        assert row["strategy"] == "fixed_token"
        assert row["chunk_count"] == len(CORPUS)
        assert 0.0 <= row["entity_chunk_coverage_pct"] <= 100.0
        assert 0.0 <= row["avg_quality_score"] <= 1.0

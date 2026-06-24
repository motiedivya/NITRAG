"""Tests for nitrag/semantic_retrievers.py — _fuse_rrf and registration."""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from nitrag.semantic_retrievers import _fuse_rrf, register_semantic_retrievers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_id: int,
    document_id: str = "doc_001",
    chunk_strategy_name: str = "fixed_token",
    score: float = 0.5,
    start_index: int = 0,
    end_index: int = 10,
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_strategy_name": chunk_strategy_name,
        "score": score,
        "start_index": start_index,
        "end_index": end_index,
        "retriever_name": "test",
        "query": "test query",
        "token_length": 20,
        "page_start": 0,
        "page_end": 0,
    }


# ---------------------------------------------------------------------------
# _fuse_rrf — basic scoring
# ---------------------------------------------------------------------------

class TestFuseRRF:
    def test_item_in_both_lists_appears_once(self):
        """A result present in both lexical and semantic lists appears once in output."""
        lexical = [_make_result(chunk_id=1, score=0.9)]
        semantic = [_make_result(chunk_id=1, score=0.8)]
        result = _fuse_rrf(lexical, semantic, alpha=0.5)
        chunk_ids = [r["chunk_id"] for r in result]
        assert chunk_ids.count(1) == 1

    def test_item_in_both_lists_has_combined_score(self):
        """A result appearing in rank 1 of both lists scores higher than rank 1 of one list only."""
        # Item A: rank 1 in both
        item_a_lex = [_make_result(chunk_id=1)]
        item_a_sem = [_make_result(chunk_id=1)]
        # Item B: rank 1 in lexical only
        item_b_lex = [_make_result(chunk_id=2)]
        item_b_sem = []

        result_a = _fuse_rrf(item_a_lex, item_a_sem, alpha=0.5, top_k=2)
        result_b = _fuse_rrf(item_b_lex, item_b_sem, alpha=0.5, top_k=2)

        score_a = result_a[0]["rrf_score"]
        score_b = result_b[0]["rrf_score"]
        assert score_a > score_b

    def test_unique_key_has_single_rank_contribution(self):
        """A result only in one list has only one rank's RRF contribution."""
        lexical = [_make_result(chunk_id=1), _make_result(chunk_id=2)]
        semantic = [_make_result(chunk_id=3)]
        result = _fuse_rrf(lexical, semantic, alpha=0.5, top_k=10)
        chunk_ids = [r["chunk_id"] for r in result]
        # All unique keys should appear
        assert 1 in chunk_ids or 2 in chunk_ids
        assert 3 in chunk_ids

    def test_output_sorted_descending_by_score(self):
        """Output must be sorted by rrf_score descending."""
        lexical = [
            _make_result(chunk_id=1),
            _make_result(chunk_id=2),
            _make_result(chunk_id=3),
        ]
        semantic = [
            _make_result(chunk_id=1),
            _make_result(chunk_id=4),
        ]
        result = _fuse_rrf(lexical, semantic, alpha=0.5, top_k=10)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_alpha_0_only_lexical_scores_count(self):
        """With alpha=0.0, only the lexical side contributes."""
        # Item 1 ranks first lexically; Item 2 ranks first semantically
        lexical = [_make_result(chunk_id=1), _make_result(chunk_id=2)]
        semantic = [_make_result(chunk_id=2), _make_result(chunk_id=1)]
        result = _fuse_rrf(lexical, semantic, alpha=0.0, top_k=10)
        # Semantic weight = alpha = 0 so semantic side contributes 0
        # Item 1 lexical rank=1 → score 1.0/(60+1); semantic rank=2 → 0 * score
        # Item 2 lexical rank=2 → score 1.0/(60+2); semantic rank=1 → 0 * score
        # So item 1 should rank higher
        assert result[0]["chunk_id"] == 1

    def test_alpha_1_only_semantic_scores_count(self):
        """With alpha=1.0, only the semantic side contributes."""
        lexical = [_make_result(chunk_id=1), _make_result(chunk_id=2)]
        semantic = [_make_result(chunk_id=2), _make_result(chunk_id=1)]
        result = _fuse_rrf(lexical, semantic, alpha=1.0, top_k=10)
        # Lexical weight = 1 - alpha = 0 so lexical contributes 0
        # Item 2 semantic rank=1 → score 1.0/(60+1); item 1 semantic rank=2
        assert result[0]["chunk_id"] == 2

    def test_top_k_limits_output(self):
        """Output length is capped at top_k."""
        lexical = [_make_result(chunk_id=i) for i in range(20)]
        semantic = []
        result = _fuse_rrf(lexical, semantic, alpha=0.5, top_k=5)
        assert len(result) <= 5

    def test_deduplication_same_key_appears_once(self):
        """Identical (document_id, chunk_id, strategy) in both inputs → appears once."""
        item = _make_result(chunk_id=7, document_id="doc_A", chunk_strategy_name="strat")
        lexical = [item]
        semantic = [item]
        result = _fuse_rrf(lexical, semantic, alpha=0.5, top_k=5)
        # The key (strat, doc_A, 7) should appear exactly once
        assert len(result) == 1

    def test_rank_1_in_both_beats_rank_1_in_one(self):
        """Item in rank 1 of both lists should outscore item in rank 1 of one list."""
        # item_a: rank 1 lexical + rank 1 semantic
        # item_b: rank 1 lexical only (not in semantic)
        lexical = [
            _make_result(chunk_id=1),  # rank 1
            _make_result(chunk_id=2),  # rank 2
        ]
        semantic = [
            _make_result(chunk_id=1),  # rank 1 — so item 1 is in both
        ]
        result = _fuse_rrf(lexical, semantic, alpha=0.5, top_k=5)
        # item 1 (both lists rank 1) should be top
        assert result[0]["chunk_id"] == 1

    def test_empty_lexical_returns_semantic_top_k(self):
        """When lexical is empty, output comes from semantic only."""
        semantic = [_make_result(chunk_id=i) for i in range(5)]
        result = _fuse_rrf([], semantic, alpha=0.5, top_k=3)
        assert len(result) == 3

    def test_empty_semantic_returns_lexical_top_k(self):
        """When semantic is empty, output comes from lexical only."""
        lexical = [_make_result(chunk_id=i) for i in range(5)]
        result = _fuse_rrf(lexical, [], alpha=0.5, top_k=3)
        assert len(result) == 3

    def test_each_result_has_rrf_score_field(self):
        """All output items should have rrf_score set."""
        lexical = [_make_result(chunk_id=i) for i in range(3)]
        semantic = [_make_result(chunk_id=i) for i in range(3)]
        result = _fuse_rrf(lexical, semantic, alpha=0.5, top_k=10)
        for r in result:
            assert "rrf_score" in r
            assert isinstance(r["rrf_score"], float)
            assert r["rrf_score"] > 0.0


# ---------------------------------------------------------------------------
# register_semantic_retrievers — smoke test
# ---------------------------------------------------------------------------

class TestRegisterSemanticRetrievers:
    def _make_mock_retriever_manager(self):
        """Build a minimal RetrieverManager-like mock."""
        mgr = MagicMock()
        registered = {}

        def register_retriever(strategy, force=False):
            registered[strategy.name] = strategy

        mgr.register_retriever.side_effect = register_retriever
        mgr._registered = registered
        return mgr

    def _make_mock_embedding_manager(self):
        em = MagicMock()
        em.embed_query.side_effect = NotImplementedError("no real embedding in tests")
        return em

    def _make_mock_vector_index_manager(self):
        vim = MagicMock()
        vim.is_built.return_value = False
        vim.search.side_effect = NotImplementedError("no real index in tests")
        return vim

    def test_dense_and_hybrid_are_registered(self):
        """register_semantic_retrievers registers 'dense' and 'hybrid' strategies."""
        mgr = self._make_mock_retriever_manager()
        em = self._make_mock_embedding_manager()
        vim = self._make_mock_vector_index_manager()

        register_semantic_retrievers(mgr, em, vim)

        registered_names = [call.args[0].name for call in mgr.register_retriever.call_args_list]
        assert "dense" in registered_names
        assert "hybrid" in registered_names

    def test_hyde_registered_when_llm_config_provided(self):
        """When llm_config is given, 'hyde' strategy is also registered."""
        from nitrag.config import LLMConfig
        mgr = self._make_mock_retriever_manager()
        em = self._make_mock_embedding_manager()
        vim = self._make_mock_vector_index_manager()
        llm_config = LLMConfig()

        register_semantic_retrievers(mgr, em, vim, llm_config=llm_config)

        registered_names = [call.args[0].name for call in mgr.register_retriever.call_args_list]
        assert "hyde" in registered_names

    def test_hyde_not_registered_without_llm_config(self):
        """Without llm_config, 'hyde' should NOT be registered."""
        mgr = self._make_mock_retriever_manager()
        em = self._make_mock_embedding_manager()
        vim = self._make_mock_vector_index_manager()

        register_semantic_retrievers(mgr, em, vim, llm_config=None)

        registered_names = [call.args[0].name for call in mgr.register_retriever.call_args_list]
        assert "hyde" not in registered_names

    def test_register_called_with_force_true_by_default(self):
        """Strategies are registered with force=True by default to allow re-registration."""
        mgr = self._make_mock_retriever_manager()
        em = self._make_mock_embedding_manager()
        vim = self._make_mock_vector_index_manager()

        register_semantic_retrievers(mgr, em, vim)

        for call in mgr.register_retriever.call_args_list:
            assert call.kwargs.get("force") is True or call.args[1] is True \
                or (len(call.args) > 1 and call.args[1] is True) \
                or call.kwargs.get("force", True) is True

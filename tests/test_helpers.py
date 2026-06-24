"""
Unit tests for pure helper functions across all three managers.
No I/O, no store, no parquet — fast and deterministic.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# simple_tokenize (same implementation in all three managers)
# ---------------------------------------------------------------------------

def tokenize(text: str):
    import re
    return re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())


class TestSimpleTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert tokenize("blood pressure: 145/90") == ["blood", "pressure", "145", "90"]

    def test_empty(self):
        assert tokenize("") == []

    def test_none(self):
        assert tokenize(None) == []

    def test_medical_term(self):
        tokens = tokenize("HbA1c 8.4% WBC 12.5 k/uL")
        assert "hba1c" in tokens
        assert "8" in tokens
        assert "wbc" in tokens

    def test_preserves_alphanumeric(self):
        assert tokenize("ekg12abc") == ["ekg12abc"]


# ---------------------------------------------------------------------------
# passes_filters (retriever_manager)
# ---------------------------------------------------------------------------

class TestPassesFilters:
    def _pf(self, row, filters):
        from nitrag.retriever_manager import passes_filters
        return passes_filters(row, filters)

    def test_no_filters(self):
        assert self._pf({"a": 1}, None) is True
        assert self._pf({"a": 1}, {}) is True

    def test_eq_match(self):
        assert self._pf({"section": "Assessment"}, {"section": {"$eq": "Assessment"}})

    def test_eq_no_match(self):
        assert not self._pf({"section": "Plan"}, {"section": {"$eq": "Assessment"}})

    def test_ne(self):
        assert self._pf({"section": "Plan"}, {"section": {"$ne": "Assessment"}})
        assert not self._pf({"section": "Assessment"}, {"section": {"$ne": "Assessment"}})

    def test_in(self):
        assert self._pf({"section": "Assessment"}, {"section": {"$in": ["Assessment", "Plan"]}})
        assert not self._pf({"section": "Radiology"}, {"section": {"$in": ["Assessment", "Plan"]}})

    def test_gte(self):
        assert self._pf({"score": 0.8}, {"score": {"$gte": 0.5}})
        assert not self._pf({"score": 0.3}, {"score": {"$gte": 0.5}})

    def test_lte(self):
        assert self._pf({"score": 0.4}, {"score": {"$lte": 0.5}})
        assert not self._pf({"score": 0.6}, {"score": {"$lte": 0.5}})

    def test_contains(self):
        assert self._pf({"section": "Assessment and Plan"}, {"section": {"$contains": "Assessment"}})
        assert not self._pf({"section": "Radiology"}, {"section": {"$contains": "Assessment"}})

    def test_plain_equality(self):
        assert self._pf({"flag": True}, {"flag": True})
        assert not self._pf({"flag": False}, {"flag": True})

    def test_list_value(self):
        assert self._pf({"section": "Plan"}, {"section": ["Plan", "Assessment"]})
        assert not self._pf({"section": "Radiology"}, {"section": ["Plan", "Assessment"]})

    def test_missing_key(self):
        assert not self._pf({}, {"section": {"$eq": "Assessment"}})


# ---------------------------------------------------------------------------
# normalize_scores (reranker_manager)
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def _ns(self, results, score_key="score"):
        from nitrag.reranker_manager import normalize_scores
        return normalize_scores(results, score_key)

    def _r(self, chunk_id, score):
        return {"chunk_strategy_name": "s", "document_id": "d", "chunk_id": chunk_id, "score": score}

    def test_basic_normalization(self):
        results = [self._r(0, 0.0), self._r(1, 0.5), self._r(2, 1.0)]
        norms = self._ns(results)
        assert norms[("s", "d", 0)] == pytest.approx(0.0)
        assert norms[("s", "d", 1)] == pytest.approx(0.5)
        assert norms[("s", "d", 2)] == pytest.approx(1.0)

    def test_all_same_nonzero(self):
        results = [self._r(0, 0.7), self._r(1, 0.7)]
        norms = self._ns(results)
        assert norms[("s", "d", 0)] == pytest.approx(1.0)
        assert norms[("s", "d", 1)] == pytest.approx(1.0)

    def test_all_zero(self):
        results = [self._r(0, 0.0), self._r(1, 0.0)]
        norms = self._ns(results)
        assert norms[("s", "d", 0)] == pytest.approx(0.0)

    def test_empty(self):
        assert self._ns([]) == {}

    def test_custom_score_key(self):
        results = [{"chunk_strategy_name": "s", "document_id": "d", "chunk_id": 0, "rerank_score": 2.0},
                   {"chunk_strategy_name": "s", "document_id": "d", "chunk_id": 1, "rerank_score": 4.0}]
        norms = self._ns(results, score_key="rerank_score")
        assert norms[("s", "d", 1)] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion (retriever_manager)
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def _rrf(self, ranked_lists, top_k=10, k=60):
        from nitrag.retriever_manager import reciprocal_rank_fusion
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k, k=k)

    def _r(self, chunk_id, score):
        return {
            "chunk_strategy_name": "s", "document_id": "d",
            "chunk_id": chunk_id, "score": score,
            "start_index": chunk_id, "end_index": chunk_id + 1,
        }

    def test_single_list_preserves_order(self):
        ranked = [self._r(0, 1.0), self._r(1, 0.8), self._r(2, 0.5)]
        result = self._rrf([ranked])
        ids = [r["chunk_id"] for r in result]
        assert ids == [0, 1, 2]

    def test_two_lists_boost_shared(self):
        list_a = [self._r(0, 1.0), self._r(1, 0.8), self._r(2, 0.5)]
        list_b = [self._r(2, 1.0), self._r(0, 0.7), self._r(3, 0.5)]
        result = self._rrf([list_a, list_b])
        # chunk 0 and 2 both appear in both lists — should rank above chunk 3 (only list_b)
        ids = [r["chunk_id"] for r in result]
        assert ids.index(3) > ids.index(0)
        assert ids.index(3) > ids.index(2)

    def test_top_k_respected(self):
        ranked = [self._r(i, 1.0 - i * 0.1) for i in range(10)]
        result = self._rrf([ranked], top_k=3)
        assert len(result) == 3

    def test_empty_lists_skipped(self):
        ranked = [self._r(0, 1.0)]
        result = self._rrf([ranked, []])
        assert len(result) == 1

    def test_result_has_required_fields(self):
        ranked = [self._r(0, 1.0)]
        result = self._rrf([ranked])
        assert "score" in result[0]
        assert "retriever_name" in result[0]


# ---------------------------------------------------------------------------
# NumericRangeRetriever — _parse_conditions, _satisfies_all
# ---------------------------------------------------------------------------

class TestNumericRangeParser:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from nitrag.retriever_manager import NumericRangeRetrieverStrategy
        self.strategy = NumericRangeRetrieverStrategy()

    def test_greater_than_word(self):
        conds = self.strategy._parse_conditions("glucose greater than 200")
        ops = {c["op"]: c["value"] for c in conds}
        assert "gt" in ops
        assert ops["gt"] == pytest.approx(200.0)

    def test_less_than_symbol(self):
        conds = self.strategy._parse_conditions("HR < 60")
        ops = {c["op"]: c["value"] for c in conds}
        assert "lt" in ops
        assert ops["lt"] == pytest.approx(60.0)

    def test_gte_symbol(self):
        conds = self.strategy._parse_conditions("HbA1c >= 7.0")
        ops = {c["op"]: c["value"] for c in conds}
        assert "gte" in ops
        assert ops["gte"] == pytest.approx(7.0)

    def test_lte_symbol(self):
        conds = self.strategy._parse_conditions("BP <= 120")
        ops = {c["op"]: c["value"] for c in conds}
        assert "lte" in ops
        assert ops["lte"] == pytest.approx(120.0)

    def test_between(self):
        conds = self.strategy._parse_conditions("glucose between 70 and 120")
        ops = {c["op"]: c["value"] for c in conds}
        assert "gte" in ops and "lte" in ops
        assert ops["gte"] == pytest.approx(70.0)
        assert ops["lte"] == pytest.approx(120.0)

    def test_approx(self):
        conds = self.strategy._parse_conditions("around 150 mg/dL")
        assert len(conds) == 2
        assert all(c["op"] in ("gte", "lte") for c in conds)
        # 15% margin → 127.5 to 172.5
        lows = [c["value"] for c in conds if c["op"] == "gte"]
        highs = [c["value"] for c in conds if c["op"] == "lte"]
        assert lows[0] < 150.0 < highs[0]

    def test_no_condition_returns_empty(self):
        assert self.strategy._parse_conditions("show me all patients") == []

    def test_satisfies_all_gt(self):
        assert self.strategy._satisfies_all(210.0, [{"op": "gt", "value": 200.0}])
        assert not self.strategy._satisfies_all(190.0, [{"op": "gt", "value": 200.0}])

    def test_satisfies_all_combined(self):
        conds = [{"op": "gte", "value": 70.0}, {"op": "lte", "value": 120.0}]
        assert self.strategy._satisfies_all(95.0, conds)
        assert not self.strategy._satisfies_all(130.0, conds)
        assert not self.strategy._satisfies_all(60.0, conds)


# ---------------------------------------------------------------------------
# QueryExpansionBM25 — _expand_query
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from nitrag.retriever_manager import QueryExpansionBM25RetrieverStrategy
        self.strategy = QueryExpansionBM25RetrieverStrategy()

    def test_known_abbreviation_expands(self):
        variants = self.strategy._expand_query("patient with HTN and DM")
        assert len(variants) > 1
        combined = " ".join(variants)
        assert "hypertension" in combined.lower()

    def test_unknown_term_no_expansion(self):
        variants = self.strategy._expand_query("patient has a headache")
        assert variants[0] == "patient has a headache"

    def test_mi_expands(self):
        variants = self.strategy._expand_query("rule out MI")
        combined = " ".join(variants)
        assert "myocardial infarction" in combined.lower() or "heart attack" in combined.lower()

    def test_max_variants_capped(self):
        # even a query with many abbreviations stays under cap
        variants = self.strategy._expand_query("MI HTN DM COPD CHF UTI")
        assert len(variants) <= 5

    def test_original_always_first(self):
        q = "patient with HTN"
        variants = self.strategy._expand_query(q)
        assert variants[0] == q


# ---------------------------------------------------------------------------
# NegationAwareBM25 — _compute_negation_penalty
# ---------------------------------------------------------------------------

class TestNegationPenalty:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from nitrag.retriever_manager import NegationAwareBM25RetrieverStrategy, safe_json_loads
        self.strategy = NegationAwareBM25RetrieverStrategy()
        self.safe_json_loads = safe_json_loads

    def _result(self, contains_negation, entities_json_negated_text=None):
        import json
        entities = []
        if entities_json_negated_text:
            for t in entities_json_negated_text:
                entities.append({"text": t, "negated": True, "type": "diagnosis_or_problem_candidate"})
        return {
            "contains_negation": contains_negation,
            "entities": entities,
            "text_preview": "no fever no chest pain",
        }

    def test_no_negation_flag_zero_penalty(self):
        result = self._result(False, ["fever"])
        penalty = self.strategy._compute_negation_penalty(result, {"fever"})
        assert penalty == pytest.approx(0.0)

    def test_negation_no_overlap_zero_penalty(self):
        result = self._result(True, ["fever"])
        penalty = self.strategy._compute_negation_penalty(result, {"diabetes"})
        assert penalty == pytest.approx(0.0)

    def test_negation_full_overlap_full_penalty(self):
        result = self._result(True, ["fever"])
        penalty = self.strategy._compute_negation_penalty(result, {"fever"})
        assert penalty > 0.0
        assert penalty <= 1.0

    def test_negation_partial_overlap(self):
        result = self._result(True, ["fever"])
        # query has 2 terms, 1 negated → penalty = 1/2
        penalty = self.strategy._compute_negation_penalty(result, {"fever", "diabetes"})
        assert penalty == pytest.approx(0.5)

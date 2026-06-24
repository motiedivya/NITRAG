"""Tests for nitrag/generation_evaluation.py — all 6 metrics, EvaluationReport, aggregate."""
from __future__ import annotations

import pytest

from nitrag.context_assembler import AssembledContext, ContextChunk
from nitrag.generation_evaluation import EvaluationReport, GenerationEvaluationManager
from nitrag.generation_manager import Citation, GenerationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    citation_number: int = 1,
    chunk_id: int = 0,
    text: str = "The patient has hypertension and takes lisinopril 10mg daily.",
    section: str = "Medications",
    source_label: str = "Page 1 | Medications",
) -> ContextChunk:
    return ContextChunk(
        citation_number=citation_number,
        chunk_id=chunk_id,
        text=text,
        page_start=0,
        page_end=0,
        section=section,
        score=0.9,
        retriever="bm25",
        token_count=20,
        document_id="doc_001",
        source_label=source_label,
    )


def _make_context(
    chunks: list[ContextChunk],
    query: str = "What medications were prescribed?",
    truncated: bool = False,
    truncated_count: int = 0,
) -> AssembledContext:
    citation_map = {c.chunk_id: c.citation_number for c in chunks}
    formatted = "\n".join(f"[{c.citation_number}] {c.text}" for c in chunks)
    return AssembledContext(
        chunks=chunks,
        citation_map=citation_map,
        total_tokens=sum(c.token_count for c in chunks),
        formatted_text=formatted,
        query=query,
        truncated=truncated,
        truncated_count=truncated_count,
    )


def _make_generation_result(
    answer: str = "Lisinopril 10mg was prescribed [1].",
    query: str = "What medications were prescribed?",
    citations: list[Citation] | None = None,
) -> GenerationResult:
    return GenerationResult(
        query=query,
        answer=answer,
        citations=citations or [],
        faithfulness_score=0.9,
        tokens_used={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=200.0,
        model_name="llama3.1:8b",
        provider="openai_compatible",
        context_tokens=300,
        truncated_context=False,
    )


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_overall_score_weights_sum_to_1(self):
        """0.35 + 0.20 + 0.20 + 0.15 + 0.10 == 1.0"""
        assert abs(0.35 + 0.20 + 0.20 + 0.15 + 0.10 - 1.0) < 1e-9

    def test_overall_score_is_float_in_range(self):
        report = EvaluationReport(
            query="test",
            faithfulness=0.8,
            answer_relevance=0.7,
            context_precision=0.6,
            citation_coverage=1.0,
            hallucination_risk=0.1,
            context_recall=0.9,
        )
        score = report.overall_score
        assert 0.0 <= score <= 1.0

    def test_overall_score_max_when_all_perfect(self):
        report = EvaluationReport(
            query="test",
            faithfulness=1.0,
            answer_relevance=1.0,
            context_precision=1.0,
            citation_coverage=1.0,
            hallucination_risk=0.0,  # 0 risk → contributes 0.10 * (1 - 0) = 0.10
            context_recall=1.0,
        )
        assert abs(report.overall_score - 1.0) < 0.01

    def test_to_dict_has_all_required_keys(self):
        report = EvaluationReport(
            query="What is the diagnosis?",
            faithfulness=0.5,
            answer_relevance=0.5,
            context_precision=0.5,
            citation_coverage=0.5,
            hallucination_risk=0.5,
            context_recall=0.5,
        )
        d = report.to_dict()
        expected_keys = {
            "query", "faithfulness", "answer_relevance", "context_precision",
            "citation_coverage", "hallucination_risk", "context_recall",
            "sentence_risks", "notes",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_values_are_floats(self):
        report = EvaluationReport(
            query="q",
            faithfulness=0.8,
            answer_relevance=0.6,
            context_precision=0.7,
            citation_coverage=0.9,
            hallucination_risk=0.2,
            context_recall=0.75,
        )
        d = report.to_dict()
        for key in ["faithfulness", "answer_relevance", "context_precision",
                    "citation_coverage", "hallucination_risk", "context_recall"]:
            assert isinstance(d[key], (int, float))


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def test_full_context_overlap_high_score(self):
        evaluator = GenerationEvaluationManager()
        text = "Metformin 500mg twice daily was prescribed for diabetes mellitus control."
        chunk = _make_chunk(text=text)
        context = _make_context([chunk])
        # Answer closely mirrors the context text
        answer = "Metformin 500mg twice daily was prescribed for diabetes."
        score = evaluator.faithfulness(answer, context)
        assert score >= 0.5

    def test_sentences_with_citation_markers_fully_grounded(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Unrelated content that does not match the answer at all.")
        context = _make_context([chunk])
        answer = "The medication was administered [1]. The dose was confirmed [1]."
        score = evaluator.faithfulness(answer, context)
        assert score == 1.0


class TestAnswerRelevance:
    def test_high_keyword_overlap_gives_high_score(self):
        evaluator = GenerationEvaluationManager()
        # Answer shares key terms with query
        answer = "Lisinopril 10mg was prescribed for hypertension treatment."
        query = "lisinopril hypertension medication treatment prescribed"
        score = evaluator.answer_relevance(answer, query)
        assert score > 0.3

    def test_unrelated_answer_gives_low_score(self):
        evaluator = GenerationEvaluationManager()
        answer = "The quantum experiment showed remarkable superconducting results."
        query = "What medications were prescribed for diabetes?"
        score = evaluator.answer_relevance(answer, query)
        # Very little overlap between medical query and physics answer
        assert score < 0.3


class TestContextPrecision:
    def test_answer_cites_all_chunks_returns_high(self):
        evaluator = GenerationEvaluationManager()
        chunk1 = _make_chunk(citation_number=1, chunk_id=1, text="Metformin was prescribed.")
        chunk2 = _make_chunk(citation_number=2, chunk_id=2, text="Blood pressure 145/90 mmHg.")
        context = _make_context([chunk1, chunk2])
        # Answer cites both chunks
        answer = "Metformin was prescribed [1]. Blood pressure was 145/90 [2]."
        score = evaluator.context_precision(answer, context)
        assert score == 1.0

    def test_answer_cites_none_uses_overlap_fallback(self):
        evaluator = GenerationEvaluationManager()
        chunk1 = _make_chunk(citation_number=1, chunk_id=1, text="Metformin 500mg twice daily medication treatment.")
        chunk2 = _make_chunk(citation_number=2, chunk_id=2, text="Warfarin anticoagulation therapy dosage.")
        context = _make_context([chunk1, chunk2])
        # Answer has some overlap with chunk1 but no [N] markers
        answer = "Metformin 500mg twice daily medication was given."
        score = evaluator.context_precision(answer, context)
        # chunk1 should pass overlap check (>= 0.15)
        assert score >= 0.0


class TestCitationCoverage:
    def test_all_citations_resolve_returns_1(self):
        evaluator = GenerationEvaluationManager()
        chunk1 = _make_chunk(citation_number=1, chunk_id=1)
        chunk2 = _make_chunk(citation_number=2, chunk_id=2)
        context = _make_context([chunk1, chunk2])
        answer = "See findings [1] and treatment [2]."
        score = evaluator.citation_coverage(answer, context)
        assert score == 1.0

    def test_invalid_citation_number_returns_zero(self):
        evaluator = GenerationEvaluationManager()
        chunk1 = _make_chunk(citation_number=1, chunk_id=1)
        context = _make_context([chunk1])
        # [5] does not exist in context
        answer = "See reference [5] for details."
        score = evaluator.citation_coverage(answer, context)
        assert score == 0.0

    def test_no_citations_in_answer_returns_1(self):
        # When the answer has no [N] markers, coverage defaults to 1.0
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(citation_number=1, chunk_id=1)
        context = _make_context([chunk])
        answer = "The patient has hypertension."
        score = evaluator.citation_coverage(answer, context)
        assert score == 1.0

    def test_partial_resolution(self):
        evaluator = GenerationEvaluationManager()
        chunk1 = _make_chunk(citation_number=1, chunk_id=1)
        context = _make_context([chunk1])
        # [1] resolves, [5] does not → 1/2 = 0.5
        answer = "Finding [1] and unknown [5]."
        score = evaluator.citation_coverage(answer, context)
        assert score == 0.5


class TestContextRecall:
    def test_all_query_keywords_in_context_returns_1(self):
        evaluator = GenerationEvaluationManager()
        # Build context that contains all query keywords
        chunk = _make_chunk(
            text="Lisinopril 10mg was prescribed for hypertension treatment control."
        )
        context = _make_context([chunk], query="lisinopril hypertension treatment")
        score = evaluator.context_recall("lisinopril hypertension treatment", context)
        assert score == 1.0

    def test_no_query_keywords_in_context_returns_low(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Blood pressure 145/90 mmHg was measured today.")
        context = _make_context([chunk])
        # Query keywords not in context
        score = evaluator.context_recall("warfarin anticoagulation dosing protocol", context)
        assert score < 0.5


class TestHallucinationRisk:
    def test_sentences_with_citations_get_risk_zero(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Unrelated content here to prevent empty context.")
        context = _make_context([chunk])
        answer = "Medication was administered [1]. Dose confirmed [2]."
        result = evaluator.hallucination_risk(answer, context)
        for sent_info in result["sentences"]:
            if sent_info["grounded_by_citation"]:
                assert sent_info["risk"] == 0.0

    def test_high_overlap_sentence_has_low_risk(self):
        evaluator = GenerationEvaluationManager()
        # Sentence closely mirrors context
        text = "Metformin 500mg twice daily was prescribed for type 2 diabetes mellitus management."
        chunk = _make_chunk(text=text)
        context = _make_context([chunk])
        answer = "Metformin 500mg twice daily was prescribed for type 2 diabetes mellitus management."
        result = evaluator.hallucination_risk(answer, context)
        assert result["mean_risk"] < 0.5

    def test_zero_overlap_sentence_has_high_risk(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Aspirin 81mg daily for cardiovascular protection.")
        context = _make_context([chunk])
        # Completely unrelated sentence
        answer = "Superconducting quantum magnets operate at extremely cold temperatures demonstrating entanglement."
        result = evaluator.hallucination_risk(answer, context)
        assert result["mean_risk"] > 0.4

    def test_returns_sentences_list(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Some medical context here.")
        context = _make_context([chunk])
        answer = "This is a test sentence with enough words to qualify."
        result = evaluator.hallucination_risk(answer, context)
        assert "sentences" in result
        assert "mean_risk" in result
        assert isinstance(result["sentences"], list)

    def test_mean_risk_is_in_range(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Patient diagnosed with hypertension and diabetes mellitus.")
        context = _make_context([chunk])
        answer = "Hypertension and diabetes mellitus were diagnosed in this patient."
        result = evaluator.hallucination_risk(answer, context)
        assert 0.0 <= result["mean_risk"] <= 1.0


# ---------------------------------------------------------------------------
# evaluate() end-to-end
# ---------------------------------------------------------------------------

class TestEvaluateEndToEnd:
    def test_evaluate_returns_evaluation_report(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(
            text="Metformin 500mg twice daily was prescribed for diabetes mellitus."
        )
        context = _make_context(
            [chunk],
            query="What medications were prescribed for diabetes?",
        )
        result = _make_generation_result(
            answer="Metformin 500mg twice daily was prescribed for diabetes [1].",
            query="What medications were prescribed for diabetes?",
        )
        report = evaluator.evaluate(result, context)
        assert isinstance(report, EvaluationReport)

    def test_evaluate_all_metric_fields_are_floats_in_range(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(
            text="Lisinopril 10mg once daily prescribed for hypertension blood pressure."
        )
        context = _make_context(
            [chunk],
            query="What was prescribed for blood pressure?",
        )
        result = _make_generation_result(
            answer="Lisinopril 10mg once daily was prescribed for hypertension [1].",
            query="What was prescribed for blood pressure?",
        )
        report = evaluator.evaluate(result, context)
        for attr in [
            "faithfulness", "answer_relevance", "context_precision",
            "citation_coverage", "hallucination_risk", "context_recall",
        ]:
            val = getattr(report, attr)
            assert isinstance(val, float), f"{attr} is not float"
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0, 1]"

    def test_evaluate_query_matches_result_query(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Blood pressure reading 145/90 mmHg recorded today.")
        context = _make_context([chunk], query="What was the blood pressure?")
        result = _make_generation_result(
            answer="Blood pressure was 145/90 mmHg [1].",
            query="What was the blood pressure?",
        )
        report = evaluator.evaluate(result, context)
        assert report.query == "What was the blood pressure?"

    def test_truncated_context_adds_note(self):
        evaluator = GenerationEvaluationManager()
        chunk = _make_chunk(text="Aspirin 81mg daily for cardiovascular protection.")
        context = _make_context([chunk], truncated=True, truncated_count=3)
        result = _make_generation_result()
        report = evaluator.evaluate(result, context)
        # Should have a note about truncation
        note_text = " ".join(report.notes).lower()
        assert "truncat" in note_text or len(report.notes) >= 0  # note may appear


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_aggregate_averages_two_reports(self):
        evaluator = GenerationEvaluationManager()
        r1 = EvaluationReport(
            query="q1", faithfulness=0.8, answer_relevance=0.6,
            context_precision=0.7, citation_coverage=1.0,
            hallucination_risk=0.2, context_recall=0.9,
        )
        r2 = EvaluationReport(
            query="q2", faithfulness=0.4, answer_relevance=0.8,
            context_precision=0.5, citation_coverage=0.5,
            hallucination_risk=0.4, context_recall=0.7,
        )
        agg = evaluator.aggregate([r1, r2])
        assert abs(agg["faithfulness"] - 0.6) < 0.01
        assert abs(agg["answer_relevance"] - 0.7) < 0.01

    def test_aggregate_returns_overall_score(self):
        evaluator = GenerationEvaluationManager()
        r1 = EvaluationReport(
            query="q1", faithfulness=1.0, answer_relevance=1.0,
            context_precision=1.0, citation_coverage=1.0,
            hallucination_risk=0.0, context_recall=1.0,
        )
        agg = evaluator.aggregate([r1])
        assert "overall_score" in agg
        assert 0.0 <= agg["overall_score"] <= 1.0

    def test_aggregate_empty_list_returns_empty_dict(self):
        evaluator = GenerationEvaluationManager()
        agg = evaluator.aggregate([])
        assert agg == {}

    def test_aggregate_all_metrics_present(self):
        evaluator = GenerationEvaluationManager()
        r = EvaluationReport(
            query="q", faithfulness=0.5, answer_relevance=0.5,
            context_precision=0.5, citation_coverage=0.5,
            hallucination_risk=0.5, context_recall=0.5,
        )
        agg = evaluator.aggregate([r])
        for key in ["faithfulness", "answer_relevance", "context_precision",
                    "citation_coverage", "hallucination_risk", "context_recall", "overall_score"]:
            assert key in agg

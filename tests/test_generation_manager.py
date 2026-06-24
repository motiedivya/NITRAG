"""Tests for nitrag/generation_manager.py — pure functions only, no LLM calls."""
from __future__ import annotations

from nitrag.context_assembler import AssembledContext, ContextChunk
from nitrag.generation_manager import (
    Citation,
    GenerationResult,
    MEDICAL_SYSTEM_PROMPT,
    _best_supporting_quote,
    compute_faithfulness,
    extract_citation_numbers,
    resolve_citations,
)


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


def _make_context(chunks: list[ContextChunk], query: str = "What medications?") -> AssembledContext:
    citation_map = {c.chunk_id: c.citation_number for c in chunks}
    formatted = "\n".join(f"[{c.citation_number}] {c.text}" for c in chunks)
    return AssembledContext(
        chunks=chunks,
        citation_map=citation_map,
        total_tokens=sum(c.token_count for c in chunks),
        formatted_text=formatted,
        query=query,
    )


# ---------------------------------------------------------------------------
# extract_citation_numbers
# ---------------------------------------------------------------------------

class TestExtractCitationNumbers:
    def test_extracts_multiple_citations(self):
        result = extract_citation_numbers("[1] foo [2] bar [3]")
        assert result == [1, 2, 3]

    def test_returns_empty_for_no_citations(self):
        result = extract_citation_numbers("no cites here")
        assert result == []

    def test_handles_single_citation(self):
        result = extract_citation_numbers("The patient has HTN [1].")
        assert result == [1]

    def test_handles_adjacent_citations(self):
        result = extract_citation_numbers("Both [1][2] support this claim.")
        assert result == [1, 2]

    def test_handles_high_citation_numbers(self):
        result = extract_citation_numbers("See [10] and [99]")
        assert 10 in result
        assert 99 in result

    def test_empty_string_returns_empty(self):
        assert extract_citation_numbers("") == []

    def test_ignores_non_bracketed_numbers(self):
        result = extract_citation_numbers("Step 1 and step 2 but not a citation")
        assert result == []


# ---------------------------------------------------------------------------
# resolve_citations
# ---------------------------------------------------------------------------

class TestResolveCitations:
    def test_resolves_two_citations(self):
        chunk1 = _make_chunk(citation_number=1, chunk_id=10, text="Metformin 500mg twice daily.")
        chunk2 = _make_chunk(citation_number=2, chunk_id=20, text="Blood pressure 145/90 mmHg.")
        context = _make_context([chunk1, chunk2])

        answer = "The patient takes Metformin [1] and has elevated blood pressure [2]."
        citations = resolve_citations(answer, context)

        assert len(citations) == 2
        numbers = [c.number for c in citations]
        assert 1 in numbers
        assert 2 in numbers

    def test_resolved_citation_has_correct_chunk_id(self):
        chunk = _make_chunk(citation_number=1, chunk_id=42)
        context = _make_context([chunk])
        citations = resolve_citations("Hypertension noted [1].", context)
        assert len(citations) == 1
        assert citations[0].chunk_id == 42

    def test_resolved_citation_has_source_label(self):
        chunk = _make_chunk(citation_number=1, source_label="Page 2 | Assessment")
        context = _make_context([chunk])
        citations = resolve_citations("Finding [1].", context)
        assert citations[0].source_label == "Page 2 | Assessment"

    def test_missing_citation_not_resolved(self):
        # Answer references [5] but context only has chunk #1
        chunk = _make_chunk(citation_number=1, chunk_id=1)
        context = _make_context([chunk])
        citations = resolve_citations("Missing ref [5].", context)
        assert len(citations) == 0

    def test_duplicate_citation_resolved_once(self):
        chunk = _make_chunk(citation_number=1, chunk_id=1)
        context = _make_context([chunk])
        citations = resolve_citations("See [1] and also [1] again.", context)
        assert len(citations) == 1

    def test_sorted_by_citation_number(self):
        chunk2 = _make_chunk(citation_number=2, chunk_id=20)
        chunk1 = _make_chunk(citation_number=1, chunk_id=10)
        context = _make_context([chunk1, chunk2])
        citations = resolve_citations("First [2] then [1].", context)
        assert [c.number for c in citations] == [1, 2]


# ---------------------------------------------------------------------------
# _best_supporting_quote
# ---------------------------------------------------------------------------

class TestBestSupportingQuote:
    def test_returns_most_overlapping_sentence(self):
        # Sentence 2 shares the most tokens with the answer
        chunk_text = (
            "The patient was admitted yesterday. "
            "Lisinopril 10mg was prescribed for blood pressure control. "
            "Follow-up in three months."
        )
        answer = "Lisinopril 10mg was given for blood pressure."
        quote, score = _best_supporting_quote(answer, chunk_text)
        assert "Lisinopril" in quote or "lisinopril" in quote.lower()
        assert score > 0.0

    def test_score_positive_for_overlapping_text(self):
        chunk_text = "Metformin 500mg twice daily for diabetes management."
        answer = "Metformin is used for diabetes."
        _, score = _best_supporting_quote(answer, chunk_text)
        assert score > 0.0

    def test_score_low_for_non_overlapping_text(self):
        chunk_text = "Blood pressure 145/90 mmHg. Heart rate 88 bpm."
        answer = "The patient has no allergies to shellfish or penicillin."
        _, score = _best_supporting_quote(answer, chunk_text)
        # Very little overlap between these two
        assert score < 0.5

    def test_quote_truncated_to_300_chars(self):
        long_chunk = "A " * 200 + ". " + "B " * 200 + "."
        answer = "A " * 10
        quote, _ = _best_supporting_quote(answer, long_chunk)
        assert len(quote) <= 300

    def test_returns_empty_for_short_sentences(self):
        # All sentences < 10 chars — none qualify
        chunk_text = "Hi. Yes. No."
        answer = "Something relevant here."
        quote, score = _best_supporting_quote(answer, chunk_text)
        assert quote == "" or score == 0.0


# ---------------------------------------------------------------------------
# compute_faithfulness
# ---------------------------------------------------------------------------

class TestComputeFaithfulness:
    def test_full_overlap_returns_high_score(self):
        # Answer sentences closely mirror the context
        chunk = _make_chunk(
            text=(
                "Metformin 500mg twice daily prescribed. "
                "Lisinopril 10mg once daily for blood pressure. "
                "Aspirin 81mg daily."
            )
        )
        context = _make_context([chunk])
        answer = (
            "Metformin 500mg twice daily was prescribed. "
            "Lisinopril 10mg once daily for blood pressure control. "
            "Aspirin 81mg daily was given."
        )
        score = compute_faithfulness(answer, context)
        assert score > 0.6

    def test_zero_overlap_returns_low_score(self):
        chunk = _make_chunk(text="Blood pressure 145/90 mmHg measured today.")
        context = _make_context([chunk])
        answer = (
            "The quantum entanglement experiment demonstrated remarkable results. "
            "Superconducting magnets operate below critical temperature."
        )
        score = compute_faithfulness(answer, context)
        assert score < 0.5

    def test_sentences_with_citations_are_grounded(self):
        # Even if there's no lexical overlap, a sentence with [N] counts as supported
        chunk = _make_chunk(text="Unrelated content about other topics entirely.")
        context = _make_context([chunk])
        answer = (
            "The medication was administered [1]. "
            "Another treatment was also prescribed [1]."
        )
        score = compute_faithfulness(answer, context)
        assert score == 1.0

    def test_empty_answer_returns_one(self):
        chunk = _make_chunk()
        context = _make_context([chunk])
        assert compute_faithfulness("", context) == 1.0

    def test_short_answer_below_threshold_ignored(self):
        # Sentences shorter than 20 chars don't count
        chunk = _make_chunk(text="Completely unrelated text.")
        context = _make_context([chunk])
        score = compute_faithfulness("Short.", context)
        assert score == 1.0  # no sentences ≥ 20 chars → default 1.0


# ---------------------------------------------------------------------------
# Citation dataclass
# ---------------------------------------------------------------------------

class TestCitationDataclass:
    def test_construct_and_access_fields(self):
        c = Citation(
            number=3,
            chunk_id=7,
            page_start=1,
            page_end=2,
            section="Assessment",
            quote="The patient has DM2.",
            confidence=0.85,
            source_label="Page 2 | Assessment",
        )
        assert c.number == 3
        assert c.chunk_id == 7
        assert c.page_start == 1
        assert c.page_end == 2
        assert c.section == "Assessment"
        assert "DM2" in c.quote
        assert c.confidence == 0.85
        assert c.source_label == "Page 2 | Assessment"


# ---------------------------------------------------------------------------
# GenerationResult dataclass
# ---------------------------------------------------------------------------

class TestGenerationResultDataclass:
    def test_construct_and_access_fields(self):
        gr = GenerationResult(
            query="What medications were prescribed?",
            answer="Metformin 500mg [1].",
            citations=[],
            faithfulness_score=0.9,
            tokens_used={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=123.4,
            model_name="llama3.1:8b",
            provider="openai_compatible(localhost:11434)",
            context_tokens=300,
            truncated_context=False,
        )
        assert gr.query == "What medications were prescribed?"
        assert "Metformin" in gr.answer
        assert gr.faithfulness_score == 0.9
        assert gr.tokens_used["total_tokens"] == 150
        assert gr.latency_ms == 123.4
        assert gr.model_name == "llama3.1:8b"
        assert gr.context_tokens == 300
        assert gr.truncated_context is False


# ---------------------------------------------------------------------------
# MEDICAL_SYSTEM_PROMPT
# ---------------------------------------------------------------------------

class TestMedicalSystemPrompt:
    def test_contains_citation_keyword(self):
        assert "citation" in MEDICAL_SYSTEM_PROMPT.lower()

    def test_contains_evidence_keyword(self):
        assert "evidence" in MEDICAL_SYSTEM_PROMPT.lower()

    def test_is_non_empty_string(self):
        assert isinstance(MEDICAL_SYSTEM_PROMPT, str)
        assert len(MEDICAL_SYSTEM_PROMPT) > 100

    def test_mentions_citation_format(self):
        # Should reference [N] notation
        assert "[N]" in MEDICAL_SYSTEM_PROMPT or "[1]" in MEDICAL_SYSTEM_PROMPT

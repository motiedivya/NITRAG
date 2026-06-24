"""RAGAS-style generation evaluation for medical RAG.

Metrics — all lexical heuristics, no external LLM judge required
-----------------------------------------------------------------
faithfulness         fraction of answer sentences supported by retrieved context
answer_relevance     keyword overlap between answer and query
context_precision    fraction of retrieved chunks that contributed to the answer
citation_coverage    fraction of [N] references that resolve to real chunks
hallucination_risk   per-sentence risk score (0 = fully grounded, 1 = no support)
context_recall       coverage of query keywords across retrieved chunks (proxy for recall)

All scores are in [0, 1]. Higher is better except hallucination_risk (lower is better).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_assembler import AssembledContext, ContextChunk
from .generation_manager import GenerationResult, _CITATION_RE, compute_faithfulness


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    query: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    citation_coverage: float
    hallucination_risk: float          # mean per-sentence risk
    context_recall: float
    sentence_risks: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "citation_coverage": self.citation_coverage,
            "hallucination_risk": self.hallucination_risk,
            "context_recall": self.context_recall,
            "sentence_risks": self.sentence_risks,
            "notes": self.notes,
        }

    @property
    def overall_score(self) -> float:
        """Weighted composite score."""
        return round(
            0.35 * self.faithfulness
            + 0.20 * self.answer_relevance
            + 0.20 * self.context_precision
            + 0.15 * self.citation_coverage
            + 0.10 * (1.0 - self.hallucination_risk),
            3,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "and", "or", "not", "but", "from", "that",
    "this", "it", "its", "as", "so", "if", "then", "when", "who", "what",
    "which", "there", "their", "they", "he", "she", "we", "you", "i",
    "no", "yes", "also", "about", "after", "before", "between", "into",
    "through", "during", "patient", "patients",
}


def _overlap_score(text_a: str, text_b: str) -> float:
    a, b = _tokenize(text_a), _tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), 1)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


# ─────────────────────────────────────────────────────────────────────────────
# GenerationEvaluationManager
# ─────────────────────────────────────────────────────────────────────────────

class GenerationEvaluationManager:
    """Evaluate the quality of a RAG generation result.

    Usage
    -----
    from nitrag.generation_evaluation import GenerationEvaluationManager

    evaluator = GenerationEvaluationManager()
    report = evaluator.evaluate(generation_result, context)
    print(report.faithfulness, report.hallucination_risk, report.overall_score)
    """

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def evaluate(
        self,
        result: GenerationResult,
        context: AssembledContext,
    ) -> EvaluationReport:
        """Run all metrics and return an EvaluationReport."""
        notes: List[str] = []

        faithfulness = self.faithfulness(result.answer, context)
        relevance = self.answer_relevance(result.answer, result.query)
        precision = self.context_precision(result.answer, context)
        coverage = self.citation_coverage(result.answer, context)
        risk_data = self.hallucination_risk(result.answer, context)
        recall = self.context_recall(result.query, context)

        if faithfulness < 0.5:
            notes.append(f"Low faithfulness ({faithfulness:.2f}): many answer sentences lack evidence support.")
        if risk_data["mean_risk"] > 0.5:
            notes.append("High hallucination risk: multiple sentences have weak grounding.")
        if context.truncated:
            notes.append(f"Context was truncated ({context.truncated_count} chunks dropped due to token budget).")
        if coverage < 0.7:
            notes.append(f"Citation coverage {coverage:.0%}: some [N] references did not resolve to chunks.")

        return EvaluationReport(
            query=result.query,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision,
            citation_coverage=coverage,
            hallucination_risk=risk_data["mean_risk"],
            context_recall=recall,
            sentence_risks=risk_data["sentences"],
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def faithfulness(self, answer: str, context: AssembledContext) -> float:
        """Fraction of answer sentences with ≥ 40% lexical overlap with context."""
        return compute_faithfulness(answer, context)

    def answer_relevance(self, answer: str, query: str) -> float:
        """How well the answer covers query keywords (excluding stopwords)."""
        return _overlap_score(answer, query)

    def context_precision(self, answer: str, context: AssembledContext) -> float:
        """Fraction of retrieved chunks that are referenced in the answer ([N] marker or high overlap)."""
        if not context.chunks:
            return 1.0
        cited_numbers = set(_CITATION_RE.findall(answer))
        answer_tokens = _tokenize(answer)
        used = 0
        for c in context.chunks:
            if str(c.citation_number) in cited_numbers:
                used += 1
                continue
            if _overlap_score(answer, c.text) >= 0.15:
                used += 1
        return round(used / len(context.chunks), 3)

    def citation_coverage(self, answer: str, context: AssembledContext) -> float:
        """Fraction of [N] markers in the answer that map to actual retrieved chunks."""
        cited = set(int(n) for n in _CITATION_RE.findall(answer))
        if not cited:
            return 1.0
        valid_numbers = {c.citation_number for c in context.chunks}
        return round(len(cited & valid_numbers) / len(cited), 3)

    def context_recall(self, query: str, context: AssembledContext) -> float:
        """Fraction of query keywords found across retrieved chunks (proxy for recall)."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 1.0
        context_tokens = _tokenize(" ".join(c.text for c in context.chunks))
        found = query_tokens & context_tokens
        return round(len(found) / len(query_tokens), 3)

    def hallucination_risk(
        self,
        answer: str,
        context: AssembledContext,
    ) -> Dict[str, Any]:
        """Per-sentence hallucination risk score.

        A sentence has high risk if it makes a specific claim (≥4 words) but
        has low lexical overlap with the context.
        Sentences with [N] citation markers are treated as grounded.

        Returns dict with:
            mean_risk  : float [0, 1]
            sentences  : List[{sentence, risk, grounded_by_citation}]
        """
        sentences = _split_sentences(answer)
        if not sentences:
            return {"mean_risk": 0.0, "sentences": []}

        context_text = " ".join(c.text for c in context.chunks)
        results = []
        total_risk = 0.0

        for sent in sentences:
            has_citation = bool(_CITATION_RE.search(sent))
            if has_citation:
                risk = 0.0
            else:
                overlap = _overlap_score(sent, context_text)
                # Risk = 1 - overlap, but cap low-token sentences at 0.3 risk
                sent_tokens = _tokenize(sent)
                if len(sent_tokens) < 4:
                    risk = 0.0
                else:
                    risk = round(max(0.0, 1.0 - overlap * 2.5), 3)   # scale: 40% overlap → 0 risk
            total_risk += risk
            results.append({
                "sentence": sent[:120],
                "risk": risk,
                "grounded_by_citation": has_citation,
            })

        mean_risk = round(total_risk / max(len(sentences), 1), 3)
        return {"mean_risk": mean_risk, "sentences": results}

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        results: List[GenerationResult],
        contexts: List[AssembledContext],
    ) -> List[EvaluationReport]:
        """Evaluate a list of (result, context) pairs."""
        return [self.evaluate(r, c) for r, c in zip(results, contexts)]

    def aggregate(self, reports: List[EvaluationReport]) -> Dict[str, float]:
        """Aggregate multiple evaluation reports into mean scores."""
        if not reports:
            return {}
        metrics = [
            "faithfulness", "answer_relevance", "context_precision",
            "citation_coverage", "hallucination_risk", "context_recall",
        ]
        return {
            m: round(sum(getattr(r, m) for r in reports) / len(reports), 3)
            for m in metrics
        } | {"overall_score": round(sum(r.overall_score for r in reports) / len(reports), 3)}

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


def simple_tokenize(text: str) -> List[str]:
    text = str(text or "").lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def result_key(result: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(result.get("chunk_strategy_name")),
        str(result.get("document_id")),
        int(result.get("chunk_id") or 0),
    )


def normalize_scores(results: List[Dict[str, Any]], score_key: str = "score") -> Dict[Tuple[str, str, int], float]:
    scores = [safe_float(r.get(score_key), 0.0) for r in results]
    if not scores:
        return {}
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return {result_key(r): 1.0 if hi > 0 else 0.0 for r in results}
    return {result_key(r): (safe_float(r.get(score_key), 0.0) - lo) / (hi - lo) for r in results}


def jaccard(a: str, b: str) -> float:
    ta = set(simple_tokenize(a))
    tb = set(simple_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class BaseRerankerStrategy(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        store=None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pass


class ScorePassthroughReranker(BaseRerankerStrategy):
    name = "score_passthrough"
    description = "Keeps retriever ordering, preserving the original score as rerank_score."

    def rerank(self, *, query: str, results: List[Dict[str, Any]], store=None, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        out = []
        for rank, result in enumerate(results, start=1):
            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = r["original_score"]
            out.append(r)
        return out[:top_k] if top_k else out


class KeywordOverlapReranker(BaseRerankerStrategy):
    name = "keyword_overlap"
    description = "Reranks by query-term coverage and frequency in retrieved text."

    def rerank(self, *, query: str, results: List[Dict[str, Any]], store=None, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        q_terms = Counter(simple_tokenize(query))
        if not q_terms:
            return ScorePassthroughReranker().rerank(query=query, results=results, store=store, top_k=top_k)

        out = []
        for rank, result in enumerate(results, start=1):
            text_terms = Counter(simple_tokenize(result.get("text_preview", "")))
            matched = {term: min(q_tf, text_terms.get(term, 0)) for term, q_tf in q_terms.items() if text_terms.get(term, 0)}
            coverage = len(matched) / max(1, len(q_terms))
            frequency = sum(matched.values())
            score = coverage * 5.0 + math.log1p(frequency)

            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {"query_coverage": coverage, "matched_terms": sorted(matched)}
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


class PhraseProximityReranker(BaseRerankerStrategy):
    name = "phrase_proximity"
    description = "Reranks by query term proximity and phrase order in retrieved text."

    def rerank(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        store=None,
        top_k: Optional[int] = None,
        proximity_window: int = 20,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        q_terms = list(dict.fromkeys(simple_tokenize(query)))
        if not q_terms:
            return ScorePassthroughReranker().rerank(query=query, results=results, store=store, top_k=top_k)

        out = []
        for rank, result in enumerate(results, start=1):
            text_terms = simple_tokenize(result.get("text_preview", ""))
            positions = defaultdict(list)
            for pos, term in enumerate(text_terms):
                if term in q_terms:
                    positions[term].append(pos)

            matched_count = len(positions)
            span = self._min_span(list(positions.values()))
            coverage = matched_count / max(1, len(q_terms))
            proximity = 0.0
            if span is not None:
                proximity = 1.0 / max(1.0, span)
                if span <= proximity_window:
                    proximity += 1.0

            phrase_bonus = 1.0 if " ".join(q_terms[: min(3, len(q_terms))]) in " ".join(text_terms) else 0.0
            score = coverage * 4.0 + proximity + phrase_bonus

            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {"matched_terms": sorted(positions), "min_span": span, "phrase_bonus": phrase_bonus}
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out

    def _min_span(self, position_lists: List[List[int]]) -> Optional[int]:
        if len(position_lists) < 2:
            return None
        merged = []
        for term_idx, positions in enumerate(position_lists):
            for pos in positions:
                merged.append((int(pos), term_idx))
        merged.sort()

        counts = defaultdict(int)
        covered = 0
        left = 0
        best = None
        for right, (pos, term_idx) in enumerate(merged):
            if counts[term_idx] == 0:
                covered += 1
            counts[term_idx] += 1
            while covered == len(position_lists) and left <= right:
                left_pos, left_term = merged[left]
                span = pos - left_pos + 1
                best = span if best is None else min(best, span)
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    covered -= 1
                left += 1
        return best


class MetadataQualityReranker(BaseRerankerStrategy):
    name = "metadata_quality"
    description = "Reranks by clinical quality score, section/entity presence, and metadata richness."

    def rerank(self, *, query: str, results: List[Dict[str, Any]], store=None, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        out = []
        for rank, result in enumerate(results, start=1):
            quality = safe_float(result.get("clinical_quality_score"), 0.0)
            section_bonus = 0.15 if result.get("primary_section") else 0.0
            entities = result.get("entities") or []
            entity_bonus = min(0.25, 0.05 * len(entities)) if isinstance(entities, list) else 0.0
            flag_bonus = 0.05 * sum(
                1 for flag in ["contains_medication", "contains_lab", "contains_diagnosis", "contains_vital"]
                if bool(result.get(flag))
            )
            score = quality + section_bonus + entity_bonus + flag_bonus

            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {"quality": quality, "section_bonus": section_bonus, "entity_bonus": entity_bonus, "flag_bonus": flag_bonus}
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


class ClinicalIntentReranker(BaseRerankerStrategy):
    name = "clinical_intent"
    description = "Reranks by lightweight clinical intent detection and matching metadata flags/entities."

    INTENT_RULES = {
        "medication": {
            "terms": {"medication", "medications", "medicine", "drug", "dose", "tablet", "prescription", "rx"},
            "flags": ["contains_medication"],
            "entity_terms": {"medication_candidate", "medication_line_candidate"},
        },
        "diagnosis": {
            "terms": {"diagnosis", "assessment", "impression", "problem", "dx"},
            "flags": ["contains_diagnosis"],
            "entity_terms": {"diagnosis_code_candidate", "diagnosis_or_problem_candidate"},
        },
        "vital": {
            "terms": {"vital", "vitals", "blood", "pressure", "temperature", "pulse"},
            "flags": ["contains_vital"],
            "entity_terms": {"vital"},
        },
        "lab": {
            "terms": {"lab", "labs", "laboratory", "result", "test"},
            "flags": ["contains_lab"],
            "entity_terms": {"lab_result"},
        },
    }

    def rerank(self, *, query: str, results: List[Dict[str, Any]], store=None, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        q_terms = set(simple_tokenize(query))
        intents = [name for name, rule in self.INTENT_RULES.items() if q_terms & rule["terms"]]
        if not intents:
            intents = list(self.INTENT_RULES.keys())

        out = []
        for rank, result in enumerate(results, start=1):
            entity_counts = result.get("entity_type_counts") or {}
            score = 0.0
            matched_intents = []
            for intent in intents:
                rule = self.INTENT_RULES[intent]
                flag_score = sum(1.0 for flag in rule["flags"] if bool(result.get(flag)))
                entity_score = sum(safe_float(entity_counts.get(entity_type), 0.0) for entity_type in rule["entity_terms"]) if isinstance(entity_counts, dict) else 0.0
                text_score = 0.5 if q_terms & set(simple_tokenize(result.get("text_preview", ""))) & rule["terms"] else 0.0
                intent_score = flag_score + min(1.0, entity_score) + text_score
                if intent_score:
                    matched_intents.append(intent)
                score += intent_score

            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {"intents": intents, "matched_intents": matched_intents}
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


class LengthPenaltyReranker(BaseRerankerStrategy):
    name = "length_penalty"
    description = "Penalizes very short or very long chunks while preserving retrieval relevance."

    def rerank(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        store=None,
        top_k: Optional[int] = None,
        target_min: int = 128,
        target_max: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        score_norm = normalize_scores(results)
        out = []
        for rank, result in enumerate(results, start=1):
            length = safe_float(result.get("token_length"), 0.0)
            if target_min <= length <= target_max:
                length_score = 1.0
            elif length < target_min:
                length_score = max(0.0, length / max(1.0, target_min))
            else:
                length_score = max(0.0, target_max / max(1.0, length))

            score = 0.65 * score_norm.get(result_key(result), 0.0) + 0.35 * length_score
            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {"length_score": length_score, "token_length": length}
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


class RecencyDateReranker(BaseRerankerStrategy):
    name = "recency_date"
    description = "Boosts chunks with date-looking text or date entities when the query has temporal intent."

    DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
    TEMPORAL_TERMS = {"date", "recent", "latest", "today", "yesterday", "follow", "followup", "follow-up", "history"}

    def rerank(self, *, query: str, results: List[Dict[str, Any]], store=None, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        q_terms = set(simple_tokenize(query))
        temporal_intent = bool(q_terms & self.TEMPORAL_TERMS or self.DATE_PATTERN.search(query or ""))
        score_norm = normalize_scores(results)

        out = []
        for rank, result in enumerate(results, start=1):
            text = result.get("text_preview", "")
            date_hits = self.DATE_PATTERN.findall(text)
            entities = result.get("entities") or []
            entity_dates = [
                e for e in entities
                if isinstance(e, dict) and (e.get("type") == "date" or e.get("entity_type") == "date")
            ]
            date_score = min(1.0, 0.25 * (len(date_hits) + len(entity_dates)))
            score = score_norm.get(result_key(result), 0.0)
            if temporal_intent:
                score = 0.60 * score + 0.40 * date_score

            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {"temporal_intent": temporal_intent, "date_hit_count": len(date_hits) + len(entity_dates)}
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


class DiversityMMRReranker(BaseRerankerStrategy):
    name = "diversity_mmr"
    description = "Applies maximal marginal relevance to reduce duplicate context."

    def rerank(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        store=None,
        top_k: Optional[int] = None,
        lambda_mult: float = 0.75,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        limit = top_k or len(results)
        score_norm = normalize_scores(results)
        remaining = [dict(r) for r in results]
        selected = []

        while remaining and len(selected) < limit:
            best = None
            best_score = -1e9
            for candidate in remaining:
                relevance = score_norm.get(result_key(candidate), 0.0)
                diversity_penalty = max(
                    (jaccard(candidate.get("text_preview", ""), chosen.get("text_preview", "")) for chosen in selected),
                    default=0.0,
                )
                score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
                if score > best_score:
                    best = candidate
                    best_score = score

            remaining.remove(best)
            best["original_rank"] = results.index(next(r for r in results if result_key(r) == result_key(best))) + 1
            best["original_score"] = safe_float(best.get("score"), 0.0)
            best["reranker_name"] = self.name
            best["rerank_score"] = round(best_score, 6)
            selected.append(best)

        return selected


class DeduplicateReranker(BaseRerankerStrategy):
    name = "deduplicate"
    description = "Removes duplicate chunk ids/spans and near-duplicate text previews."

    def rerank(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        store=None,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.92,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        out = []
        seen_keys = set()

        for rank, result in enumerate(results, start=1):
            span_key = (
                result.get("chunk_strategy_name"),
                result.get("document_id"),
                result.get("chunk_id"),
                result.get("start_index"),
                result.get("end_index"),
            )
            if span_key in seen_keys:
                continue
            if any(jaccard(result.get("text_preview", ""), kept.get("text_preview", "")) >= similarity_threshold for kept in out):
                continue

            seen_keys.add(span_key)
            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = r["original_score"]
            out.append(r)

            if top_k and len(out) >= top_k:
                break

        return out


class HybridWeightedReranker(BaseRerankerStrategy):
    name = "hybrid_weighted"
    description = "Combines retrieval score, keyword overlap, metadata quality, clinical intent, and length quality."

    def __init__(
        self,
        retrieval_weight: float = 0.35,
        keyword_weight: float = 0.25,
        metadata_weight: float = 0.20,
        clinical_weight: float = 0.15,
        length_weight: float = 0.05,
    ):
        self.retrieval_weight = retrieval_weight
        self.keyword_weight = keyword_weight
        self.metadata_weight = metadata_weight
        self.clinical_weight = clinical_weight
        self.length_weight = length_weight
        self.keyword = KeywordOverlapReranker()
        self.metadata = MetadataQualityReranker()
        self.clinical = ClinicalIntentReranker()
        self.length = LengthPenaltyReranker()

    def rerank(self, *, query: str, results: List[Dict[str, Any]], store=None, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        retrieval_scores = normalize_scores(results)
        feature_maps = {}
        for reranker in [self.keyword, self.metadata, self.clinical, self.length]:
            ranked = reranker.rerank(query=query, results=results, store=store, top_k=None, **kwargs)
            feature_maps[reranker.name] = normalize_scores(ranked, score_key="rerank_score")

        out = []
        for rank, result in enumerate(results, start=1):
            key = result_key(result)
            score = (
                self.retrieval_weight * retrieval_scores.get(key, 0.0)
                + self.keyword_weight * feature_maps["keyword_overlap"].get(key, 0.0)
                + self.metadata_weight * feature_maps["metadata_quality"].get(key, 0.0)
                + self.clinical_weight * feature_maps["clinical_intent"].get(key, 0.0)
                + self.length_weight * feature_maps["length_penalty"].get(key, 0.0)
            )
            r = dict(result)
            r["original_rank"] = rank
            r["original_score"] = safe_float(result.get("score"), 0.0)
            r["reranker_name"] = self.name
            r["rerank_score"] = round(score, 6)
            r["rerank_features"] = {
                "retrieval_norm": retrieval_scores.get(key, 0.0),
                "keyword_norm": feature_maps["keyword_overlap"].get(key, 0.0),
                "metadata_norm": feature_maps["metadata_quality"].get(key, 0.0),
                "clinical_norm": feature_maps["clinical_intent"].get(key, 0.0),
                "length_norm": feature_maps["length_penalty"].get(key, 0.0),
            }
            out.append(r)

        out.sort(key=lambda r: r["rerank_score"], reverse=True)
        return out[:top_k] if top_k else out


class RerankerManager:
    def __init__(self, store=None):
        self.store = store
        self._rerankers: Dict[str, BaseRerankerStrategy] = {}
        self._descriptions: Dict[str, str] = {}

    def register_reranker(self, reranker: BaseRerankerStrategy, force: bool = False) -> None:
        if reranker.name in self._rerankers and not force:
            raise ValueError(f"Reranker already registered: {reranker.name}")
        self._rerankers[reranker.name] = reranker
        self._descriptions[reranker.name] = reranker.description

    def list_rerankers(self, with_descriptions: bool = False):
        if with_descriptions:
            return {name: self._descriptions.get(name, "") for name in self._rerankers}
        return list(self._rerankers.keys())

    def rerank(
        self,
        reranker_name: str,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if reranker_name not in self._rerankers:
            raise KeyError(f"Unknown reranker: {reranker_name}")
        return self._rerankers[reranker_name].rerank(
            query=query,
            results=results,
            store=self.store,
            top_k=top_k,
            **kwargs,
        )

    def rerank_pipeline(
        self,
        reranker_names: List[str],
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        current = results
        for name in reranker_names:
            current = self.rerank(name, query, current, top_k=None, **kwargs)
        return current[:top_k] if top_k else current


def register_default_rerankers(manager: RerankerManager) -> None:
    manager.register_reranker(ScorePassthroughReranker())
    manager.register_reranker(KeywordOverlapReranker())
    manager.register_reranker(PhraseProximityReranker())
    manager.register_reranker(MetadataQualityReranker())
    manager.register_reranker(ClinicalIntentReranker())
    manager.register_reranker(LengthPenaltyReranker())
    manager.register_reranker(RecencyDateReranker())
    manager.register_reranker(DiversityMMRReranker())
    manager.register_reranker(DeduplicateReranker())
    manager.register_reranker(HybridWeightedReranker())

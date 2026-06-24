from __future__ import annotations

import re
import json
import math
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import Counter, defaultdict

import pyarrow.parquet as pq


def safe_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def read_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def simple_tokenize(text: str) -> List[str]:
    text = str(text or "").lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


def make_word_ngrams(tokens: List[str], n: int) -> List[str]:
    if n <= 1:
        return tokens
    return [" ".join(tokens[i:i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def make_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    normalized = " ".join(simple_tokenize(text))
    grams = []
    for n in range(n_min, n_max + 1):
        if len(normalized) < n:
            continue
        grams.extend(normalized[i:i + n] for i in range(0, len(normalized) - n + 1))
    return grams


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return float(score) / float(max_score)


def result_key(result: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(result.get("chunk_strategy_name")),
        str(result.get("document_id")),
        int(result.get("chunk_id") or 0),
    )


def passes_filters(row: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        actual = row.get(key)

        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False

            if "$ne" in expected and actual == expected["$ne"]:
                return False

            if "$in" in expected and actual not in expected["$in"]:
                return False

            if "$gte" in expected and (actual is None or actual < expected["$gte"]):
                return False

            if "$lte" in expected and (actual is None or actual > expected["$lte"]):
                return False

            if "$contains" in expected and expected["$contains"] not in str(actual):
                return False

        elif isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False

        else:
            if actual != expected:
                return False

    return True


def make_result(
    *,
    row: Dict[str, Any],
    score: float,
    retriever_name: str,
    query: str,
    store,
    return_text_chars: int = 1200,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start = int(row["start_index"])
    end = int(row["end_index"])
    text = store.decode_span(start, end)

    out = {
        "score": round(float(score), 6),
        "retriever_name": retriever_name,
        "query": query,

        "chunk_strategy_name": row.get("chunk_strategy_name") or row.get("strategy_name"),
        "chunk_id": row.get("chunk_id"),
        "doc_idx": row.get("doc_idx"),
        "document_id": row.get("document_id"),

        "start_index": start,
        "end_index": end,
        "token_length": row.get("token_length"),

        "page_start": row.get("page_start"),
        "page_end": row.get("page_end"),

        "document_type": row.get("document_type"),
        "primary_section": row.get("primary_section"),

        "contains_medication": row.get("contains_medication"),
        "contains_lab": row.get("contains_lab"),
        "contains_diagnosis": row.get("contains_diagnosis"),
        "contains_vital": row.get("contains_vital"),
        "clinical_quality_score": row.get("clinical_quality_score"),

        "entity_type_counts": safe_json_loads(row.get("entity_type_counts_json")),
        "entities": safe_json_loads(row.get("entities_json")),

        "text_preview": text[:return_text_chars],
    }

    if extra:
        out.update(extra)

    return out


class BaseRetrieverStrategy(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pass


class BM25RetrieverStrategy(BaseRetrieverStrategy):
    name = "bm25"
    description = "BM25 retrieval from persisted BM25 index."

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("BM25RetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_term = index["postings_by_term"]
        idf = index["idf"]
        avgdl = index["avgdl"]
        doc_lens = index["doc_lens"]

        q_terms = simple_tokenize(query)
        scores = defaultdict(float)
        matched_terms = defaultdict(list)

        for term in q_terms:
            for doc_idx, tf in postings_by_term.get(term, []):
                dl = doc_lens.get(doc_idx, 0)
                term_idf = idf.get(term, 0.0)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1e-9))
                score = term_idf * numerator / max(denominator, 1e-9)

                scores[doc_idx] += score
                matched_terms[doc_idx].append(term)

        results = []

        for doc_idx, score in scores.items():
            row = docs[doc_idx]

            if not passes_filters(row, filters):
                continue

            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={
                    "matched_terms": sorted(set(matched_terms[doc_idx])),
                },
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "bm25"

        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        vocab = read_parquet(index_dir / "vocab.parquet")

        if not docs:
            raise FileNotFoundError(f"No BM25 docs found: {index_dir}")

        postings_by_term = defaultdict(list)
        for p in postings:
            postings_by_term[p["term"]].append((int(p["doc_idx"]), int(p["tf"])))

        idf = {v["term"]: float(v["idf"]) for v in vocab}

        doc_lens = {}
        for doc_idx, row in enumerate(docs):
            text_preview = row.get("text_preview") or ""
            doc_lens[doc_idx] = max(1, len(simple_tokenize(text_preview)))

        avgdl = sum(doc_lens.values()) / max(1, len(doc_lens))

        loaded = {
            "docs": docs,
            "postings_by_term": postings_by_term,
            "idf": idf,
            "doc_lens": doc_lens,
            "avgdl": avgdl,
        }

        self._cache[key] = loaded
        return loaded


class KeywordExactRetrieverStrategy(BaseRetrieverStrategy):
    name = "keyword_exact"
    description = "Exact keyword retrieval from keyword inverted index."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        require_all_terms: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("KeywordExactRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_term = index["postings_by_term"]

        q_terms = simple_tokenize(query)
        if not q_terms:
            return []

        doc_term_hits = defaultdict(Counter)

        for term in q_terms:
            for doc_idx, tf in postings_by_term.get(term, []):
                doc_term_hits[doc_idx][term] += tf

        results = []

        for doc_idx, hit_counter in doc_term_hits.items():
            if require_all_terms and not all(t in hit_counter for t in q_terms):
                continue

            row = docs[doc_idx]

            if not passes_filters(row, filters):
                continue

            matched = list(hit_counter.keys())
            score = sum(hit_counter.values()) + (len(matched) * 2.0)

            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={
                    "matched_terms": sorted(matched),
                    "term_frequencies": dict(hit_counter),
                },
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "keyword_inverted"

        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")

        if not docs:
            raise FileNotFoundError(f"No keyword docs found: {index_dir}")

        postings_by_term = defaultdict(list)
        for p in postings:
            postings_by_term[p["term"]].append((int(p["doc_idx"]), int(p["tf"])))

        loaded = {
            "docs": docs,
            "postings_by_term": postings_by_term,
        }

        self._cache[key] = loaded
        return loaded


class TFIDFRetrieverStrategy(BaseRetrieverStrategy):
    name = "tfidf"
    description = "Cosine-style retrieval over persisted TF-IDF term weights."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("TFIDFRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_term = index["postings_by_term"]
        idf = index["idf"]

        q_tf = Counter(simple_tokenize(query))
        if not q_tf:
            return []

        max_tf = max(q_tf.values())
        q_weights = {}
        q_norm_sq = 0.0
        for term, freq in q_tf.items():
            weight = (0.5 + 0.5 * freq / max_tf) * idf.get(term, 0.0)
            if weight <= 0:
                continue
            q_weights[term] = weight
            q_norm_sq += weight * weight

        q_norm = math.sqrt(q_norm_sq) or 1.0
        scores = defaultdict(float)
        matched_terms = defaultdict(list)

        for term, q_weight in q_weights.items():
            q_weight = q_weight / q_norm
            for doc_idx, doc_weight in postings_by_term.get(term, []):
                scores[doc_idx] += q_weight * doc_weight
                matched_terms[doc_idx].append(term)

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_terms": sorted(set(matched_terms[doc_idx]))},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "tfidf"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        vocab = read_parquet(index_dir / "vocab.parquet")
        if not docs:
            raise FileNotFoundError(f"No TF-IDF docs found: {index_dir}")

        postings_by_term = defaultdict(list)
        for p in postings:
            postings_by_term[p["term"]].append((int(p["doc_idx"]), safe_float(p.get("tfidf"))))

        loaded = {
            "docs": docs,
            "postings_by_term": postings_by_term,
            "idf": {v["term"]: safe_float(v.get("idf")) for v in vocab},
        }
        self._cache[key] = loaded
        return loaded


class PhraseNgramRetrieverStrategy(BaseRetrieverStrategy):
    name = "phrase_ngram"
    description = "Phrase retrieval over persisted word bigram/trigram indexes."

    def __init__(self, n_values: Optional[List[int]] = None):
        self.n_values = n_values or [2, 3]
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("PhraseNgramRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_phrase = index["postings_by_phrase"]

        tokens = simple_tokenize(query)
        phrases = []
        for n in self.n_values:
            phrases.extend(make_word_ngrams(tokens, n))
        if not phrases:
            return []

        scores = defaultdict(float)
        matched = defaultdict(list)
        for phrase in phrases:
            phrase_weight = 1.0 + phrase.count(" ")
            for doc_idx, tf in postings_by_phrase.get(phrase, []):
                scores[doc_idx] += phrase_weight * tf
                matched[doc_idx].append(phrase)

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_phrases": sorted(set(matched[doc_idx]))},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "phrase_ngram"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No phrase n-gram docs found: {index_dir}")

        postings_by_phrase = defaultdict(list)
        for p in postings:
            postings_by_phrase[p["phrase"]].append((int(p["doc_idx"]), int(p["tf"])))

        loaded = {"docs": docs, "postings_by_phrase": postings_by_phrase}
        self._cache[key] = loaded
        return loaded


class CharacterNgramRetrieverStrategy(BaseRetrieverStrategy):
    name = "char_ngram"
    description = "Character n-gram retrieval for typo-tolerant lexical matching."

    def __init__(self, n_min: int = 3, n_max: int = 5):
        self.n_min = n_min
        self.n_max = n_max
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("CharacterNgramRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_gram = index["postings_by_gram"]

        grams = Counter(make_char_ngrams(query, self.n_min, self.n_max))
        if not grams:
            return []

        scores = defaultdict(float)
        matched_counts = defaultdict(int)
        for gram, qtf in grams.items():
            for doc_idx, tf in postings_by_gram.get(gram, []):
                scores[doc_idx] += min(qtf, tf)
                matched_counts[doc_idx] += 1

        query_gram_count = sum(grams.values())
        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            normalized = score / max(1, query_gram_count)
            results.append(make_result(
                row=row,
                score=normalized,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_ngram_count": matched_counts[doc_idx]},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "char_ngram"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No character n-gram docs found: {index_dir}")

        postings_by_gram = defaultdict(list)
        for p in postings:
            postings_by_gram[p["gram"]].append((int(p["doc_idx"]), int(p["tf"])))

        loaded = {"docs": docs, "postings_by_gram": postings_by_gram}
        self._cache[key] = loaded
        return loaded


class FieldedLexicalRetrieverStrategy(BaseRetrieverStrategy):
    name = "fielded_lexical"
    description = "Weighted field-aware retrieval over body, section, entity, and flag fields."

    DEFAULT_FIELD_WEIGHTS = {
        "body": 1.0,
        "primary_section": 2.5,
        "section_name": 2.0,
        "entity_type": 2.0,
        "entity_text": 2.5,
        "entity_normalized": 2.75,
        "clinical_flag": 1.75,
        "document_type": 0.75,
    }

    def __init__(self, field_weights: Optional[Dict[str, float]] = None):
        self.field_weights = dict(self.DEFAULT_FIELD_WEIGHTS)
        if field_weights:
            self.field_weights.update(field_weights)
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("FieldedLexicalRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_term = index["postings_by_term"]

        q_terms = simple_tokenize(query)
        if not q_terms:
            return []

        scores = defaultdict(float)
        matched_fields = defaultdict(set)
        for term in q_terms:
            for field, doc_idx, tf in postings_by_term.get(term, []):
                scores[doc_idx] += self.field_weights.get(field, 1.0) * tf
                matched_fields[doc_idx].add(field)

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_fields": sorted(matched_fields[doc_idx])},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "fielded_lexical"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No fielded lexical docs found: {index_dir}")

        postings_by_term = defaultdict(list)
        for p in postings:
            postings_by_term[p["term"]].append((p["field"], int(p["doc_idx"]), int(p["tf"])))

        loaded = {"docs": docs, "postings_by_term": postings_by_term}
        self._cache[key] = loaded
        return loaded


class EntityRetrieverStrategy(BaseRetrieverStrategy):
    name = "entity"
    description = "Entity-centric retrieval from entity index postings."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        entity_types: Optional[List[str]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("EntityRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings = index["postings"]

        q_terms = set(simple_tokenize(query))
        wanted_types = {str(t).lower() for t in entity_types or []}

        scores = defaultdict(float)
        matches = defaultdict(list)

        for p in postings:
            entity_type = str(p.get("entity_type") or "")
            entity_text = str(p.get("entity_text") or "")
            normalized = str(p.get("normalized_value") or "")
            haystack_terms = set(simple_tokenize(" ".join([entity_type, entity_text, normalized])))

            if wanted_types and entity_type.lower() not in wanted_types:
                continue
            if q_terms and not (q_terms & haystack_terms):
                continue

            doc_idx = int(p["doc_idx"])
            confidence = safe_float(p.get("confidence"), 0.5)
            score = 1.0 + confidence + len(q_terms & haystack_terms)
            scores[doc_idx] += score
            matches[doc_idx].append({
                "entity_type": entity_type,
                "entity_text": entity_text,
                "normalized_value": normalized,
                "confidence": confidence,
            })

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_entities": matches[doc_idx]},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "entity"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No entity docs found: {index_dir}")

        loaded = {"docs": docs, "postings": postings}
        self._cache[key] = loaded
        return loaded


class SectionPageRetrieverStrategy(BaseRetrieverStrategy):
    name = "section_page"
    description = "Retrieves chunks by section/document/page postings, useful for scoped browsing."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str = "",
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sections: Optional[List[str]] = None,
        pages: Optional[List[int]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("SectionPageRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings = index["postings"]

        q_terms = set(simple_tokenize(query))
        wanted_sections = {str(s).lower() for s in sections or []}
        wanted_pages = {str(int(p)) for p in pages or []}

        scores = defaultdict(float)
        matched = defaultdict(list)

        for p in postings:
            field = str(p.get("field") or "")
            value = str(p.get("value") or "")
            value_terms = set(simple_tokenize(value))

            hit = False
            if wanted_sections and field in {"primary_section", "section_name"} and value.lower() in wanted_sections:
                hit = True
            if wanted_pages and field == "page" and value in wanted_pages:
                hit = True
            if q_terms and (q_terms & value_terms):
                hit = True
            if not q_terms and not wanted_sections and not wanted_pages:
                hit = True
            if not hit:
                continue

            doc_idx = int(p["doc_idx"])
            weight = 2.0 if field in {"primary_section", "section_name"} else 1.0
            scores[doc_idx] += weight
            matched[doc_idx].append({"field": field, "value": value})

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score + safe_float(row.get("clinical_quality_score"), 0.0),
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_scope": matched[doc_idx]},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "section_page"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No section/page docs found: {index_dir}")

        loaded = {"docs": docs, "postings": postings}
        self._cache[key] = loaded
        return loaded


class BooleanSetRetrieverStrategy(BaseRetrieverStrategy):
    name = "boolean_set"
    description = "Boolean AND/OR term retrieval from the boolean_set index."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        require_all_terms: bool = False,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("BooleanSetRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings_by_term = index["postings_by_term"]

        q_terms = list(dict.fromkeys(simple_tokenize(query)))
        if not q_terms:
            return []

        hits = defaultdict(set)
        for term in q_terms:
            for doc_idx in postings_by_term.get(term, []):
                hits[doc_idx].add(term)

        results = []
        for doc_idx, matched in hits.items():
            if require_all_terms and len(matched) < len(q_terms):
                continue

            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue

            score = len(matched) / max(1, len(q_terms))
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={
                    "matched_terms": sorted(matched),
                    "require_all_terms": require_all_terms,
                },
            ))

        results.sort(key=lambda r: (r["score"], safe_float(r.get("clinical_quality_score"))), reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "boolean_set"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No boolean_set docs found: {index_dir}")

        postings_by_term = defaultdict(list)
        for p in postings:
            postings_by_term[p["term"]].append(int(p["doc_idx"]))

        loaded = {"docs": docs, "postings_by_term": postings_by_term}
        self._cache[key] = loaded
        return loaded


class PositionalProximityRetrieverStrategy(BaseRetrieverStrategy):
    name = "positional_proximity"
    description = "Proximity-aware retrieval using term positions from the positional index."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        proximity_window: int = 12,
        require_all_terms: bool = False,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("PositionalProximityRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        positions_by_term = index["positions_by_term"]

        q_terms = list(dict.fromkeys(simple_tokenize(query)))
        if not q_terms:
            return []

        doc_term_positions = defaultdict(dict)
        for term in q_terms:
            for doc_idx, positions in positions_by_term.get(term, []):
                doc_term_positions[doc_idx][term] = positions

        results = []
        for doc_idx, term_positions in doc_term_positions.items():
            if require_all_terms and len(term_positions) < len(q_terms):
                continue

            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue

            min_span = self._min_position_span(list(term_positions.values()))
            coverage_score = len(term_positions) / max(1, len(q_terms))
            frequency_score = sum(len(v) for v in term_positions.values())
            proximity_bonus = 0.0
            if min_span is not None:
                proximity_bonus = 1.0 / max(1.0, float(min_span))
                if min_span <= proximity_window:
                    proximity_bonus += 1.0

            score = coverage_score * 5.0 + math.log1p(frequency_score) + proximity_bonus
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={
                    "matched_terms": sorted(term_positions),
                    "min_position_span": min_span,
                    "proximity_window": proximity_window,
                },
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _min_position_span(self, position_lists: List[List[int]]) -> Optional[int]:
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

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "positional"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No positional docs found: {index_dir}")

        positions_by_term = defaultdict(list)
        for p in postings:
            positions = safe_json_loads(p.get("positions_json")) or []
            positions_by_term[p["term"]].append((int(p["doc_idx"]), [int(v) for v in positions]))

        loaded = {"docs": docs, "positions_by_term": positions_by_term}
        self._cache[key] = loaded
        return loaded


class TemporalRetrieverStrategy(BaseRetrieverStrategy):
    name = "temporal"
    description = "Retrieves chunks by date values from the temporal index."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str = "",
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        date_values: Optional[List[str]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("TemporalRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings = index["postings"]

        wanted = {str(v).lower() for v in date_values or []}
        query_terms = set(simple_tokenize(query))
        date_like_terms = {t for t in query_terms if any(ch.isdigit() for ch in t)}

        scores = defaultdict(float)
        matched_dates = defaultdict(list)

        for p in postings:
            value = str(p.get("date_value") or "")
            value_l = value.lower()
            hit = False

            if wanted and value_l in wanted:
                hit = True
            if date_like_terms and any(term in value_l for term in date_like_terms):
                hit = True
            if not wanted and not date_like_terms and query:
                hit = bool(set(simple_tokenize(value)) & query_terms)
            if not wanted and not query:
                hit = True

            if not hit:
                continue

            doc_idx = int(p["doc_idx"])
            scores[doc_idx] += 1.0
            matched_dates[doc_idx].append(value)

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_dates": sorted(set(matched_dates[doc_idx]))},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "temporal"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No temporal docs found: {index_dir}")

        loaded = {"docs": docs, "postings": postings}
        self._cache[key] = loaded
        return loaded


class LayoutSpatialRetrieverStrategy(BaseRetrieverStrategy):
    name = "layout_spatial"
    description = "Retrieves chunks by layout page, zone, element type, or normalized x/y bins."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str = "",
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        pages: Optional[List[int]] = None,
        zones: Optional[List[str]] = None,
        element_types: Optional[List[str]] = None,
        x_bins: Optional[List[int]] = None,
        y_bins: Optional[List[int]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("LayoutSpatialRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        postings = index["postings"]

        wanted_pages = {int(p) for p in pages or []}
        wanted_zones = {str(z).lower() for z in zones or []}
        wanted_types = {str(t).lower() for t in element_types or []}
        wanted_x = {int(x) for x in x_bins or []}
        wanted_y = {int(y) for y in y_bins or []}
        query_terms = set(simple_tokenize(query))

        scores = defaultdict(float)
        matched_layout = defaultdict(list)

        for p in postings:
            hit = False
            score = 0.0

            page_number = safe_int(p.get("page_number"))
            if wanted_pages and page_number in wanted_pages:
                hit = True
                score += 2.0

            zone = str(p.get("zone") or "").lower()
            if wanted_zones and zone in wanted_zones:
                hit = True
                score += 1.5

            element_type = str(p.get("element_type") or "").lower()
            if wanted_types and element_type in wanted_types:
                hit = True
                score += 1.0

            x_bin = safe_int(p.get("x_bin"))
            y_bin = safe_int(p.get("y_bin"))
            if wanted_x and x_bin in wanted_x:
                hit = True
                score += 0.75
            if wanted_y and y_bin in wanted_y:
                hit = True
                score += 0.75

            if query_terms:
                layout_terms = set(simple_tokenize(" ".join([str(page_number), zone, element_type])))
                if query_terms & layout_terms:
                    hit = True
                    score += 1.0

            if not any([wanted_pages, wanted_zones, wanted_types, wanted_x, wanted_y, query_terms]):
                hit = True
                score = 1.0

            if not hit:
                continue

            doc_idx = int(p["doc_idx"])
            scores[doc_idx] += score
            matched_layout[doc_idx].append({
                "page_number": page_number,
                "zone": zone,
                "element_type": element_type,
                "x_bin": x_bin,
                "y_bin": y_bin,
                "element_id": p.get("element_id"),
            })

        results = []
        for doc_idx, score in scores.items():
            row = docs[doc_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={"matched_layout": matched_layout[doc_idx][:20]},
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "layout_spatial"
        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            raise FileNotFoundError(f"No layout spatial docs found: {index_dir}")

        loaded = {"docs": docs, "postings": postings}
        self._cache[key] = loaded
        return loaded


class MinHashDuplicateRetrieverStrategy(BaseRetrieverStrategy):
    name = "minhash_duplicates"
    description = "Finds near-duplicate candidate chunks from the MinHash LSH index."

    def __init__(self, seed_retriever: Optional[BaseRetrieverStrategy] = None):
        self.seed_retriever = seed_retriever or BM25RetrieverStrategy()
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str = "",
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[int] = None,
        doc_idx: Optional[int] = None,
        include_seed: bool = True,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("MinHashDuplicateRetrieverStrategy requires chunk_strategy_name")

        index = self._load_index(index_root_dir, chunk_strategy_name)
        docs = index["docs"]
        neighbors = index["neighbors"]

        seed_doc_idx = doc_idx
        if seed_doc_idx is None and chunk_id is not None:
            for i, row in enumerate(docs):
                if safe_int(row.get("chunk_id")) == int(chunk_id):
                    seed_doc_idx = i
                    break

        seed_result = None
        if seed_doc_idx is None and query:
            base = self.seed_retriever.retrieve(
                store=store,
                index_root_dir=index_root_dir,
                query=query,
                chunk_strategy_name=chunk_strategy_name,
                top_k=1,
                filters=filters,
                return_text_chars=return_text_chars,
            )
            if base:
                seed_doc_idx = safe_int(base[0].get("doc_idx"))
                seed_result = base[0]

        if seed_doc_idx is None or seed_doc_idx < 0 or seed_doc_idx >= len(docs):
            return []

        results = []
        if include_seed:
            seed_row = docs[seed_doc_idx]
            if passes_filters(seed_row, filters):
                seed = seed_result or make_result(
                    row=seed_row,
                    score=1.0,
                    retriever_name=self.name,
                    query=query,
                    store=store,
                    return_text_chars=return_text_chars,
                    extra={"duplicate_role": "seed"},
                )
                seed = dict(seed)
                seed["retriever_name"] = self.name
                seed["duplicate_role"] = "seed"
                results.append(seed)

        for neighbor_idx, shared_bucket_count in neighbors.get(seed_doc_idx, {}).items():
            row = docs[neighbor_idx]
            if not passes_filters(row, filters):
                continue
            results.append(make_result(
                row=row,
                score=shared_bucket_count,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
                extra={
                    "duplicate_role": "candidate",
                    "seed_doc_idx": seed_doc_idx,
                    "shared_bucket_count": shared_bucket_count,
                },
            ))

        results.sort(key=lambda r: (r.get("duplicate_role") != "seed", -float(r["score"])))
        return results[:top_k]

    def _load_index(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "minhash_lsh"
        docs = read_parquet(index_dir / "docs.parquet")
        candidate_pairs = read_parquet(index_dir / "candidate_pairs.parquet")
        if not docs:
            raise FileNotFoundError(f"No MinHash LSH docs found: {index_dir}")

        neighbors = defaultdict(Counter)
        for pair in candidate_pairs:
            left = int(pair["left_doc_idx"])
            right = int(pair["right_doc_idx"])
            neighbors[left][right] += 1
            neighbors[right][left] += 1

        loaded = {"docs": docs, "neighbors": neighbors}
        self._cache[key] = loaded
        return loaded


class MetadataFilterRetrieverStrategy(BaseRetrieverStrategy):
    name = "metadata_filter"
    description = "Retrieves chunks using only metadata filters."

    def __init__(self):
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str = "",
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        sort_by_quality: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("MetadataFilterRetrieverStrategy requires chunk_strategy_name")

        docs = self._load_docs(index_root_dir, chunk_strategy_name)

        results = []

        for row in docs:
            if not passes_filters(row, filters):
                continue

            score = 1.0
            if sort_by_quality:
                score += safe_float(row.get("clinical_quality_score"), 0.0)

            results.append(make_result(
                row=row,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                return_text_chars=return_text_chars,
            ))

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_docs(self, index_root_dir: Path, chunk_strategy_name: str) -> List[Dict[str, Any]]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "bm25"
        docs = read_parquet(index_dir / "docs.parquet")

        if not docs:
            index_dir = index_root_dir / chunk_strategy_name / "metadata_inverted"
            docs = read_parquet(index_dir / "docs.parquet")

        if not docs:
            raise FileNotFoundError(f"No docs found for metadata retrieval: {chunk_strategy_name}")

        self._cache[key] = docs
        return docs


class BM25MetadataBoostRetrieverStrategy(BaseRetrieverStrategy):
    name = "bm25_metadata_boost"
    description = "BM25 retrieval with metadata boosts for section/entity flags/quality."

    def __init__(
        self,
        base_bm25: Optional[BM25RetrieverStrategy] = None,
        quality_weight: float = 0.20,
        section_boost: float = 0.20,
        flag_boost: float = 0.15,
    ):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()
        self.quality_weight = quality_weight
        self.section_boost = section_boost
        self.flag_boost = flag_boost

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        preferred_sections: Optional[List[str]] = None,
        preferred_flags: Optional[List[str]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        base_results = self.base_bm25.retrieve(
            store=store,
            index_root_dir=index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=max(top_k * 5, 50),
            filters=filters,
            return_text_chars=return_text_chars,
        )

        if not base_results:
            return []

        max_base = max(r["score"] for r in base_results) or 1.0

        preferred_sections = set(preferred_sections or [])
        preferred_flags = preferred_flags or []

        boosted = []

        for r in base_results:
            base_norm = normalize_score(r["score"], max_base)
            score = base_norm

            quality = safe_float(r.get("clinical_quality_score"), 0.0)
            score += self.quality_weight * quality

            if preferred_sections and r.get("primary_section") in preferred_sections:
                score += self.section_boost

            for flag in preferred_flags:
                if bool(r.get(flag)):
                    score += self.flag_boost

            r2 = dict(r)
            r2["base_score"] = r["score"]
            r2["score"] = round(score, 6)
            r2["retriever_name"] = self.name
            boosted.append(r2)

        boosted.sort(key=lambda r: r["score"], reverse=True)
        return boosted[:top_k]


class MultiQueryBM25RetrieverStrategy(BaseRetrieverStrategy):
    name = "multi_query_bm25"
    description = "Runs multiple query variants over BM25 and fuses results with RRF."

    def __init__(self, base_bm25: Optional[BM25RetrieverStrategy] = None):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_variants: Optional[List[str]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        variants = query_variants or self.default_variants(query)

        ranked_lists = []

        for q in variants:
            rows = self.base_bm25.retrieve(
                store=store,
                index_root_dir=index_root_dir,
                query=q,
                chunk_strategy_name=chunk_strategy_name,
                top_k=max(top_k * 3, 20),
                filters=filters,
                return_text_chars=return_text_chars,
            )
            ranked_lists.append(rows)

        fused = reciprocal_rank_fusion(
            ranked_lists,
            top_k=top_k,
            k=60,
            retriever_name=self.name,
        )

        for r in fused:
            r["query_variants"] = variants

        return fused

    def default_variants(self, query: str) -> List[str]:
        q = query.strip()
        variants = [q]

        # Simple clinical-ish lexical expansions. Keep this lightweight.
        expansions = {
            "meds": "medications medicine drug prescription",
            "medication": "medication medicine drug prescription dose tablet",
            "diagnosis": "assessment impression diagnosis problem",
            "pain": "pain ache tenderness discomfort",
            "follow up": "follow up follow-up return clinic plan",
            "lab": "lab laboratory result test blood",
            "xray": "xray x-ray radiology imaging",
        }

        q_lower = q.lower()

        for key, exp in expansions.items():
            if key in q_lower:
                variants.append(q + " " + exp)

        # Keyword-only version.
        tokens = simple_tokenize(q)
        if tokens:
            variants.append(" ".join(tokens))

        return list(dict.fromkeys(variants))


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    top_k: int = 10,
    k: int = 60,
    retriever_name: str = "rrf_fusion",
) -> List[Dict[str, Any]]:
    fused_scores = defaultdict(float)
    best_result = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            key = result_key(result)
            fused_scores[key] += 1.0 / (k + rank)

            if key not in best_result or result.get("score", 0) > best_result[key].get("score", 0):
                best_result[key] = result

    fused = []

    for key, score in fused_scores.items():
        r = dict(best_result[key])
        r["score"] = round(float(score), 6)
        r["retriever_name"] = retriever_name
        r["fusion_sources"] = len(ranked_lists)
        fused.append(r)

    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:top_k]


class LexicalFusionRetrieverStrategy(BaseRetrieverStrategy):
    name = "lexical_fusion"
    description = "Fuses BM25 and exact keyword retrieval using RRF."

    def __init__(
        self,
        bm25: Optional[BM25RetrieverStrategy] = None,
        keyword: Optional[KeywordExactRetrieverStrategy] = None,
    ):
        self.bm25 = bm25 or BM25RetrieverStrategy()
        self.keyword = keyword or KeywordExactRetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25.retrieve(
            store=store,
            index_root_dir=index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=max(top_k * 3, 20),
            filters=filters,
            return_text_chars=return_text_chars,
        )

        keyword_results = self.keyword.retrieve(
            store=store,
            index_root_dir=index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=max(top_k * 3, 20),
            filters=filters,
            return_text_chars=return_text_chars,
        )

        return reciprocal_rank_fusion(
            [bm25_results, keyword_results],
            top_k=top_k,
            k=60,
            retriever_name=self.name,
        )


class AdvancedLexicalFusionRetrieverStrategy(BaseRetrieverStrategy):
    name = "advanced_lexical_fusion"
    description = "Fuses BM25, TF-IDF, exact keyword, fielded lexical, phrase, and char n-gram retrieval."

    def __init__(
        self,
        retrievers: Optional[List[BaseRetrieverStrategy]] = None,
    ):
        self.retrievers = retrievers or [
            BM25RetrieverStrategy(),
            TFIDFRetrieverStrategy(),
            KeywordExactRetrieverStrategy(),
            FieldedLexicalRetrieverStrategy(),
            PhraseNgramRetrieverStrategy(),
            CharacterNgramRetrieverStrategy(),
        ]

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        ranked_lists = []

        for retriever in self.retrievers:
            try:
                ranked_lists.append(retriever.retrieve(
                    store=store,
                    index_root_dir=index_root_dir,
                    query=query,
                    chunk_strategy_name=chunk_strategy_name,
                    top_k=max(top_k * 3, 20),
                    filters=filters,
                    return_text_chars=return_text_chars,
                    **kwargs,
                ))
            except Exception as e:
                print(f"Advanced fusion skipping {retriever.name}: {e}")

        return reciprocal_rank_fusion(
            ranked_lists,
            top_k=top_k,
            k=60,
            retriever_name=self.name,
        )


class CrossChunkStrategyFusionRetriever(BaseRetrieverStrategy):
    name = "cross_chunk_fusion"
    description = "Runs a base retriever across many chunk strategies and fuses with RRF."

    def __init__(
        self,
        base_retriever: Optional[BaseRetrieverStrategy] = None,
    ):
        self.base_retriever = base_retriever or BM25MetadataBoostRetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        chunk_strategy_names: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if chunk_strategy_names is None:
            chunk_strategy_names = [
                p.name
                for p in index_root_dir.iterdir()
                if p.is_dir()
            ]

        ranked_lists = []

        for cs in chunk_strategy_names:
            try:
                rows = self.base_retriever.retrieve(
                    store=store,
                    index_root_dir=index_root_dir,
                    query=query,
                    chunk_strategy_name=cs,
                    top_k=max(top_k * 2, 20),
                    filters=filters,
                    return_text_chars=return_text_chars,
                    **kwargs,
                )
                ranked_lists.append(rows)
            except Exception as e:
                print(f"Skipping chunk strategy {cs}: {e}")

        return reciprocal_rank_fusion(
            ranked_lists,
            top_k=top_k,
            k=60,
            retriever_name=self.name,
        )


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(simple_tokenize(a))
    tb = set(simple_tokenize(b))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


class MMRDiversityRetrieverStrategy(BaseRetrieverStrategy):
    name = "mmr_diversity"
    description = "Runs a base retriever, then applies MMR diversity selection."

    def __init__(
        self,
        base_retriever: Optional[BaseRetrieverStrategy] = None,
        lambda_mult: float = 0.75,
    ):
        self.base_retriever = base_retriever or BM25MetadataBoostRetrieverStrategy()
        self.lambda_mult = lambda_mult

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        fetch_k: int = 50,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        candidates = self.base_retriever.retrieve(
            store=store,
            index_root_dir=index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=max(fetch_k, top_k),
            filters=filters,
            return_text_chars=max(return_text_chars, 1200),
            **kwargs,
        )

        if not candidates:
            return []

        max_score = max(c["score"] for c in candidates) or 1.0

        selected = []
        remaining = list(candidates)

        while remaining and len(selected) < top_k:
            best = None
            best_mmr = -1e9

            for c in remaining:
                relevance = normalize_score(c["score"], max_score)

                if selected:
                    diversity_penalty = max(
                        jaccard_similarity(c["text_preview"], s["text_preview"])
                        for s in selected
                    )
                else:
                    diversity_penalty = 0.0

                mmr_score = self.lambda_mult * relevance - (1 - self.lambda_mult) * diversity_penalty

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = c

            remaining.remove(best)
            best = dict(best)
            best["score"] = round(float(best_mmr), 6)
            best["retriever_name"] = self.name
            selected.append(best)

        return selected


class ContextExpansionRetrieverStrategy(BaseRetrieverStrategy):
    name = "context_expansion"
    description = "Retrieves chunks, then expands answer context using parent_start/parent_end when available."

    def __init__(self, base_retriever: Optional[BaseRetrieverStrategy] = None):
        self.base_retriever = base_retriever or BM25MetadataBoostRetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 2000,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = self.base_retriever.retrieve(
            store=store,
            index_root_dir=index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=top_k,
            filters=filters,
            return_text_chars=return_text_chars,
            **kwargs,
        )

        expanded = []

        for r in results:
            parent_start = safe_int(r.get("parent_start"))
            parent_end = safe_int(r.get("parent_end"))

            # Some index docs may not expose these unless included in docs.parquet.
            # Fallback to original chunk span.
            if parent_start is None or parent_end is None:
                parent_start = safe_int(r.get("start_index"))
                parent_end = safe_int(r.get("end_index"))

            expanded_text = store.decode_span(parent_start, parent_end)

            r2 = dict(r)
            r2["retriever_name"] = self.name
            r2["expanded_start_index"] = parent_start
            r2["expanded_end_index"] = parent_end
            r2["expanded_text_preview"] = expanded_text[:return_text_chars]

            expanded.append(r2)

        return expanded


class GraphExpansionRetrieverStrategy(BaseRetrieverStrategy):
    name = "graph_expansion"
    description = "Retrieves base hits, then adds related chunks from the chunk graph index."

    def __init__(
        self,
        base_retriever: Optional[BaseRetrieverStrategy] = None,
        edge_types: Optional[List[str]] = None,
    ):
        self.base_retriever = base_retriever or BM25MetadataBoostRetrieverStrategy()
        self.edge_types = set(edge_types or [
            "previous_next",
            "next_previous",
            "token_overlap",
            "same_primary_section",
            "parent",
            "child",
        ])
        self._cache = {}

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        base_top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("GraphExpansionRetrieverStrategy requires chunk_strategy_name")

        graph = self._load_graph(index_root_dir, chunk_strategy_name)
        docs = graph["docs"]
        edges_by_source = graph["edges_by_source"]

        base_results = self.base_retriever.retrieve(
            store=store,
            index_root_dir=index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=base_top_k or max(top_k, 10),
            filters=filters,
            return_text_chars=return_text_chars,
            **kwargs,
        )

        scored = {}
        for result in base_results:
            doc_idx = safe_int(result.get("doc_idx"))
            if doc_idx is None:
                continue
            key = doc_idx
            candidate = dict(result)
            candidate["retriever_name"] = self.name
            candidate["graph_role"] = "base"
            scored[key] = candidate

            for edge in edges_by_source.get(doc_idx, []):
                if edge["edge_type"] not in self.edge_types:
                    continue
                target_idx = int(edge["target_doc_idx"])
                if target_idx < 0 or target_idx >= len(docs):
                    continue

                row = docs[target_idx]
                if not passes_filters(row, filters):
                    continue

                edge_score = safe_float(result.get("score"), 0.0) * safe_float(edge.get("weight"), 0.0)
                if target_idx not in scored or edge_score > scored[target_idx].get("score", 0):
                    scored[target_idx] = make_result(
                        row=row,
                        score=edge_score,
                        retriever_name=self.name,
                        query=query,
                        store=store,
                        return_text_chars=return_text_chars,
                        extra={
                            "graph_role": "neighbor",
                            "source_doc_idx": doc_idx,
                            "edge_type": edge["edge_type"],
                            "edge_weight": edge["weight"],
                        },
                    )

        results = list(scored.values())
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _load_graph(self, index_root_dir: Path, chunk_strategy_name: str) -> Dict[str, Any]:
        key = (str(index_root_dir), chunk_strategy_name, self.name)
        if key in self._cache:
            return self._cache[key]

        index_dir = index_root_dir / chunk_strategy_name / "chunk_graph"
        docs = read_parquet(index_dir / "docs.parquet")
        edges = read_parquet(index_dir / "edges.parquet")
        if not docs:
            raise FileNotFoundError(f"No chunk graph docs found: {index_dir}")

        edges_by_source = defaultdict(list)
        for edge in edges:
            edges_by_source[int(edge["source_doc_idx"])].append(edge)

        loaded = {"docs": docs, "edges_by_source": edges_by_source}
        self._cache[key] = loaded
        return loaded


class RetrieverManager:
    """
    Register and run retrieval strategies.

    Similar to:
      ChunkManager -> chunking strategies
      IndexManager -> indexing strategies
      RetrieverManager -> retrieval strategies
    """

    def __init__(
        self,
        store,
        index_root_dir: Optional[Union[str, Path]] = None,
    ):
        self.store = store
        self.document_dir = self.store.paths.document_dir
        self.index_root_dir = Path(index_root_dir) if index_root_dir else self.document_dir / "indexes"

        self._retrievers: Dict[str, BaseRetrieverStrategy] = {}
        self._descriptions: Dict[str, str] = {}
        self._errors: Dict[str, str] = {}

    def register_retriever(
        self,
        retriever: BaseRetrieverStrategy,
        force: bool = False,
    ) -> None:
        name = retriever.name

        if name in self._retrievers and not force:
            raise ValueError(f"Retriever already registered: {name}")

        self._retrievers[name] = retriever
        self._descriptions[name] = retriever.description

    def list_retrievers(self, with_descriptions: bool = False):
        if with_descriptions:
            return {
                name: self._descriptions.get(name, "")
                for name in self._retrievers
            }

        return list(self._retrievers.keys())

    def list_chunk_strategies(self) -> List[str]:
        if not self.index_root_dir.exists():
            return []

        return sorted(p.name for p in self.index_root_dir.iterdir() if p.is_dir())

    def retrieve(
        self,
        retriever_name: str,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if retriever_name not in self._retrievers:
            raise KeyError(f"Unknown retriever: {retriever_name}")

        return self._retrievers[retriever_name].retrieve(
            store=self.store,
            index_root_dir=self.index_root_dir,
            query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=top_k,
            filters=filters,
            **kwargs,
        )

    def retrieve_all(
        self,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        retriever_names: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        continue_on_error: bool = True,
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        outputs = {}
        self._errors.clear()

        retriever_names = retriever_names or list(self._retrievers.keys())

        for name in retriever_names:
            try:
                outputs[name] = self.retrieve(
                    retriever_name=name,
                    query=query,
                    chunk_strategy_name=chunk_strategy_name,
                    top_k=top_k,
                    filters=filters,
                    **kwargs,
                )
            except Exception as e:
                error = "".join(traceback.format_exception_only(type(e), e)).strip()
                self._errors[name] = error
                print(f"Retriever failed: {name}: {error}")

                if not continue_on_error:
                    raise

        return outputs

    def retrieve_fused(
        self,
        query: str,
        retriever_names: List[str],
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        ranked_lists = []

        for name in retriever_names:
            rows = self.retrieve(
                retriever_name=name,
                query=query,
                chunk_strategy_name=chunk_strategy_name,
                top_k=max(top_k * 3, 20),
                filters=filters,
                **kwargs,
            )
            ranked_lists.append(rows)

        return reciprocal_rank_fusion(
            ranked_lists,
            top_k=top_k,
            k=60,
            retriever_name="manager_rrf_fusion",
        )

    def errors(self) -> Dict[str, str]:
        return dict(self._errors)


class QueryExpansionBM25RetrieverStrategy(BaseRetrieverStrategy):
    """
    BM25 retrieval with medical query expansion.

    Expands query terms using a built-in medical abbreviation dictionary and
    simple morphological variants (plurals, verb forms), then runs BM25 on
    the union of original + expanded terms.  Results from all query variants
    are fused with Reciprocal Rank Fusion.

    Covers: MI→myocardial infarction, HTN→hypertension, DM→diabetes mellitus,
    COPD, CHF, UTI, PE, DVT, CAD, CABG, PCI, BPH, GERD, CVA, TIA, and more.
    """

    name = "query_expansion_bm25"
    description = "BM25 retrieval with built-in medical abbreviation expansion and RRF fusion across query variants."

    MEDICAL_ABBREVIATIONS: Dict[str, List[str]] = {
        "mi": ["myocardial infarction", "heart attack"],
        "htn": ["hypertension", "high blood pressure"],
        "dm": ["diabetes mellitus", "diabetes"],
        "dm2": ["type 2 diabetes", "diabetes mellitus type 2"],
        "t2dm": ["type 2 diabetes", "diabetes mellitus"],
        "t1dm": ["type 1 diabetes", "insulin dependent diabetes"],
        "copd": ["chronic obstructive pulmonary disease"],
        "chf": ["congestive heart failure", "heart failure"],
        "uti": ["urinary tract infection"],
        "pe": ["pulmonary embolism"],
        "dvt": ["deep vein thrombosis"],
        "cad": ["coronary artery disease"],
        "cabg": ["coronary artery bypass graft"],
        "pci": ["percutaneous coronary intervention", "stent"],
        "bph": ["benign prostatic hyperplasia"],
        "gerd": ["gastroesophageal reflux disease", "acid reflux"],
        "cva": ["cerebrovascular accident", "stroke"],
        "tia": ["transient ischemic attack", "mini stroke"],
        "afib": ["atrial fibrillation"],
        "af": ["atrial fibrillation"],
        "chemo": ["chemotherapy"],
        "bp": ["blood pressure"],
        "hr": ["heart rate"],
        "rr": ["respiratory rate"],
        "temp": ["temperature"],
        "spo2": ["oxygen saturation"],
        "o2": ["oxygen"],
        "wbc": ["white blood cell", "leukocyte"],
        "rbc": ["red blood cell", "erythrocyte"],
        "hgb": ["hemoglobin"],
        "hct": ["hematocrit"],
        "plt": ["platelet"],
        "cr": ["creatinine"],
        "bun": ["blood urea nitrogen"],
        "k": ["potassium"],
        "na": ["sodium"],
        "hba1c": ["hemoglobin a1c", "glycated hemoglobin"],
        "ldl": ["low density lipoprotein", "bad cholesterol"],
        "hdl": ["high density lipoprotein", "good cholesterol"],
        "bnp": ["b-type natriuretic peptide", "brain natriuretic peptide"],
        "ekg": ["electrocardiogram", "ecg"],
        "ecg": ["electrocardiogram"],
        "echo": ["echocardiogram"],
        "ct": ["computed tomography", "cat scan"],
        "mri": ["magnetic resonance imaging"],
        "cxr": ["chest x-ray", "chest radiograph"],
        "icu": ["intensive care unit"],
        "ed": ["emergency department", "emergency room"],
        "or": ["operating room", "surgery"],
        "po": ["by mouth", "oral"],
        "iv": ["intravenous"],
        "im": ["intramuscular"],
        "bid": ["twice daily", "twice a day"],
        "tid": ["three times daily"],
        "qd": ["once daily", "every day"],
        "qid": ["four times daily"],
        "prn": ["as needed"],
    }

    def __init__(self, base_bm25: Optional[BM25RetrieverStrategy] = None):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        query_variants = self._expand_query(query)
        ranked_lists = []
        for variant in query_variants:
            try:
                results = self.base_bm25.retrieve(
                    store=store,
                    index_root_dir=index_root_dir,
                    query=variant,
                    chunk_strategy_name=chunk_strategy_name,
                    top_k=top_k * 2,
                    filters=filters,
                    return_text_chars=return_text_chars,
                )
                if results:
                    ranked_lists.append(results)
            except Exception:
                pass
        if not ranked_lists:
            return []
        fused = reciprocal_rank_fusion(ranked_lists, top_k=top_k, retriever_name=self.name)
        for r in fused:
            r["expanded_variants"] = query_variants
        return fused

    def _expand_query(self, query: str) -> List[str]:
        variants = [query]
        tokens = simple_tokenize(query)
        expansions_added = set()
        for token in tokens:
            for expansion in self.MEDICAL_ABBREVIATIONS.get(token.lower(), []):
                if expansion not in expansions_added:
                    variants.append(re.sub(r"\b" + re.escape(token) + r"\b", expansion, query, flags=re.IGNORECASE))
                    expansions_added.add(expansion)
        return variants[:5]  # cap at 5 variants to avoid exploding latency


class EntityCentricFusionRetrieverStrategy(BaseRetrieverStrategy):
    """
    Entity-centric retrieval with RRF fusion.

    Extracts candidate entity terms from the query (capitalized words,
    known medical terms, multi-word phrases), retrieves separately for each
    via the entity index, then fuses with RRF.  Outperforms plain BM25 for
    queries like "metformin side effects in patients with renal failure"
    where several distinct entities carry the query intent.
    """

    name = "entity_centric_fusion"
    description = "Extracts entity terms from the query, retrieves per entity via entity index, fuses with RRF."

    # Patterns that suggest a medical entity in the query
    _ENTITY_HINT_RE = re.compile(
        r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*"  # capitalized multi-word
        r"|(?:mg|mcg|mL|mg/dL|mmol/L|IU/L|bpm|mmHg)\b"  # units → numeric context
        r")\b"
    )

    def __init__(self, base_bm25: Optional[BM25RetrieverStrategy] = None):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("entity_centric_fusion requires chunk_strategy_name")

        entity_terms = self._extract_entity_terms(query)
        ranked_lists = []

        # Always include a baseline BM25 run
        try:
            baseline = self.base_bm25.retrieve(
                store=store, index_root_dir=index_root_dir, query=query,
                chunk_strategy_name=chunk_strategy_name, top_k=top_k * 2,
                filters=filters, return_text_chars=return_text_chars,
            )
            if baseline:
                ranked_lists.append(baseline)
        except Exception:
            pass

        # Per-entity entity-index lookups
        entity_index_dir = index_root_dir / chunk_strategy_name / "entity"
        if entity_index_dir.exists():
            docs = read_parquet(entity_index_dir / "docs.parquet")
            postings = read_parquet(entity_index_dir / "postings.parquet")
            if docs and postings:
                docs_by_idx = {int(d["doc_idx"]): d for d in docs}
                postings_by_term: Dict[str, List[int]] = defaultdict(list)
                for p in postings:
                    term_key = str(p.get("entity_normalized") or p.get("term") or "").lower()
                    if term_key:
                        postings_by_term[term_key].append(int(p["doc_idx"]))

                for term in entity_terms:
                    term_lower = term.lower()
                    matching_doc_idxs = postings_by_term.get(term_lower, [])
                    if not matching_doc_idxs:
                        # fuzzy: check partial matches
                        matching_doc_idxs = [
                            idx for t, idxs in postings_by_term.items()
                            if term_lower in t or t in term_lower
                            for idx in idxs
                        ][:50]
                    if not matching_doc_idxs:
                        continue
                    entity_results = []
                    seen = set()
                    for doc_idx in matching_doc_idxs:
                        if doc_idx in seen:
                            continue
                        seen.add(doc_idx)
                        row = docs_by_idx.get(doc_idx)
                        if row and passes_filters(row, filters):
                            entity_results.append(make_result(
                                row=row, score=1.0 / (1 + len(entity_results)),
                                retriever_name=self.name, query=query, store=store,
                                return_text_chars=return_text_chars,
                                extra={"matched_entity_term": term},
                            ))
                    if entity_results:
                        ranked_lists.append(entity_results[:top_k * 2])

        if not ranked_lists:
            return []
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k, retriever_name=self.name)

    def _extract_entity_terms(self, query: str) -> List[str]:
        terms = []
        # capitalized words / medical capitalized phrases
        for m in self._ENTITY_HINT_RE.finditer(query):
            t = m.group(1).strip()
            if len(t) > 2:
                terms.append(t)
        # also include all tokens >= 4 chars (likely content words)
        for tok in simple_tokenize(query):
            if len(tok) >= 4 and tok not in [t.lower() for t in terms]:
                terms.append(tok)
        return list(dict.fromkeys(terms))[:8]  # deduplicate, cap at 8


class NumericRangeRetrieverStrategy(BaseRetrieverStrategy):
    """
    Numeric range retrieval from the numeric_range index.

    Parses the query for numeric conditions expressed as:
      - explicit comparisons: "> 200", "< 60", ">= 7.0", "<= 120"
      - approximate values: "around 150", "~90"
      - ranges: "between 70 and 120", "70–120"

    Returns chunks that contain at least one numeric value satisfying ALL
    parsed conditions.  Falls back to BM25 when no numeric conditions are found.
    """

    name = "numeric_range_retriever"
    description = "Parses numeric conditions from the query and retrieves chunks via the numeric_range index."

    _GT_RE  = re.compile(r"(?:greater than|above|over|>)\s*=?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
    _LT_RE  = re.compile(r"(?:less than|below|under|<)\s*=?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
    _GTE_RE = re.compile(r">=\s*(\d+(?:\.\d+)?)")
    _LTE_RE = re.compile(r"<=\s*(\d+(?:\.\d+)?)")
    _BTW_RE = re.compile(r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
    _APPROX_RE = re.compile(r"(?:around|approximately|~)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

    def __init__(self, base_bm25: Optional[BM25RetrieverStrategy] = None):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("numeric_range_retriever requires chunk_strategy_name")

        conditions = self._parse_conditions(query)
        if not conditions:
            # no numeric condition detected — fall back to BM25
            return self.base_bm25.retrieve(
                store=store, index_root_dir=index_root_dir, query=query,
                chunk_strategy_name=chunk_strategy_name, top_k=top_k,
                filters=filters, return_text_chars=return_text_chars,
            )

        index_dir = index_root_dir / chunk_strategy_name / "numeric_range"
        if not index_dir.exists():
            return self.base_bm25.retrieve(
                store=store, index_root_dir=index_root_dir, query=query,
                chunk_strategy_name=chunk_strategy_name, top_k=top_k,
                filters=filters, return_text_chars=return_text_chars,
            )

        docs = read_parquet(index_dir / "docs.parquet")
        postings = read_parquet(index_dir / "postings.parquet")
        if not docs:
            return []

        docs_by_idx = {int(d["doc_idx"]): d for d in docs}
        matching: Dict[int, float] = defaultdict(float)
        for p in postings:
            value = float(p.get("numeric_value") or 0)
            if self._satisfies_all(value, conditions):
                doc_idx = int(p["doc_idx"])
                # score: prefer exact/central matches
                matching[doc_idx] += 1.0

        results = []
        for doc_idx, score in sorted(matching.items(), key=lambda x: -x[1]):
            row = docs_by_idx.get(doc_idx)
            if row and passes_filters(row, filters):
                results.append(make_result(
                    row=row, score=score, retriever_name=self.name, query=query,
                    store=store, return_text_chars=return_text_chars,
                    extra={"numeric_conditions": conditions},
                ))
        return results[:top_k]

    def _parse_conditions(self, query: str) -> List[Dict[str, float]]:
        conditions = []
        for m in self._GTE_RE.finditer(query):
            conditions.append({"op": "gte", "value": float(m.group(1))})
        for m in self._LTE_RE.finditer(query):
            conditions.append({"op": "lte", "value": float(m.group(1))})
        for m in self._GT_RE.finditer(query):
            if not any(c["op"] == "gte" and c["value"] == float(m.group(1)) for c in conditions):
                conditions.append({"op": "gt", "value": float(m.group(1))})
        for m in self._LT_RE.finditer(query):
            if not any(c["op"] == "lte" and c["value"] == float(m.group(1)) for c in conditions):
                conditions.append({"op": "lt", "value": float(m.group(1))})
        for m in self._BTW_RE.finditer(query):
            conditions.append({"op": "gte", "value": float(m.group(1))})
            conditions.append({"op": "lte", "value": float(m.group(2))})
        for m in self._APPROX_RE.finditer(query):
            v = float(m.group(1))
            margin = v * 0.15
            conditions.append({"op": "gte", "value": v - margin})
            conditions.append({"op": "lte", "value": v + margin})
        return conditions

    @staticmethod
    def _satisfies_all(value: float, conditions: List[Dict[str, float]]) -> bool:
        for c in conditions:
            op, threshold = c["op"], c["value"]
            if op == "gt" and not (value > threshold):
                return False
            if op == "gte" and not (value >= threshold):
                return False
            if op == "lt" and not (value < threshold):
                return False
            if op == "lte" and not (value <= threshold):
                return False
        return True


class NegationAwareBM25RetrieverStrategy(BaseRetrieverStrategy):
    """
    BM25 retrieval with negation demotion.

    Runs standard BM25, then re-scores by penalising chunks where the
    query's content words appear primarily in negated entity contexts
    (contains_negation=True AND entity text overlaps with query terms).
    Useful when the query is about a positive finding but many retrieved
    chunks discuss its absence ("no fever", "denies chest pain").
    """

    name = "negation_aware_bm25"
    description = "BM25 retrieval that demotes chunks where key query terms appear in negated clinical contexts."

    def __init__(
        self,
        base_bm25: Optional[BM25RetrieverStrategy] = None,
        negation_penalty: float = 0.35,
    ):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()
        self.negation_penalty = negation_penalty

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        candidates = self.base_bm25.retrieve(
            store=store, index_root_dir=index_root_dir, query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=top_k * 3,  # over-fetch before demotion
            filters=filters, return_text_chars=return_text_chars,
        )
        if not candidates:
            return []

        query_terms = set(simple_tokenize(query))
        results = []
        for r in candidates:
            score = float(r.get("score") or 0)
            penalty = self._compute_negation_penalty(r, query_terms)
            r = dict(r)
            r["score"] = round(score * (1.0 - penalty * self.negation_penalty), 6)
            r["negation_penalty"] = round(penalty, 4)
            r["retriever_name"] = self.name
            results.append(r)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _compute_negation_penalty(result: Dict[str, Any], query_terms: set) -> float:
        if not result.get("contains_negation"):
            return 0.0
        entities = safe_json_loads(result.get("entities")) or []
        if not isinstance(entities, list):
            return 0.0
        negated_terms: set = set()
        for e in entities:
            if not isinstance(e, dict):
                continue
            if bool(e.get("negated") or e.get("is_negated")):
                text = str(e.get("text") or e.get("normalized_value") or "")
                negated_terms.update(simple_tokenize(text))
        overlap = query_terms & negated_terms
        if not overlap or not query_terms:
            return 0.0
        # penalty proportional to fraction of query terms that are negated
        return len(overlap) / len(query_terms)


class ClinicalSectionScopedRetrieverStrategy(BaseRetrieverStrategy):
    """
    Section-scoped BM25 retrieval.

    Restricts the candidate pool to chunks from high-signal clinical sections
    before running BM25 scoring.  Falls back to unrestricted BM25 when not
    enough section-scoped results are found.

    High-signal sections (configurable): Assessment, Plan, Impression,
    Findings, Diagnosis, Discharge Summary, History of Present Illness,
    Physical Exam, Results, Recommendations.
    """

    name = "clinical_section_scoped"
    description = "BM25 retrieval scoped to high-signal clinical sections; falls back to full-corpus BM25 when needed."

    HIGH_SIGNAL_SECTIONS = {
        "assessment", "assessment and plan", "plan", "impression",
        "findings", "diagnosis", "diagnoses", "discharge summary",
        "discharge diagnosis", "history of present illness", "hpi",
        "physical examination", "physical exam", "exam", "results",
        "laboratory results", "lab results", "recommendations",
        "clinical impression", "problem list", "active problems",
        "chief complaint", "summary", "conclusion", "conclusion and recommendations",
    }

    def __init__(
        self,
        base_bm25: Optional[BM25RetrieverStrategy] = None,
        min_scoped_results: int = 3,
        high_signal_sections: Optional[set] = None,
    ):
        self.base_bm25 = base_bm25 or BM25RetrieverStrategy()
        self.min_scoped_results = min_scoped_results
        self.high_signal_sections = high_signal_sections or self.HIGH_SIGNAL_SECTIONS

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_text_chars: int = 1200,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        # Fetch a large candidate pool first, then scope
        candidates = self.base_bm25.retrieve(
            store=store, index_root_dir=index_root_dir, query=query,
            chunk_strategy_name=chunk_strategy_name,
            top_k=top_k * 4,
            filters=filters, return_text_chars=return_text_chars,
        )
        if not candidates:
            return []

        scoped = [
            r for r in candidates
            if str(r.get("primary_section") or "").lower().strip() in self.high_signal_sections
        ]

        if len(scoped) >= self.min_scoped_results:
            results = scoped[:top_k]
        else:
            # soft blend: give section-scoped chunks a 25% boost, then take top_k
            section_set = {id(r) for r in scoped}
            for r in candidates:
                if id(r) in section_set:
                    r = dict(r)
                    r["score"] = float(r.get("score") or 0) * 1.25
            candidates.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
            results = candidates[:top_k]

        for r in results:
            r["retriever_name"] = self.name
        return results


def register_default_retrievers(manager: RetrieverManager) -> None:
    bm25 = BM25RetrieverStrategy()
    keyword = KeywordExactRetrieverStrategy()
    tfidf = TFIDFRetrieverStrategy()
    phrase = PhraseNgramRetrieverStrategy()
    char_ngram = CharacterNgramRetrieverStrategy()
    fielded = FieldedLexicalRetrieverStrategy()
    boolean_set = BooleanSetRetrieverStrategy()
    positional = PositionalProximityRetrieverStrategy()

    manager.register_retriever(bm25)
    manager.register_retriever(keyword)
    manager.register_retriever(tfidf)
    manager.register_retriever(phrase)
    manager.register_retriever(char_ngram)
    manager.register_retriever(fielded)
    manager.register_retriever(boolean_set)
    manager.register_retriever(positional)
    manager.register_retriever(EntityRetrieverStrategy())
    manager.register_retriever(SectionPageRetrieverStrategy())
    manager.register_retriever(TemporalRetrieverStrategy())
    manager.register_retriever(LayoutSpatialRetrieverStrategy())
    manager.register_retriever(MinHashDuplicateRetrieverStrategy(seed_retriever=bm25))
    manager.register_retriever(MetadataFilterRetrieverStrategy())
    manager.register_retriever(BM25MetadataBoostRetrieverStrategy(base_bm25=bm25))
    manager.register_retriever(MultiQueryBM25RetrieverStrategy(base_bm25=bm25))
    manager.register_retriever(LexicalFusionRetrieverStrategy(bm25=bm25, keyword=keyword))
    manager.register_retriever(AdvancedLexicalFusionRetrieverStrategy(
        retrievers=[bm25, tfidf, keyword, fielded, phrase, char_ngram, boolean_set, positional]
    ))
    manager.register_retriever(MMRDiversityRetrieverStrategy())
    manager.register_retriever(ContextExpansionRetrieverStrategy())
    manager.register_retriever(GraphExpansionRetrieverStrategy())
    manager.register_retriever(CrossChunkStrategyFusionRetriever())
    manager.register_retriever(QueryExpansionBM25RetrieverStrategy(base_bm25=bm25))
    manager.register_retriever(EntityCentricFusionRetrieverStrategy(base_bm25=bm25))
    manager.register_retriever(NumericRangeRetrieverStrategy(base_bm25=bm25))
    manager.register_retriever(NegationAwareBM25RetrieverStrategy(base_bm25=bm25))
    manager.register_retriever(ClinicalSectionScopedRetrieverStrategy(base_bm25=bm25))

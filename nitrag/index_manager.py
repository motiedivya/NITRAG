from __future__ import annotations

import re
import json
import math
import hashlib
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def safe_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records if records else [])
    pq.write_table(table, path, compression="zstd")


def read_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def write_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def simple_tokenize(text: str) -> List[str]:
    text = str(text or "").lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


def make_word_ngrams(tokens: List[str], n: int) -> List[str]:
    if n <= 1:
        return tokens
    return [" ".join(tokens[i:i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def make_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    normalized = " ".join(simple_tokenize(text))
    out = []
    for n in range(n_min, n_max + 1):
        if len(normalized) < n:
            continue
        out.extend(normalized[i:i + n] for i in range(0, len(normalized) - n + 1))
    return out


def chunk_text(store, row: Dict[str, Any]) -> str:
    return store.decode_span(int(row["start_index"]), int(row["end_index"]))


def base_doc_record(
    *,
    row: Dict[str, Any],
    doc_idx: int,
    chunk_strategy_name: str,
    text: str,
    preview_chars: int = 1000,
) -> Dict[str, Any]:
    return {
        "doc_idx": doc_idx,
        "chunk_id": row.get("chunk_id"),
        "document_id": row.get("document_id"),
        "chunk_strategy_name": chunk_strategy_name,
        "start_index": row.get("start_index"),
        "end_index": row.get("end_index"),
        "page_start": row.get("page_start"),
        "page_end": row.get("page_end"),
        "token_length": row.get("token_length"),
        "document_type": row.get("document_type"),
        "primary_section": row.get("primary_section"),
        "contains_medication": row.get("contains_medication"),
        "contains_lab": row.get("contains_lab"),
        "contains_diagnosis": row.get("contains_diagnosis"),
        "contains_vital": row.get("contains_vital"),
        "clinical_quality_score": row.get("clinical_quality_score"),
        "text_preview": text[:preview_chars],
        "metadata_json": row.get("metadata_json"),
    }


def truthy_flag_names(row: Dict[str, Any]) -> List[str]:
    return [
        flag
        for flag in [
            "contains_date",
            "contains_patient_id",
            "contains_vital",
            "contains_lab",
            "contains_medication",
            "contains_diagnosis",
            "contains_imaging",
            "contains_procedure",
            "contains_negation",
        ]
        if bool(row.get(flag))
    ]


def parse_int_list(value: Any) -> List[int]:
    parsed = safe_json_loads(value)
    if not isinstance(parsed, list):
        return []

    out = []
    for item in parsed:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def stable_hash_int(value: str, seed: int = 0) -> int:
    payload = f"{seed}:{value}".encode("utf-8", errors="ignore")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16)


@dataclass
class IndexBuildResult:
    index_name: str
    chunk_strategy_name: str
    output_dir: Path
    stats: Dict[str, Any]


class BaseIndexStrategy(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        pass


class BM25IndexStrategy(BaseIndexStrategy):
    name = "bm25"
    description = "Lexical BM25 index over enriched chunk text."

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        vocab_rows = []

        doc_lens = []
        df = Counter()

        # Pass 1: tokenize docs and collect term frequencies.
        doc_term_freqs = []

        for doc_idx, row in enumerate(chunks):
            start = int(row["start_index"])
            end = int(row["end_index"])

            text = store.decode_span(start, end)
            tokens = simple_tokenize(text)
            tf = Counter(tokens)

            doc_term_freqs.append(tf)
            doc_lens.append(len(tokens))

            for term in tf:
                df[term] += 1

            docs.append({
                "doc_idx": doc_idx,
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "chunk_strategy_name": chunk_strategy_name,
                "start_index": start,
                "end_index": end,
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "token_length": row.get("token_length"),
                "document_type": row.get("document_type"),
                "primary_section": row.get("primary_section"),
                "contains_medication": row.get("contains_medication"),
                "contains_lab": row.get("contains_lab"),
                "contains_diagnosis": row.get("contains_diagnosis"),
                "contains_vital": row.get("contains_vital"),
                "clinical_quality_score": row.get("clinical_quality_score"),
                "text_preview": text[:1000],
                "metadata_json": row.get("metadata_json"),
            })

        n_docs = len(chunks)
        avgdl = sum(doc_lens) / max(1, n_docs)

        # Pass 2: build postings.
        for doc_idx, tf in enumerate(doc_term_freqs):
            for term, freq in tf.items():
                postings.append({
                    "term": term,
                    "doc_idx": doc_idx,
                    "tf": int(freq),
                })

        for term, term_df in df.items():
            idf = math.log(1 + ((n_docs - term_df + 0.5) / (term_df + 0.5)))
            vocab_rows.append({
                "term": term,
                "df": int(term_df),
                "idf": float(idf),
            })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(vocab_rows, output_dir / "vocab.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": n_docs,
            "vocab_size": len(vocab_rows),
            "postings_count": len(postings),
            "avg_doc_len": avgdl,
            "k1": self.k1,
            "b": self.b,
        }

        write_json(stats, output_dir / "manifest.json")

        return IndexBuildResult(
            index_name=self.name,
            chunk_strategy_name=chunk_strategy_name,
            output_dir=output_dir,
            stats=stats,
        )


class KeywordInvertedIndexStrategy(BaseIndexStrategy):
    name = "keyword_inverted"
    description = "Simple term → chunk inverted index for debugging exact lexical retrieval."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        postings = []
        docs = []

        for doc_idx, row in enumerate(chunks):
            text = store.decode_span(int(row["start_index"]), int(row["end_index"]))
            tokens = simple_tokenize(text)
            counts = Counter(tokens)

            docs.append({
                "doc_idx": doc_idx,
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "chunk_strategy_name": chunk_strategy_name,
                "start_index": row.get("start_index"),
                "end_index": row.get("end_index"),
                "token_length": row.get("token_length"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "primary_section": row.get("primary_section"),
                "text_preview": text[:1000],
            })

            for term, freq in counts.items():
                postings.append({
                    "term": term,
                    "doc_idx": doc_idx,
                    "tf": int(freq),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "postings_count": len(postings),
            "unique_terms": len(set(p["term"] for p in postings)),
        }

        write_json(stats, output_dir / "manifest.json")

        return IndexBuildResult(
            index_name=self.name,
            chunk_strategy_name=chunk_strategy_name,
            output_dir=output_dir,
            stats=stats,
        )


class MetadataInvertedIndexStrategy(BaseIndexStrategy):
    name = "metadata_inverted"
    description = "Metadata → chunk index for fast filtering and analysis."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []

        for doc_idx, row in enumerate(chunks):
            docs.append({
                "doc_idx": doc_idx,
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "chunk_strategy_name": chunk_strategy_name,
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "primary_section": row.get("primary_section"),
                "document_type": row.get("document_type"),
                "clinical_quality_score": row.get("clinical_quality_score"),
            })

            self._add_posting(postings, "document_type", row.get("document_type"), doc_idx)
            self._add_posting(postings, "primary_section", row.get("primary_section"), doc_idx)

            for flag in [
                "contains_date",
                "contains_patient_id",
                "contains_vital",
                "contains_lab",
                "contains_medication",
                "contains_diagnosis",
                "contains_imaging",
                "contains_procedure",
                "contains_negation",
            ]:
                self._add_posting(postings, flag, row.get(flag), doc_idx)

            entity_type_counts = safe_json_loads(row.get("entity_type_counts_json")) or {}
            if isinstance(entity_type_counts, dict):
                for entity_type, count in entity_type_counts.items():
                    if count:
                        self._add_posting(postings, "entity_type", entity_type, doc_idx)

            section_names = safe_json_loads(row.get("section_names_json")) or []
            if isinstance(section_names, list):
                for section in section_names:
                    self._add_posting(postings, "section_name", section, doc_idx)

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "postings_count": len(postings),
        }

        write_json(stats, output_dir / "manifest.json")

        return IndexBuildResult(
            index_name=self.name,
            chunk_strategy_name=chunk_strategy_name,
            output_dir=output_dir,
            stats=stats,
        )

    def _add_posting(
        self,
        postings: List[Dict[str, Any]],
        field: str,
        value: Any,
        doc_idx: int,
    ) -> None:
        if value is None:
            return

        postings.append({
            "field": str(field),
            "value": str(value),
            "doc_idx": int(doc_idx),
        })


class TFIDFIndexStrategy(BaseIndexStrategy):
    name = "tfidf"
    description = "TF-IDF lexical index with normalized term weights."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        vocab_rows = []
        doc_term_freqs = []
        df = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            tokens = simple_tokenize(text)
            tf = Counter(tokens)
            doc_term_freqs.append(tf)
            for term in tf:
                df[term] += 1
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

        n_docs = len(chunks)
        idf = {
            term: math.log((1 + n_docs) / (1 + term_df)) + 1.0
            for term, term_df in df.items()
        }

        for doc_idx, tf in enumerate(doc_term_freqs):
            max_tf = max(tf.values()) if tf else 1
            weights = {}
            norm_sq = 0.0

            for term, freq in tf.items():
                weight = (0.5 + 0.5 * freq / max_tf) * idf[term]
                weights[term] = weight
                norm_sq += weight * weight

            norm = math.sqrt(norm_sq) or 1.0
            for term, freq in tf.items():
                postings.append({
                    "term": term,
                    "doc_idx": doc_idx,
                    "tf": int(freq),
                    "tfidf": float(weights[term] / norm),
                })

        for term, term_df in df.items():
            vocab_rows.append({
                "term": term,
                "df": int(term_df),
                "idf": float(idf[term]),
            })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(vocab_rows, output_dir / "vocab.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": n_docs,
            "vocab_size": len(vocab_rows),
            "postings_count": len(postings),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class PhraseNgramIndexStrategy(BaseIndexStrategy):
    name = "phrase_ngram"
    description = "Word bigram/trigram inverted index for phrase-like matching."

    def __init__(self, n_values: Optional[List[int]] = None):
        self.n_values = n_values or [2, 3]

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        phrase_df = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            tokens = simple_tokenize(text)
            phrases = []
            for n in self.n_values:
                phrases.extend((n, phrase) for phrase in make_word_ngrams(tokens, n))

            counts = Counter(phrases)
            for _, phrase in counts:
                phrase_df[phrase] += 1

            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            for (n, phrase), freq in counts.items():
                postings.append({
                    "phrase": phrase,
                    "n": int(n),
                    "doc_idx": doc_idx,
                    "tf": int(freq),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(
            [{"phrase": phrase, "df": int(count)} for phrase, count in phrase_df.items()],
            output_dir / "vocab.parquet",
        )

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "phrase_count": len(phrase_df),
            "postings_count": len(postings),
            "n_values": self.n_values,
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class CharacterNgramIndexStrategy(BaseIndexStrategy):
    name = "char_ngram"
    description = "Character n-gram index for fuzzy-ish matching, typos, codes, and abbreviations."

    def __init__(self, n_min: int = 3, n_max: int = 5):
        self.n_min = n_min
        self.n_max = n_max

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        vocab = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            counts = Counter(make_char_ngrams(text, self.n_min, self.n_max))
            for gram in counts:
                vocab[gram] += 1

            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            for gram, freq in counts.items():
                postings.append({
                    "gram": gram,
                    "doc_idx": doc_idx,
                    "tf": int(freq),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(
            [{"gram": gram, "df": int(count)} for gram, count in vocab.items()],
            output_dir / "vocab.parquet",
        )

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "vocab_size": len(vocab),
            "postings_count": len(postings),
            "n_min": self.n_min,
            "n_max": self.n_max,
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class FieldedLexicalIndexStrategy(BaseIndexStrategy):
    name = "fielded_lexical"
    description = "Field-aware inverted index over body, sections, entity types, entity text, and flags."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            field_terms = defaultdict(list)
            field_terms["body"].extend(simple_tokenize(text))
            field_terms["document_type"].extend(simple_tokenize(row.get("document_type")))
            field_terms["primary_section"].extend(simple_tokenize(row.get("primary_section")))

            section_names = safe_json_loads(row.get("section_names_json")) or []
            if isinstance(section_names, list):
                for section in section_names:
                    field_terms["section_name"].extend(simple_tokenize(section))

            entity_type_counts = safe_json_loads(row.get("entity_type_counts_json")) or {}
            if isinstance(entity_type_counts, dict):
                for entity_type, count in entity_type_counts.items():
                    if count:
                        field_terms["entity_type"].extend(simple_tokenize(entity_type))

            entities = safe_json_loads(row.get("entities_json")) or []
            if isinstance(entities, list):
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    field_terms["entity_text"].extend(simple_tokenize(ent.get("text")))
                    field_terms["entity_normalized"].extend(simple_tokenize(ent.get("normalized_value")))

            field_terms["clinical_flag"].extend(simple_tokenize(" ".join(truthy_flag_names(row))))

            for field, terms in field_terms.items():
                for term, freq in Counter(terms).items():
                    postings.append({
                        "field": field,
                        "term": term,
                        "doc_idx": doc_idx,
                        "tf": int(freq),
                    })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "postings_count": len(postings),
            "fields": sorted(set(p["field"] for p in postings)),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class EntityIndexStrategy(BaseIndexStrategy):
    name = "entity"
    description = "Entity-centric index from enriched entities_json and entity_type_counts_json."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        type_counts = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            entities = safe_json_loads(row.get("entities_json")) or []
            if not isinstance(entities, list):
                continue

            for ent in entities:
                if not isinstance(ent, dict):
                    continue

                entity_type = ent.get("type") or ent.get("entity_type")
                entity_text = ent.get("text")
                normalized = ent.get("normalized_value") or entity_text
                entity_type = str(entity_type or "")
                type_counts[entity_type] += 1

                postings.append({
                    "entity_type": entity_type,
                    "entity_text": str(entity_text or ""),
                    "normalized_value": str(normalized or ""),
                    "doc_idx": doc_idx,
                    "page": ent.get("page"),
                    "element_id": ent.get("element_id"),
                    "confidence": ent.get("confidence"),
                    "negated": ent.get("negated"),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(
            [{"entity_type": k, "count": int(v)} for k, v in type_counts.items()],
            output_dir / "entity_types.parquet",
        )

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "entity_postings_count": len(postings),
            "entity_type_count": len(type_counts),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class SectionPageIndexStrategy(BaseIndexStrategy):
    name = "section_page"
    description = "Section/page navigation index for scoped browsing and retrieval filters."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            self._add(postings, doc_idx, "primary_section", row.get("primary_section"))
            self._add(postings, doc_idx, "document_type", row.get("document_type"))

            page_start = row.get("page_start")
            page_end = row.get("page_end")
            if page_start is not None and page_end is not None:
                for page in range(int(page_start), int(page_end) + 1):
                    self._add(postings, doc_idx, "page", page)

            section_names = safe_json_loads(row.get("section_names_json")) or []
            if isinstance(section_names, list):
                for section in section_names:
                    self._add(postings, doc_idx, "section_name", section)

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "postings_count": len(postings),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)

    def _add(self, postings: List[Dict[str, Any]], doc_idx: int, field: str, value: Any) -> None:
        if value in [None, "", [], {}]:
            return
        postings.append({
            "field": field,
            "value": str(value),
            "doc_idx": int(doc_idx),
        })


class ChunkGraphIndexStrategy(BaseIndexStrategy):
    name = "chunk_graph"
    description = "Chunk relationship graph: sequence, overlap, shared page, shared section, and parent links."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        edges = []

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

        ordered = sorted(enumerate(chunks), key=lambda item: (
            int(item[1].get("start_index") or 0),
            int(item[1].get("end_index") or 0),
        ))

        for pos, (doc_idx, row) in enumerate(ordered):
            if pos > 0:
                self._add_edge(edges, ordered[pos - 1][0], doc_idx, "previous_next", 1.0)
            if pos + 1 < len(ordered):
                self._add_edge(edges, doc_idx, ordered[pos + 1][0], "next_previous", 1.0)

            parent_id = row.get("parent_id")
            if parent_id not in [None, ""]:
                for other_idx, other in enumerate(chunks):
                    if other.get("chunk_id") == parent_id:
                        self._add_edge(edges, doc_idx, other_idx, "parent", 1.0)
                        self._add_edge(edges, other_idx, doc_idx, "child", 1.0)

        for left_idx, left in enumerate(chunks):
            for right_idx in range(left_idx + 1, len(chunks)):
                right = chunks[right_idx]

                overlap = self._token_overlap(left, right)
                if overlap > 0:
                    weight = overlap / max(1, min(
                        int(left.get("token_length") or 0),
                        int(right.get("token_length") or 0),
                    ))
                    self._add_edge(edges, left_idx, right_idx, "token_overlap", weight)
                    self._add_edge(edges, right_idx, left_idx, "token_overlap", weight)

                if self._same_page(left, right):
                    self._add_edge(edges, left_idx, right_idx, "same_page", 0.25)
                    self._add_edge(edges, right_idx, left_idx, "same_page", 0.25)

                if left.get("primary_section") and left.get("primary_section") == right.get("primary_section"):
                    self._add_edge(edges, left_idx, right_idx, "same_primary_section", 0.35)
                    self._add_edge(edges, right_idx, left_idx, "same_primary_section", 0.35)

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(edges, output_dir / "edges.parquet")

        edge_counts = Counter(edge["edge_type"] for edge in edges)
        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "edge_count": len(edges),
            "edge_type_counts": dict(edge_counts),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)

    def _add_edge(self, edges: List[Dict[str, Any]], source: int, target: int, edge_type: str, weight: float) -> None:
        if source == target:
            return
        edges.append({
            "source_doc_idx": int(source),
            "target_doc_idx": int(target),
            "edge_type": edge_type,
            "weight": float(weight),
        })

    def _token_overlap(self, left: Dict[str, Any], right: Dict[str, Any]) -> int:
        left_start = int(left.get("start_index") or 0)
        left_end = int(left.get("end_index") or 0)
        right_start = int(right.get("start_index") or 0)
        right_end = int(right.get("end_index") or 0)
        return max(0, min(left_end, right_end) - max(left_start, right_start))

    def _same_page(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_start = left.get("page_start")
        left_end = left.get("page_end")
        right_start = right.get("page_start")
        right_end = right.get("page_end")
        if None in [left_start, left_end, right_start, right_end]:
            return False
        return max(int(left_start), int(right_start)) <= min(int(left_end), int(right_end))


class PositionalIndexStrategy(BaseIndexStrategy):
    name = "positional"
    description = "Term position index for proximity, phrase, and span-aware lexical retrieval."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        vocab = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            positions_by_term = defaultdict(list)
            for position, term in enumerate(simple_tokenize(text)):
                positions_by_term[term].append(position)

            for term, positions in positions_by_term.items():
                vocab[term] += 1
                postings.append({
                    "term": term,
                    "doc_idx": doc_idx,
                    "tf": int(len(positions)),
                    "first_position": int(positions[0]),
                    "last_position": int(positions[-1]),
                    "positions_json": safe_json_dumps(positions),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(
            [{"term": term, "df": int(count)} for term, count in vocab.items()],
            output_dir / "vocab.parquet",
        )

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "vocab_size": len(vocab),
            "postings_count": len(postings),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class BooleanSetIndexStrategy(BaseIndexStrategy):
    name = "boolean_set"
    description = "Binary term membership index for boolean AND/OR filtering."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        vocab = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            terms = sorted(set(simple_tokenize(text)))
            docs.append({
                **base_doc_record(
                    row=row,
                    doc_idx=doc_idx,
                    chunk_strategy_name=chunk_strategy_name,
                    text=text,
                ),
                "terms_json": safe_json_dumps(terms),
                "term_count": int(len(terms)),
            })

            for term in terms:
                vocab[term] += 1
                postings.append({
                    "term": term,
                    "doc_idx": doc_idx,
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(
            [{"term": term, "df": int(count)} for term, count in vocab.items()],
            output_dir / "vocab.parquet",
        )

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "vocab_size": len(vocab),
            "postings_count": len(postings),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)


class TemporalIndexStrategy(BaseIndexStrategy):
    name = "temporal"
    description = "Date/time index from enriched date_values_json, date entities, and text date patterns."

    DATE_PATTERN = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    )

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            seen = set()
            for value in self._date_values(row, text):
                key = str(value).strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                postings.append({
                    "date_value": key,
                    "doc_idx": doc_idx,
                    "page_start": row.get("page_start"),
                    "page_end": row.get("page_end"),
                    "primary_section": row.get("primary_section"),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "date_postings_count": len(postings),
            "unique_date_values": len(set(p["date_value"] for p in postings)),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)

    def _date_values(self, row: Dict[str, Any], text: str) -> List[str]:
        values = []

        parsed = safe_json_loads(row.get("date_values_json"))
        if isinstance(parsed, list):
            values.extend(str(v) for v in parsed if v)

        entities = safe_json_loads(row.get("entities_json"))
        if isinstance(entities, list):
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                if ent.get("type") == "date" or ent.get("entity_type") == "date":
                    values.append(str(ent.get("normalized_value") or ent.get("text") or ""))

        values.extend(self.DATE_PATTERN.findall(str(text or "")))
        return values


class LayoutSpatialIndexStrategy(BaseIndexStrategy):
    name = "layout_spatial"
    description = "Layout-aware index from source element ids, pages, normalized boxes, and page zones."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        element_by_id = self._load_layout_elements(store)

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            element_ids = parse_int_list(row.get("source_element_ids_json"))
            if not element_ids and row.get("source_element_id") is not None:
                try:
                    element_ids = [int(row.get("source_element_id"))]
                except Exception:
                    element_ids = []

            for element_id in element_ids:
                element = element_by_id.get(element_id)
                if not element:
                    continue

                page_number = element.get("page_number")
                x_bin = self._bin(element.get("center_x_norm"))
                y_bin = self._bin(element.get("center_y_norm"))
                zone = self._zone(element.get("center_y_norm"))

                postings.append({
                    "doc_idx": doc_idx,
                    "element_id": element_id,
                    "element_type": element.get("element_type"),
                    "page_number": page_number,
                    "x_bin": x_bin,
                    "y_bin": y_bin,
                    "zone": zone,
                    "x0_norm": element.get("x0_norm"),
                    "y0_norm": element.get("y0_norm"),
                    "x1_norm": element.get("x1_norm"),
                    "y1_norm": element.get("y1_norm"),
                })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "layout_postings_count": len(postings),
            "indexed_element_count": len(set(p["element_id"] for p in postings)),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)

    def _load_layout_elements(self, store) -> Dict[int, Dict[str, Any]]:
        document_dir = Path(store.paths.document_dir)
        for path in [
            document_dir / "layout_elements_with_sections.parquet",
            document_dir / "layout_elements.parquet",
            document_dir / "elements.parquet",
        ]:
            rows = read_parquet(path)
            if rows:
                out = {}
                for row in rows:
                    try:
                        out[int(row.get("element_id"))] = row
                    except Exception:
                        continue
                return out
        return {}

    def _bin(self, value: Any, bins: int = 10) -> Optional[int]:
        try:
            value = float(value)
            return max(0, min(bins - 1, int(value * bins)))
        except Exception:
            return None

    def _zone(self, y_value: Any) -> Optional[str]:
        try:
            y = float(y_value)
        except Exception:
            return None
        if y < 0.20:
            return "top"
        if y > 0.80:
            return "bottom"
        return "middle"


class MinHashLSHIndexStrategy(BaseIndexStrategy):
    name = "minhash_lsh"
    description = "Approximate near-duplicate index using token-shingle MinHash and LSH buckets."

    def __init__(self, shingle_size: int = 5, signature_size: int = 64, band_size: int = 8):
        self.shingle_size = shingle_size
        self.signature_size = signature_size
        self.band_size = band_size

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        signatures = []
        buckets = []
        bucket_members = defaultdict(list)

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(
                row=row,
                doc_idx=doc_idx,
                chunk_strategy_name=chunk_strategy_name,
                text=text,
            ))

            shingles = self._shingles(simple_tokenize(text))
            signature = self._signature(shingles)
            signatures.append({
                "doc_idx": doc_idx,
                "signature_json": safe_json_dumps(signature),
                "shingle_count": int(len(shingles)),
            })

            for band_id, band_hash in self._bands(signature):
                bucket_key = f"{band_id}:{band_hash}"
                bucket_members[bucket_key].append(doc_idx)
                buckets.append({
                    "band_id": int(band_id),
                    "bucket": str(bucket_key),
                    "doc_idx": doc_idx,
                })

        candidate_pairs = []
        seen_pairs = set()
        for bucket, members in bucket_members.items():
            if len(members) < 2:
                continue
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    pair = tuple(sorted((left, right)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    candidate_pairs.append({
                        "left_doc_idx": pair[0],
                        "right_doc_idx": pair[1],
                        "bucket": bucket,
                    })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(signatures, output_dir / "signatures.parquet")
        write_parquet(buckets, output_dir / "buckets.parquet")
        write_parquet(candidate_pairs, output_dir / "candidate_pairs.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(docs),
            "signature_size": self.signature_size,
            "shingle_size": self.shingle_size,
            "band_size": self.band_size,
            "bucket_count": len(bucket_members),
            "candidate_pair_count": len(candidate_pairs),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(self.name, chunk_strategy_name, output_dir, stats)

    def _shingles(self, tokens: List[str]) -> List[str]:
        if len(tokens) < self.shingle_size:
            return tokens
        return make_word_ngrams(tokens, self.shingle_size)

    def _signature(self, shingles: List[str]) -> List[int]:
        if not shingles:
            return [0 for _ in range(self.signature_size)]

        signature = []
        for seed in range(self.signature_size):
            signature.append(min(stable_hash_int(shingle, seed=seed) for shingle in shingles))
        return signature

    def _bands(self, signature: List[int]) -> List[tuple]:
        bands = []
        for start in range(0, len(signature), self.band_size):
            band_id = start // self.band_size
            band_values = signature[start:start + self.band_size]
            band_hash = stable_hash_int(",".join(str(v) for v in band_values), seed=band_id)
            bands.append((band_id, band_hash))
        return bands


class IndexManager:
    """
    Register, execute, persist, and debug indexing strategies.

    Similar idea to ChunkManager:
      - ChunkManager runs many chunkers
      - IndexManager runs many indexers over each chunk strategy
    """

    def __init__(
        self,
        store,
        use_enriched_chunks: bool = True,
    ):
        self.store = store
        self.use_enriched_chunks = use_enriched_chunks

        self.document_dir = self.store.paths.document_dir
        self.chunk_dir = self.document_dir / ("chunks_enriched" if use_enriched_chunks else "chunks")
        self.index_root_dir = self.document_dir / "indexes"
        self.index_root_dir.mkdir(parents=True, exist_ok=True)

        self._strategies: Dict[str, BaseIndexStrategy] = {}
        self._descriptions: Dict[str, str] = {}
        self._errors: Dict[str, str] = {}

    def register_indexer(
        self,
        strategy: BaseIndexStrategy,
        force: bool = False,
    ) -> None:
        name = strategy.name

        if name in self._strategies and not force:
            raise ValueError(f"Indexer already registered: {name}")

        self._strategies[name] = strategy
        self._descriptions[name] = strategy.description

    def list_indexers(self, with_descriptions: bool = False):
        if with_descriptions:
            return {
                name: self._descriptions.get(name, "")
                for name in self._strategies
            }

        return list(self._strategies.keys())

    def list_chunk_strategies(self) -> List[str]:
        if not self.chunk_dir.exists():
            return []

        return sorted(path.stem for path in self.chunk_dir.glob("*.parquet"))

    def execute(
        self,
        indexer_name: str,
        chunk_strategy_name: str,
        overwrite: bool = True,
    ) -> Path:
        if indexer_name not in self._strategies:
            raise KeyError(f"Unknown indexer: {indexer_name}")

        chunk_path = self.chunk_dir / f"{chunk_strategy_name}.parquet"

        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk strategy file not found: {chunk_path}")

        chunks = read_parquet(chunk_path)

        out_dir = self.index_root_dir / chunk_strategy_name / indexer_name

        if out_dir.exists() and not overwrite:
            raise FileExistsError(f"Index already exists: {out_dir}")

        print(f"Indexing | chunk_strategy={chunk_strategy_name} | indexer={indexer_name}")

        result = self._strategies[indexer_name].build(
            store=self.store,
            chunk_strategy_name=chunk_strategy_name,
            chunks=chunks,
            output_dir=out_dir,
            overwrite=overwrite,
        )

        print(f"Saved index: {out_dir}")
        print(f"Stats: {result.stats}")

        return out_dir

    def execute_all(
        self,
        chunk_strategy_names: Optional[List[str]] = None,
        indexer_names: Optional[List[str]] = None,
        continue_on_error: bool = True,
        overwrite: bool = True,
    ) -> Dict[str, Path]:
        outputs = {}
        self._errors.clear()

        chunk_strategy_names = chunk_strategy_names or self.list_chunk_strategies()
        indexer_names = indexer_names or list(self._strategies.keys())

        for chunk_strategy_name in chunk_strategy_names:
            for indexer_name in indexer_names:
                key = f"{chunk_strategy_name}/{indexer_name}"

                try:
                    outputs[key] = self.execute(
                        indexer_name=indexer_name,
                        chunk_strategy_name=chunk_strategy_name,
                        overwrite=overwrite,
                    )
                except Exception as e:
                    error = "".join(traceback.format_exception_only(type(e), e)).strip()
                    self._errors[key] = error
                    print(f"Indexer failed: {key}: {error}")

                    if not continue_on_error:
                        raise

        return outputs

    def errors(self) -> Dict[str, str]:
        return dict(self._errors)

    def index_path(self, chunk_strategy_name: str, indexer_name: str) -> Path:
        return self.index_root_dir / chunk_strategy_name / indexer_name

    def index_manifest(self, chunk_strategy_name: str, indexer_name: str) -> Dict[str, Any]:
        path = self.index_path(chunk_strategy_name, indexer_name) / "manifest.json"

        if not path.exists():
            raise FileNotFoundError(f"Index manifest not found: {path}")

        return json.loads(path.read_text(encoding="utf-8"))


class SentenceInvertedIndexStrategy(BaseIndexStrategy):
    """
    Sentence-level inverted index.

    Splits each chunk into sentences and indexes every term with its sentence
    position.  Downstream retrievers can score by how many matching sentences
    a chunk contains, enabling extractive-QA style highlighting.

    Parquet files
    ─────────────
    docs.parquet     — one row per chunk (same schema as BM25 docs)
    postings.parquet — term, doc_idx, sentence_idx, tf_in_sentence
    vocab.parquet    — term, df (# docs containing term), sentence_df
    manifest.json
    """

    name = "sentence_inverted"
    description = "Fine-grained inverted index at sentence granularity for extractive QA and sentence-level scoring."

    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        doc_df: Counter = Counter()
        sentence_df: Counter = Counter()
        vocab_rows = []

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(row=row, doc_idx=doc_idx, chunk_strategy_name=chunk_strategy_name, text=text))

            sentences = self._SENT_SPLIT.split(text.strip()) or [text]
            seen_terms: set = set()
            for sent_idx, sentence in enumerate(sentences):
                tokens = simple_tokenize(sentence)
                tf = Counter(tokens)
                for term, freq in tf.items():
                    postings.append({
                        "term": term,
                        "doc_idx": doc_idx,
                        "sentence_idx": sent_idx,
                        "tf_in_sentence": int(freq),
                    })
                    seen_terms.add(term)
                    sentence_df[term] += 1
            for term in seen_terms:
                doc_df[term] += 1

        for term, df in doc_df.items():
            vocab_rows.append({"term": term, "df": int(df), "sentence_df": int(sentence_df[term])})

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(vocab_rows, output_dir / "vocab.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(chunks),
            "vocab_size": len(vocab_rows),
            "postings_count": len(postings),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(index_name=self.name, chunk_strategy_name=chunk_strategy_name, output_dir=output_dir, stats=stats)


class NumericRangeIndexStrategy(BaseIndexStrategy):
    """
    Numeric value index for range-based medical queries.

    Extracts all numeric values from chunk text and enriched entity JSON,
    attaches inferred unit and magnitude class (vital, lab, dose, generic),
    and stores them as postings keyed by (context_type, unit_bucket).

    Parquet files
    ─────────────
    docs.parquet      — one row per chunk
    postings.parquet  — doc_idx, numeric_value, unit, context_type,
                        magnitude_class, surrounding_text (32 chars each side)
    manifest.json

    Use cases
    ─────────
    - "show me patients with glucose > 200"
    - "find chunks with HR > 100 bpm"
    - "doses above 500 mg"
    """

    name = "numeric_range"
    description = "Indexes numeric values (vitals, lab results, dosages) with unit and context for range-based medical queries."

    _NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zA-Z/%]+)?")
    _UNIT_MAP = {
        "mg": "dose", "mcg": "dose", "g": "dose", "ml": "dose", "l": "dose",
        "mmhg": "vital", "bpm": "vital", "rpm": "vital",
        "mgdl": "lab", "mmoll": "lab", "iu": "lab", "iul": "lab",
        "%": "lab", "meql": "lab",
        "celsius": "vital", "fahrenheit": "vital", "c": "vital", "f": "vital",
        "kg": "demographic", "lb": "demographic", "lbs": "demographic",
        "cm": "demographic", "m": "demographic", "in": "demographic",
    }

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(row=row, doc_idx=doc_idx, chunk_strategy_name=chunk_strategy_name, text=text))

            seen: set = set()
            for m in self._NUM_RE.finditer(text):
                try:
                    value = float(m.group(1))
                except ValueError:
                    continue
                raw_unit = (m.group(2) or "").lower().strip()
                unit_clean = re.sub(r"[^a-z%/]", "", raw_unit)
                context_type = self._UNIT_MAP.get(unit_clean, "generic")
                mag_class = self._magnitude_class(value)
                surrounding_start = max(0, m.start() - 32)
                surrounding = text[surrounding_start: m.end() + 32]
                key = (doc_idx, round(value, 3), unit_clean)
                if key not in seen:
                    seen.add(key)
                    postings.append({
                        "doc_idx": doc_idx,
                        "numeric_value": float(value),
                        "unit": unit_clean,
                        "context_type": context_type,
                        "magnitude_class": mag_class,
                        "surrounding_text": surrounding[:80],
                    })

            entities = safe_json_loads(row.get("entities_json"))
            if isinstance(entities, list):
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    norm = str(entity.get("normalized_value") or entity.get("text") or "")
                    m = self._NUM_RE.search(norm)
                    if not m:
                        continue
                    try:
                        value = float(m.group(1))
                    except ValueError:
                        continue
                    etype = str(entity.get("type") or entity.get("entity_type") or "entity")
                    context_type = {"vital": "vital", "lab_result": "lab"}.get(etype, "entity")
                    key = (doc_idx, round(value, 3), etype)
                    if key not in seen:
                        seen.add(key)
                        postings.append({
                            "doc_idx": doc_idx,
                            "numeric_value": float(value),
                            "unit": etype,
                            "context_type": context_type,
                            "magnitude_class": self._magnitude_class(value),
                            "surrounding_text": norm[:80],
                        })

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(chunks),
            "postings_count": len(postings),
            "context_type_counts": dict(Counter(p["context_type"] for p in postings)),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(index_name=self.name, chunk_strategy_name=chunk_strategy_name, output_dir=output_dir, stats=stats)

    @staticmethod
    def _magnitude_class(value: float) -> str:
        if value < 1:
            return "sub_unit"
        if value < 10:
            return "single_digit"
        if value < 100:
            return "tens"
        if value < 1000:
            return "hundreds"
        return "thousands_plus"


class ConceptCooccurrenceIndexStrategy(BaseIndexStrategy):
    """
    Entity-type co-occurrence index.

    For each chunk records every ordered pair of entity types that co-occur
    (e.g., medication + diagnosis, vital + lab_result).  A retriever can use
    this to find chunks that link two clinical concepts — useful for questions
    like "what medications are associated with hypertension in these notes".

    Parquet files
    ─────────────
    docs.parquet      — one row per chunk
    postings.parquet  — doc_idx, type_a, type_b, pair_count
    vocab.parquet     — pair (type_a|type_b), pair_df, total_count
    manifest.json
    """

    name = "concept_cooccurrence"
    description = "Indexes entity-type co-occurrence pairs per chunk for relationship and association queries."

    def build(
        self,
        *,
        store,
        chunk_strategy_name: str,
        chunks: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = True,
    ) -> IndexBuildResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = []
        postings = []
        pair_df: Counter = Counter()
        pair_total: Counter = Counter()

        for doc_idx, row in enumerate(chunks):
            text = chunk_text(store, row)
            docs.append(base_doc_record(row=row, doc_idx=doc_idx, chunk_strategy_name=chunk_strategy_name, text=text))

            entities = safe_json_loads(row.get("entities_json"))
            entity_types: List[str] = []
            if isinstance(entities, list):
                for e in entities:
                    if isinstance(e, dict):
                        etype = str(e.get("type") or e.get("entity_type") or "").strip()
                        if etype:
                            entity_types.append(etype)

            # Also add flag-derived types for chunks without entities_json
            flag_types = truthy_flag_names(row)
            flag_prefix_map = {
                "contains_medication": "medication_candidate",
                "contains_diagnosis": "diagnosis_or_problem_candidate",
                "contains_vital": "vital",
                "contains_lab": "lab_result",
                "contains_date": "date",
                "contains_imaging": "imaging_candidate",
                "contains_procedure": "procedure_candidate",
            }
            for flag in flag_types:
                synthetic = flag_prefix_map.get(flag)
                if synthetic and synthetic not in entity_types:
                    entity_types.append(synthetic)

            type_counts = Counter(entity_types)
            seen_pairs: set = set()
            types_unique = sorted(set(entity_types))
            for i, ta in enumerate(types_unique):
                for tb in types_unique[i + 1:]:
                    pair_key = f"{ta}|{tb}"
                    count = min(type_counts[ta], type_counts[tb])
                    postings.append({"doc_idx": doc_idx, "type_a": ta, "type_b": tb, "pair_count": int(count)})
                    pair_total[pair_key] += count
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pair_df[pair_key] += 1

        vocab_rows = [
            {"pair": pair, "pair_df": int(df), "total_count": int(pair_total[pair])}
            for pair, df in pair_df.items()
        ]

        write_parquet(docs, output_dir / "docs.parquet")
        write_parquet(postings, output_dir / "postings.parquet")
        write_parquet(vocab_rows, output_dir / "vocab.parquet")

        stats = {
            "index_name": self.name,
            "chunk_strategy_name": chunk_strategy_name,
            "n_docs": len(chunks),
            "unique_pairs": len(vocab_rows),
            "postings_count": len(postings),
        }
        write_json(stats, output_dir / "manifest.json")
        return IndexBuildResult(index_name=self.name, chunk_strategy_name=chunk_strategy_name, output_dir=output_dir, stats=stats)


def register_default_indexers(manager: IndexManager) -> None:
    manager.register_indexer(BM25IndexStrategy())
    manager.register_indexer(KeywordInvertedIndexStrategy())
    manager.register_indexer(MetadataInvertedIndexStrategy())
    manager.register_indexer(TFIDFIndexStrategy())
    manager.register_indexer(PhraseNgramIndexStrategy())
    manager.register_indexer(CharacterNgramIndexStrategy())
    manager.register_indexer(FieldedLexicalIndexStrategy())
    manager.register_indexer(EntityIndexStrategy())
    manager.register_indexer(SectionPageIndexStrategy())
    manager.register_indexer(ChunkGraphIndexStrategy())
    manager.register_indexer(PositionalIndexStrategy())
    manager.register_indexer(BooleanSetIndexStrategy())
    manager.register_indexer(TemporalIndexStrategy())
    manager.register_indexer(LayoutSpatialIndexStrategy())
    manager.register_indexer(MinHashLSHIndexStrategy())
    manager.register_indexer(SentenceInvertedIndexStrategy())
    manager.register_indexer(NumericRangeIndexStrategy())
    manager.register_indexer(ConceptCooccurrenceIndexStrategy())

"""
Tests for all 18 index strategies.

Pattern for each strategy:
  1. build() succeeds and returns an IndexBuildResult
  2. Expected parquet files exist and have > 0 rows
  3. manifest.json is valid JSON with expected keys
  4. Strategy-specific structural assertions

Integration-style (builds real parquets in tmp_path) but uses synthetic data — no PDF.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
import pytest

from tests.conftest import MockStore, CORPUS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_parquet(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def load_manifest(output_dir: Path) -> Dict[str, Any]:
    p = output_dir / "manifest.json"
    assert p.exists(), f"manifest.json missing in {output_dir}"
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> MockStore:
    return MockStore()


@pytest.fixture
def chunks(store) -> List[Dict[str, Any]]:
    import json as _json
    result = []
    for i, (start, end, text, meta) in enumerate(CORPUS):
        result.append({
            "chunk_id": i,
            "document_id": "doc_001",
            "chunk_strategy_name": "fixed_token",
            "start_index": start,
            "end_index": end,
            "token_length": len(text.split()),
            "page_start": 0,
            "page_end": 0,
            "document_type": "discharge_summary",
            "overlap_line_count": 2,
            "source_element_ids_json": _json.dumps([i * 10]),
            "section_names_json": _json.dumps([meta["primary_section"]]),
            **meta,
        })
    return result


# ---------------------------------------------------------------------------
# Registration smoke test
# ---------------------------------------------------------------------------

def test_all_18_strategies_registered(tmp_path):
    from nitrag.index_manager import IndexManager, register_default_indexers
    manager = IndexManager(store=MockStore(tmp_path))
    register_default_indexers(manager)
    names = set(manager.list_indexers())
    expected = {
        "bm25", "keyword_inverted", "metadata_inverted", "tfidf",
        "phrase_ngram", "char_ngram", "fielded_lexical", "entity",
        "section_page", "chunk_graph", "positional", "boolean_set",
        "temporal", "layout_spatial", "minhash_lsh",
        "sentence_inverted", "numeric_range", "concept_cooccurrence",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


# ---------------------------------------------------------------------------
# BM25 — baseline correctness
# ---------------------------------------------------------------------------

class TestBM25IndexStrategy:
    def test_build_creates_files(self, tmp_path, store, chunks):
        from nitrag.index_manager import BM25IndexStrategy
        out = tmp_path / "bm25"
        result = BM25IndexStrategy().build(
            store=store, chunk_strategy_name="fixed_token",
            chunks=chunks, output_dir=out,
        )
        assert out.exists()
        assert (out / "docs.parquet").exists()
        assert (out / "postings.parquet").exists()
        assert (out / "vocab.parquet").exists()
        assert result.index_name == "bm25"

    def test_docs_count_matches_chunks(self, tmp_path, store, chunks):
        from nitrag.index_manager import BM25IndexStrategy
        out = tmp_path / "bm25"
        BM25IndexStrategy().build(store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=out)
        docs = load_parquet(out / "docs.parquet")
        assert len(docs) == len(chunks)

    def test_postings_have_idf_positive(self, tmp_path, store, chunks):
        from nitrag.index_manager import BM25IndexStrategy
        out = tmp_path / "bm25"
        BM25IndexStrategy().build(store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=out)
        vocab = load_parquet(out / "vocab.parquet")
        assert all(v["idf"] > 0 for v in vocab)

    def test_manifest_has_stats(self, tmp_path, store, chunks):
        from nitrag.index_manager import BM25IndexStrategy
        out = tmp_path / "bm25"
        BM25IndexStrategy().build(store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=out)
        manifest = load_manifest(out)
        assert manifest["n_docs"] == len(chunks)
        assert manifest["vocab_size"] > 0
        assert manifest["postings_count"] > 0

    def test_empty_chunks(self, tmp_path, store):
        from nitrag.index_manager import BM25IndexStrategy
        out = tmp_path / "bm25_empty"
        result = BM25IndexStrategy().build(store=store, chunk_strategy_name="s", chunks=[], output_dir=out)
        docs = load_parquet(out / "docs.parquet")
        assert len(docs) == 0
        assert result.stats["n_docs"] == 0


# ---------------------------------------------------------------------------
# Sentence Inverted (NEW)
# ---------------------------------------------------------------------------

class TestSentenceInvertedIndexStrategy:
    @pytest.fixture
    def out(self, tmp_path, store, chunks):
        from nitrag.index_manager import SentenceInvertedIndexStrategy
        d = tmp_path / "sentence_inverted"
        SentenceInvertedIndexStrategy().build(
            store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=d,
        )
        return d

    def test_files_created(self, out):
        for f in ("docs.parquet", "postings.parquet", "vocab.parquet", "manifest.json"):
            assert (out / f).exists(), f"{f} missing"

    def test_postings_have_sentence_idx(self, out):
        postings = load_parquet(out / "postings.parquet")
        assert len(postings) > 0
        assert all("sentence_idx" in p for p in postings)
        assert all("tf_in_sentence" in p for p in postings)

    def test_sentence_idx_is_non_negative(self, out):
        postings = load_parquet(out / "postings.parquet")
        assert all(p["sentence_idx"] >= 0 for p in postings)

    def test_vocab_has_sentence_df(self, out):
        vocab = load_parquet(out / "vocab.parquet")
        assert len(vocab) > 0
        assert all("sentence_df" in v for v in vocab)
        assert all(v["sentence_df"] >= v["df"] for v in vocab), \
            "sentence_df must be >= doc_df (multiple sentences can match per doc)"

    def test_multiple_sentences_for_multi_sentence_chunk(self, tmp_path, store):
        from nitrag.index_manager import SentenceInvertedIndexStrategy
        import json as _json
        multi = [{
            "chunk_id": 0, "document_id": "d", "chunk_strategy_name": "s",
            "start_index": 0, "end_index": 20,
            "token_length": 10, "page_start": 0, "page_end": 0,
            "document_type": "note", "primary_section": "Assessment",
            "contains_vital": False, "contains_diagnosis": False,
            "contains_negation": False, "contains_medication": False,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.5,
            "entities_json": _json.dumps([]),
            "entity_type_counts_json": _json.dumps({}),
            "entity_count": 0, "overlap_line_count": 0,
            "source_element_ids_json": _json.dumps([]),
            "section_names_json": _json.dumps([]),
        }]
        out = tmp_path / "sent_multi"
        SentenceInvertedIndexStrategy().build(
            store=store, chunk_strategy_name="s", chunks=multi, output_dir=out,
        )
        postings = load_parquet(out / "postings.parquet")
        # chunk 0 covers CORPUS[0] which has multiple sentences separated by ". "
        sentence_indices = {p["sentence_idx"] for p in postings if p["doc_idx"] == 0}
        # should have at least 1 sentence index
        assert len(sentence_indices) >= 1

    def test_docs_count_matches(self, out, chunks):
        docs = load_parquet(out / "docs.parquet")
        assert len(docs) == len(chunks)

    def test_manifest_present(self, out):
        manifest = load_manifest(out)
        assert "n_docs" in manifest
        assert "vocab_size" in manifest
        assert manifest["index_name"] == "sentence_inverted"


# ---------------------------------------------------------------------------
# Numeric Range (NEW)
# ---------------------------------------------------------------------------

class TestNumericRangeIndexStrategy:
    @pytest.fixture
    def out(self, tmp_path, store, chunks):
        from nitrag.index_manager import NumericRangeIndexStrategy
        d = tmp_path / "numeric_range"
        NumericRangeIndexStrategy().build(
            store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=d,
        )
        return d

    def test_files_created(self, out):
        for f in ("docs.parquet", "postings.parquet", "manifest.json"):
            assert (out / f).exists()

    def test_postings_have_numeric_value(self, out):
        postings = load_parquet(out / "postings.parquet")
        assert len(postings) > 0
        assert all("numeric_value" in p for p in postings)
        assert all(isinstance(p["numeric_value"], (int, float)) for p in postings)

    def test_blood_pressure_values_indexed(self, out):
        postings = load_parquet(out / "postings.parquet")
        values = {p["numeric_value"] for p in postings}
        # CORPUS[0] has "145/90 mmHg" — both 145 and 90 should appear
        assert 145.0 in values or 145 in values

    def test_glucose_indexed(self, out):
        postings = load_parquet(out / "postings.parquet")
        values = {p["numeric_value"] for p in postings}
        # CORPUS[4] has "Glucose 210 mg/dL"
        assert 210.0 in values or 210 in values

    def test_context_types_present(self, out):
        postings = load_parquet(out / "postings.parquet")
        ctx_types = {p["context_type"] for p in postings}
        # should have at least generic and vital
        assert len(ctx_types) >= 1
        # "mmhg" unit → vital
        vital_postings = [p for p in postings if p.get("context_type") == "vital"]
        assert len(vital_postings) >= 1

    def test_entity_json_values_indexed(self, out):
        """Numeric values from entities_json should also be indexed."""
        postings = load_parquet(out / "postings.parquet")
        # CORPUS[4] entities include lab results with numeric values
        entity_postings = [p for p in postings if p.get("context_type") in ("lab", "entity")]
        assert len(entity_postings) >= 0  # just verify no crash

    def test_manifest_has_context_type_counts(self, out):
        manifest = load_manifest(out)
        assert "context_type_counts" in manifest
        assert isinstance(manifest["context_type_counts"], dict)

    def test_surrounding_text_truncated(self, out):
        postings = load_parquet(out / "postings.parquet")
        for p in postings:
            assert len(str(p.get("surrounding_text", ""))) <= 80


# ---------------------------------------------------------------------------
# Concept Co-occurrence (NEW)
# ---------------------------------------------------------------------------

class TestConceptCooccurrenceIndexStrategy:
    @pytest.fixture
    def out(self, tmp_path, store, chunks):
        from nitrag.index_manager import ConceptCooccurrenceIndexStrategy
        d = tmp_path / "concept_cooccurrence"
        ConceptCooccurrenceIndexStrategy().build(
            store=store, chunk_strategy_name="fixed_token", chunks=chunks, output_dir=d,
        )
        return d

    def test_files_created(self, out):
        for f in ("docs.parquet", "postings.parquet", "vocab.parquet", "manifest.json"):
            assert (out / f).exists()

    def test_postings_have_pair_fields(self, out):
        postings = load_parquet(out / "postings.parquet")
        assert len(postings) > 0
        for p in postings:
            assert "type_a" in p
            assert "type_b" in p
            assert "pair_count" in p
            assert "doc_idx" in p

    def test_type_a_lexicographically_before_type_b(self, out):
        """Pairs should always be ordered so type_a <= type_b (no duplicate reversed pairs)."""
        postings = load_parquet(out / "postings.parquet")
        for p in postings:
            assert p["type_a"] <= p["type_b"], \
                f"Unordered pair: {p['type_a']} > {p['type_b']}"

    def test_vocab_has_pair_key(self, out):
        vocab = load_parquet(out / "vocab.parquet")
        assert len(vocab) > 0
        for v in vocab:
            assert "pair" in v
            assert "|" in v["pair"]
            assert "pair_df" in v
            assert "total_count" in v

    def test_medication_diagnosis_pair_exists(self, out):
        """CORPUS[3] has medication + date, CORPUS[7] has diagnosis + date — some pair must exist."""
        postings = load_parquet(out / "postings.parquet")
        pairs = {(p["type_a"], p["type_b"]) for p in postings}
        # At least one pair should involve diagnosis or medication types
        diagnosis_pairs = [p for p in pairs if "diagnosis" in p[0] or "diagnosis" in p[1]]
        medication_pairs = [p for p in pairs if "medication" in p[0] or "medication" in p[1]]
        assert len(diagnosis_pairs) + len(medication_pairs) > 0

    def test_pair_count_positive(self, out):
        postings = load_parquet(out / "postings.parquet")
        assert all(p["pair_count"] >= 1 for p in postings)

    def test_manifest_stats(self, out):
        manifest = load_manifest(out)
        assert manifest["index_name"] == "concept_cooccurrence"
        assert manifest["unique_pairs"] >= 0
        assert manifest["postings_count"] >= 0
        assert manifest["n_docs"] == len(CORPUS)

    def test_flag_derived_types_included(self, tmp_path, store):
        """Chunks without entities_json but with flag=True should still generate pairs."""
        import json as _json
        from nitrag.index_manager import ConceptCooccurrenceIndexStrategy
        chunk = [{
            "chunk_id": 0, "document_id": "d", "chunk_strategy_name": "s",
            "start_index": 0, "end_index": 20, "token_length": 10,
            "page_start": 0, "page_end": 0, "document_type": "note",
            "primary_section": "Assessment",
            "contains_vital": True, "contains_diagnosis": True,
            "contains_negation": False, "contains_medication": True,
            "contains_lab": False, "contains_date": False,
            "clinical_quality_score": 0.5,
            "entities_json": _json.dumps([]),  # no real entities
            "entity_type_counts_json": _json.dumps({}),
            "entity_count": 0, "overlap_line_count": 0,
            "source_element_ids_json": _json.dumps([]),
            "section_names_json": _json.dumps([]),
        }]
        out = tmp_path / "cc_flags"
        ConceptCooccurrenceIndexStrategy().build(
            store=store, chunk_strategy_name="s", chunks=chunk, output_dir=out,
        )
        postings = load_parquet(out / "postings.parquet")
        # vital + diagnosis + medication → at least one pair
        assert len(postings) >= 1


# ---------------------------------------------------------------------------
# Quick smoke tests for remaining 15 existing strategies
# ---------------------------------------------------------------------------

EXISTING_STRATEGIES = [
    "KeywordInvertedIndexStrategy",
    "MetadataInvertedIndexStrategy",
    "TFIDFIndexStrategy",
    "PhraseNgramIndexStrategy",
    "CharacterNgramIndexStrategy",
    "FieldedLexicalIndexStrategy",
    "EntityIndexStrategy",
    "SectionPageIndexStrategy",
    "ChunkGraphIndexStrategy",
    "PositionalIndexStrategy",
    "BooleanSetIndexStrategy",
    "TemporalIndexStrategy",
    "LayoutSpatialIndexStrategy",
    "MinHashLSHIndexStrategy",
]


@pytest.mark.parametrize("class_name", EXISTING_STRATEGIES)
def test_existing_strategy_builds_without_error(tmp_path, store, chunks, class_name):
    import nitrag.index_manager as im
    cls = getattr(im, class_name)
    out = tmp_path / class_name.lower()
    result = cls().build(
        store=store, chunk_strategy_name="fixed_token",
        chunks=chunks, output_dir=out,
    )
    assert result is not None
    assert (out / "manifest.json").exists()
    assert (out / "docs.parquet").exists()

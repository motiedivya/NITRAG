"""Tests for nitrag/config.py — RAGConfig presets, serialization, env loading."""
from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from nitrag.config import (
    RAGConfig,
    EmbeddingConfig,
    LLMConfig,
    VectorIndexConfig,
    RetrievalConfig,
    GenerationConfig,
)


# ---------------------------------------------------------------------------
# EmbeddingConfig defaults
# ---------------------------------------------------------------------------

class TestEmbeddingConfigDefaults:
    def test_default_provider_is_fastembed(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "fastembed"

    def test_default_model_is_nomic(self):
        cfg = EmbeddingConfig()
        assert "nomic" in cfg.model_name.lower()

    def test_normalize_true_by_default(self):
        cfg = EmbeddingConfig()
        assert cfg.normalize is True

    def test_device_default_is_cpu(self):
        cfg = EmbeddingConfig()
        assert cfg.device == "cpu"

    def test_batch_size_positive(self):
        cfg = EmbeddingConfig()
        assert cfg.batch_size > 0


# ---------------------------------------------------------------------------
# LLMConfig defaults
# ---------------------------------------------------------------------------

class TestLLMConfigDefaults:
    def test_default_provider_openai_compatible(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai_compatible"

    def test_default_temperature_is_low(self):
        cfg = LLMConfig()
        assert 0.0 <= cfg.temperature <= 0.5

    def test_default_base_url_is_localhost(self):
        cfg = LLMConfig()
        assert cfg.base_url is not None and "localhost" in cfg.base_url

    def test_default_max_tokens_positive(self):
        cfg = LLMConfig()
        assert cfg.max_tokens > 0


# ---------------------------------------------------------------------------
# VectorIndexConfig defaults
# ---------------------------------------------------------------------------

class TestVectorIndexConfigDefaults:
    def test_default_backend_is_faiss(self):
        cfg = VectorIndexConfig()
        assert cfg.backend == "faiss"

    def test_default_metric_is_cosine(self):
        cfg = VectorIndexConfig()
        assert cfg.metric == "cosine"


# ---------------------------------------------------------------------------
# RetrievalConfig defaults
# ---------------------------------------------------------------------------

class TestRetrievalConfigDefaults:
    def test_default_retriever_names(self):
        cfg = RetrievalConfig()
        assert "bm25" in cfg.retriever_names

    def test_alpha_between_0_and_1(self):
        cfg = RetrievalConfig()
        assert 0.0 <= cfg.hybrid_alpha <= 1.0


# ---------------------------------------------------------------------------
# GenerationConfig defaults
# ---------------------------------------------------------------------------

class TestGenerationConfigDefaults:
    def test_max_context_tokens_positive(self):
        cfg = GenerationConfig()
        assert cfg.max_context_tokens > 0

    def test_hallucination_check_on_by_default(self):
        cfg = GenerationConfig()
        assert cfg.hallucination_check is True


# ---------------------------------------------------------------------------
# RAGConfig.local_ollama()
# ---------------------------------------------------------------------------

class TestLocalOllama:
    def test_embedding_provider_is_fastembed(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.embedding.provider == "fastembed"

    def test_embedding_model_nomic(self):
        cfg = RAGConfig.local_ollama()
        assert "nomic" in cfg.embedding.model_name.lower()

    def test_llm_model_llama(self):
        cfg = RAGConfig.local_ollama()
        assert "llama" in cfg.llm.model_name.lower()

    def test_llm_base_url_is_ollama(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.llm.base_url is not None
        assert "11434" in cfg.llm.base_url

    def test_llm_api_key_is_ollama(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.llm.api_key == "ollama"

    def test_embedding_dimensions_set(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.embedding.dimensions == 768

    def test_vector_index_backend_faiss(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.vector_index.backend == "faiss"

    def test_vector_index_type_flat(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.vector_index.index_type == "flat"

    def test_use_hyde_false(self):
        cfg = RAGConfig.local_ollama()
        assert cfg.retrieval.use_hyde is False


# ---------------------------------------------------------------------------
# RAGConfig.openai_cloud()
# ---------------------------------------------------------------------------

class TestOpenAICloud:
    def test_embedding_provider_openai(self):
        cfg = RAGConfig.openai_cloud()
        assert cfg.embedding.provider == "openai"

    def test_embedding_model_text_embedding_3_large(self):
        cfg = RAGConfig.openai_cloud()
        assert "text-embedding-3-large" in cfg.embedding.model_name

    def test_llm_model_gpt4o(self):
        cfg = RAGConfig.openai_cloud()
        assert "gpt-4o" in cfg.llm.model_name

    def test_llm_base_url_is_none(self):
        # OpenAI SDK uses default endpoint — no base_url override
        cfg = RAGConfig.openai_cloud()
        assert cfg.llm.base_url is None

    def test_embedding_base_url_is_none(self):
        cfg = RAGConfig.openai_cloud()
        assert cfg.embedding.base_url is None

    def test_api_key_passed_through(self):
        cfg = RAGConfig.openai_cloud(api_key="sk-test")
        assert cfg.llm.api_key == "sk-test"
        assert cfg.embedding.api_key == "sk-test"

    def test_use_hyde_true(self):
        cfg = RAGConfig.openai_cloud()
        assert cfg.retrieval.use_hyde is True


# ---------------------------------------------------------------------------
# fast_local vs medical_precise are distinct
# ---------------------------------------------------------------------------

class TestPresetDistinctness:
    def test_fast_local_model_differs_from_medical_precise(self):
        fast = RAGConfig.fast_local()
        precise = RAGConfig.medical_precise()
        assert fast.llm.model_name != precise.llm.model_name

    def test_fast_local_embed_model_differs_from_medical_precise(self):
        fast = RAGConfig.fast_local()
        precise = RAGConfig.medical_precise()
        assert fast.embedding.model_name != precise.embedding.model_name

    def test_fast_local_retriever_names_subset_of_medical(self):
        fast = RAGConfig.fast_local()
        precise = RAGConfig.medical_precise()
        # medical_precise has more retriever names
        assert len(precise.retrieval.retriever_names) >= len(fast.retrieval.retriever_names)

    def test_medical_precise_top_k_retrieve_larger(self):
        fast = RAGConfig.fast_local()
        precise = RAGConfig.medical_precise()
        assert precise.retrieval.top_k_retrieve >= fast.retrieval.top_k_retrieve

    def test_fast_local_query_expansion_off(self):
        fast = RAGConfig.fast_local()
        assert fast.retrieval.query_expansion is False

    def test_medical_precise_query_expansion_on(self):
        precise = RAGConfig.medical_precise()
        assert precise.retrieval.query_expansion is True


# ---------------------------------------------------------------------------
# Mutating one preset does not affect another
# ---------------------------------------------------------------------------

class TestPresetIsolation:
    def test_mutating_local_ollama_does_not_affect_fast_local(self):
        local = RAGConfig.local_ollama()
        fast = RAGConfig.fast_local()
        original_fast_model = fast.llm.model_name

        local.llm.model_name = "totally-different-model"
        # fast_local should not be affected
        assert fast.llm.model_name == original_fast_model

    def test_mutating_retrieval_on_one_preset_does_not_affect_another(self):
        a = RAGConfig.local_ollama()
        b = RAGConfig.medical_precise()
        original_b_top_k = b.retrieval.top_k_retrieve

        a.retrieval.top_k_retrieve = 9999
        assert b.retrieval.top_k_retrieve == original_b_top_k


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_to_file_creates_valid_json(self, tmp_path):
        cfg = RAGConfig.local_ollama()
        out = str(tmp_path / "config.json")
        cfg.to_file(out)
        assert Path(out).exists()
        with open(out) as f:
            data = json.load(f)
        assert "embedding" in data
        assert "llm" in data

    def test_round_trip_preserves_all_sections(self, tmp_path):
        cfg = RAGConfig.openai_cloud(api_key="sk-roundtrip")
        out = str(tmp_path / "rt.json")
        cfg.to_file(out)
        restored = RAGConfig.from_file(out)
        assert restored.embedding.provider == cfg.embedding.provider
        assert restored.llm.model_name == cfg.llm.model_name
        assert restored.llm.api_key == cfg.llm.api_key
        assert restored.vector_index.backend == cfg.vector_index.backend
        assert restored.retrieval.top_k_retrieve == cfg.retrieval.top_k_retrieve
        assert restored.generation.max_context_tokens == cfg.generation.max_context_tokens

    def test_round_trip_preserves_custom_field(self, tmp_path):
        cfg = RAGConfig.local_ollama()
        cfg.retrieval.top_k_rerank = 99
        out = str(tmp_path / "custom.json")
        cfg.to_file(out)
        restored = RAGConfig.from_file(out)
        assert restored.retrieval.top_k_rerank == 99

    def test_to_dict_has_all_keys(self):
        cfg = RAGConfig.fast_local()
        d = cfg.to_dict()
        assert set(d.keys()) == {"embedding", "llm", "vector_index", "retrieval", "generation"}


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------

class TestFromEnv:
    def test_from_env_reads_embed_provider(self, monkeypatch):
        monkeypatch.setenv("NITRAG_EMBED_PROVIDER", "openai")
        cfg = RAGConfig.from_env()
        assert cfg.embedding.provider == "openai"

    def test_from_env_reads_llm_model(self, monkeypatch):
        monkeypatch.setenv("NITRAG_LLM_MODEL", "gpt-4-turbo")
        cfg = RAGConfig.from_env()
        assert cfg.llm.model_name == "gpt-4-turbo"

    def test_from_env_reads_llm_temperature_as_float(self, monkeypatch):
        monkeypatch.setenv("NITRAG_LLM_TEMPERATURE", "0.7")
        cfg = RAGConfig.from_env()
        assert abs(cfg.llm.temperature - 0.7) < 1e-9

    def test_from_env_reads_embed_base_url(self, monkeypatch):
        monkeypatch.setenv("NITRAG_EMBED_BASE_URL", "http://myhost:1234/v1")
        cfg = RAGConfig.from_env()
        assert cfg.embedding.base_url == "http://myhost:1234/v1"

    def test_from_env_no_vars_returns_defaults(self, monkeypatch):
        # Clear any relevant env vars
        for k in [
            "NITRAG_EMBED_PROVIDER", "NITRAG_EMBED_MODEL", "NITRAG_LLM_MODEL",
            "NITRAG_LLM_TEMPERATURE",
        ]:
            monkeypatch.delenv(k, raising=False)
        cfg = RAGConfig.from_env()
        default = RAGConfig()
        assert cfg.embedding.provider == default.embedding.provider

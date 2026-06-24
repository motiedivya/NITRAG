from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding provider.

    provider options
    ----------------
    "fastembed"           — ONNX-based, no PyTorch, self-hosted (default)
    "openai"              — OpenAI embeddings API or OpenAI-compatible endpoint
    "sentence_transformers" — full HuggingFace model support (requires torch)
    """
    provider: str = "fastembed"
    # nomic-ai/nomic-embed-text-v1.5: 8 192-token context, Apache 2.0, strong on clinical
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    base_url: Optional[str] = None          # e.g. "http://localhost:11434/v1" for Ollama embed
    api_key: Optional[str] = None           # None → OPENAI_API_KEY env var
    dimensions: Optional[int] = None        # None = auto-detected from model
    batch_size: int = 64
    normalize: bool = True                  # L2-normalise for cosine similarity
    device: str = "cpu"                     # "cpu" | "cuda" | "mps"
    max_length: Optional[int] = None        # None = model default


@dataclass
class LLMConfig:
    """Configuration for the LLM provider.

    provider options
    ----------------
    "openai_compatible"   — OpenAI SDK (OpenAI, Ollama, vLLM, LMStudio, Groq…)
    "anthropic"           — Anthropic SDK (claude-sonnet-4-6, claude-haiku-4-5…)
    """
    provider: str = "openai_compatible"
    model_name: str = "llama3.1:8b"
    base_url: Optional[str] = "http://localhost:11434/v1"   # None = OpenAI default
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 120
    system_prompt: Optional[str] = None     # None → default medical system prompt


@dataclass
class VectorIndexConfig:
    """Configuration for the vector index backend."""
    backend: str = "faiss"                  # "faiss" | "qdrant"
    index_type: str = "flat"                # "flat" (exact) | "hnsw" (ANN)
    metric: str = "cosine"                  # "cosine" | "dot" | "l2"
    # HNSW params (only used when index_type="hnsw")
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16
    hnsw_ef_search: int = 100
    # Qdrant-only params
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_prefix: str = "nitrag"


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval stage."""
    retriever_names: List[str] = field(
        default_factory=lambda: ["bm25", "dense", "hybrid"]
    )
    chunk_strategy_name: str = "block_group_800_overlap_1"
    reranker_name: str = "hybrid_weighted"
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    hybrid_alpha: float = 0.5               # 1.0 = all semantic, 0.0 = all lexical
    use_hyde: bool = False                  # LLM call per query; disable if LLM is slow
    query_expansion: bool = True            # medical abbreviation/synonym expansion


@dataclass
class GenerationConfig:
    """Configuration for the generation stage."""
    max_context_tokens: int = 3500
    context_ordering: str = "score"         # "score" | "page" | "mixed"
    include_metadata_in_context: bool = True
    min_citation_confidence: float = 0.25
    hallucination_check: bool = True        # post-process: verify claims have evidence
    structured_output: bool = True


@dataclass
class RAGConfig:
    """Master configuration for the full NITRAG pipeline.

    Usage
    -----
    # Preset factories (most common):
    config = RAGConfig.local_ollama()
    config = RAGConfig.openai_cloud(api_key="sk-...")
    config = RAGConfig.medical_precise()

    # Load from file:
    config = RAGConfig.from_file("configs/local_ollama.json")

    # Load from environment variables:
    config = RAGConfig.from_env()           # reads NITRAG_* env vars

    # Customise one field without rebuilding everything:
    config = RAGConfig.local_ollama()
    config.llm.model_name = "llama3.1:70b"
    config.retrieval.top_k_retrieve = 30
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": asdict(self.embedding),
            "llm": asdict(self.llm),
            "vector_index": asdict(self.vector_index),
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
        }

    def to_file(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        def _merge(dcls, overrides: dict):
            import dataclasses
            defaults = {f.name: f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
                        for f in dataclasses.fields(dcls)}
            defaults.update({k: v for k, v in overrides.items() if k in defaults})
            return dcls(**defaults)

        return cls(
            embedding=_merge(EmbeddingConfig, data.get("embedding", {})),
            llm=_merge(LLMConfig, data.get("llm", {})),
            vector_index=_merge(VectorIndexConfig, data.get("vector_index", {})),
            retrieval=_merge(RetrievalConfig, data.get("retrieval", {})),
            generation=_merge(GenerationConfig, data.get("generation", {})),
        )

    @classmethod
    def from_file(cls, path: str) -> "RAGConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = "NITRAG_") -> "RAGConfig":
        """Load from environment variables (NITRAG_* by default).

        Supported vars
        --------------
        NITRAG_EMBED_PROVIDER, NITRAG_EMBED_MODEL, NITRAG_EMBED_BASE_URL, NITRAG_EMBED_API_KEY
        NITRAG_LLM_PROVIDER, NITRAG_LLM_MODEL, NITRAG_LLM_BASE_URL, NITRAG_LLM_API_KEY, NITRAG_LLM_TEMPERATURE
        """
        config = cls()
        _str_map: Dict[str, tuple] = {
            f"{prefix}EMBED_PROVIDER": ("embedding", "provider"),
            f"{prefix}EMBED_MODEL":    ("embedding", "model_name"),
            f"{prefix}EMBED_BASE_URL": ("embedding", "base_url"),
            f"{prefix}EMBED_API_KEY":  ("embedding", "api_key"),
            f"{prefix}LLM_PROVIDER":   ("llm", "provider"),
            f"{prefix}LLM_MODEL":      ("llm", "model_name"),
            f"{prefix}LLM_BASE_URL":   ("llm", "base_url"),
            f"{prefix}LLM_API_KEY":    ("llm", "api_key"),
        }
        _typed_map: Dict[str, tuple] = {
            f"{prefix}LLM_TEMPERATURE": ("llm", "temperature", float),
            f"{prefix}LLM_MAX_TOKENS":  ("llm", "max_tokens", int),
        }
        for env_key, (section, attr) in _str_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                setattr(getattr(config, section), attr, val)
        for env_key, (section, attr, typ) in _typed_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                setattr(getattr(config, section), attr, typ(val))
        return config

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def local_ollama(cls) -> "RAGConfig":
        """Fully local: fastembed (nomic-embed-v1.5, 8k ctx) + Ollama (llama3.1:8b).

        Requirements: ``uv pip install fastembed faiss-cpu openai``
        Ollama must be running: ``ollama serve && ollama pull llama3.1:8b``
        """
        return cls(
            embedding=EmbeddingConfig(
                provider="fastembed",
                model_name="nomic-ai/nomic-embed-text-v1.5",
                dimensions=768,
                batch_size=64,
            ),
            llm=LLMConfig(
                provider="openai_compatible",
                model_name="llama3.1:8b",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0.1,
                max_tokens=2048,
            ),
            vector_index=VectorIndexConfig(backend="faiss", index_type="flat"),
            retrieval=RetrievalConfig(
                retriever_names=["bm25", "dense", "hybrid"],
                chunk_strategy_name="block_group_800_overlap_1",
                reranker_name="hybrid_weighted",
                top_k_retrieve=20,
                top_k_rerank=5,
                use_hyde=False,
                query_expansion=True,
            ),
            generation=GenerationConfig(
                max_context_tokens=3500,
                hallucination_check=True,
            ),
        )

    @classmethod
    def openai_cloud(cls, api_key: Optional[str] = None) -> "RAGConfig":
        """Cloud: text-embedding-3-large + gpt-4o.

        Requirements: ``uv pip install fastembed faiss-cpu openai``
        Set OPENAI_API_KEY or pass api_key=.
        """
        return cls(
            embedding=EmbeddingConfig(
                provider="openai",
                model_name="text-embedding-3-large",
                base_url=None,
                api_key=api_key,
                dimensions=1024,            # reduced dims for efficiency
                batch_size=32,
            ),
            llm=LLMConfig(
                provider="openai_compatible",
                model_name="gpt-4o",
                base_url=None,
                api_key=api_key,
                temperature=0.1,
                max_tokens=4096,
            ),
            vector_index=VectorIndexConfig(backend="faiss", index_type="hnsw"),
            retrieval=RetrievalConfig(
                retriever_names=["bm25", "dense", "hybrid"],
                chunk_strategy_name="block_group_800_overlap_1",
                reranker_name="hybrid_weighted",
                top_k_retrieve=20,
                top_k_rerank=5,
                use_hyde=True,
                query_expansion=True,
            ),
            generation=GenerationConfig(
                max_context_tokens=5000,
                hallucination_check=True,
            ),
        )

    @classmethod
    def fast_local(cls) -> "RAGConfig":
        """Fastest local: fastembed (bge-small) + Ollama (mistral:7b).

        Optimised for speed; good for development and testing.
        """
        return cls(
            embedding=EmbeddingConfig(
                provider="fastembed",
                model_name="BAAI/bge-small-en-v1.5",
                dimensions=384,
                batch_size=128,
            ),
            llm=LLMConfig(
                provider="openai_compatible",
                model_name="mistral:7b-instruct",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                max_tokens=1024,
            ),
            vector_index=VectorIndexConfig(backend="faiss", index_type="flat"),
            retrieval=RetrievalConfig(
                retriever_names=["bm25", "dense"],
                chunk_strategy_name="block_group_800_overlap_1",
                reranker_name="keyword_overlap",
                top_k_retrieve=10,
                top_k_rerank=5,
                use_hyde=False,
                query_expansion=False,
            ),
        )

    @classmethod
    def medical_precise(cls) -> "RAGConfig":
        """High-accuracy medical: bge-large-en-v1.5 + large LLM.

        Best for citation accuracy and faithfulness. Slower than local_ollama.
        Swap llm.model_name to "gpt-4o" and llm.base_url to None for OpenAI.
        """
        return cls(
            embedding=EmbeddingConfig(
                provider="fastembed",
                model_name="BAAI/bge-large-en-v1.5",
                dimensions=1024,
                batch_size=32,
            ),
            llm=LLMConfig(
                provider="openai_compatible",
                model_name="llama3.1:70b",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0.05,
                max_tokens=4096,
            ),
            vector_index=VectorIndexConfig(backend="faiss", index_type="hnsw"),
            retrieval=RetrievalConfig(
                retriever_names=["bm25", "dense", "hybrid", "clinical_section_scoped"],
                chunk_strategy_name="block_group_800_overlap_1",
                reranker_name="hybrid_weighted",
                top_k_retrieve=30,
                top_k_rerank=8,
                use_hyde=True,
                query_expansion=True,
            ),
            generation=GenerationConfig(
                max_context_tokens=5000,
                hallucination_check=True,
                min_citation_confidence=0.2,
            ),
        )

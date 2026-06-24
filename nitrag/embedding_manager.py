"""Embedding provider abstraction and per-strategy chunk embedding storage.

Storage layout
--------------
  {doc_dir}/embeddings/{chunk_strategy_name}/
      vectors.npy      — float32 array [N, D]
      chunk_ids.npy    — int64 array [N]  (aligned row-for-row with vectors)
      manifest.json    — model name, dimensions, created_at, chunk count
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from .config import EmbeddingConfig


# ─────────────────────────────────────────────────────────────────────────────
# Provider ABC
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    """Abstract base for any embedding backend."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return float32 array [len(texts), dimensions]."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query; may apply a query prefix if the model needs one."""
        return self.embed_texts([query])[0]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensionality."""


# ─────────────────────────────────────────────────────────────────────────────
# FastEmbed provider (ONNX, no PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class FastEmbedProvider(EmbeddingProvider):
    """Self-hosted ONNX-based embeddings via the fastembed library.

    No PyTorch required. Models are downloaded to ~/.cache/fastembed on first use.

    Recommended models
    ------------------
    nomic-ai/nomic-embed-text-v1.5  — 768d, 8 192-token context, Apache 2.0 (default)
    BAAI/bge-large-en-v1.5          — 1024d, top MTEB, strong on medical
    BAAI/bge-small-en-v1.5          — 384d, fast, good for dev
    BAAI/bge-m3                     — 1024d, multilingual
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed is required for FastEmbedProvider. "
                "Install with: uv pip install fastembed"
            ) from e

        kwargs: Dict[str, Any] = {"model_name": config.model_name}
        if config.max_length is not None:
            kwargs["max_length"] = config.max_length
        self._model = TextEmbedding(**kwargs)
        self._batch_size = config.batch_size
        self._normalize = config.normalize
        self._dims: Optional[int] = config.dimensions

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = list(
            self._model.embed(texts, batch_size=self._batch_size)
        )
        arr = np.array(embeddings, dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        embeddings = list(self._model.query_embed([query]))
        arr = np.array(embeddings, dtype=np.float32)
        if self._normalize:
            norm = np.linalg.norm(arr, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            arr = arr / norm
        return arr[0]

    @property
    def dimensions(self) -> int:
        if self._dims is not None:
            return self._dims
        probe = self.embed_texts(["probe"])
        self._dims = int(probe.shape[1])
        return self._dims


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible provider
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings API or any OpenAI-compatible endpoint.

    Works with: OpenAI, Azure OpenAI, Ollama (embed endpoint), vLLM.
    Set base_url for non-OpenAI endpoints.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai is required. Install with: uv pip install openai") from e

        import os
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self._client = openai.OpenAI(**kwargs)
        self._model_name = config.model_name
        self._batch_size = config.batch_size
        self._normalize = config.normalize
        self._dims = config.dimensions
        # text-embedding-3-* supports reduced dimensions
        self._dimensions_param = config.dimensions

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            kwargs: Dict[str, Any] = {"model": self._model_name, "input": batch}
            if self._dimensions_param is not None and "text-embedding-3" in self._model_name:
                kwargs["dimensions"] = self._dimensions_param
            resp = self._client.embeddings.create(**kwargs)
            batch_vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_vecs)
        arr = np.array(all_embeddings, dtype=np.float32)
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr

    @property
    def dimensions(self) -> int:
        if self._dims is not None:
            return self._dims
        probe = self.embed_texts(["probe"])
        self._dims = int(probe.shape[1])
        return self._dims


# ─────────────────────────────────────────────────────────────────────────────
# SentenceTransformers provider (optional; requires torch)
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformersProvider(EmbeddingProvider):
    """HuggingFace sentence-transformers embedding.

    Full model support but requires PyTorch.
    Install: uv pip install 'nitrag[sentence-transformers]'
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: uv pip install 'nitrag[sentence-transformers]'"
            ) from e

        self._model = SentenceTransformer(
            config.model_name,
            device=config.device,
        )
        self._batch_size = config.batch_size
        self._normalize = config.normalize
        self._dims = config.dimensions

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        arr = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.array(arr, dtype=np.float32)

    @property
    def dimensions(self) -> int:
        if self._dims is not None:
            return self._dims
        self._dims = self._model.get_sentence_embedding_dimension()
        return self._dims  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "fastembed":
        return FastEmbedProvider(config)
    if config.provider == "openai":
        return OpenAIEmbeddingProvider(config)
    if config.provider == "sentence_transformers":
        return SentenceTransformersProvider(config)
    raise ValueError(
        f"Unknown embedding provider: {config.provider!r}. "
        "Options: 'fastembed', 'openai', 'sentence_transformers'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingManager
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingManager:
    """Orchestrates chunk embedding and storage for a single document.

    Storage layout (per chunk strategy)
    ------------------------------------
      {doc_dir}/embeddings/{strategy}/vectors.npy
      {doc_dir}/embeddings/{strategy}/chunk_ids.npy
      {doc_dir}/embeddings/{strategy}/manifest.json

    Usage
    -----
    from nitrag.embedding_manager import EmbeddingManager
    from nitrag.config import RAGConfig

    config = RAGConfig.local_ollama()
    mgr = EmbeddingManager(store, config.embedding)
    mgr.embed_chunks("block_group_800_overlap_1")
    vectors, chunk_ids = mgr.load_embeddings("block_group_800_overlap_1")
    q_vec = mgr.embed_query("What medications were prescribed?")
    """

    def __init__(self, store, config: EmbeddingConfig) -> None:
        self.store = store
        self.config = config
        self.document_dir = Path(store.paths.document_dir)
        self.embeddings_dir = self.document_dir / "embeddings"
        self._provider: Optional[EmbeddingProvider] = None

    # ------------------------------------------------------------------
    # Provider (lazy)
    # ------------------------------------------------------------------

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = create_embedding_provider(self.config)
        return self._provider

    # ------------------------------------------------------------------
    # Chunk embedding
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunk_strategy_name: str,
        use_enriched: bool = True,
        overwrite: bool = False,
    ) -> Path:
        """Embed all chunks for *chunk_strategy_name* and save to disk.

        Returns the directory where vectors were saved.
        """
        out_dir = self.embeddings_dir / chunk_strategy_name
        manifest_path = out_dir / "manifest.json"

        if not overwrite and manifest_path.exists():
            return out_dir

        # Load chunks parquet
        chunks_root = self.document_dir / ("chunks_enriched" if use_enriched else "chunks")
        parquet_path = chunks_root / f"{chunk_strategy_name}.parquet"
        if not parquet_path.exists():
            # fall back to unenriched
            parquet_path = self.document_dir / "chunks" / f"{chunk_strategy_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"No chunks found for strategy {chunk_strategy_name!r}: {parquet_path}")

        rows = pq.read_table(parquet_path).to_pylist()
        if not rows:
            raise ValueError(f"Empty chunk file: {parquet_path}")

        # Decode text for each chunk
        texts: List[str] = []
        chunk_ids: List[int] = []
        for row in rows:
            text = self.store.decode_span(int(row["start_index"]), int(row["end_index"]))
            texts.append(text)
            chunk_ids.append(int(row["chunk_id"]))

        # Embed in batches
        t0 = time.time()
        vectors = self.provider.embed_texts(texts)
        elapsed = time.time() - t0

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_dir / "vectors.npy"), vectors.astype(np.float32))
        np.save(str(out_dir / "chunk_ids.npy"), np.array(chunk_ids, dtype=np.int64))

        manifest: Dict[str, Any] = {
            "chunk_strategy_name": chunk_strategy_name,
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "dimensions": int(vectors.shape[1]),
            "num_chunks": len(chunk_ids),
            "use_enriched": use_enriched,
            "elapsed_seconds": round(elapsed, 2),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return out_dir

    def embed_all_strategies(
        self,
        use_enriched: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Embed all available chunk strategies."""
        chunks_dir = self.document_dir / ("chunks_enriched" if use_enriched else "chunks")
        if not chunks_dir.exists():
            chunks_dir = self.document_dir / "chunks"

        results: Dict[str, Any] = {}
        for parquet in sorted(chunks_dir.glob("*.parquet")):
            strategy_name = parquet.stem
            try:
                out_dir = self.embed_chunks(strategy_name, use_enriched=use_enriched, overwrite=overwrite)
                results[strategy_name] = str(out_dir)
            except Exception as e:
                results[strategy_name] = f"ERROR: {e}"
        return results

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_embeddings(self, chunk_strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load persisted embeddings.

        Returns
        -------
        vectors   : float32 ndarray [N, D]
        chunk_ids : int64 ndarray [N]
        """
        out_dir = self.embeddings_dir / chunk_strategy_name
        vectors_path = out_dir / "vectors.npy"
        ids_path = out_dir / "chunk_ids.npy"

        if not vectors_path.exists():
            raise FileNotFoundError(
                f"No embeddings found for strategy {chunk_strategy_name!r}. "
                f"Run embed_chunks('{chunk_strategy_name}') first."
            )

        vectors = np.load(str(vectors_path))
        chunk_ids = np.load(str(ids_path))
        return vectors, chunk_ids

    def is_embedded(self, chunk_strategy_name: str) -> bool:
        """Return True if embeddings exist for this strategy."""
        return (self.embeddings_dir / chunk_strategy_name / "manifest.json").exists()

    def list_embedded_strategies(self) -> List[str]:
        """List all chunk strategies that have been embedded."""
        if not self.embeddings_dir.exists():
            return []
        return sorted(
            p.name for p in self.embeddings_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        )

    def get_manifest(self, chunk_strategy_name: str) -> Dict[str, Any]:
        """Return the embedding manifest for a strategy."""
        path = self.embeddings_dir / chunk_strategy_name / "manifest.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns float32 ndarray [D]."""
        return self.provider.embed_query(query)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed multiple queries. Returns float32 ndarray [len(queries), D]."""
        return self.provider.embed_texts(queries)

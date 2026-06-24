"""FAISS (and optional Qdrant) vector index management.

Storage layout
--------------
  {doc_dir}/indexes/{chunk_strategy_name}/dense/
      faiss.index      — FAISS index (Flat or HNSW)
      chunk_ids.npy    — int64 array [N], maps FAISS row → chunk_id
      docs.parquet     — chunk metadata rows (for post-retrieval filtering / result assembly)
      manifest.json    — index type, metric, dimensions, chunk count
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from .config import VectorIndexConfig
from .embedding_manager import EmbeddingManager


# ─────────────────────────────────────────────────────────────────────────────
# Backend ABC
# ─────────────────────────────────────────────────────────────────────────────

class VectorIndexBackend(ABC):
    @abstractmethod
    def build(
        self,
        vectors: np.ndarray,
        chunk_ids: np.ndarray,
        metadata_rows: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        output_dir: Path,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    @staticmethod
    def _passes_filters(row: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
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
            elif actual != expected:
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# FAISS backend (flat + HNSW)
# ─────────────────────────────────────────────────────────────────────────────

class FAISSBackend(VectorIndexBackend):
    """FAISS vector index, disk-persistent. Supports exact (flat) and ANN (HNSW).

    For cosine similarity, vectors must be L2-normalised before indexing
    (EmbeddingManager does this by default with normalize=True).
    """

    def __init__(self, config: VectorIndexConfig) -> None:
        self.config = config

    def _import_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError as e:
            raise ImportError("faiss-cpu is required. Install with: uv pip install faiss-cpu") from e

    def _make_index(self, faiss, dimensions: int):
        metric = self.config.metric
        if metric == "cosine":
            # L2-normalised vectors + inner product ≡ cosine similarity
            inner = faiss.METRIC_INNER_PRODUCT
        elif metric == "dot":
            inner = faiss.METRIC_INNER_PRODUCT
        else:
            inner = faiss.METRIC_L2

        if self.config.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimensions, self.config.hnsw_m, inner)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search
        else:
            # Flat index — exact nearest neighbour
            if metric == "l2":
                index = faiss.IndexFlatL2(dimensions)
            else:
                index = faiss.IndexFlatIP(dimensions)

        return index

    def build(
        self,
        vectors: np.ndarray,
        chunk_ids: np.ndarray,
        metadata_rows: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        faiss = self._import_faiss()
        output_dir.mkdir(parents=True, exist_ok=True)

        vectors_f32 = vectors.astype(np.float32)
        dimensions = vectors_f32.shape[1]

        index = self._make_index(faiss, dimensions)
        index.add(vectors_f32)

        faiss.write_index(index, str(output_dir / "faiss.index"))
        np.save(str(output_dir / "chunk_ids.npy"), chunk_ids.astype(np.int64))

        # Save metadata parquet for result assembly
        import pandas as pd
        import pyarrow as pa
        df = pd.DataFrame(metadata_rows)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_dir / "docs.parquet")

        manifest = {
            "index_type": self.config.index_type,
            "metric": self.config.metric,
            "dimensions": int(dimensions),
            "num_vectors": int(len(vectors_f32)),
            "backend": "faiss",
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        output_dir: Path,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        faiss = self._import_faiss()
        index_path = output_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        index = faiss.read_index(str(index_path))
        chunk_ids = np.load(str(output_dir / "chunk_ids.npy"))

        # Adjust top_k to account for potential filter rejects
        fetch_k = min(top_k * 4 if filters else top_k, int(chunk_ids.shape[0]))
        if fetch_k == 0:
            return []

        if self.config.index_type == "hnsw":
            index.hnsw.efSearch = max(self.config.hnsw_ef_search, fetch_k)

        q = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(q, fetch_k)

        docs_rows = pq.read_table(output_dir / "docs.parquet").to_pylist()
        docs_by_chunk_id: Dict[int, Dict[str, Any]] = {
            int(r.get("chunk_id", i)): r for i, r in enumerate(docs_rows)
        }

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            cid = int(chunk_ids[idx])
            row = docs_by_chunk_id.get(cid, {})
            if not self._passes_filters(row, filters):
                continue

            # Inner product on L2-normalised vectors = cosine similarity ∈ [-1, 1]
            # Shift to [0, 1] for display consistency with lexical scores
            score = float(dist)
            if self.config.metric in ("cosine", "dot"):
                score = (score + 1.0) / 2.0

            results.append({**row, "dense_score": round(score, 6), "faiss_idx": int(idx)})
            if len(results) >= top_k:
                break

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant backend (optional)
# ─────────────────────────────────────────────────────────────────────────────

class QdrantBackend(VectorIndexBackend):
    """Qdrant vector database backend.

    Requires a running Qdrant server. Install: uv pip install 'nitrag[qdrant]'
    """

    def __init__(self, config: VectorIndexConfig) -> None:
        self.config = config

    def _client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required. Install with: uv pip install 'nitrag[qdrant]'"
            ) from e
        return QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key)

    def _collection_name(self, output_dir: Path) -> str:
        return f"{self.config.qdrant_collection_prefix}_{output_dir.parent.name}"

    def build(
        self,
        vectors: np.ndarray,
        chunk_ids: np.ndarray,
        metadata_rows: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        from qdrant_client.models import Distance, VectorParams, PointStruct
        client = self._client()
        collection = self._collection_name(output_dir)
        dimensions = int(vectors.shape[1])
        distance_map = {"cosine": Distance.COSINE, "dot": Distance.DOT, "l2": Distance.EUCLID}
        distance = distance_map.get(self.config.metric, Distance.COSINE)

        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(collection, vectors_config=VectorParams(size=dimensions, distance=distance))

        points = [
            PointStruct(id=int(chunk_ids[i]), vector=vectors[i].tolist(), payload=metadata_rows[i])
            for i in range(len(vectors))
        ]
        client.upsert(collection_name=collection, points=points)

        # Save manifest only (no local files needed for Qdrant)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "manifest.json", "w") as f:
            json.dump({"backend": "qdrant", "collection": collection, "dimensions": dimensions}, f)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        output_dir: Path,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = self._client()
        collection = self._collection_name(output_dir)

        qdrant_filter = None
        if filters:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v))
                          for k, v in filters.items() if not isinstance(v, dict)]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        hits = client.search(
            collection_name=collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [{**hit.payload, "dense_score": round(float(hit.score), 6)} for hit in hits]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_vector_backend(config: VectorIndexConfig) -> VectorIndexBackend:
    if config.backend == "faiss":
        return FAISSBackend(config)
    if config.backend == "qdrant":
        return QdrantBackend(config)
    raise ValueError(f"Unknown vector index backend: {config.backend!r}. Options: 'faiss', 'qdrant'")


# ─────────────────────────────────────────────────────────────────────────────
# VectorIndexManager
# ─────────────────────────────────────────────────────────────────────────────

class VectorIndexManager:
    """Builds and searches FAISS (or Qdrant) vector indexes.

    Works in tandem with EmbeddingManager — embeddings must exist before indexing.

    Storage
    -------
    Indexes are stored alongside lexical indexes:
      {doc_dir}/indexes/{chunk_strategy}/dense/

    Usage
    -----
    from nitrag.vector_index_manager import VectorIndexManager

    vim = VectorIndexManager(store, embedding_manager, config.vector_index)
    vim.build("block_group_800_overlap_1")
    results = vim.search("block_group_800_overlap_1", query_vector, top_k=10)
    """

    def __init__(
        self,
        store,
        embedding_manager: EmbeddingManager,
        config: VectorIndexConfig,
    ) -> None:
        self.store = store
        self.embedding_manager = embedding_manager
        self.config = config
        self.document_dir = Path(store.paths.document_dir)
        self.index_root_dir = self.document_dir / "indexes"
        self._backend: Optional[VectorIndexBackend] = None

    @property
    def backend(self) -> VectorIndexBackend:
        if self._backend is None:
            self._backend = create_vector_backend(self.config)
        return self._backend

    def _index_dir(self, chunk_strategy_name: str) -> Path:
        return self.index_root_dir / chunk_strategy_name / "dense"

    def _load_chunk_metadata(self, chunk_strategy_name: str) -> List[Dict[str, Any]]:
        """Load chunk rows for metadata-aware result assembly."""
        for sub in ("chunks_enriched", "chunks"):
            p = self.document_dir / sub / f"{chunk_strategy_name}.parquet"
            if p.exists():
                return pq.read_table(p).to_pylist()
        return []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunk_strategy_name: str,
        overwrite: bool = False,
    ) -> Path:
        """Build a vector index for *chunk_strategy_name*."""
        out_dir = self._index_dir(chunk_strategy_name)
        if not overwrite and (out_dir / "manifest.json").exists():
            return out_dir

        vectors, chunk_ids = self.embedding_manager.load_embeddings(chunk_strategy_name)
        metadata_rows = self._load_chunk_metadata(chunk_strategy_name)

        # Align metadata rows to chunk_ids order
        meta_by_cid: Dict[int, Dict[str, Any]] = {
            int(r.get("chunk_id", i)): r for i, r in enumerate(metadata_rows)
        }
        ordered_meta = [meta_by_cid.get(int(cid), {}) for cid in chunk_ids]

        self.backend.build(vectors, chunk_ids, ordered_meta, out_dir)
        return out_dir

    def build_all(
        self,
        chunk_strategy_names: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Build vector indexes for all embedded strategies."""
        if chunk_strategy_names is None:
            chunk_strategy_names = self.embedding_manager.list_embedded_strategies()

        results: Dict[str, Any] = {}
        for strategy in chunk_strategy_names:
            try:
                out_dir = self.build(strategy, overwrite=overwrite)
                results[strategy] = str(out_dir)
            except Exception as e:
                results[strategy] = f"ERROR: {e}"
        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        chunk_strategy_name: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbour search against the built index."""
        out_dir = self._index_dir(chunk_strategy_name)
        return self.backend.search(query_vector, top_k, out_dir, filters)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_built(self, chunk_strategy_name: str) -> bool:
        return (self._index_dir(chunk_strategy_name) / "manifest.json").exists()

    def list_built_strategies(self) -> List[str]:
        if not self.index_root_dir.exists():
            return []
        return sorted(
            p.parent.name
            for p in self.index_root_dir.rglob("dense/manifest.json")
        )

    def get_manifest(self, chunk_strategy_name: str) -> Dict[str, Any]:
        path = self._index_dir(chunk_strategy_name) / "manifest.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

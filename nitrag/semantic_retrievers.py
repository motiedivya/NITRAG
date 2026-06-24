"""Semantic (dense + hybrid) retrieval strategies for the RetrieverManager.

New strategies
--------------
dense   — direct cosine similarity via FAISS
hybrid  — BM25 + dense fused with Reciprocal Rank Fusion (RRF)
hyde    — Hypothetical Document Embedding: generate a hypothetical answer,
          embed it, then search — improves recall for clinical queries

Registration
------------
from nitrag.semantic_retrievers import register_semantic_retrievers

register_semantic_retrievers(
    retriever_manager,
    embedding_manager,
    vector_index_manager,
    llm_config=config.llm,   # optional; only needed for hyde
)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .retriever_manager import BaseRetrieverStrategy, make_result, passes_filters, result_key

if TYPE_CHECKING:
    from .embedding_manager import EmbeddingManager
    from .vector_index_manager import VectorIndexManager
    from .config import LLMConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for rank (1-indexed)."""
    return 1.0 / (k + rank)


def _fuse_rrf(
    lexical: List[Dict[str, Any]],
    semantic: List[Dict[str, Any]],
    alpha: float = 0.5,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    alpha controls the weight: 1.0 = all semantic, 0.0 = all lexical.
    """
    from collections import defaultdict
    scores: Dict[tuple, float] = defaultdict(float)
    items: Dict[tuple, Dict[str, Any]] = {}

    for rank, r in enumerate(lexical, start=1):
        key = result_key(r)
        scores[key] += (1.0 - alpha) * _rrf_score(rank)
        if key not in items:
            items[key] = r

    for rank, r in enumerate(semantic, start=1):
        key = result_key(r)
        scores[key] += alpha * _rrf_score(rank)
        if key not in items:
            items[key] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, score in ranked[:top_k]:
        item = dict(items[key])
        item["rrf_score"] = round(score, 6)
        results.append(item)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Dense retriever
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetrieverStrategy(BaseRetrieverStrategy):
    """Retrieve via dense vector similarity (FAISS).

    Embeds the query using EmbeddingManager then searches the FAISS index
    built by VectorIndexManager. Results include a ``dense_score`` field
    (cosine similarity, shifted to [0, 1]) in addition to the standard
    result fields.
    """
    name = "dense"
    description = "Dense semantic retrieval via FAISS cosine similarity."

    def __init__(
        self,
        embedding_manager: "EmbeddingManager",
        vector_index_manager: "VectorIndexManager",
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vector_index_manager = vector_index_manager

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
        if not chunk_strategy_name:
            raise ValueError("DenseRetrieverStrategy requires chunk_strategy_name")

        if not self._vector_index_manager.is_built(chunk_strategy_name):
            raise RuntimeError(
                f"Dense index not built for strategy {chunk_strategy_name!r}. "
                "Run VectorIndexManager.build() first."
            )

        q_vec = self._embedding_manager.embed_query(query)
        raw = self._vector_index_manager.search(
            chunk_strategy_name=chunk_strategy_name,
            query_vector=q_vec,
            top_k=top_k,
            filters=filters,
        )

        results = []
        for r in raw:
            score = float(r.get("dense_score") or 0.0)
            result = make_result(
                row=r,
                score=score,
                retriever_name=self.name,
                query=query,
                store=store,
                extra={"dense_score": score},
            )
            results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever (BM25 + dense, RRF fusion)
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetrieverStrategy(BaseRetrieverStrategy):
    """Hybrid retrieval: BM25 + dense cosine, fused via Reciprocal Rank Fusion.

    Parameters
    ----------
    alpha       : 0.0 = all BM25, 1.0 = all dense, 0.5 = equal weight
    fetch_k     : number of candidates fetched from each side before fusion
    """
    name = "hybrid"
    description = "Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion."

    def __init__(
        self,
        embedding_manager: "EmbeddingManager",
        vector_index_manager: "VectorIndexManager",
        alpha: float = 0.5,
        fetch_k: int = 40,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vector_index_manager = vector_index_manager
        self.alpha = alpha
        self.fetch_k = fetch_k

    def retrieve(
        self,
        *,
        store,
        index_root_dir: Path,
        query: str,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not chunk_strategy_name:
            raise ValueError("HybridRetrieverStrategy requires chunk_strategy_name")

        effective_alpha = alpha if alpha is not None else self.alpha

        # BM25 side
        from .retriever_manager import BM25RetrieverStrategy
        bm25 = BM25RetrieverStrategy()
        try:
            lexical = bm25.retrieve(
                store=store,
                index_root_dir=index_root_dir,
                query=query,
                chunk_strategy_name=chunk_strategy_name,
                top_k=self.fetch_k,
                filters=filters,
            )
        except Exception:
            lexical = []

        # Dense side
        semantic: List[Dict[str, Any]] = []
        if self._vector_index_manager.is_built(chunk_strategy_name):
            q_vec = self._embedding_manager.embed_query(query)
            raw = self._vector_index_manager.search(
                chunk_strategy_name=chunk_strategy_name,
                query_vector=q_vec,
                top_k=self.fetch_k,
                filters=filters,
            )
            for r in raw:
                score = float(r.get("dense_score") or 0.0)
                semantic.append(make_result(
                    row=r, score=score, retriever_name="dense",
                    query=query, store=store, extra={"dense_score": score},
                ))

        if not lexical and not semantic:
            return []
        if not lexical:
            fused = semantic[:top_k]
        elif not semantic:
            fused = lexical[:top_k]
        else:
            fused = _fuse_rrf(lexical, semantic, alpha=effective_alpha, top_k=top_k * 2)

        # Tag results with hybrid metadata
        results = []
        for r in fused[:top_k]:
            r = dict(r)
            r["retriever_name"] = self.name
            r["score"] = r.get("rrf_score") or r.get("score") or 0.0
            results.append(r)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# HyDE retriever
# ─────────────────────────────────────────────────────────────────────────────

class HyDERetrieverStrategy(BaseRetrieverStrategy):
    """Hypothetical Document Embedding retrieval.

    Generates a hypothetical document that answers the query using an LLM,
    then embeds *that* passage instead of the raw query. This improves
    semantic retrieval for clinical questions where the query phrasing
    differs significantly from how clinical notes express the answer.

    Requires an LLM config. Falls back to regular dense retrieval on error.
    """
    name = "hyde"
    description = "HyDE: embed a generated hypothetical answer for better semantic recall."

    def __init__(
        self,
        embedding_manager: "EmbeddingManager",
        vector_index_manager: "VectorIndexManager",
        llm_config: "LLMConfig",
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vector_index_manager = vector_index_manager
        self._llm_config = llm_config
        self._query_manager: Optional[object] = None

    def _get_query_manager(self):
        if self._query_manager is None:
            from .query_manager import QueryManager
            self._query_manager = QueryManager(llm_config=self._llm_config)
        return self._query_manager

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
        if not chunk_strategy_name:
            raise ValueError("HyDERetrieverStrategy requires chunk_strategy_name")

        # Generate hypothetical document
        embed_text = query
        try:
            qm = self._get_query_manager()
            hyde_passage = qm.generate_hyde(query)  # type: ignore[union-attr]
            if hyde_passage:
                embed_text = hyde_passage
        except Exception:
            pass  # fall back to embedding the raw query

        q_vec = self._embedding_manager.embed_query(embed_text)

        if not self._vector_index_manager.is_built(chunk_strategy_name):
            raise RuntimeError(
                f"Dense index not built for {chunk_strategy_name!r}. "
                "Run VectorIndexManager.build() first."
            )

        raw = self._vector_index_manager.search(
            chunk_strategy_name=chunk_strategy_name,
            query_vector=q_vec,
            top_k=top_k,
            filters=filters,
        )

        results = []
        for r in raw:
            score = float(r.get("dense_score") or 0.0)
            result = make_result(
                row=r, score=score, retriever_name=self.name,
                query=query, store=store, extra={"dense_score": score, "hyde_used": True},
            )
            results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Registration helper
# ─────────────────────────────────────────────────────────────────────────────

def register_semantic_retrievers(
    retriever_manager,
    embedding_manager: "EmbeddingManager",
    vector_index_manager: "VectorIndexManager",
    alpha: float = 0.5,
    llm_config: Optional["LLMConfig"] = None,
    force: bool = True,
) -> None:
    """Register dense, hybrid, and (optionally) hyde into an existing RetrieverManager.

    Parameters
    ----------
    retriever_manager    : existing RetrieverManager instance (already has lexical strategies)
    embedding_manager    : EmbeddingManager with an initialised provider
    vector_index_manager : VectorIndexManager with built indexes
    alpha                : hybrid fusion weight (0.0 = all BM25, 1.0 = all dense)
    llm_config           : required for hyde strategy; omit to skip hyde registration
    force                : overwrite if already registered
    """
    retriever_manager.register_retriever(
        DenseRetrieverStrategy(embedding_manager, vector_index_manager),
        force=force,
    )
    retriever_manager.register_retriever(
        HybridRetrieverStrategy(embedding_manager, vector_index_manager, alpha=alpha),
        force=force,
    )
    if llm_config is not None:
        retriever_manager.register_retriever(
            HyDERetrieverStrategy(embedding_manager, vector_index_manager, llm_config),
            force=force,
        )

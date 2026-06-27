"""End-to-end RAG pipeline orchestrator.

Single entry point: RAGPipeline.answer(query) → RAGResponse

Pipeline stages
---------------
1. Query understanding (expand medical abbreviations, classify, HyDE)
2. Retrieve: lexical + semantic (configurable blend) → candidates
3. Rerank: combined signal reranking
4. Assemble: token-budget context with citation numbering
5. Generate: LLM answer with grounded [N] citations
6. Evaluate (optional): faithfulness, hallucination risk, context precision

Usage
-----
    # Option A — preset factory (recommended)
    pipeline = RAGPipeline.local_ollama(store)
    response = pipeline.answer("What medications were prescribed?")

    # Option B — from config file
    pipeline = RAGPipeline.from_config_file(store, "configs/local_ollama.json")

    # Option C — explicit config
    from nitrag.config import RAGConfig
    config = RAGConfig.openai_cloud(api_key="sk-...")
    pipeline = RAGPipeline(store, config)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .config import RAGConfig, LLMConfig
from .context_assembler import AssembledContext, ContextAssembler
from .embedding_manager import EmbeddingManager
from .generation_evaluation import EvaluationReport, GenerationEvaluationManager
from .generation_manager import Citation, GenerationManager, GenerationResult
from .query_manager import QueryManager
from .semantic_retrievers import register_semantic_retrievers
from .vector_index_manager import VectorIndexManager


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    query: str
    answer: str
    citations: List[Citation]
    context: AssembledContext
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    generation_result: GenerationResult
    evaluation: Optional[EvaluationReport]
    latency: Dict[str, float]           # query_ms, retrieve_ms, rerank_ms, assemble_ms, generate_ms, total_ms
    config_snapshot: Dict[str, Any]

    def __str__(self) -> str:
        lines = [
            f"Query: {self.query}",
            "",
            f"Answer:\n{self.answer}",
            "",
            f"Citations ({len(self.citations)}):",
        ]
        for c in self.citations:
            lines.append(f"  [{c.number}] {c.source_label}")
            if c.quote:
                lines.append(f"       \"{c.quote[:120]}\"")
        if self.evaluation:
            lines.append("")
            lines.append(
                f"Evaluation: faithfulness={self.evaluation.faithfulness:.2f} "
                f"hallucination_risk={self.evaluation.hallucination_risk:.2f} "
                f"overall={self.evaluation.overall_score:.2f}"
            )
        lines.append("")
        total = self.latency.get("total_ms", 0)
        lines.append(f"Latency: {total:.0f}ms total")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """Complete medical RAG pipeline.

    Initialisation
    --------------
    The pipeline is lazy: heavy components (embedding model, FAISS index, LLM)
    are loaded on first use. Call ``prepare()`` to pre-warm all components.
    """

    def __init__(
        self,
        store,
        config: RAGConfig,
        auto_build: bool = False,
    ) -> None:
        self.store = store
        self.config = config
        self._auto_build = auto_build

        # Components — all lazy
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._vector_index_manager: Optional[VectorIndexManager] = None
        self._retriever_manager = None
        self._reranker_manager = None
        self._context_assembler: Optional[ContextAssembler] = None
        self._generation_manager: Optional[GenerationManager] = None
        self._query_manager: Optional[QueryManager] = None
        self._evaluator: Optional[GenerationEvaluationManager] = None

        self._retriever_initialized = False

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    def _get_embedding_manager(self) -> EmbeddingManager:
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager(self.store, self.config.embedding)
        return self._embedding_manager

    def _get_vector_index_manager(self) -> VectorIndexManager:
        if self._vector_index_manager is None:
            self._vector_index_manager = VectorIndexManager(
                self.store, self._get_embedding_manager(), self.config.vector_index
            )
        return self._vector_index_manager

    def _get_retriever_manager(self):
        if not self._retriever_initialized:
            from .retriever_manager import RetrieverManager, register_default_retrievers
            rm = RetrieverManager(self.store)
            register_default_retrievers(rm)
            register_semantic_retrievers(
                rm,
                self._get_embedding_manager(),
                self._get_vector_index_manager(),
                alpha=self.config.retrieval.hybrid_alpha,
                llm_config=self.config.llm if self.config.retrieval.use_hyde else None,
            )
            self._retriever_manager = rm
            self._retriever_initialized = True
        return self._retriever_manager

    def _get_reranker_manager(self):
        if self._reranker_manager is None:
            from .reranker_manager import RerankerManager, register_default_rerankers
            rm = RerankerManager(self.store)
            register_default_rerankers(rm)
            self._reranker_manager = rm
        return self._reranker_manager

    def _get_context_assembler(self) -> ContextAssembler:
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler(self.store, self.config.generation)
        return self._context_assembler

    def _get_generation_manager(self) -> GenerationManager:
        if self._generation_manager is None:
            self._generation_manager = GenerationManager(self.config.llm, self.config.generation)
        return self._generation_manager

    def _get_query_manager(self) -> QueryManager:
        if self._query_manager is None:
            llm_config = self.config.llm if self.config.retrieval.use_hyde else None
            self._query_manager = QueryManager(llm_config=llm_config)
        return self._query_manager

    def _get_evaluator(self) -> GenerationEvaluationManager:
        if self._evaluator is None:
            self._evaluator = GenerationEvaluationManager()
        return self._evaluator

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare(
        self,
        chunk_strategy_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Pre-build embeddings and vector indexes for all (or specified) strategies.

        Call this once after ingestion before serving queries.
        """
        strategy = chunk_strategy_name or self.config.retrieval.chunk_strategy_name
        em = self._get_embedding_manager()
        vim = self._get_vector_index_manager()

        embed_results: Dict[str, Any] = {}
        index_results: Dict[str, Any] = {}

        if not em.is_embedded(strategy) or overwrite:
            embed_results[strategy] = str(em.embed_chunks(strategy, overwrite=overwrite))
        else:
            embed_results[strategy] = "already_embedded"

        if not vim.is_built(strategy) or overwrite:
            index_results[strategy] = str(vim.build(strategy, overwrite=overwrite))
        else:
            index_results[strategy] = "already_built"

        return {"embeddings": embed_results, "vector_indexes": index_results}

    def prepare_all(self, overwrite: bool = False) -> Dict[str, Any]:
        """Embed and index all available chunk strategies."""
        em = self._get_embedding_manager()
        vim = self._get_vector_index_manager()
        embed_results = em.embed_all_strategies(overwrite=overwrite)
        index_results = vim.build_all(overwrite=overwrite)
        return {"embeddings": embed_results, "vector_indexes": index_results}

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        stream: bool = False,
        evaluate: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Union[RAGResponse, Iterator[str]]:
        """Run the full pipeline and return a RAGResponse.

        Parameters
        ----------
        query              : clinical question
        top_k_retrieve     : override config.retrieval.top_k_retrieve
        top_k_rerank       : override config.retrieval.top_k_rerank
        max_context_tokens : override config.generation.max_context_tokens
        stream             : if True, return a token iterator (skips evaluation)
        evaluate           : if True, compute generation evaluation metrics
        filters            : optional metadata filters (e.g. {"contains_medication": True})
        """
        t_start = time.time()
        latency: Dict[str, float] = {}

        rc = self.config.retrieval
        gc = self.config.generation
        k_retrieve = top_k_retrieve or rc.top_k_retrieve
        k_rerank = top_k_rerank or rc.top_k_rerank
        chunk_strategy = rc.chunk_strategy_name

        # 1. Query understanding
        t0 = time.time()
        qm = self._get_query_manager()
        query_info = qm.process(
            query,
            use_hyde=(rc.use_hyde and not stream),
        )
        expanded_queries = query_info["expanded"] if rc.query_expansion else [query]
        latency["query_ms"] = round((time.time() - t0) * 1000, 1)

        # 2. Retrieve
        t0 = time.time()
        retrieved = self._retrieve(
            queries=expanded_queries,
            chunk_strategy=chunk_strategy,
            top_k=k_retrieve,
            filters=filters,
        )
        latency["retrieve_ms"] = round((time.time() - t0) * 1000, 1)

        # 3. Rerank
        t0 = time.time()
        reranked = self._rerank(
            results=retrieved,
            query=query,
            top_k=k_rerank,
        )
        latency["rerank_ms"] = round((time.time() - t0) * 1000, 1)

        # 4. Assemble context
        t0 = time.time()
        assembler = self._get_context_assembler()
        context = assembler.assemble(
            reranked,
            query=query,
            max_tokens=max_context_tokens or gc.max_context_tokens,
        )
        latency["assemble_ms"] = round((time.time() - t0) * 1000, 1)

        # 5. Generate (or stream)
        t0 = time.time()
        gen = self._get_generation_manager()

        if stream:
            return gen.answer(query, context, stream=True)

        gen_result: GenerationResult = gen.answer(
            query,
            context,
            min_citation_confidence=gc.min_citation_confidence,
        )
        latency["generate_ms"] = round((time.time() - t0) * 1000, 1)
        latency["total_ms"] = round((time.time() - t_start) * 1000, 1)

        # 6. Optional evaluation
        eval_report: Optional[EvaluationReport] = None
        if evaluate:
            eval_report = self._get_evaluator().evaluate(gen_result, context)

        config_snapshot = {
            "embedding_model": self.config.embedding.model_name,
            "llm_model": self.config.llm.model_name,
            "retrievers": rc.retriever_names,
            "reranker": rc.reranker_name,
            "chunk_strategy": chunk_strategy,
        }

        return RAGResponse(
            query=query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            context=context,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            generation_result=gen_result,
            evaluation=eval_report,
            latency=latency,
            config_snapshot=config_snapshot,
        )

    # ------------------------------------------------------------------
    # Full-document answer (bypasses retrieval/reranking)
    # ------------------------------------------------------------------

    def answer_full_document(
        self,
        query: str,
        chunk_strategy: str = "sentence_based",
    ) -> RAGResponse:
        """Generate an answer from ALL chunks in a strategy, ordered by page.

        No BM25 / dense retrieval or reranking — every sentence in the document
        is included in context order (chronological).  Ideal for narrative
        summaries where completeness matters more than relevance ranking.
        """
        t0 = time.time()

        # Load enriched chunks when available; fall back to plain chunks
        enriched_path = self.store.paths.chunks_enriched_dir / f"{chunk_strategy}.parquet"
        plain_path    = self.store.paths.chunks_dir / f"{chunk_strategy}.parquet"

        from .chunk_manager import read_parquet_mmap
        if enriched_path.exists():
            rows: List[Dict[str, Any]] = read_parquet_mmap(enriched_path).to_pylist()
        elif plain_path.exists():
            rows = read_parquet_mmap(plain_path).to_pylist()
        else:
            raise FileNotFoundError(
                f"No chunks parquet found for strategy '{chunk_strategy}'. "
                "Run the chunker first."
            )

        # Chronological order: page first, then chunk_id (= sentence_index)
        rows.sort(key=lambda r: (int(r.get("page_start") or 0), int(r.get("chunk_id") or 0)))

        for r in rows:
            r.setdefault("score", 1.0)
            r.setdefault("rerank_score", 1.0)
            r.setdefault("retriever_name", "full_document")
            # Ensure strategy name propagates to ContextChunk
            r.setdefault("chunk_strategy_name", chunk_strategy)

        gc = self.config.generation
        assembler = self._get_context_assembler()
        context = assembler.assemble(rows, query=query, max_tokens=gc.max_context_tokens)

        gen = self._get_generation_manager()
        gen_result: GenerationResult = gen.answer(
            query,
            context,
            min_citation_confidence=gc.min_citation_confidence,
        )

        latency_ms = round((time.time() - t0) * 1000, 1)
        return RAGResponse(
            query=query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            context=context,
            retrieved_chunks=rows,
            reranked_chunks=rows,
            generation_result=gen_result,
            evaluation=None,
            latency={"full_document_ms": latency_ms, "total_ms": latency_ms},
            config_snapshot={"chunk_strategy": chunk_strategy, "mode": "full_document"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        queries: List[str],
        chunk_strategy: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rm = self._get_retriever_manager()
        rc = self.config.retrieval
        all_results: List[Dict[str, Any]] = []
        seen: Dict[tuple, float] = {}

        primary_query = queries[0]

        for retriever_name in rc.retriever_names:
            if retriever_name not in rm.list_retrievers():
                continue
            for q in queries:
                try:
                    results = rm.retrieve(
                        retriever_name=retriever_name,
                        query=q,
                        chunk_strategy_name=chunk_strategy,
                        top_k=top_k,
                        filters=filters,
                    )
                    for r in results:
                        key = (str(r.get("chunk_strategy_name")), str(r.get("document_id")), int(r.get("chunk_id") or 0))
                        score = float(r.get("score") or 0.0)
                        if key not in seen or score > seen[key]:
                            seen[key] = score
                            r["query"] = primary_query
                            all_results.append(r)
                except Exception:
                    continue

        # Deduplicate keeping highest score
        deduped: Dict[tuple, Dict[str, Any]] = {}
        for r in all_results:
            key = (str(r.get("chunk_strategy_name")), str(r.get("document_id")), int(r.get("chunk_id") or 0))
            score = float(r.get("score") or 0.0)
            if key not in deduped or score > float(deduped[key].get("score") or 0.0):
                deduped[key] = r

        results = sorted(deduped.values(), key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return results[:top_k * 2]

    def _rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []
        rm = self._get_reranker_manager()
        reranker_name = self.config.retrieval.reranker_name
        if reranker_name not in rm.list_rerankers():
            reranker_name = "keyword_overlap"
        try:
            return rm.rerank(reranker_name=reranker_name, query=query, results=results, top_k=top_k)
        except Exception:
            return results[:top_k]

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def local_ollama(cls, store, auto_build: bool = False) -> "RAGPipeline":
        """Self-hosted: fastembed (nomic-embed-v1.5) + Ollama (llama3.1:8b)."""
        return cls(store, RAGConfig.local_ollama(), auto_build=auto_build)

    @classmethod
    def openai_cloud(
        cls,
        store,
        api_key: Optional[str] = None,
        auto_build: bool = False,
    ) -> "RAGPipeline":
        """Cloud: text-embedding-3-large + gpt-4o."""
        return cls(store, RAGConfig.openai_cloud(api_key=api_key), auto_build=auto_build)

    @classmethod
    def fast_local(cls, store, auto_build: bool = False) -> "RAGPipeline":
        """Fast local: bge-small + Ollama mistral:7b (dev/testing)."""
        return cls(store, RAGConfig.fast_local(), auto_build=auto_build)

    @classmethod
    def medical_precise(cls, store, auto_build: bool = False) -> "RAGPipeline":
        """High-accuracy: bge-large + large LLM, HyDE, wide retrieval."""
        return cls(store, RAGConfig.medical_precise(), auto_build=auto_build)

    @classmethod
    def from_config_file(cls, store, path: str, auto_build: bool = False) -> "RAGPipeline":
        """Load from a JSON config file (e.g., configs/local_ollama.json)."""
        return cls(store, RAGConfig.from_file(path), auto_build=auto_build)

    @classmethod
    def from_config(cls, store, config: RAGConfig, auto_build: bool = False) -> "RAGPipeline":
        return cls(store, config, auto_build=auto_build)

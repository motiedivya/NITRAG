from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nitrag.chunk_manager import PdfTokenStore, ChunkManager, register_default_chunkers
from nitrag.chunking_evaluation import ChunkingEvaluationManager
from nitrag.chunk_metadata_enrichment_evaluation import ChunkMetadataEnrichmentEvaluationManager
from nitrag.chunk_metadata_enricher import ChunkMetadataEnricher
from nitrag.clinical_metadata_extractor import ClinicalMetadataExtractor
from nitrag.document_metadata_extractor import PyMuPDFLayoutExtractor
from nitrag.final_evaluation import FinalEvaluationManager
from nitrag.index_manager import IndexManager, register_default_indexers, read_parquet
from nitrag.indexing_evaluation import IndexingEvaluationManager
from nitrag.reranker_manager import RerankerManager, register_default_rerankers
from nitrag.retriever_manager import RetrieverManager, register_default_retrievers
from nitrag.rag_diagnostics_manager import RAGDiagnosticsManager
# Semantic + generation stages
from nitrag.config import RAGConfig
from nitrag.embedding_manager import EmbeddingManager
from nitrag.vector_index_manager import VectorIndexManager
from nitrag.semantic_retrievers import register_semantic_retrievers
from nitrag.context_assembler import ContextAssembler
from nitrag.generation_manager import GenerationManager
from nitrag.generation_evaluation import GenerationEvaluationManager
from nitrag.rag_pipeline import RAGPipeline


PDF_PATH = PROJECT_ROOT / "data" / "Visit Note - 10-14-2020.pdf"
RAG_STORE = PROJECT_ROOT / "rag_store"


def main() -> None:
    # Document layout metadata extraction.
    extractor = PyMuPDFLayoutExtractor(
        encoding_model_name="gpt-4o",
        root_dir=RAG_STORE,
    )
    manifest = extractor.extract(PDF_PATH, overwrite=True)
    doc_dir = Path(manifest["paths"]["document_dir"])

    pages = pq.read_table(doc_dir / "layout_pages.parquet").to_pylist()
    elements = pq.read_table(doc_dir / "layout_elements.parquet").to_pylist()
    spans = pq.read_table(doc_dir / "layout_spans.parquet").to_pylist()
    words = pq.read_table(doc_dir / "layout_words.parquet").to_pylist()
    print("layout:", len(pages), len(elements), len(spans), len(words))

    # Clinical metadata extraction.
    clinical_extractor = ClinicalMetadataExtractor(doc_dir)
    clinical_result = clinical_extractor.run()
    print("clinical:", clinical_result)

    # Token store and chunking.
    store = PdfTokenStore(
        encoding_model_name="gpt-4o",
        root_dir=RAG_STORE,
    )
    store.ingest_pdf(PDF_PATH, overwrite=True)

    manager = ChunkManager(store)
    register_default_chunkers(manager)
    manager.execute_all(continue_on_error=True)
    print("chunk errors:", manager.errors())

    # First-stage chunking evaluation.
    chunk_eval = ChunkingEvaluationManager(store)
    chunking_summary = chunk_eval.generate_report()
    print("chunking report:", chunking_summary["report_dir"])
    chunking_metrics = pd.read_csv(chunk_eval.metrics_dir / "chunking_metrics.csv")
    print(chunking_metrics.sort_values(
        ["coverage_pct", "redundancy_factor", "median_tokens"],
        ascending=[False, True, True],
    ))

    # Metadata enrichment.
    enricher = ChunkMetadataEnricher(store.paths.document_dir)
    enrichment_outputs = enricher.enrich_all(overwrite=True)
    print("enriched chunks:", enrichment_outputs)

    enrichment_eval = ChunkMetadataEnrichmentEvaluationManager(store)
    enrichment_eval_summary = enrichment_eval.generate_report()
    print("chunk metadata enrichment evaluation:", enrichment_eval_summary["report_dir"])
    enrichment_metrics = pd.read_csv(enrichment_eval.metrics_dir / "enrichment_metrics.csv")
    print(enrichment_metrics.sort_values(
        ["source_entity_recall_pct", "avg_quality_score"],
        ascending=[False, False],
    ))

    # Indexing.
    index_manager = IndexManager(store=store, use_enriched_chunks=True)
    register_default_indexers(index_manager)
    print(index_manager.list_indexers(with_descriptions=True))
    print(index_manager.list_chunk_strategies())
    index_outputs = index_manager.execute_all(continue_on_error=True, overwrite=True)
    print("index outputs:", index_outputs)

    indexing_eval = IndexingEvaluationManager(store)
    indexing_eval_summary = indexing_eval.generate_report()
    print("indexing evaluation:", indexing_eval_summary["report_dir"])
    indexing_metrics = pd.read_csv(indexing_eval.metrics_dir / "index_metrics.csv")
    print(indexing_metrics[[
        "chunk_strategy",
        "index_name",
        "docs_rows",
        "postings_rows",
        "vocab_rows",
        "total_size_bytes",
    ]].sort_values(["chunk_strategy", "index_name"]).head(30))
    indexing_scorecard = pd.read_csv(indexing_eval.metrics_dir / "index_scorecard.csv")
    print(indexing_scorecard.sort_values(
        ["health_score", "chunk_strategy", "index_name"],
        ascending=[True, True, True],
    ).head(20))

    # Example index reads from the notebook.
    bm25_docs = read_parquet(store.paths.document_dir / "indexes" / "block_group_800_overlap_1" / "bm25" / "docs.parquet")
    bm25_vocab = read_parquet(store.paths.document_dir / "indexes" / "block_group_800_overlap_1" / "bm25" / "vocab.parquet")
    bm25_postings = read_parquet(store.paths.document_dir / "indexes" / "block_group_800_overlap_1" / "bm25" / "postings.parquet")
    print("bm25 rows:", len(bm25_docs), len(bm25_vocab), len(bm25_postings))

    # Retrieval.
    retriever_manager = RetrieverManager(store)
    register_default_retrievers(retriever_manager)
    print(retriever_manager.list_retrievers(with_descriptions=True))
    print(retriever_manager.list_chunk_strategies())

    reranker_manager = RerankerManager(store)
    register_default_rerankers(reranker_manager)
    print(reranker_manager.list_rerankers(with_descriptions=True))

    examples = [
        ("bm25", "pain medication follow up", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("tfidf", "pain medication follow up", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("phrase_ngram", "follow up medication", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("char_ngram", "medicatoin dose pain", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("fielded_lexical", "medication pain diagnosis", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("boolean_set", "pain medication follow", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5, "require_all_terms": False}),
        ("positional_proximity", "pain medication follow", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5, "proximity_window": 20}),
        ("advanced_lexical_fusion", "medication dose pain follow up", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("graph_expansion", "pain medication follow up", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 5}),
        ("bm25_metadata_boost", "medication dose pain", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 10, "preferred_flags": ["contains_medication"]}),
        ("metadata_filter", "", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 10, "filters": {"contains_medication": True}}),
        ("section_page", "", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 10, "pages": [1]}),
        ("layout_spatial", "", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 10, "pages": [1], "zones": ["middle"]}),
        ("temporal", "", {"chunk_strategy_name": "block_group_800_overlap_1", "top_k": 10}),
        ("minhash_duplicates", "pain medication follow up", {"chunk_strategy_name": "line_based_debug", "top_k": 10}),
        ("cross_chunk_fusion", "diagnosis pain medication follow up", {"top_k": 10}),
    ]
    for retriever_name, query, kwargs in examples:
        results = retriever_manager.retrieve(retriever_name=retriever_name, query=query, **kwargs)
        print("retrieval:", retriever_name, "results:", len(results))
        for r in results[:3]:
            print(r.get("score"), r.get("chunk_strategy_name"), r.get("chunk_id"), r.get("page_start"), r.get("page_end"))

    rerank_query = "medication dose pain follow up"
    rerank_candidates = retriever_manager.retrieve(
        retriever_name="advanced_lexical_fusion",
        query=rerank_query,
        chunk_strategy_name="block_group_800_overlap_1",
        top_k=20,
    )
    for reranker_name in ["keyword_overlap", "phrase_proximity", "metadata_quality", "clinical_intent", "hybrid_weighted", "diversity_mmr"]:
        reranked = reranker_manager.rerank(
            reranker_name=reranker_name,
            query=rerank_query,
            results=rerank_candidates,
            top_k=5,
        )
        print("rerank:", reranker_name, "results:", len(reranked))
        for r in reranked[:3]:
            print(r.get("rerank_score"), r.get("original_score"), r.get("chunk_strategy_name"), r.get("chunk_id"))

    # Static and retrieval diagnostics.
    diag = RAGDiagnosticsManager(
        store=store,
        retriever_manager=retriever_manager,
        use_enriched_chunks=True,
    )
    static_summary = diag.generate_static_report()
    print("diagnostics report:", static_summary["report_dir"])

    query_suite = [
        {
            "query": "medication dose pain follow up",
            "expected_keywords": ["pain", "medication", "follow"],
            "preferred_flags": ["contains_medication"],
        },
        {
            "query": "diagnosis assessment impression",
            "expected_keywords": ["diagnosis", "assessment", "impression"],
            "preferred_flags": ["contains_diagnosis"],
        },
        {
            "query": "vital signs blood pressure temperature",
            "expected_keywords": ["blood", "pressure", "temperature"],
            "preferred_flags": ["contains_vital"],
        },
    ]

    final_eval = FinalEvaluationManager(
        store=store,
        retriever_manager=retriever_manager,
        reranker_manager=reranker_manager,
    )
    final_summary = final_eval.generate_report(
        query_suite=query_suite,
        retriever_names=[
            "bm25",
            "tfidf",
            "keyword_exact",
            "phrase_ngram",
            "char_ngram",
            "fielded_lexical",
            "boolean_set",
            "positional_proximity",
            "bm25_metadata_boost",
            "multi_query_bm25",
            "advanced_lexical_fusion",
            "mmr_diversity",
            "graph_expansion",
            "layout_spatial",
        ],
        chunk_strategy_names=[
            "fixed_512",
            "fixed_512_overlap_100",
            "block_group_800_overlap_1",
            "line_group_512_overlap_2",
            "hierarchical_child_256_parent_1200",
        ],
        reranker_names=[
            "baseline",
            "keyword_overlap",
            "phrase_proximity",
            "metadata_quality",
            "clinical_intent",
            "hybrid_weighted",
            "diversity_mmr",
            "deduplicate",
        ],
        candidate_k=20,
        top_k=10,
    )
    print("final evaluation:", final_summary["report_dir"])
    pipeline_rankings = pd.read_csv(final_eval.metrics_dir / "pipeline_rankings.csv")
    print(pipeline_rankings.head(30))

    # ── Stage 6: Embedding ────────────────────────────────────────────────
    print("\n=== Stage 6: Embedding ===")
    rag_config = RAGConfig.local_ollama()   # swap to .openai_cloud() for cloud
    embedding_manager = EmbeddingManager(store, rag_config.embedding)
    embed_results = embedding_manager.embed_all_strategies(use_enriched=True, overwrite=False)
    print("embedding results:", embed_results)
    print("embedded strategies:", embedding_manager.list_embedded_strategies())

    # ── Stage 6b: Vector Indexing ─────────────────────────────────────────
    print("\n=== Stage 6b: Vector Indexing ===")
    vector_index_manager = VectorIndexManager(store, embedding_manager, rag_config.vector_index)
    vim_results = vector_index_manager.build_all(overwrite=False)
    print("vector index results:", vim_results)
    print("indexed strategies:", vector_index_manager.list_built_strategies())

    # Register semantic retrievers into existing retriever_manager
    register_semantic_retrievers(
        retriever_manager,
        embedding_manager,
        vector_index_manager,
        alpha=rag_config.retrieval.hybrid_alpha,
    )
    print("retrievers after semantic registration:", retriever_manager.list_retrievers())

    # ── Stage 9–11: End-to-end RAG pipeline ──────────────────────────────
    print("\n=== Stage 9–11: End-to-end RAG pipeline ===")
    pipeline = RAGPipeline(store, rag_config)

    test_queries = [
        "What medications were prescribed and at what dose?",
        "What is the primary diagnosis or clinical impression?",
        "What were the patient's vital signs?",
    ]

    evaluator = GenerationEvaluationManager()
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            response = pipeline.answer(query, evaluate=True)
            print(f"Answer: {response.answer[:300]}...")
            print(f"Citations: {len(response.citations)}")
            for c in response.citations[:3]:
                print(f"  [{c.number}] {c.source_label}  confidence={c.confidence:.2f}")
            if response.evaluation:
                ev = response.evaluation
                print(
                    f"Eval: faithfulness={ev.faithfulness:.2f}  "
                    f"hallucination_risk={ev.hallucination_risk:.2f}  "
                    f"overall={ev.overall_score:.2f}"
                )
            print(f"Latency: {response.latency.get('total_ms', 0):.0f}ms")
        except Exception as e:
            print(f"  [skipped — LLM unavailable: {e}]")


if __name__ == "__main__":
    main()

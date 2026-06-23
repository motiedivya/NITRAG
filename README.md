# NIT-RAG

File-based version of `NITRAG.ipynb`, split by notebook section.

## Layout

- `data/Visit Note - 10-14-2020.pdf` - copy of the PDF used by the notebook.
- `nitrag/document_metadata_extractor.py` - Document MetadataExtractor.
- `nitrag/final_evaluation.py` - cross-stage final metrics and full pipeline rankings.
- `nitrag/clinical_metadata_extractor.py` - Clinical MetadataExtractor v1.
- `nitrag/chunk_manager.py` - ChunkManager and chunking strategies.
- `nitrag/chunking_evaluation.py` - first-stage chunking metrics and plots.
- `nitrag/chunk_metadata_enricher.py` - ChunkMetadataEnricher.
- `nitrag/chunk_metadata_enrichment_evaluation.py` - focused evaluation for enriched chunk metadata.
- `nitrag/index_manager.py` - IndexManager and lexical, TF-IDF, phrase, character n-gram, fielded, metadata, entity, section/page, graph, positional, boolean, temporal, layout-spatial, and MinHash LSH index strategies.
- `nitrag/indexing_evaluation.py` - focused validation and metrics for persisted indexing outputs.
- `nitrag/reranker_manager.py` - post-retrieval rerankers for keyword, phrase proximity, metadata quality, clinical intent, length, recency, diversity, deduplication, and hybrid scoring.
- `nitrag/reranking_evaluation.py` - benchmark and plots for reranker quality, rank movement, diversity, and latency.
- `nitrag/retriever_manager.py` - RetrieverManager and BM25, TF-IDF, exact keyword, phrase, character n-gram, fielded, boolean, positional proximity, entity, section/page, temporal, layout-spatial, MinHash duplicate, graph expansion, and fusion retrieval strategies.
- `nitrag/rag_diagnostics_manager.py` - static and retrieval diagnostics.
- `scripts/run_pipeline.py` - runnable pipeline equivalent to the notebook flow.

## Run

From this directory, using the venv from `/home/neuralit`:

```bash
/home/neuralit/.venv/bin/python scripts/run_pipeline.py
```

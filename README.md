# NIT-RAG

A production-grade, generic RAG system that implements all major strategies at each pipeline stage. The goal is to eventually drive an orchestrator/agent that selects the optimal RAG configuration for **Medical Data** workloads automatically.

---

## Vision

Most RAG systems pick one strategy per stage and bake it in. NIT-RAG registers *all* known strategies at every stage (chunking, indexing, retrieval, reranking, etc.) so a future orchestrator can experiment, benchmark, and select the best configuration per document type and query intent — specifically targeting clinical/medical documents.

The system is intentionally modular: each stage can be developed, evaluated, and swapped independently. Evaluation plots and metrics are planned for every stage so improvements can be measured objectively.

---

## Pipeline Stages

| # | Stage | Status | Notes |
|---|-------|--------|-------|
| 1 | **Data Ingestion** | ✅ Complete (PDF) | Full PDF ingestion via `PDFIngestionPipeline`: layout extraction, text normalisation, page-type detection, column detection, reading-order correction, OCR-aware heading detection. Non-PDF formats out of scope for now. |
| 2 | **Metadata Extraction** | ✅ Complete | Document layout metadata + clinical metadata (entities, sections, dates, providers) |
| 3 | **Chunking** | ✅ Complete | 8+ strategies: fixed-token, page, page-window, element, block-group, line-group, hierarchical, debug |
| 4 | **Metadata Enrichment** | ✅ Complete | Enriches chunks with clinical entities, sections, layout zones, quality scores |
| 5 | **Indexing (Lexical)** | ✅ Complete | 14 strategies: BM25, TF-IDF, phrase/char n-gram, fielded, entity, section/page, graph, positional, boolean, temporal, spatial, MinHash LSH |
| 6 | **Embedding** | 🔴 Not Started | Config present (`text-embedding-3-small`) but no implementation; needed for semantic/vector retrieval |
| 6b | **Vector Indexing** | 🔴 Not Started | Depends on Embedding; FAISS, Chroma, Qdrant, or similar vector store integration |
| 7 | **Retrieval** | 🟡 Partial | 22+ lexical strategies (BM25, fusion, MMR, graph expansion, etc.); **zero semantic/vector retrieval** — blocked on Embedding |
| 8 | **Reranking** | ✅ Complete | 10 strategies: keyword overlap, phrase proximity, metadata quality, clinical intent, length penalty, recency, diversity MMR, deduplication, hybrid |
| 9 | **Context Assembly** | 🔴 Not Started | Assembling retrieved + reranked chunks into a prompt context window (token budget management, ordering, deduplication) |
| 10 | **Generation** | 🔴 Not Started | LLM call with assembled context; streaming, structured output, citation tracking |
| 11 | **Query Understanding** | 🔴 Not Started | Query expansion, classification, intent detection, HyDE — needed before retrieval |
| 12 | **Evaluation Framework** | 🔴 Not Started | End-to-end RAGAS-style metrics: faithfulness, answer relevance, context precision/recall |

> Stage numbers 6b, 11, 12 are additions beyond the original 9-stage plan — they are gaps that surface once those stages are implemented.

---

## Architecture

```
nitrag/
├── document_metadata_extractor.py      # Stage 1+2: PDF layout extraction
├── clinical_metadata_extractor.py      # Stage 2: Clinical entities, sections, dates
├── chunk_manager.py                    # Stage 3: Chunking strategies + token store
├── chunking_evaluation.py              # Stage 3: Coverage, redundancy, boundary metrics
├── chunk_metadata_enricher.py          # Stage 4: Chunk enrichment with clinical metadata
├── chunk_metadata_enrichment_evaluation.py  # Stage 4: Enrichment quality metrics
├── index_manager.py                    # Stage 5: 14 lexical index strategies
├── indexing_evaluation.py              # Stage 5: Index health, size, postings metrics
├── retriever_manager.py                # Stage 7: 22+ retrieval strategies (lexical)
├── reranker_manager.py                 # Stage 8: 10 reranking strategies
├── reranking_evaluation.py             # Stage 8: Rank movement, diversity, latency
├── rag_diagnostics_manager.py          # Cross-stage static + retrieval diagnostics
└── final_evaluation.py                 # Cross-pipeline ranking and comparison
scripts/
└── run_pipeline.py                     # End-to-end pipeline runner
```

### Storage layout

```
rag_store/{doc_id}/
├── tokens.dat                          # Encoded token stream
├── layout_{pages,elements,spans,words}.parquet
├── layout_manifest.json
├── clinical_document_metadata.json
├── clinical_{sections,element_metadata,entities}.parquet
├── chunks/{strategy}.parquet
├── chunks_enriched/{strategy}.parquet
├── indexes/{chunk_strategy}/{index_name}/{docs,vocab,postings}.parquet
└── reports/{stage}/                    # Metrics + plots per evaluation run
```

---

## TODO

### Phase 1 — Data Ingestion

**PDF ingestion — complete** (`nitrag/pdf_ingestion.py`, `scripts/ingest_pdf.py`)

- [x] Layout extraction (pages, blocks, lines, spans, words, images, drawings)
- [x] Text normalisation (Unicode NFC, ligature expansion, zero-width char removal)
- [x] Page type detection: native vs. scanned/OCR'd vs. image-only
- [x] Multi-column layout detection + reading-order correction
- [x] OCR-aware heading detection (position/shape-based; works without font cues)
- [x] Per-page quality metrics (font variety, image area ratio, OCR quality proxy)
- [x] Standalone script: `scripts/ingest_pdf.py`

**Remaining (non-PDF formats — deferred)**

- [ ] Ingestion evaluation metrics and plots (page-type distribution, normalisation diff, heading accuracy)
- [ ] Multi-document batch ingestion runner
- [ ] Other formats: DOCX, plain text, HTML, CSV (separate format adapters, later phase)

### Phase 2 — Embedding

- [ ] Implement embedding generation (`text-embedding-3-small` baseline)
- [ ] Support multiple embedding models (OpenAI, HuggingFace sentence-transformers, BiomedBERT, ClinicalBERT)
- [ ] Chunk-level embedding with caching (avoid re-embedding unchanged chunks)
- [ ] Embedding evaluation: dimensionality, cosine similarity distributions, coverage

### Phase 3 — Vector Indexing

- [ ] FAISS flat/IVF index integration
- [ ] Chroma or Qdrant for persistent vector storage
- [ ] Hybrid index: lexical (BM25) + dense (FAISS) side-by-side under IndexManager
- [ ] Vector index evaluation: recall@k, index size, query latency

### Phase 4 — Semantic Retrieval

- [ ] Dense vector retrieval strategy in RetrieverManager
- [ ] Hybrid retrieval: lexical + semantic fusion (RRF and weighted score fusion)
- [ ] HyDE (Hypothetical Document Embeddings) retrieval
- [ ] Semantic retrieval evaluation: MRR, recall@k, latency

### Phase 5 — Query Understanding

- [ ] Query classification (factoid vs. summary vs. temporal vs. comparison)
- [ ] Query expansion (synonyms, medical term expansion via UMLS)
- [ ] HyDE query reformulation
- [ ] Query evaluation: expansion quality, downstream retrieval impact

### Phase 6 — Context Assembly

- [ ] Token-budget-aware context window builder
- [ ] Chunk ordering strategies (by score, by document order, by page)
- [ ] Deduplication across retrieved chunks before assembly
- [ ] Citation / provenance tracking per assembled context segment

### Phase 7 — Generation

- [ ] LLM integration (OpenAI `gpt-4o` baseline)
- [ ] Streaming response support
- [ ] Structured output (JSON schema for medical Q&A)
- [ ] Citation grounding — map each claim back to a source chunk
- [ ] Hallucination guard: faithfulness check against retrieved context

### Phase 8 — End-to-End Evaluation Framework

- [ ] RAGAS-style metrics: faithfulness, answer relevance, context precision, context recall
- [ ] Medical-domain metrics: clinical accuracy, entity grounding
- [ ] Per-stage latency breakdown
- [ ] Cross-pipeline comparison dashboard (matplotlib or similar)
- [ ] Golden test set for the medical domain

### Phase 9 — Orchestrator / Agent

- [ ] Hyperparameter space definition (which chunker × which indexer × which retriever × which reranker)
- [ ] Search strategy: grid search → Bayesian optimization → RL-based
- [ ] Objective function: downstream answer quality (RAGAS score) for medical queries
- [ ] Per-document-type profiles (Visit Note, Discharge Summary, Radiology, Lab)
- [ ] Configuration persistence and experiment tracking

---

## Run

```bash
# Using uv (preferred)
uv run scripts/run_pipeline.py

# Or using the project venv
/home/neuralit/.venv/bin/python scripts/run_pipeline.py
```

Requires a PDF at `data/` and the env vars from `.env.example`:

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (needed once Embedding / Generation stages are implemented)
```

---

## Production Infrastructure

When this system moves to production it will run on the following stack. All choices follow a **reliable + KISS** principle — no tool is added unless it is well-proven and operationally simple.

| Concern | Tool | Role |
|---------|------|------|
| Queue management | **NSQ** | Async document ingestion jobs, pipeline task dispatch |
| API layer | **FastAPI** | REST endpoints for ingestion, retrieval, generation |
| Real-time push | **Centrifugo** | WebSocket delivery of streaming generation responses |
| Caching | **Redis** | Embedding cache, query result cache, session state |
| Operational data | **MongoDB** | Document metadata, pipeline run records, user/org data (default; re-evaluate if relational needs arise) |
| Vector storage | **TBD** | Embedding index — candidates: Qdrant (self-hosted, reliable, easy), pgvector (zero extra infra if Postgres is already present) |

> Tool selection philosophy: only pick tools that are widely battle-tested and have a low operational surface area. When two tools solve the same problem equally well, pick the simpler one.

---

## Design Principles

- **One strategy per class** — every chunker, indexer, retriever, reranker is a self-contained class registered with a manager. Adding a new strategy never changes existing ones.
- **Parquet-native storage** — all data lives in typed Parquet files for efficient I/O, schema enforcement, and compatibility with downstream tools.
- **Token-indexed architecture** — chunks track start/end token positions so any stage can reconstruct the exact document span.
- **Evaluation at every stage** — no stage is considered complete until it ships metrics and plots that let you see whether a strategy is better or worse than the baseline.

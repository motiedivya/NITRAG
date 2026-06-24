# NITRAG — Medical Evidence RAG

A production-grade, configuration-driven RAG system built around clinical documents. Every strategy at every pipeline stage is registered and benchmarked, so a future orchestrator can select the best configuration per document type and query intent automatically.

**Current state:** all 12 pipeline stages are implemented and passing 486 tests. The system ships with a web UI.

---

## Quick start

```bash
# Install dependencies
uv sync

# Process a document (stages 1–6b)
uv run python scripts/run_pipeline.py

# Start the web UI
uv run python scripts/start_server.py
# → http://localhost:8000
```

Requires Python ≥ 3.12 and `uv`. For LLM-backed generation, Ollama must be running locally:

```bash
ollama serve
ollama pull llama3.1:8b        # default generation model
```

---

## Pipeline stages

| # | Stage | Module | Notes |
|---|-------|--------|-------|
| 1 | **PDF Ingestion** | `document_metadata_extractor.py` | Layout extraction, reading-order correction, page-type detection, OCR-aware heading detection |
| 2 | **Clinical Metadata** | `clinical_metadata_extractor.py` | Entity extraction (medications, diagnoses, labs, vitals), section detection, date normalisation |
| 3 | **Chunking** | `chunk_manager.py` | 8+ strategies: fixed-token, page, page-window, element, block-group, line-group, hierarchical, debug |
| 3e | **Chunk Evaluation** | `chunking_evaluation.py` | Coverage, redundancy, boundary quality, token-length distribution |
| 4 | **Metadata Enrichment** | `chunk_metadata_enricher.py` | Attaches clinical entities, sections, layout zones, quality scores to each chunk |
| 4e | **Enrichment Evaluation** | `chunk_metadata_enrichment_evaluation.py` | Entity recall, quality score distributions |
| 5 | **Lexical Indexing** | `index_manager.py` | 14 strategies: BM25, TF-IDF, phrase/char n-gram, fielded, entity, section/page, graph, positional, boolean, temporal, spatial, MinHash LSH |
| 5e | **Index Evaluation** | `indexing_evaluation.py` | Postings coverage, vocabulary size, index health |
| 6 | **Embeddings** | `embedding_manager.py` | fastembed (ONNX, no PyTorch) default; OpenAI and sentence-transformers also supported. Default: `nomic-ai/nomic-embed-text-v1.5` |
| 6b | **Vector Index** | `vector_index_manager.py` | FAISS flat (exact cosine) and HNSW (ANN). L2-normalised vectors for cosine via inner product |
| 7 | **Retrieval** | `retriever_manager.py` + `semantic_retrievers.py` | 22+ lexical strategies + dense, hybrid (BM25+dense RRF), HyDE |
| 8 | **Reranking** | `reranker_manager.py` | 10 strategies: keyword overlap, phrase proximity, metadata quality, clinical intent, length penalty, recency, MMR diversity, deduplication, hybrid |
| 9 | **Query Understanding** | `query_manager.py` | Medical abbreviation expansion (90+ terms), query classification (6 types), HyDE passage generation |
| 10 | **Context Assembly** | `context_assembler.py` | Token-budget assembly, citation numbering, page/score/mixed ordering, evidence block formatting |
| 11 | **Generation** | `generation_manager.py` | OpenAI-compatible provider (Ollama, vLLM, LMStudio, Groq, OpenAI) + Anthropic. 7-rule medical citation prompt |
| 11e | **Generation Evaluation** | `generation_evaluation.py` | Faithfulness, answer relevance, context precision, citation coverage, hallucination risk, context recall |
| 12 | **End-to-end Pipeline** | `rag_pipeline.py` | Orchestrates stages 9→11e. Preset factories: `local_ollama()`, `openai_cloud()`, `fast_local()`, `medical_precise()` |
| — | **Web UI** | `server.py` + `ui/index.html` | FastAPI server + SPA with citation-first evidence display |

---

## Architecture

```
nitrag/
├── document_metadata_extractor.py       # Stage 1: PDF layout extraction
├── clinical_metadata_extractor.py       # Stage 2: Clinical entities, sections, dates
├── pdf_ingestion.py                     # Stage 1 low-level: PDFIngestionPipeline
├── chunk_manager.py                     # Stage 3: Chunking strategies + PdfTokenStore
├── chunking_evaluation.py               # Stage 3e: Coverage, redundancy, boundary metrics
├── chunk_metadata_enricher.py           # Stage 4: Chunk enrichment
├── chunk_metadata_enrichment_evaluation.py  # Stage 4e: Enrichment quality metrics
├── index_manager.py                     # Stage 5: 14 lexical index strategies
├── indexing_evaluation.py               # Stage 5e: Index health + postings metrics
├── config.py                            # RAGConfig + 5 sub-configs + preset factories
├── embedding_manager.py                 # Stage 6: fastembed / OpenAI / sentence-transformers
├── vector_index_manager.py              # Stage 6b: FAISS flat + HNSW
├── query_manager.py                     # Stage 9: Query expansion + classification + HyDE
├── retriever_manager.py                 # Stage 7: 22+ lexical retrieval strategies
├── semantic_retrievers.py               # Stage 7: Dense, hybrid (RRF), HyDE retrieval
├── reranker_manager.py                  # Stage 8: 10 reranking strategies
├── reranking_evaluation.py              # Stage 8: Rank movement, diversity, latency
├── context_assembler.py                 # Stage 10: Token-budget context + citation numbering
├── generation_manager.py                # Stage 11: LLM generation + citation grounding
├── generation_evaluation.py             # Stage 11e: Faithfulness + hallucination risk
├── rag_pipeline.py                      # Stage 12: End-to-end orchestrator
├── rag_diagnostics_manager.py           # Cross-stage static + retrieval diagnostics
├── retrieval_evaluation.py              # Retrieval-specific metrics
├── final_evaluation.py                  # Cross-pipeline strategy comparison
├── server.py                            # FastAPI web server
└── ui/
    └── index.html                       # Single-file SPA (vanilla JS)
scripts/
├── run_pipeline.py                      # End-to-end pipeline runner
└── start_server.py                      # Launch the web UI server
configs/
├── local_ollama.json
├── openai_cloud.json
└── medical_precise.json
tests/
├── conftest.py                          # MockStore, CORPUS, synthetic_results fixtures
├── test_config.py
├── test_context_assembler.py
├── test_evaluation_managers.py
├── test_generation_evaluation.py
├── test_generation_manager.py
├── test_helpers.py
├── test_index_strategies.py
├── test_query_manager.py
├── test_reranker_strategies.py
├── test_retriever_strategies.py
├── test_semantic_retrievers.py
└── test_server.py
```

### Storage layout

```
rag_store/{doc_id}/
├── manifest.json
├── tokens.dat                           # Encoded token stream
├── layout_{pages,elements,spans,words}.parquet
├── layout_manifest.json
├── clinical_document_metadata.json
├── clinical_{sections,element_metadata,entities}.parquet
├── chunks/{strategy}.parquet
├── chunks_enriched/{strategy}.parquet
├── indexes/{chunk_strategy}/
│   ├── {index_name}/{docs,vocab,postings}.parquet   # lexical
│   └── dense/
│       ├── faiss.index
│       ├── chunk_ids.npy
│       ├── docs.parquet
│       └── manifest.json
├── embeddings/{chunk_strategy}/
│   ├── vectors.npy
│   ├── chunk_ids.npy
│   └── manifest.json
└── reports/{stage}/                     # Metrics + plots per evaluation run
```

---

## Configuration

All backends are swapped with one field change. The system ships with four preset factories:

```python
from nitrag.config import RAGConfig

# Self-hosted: nomic-embed-text-v1.5 (fastembed) + llama3.1:8b (Ollama)
config = RAGConfig.local_ollama()

# Cloud: text-embedding-3-large + gpt-4o (set OPENAI_API_KEY)
config = RAGConfig.openai_cloud()

# Fast dev: bge-small-en + mistral:7b
config = RAGConfig.fast_local()

# High-accuracy: bge-large-en + wide retrieval + HyDE
config = RAGConfig.medical_precise()

# Load from JSON file
config = RAGConfig.from_file("configs/local_ollama.json")

# Mutate any field
config.llm.base_url = "http://my-server:11434/v1"
config.retrieval.top_k_retrieve = 30
config.embedding.model_name = "BAAI/bge-large-en-v1.5"
```

**Environment variable overrides** (all optional):

| Variable | Default |
|---|---|
| `NITRAG_EMBEDDING_PROVIDER` | `fastembed` |
| `NITRAG_EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` |
| `NITRAG_LLM_PROVIDER` | `openai_compatible` |
| `NITRAG_LLM_MODEL` | `llama3.1:8b` |
| `NITRAG_LLM_BASE_URL` | `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | — |
| `ANTHROPIC_API_KEY` | — |

---

## Web UI

```bash
uv run python scripts/start_server.py [--port 8000] [--reload]
```

The UI is a three-panel clinical tool:

- **Left sidebar** — document selector, config preset, pipeline status per stage
- **Center workspace** — query input, answer with inline `[N]` citation markers, evaluation strip (faithfulness, relevance, hallucination risk, overall score)
- **Right evidence panel** — citation cards (quote, page, section, confidence bar, content-type chips); "All Chunks" tab shows every retrieved passage with retriever name and score

**Citation interaction:** hover or click a `[N]` marker in the answer to highlight and scroll to the corresponding evidence card.

**Other features:**

- **Upload → process flow** — upload a PDF from the sidebar; a modal tracks all 7 pipeline stages in real time via 2 s polling
- **Query history** — previous queries listed in the sidebar; click any to restore the full response
- **Keyboard shortcuts** — `Ctrl+K` focuses the query input; `Enter` submits; `Shift+Enter` adds a newline
- **Export** — downloads the answer, citations, and evaluation as a Markdown file
- **New query button** — clears the workspace and resets to the welcome state

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the SPA |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/documents` | List all processed documents |
| `GET` | `/api/documents/{doc_id}` | Single document info + stage status |
| `GET` | `/api/config/presets` | Available configuration presets |
| `POST` | `/api/upload` | Upload a PDF (stores to `data/`) |
| `POST` | `/api/process` | Run full pipeline on an uploaded PDF |
| `GET` | `/api/process/status/{doc_id}` | Poll processing progress |
| `POST` | `/api/query` | Run a RAG query, return JSON with answer + citations + evaluation |

---

## Usage

### Programmatic

```python
from nitrag.rag_pipeline import RAGPipeline
from nitrag.chunk_manager import PdfTokenStore

store = PdfTokenStore(encoding_model_name="gpt-4o", root_dir="rag_store")
store.load("doc_e7fb48687c98a19c")

pipeline = RAGPipeline.local_ollama(store)
response = pipeline.answer("What medications were prescribed?", evaluate=True)

print(response.answer)
for c in response.citations:
    print(f"  [{c.number}] {c.source_label}  confidence={c.confidence:.0%}")
    print(f"  \"{c.quote}\"")

print(f"Faithfulness: {response.evaluation.faithfulness:.0%}")
print(f"Hallucination risk: {response.evaluation.hallucination_risk:.0%}")
```

### Building embeddings and indexes for a new document

```python
from nitrag.config import RAGConfig
from nitrag.embedding_manager import EmbeddingManager
from nitrag.vector_index_manager import VectorIndexManager

config = RAGConfig.local_ollama()
em = EmbeddingManager(store, config.embedding)
em.embed_all_strategies()                    # ~30s for a 3-page note

vim = VectorIndexManager(store, em, config.vector_index)
vim.build_all()                              # <1s for FAISS flat
```

### Running evaluations

```python
from nitrag.generation_evaluation import GenerationEvaluationManager

evaluator = GenerationEvaluationManager()
report = evaluator.evaluate(response.generation_result, response.context)
print(f"Overall: {report.overall_score:.2f}")
print(f"Faithfulness: {report.faithfulness:.2f}")
print(f"Hallucination risk: {report.hallucination_risk:.2f}")
# Note-level warnings
for note in report.notes:
    print(f"  ⚠ {note}")
```

---

## Tests

```bash
uv run pytest                    # all 486 tests
uv run pytest -x                 # stop on first failure
uv run pytest tests/test_generation_manager.py -v
uv run pytest --cov=nitrag --cov-report=term-missing
```

Tests use a synthetic medical corpus (8 clinical passages covering HTN, DM, medications, labs, negation, imaging) and a `MockStore` that maps `(start, end) → text` without any filesystem I/O. No LLM, no network, no FAISS needed for unit tests.

---

## Design principles

- **One strategy per class** — every chunker, indexer, retriever, reranker is a self-contained class registered with a manager. Adding a new strategy never changes existing ones.
- **Configuration-driven** — every backend is swappable by changing one field in `RAGConfig`. Switching from Ollama to OpenAI or from fastembed to sentence-transformers is one line.
- **No PyTorch for embeddings** — fastembed uses ONNX Runtime; runs on CPU with no GPU or PyTorch dependency.
- **Parquet-native storage** — all data lives in typed Parquet files for efficient I/O, schema enforcement, and compatibility with downstream tools.
- **Token-indexed architecture** — chunks track `(start_index, end_index)` token positions so any stage can reconstruct the exact document span.
- **Citation chain** — every answer sentence is grounded: PDF page → document element → token span → chunk → `[N]` in answer.
- **Evaluation at every stage** — no stage is complete without metrics and plots that show whether a strategy beats the baseline.

---

## Production infrastructure

When this system moves to production it will run on the following stack. All choices follow a **reliable + KISS** principle.

| Concern | Tool | Role |
|---------|------|------|
| Queue management | **NSQ** | Async document ingestion jobs, pipeline task dispatch |
| API layer | **FastAPI** | REST endpoints — already implemented |
| Real-time push | **Centrifugo** | WebSocket delivery of streaming generation responses |
| Caching | **Redis** | Embedding cache, query result cache, session state |
| Operational data | **MongoDB** | Document metadata, pipeline run records, user/org data |
| Vector storage | **Qdrant** | Self-hosted, HNSW index, supports named collections + filtering |

---

## Roadmap

### Near-term

- [ ] Streaming generation via SSE (server-sent events) endpoint
- [ ] Multi-document retrieval (query across all documents in rag_store)
- [ ] Document-type-aware configuration profiles (visit note vs. discharge summary vs. radiology)
- [ ] Structured output mode (JSON schema for medication extraction, diagnosis listing)

### Orchestrator / agent

- [ ] Hyperparameter space: chunker × indexer × retriever × reranker × embedding model
- [ ] Bayesian optimisation over RAGAS faithfulness on a held-out query set
- [ ] Per-document-type profiles learned from evaluation runs
- [ ] Configuration persistence and experiment tracking (MLflow or W&B)

### Data expansion

- [ ] Batch ingestion runner (folder of PDFs → rag_store)
- [ ] DICOM/HL7 adapters for structured clinical data
- [ ] De-identification pipeline before ingestion (PHI scrubbing)

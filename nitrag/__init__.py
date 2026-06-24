"""NIT-RAG — production medical RAG framework."""

from .chunk_manager import PdfTokenStore, ChunkManager, register_default_chunkers
from .chunking_evaluation import ChunkingEvaluationManager
from .chunk_metadata_enrichment_evaluation import ChunkMetadataEnrichmentEvaluationManager
from .chunk_metadata_enricher import ChunkMetadataEnricher
from .clinical_metadata_extractor import ClinicalMetadataExtractor
from .document_metadata_extractor import PyMuPDFLayoutExtractor
from .pdf_ingestion import PDFIngestionPipeline, IngestionConfig, PageType
from .final_evaluation import FinalEvaluationManager
from .index_manager import IndexManager, register_default_indexers
from .indexing_evaluation import IndexingEvaluationManager
from .reranker_manager import RerankerManager, register_default_rerankers
from .reranking_evaluation import RerankingEvaluationManager
from .retriever_manager import RetrieverManager, register_default_retrievers
from .rag_diagnostics_manager import RAGDiagnosticsManager

# Semantic + generation stack (new)
from .config import RAGConfig, EmbeddingConfig, LLMConfig, VectorIndexConfig, RetrievalConfig, GenerationConfig
from .embedding_manager import EmbeddingManager
from .vector_index_manager import VectorIndexManager
from .semantic_retrievers import register_semantic_retrievers
from .query_manager import QueryManager, QueryType, MEDICAL_ABBREVIATIONS
from .context_assembler import ContextAssembler, AssembledContext, ContextChunk
from .generation_manager import GenerationManager, GenerationResult, Citation
from .generation_evaluation import GenerationEvaluationManager, EvaluationReport
from .rag_pipeline import RAGPipeline, RAGResponse

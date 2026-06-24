"""
ingest_pdf.py — Standalone PDF ingestion script for NIT-RAG.

Runs the full PDFIngestionPipeline on one PDF and writes the augmented
parquets + manifest to the RAG store.

Usage
─────
    uv run scripts/ingest_pdf.py path/to/doc.pdf
    uv run scripts/ingest_pdf.py path/to/doc.pdf --store-dir /data/rag_store
    uv run scripts/ingest_pdf.py path/to/doc.pdf --doc-id my_visit_note_001
    uv run scripts/ingest_pdf.py path/to/doc.pdf --no-normalize --no-columns
    uv run scripts/ingest_pdf.py path/to/doc.pdf --overwrite false

Output
──────
    rag_store/{doc_id}/
        layout_pages.parquet        ← + page_type, column_count, quality metrics
        layout_elements.parquet     ← + normalized_text, reading_order_index,
                                         heading_score_final, is_heading_candidate_final
        layout_spans.parquet        ← + normalized_text
        layout_words.parquet
        layout_images.parquet
        layout_drawings.parquet
        layout_manifest.json        ← updated with ingestion_augmentation stats
        tokens.i32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nitrag.pdf_ingestion import IngestionConfig, PDFIngestionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the NIT-RAG PDF ingestion pipeline on a single PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdf", type=Path, help="Path to the PDF file to ingest.")
    p.add_argument(
        "--store-dir", type=Path, default=PROJECT_ROOT / "rag_store",
        help="Root directory for the RAG store.",
    )
    p.add_argument(
        "--doc-id", type=str, default=None,
        help="Override the auto-generated document ID (default: sha256-based).",
    )
    p.add_argument(
        "--overwrite", type=lambda v: v.lower() != "false", default=True,
        metavar="true|false",
        help="Overwrite an existing document directory.",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Skip text normalisation (ligature expansion, Unicode NFC, etc.).",
    )
    p.add_argument(
        "--join-hyphens", action="store_true",
        help="Merge words broken across lines by a trailing hyphen. Off by default.",
    )
    p.add_argument(
        "--no-columns", action="store_true",
        help="Skip 2-column layout detection.",
    )
    p.add_argument(
        "--no-ocr-headings", action="store_true",
        help="Skip OCR-aware heading detection (use font-based scores only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"WARNING: File does not have a .pdf extension: {pdf_path}")

    config = IngestionConfig(
        normalize_text      = not args.no_normalize,
        join_broken_hyphens = args.join_hyphens,
        detect_columns      = not args.no_columns,
        ocr_aware_headings  = not args.no_ocr_headings,
    )

    pipeline = PDFIngestionPipeline(config=config, root_dir=args.store_dir)

    print(f"PDF        : {pdf_path}")
    print(f"Store dir  : {args.store_dir}")
    print(f"Config     : normalize={config.normalize_text}  "
          f"join_hyphens={config.join_broken_hyphens}  "
          f"columns={config.detect_columns}  "
          f"ocr_headings={config.ocr_aware_headings}")
    print()

    manifest = pipeline.ingest(pdf_path, document_id=args.doc_id, overwrite=args.overwrite)

    aug = manifest.get("ingestion_augmentation", {})
    doc_dir = manifest["paths"]["document_dir"]

    print()
    print("=" * 60)
    print("Ingestion summary")
    print("=" * 60)
    print(f"  Document ID        : {manifest['document_id']}")
    print(f"  Output dir         : {doc_dir}")
    print(f"  Total pages        : {aug.get('total_pages', '?')}")
    print(f"  Native pages       : {aug.get('native_pages', '?')}")
    print(f"  Scanned/OCR pages  : {aug.get('scanned_ocr_pages', '?')}")
    print(f"  2-column pages     : {aug.get('two_column_pages', '?')}")
    print(f"  Total tokens       : {manifest.get('total_tokens', '?')}")
    print(f"  Total elements     : {manifest.get('total_elements', '?')}")
    print(f"  Total images       : {manifest.get('total_images', '?')}")
    print(f"  Pipeline version   : {aug.get('pipeline_version', '?')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

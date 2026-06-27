from __future__ import annotations

import json
import uuid
import hashlib
import datetime as dt
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken


# ============================================================
# Data models
# ============================================================

@dataclass
class ChunkSpan:
    start: int
    end: int
    kind: str = "chunk"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_length(self) -> int:
        return self.end - self.start

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)


RawSpan = Union[Tuple[int, int], ChunkSpan]


@dataclass
class DocumentPaths:
    document_dir: Path
    tokens_path: Path
    pages_path: Path
    elements_path: Path
    chunks_dir: Path
    chunks_enriched_dir: Path
    manifest_path: Path


# ============================================================
# Small utilities
# ============================================================

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    path = Path(path)

    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)

    return h.hexdigest()


def safe_json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj or {}, ensure_ascii=False, default=str)


def safe_json_loads(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s}


def read_parquet_mmap(path: Union[str, Path], filters=None) -> pa.Table:
    """
    Memory-map-backed Parquet read.

    Note:
    Parquet still may decompress/decode internally.
    This is perfect for metadata/chunks/metrics.
    For token slicing, we use np.memmap instead.
    """
    path = Path(path)

    with pa.memory_map(str(path), "r") as source:
        return pq.read_table(source, filters=filters)


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


# ============================================================
# Main production storage class
# ============================================================

class PdfTokenStore:
    """
    Production-style PDF token store.

    Heavy data:
      - tokens.i32       : raw int32 token ids, np.memmap friendly

    Metadata:
      - pages.parquet    : page boundaries
      - elements.parquet : layout/text elements with token spans
      - manifest.json    : document info

    Chunk outputs:
      - chunks/{strategy_name}.parquet
    """

    def __init__(
        self,
        encoding_model_name: str = "gpt-4o",
        root_dir: Union[str, Path] = "rag_store",
    ):
        self.encoding_model_name = encoding_model_name
        self.encoding = tiktoken.encoding_for_model(encoding_model_name)

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.document_id: Optional[str] = None
        self.paths: Optional[DocumentPaths] = None
        self.manifest: Optional[Dict[str, Any]] = None

        self._tokens_memmap: Optional[np.memmap] = None

    # ----------------------------
    # Ingestion
    # ----------------------------

    def ingest_pdf(
        self,
        input_pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        input_pdf_path = Path(input_pdf_path)

        if not input_pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {input_pdf_path}")

        file_hash = sha256_file(input_pdf_path)
        document_id = document_id or f"doc_{file_hash[:16]}"

        document_dir = self.root_dir / document_id

        if document_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Document already exists: {document_dir}. "
                f"Use overwrite=True or choose a different document_id."
            )

        document_dir.mkdir(parents=True, exist_ok=True)

        self.document_id = document_id
        self.paths = DocumentPaths(
            document_dir=document_dir,
            tokens_path=document_dir / "tokens.i32",
            pages_path=document_dir / "pages.parquet",
            elements_path=document_dir / "elements.parquet",
            chunks_dir=document_dir / "chunks",
            chunks_enriched_dir=document_dir / "chunks_enriched",
            manifest_path=document_dir / "manifest.json",
        )
        self.paths.chunks_dir.mkdir(parents=True, exist_ok=True)

        pages: List[Dict[str, Any]] = []
        elements: List[Dict[str, Any]] = []

        global_token_index = 0
        element_id = 0

        started_at = utc_now_iso()

        print(f"Starting PDF ingestion: {input_pdf_path}")
        print(f"Document ID: {document_id}")

        doc = fitz.open(str(input_pdf_path))

        with self.paths.tokens_path.open("wb") as token_file:
            for page_number, page in enumerate(doc):
                page_start = global_token_index
                page_line_count = 0
                page_block_count = 0

                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                try:
                    page_dict = page.get_text("dict", sort=True)
                except TypeError:
                    page_dict = page.get_text("dict")

                blocks = page_dict.get("blocks", [])

                for block_number, block in enumerate(blocks):
                    if block.get("type") != 0:
                        continue

                    block_start: Optional[int] = None
                    block_end: Optional[int] = None
                    block_text_chars = 0

                    block_bbox = block.get("bbox", [None, None, None, None])

                    for line_number, line in enumerate(block.get("lines", [])):
                        spans = line.get("spans", [])
                        line_text = "".join(span.get("text", "") for span in spans)

                        if not line_text.strip():
                            continue

                        # This is the canonical unit we actually write to the token stream.
                        # Because we encode exactly this text, line start/end spans are exact.
                        canonical_text = line_text.rstrip() + "\n"

                        token_ids = self.encoding.encode(canonical_text)
                        token_arr = np.asarray(token_ids, dtype=np.int32)

                        if token_arr.size == 0:
                            continue

                        start = global_token_index
                        token_arr.tofile(token_file)
                        global_token_index += int(token_arr.size)
                        end = global_token_index

                        line_bbox = line.get("bbox", [None, None, None, None])

                        elements.append({
                            "element_id": element_id,
                            "document_id": document_id,
                            "element_type": "line",
                            "page_number": page_number,
                            "block_number": block_number,
                            "line_number": line_number,
                            "start_index": start,
                            "end_index": end,
                            "token_length": end - start,
                            "text_chars": len(canonical_text),
                            "text_preview": canonical_text[:300],
                            "x0": float(line_bbox[0]) if line_bbox[0] is not None else None,
                            "y0": float(line_bbox[1]) if line_bbox[1] is not None else None,
                            "x1": float(line_bbox[2]) if line_bbox[2] is not None else None,
                            "y1": float(line_bbox[3]) if line_bbox[3] is not None else None,
                            "metadata_json": safe_json_dumps({
                                "source": "pymupdf",
                                "exact_token_span": True,
                            }),
                        })
                        element_id += 1

                        page_line_count += 1
                        block_text_chars += len(canonical_text)

                        if block_start is None:
                            block_start = start
                        block_end = end

                    if block_start is not None and block_end is not None:
                        elements.append({
                            "element_id": element_id,
                            "document_id": document_id,
                            "element_type": "block",
                            "page_number": page_number,
                            "block_number": block_number,
                            "line_number": None,
                            "start_index": block_start,
                            "end_index": block_end,
                            "token_length": block_end - block_start,
                            "text_chars": block_text_chars,
                            "text_preview": "",
                            "x0": float(block_bbox[0]) if block_bbox[0] is not None else None,
                            "y0": float(block_bbox[1]) if block_bbox[1] is not None else None,
                            "x1": float(block_bbox[2]) if block_bbox[2] is not None else None,
                            "y1": float(block_bbox[3]) if block_bbox[3] is not None else None,
                            "metadata_json": safe_json_dumps({
                                "source": "pymupdf",
                                "exact_token_span": True,
                                "derived_from": "line_elements",
                            }),
                        })
                        element_id += 1
                        page_block_count += 1

                # Add one newline between pages if the page had text.
                if global_token_index > page_start:
                    sep_tokens = np.asarray(self.encoding.encode("\n"), dtype=np.int32)
                    sep_tokens.tofile(token_file)
                    global_token_index += int(sep_tokens.size)

                page_end = global_token_index

                pages.append({
                    "document_id": document_id,
                    "page_number": page_number,
                    "start_index": page_start,
                    "end_index": page_end,
                    "token_length": page_end - page_start,
                    "line_count": page_line_count,
                    "block_count": page_block_count,
                    "page_width": page_width,
                    "page_height": page_height,
                    "has_text": page_end > page_start,
                })

                elements.append({
                    "element_id": element_id,
                    "document_id": document_id,
                    "element_type": "page",
                    "page_number": page_number,
                    "block_number": None,
                    "line_number": None,
                    "start_index": page_start,
                    "end_index": page_end,
                    "token_length": page_end - page_start,
                    "text_chars": None,
                    "text_preview": "",
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": page_width,
                    "y1": page_height,
                    "metadata_json": safe_json_dumps({
                        "source": "pymupdf",
                        "exact_token_span": True,
                    }),
                })
                element_id += 1

                if page_number % 50 == 0:
                    print(f"Ingested page {page_number}, total tokens: {global_token_index}")

        doc.close()

        write_parquet(pages, self.paths.pages_path)
        write_parquet(elements, self.paths.elements_path)

        finished_at = utc_now_iso()

        self.manifest = {
            "document_id": document_id,
            "source_pdf_path": str(input_pdf_path),
            "source_pdf_name": input_pdf_path.name,
            "source_sha256": file_hash,
            "encoding_model_name": self.encoding_model_name,
            "total_tokens": global_token_index,
            "total_pages": len(pages),
            "total_elements": len(elements),
            "tokens_path": str(self.paths.tokens_path),
            "pages_path": str(self.paths.pages_path),
            "elements_path": str(self.paths.elements_path),
            "chunks_dir": str(self.paths.chunks_dir),
            "created_at": started_at,
            "finished_at": finished_at,
            "storage_version": "v1_memmap_tokens_parquet_metadata",
        }

        self.paths.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._tokens_memmap = None

        print("Ingestion complete.")
        print(f"Total pages: {len(pages)}")
        print(f"Total elements: {len(elements)}")
        print(f"Total tokens: {global_token_index}")
        print(f"Document dir: {self.paths.document_dir}")

        return document_id

    # ----------------------------
    # Load existing document
    # ----------------------------

    def load(self, document_id: str) -> "PdfTokenStore":
        document_dir = self.root_dir / document_id
        manifest_path = document_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.document_id = manifest["document_id"]
        self.manifest = manifest
        self.paths = DocumentPaths(
            document_dir=document_dir,
            tokens_path=Path(manifest["tokens_path"]),
            pages_path=Path(manifest["pages_path"]),
            elements_path=Path(manifest["elements_path"]),
            chunks_dir=Path(manifest["chunks_dir"]),
            chunks_enriched_dir=document_dir / "chunks_enriched",
            manifest_path=manifest_path,
        )
        self._tokens_memmap = None

        return self

    # ----------------------------
    # Internal checks
    # ----------------------------

    def _require_loaded(self) -> None:
        if self.document_id is None or self.paths is None or self.manifest is None:
            raise RuntimeError("No document loaded. Call ingest_pdf() or load().")

    # ----------------------------
    # Token access
    # ----------------------------

    @property
    def total_tokens(self) -> int:
        self._require_loaded()
        return int(self.manifest["total_tokens"])

    @property
    def tokens(self) -> np.memmap:
        self._require_loaded()

        if self._tokens_memmap is None:
            self._tokens_memmap = np.memmap(
                self.paths.tokens_path,
                dtype=np.int32,
                mode="r",
            )

        return self._tokens_memmap

    def decode_span(self, start_index: int, end_index: int) -> str:
        self._require_loaded()

        start_index = max(0, int(start_index))
        end_index = min(int(end_index), self.total_tokens)

        if end_index <= start_index:
            return ""

        token_slice = self.tokens[start_index:end_index]
        return self.encoding.decode(token_slice.tolist())

    # ----------------------------
    # Metadata access
    # ----------------------------

    def pages_table(self) -> pa.Table:
        self._require_loaded()
        return read_parquet_mmap(self.paths.pages_path)

    def elements_table(
        self,
        element_type: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> pa.Table:
        self._require_loaded()

        filters = []

        if element_type is not None:
            filters.append(("element_type", "=", element_type))

        if page_number is not None:
            filters.append(("page_number", "=", int(page_number)))

        return read_parquet_mmap(
            self.paths.elements_path,
            filters=filters if filters else None,
        )

    def pages(self) -> List[Dict[str, Any]]:
        return self.pages_table().to_pylist()

    def elements(
        self,
        element_type: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.elements_table(
            element_type=element_type,
            page_number=page_number,
        ).to_pylist()

    def get_page_text(self, page_number: int) -> str:
        page_rows = [
            p for p in self.pages()
            if int(p["page_number"]) == int(page_number)
        ]

        if not page_rows:
            raise KeyError(f"Page not found: {page_number}")

        page = page_rows[0]
        return self.decode_span(page["start_index"], page["end_index"])

    def get_element(self, element_id: int) -> Dict[str, Any]:
        table = read_parquet_mmap(
            self.paths.elements_path,
            filters=[("element_id", "=", int(element_id))],
        )
        rows = table.to_pylist()

        if not rows:
            raise KeyError(f"Element not found: {element_id}")

        return rows[0]

    def get_element_text(self, element_id: int) -> str:
        element = self.get_element(element_id)
        return self.decode_span(element["start_index"], element["end_index"])

    def preview_elements(
        self,
        element_type: Optional[str] = None,
        page_number: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        rows = self.elements(element_type=element_type, page_number=page_number)

        output = []
        for row in rows[:limit]:
            output.append({
                "element_id": row["element_id"],
                "element_type": row["element_type"],
                "page_number": row["page_number"],
                "start_index": row["start_index"],
                "end_index": row["end_index"],
                "token_length": row["token_length"],
                "text_preview": row.get("text_preview", ""),
            })

        return output


# ============================================================
# Chunk manager
# ============================================================

class ChunkManager:
    """
    Register, execute, persist, and debug chunking strategies.
    """

    def __init__(self, store: PdfTokenStore):
        self.store = store
        self._algorithms: Dict[str, Callable[[PdfTokenStore], Iterable[RawSpan]]] = {}
        self._descriptions: Dict[str, str] = {}
        self._errors: Dict[str, str] = {}

    def register_chunker(
        self,
        name: str,
        logic_function: Callable[[PdfTokenStore], Iterable[RawSpan]],
        description: str = "",
        force: bool = False,
    ) -> None:
        if name in self._algorithms and not force:
            raise ValueError(f"Chunker already registered: {name}")

        self._algorithms[name] = logic_function
        self._descriptions[name] = description

    def list_algorithms(self, with_descriptions: bool = False):
        if with_descriptions:
            return {
                name: self._descriptions.get(name, "")
                for name in self._algorithms
            }

        return list(self._algorithms.keys())

    def execute(
        self,
        name: str,
        overwrite: bool = True,
    ) -> Path:
        self.store._require_loaded()

        if name not in self._algorithms:
            raise KeyError(f"Unknown chunker: {name}")

        out_path = self.store.paths.chunks_dir / f"{name}.parquet"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Chunks already exist: {out_path}")

        print(f"Executing chunker: {name}")

        raw_spans = list(self._algorithms[name](self.store))
        chunk_spans = [
            self._normalize_span(span, strategy_name=name)
            for span in raw_spans
        ]
        chunk_spans = [s for s in chunk_spans if s.end > s.start]

        rows = self._spans_to_rows(name, chunk_spans)
        write_parquet(rows, out_path)

        print(f"Chunker '{name}' generated {len(rows)} chunks.")
        print(f"Saved: {out_path}")

        return out_path

    def execute_all(self, continue_on_error: bool = True) -> Dict[str, Path]:
        outputs = {}
        self._errors.clear()

        for name in self._algorithms:
            try:
                outputs[name] = self.execute(name)
            except Exception as e:
                error = "".join(traceback.format_exception_only(type(e), e)).strip()
                self._errors[name] = error
                print(f"Chunker failed: {name}: {error}")

                if not continue_on_error:
                    raise

        return outputs

    def errors(self) -> Dict[str, str]:
        return dict(self._errors)

    def chunks_table(self, strategy_name: str) -> pa.Table:
        self.store._require_loaded()
        path = self.store.paths.chunks_dir / f"{strategy_name}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        return read_parquet_mmap(path)

    def chunks(self, strategy_name: str) -> List[Dict[str, Any]]:
        return self.chunks_table(strategy_name).to_pylist()

    def debug_chunk(self, strategy_name: str, chunk_id: int) -> Dict[str, Any]:
        rows = [
            r for r in self.chunks(strategy_name)
            if int(r["chunk_id"]) == int(chunk_id)
        ]

        if not rows:
            raise KeyError(f"Chunk not found: {strategy_name} / {chunk_id}")

        row = rows[0]
        text = self.store.decode_span(row["start_index"], row["end_index"])

        return {
            "chunk": row,
            "metadata": safe_json_loads(row.get("metadata_json")),
            "text": text,
        }

    def _normalize_span(self, span: RawSpan, strategy_name: str) -> ChunkSpan:
        total = self.store.total_tokens

        if isinstance(span, ChunkSpan):
            s = span
        else:
            s = ChunkSpan(start=int(span[0]), end=int(span[1]))

        s.start = max(0, min(int(s.start), total))
        s.end = max(0, min(int(s.end), total))

        s.metadata = dict(s.metadata or {})
        s.metadata.setdefault("strategy_name", strategy_name)
        s.metadata.setdefault("token_length", s.end - s.start)

        return s

    def _spans_to_rows(
        self,
        strategy_name: str,
        spans: List[ChunkSpan],
    ) -> List[Dict[str, Any]]:
        pages = self.store.pages()

        page_starts = np.asarray([int(p["start_index"]) for p in pages], dtype=np.int64)
        page_ends = np.asarray([int(p["end_index"]) for p in pages], dtype=np.int64)
        page_numbers = np.asarray([int(p["page_number"]) for p in pages], dtype=np.int64)

        rows = []

        for chunk_id, span in enumerate(spans):
            page_start, page_end = self._page_range_for_span(
                span.start,
                span.end,
                page_starts,
                page_ends,
                page_numbers,
            )

            rows.append({
                "chunk_id": chunk_id,
                "document_id": self.store.document_id,
                "strategy_name": strategy_name,
                "chunk_kind": span.kind,
                "start_index": span.start,
                "end_index": span.end,
                "token_length": span.end - span.start,
                "page_start": page_start,
                "page_end": page_end,
                "metadata_json": safe_json_dumps(span.metadata),
                "created_at": utc_now_iso(),
            })

        return rows

    @staticmethod
    def _page_range_for_span(
        start: int,
        end: int,
        page_starts: np.ndarray,
        page_ends: np.ndarray,
        page_numbers: np.ndarray,
    ) -> Tuple[Optional[int], Optional[int]]:
        if end <= start or len(page_starts) == 0:
            return None, None

        # Pages overlapping [start, end)
        mask = (page_ends > start) & (page_starts < end)

        if not np.any(mask):
            return None, None

        matched = page_numbers[mask]
        return int(matched.min()), int(matched.max())


# ============================================================
# Chunker factories
# ============================================================

def _safe_stride(chunk_size: int, overlap: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    return chunk_size - overlap


def fixed_token_chunker_factory(
    chunk_size: int = 512,
    overlap: int = 0,
):
    def chunker(store: PdfTokenStore) -> List[ChunkSpan]:
        stride = _safe_stride(chunk_size, overlap)
        chunks = []

        for start in range(0, store.total_tokens, stride):
            end = min(start + chunk_size, store.total_tokens)

            chunks.append(ChunkSpan(
                start=start,
                end=end,
                kind="fixed_token",
                metadata={
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
            ))

            if end == store.total_tokens:
                break

        return chunks

    return chunker


def page_chunker(store: PdfTokenStore) -> List[ChunkSpan]:
    chunks = []

    for p in store.pages():
        if int(p["end_index"]) <= int(p["start_index"]):
            continue

        chunks.append(ChunkSpan(
            start=int(p["start_index"]),
            end=int(p["end_index"]),
            kind="page",
            metadata={
                "page_number": int(p["page_number"]),
            },
        ))

    return chunks


def page_window_chunker_factory(
    window_pages: int = 2,
    overlap_pages: int = 1,
):
    if window_pages <= 0:
        raise ValueError("window_pages must be > 0")

    if overlap_pages >= window_pages:
        raise ValueError("overlap_pages must be smaller than window_pages")

    def chunker(store: PdfTokenStore) -> List[ChunkSpan]:
        pages = [p for p in store.pages() if int(p["end_index"]) > int(p["start_index"])]
        stride = window_pages - overlap_pages
        chunks = []

        for i in range(0, len(pages), stride):
            group = pages[i:i + window_pages]

            if not group:
                continue

            chunks.append(ChunkSpan(
                start=int(group[0]["start_index"]),
                end=int(group[-1]["end_index"]),
                kind="page_window",
                metadata={
                    "window_pages": window_pages,
                    "overlap_pages": overlap_pages,
                    "page_start": int(group[0]["page_number"]),
                    "page_end": int(group[-1]["page_number"]),
                },
            ))

            if i + window_pages >= len(pages):
                break

        return chunks

    return chunker


def element_chunker_factory(
    element_type: str = "block",
    max_tokens: Optional[int] = None,
):
    """
    One element per chunk.

    Good for:
      - block-based chunking
      - line-based debugging
      - page elements if element_type='page'
    """
    def chunker(store: PdfTokenStore) -> List[ChunkSpan]:
        elements = store.elements(element_type=element_type)
        chunks = []

        for e in elements:
            start = int(e["start_index"])
            end = int(e["end_index"])

            if end <= start:
                continue

            if max_tokens is not None and end - start > max_tokens:
                # Split oversized element into fixed windows.
                sub = fixed_window_inside_span(
                    start=start,
                    end=end,
                    chunk_size=max_tokens,
                    overlap=max(0, min(64, max_tokens // 10)),
                    kind=f"{element_type}_window",
                    metadata={
                        "source_element_id": int(e["element_id"]),
                        "source_element_type": element_type,
                    },
                )
                chunks.extend(sub)
            else:
                chunks.append(ChunkSpan(
                    start=start,
                    end=end,
                    kind=element_type,
                    metadata={
                        "source_element_id": int(e["element_id"]),
                        "element_type": element_type,
                        "page_number": int(e["page_number"]),
                    },
                ))

        return chunks

    return chunker


def fixed_window_inside_span(
    start: int,
    end: int,
    chunk_size: int,
    overlap: int,
    kind: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[ChunkSpan]:
    stride = _safe_stride(chunk_size, overlap)
    metadata = metadata or {}

    chunks = []
    cur = start

    while cur < end:
        nxt = min(cur + chunk_size, end)
        chunks.append(ChunkSpan(
            start=cur,
            end=nxt,
            kind=kind,
            metadata=dict(metadata),
        ))

        if nxt == end:
            break

        cur += stride

    return chunks


def group_elements_by_budget_chunker_factory(
    element_type: str = "block",
    max_tokens: int = 800,
    overlap_elements: int = 0,
):
    """
    Groups exact elements into token-budget chunks.

    This is a strong KISS production default:
      - no mid-block split unless required
      - good metadata
      - easy debugging
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    def chunker(store: PdfTokenStore) -> List[ChunkSpan]:
        elements = [
            e for e in store.elements(element_type=element_type)
            if int(e["end_index"]) > int(e["start_index"])
        ]

        chunks = []
        i = 0

        while i < len(elements):
            start = int(elements[i]["start_index"])
            end = int(elements[i]["end_index"])

            source_ids = [int(elements[i]["element_id"])]
            pages = {int(elements[i]["page_number"])}

            j = i + 1

            while j < len(elements):
                candidate_end = int(elements[j]["end_index"])

                if candidate_end - start > max_tokens:
                    break

                end = candidate_end
                source_ids.append(int(elements[j]["element_id"]))
                pages.add(int(elements[j]["page_number"]))
                j += 1

            chunks.append(ChunkSpan(
                start=start,
                end=end,
                kind=f"{element_type}_group",
                metadata={
                    "element_type": element_type,
                    "max_tokens": max_tokens,
                    "overlap_elements": overlap_elements,
                    "source_element_ids": source_ids,
                    "pages": sorted(pages),
                },
            ))

            if j >= len(elements):
                break

            if overlap_elements > 0:
                i = max(i + 1, j - overlap_elements)
            else:
                i = j

        return chunks

    return chunker


def hierarchical_child_chunker_factory(
    child_tokens: int = 256,
    child_overlap: int = 32,
    parent_tokens: int = 1200,
    parent_overlap: int = 120,
):
    """
    Production-grade parent-child strategy.

    Retrieval indexes child chunks.
    Generation can use parent_start/parent_end metadata for bigger context.
    """
    def chunker(store: PdfTokenStore) -> List[ChunkSpan]:
        parents = fixed_window_inside_span(
            start=0,
            end=store.total_tokens,
            chunk_size=parent_tokens,
            overlap=parent_overlap,
            kind="parent",
            metadata={
                "parent_tokens": parent_tokens,
                "parent_overlap": parent_overlap,
            },
        )

        children = []

        for parent_id, parent in enumerate(parents):
            child_spans = fixed_window_inside_span(
                start=parent.start,
                end=parent.end,
                chunk_size=child_tokens,
                overlap=child_overlap,
                kind="hierarchical_child",
                metadata={
                    "parent_id": parent_id,
                    "parent_start": parent.start,
                    "parent_end": parent.end,
                    "child_tokens": child_tokens,
                    "child_overlap": child_overlap,
                },
            )

            children.extend(child_spans)

        return children

    return chunker


def sentence_based_chunker(store: PdfTokenStore) -> List[ChunkSpan]:
    """One sentence per chunk with a global sentence_index.

    Uses block-level elements (the most granular unit available from the token
    store).  Blocks are always single-page by construction, so page_start ==
    page_end is guaranteed for every chunk produced here — which means the PDF
    highlight endpoint can search the exact page with the exact quote text.

    Large blocks (> SPLIT_THRESHOLD tokens) are split into individual sentences
    via regex; each sentence is re-tokenised to carve out a precise sub-range
    inside the block's [start_index, end_index].
    """
    import re
    import tiktoken

    SPLIT_THRESHOLD = 40   # blocks larger than this get sentence-split

    # Prefer 'line' elements when available (finer granularity); fall back to 'block'
    available_types = {e["element_type"] for e in store.elements()}
    element_type = "line" if "line" in available_types else "block"
    elements = store.elements(element_type=element_type)

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    def tokenise(text: str) -> int:
        if enc is None:
            return len(text.split())
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            return len(text.split())

    def split_sentences(text: str) -> List[str]:
        """Split text into sentences, preserving medical abbreviations."""
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        result = []
        for p in parts:
            p = p.strip()
            if p:
                result.append(p)
        return result or [text.strip()]

    chunks: List[ChunkSpan] = []
    sentence_idx = 0

    for e in elements:
        start = int(e["start_index"])
        end   = int(e["end_index"])
        tok_len = end - start
        if tok_len <= 0:
            continue

        page_num = int(e["page_number"])
        elem_id  = int(e["element_id"])

        # Small blocks → keep as one chunk
        if tok_len <= SPLIT_THRESHOLD or enc is None:
            chunks.append(ChunkSpan(
                start=start,
                end=end,
                kind="sentence",
                metadata={
                    "sentence_index":    sentence_idx,
                    "page_number":       page_num,
                    "source_element_id": elem_id,
                },
            ))
            sentence_idx += 1
            continue

        # Large blocks → decode and split into sentences
        full_text = store.decode_span(start, end)
        sents = split_sentences(full_text)

        if len(sents) <= 1:
            # Can't split — keep whole block
            chunks.append(ChunkSpan(
                start=start,
                end=end,
                kind="sentence",
                metadata={
                    "sentence_index":    sentence_idx,
                    "page_number":       page_num,
                    "source_element_id": elem_id,
                },
            ))
            sentence_idx += 1
            continue

        # Allocate sub-ranges proportionally by re-tokenised sentence length
        sent_lens = [tokenise(s) for s in sents]
        total_sent_toks = sum(sent_lens) or 1
        cur = start
        for i, (s, s_len) in enumerate(zip(sents, sent_lens)):
            if i == len(sents) - 1:
                sub_end = end
            else:
                sub_end = cur + max(1, round(tok_len * s_len / total_sent_toks))
                sub_end = min(sub_end, end - (len(sents) - i - 1))
            chunks.append(ChunkSpan(
                start=cur,
                end=sub_end,
                kind="sentence",
                metadata={
                    "sentence_index":    sentence_idx,
                    "page_number":       page_num,
                    "source_element_id": elem_id,
                    "sentence_text":     s[:200],  # for debugging
                },
            ))
            sentence_idx += 1
            cur = sub_end

    return chunks


def register_default_chunkers(manager: ChunkManager) -> None:
    manager.register_chunker(
        "fixed_512",
        fixed_token_chunker_factory(chunk_size=512, overlap=0),
        description="Plain fixed 512-token chunks.",
    )

    manager.register_chunker(
        "fixed_512_overlap_100",
        fixed_token_chunker_factory(chunk_size=512, overlap=100),
        description="Fixed 512-token chunks with 100-token overlap.",
    )

    manager.register_chunker(
        "fixed_1024_overlap_128",
        fixed_token_chunker_factory(chunk_size=1024, overlap=128),
        description="Larger fixed chunks with overlap.",
    )

    manager.register_chunker(
        "page_based",
        page_chunker,
        description="One PDF page per chunk.",
    )

    manager.register_chunker(
        "page_window_2_overlap_1",
        page_window_chunker_factory(window_pages=2, overlap_pages=1),
        description="Two-page window with one-page overlap.",
    )

    manager.register_chunker(
        "block_based",
        element_chunker_factory(element_type="block"),
        description="One PyMuPDF text block per chunk.",
    )

    manager.register_chunker(
        "line_based_debug",
        element_chunker_factory(element_type="line"),
        description="One PDF text line per chunk. Useful for debugging.",
    )

    manager.register_chunker(
        "sentence_based",
        sentence_based_chunker,
        description="One PDF line per chunk with global sentence_index for precise citation.",
    )

    manager.register_chunker(
        "block_group_800_overlap_1",
        group_elements_by_budget_chunker_factory(
            element_type="block",
            max_tokens=800,
            overlap_elements=1,
        ),
        description="Groups PDF blocks up to 800 tokens with one-block overlap.",
    )

    manager.register_chunker(
        "line_group_512_overlap_2",
        group_elements_by_budget_chunker_factory(
            element_type="line",
            max_tokens=512,
            overlap_elements=2,
        ),
        description="Groups PDF lines up to 512 tokens with two-line overlap.",
    )

    manager.register_chunker(
        "hierarchical_child_256_parent_1200",
        hierarchical_child_chunker_factory(
            child_tokens=256,
            child_overlap=32,
            parent_tokens=1200,
            parent_overlap=120,
        ),
        description="Parent-child strategy: retrieve children, expand to parent context.",
    )

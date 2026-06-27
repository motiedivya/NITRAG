"""Citation-aware context assembly for medical RAG.

Responsibilities
----------------
- Deduplicate retrieval results by chunk_id
- Apply token budget (don't exceed LLM context window)
- Assign sequential citation numbers [1], [2], … per chunk
- Decode full chunk text from the token store
- Format as structured evidence blocks for the LLM prompt
- Support score-based, page-based, or mixed ordering

Output
------
  AssembledContext.formatted_text  — the full evidence string to inject into the LLM prompt
  AssembledContext.citation_map    — chunk_id → citation_number
  AssembledContext.chunks          — list of ContextChunk with all metadata
"""
from __future__ import annotations

import json
import re
import tiktoken
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import GenerationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextChunk:
    citation_number: int
    chunk_id: int
    text: str
    page_start: int
    page_end: int
    section: str
    score: float
    retriever: str
    token_count: int
    document_id: str
    source_label: str           # "Page 2 | Assessment and Plan"
    contains_medication: bool = False
    contains_lab: bool = False
    contains_diagnosis: bool = False
    contains_vital: bool = False
    clinical_quality_score: float = 0.0
    chunk_strategy_name: str = ""
    sentence_index: int = -1


@dataclass
class AssembledContext:
    chunks: List[ContextChunk]
    citation_map: Dict[int, int]        # chunk_id → citation_number
    total_tokens: int
    formatted_text: str                 # evidence block for the LLM prompt
    query: str
    truncated: bool = False             # True if token budget was hit
    truncated_count: int = 0            # how many chunks were dropped


# ─────────────────────────────────────────────────────────────────────────────
# ContextAssembler
# ─────────────────────────────────────────────────────────────────────────────

class ContextAssembler:
    """Assemble retrieval results into a citation-aware context block.

    Usage
    -----
    assembler = ContextAssembler(store, config.generation)
    context = assembler.assemble(reranked_results, query="What medications were prescribed?")
    print(context.formatted_text)
    # ─── Evidence [1] | Page 2 | Medications ────────────────
    # Metformin 1000mg twice daily. Lisinopril 10mg once daily.
    # ────────────────────────────────────────────────────────
    """

    def __init__(self, store, config: GenerationConfig) -> None:
        self.store = store
        self.config = config
        self._enc: Optional[tiktoken.Encoding] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        results: List[Dict[str, Any]],
        query: str,
        max_tokens: Optional[int] = None,
    ) -> AssembledContext:
        """Assemble retrieval results into a citation-annotated context.

        Parameters
        ----------
        results     : retrieval / reranking result dicts (from RetrieverManager / RerankerManager)
        query       : the original user query (used for ordering heuristics)
        max_tokens  : token budget override; falls back to config.max_context_tokens
        """
        budget = max_tokens if max_tokens is not None else self.config.max_context_tokens

        # 1. Deduplicate by chunk_id, keep highest score
        seen: Dict[int, Dict[str, Any]] = {}
        for r in results:
            cid = int(r.get("chunk_id") or 0)
            score = float(r.get("rerank_score") or r.get("score") or 0.0)
            if cid not in seen or score > float(seen[cid].get("rerank_score") or seen[cid].get("score") or 0.0):
                seen[cid] = r

        unique = list(seen.values())

        # 2. Order
        unique = self._order(unique)

        # 3. Apply token budget
        chosen: List[ContextChunk] = []
        used_tokens = 0
        truncated = False
        dropped = 0

        for i, r in enumerate(unique):
            cid = int(r.get("chunk_id") or 0)
            text = self._decode_text(r)
            token_count = self._count_tokens(text)

            if used_tokens + token_count > budget:
                truncated = True
                dropped = len(unique) - i
                break

            page_start = int(r.get("page_start") or 0)
            page_end = int(r.get("page_end") or page_start)
            section = str(r.get("primary_section") or "")
            meta_json = str(r.get("metadata_json") or "")
            source_label = self._source_label(page_start, page_end, section, meta_json)

            sentence_index = -1
            if meta_json:
                try:
                    sentence_index = int(json.loads(meta_json).get("sentence_index", -1))
                except Exception:
                    pass

            citation_number = len(chosen) + 1
            chosen.append(ContextChunk(
                citation_number=citation_number,
                chunk_id=cid,
                text=text,
                page_start=page_start,
                page_end=page_end,
                section=section,
                score=float(r.get("rerank_score") or r.get("score") or 0.0),
                retriever=str(r.get("retriever_name") or ""),
                token_count=token_count,
                document_id=str(r.get("document_id") or ""),
                source_label=source_label,
                contains_medication=bool(r.get("contains_medication")),
                contains_lab=bool(r.get("contains_lab")),
                contains_diagnosis=bool(r.get("contains_diagnosis")),
                contains_vital=bool(r.get("contains_vital")),
                clinical_quality_score=float(r.get("clinical_quality_score") or 0.0),
                chunk_strategy_name=str(r.get("chunk_strategy_name") or r.get("strategy_name") or ""),
                sentence_index=sentence_index,
            ))
            used_tokens += token_count

        # 4. Build outputs
        citation_map = {c.chunk_id: c.citation_number for c in chosen}
        formatted = self._format_for_llm(chosen)

        return AssembledContext(
            chunks=chosen,
            citation_map=citation_map,
            total_tokens=used_tokens,
            formatted_text=formatted,
            query=query,
            truncated=truncated,
            truncated_count=dropped,
        )

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def _order(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordering = self.config.context_ordering

        if ordering == "page":
            return sorted(results, key=lambda r: (int(r.get("page_start") or 0), int(r.get("chunk_id") or 0)))

        if ordering == "mixed":
            # Top half by score, bottom half by page order
            by_score = sorted(results, key=lambda r: float(r.get("rerank_score") or r.get("score") or 0), reverse=True)
            top_half = by_score[: max(1, len(by_score) // 2)]
            bottom_half = sorted(by_score[len(top_half):], key=lambda r: int(r.get("page_start") or 0))
            return top_half + bottom_half

        # default: by descending score
        return sorted(results, key=lambda r: float(r.get("rerank_score") or r.get("score") or 0), reverse=True)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_for_llm(self, chunks: List[ContextChunk]) -> str:
        """Format chunks as clearly-delimited evidence blocks with citation markers."""
        if not chunks:
            return "No relevant evidence found in the document."

        lines = []
        for c in chunks:
            header = f"[{c.citation_number}] {c.source_label}"
            if self.config.include_metadata_in_context:
                flags = []
                if c.contains_medication:
                    flags.append("medication")
                if c.contains_lab:
                    flags.append("lab")
                if c.contains_diagnosis:
                    flags.append("diagnosis")
                if c.contains_vital:
                    flags.append("vital")
                if flags:
                    header += f" ({', '.join(flags)})"
            separator = "─" * min(60, len(header) + 4)
            lines.append(separator)
            lines.append(header)
            lines.append(separator)
            lines.append(c.text.strip())
            lines.append("")

        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_text(self, result: Dict[str, Any]) -> str:
        """Decode full text from token store, falling back to text_preview."""
        try:
            start = int(result["start_index"])
            end = int(result["end_index"])
            text = self.store.decode_span(start, end)
            if text:
                return text
        except Exception:
            pass
        return str(result.get("text_preview") or "")

    def _count_tokens(self, text: str) -> int:
        """Count tiktoken tokens (cl100k_base approximation)."""
        if self._enc is None:
            try:
                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return len(text.split())
        try:
            return len(self._enc.encode(text, disallowed_special=()))
        except Exception:
            return len(text.split())

    @staticmethod
    def _source_label(
        page_start: int,
        page_end: int,
        section: str,
        metadata_json: str = "",
    ) -> str:
        if page_start == page_end:
            page_str = f"Page {page_start + 1}"
        else:
            page_str = f"Pages {page_start + 1}–{page_end + 1}"
        if metadata_json:
            try:
                meta = json.loads(metadata_json)
                sidx = meta.get("sentence_index")
                if sidx is not None:
                    page_str = f"Sentence {int(sidx) + 1} | {page_str}"
            except Exception:
                pass
        if section:
            return f"{page_str} | {section}"
        return page_str

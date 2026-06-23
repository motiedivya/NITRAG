from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def safe_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(records if records else [])
    pq.write_table(table, path, compression="zstd")


def read_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def token_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compact_unique(values: List[Any], limit: int = 30) -> List[Any]:
    out = []
    seen = set()

    for v in values:
        if v in [None, "", [], {}]:
            continue

        key = str(v)
        if key in seen:
            continue

        seen.add(key)
        out.append(v)

        if len(out) >= limit:
            break

    return out


class ChunkMetadataEnricher:
    """
    Enriches chunk parquet files using:
      - layout elements
      - clinical entities
      - clinical element metadata
      - clinical document metadata

    Input:
      rag_store/<doc_id>/chunks/*.parquet

    Output:
      rag_store/<doc_id>/chunks_enriched/*.parquet
    """

    def __init__(
        self,
        document_dir: Union[str, Path],
        max_entities_per_chunk: int = 50,
        max_element_ids_per_chunk: int = 200,
    ):
        self.document_dir = Path(document_dir)
        self.chunks_dir = self.document_dir / "chunks"
        self.out_dir = self.document_dir / "chunks_enriched"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.max_entities_per_chunk = max_entities_per_chunk
        self.max_element_ids_per_chunk = max_element_ids_per_chunk

        self.manifest = self._load_json_if_exists(self.document_dir / "layout_manifest.json")
        self.doc_metadata = self._load_json_if_exists(self.document_dir / "clinical_document_metadata.json")

        self.elements = self._load_elements()
        self.line_elements = [
            e for e in self.elements
            if e.get("element_type") == "line"
            and safe_int(e.get("end_index"), 0) > safe_int(e.get("start_index"), 0)
        ]

        self.clinical_entities = read_parquet(self.document_dir / "clinical_entities.parquet")
        self.clinical_element_meta = read_parquet(self.document_dir / "clinical_element_metadata.parquet")
        self.clinical_sections = read_parquet(self.document_dir / "clinical_sections.parquet")

        self.element_meta_by_id = {
            safe_int(e.get("element_id")): e
            for e in self.clinical_element_meta
            if safe_int(e.get("element_id")) is not None
        }

        self.entities_by_element_id = defaultdict(list)
        for ent in self.clinical_entities:
            eid = safe_int(ent.get("element_id"))
            if eid is not None:
                self.entities_by_element_id[eid].append(ent)

        self.entities_by_page = defaultdict(list)
        for ent in self.clinical_entities:
            page = safe_int(ent.get("page_number"))
            if page is not None:
                self.entities_by_page[page].append(ent)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def enrich_all(self, overwrite: bool = True) -> Dict[str, Path]:
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks dir not found: {self.chunks_dir}")

        outputs = {}

        for chunk_path in sorted(self.chunks_dir.glob("*.parquet")):
            strategy_name = chunk_path.stem
            outputs[strategy_name] = self.enrich_strategy(strategy_name, overwrite=overwrite)

        print("Chunk metadata enrichment complete.")
        print(f"Strategies enriched: {len(outputs)}")
        print(f"Output dir: {self.out_dir}")

        return outputs

    def enrich_strategy(self, strategy_name: str, overwrite: bool = True) -> Path:
        chunk_path = self.chunks_dir / f"{strategy_name}.parquet"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        out_path = self.out_dir / f"{strategy_name}.parquet"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Enriched chunks already exist: {out_path}")

        chunks = read_parquet(chunk_path)
        enriched = []

        for chunk in chunks:
            enriched.append(self.enrich_chunk(chunk))

        write_parquet(enriched, out_path)

        print(f"Enriched {strategy_name}: {len(enriched)} chunks → {out_path}")

        return out_path

    def preview(self, strategy_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        path = self.out_dir / f"{strategy_name}.parquet"
        rows = read_parquet(path)

        preview_rows = []

        for r in rows[:limit]:
            preview_rows.append({
                "chunk_id": r.get("chunk_id"),
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
                "primary_section": r.get("primary_section"),
                "section_names_json": r.get("section_names_json"),
                "entity_type_counts_json": r.get("entity_type_counts_json"),
                "contains_medication": r.get("contains_medication"),
                "contains_lab": r.get("contains_lab"),
                "contains_diagnosis": r.get("contains_diagnosis"),
                "clinical_quality_score": r.get("clinical_quality_score"),
            })

        return preview_rows

    # ------------------------------------------------------------
    # Chunk enrichment
    # ------------------------------------------------------------

    def enrich_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        c_start = safe_int(chunk.get("start_index"), 0)
        c_end = safe_int(chunk.get("end_index"), 0)
        c_page_start = safe_int(chunk.get("page_start"))
        c_page_end = safe_int(chunk.get("page_end"))

        overlapping_lines = self._find_overlapping_lines(
            c_start=c_start,
            c_end=c_end,
            page_start=c_page_start,
            page_end=c_page_end,
        )

        line_element_ids = [
            safe_int(e.get("element_id"))
            for e in overlapping_lines
            if safe_int(e.get("element_id")) is not None
        ]

        line_element_ids = line_element_ids[:self.max_element_ids_per_chunk]

        element_meta_rows = [
            self.element_meta_by_id[eid]
            for eid in line_element_ids
            if eid in self.element_meta_by_id
        ]

        entities = self._entities_for_chunk(
            line_element_ids=line_element_ids,
            page_start=c_page_start,
            page_end=c_page_end,
        )

        sections = self._section_summary(overlapping_lines, element_meta_rows)
        entity_summary = self._entity_summary(entities)
        flags = self._clinical_flags(entities, element_meta_rows)
        quality = self._quality_score(
            chunk=chunk,
            overlapping_lines=overlapping_lines,
            sections=sections,
            entities=entities,
            flags=flags,
        )

        existing_metadata = safe_json_loads(chunk.get("metadata_json")) or {}

        clinical_metadata = {
            "primary_section": sections["primary_section"],
            "section_names": sections["section_names"],
            "heading_paths": sections["heading_paths"],
            "source_element_ids": line_element_ids,
            "entity_type_counts": entity_summary["entity_type_counts"],
            "entities": entity_summary["entities"],
            "date_values": entity_summary["date_values"],
            "clinical_flags": flags,
            "clinical_quality_score": quality,
            "document_type": self.doc_metadata.get("document_type"),
            "document_type_confidence": self.doc_metadata.get("document_type_confidence"),
            "metadata_enricher_version": "chunk_metadata_enricher_v1",
        }

        merged_metadata = dict(existing_metadata)
        merged_metadata["clinical"] = clinical_metadata

        enriched = dict(chunk)

        # Flat columns for fast filtering.
        enriched.update({
            "document_type": self.doc_metadata.get("document_type"),
            "document_type_confidence": self.doc_metadata.get("document_type_confidence"),

            "primary_section": sections["primary_section"],
            "section_names_json": safe_json_dumps(sections["section_names"]),
            "heading_paths_json": safe_json_dumps(sections["heading_paths"]),

            "source_element_ids_json": safe_json_dumps(line_element_ids),
            "overlap_line_count": len(overlapping_lines),

            "entity_count": len(entities),
            "entity_type_counts_json": safe_json_dumps(entity_summary["entity_type_counts"]),
            "entities_json": safe_json_dumps(entity_summary["entities"]),
            "date_values_json": safe_json_dumps(entity_summary["date_values"]),

            "contains_date": flags["contains_date"],
            "contains_patient_id": flags["contains_patient_id"],
            "contains_vital": flags["contains_vital"],
            "contains_lab": flags["contains_lab"],
            "contains_medication": flags["contains_medication"],
            "contains_diagnosis": flags["contains_diagnosis"],
            "contains_imaging": flags["contains_imaging"],
            "contains_procedure": flags["contains_procedure"],
            "contains_negation": flags["contains_negation"],

            "clinical_quality_score": quality,
            "metadata_json": safe_json_dumps(merged_metadata),
        })

        # Preserve parent-child info as easy columns if present.
        enriched.update(self._extract_chunk_relationship_columns(existing_metadata))

        return enriched

    # ------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------

    def _find_overlapping_lines(
        self,
        c_start: int,
        c_end: int,
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> List[Dict[str, Any]]:
        candidates = []

        for e in self.line_elements:
            e_page = safe_int(e.get("page_number"))

            if page_start is not None and page_end is not None and e_page is not None:
                if e_page < page_start or e_page > page_end:
                    continue

            e_start = safe_int(e.get("start_index"), 0)
            e_end = safe_int(e.get("end_index"), 0)

            ov = token_overlap(c_start, c_end, e_start, e_end)
            if ov > 0:
                row = dict(e)
                row["_token_overlap"] = ov
                candidates.append(row)

        candidates.sort(key=lambda x: (
            safe_int(x.get("page_number"), 0),
            safe_float(x.get("y0"), 0.0),
            safe_float(x.get("x0"), 0.0),
            safe_int(x.get("element_id"), 0),
        ))

        return candidates

    def _entities_for_chunk(
        self,
        line_element_ids: List[int],
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> List[Dict[str, Any]]:
        out = []

        seen = set()

        # Strongest: entity belongs to overlapping element.
        for eid in line_element_ids:
            for ent in self.entities_by_element_id.get(eid, []):
                key = self._entity_key(ent)
                if key not in seen:
                    seen.add(key)
                    out.append(ent)

        # Fallback: if entity lacks element mapping or chunk has page-based span.
        if page_start is not None and page_end is not None:
            for page in range(page_start, page_end + 1):
                for ent in self.entities_by_page.get(page, []):
                    eid = safe_int(ent.get("element_id"))
                    if eid is not None and eid not in line_element_ids:
                        continue

                    key = self._entity_key(ent)
                    if key not in seen:
                        seen.add(key)
                        out.append(ent)

        out.sort(key=lambda e: safe_float(e.get("confidence"), 0.0), reverse=True)

        return out[:self.max_entities_per_chunk]

    def _entity_key(self, ent: Dict[str, Any]) -> tuple:
        return (
            ent.get("entity_type"),
            str(ent.get("normalized_value") or ent.get("text") or "").lower(),
            safe_int(ent.get("page_number")),
            safe_int(ent.get("element_id")),
        )

    # ------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------

    def _section_summary(
        self,
        overlapping_lines: List[Dict[str, Any]],
        element_meta_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        section_values = []

        for e in overlapping_lines:
            section = e.get("section_name")
            if section:
                section_values.append(str(section))

            heading_path_json = e.get("heading_path_json")
            hp = safe_json_loads(heading_path_json)
            if isinstance(hp, list):
                for h in hp:
                    if h:
                        section_values.append(str(h))

        for m in element_meta_rows:
            section = m.get("section_name") or m.get("section_detected")
            if section:
                section_values.append(str(section))

        section_values = [s.strip() for s in section_values if str(s).strip()]

        counts = Counter(section_values)
        primary = counts.most_common(1)[0][0] if counts else None

        heading_paths = []

        for e in overlapping_lines:
            hp = safe_json_loads(e.get("heading_path_json"))
            if isinstance(hp, list) and hp:
                heading_paths.append(hp)

        return {
            "primary_section": primary,
            "section_names": compact_unique(section_values, limit=20),
            "heading_paths": compact_unique(heading_paths, limit=10),
        }

    def _entity_summary(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        type_counts = Counter(e.get("entity_type") for e in entities if e.get("entity_type"))

        compact_entities = []
        date_values = []

        for e in entities:
            etype = e.get("entity_type")
            text = e.get("text")
            norm = e.get("normalized_value")

            if etype == "date":
                date_values.append(norm)

            compact_entities.append({
                "type": etype,
                "text": text,
                "normalized_value": norm,
                "page": e.get("page_number"),
                "element_id": e.get("element_id"),
                "section": e.get("section_name"),
                "confidence": e.get("confidence"),
                "negated": e.get("is_negated"),
            })

        return {
            "entity_type_counts": dict(type_counts),
            "entities": compact_entities[:self.max_entities_per_chunk],
            "date_values": compact_unique(date_values, limit=20),
        }

    def _clinical_flags(
        self,
        entities: List[Dict[str, Any]],
        element_meta_rows: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        entity_types = {e.get("entity_type") for e in entities}

        def any_meta_flag(name: str) -> bool:
            return any(bool(m.get(name)) for m in element_meta_rows)

        return {
            "contains_date": "date" in entity_types or any_meta_flag("contains_date"),
            "contains_patient_id": "patient_identifier" in entity_types or any_meta_flag("contains_patient_id"),
            "contains_vital": "vital" in entity_types or any_meta_flag("contains_vital"),
            "contains_lab": "lab_result" in entity_types or any_meta_flag("contains_lab_candidate"),
            "contains_medication": (
                "medication_candidate" in entity_types
                or "medication_line_candidate" in entity_types
                or any_meta_flag("contains_medication_cue")
            ),
            "contains_diagnosis": (
                "diagnosis_code_candidate" in entity_types
                or "diagnosis_or_problem_candidate" in entity_types
            ),
            "contains_imaging": "imaging_candidate" in entity_types,
            "contains_procedure": "procedure_candidate" in entity_types,
            "contains_negation": any(bool(e.get("is_negated")) for e in entities) or any_meta_flag("contains_negation"),
        }

    def _quality_score(
        self,
        chunk: Dict[str, Any],
        overlapping_lines: List[Dict[str, Any]],
        sections: Dict[str, Any],
        entities: List[Dict[str, Any]],
        flags: Dict[str, bool],
    ) -> float:
        score = 0.35

        if overlapping_lines:
            score += 0.15

        if sections.get("primary_section"):
            score += 0.15

        if entities:
            score += min(0.20, 0.04 * len(entities))

        if any(flags.values()):
            score += 0.10

        token_length = safe_int(chunk.get("token_length"), 0)
        if 128 <= token_length <= 1200:
            score += 0.05
        elif token_length > 2500:
            score -= 0.10

        return round(max(0.0, min(1.0, score)), 4)

    def _extract_chunk_relationship_columns(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "parent_id": metadata.get("parent_id"),
            "parent_start": metadata.get("parent_start"),
            "parent_end": metadata.get("parent_end"),
            "source_element_id": metadata.get("source_element_id"),
            "source_element_ids_json_from_chunker": safe_json_dumps(metadata.get("source_element_ids")),
        }

    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------

    def _load_json_if_exists(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_elements(self) -> List[Dict[str, Any]]:
        preferred = self.document_dir / "layout_elements_with_sections.parquet"
        fallback = self.document_dir / "layout_elements.parquet"
        old_fallback = self.document_dir / "elements.parquet"

        if preferred.exists():
            return read_parquet(preferred)

        if fallback.exists():
            return read_parquet(fallback)

        if old_fallback.exists():
            return read_parquet(old_fallback)

        return []

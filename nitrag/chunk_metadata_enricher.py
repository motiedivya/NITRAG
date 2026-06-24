"""Chunk Metadata Enricher v2.

Reads assembled chunks and entity/section metadata produced by earlier pipeline stages;
writes enriched columns that downstream retrieval, ranking, and evaluation stages consume.

Preserved downstream contracts (column names MUST remain unchanged):
    contains_medication, contains_lab, contains_diagnosis, contains_vital,
    contains_imaging, contains_procedure, contains_negation, contains_date,
    contains_patient_id, clinical_quality_score, primary_section,
    entity_type_counts_json, entities_json

New columns added in v2:
    clinical_importance_score, concept_density, is_boilerplate,
    contains_allergy, contains_demographics,
    is_radiology_content, is_operative_content,
    is_medication_focused, is_diagnostic_focused,
    chunk_metadata_enricher_version
"""
from __future__ import annotations

import json
import re
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq


_VERSION = "chunk_metadata_enricher_v2"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def safe_json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s

def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)

def lower_clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()

def _coerce_value(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pq.write_table(pa.table({}), path, compression="zstd")
        return
    clean = [{k: _coerce_value(v) for k, v in rec.items()} for rec in records]
    if len(clean) > 1:
        for key in list(clean[0].keys()):
            seen_types: set = set()
            for row in clean:
                val = row.get(key)
                if val is not None:
                    seen_types.add(type(val))
            if len(seen_types) > 1:
                for row in clean:
                    if row.get(key) is not None:
                        row[key] = str(row[key])
    pq.write_table(pa.Table.from_pylist(clean), path, compression="zstd")

def read_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    return pq.read_table(path).to_pylist()

def _token_count(text: str) -> int:
    return max(1, len(text.split()))


# ─── Keyword sets for flag evaluation ─────────────────────────────────────────

_RADIOLOGY_KEYWORDS = frozenset({
    "ct", "mri", "x-ray", "xray", "radiograph", "ultrasound", "pet",
    "spect", "fluoroscopy", "mammogram", "angiogram", "bone", "dexa",
    "scan", "imaging", "radiology", "nuclear", "contrast",
    "findings", "impression", "technique", "views", "projection",
    "lucency", "opacity", "hyperdense", "hypodense", "signal",
    "enhancement", "lesion", "nodule", "mass", "infiltrate", "effusion",
    "consolidation", "atelectasis", "pneumothorax", "fracture",
    "herniation", "stenosis", "occlusion",
})

_OPERATIVE_KEYWORDS = frozenset({
    "operative", "preoperative", "postoperative", "intraoperative",
    "surgery", "incision", "dissection", "excision", "resection",
    "anastomosis", "hemostasis", "closure", "drain", "specimen",
    "anesthesia", "surgeon", "estimated", "blood", "loss",
    "sterile", "prep", "drape", "cauterization",
    "laparotomy", "laparoscopy", "thoracotomy", "craniotomy",
})

_MEDICATION_KEYWORDS = frozenset({
    "mg", "mcg", "tablet", "capsule", "injection", "infusion",
    "dose", "daily", "bid", "tid", "qid", "prn", "po", "iv", "im",
    "subcutaneous", "topical", "sublingual", "transdermal", "patch",
    "prescription", "dispense", "refill", "sig", "pharmacy",
    "route", "frequency", "duration", "medication", "drug",
})

_DIAGNOSTIC_KEYWORDS = frozenset({
    "diagnosis", "diagnoses", "assessment", "impression",
    "differential", "rule", "consistent", "suggestive", "etiology",
    "icd", "problem", "chief", "complaint", "history",
})

_BOILERPLATE_SIGNALS = frozenset({
    "page", "continued", "confidential", "document",
    "electronically", "signed", "printed", "generated", "report",
    "version", "copyright", "rights", "reserved", "disclaimer",
    "internal", "authorized", "personnel",
    "tel", "fax", "phone", "address", "zip", "suite",
})

_CLINICAL_KEYWORDS_ANY = frozenset({
    "patient", "diagnosis", "medication", "treatment", "surgery", "procedure",
    "vital", "blood", "pressure", "heart", "rate", "temperature", "symptom",
    "complaint", "history", "exam", "lab", "imaging", "allergy", "dose",
    "hospital", "clinic", "physician", "nurse", "therapy",
})


# ─── Importance weights by entity type ────────────────────────────────────────

_IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "diagnosis_or_problem":      0.30,
    "past_diagnosis":            0.22,
    "diagnosis_code":            0.22,
    "vital":                     0.20,
    "medication":                0.18,
    "lab_result":                0.18,
    "allergy":                   0.15,
    "allergy_nkda":              0.08,
    "procedure":                 0.15,
    "procedure_code":            0.12,
    "imaging":                   0.10,
    "demographics_age":          0.07,
    "demographics_gender":       0.05,
    "demographics_dob":          0.09,
    "demographics_name":         0.05,
    "social_smoking":            0.08,
    "social_alcohol":            0.08,
    "social_substance":          0.10,
    "social_occupation":         0.04,
    "date":                      0.05,
    "patient_identifier":        0.08,
    "medication_class":          0.10,
    "medication_candidate":      0.08,
    "medication_line_candidate": 0.06,
    "lab_reference_range":       0.05,
}


# ─── Main enricher ────────────────────────────────────────────────────────────

class ChunkMetadataEnricher:
    """
    Enriches chunks produced by the chunking stage with clinical metadata
    derived from entities and element-level clinical metadata.

    Input files (all in document_dir):
      - chunks/*.parquet                    (one file per chunking strategy)
      - manifest.json                       (for PdfTokenStore bootstrap)
      - tokens.i32                          (raw token ids for text decode)
      - clinical_entities.parquet           (from ClinicalMetadataExtractor)
      - clinical_element_metadata.parquet   (from ClinicalMetadataExtractor)
      - clinical_sections.parquet           (from ClinicalMetadataExtractor)

    Output:
      - enriched_chunks.parquet
    """

    def __init__(self, document_dir: Union[str, Path]):
        self.document_dir = Path(document_dir)
        self.chunks_dir        = self.document_dir / "chunks"
        self.entities_path     = self.document_dir / "clinical_entities.parquet"
        self.element_meta_path = self.document_dir / "clinical_element_metadata.parquet"
        self.sections_path     = self.document_dir / "clinical_sections.parquet"
        self.output_path       = self.document_dir / "enriched_chunks.parquet"
        self._store: Any = None  # PdfTokenStore, lazy-loaded

    def _load_token_store(self) -> Any:
        """Lazy-load PdfTokenStore for text decoding."""
        if self._store is not None:
            return self._store
        try:
            from nitrag.chunk_manager import PdfTokenStore
            import json as _json
            manifest_path = self.document_dir / "manifest.json"
            if not manifest_path.exists():
                return None
            manifest = _json.loads(manifest_path.read_text())
            enc_model = manifest.get("encoding_model_name", "gpt-4o")
            doc_id = manifest.get("document_id", self.document_dir.name)
            store = PdfTokenStore(encoding_model_name=enc_model,
                                  root_dir=self.document_dir.parent)
            store.load(doc_id)
            self._store = store
            return store
        except Exception as exc:
            print(f"[enricher] warning: could not load token store: {exc}")
            return None

    def _decode_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """Decode chunk text from token store, with fallbacks."""
        # Fast path: text already in row (shouldn't happen with current schema but safe)
        if chunk.get("text"):
            return str(chunk["text"])
        # Decode from token store using start/end indices
        try:
            start = int(chunk["start_index"])
            end = int(chunk["end_index"])
            if end > start:
                store = self._load_token_store()
                if store is not None:
                    text = store.decode_span(start, end)
                    if text:
                        return text
        except Exception:
            pass
        # Last resort: text_preview from metadata_json
        try:
            meta = json.loads(chunk.get("metadata_json") or "{}")
            preview = meta.get("text_preview") or ""
            if preview:
                return preview
        except Exception:
            pass
        return ""

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunk rows from chunks/ directory."""
        if not self.chunks_dir.is_dir():
            return []
        seen: set = set()
        chunks: List[Dict[str, Any]] = []
        for pq_file in sorted(self.chunks_dir.glob("*.parquet")):
            try:
                rows = read_parquet(pq_file)
                for row in rows:
                    key = (row.get("strategy_name"), row.get("chunk_id"))
                    if key not in seen:
                        seen.add(key)
                        chunks.append(row)
            except Exception as exc:
                print(f"[enricher] warning: could not read {pq_file.name}: {exc}")
        return chunks

    def _build_lookup_tables(self) -> tuple:
        """Load and index clinical metadata tables. Returns (elem_by_id, entities_by_elem, entities_by_page, section_by_page)."""
        entities     = read_parquet(self.entities_path)     if self.entities_path.exists()     else []
        element_meta = read_parquet(self.element_meta_path) if self.element_meta_path.exists() else []
        sections     = read_parquet(self.sections_path)     if self.sections_path.exists()     else []

        elem_by_id: Dict[Any, Dict[str, Any]] = {
            e["element_id"]: e for e in element_meta if e.get("element_id") is not None
        }
        entities_by_elem: Dict[Any, List[Dict[str, Any]]] = {}
        for ent in entities:
            key = ent.get("element_id")
            if key is not None:
                entities_by_elem.setdefault(key, []).append(ent)

        entities_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for ent in entities:
            page = ent.get("page_number")
            if page is not None:
                entities_by_page.setdefault(int(page), []).append(ent)

        section_by_page = self._build_section_page_map(sections)
        return elem_by_id, entities_by_elem, entities_by_page, section_by_page

    def enrich_all(self, overwrite: bool = True) -> Dict[str, Any]:
        """
        Enrich all chunking strategies and write per-strategy files to
        chunks_enriched/{strategy}.parquet  (layout expected by IndexManager
        and EmbeddingManager).
        """
        if not self.chunks_dir.is_dir():
            print(f"[enricher] no chunks dir at {self.chunks_dir}")
            return {"enriched_count": 0, "strategies": [], "version": _VERSION}

        out_dir = self.document_dir / "chunks_enriched"
        out_dir.mkdir(parents=True, exist_ok=True)

        elem_by_id, entities_by_elem, entities_by_page, section_by_page = self._build_lookup_tables()

        total = 0
        strategies: List[str] = []
        for pq_file in sorted(self.chunks_dir.glob("*.parquet")):
            strategy = pq_file.stem
            out_path = out_dir / pq_file.name

            if not overwrite and out_path.exists():
                strategies.append(strategy)
                continue

            try:
                chunks = read_parquet(pq_file)
            except Exception as exc:
                print(f"[enricher] warning: could not read {pq_file.name}: {exc}")
                continue

            enriched: List[Dict[str, Any]] = []
            for chunk in chunks:
                try:
                    row = self._enrich_chunk(chunk, elem_by_id, entities_by_elem, entities_by_page, section_by_page)
                    enriched.append(row)
                except Exception as exc:
                    chunk["_enrichment_error"] = str(exc)
                    enriched.append(chunk)

            write_parquet(enriched, out_path)
            total += len(enriched)
            strategies.append(strategy)
            print(f"[enricher] {strategy}: {len(enriched)} chunks → {out_path}")

        print(f"Chunk enrichment complete: {total} chunks across {len(strategies)} strategies")
        return {"enriched_count": total, "strategies": strategies, "version": _VERSION}

    def run(self) -> Dict[str, Any]:
        """Convenience wrapper — enriches all strategies."""
        return self.enrich_all(overwrite=True)

    # ── Core enrichment logic ─────────────────────────────────────────────────

    def _enrich_chunk(
        self,
        chunk: Dict[str, Any],
        elem_by_id: Dict[Any, Dict[str, Any]],
        entities_by_elem: Dict[Any, List[Dict[str, Any]]],
        entities_by_page: Dict[int, List[Dict[str, Any]]],
        section_by_page: Dict[int, str],
    ) -> Dict[str, Any]:
        text = self._decode_chunk_text(chunk)
        text_l = lower_clean(text)
        token_count = _token_count(text)

        chunk_entities = self._gather_chunk_entities(
            chunk, elem_by_id, entities_by_elem, entities_by_page,
        )

        entity_type_counts: Dict[str, int] = {}
        for ent in chunk_entities:
            et = ent.get("entity_type") or "unknown"
            entity_type_counts[et] = entity_type_counts.get(et, 0) + 1

        # ── Preserved presence flags ──────────────────────────────────────────
        contains_medication = self._has_types(
            entity_type_counts,
            {"medication", "medication_class", "medication_candidate", "medication_line_candidate"},
        ) or bool(re.search(
            r"\b(mg|mcg|tablet|capsule|injection|dose|daily|bid|tid|qid|prn)\b", text_l,
        ))

        contains_lab = self._has_types(
            entity_type_counts, {"lab_result", "lab_reference_range"},
        ) or bool(re.search(
            r"\b(hemoglobin|wbc|platelet|sodium|potassium|creatinine|bun|glucose|hba1c|cholesterol|troponin)\b",
            text_l,
        ))

        contains_diagnosis = self._has_types(
            entity_type_counts, {"diagnosis_or_problem", "past_diagnosis", "diagnosis_code"},
        ) or bool(re.search(
            r"\b(diagnosis|diagnoses|impression|assessment|history of|hx of|h/o|icd)\b", text_l,
        ))

        contains_vital = self._has_types(entity_type_counts, {"vital"}) or bool(re.search(
            r"\b(bp|blood pressure|heart rate|hr|temperature|temp|rr|respiratory rate|spo2|o2 sat|weight|height|bmi|pain score|gcs)\b",
            text_l,
        ))

        contains_imaging = self._has_types(entity_type_counts, {"imaging"}) or bool(re.search(
            r"\b(x[- ]?ray|ct|mri|ultrasound|xray|pet scan|nuclear medicine|mammogram|angiogram)\b",
            text_l,
        ))

        contains_procedure = self._has_types(
            entity_type_counts, {"procedure", "procedure_code"},
        ) or bool(re.search(
            r"\b(surgery|operation|biopsy|endoscopy|colonoscopy|catheterization|dialysis|procedure)\b",
            text_l,
        ))

        contains_date = self._has_types(entity_type_counts, {"date"}) or bool(re.search(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text,
        ))

        contains_patient_id = self._has_types(
            entity_type_counts, {"patient_identifier"},
        ) or bool(re.search(
            r"\b(?:mrn|uhid|patient\s*id|account\s*no)\s*[:#\-]?\s*[a-z0-9]{4,20}\b", text_l,
        ))

        contains_negation = any(
            bool(re.search(p, text, flags=re.I))
            for p in [r"\bno\b", r"\bnot\b", r"\bnone\b", r"\bdenies\b",
                      r"\bnegative for\b", r"\bwithout\b", r"\bw/o\b", r"\bruled out\b"]
        )

        # ── New presence flags ────────────────────────────────────────────────
        contains_allergy = self._has_types(
            entity_type_counts, {"allergy", "allergy_nkda"},
        ) or bool(re.search(r"\b(allerg|nkda|nka|adverse reaction)\w*\b", text_l))

        contains_demographics = self._has_types(
            entity_type_counts,
            {"demographics_age", "demographics_gender", "demographics_dob", "demographics_name"},
        ) or bool(re.search(
            r"\b(\d{1,3})[- ]?(?:year[s]?[- ]?old|yo|y/?o)|dob|date of birth\b", text_l,
        ))

        # ── Section ───────────────────────────────────────────────────────────
        primary_section = self._resolve_primary_section(chunk, section_by_page, text_l)

        # ── Specialty flags ───────────────────────────────────────────────────
        is_radiology_content  = self._is_radiology(chunk_entities, text_l, primary_section)
        is_operative_content  = self._is_operative(chunk_entities, text_l, primary_section)
        is_medication_focused = self._is_medication_focused(chunk_entities, text_l, primary_section)
        is_diagnostic_focused = self._is_diagnostic_focused(chunk_entities, text_l, primary_section)

        # ── Boilerplate ───────────────────────────────────────────────────────
        is_boilerplate = self._detect_boilerplate(text, text_l, token_count, entity_type_counts)

        # ── Clinical importance score ─────────────────────────────────────────
        clinical_importance_score = self._compute_importance_score(
            entity_type_counts, chunk_entities,
            contains_medication, contains_lab, contains_diagnosis, contains_vital,
            contains_allergy, contains_imaging, contains_procedure, is_boilerplate,
        )

        # ── Concept density ───────────────────────────────────────────────────
        total_entities = sum(entity_type_counts.values())
        concept_density = round(total_entities * 100.0 / max(token_count, 1), 4)

        # ── Quality score (preserved column, improved formula) ────────────────
        clinical_quality_score = self._compute_quality_score(
            text, text_l, token_count,
            contains_medication, contains_lab, contains_diagnosis, contains_vital,
            contains_imaging, contains_procedure, contains_allergy,
            primary_section, clinical_importance_score, concept_density, is_boilerplate,
        )

        # ── entities_json ─────────────────────────────────────────────────────
        top_entities = sorted(
            chunk_entities,
            key=lambda e: float(e.get("confidence") or 0),
            reverse=True,
        )[:30]
        entities_json = safe_json_dumps([
            {
                "entity_type":      e.get("entity_type"),
                "text":             (e.get("text") or "")[:200],
                "normalized_value": e.get("normalized_value"),
                "confidence":       round(float(e.get("confidence") or 0), 4),
                "section_name":     e.get("section_name"),
                "is_negated":       e.get("is_negated"),
            }
            for e in top_entities
        ])

        enriched = dict(chunk)
        enriched.update({
            # Preserved columns
            "contains_medication":    contains_medication,
            "contains_lab":           contains_lab,
            "contains_diagnosis":     contains_diagnosis,
            "contains_vital":         contains_vital,
            "contains_imaging":       contains_imaging,
            "contains_procedure":     contains_procedure,
            "contains_negation":      contains_negation,
            "contains_date":          contains_date,
            "contains_patient_id":    contains_patient_id,
            "clinical_quality_score": round(float(clinical_quality_score), 4),
            "primary_section":        primary_section,
            "entity_type_counts_json":safe_json_dumps(entity_type_counts),
            "entities_json":          entities_json,
            # New v2 columns
            "clinical_importance_score": round(float(clinical_importance_score), 4),
            "concept_density":           concept_density,
            "is_boilerplate":            is_boilerplate,
            "contains_allergy":          contains_allergy,
            "contains_demographics":     contains_demographics,
            "is_radiology_content":      is_radiology_content,
            "is_operative_content":      is_operative_content,
            "is_medication_focused":     is_medication_focused,
            "is_diagnostic_focused":     is_diagnostic_focused,
            "chunk_metadata_enricher_version": _VERSION,
            "enriched_at":               utc_now_iso(),
        })
        return enriched

    # ── Static presence helper ────────────────────────────────────────────────

    @staticmethod
    def _has_types(counts: Dict[str, int], types: set) -> bool:
        return any(counts.get(t, 0) > 0 for t in types)

    # ── Clinical importance score ─────────────────────────────────────────────

    def _compute_importance_score(
        self,
        entity_type_counts: Dict[str, int],
        chunk_entities: List[Dict[str, Any]],
        contains_medication: bool,
        contains_lab: bool,
        contains_diagnosis: bool,
        contains_vital: bool,
        contains_allergy: bool,
        contains_imaging: bool,
        contains_procedure: bool,
        is_boilerplate: bool,
    ) -> float:
        if is_boilerplate:
            return 0.05

        score = 0.0
        for et, count in entity_type_counts.items():
            weight = _IMPORTANCE_WEIGHTS.get(et, 0.04)
            score += weight + (count - 1) * min(0.05, weight * 0.2)

        score += (
            0.06 * contains_diagnosis
            + 0.05 * contains_vital
            + 0.05 * contains_medication
            + 0.04 * contains_lab
            + 0.05 * contains_allergy
            + 0.03 * contains_imaging
            + 0.03 * contains_procedure
        )

        high_conf = sum(1 for e in chunk_entities if float(e.get("confidence") or 0) >= 0.80)
        score += high_conf * 0.02

        return round(min(score, 1.0), 4)

    # ── Quality score ─────────────────────────────────────────────────────────

    def _compute_quality_score(
        self,
        text: str,
        text_l: str,
        token_count: int,
        contains_medication: bool,
        contains_lab: bool,
        contains_diagnosis: bool,
        contains_vital: bool,
        contains_imaging: bool,
        contains_procedure: bool,
        contains_allergy: bool,
        primary_section: Optional[str],
        clinical_importance_score: float,
        concept_density: float,
        is_boilerplate: bool,
    ) -> float:
        if is_boilerplate:
            return 0.05

        score = 0.20

        if token_count >= 10:  score += 0.05
        if token_count >= 40:  score += 0.05
        if token_count >= 100: score += 0.05
        if token_count >= 200: score += 0.03

        if primary_section and primary_section not in ("unknown", "other"):
            score += 0.08

        score += (
            0.08 * contains_diagnosis
            + 0.07 * contains_vital
            + 0.06 * contains_medication
            + 0.06 * contains_lab
            + 0.06 * contains_allergy
            + 0.04 * contains_imaging
            + 0.04 * contains_procedure
        )

        score += clinical_importance_score * 0.20
        score += min(concept_density / 50.0, 0.10)

        keyword_hits = sum(1 for w in _CLINICAL_KEYWORDS_ANY if w in text_l.split())
        score += min(keyword_hits * 0.01, 0.08)

        sentence_count = len(re.findall(r"[.!?]", text))
        if sentence_count >= 3: score += 0.04
        if sentence_count >= 8: score += 0.04

        if token_count < 10:
            score -= 0.20
        elif token_count < 20:
            score -= 0.10

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) == 1 and len(lines[0]) < 50:
            score -= 0.10

        return round(max(0.0, min(score, 1.0)), 4)

    # ── Specialty flags ───────────────────────────────────────────────────────

    def _is_radiology(
        self,
        entities: List[Dict[str, Any]],
        text_l: str,
        primary_section: Optional[str],
    ) -> bool:
        if primary_section in ("imaging", "findings", "impression", "technique",
                               "clinical_history", "comparison"):
            return True
        if sum(1 for e in entities if e.get("entity_type") == "imaging") >= 2:
            return True
        words = set(text_l.split())
        return sum(1 for w in _RADIOLOGY_KEYWORDS if w in words) >= 3

    def _is_operative(
        self,
        entities: List[Dict[str, Any]],
        text_l: str,
        primary_section: Optional[str],
    ) -> bool:
        if primary_section in ("operative_findings", "preoperative_diagnosis",
                               "postoperative_diagnosis", "procedure_performed",
                               "anesthesia", "procedure", "specimens", "complications"):
            return True
        if sum(1 for e in entities if e.get("entity_type") in ("procedure", "procedure_code")) >= 2:
            return True
        return sum(1 for w in _OPERATIVE_KEYWORDS if w in text_l) >= 3

    def _is_medication_focused(
        self,
        entities: List[Dict[str, Any]],
        text_l: str,
        primary_section: Optional[str],
    ) -> bool:
        if primary_section == "medications":
            return True
        med_types = {"medication", "medication_class", "medication_candidate", "medication_line_candidate"}
        if sum(1 for e in entities if e.get("entity_type") in med_types) >= 3:
            return True
        words = set(text_l.split())
        return sum(1 for w in _MEDICATION_KEYWORDS if w in words) >= 4

    def _is_diagnostic_focused(
        self,
        entities: List[Dict[str, Any]],
        text_l: str,
        primary_section: Optional[str],
    ) -> bool:
        if primary_section in ("assessment", "impression", "plan", "chief_complaint", "hpi"):
            return True
        diag_types = {"diagnosis_or_problem", "past_diagnosis", "diagnosis_code"}
        if sum(1 for e in entities if e.get("entity_type") in diag_types) >= 2:
            return True
        return sum(1 for w in _DIAGNOSTIC_KEYWORDS if w in text_l) >= 2

    # ── Boilerplate detection ─────────────────────────────────────────────────

    def _detect_boilerplate(
        self,
        text: str,
        text_l: str,
        token_count: int,
        entity_type_counts: Dict[str, int],
    ) -> bool:
        total_entities = sum(entity_type_counts.values())

        if token_count < 8 and total_entities == 0:
            return True

        words = set(text_l.split())
        if sum(1 for sig in _BOILERPLATE_SIGNALS if sig in words) >= 2:
            return True

        stripped = re.sub(r"[\s\-_=|/.,:;#*()]", "", text)
        if stripped and stripped.isdigit():
            return True

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if (
            len(lines) == 1
            and len(lines[0]) < 60
            and lines[0].isupper()
            and total_entities == 0
            and not any(w in text_l.split() for w in _CLINICAL_KEYWORDS_ANY)
        ):
            return True

        return False

    # ── Section resolution ────────────────────────────────────────────────────

    def _resolve_primary_section(
        self,
        chunk: Dict[str, Any],
        section_by_page: Dict[int, str],
        text_l: str,
    ) -> Optional[str]:
        section = chunk.get("section_name") or chunk.get("section_key")
        if section:
            return str(section).strip()

        start_page = chunk.get("start_page") or chunk.get("page_start")
        if start_page is not None:
            try:
                s = section_by_page.get(int(start_page))
                if s:
                    return s
            except (TypeError, ValueError):
                pass

        return self._infer_section_from_text(text_l)

    def _infer_section_from_text(self, text_l: str) -> Optional[str]:
        checks = [
            (r"\b(?:nkda|nka|allerg)\w*\b",                          "allergies"),
            (r"\b(?:chief complaint|reason for visit)\b",             "chief_complaint"),
            (r"\b(?:history of present illness|hpi)\b",               "hpi"),
            (r"\b(?:review of systems|ros)\b",                        "ros"),
            (r"\b(?:vital signs?|blood pressure|heart rate|rr)\b",    "vitals"),
            (r"\b(?:physical exam|physical examination)\b",           "physical_exam"),
            (r"\b(?:assessment|impression|diagnosis|diagnoses)\b",    "assessment"),
            (r"\b(?:medication|drug|prescription|take tablet)\b",     "medications"),
            (r"\b(?:lab(?:oratory)?|blood work|test result)\b",       "labs"),
            (r"\b(?:ct|mri|x-?ray|ultrasound|imaging|radiology)\b",  "imaging"),
            (r"\b(?:plan|recommendation|next step|follow.?up)\b",     "plan"),
            (r"\b(?:family history|fhx)\b",                           "family_history"),
            (r"\b(?:social history|shx|tobacco|smoking|alcohol)\b",   "social_history"),
            (r"\b(?:past medical history|pmh)\b",                     "past_medical_history"),
            (r"\b(?:discharge|hospital course|length of stay)\b",     "hospital_course"),
            (r"\b(?:operative|surgery|procedure performed|preoperative|postoperative)\b", "procedure"),
        ]
        for pattern, section_key in checks:
            if re.search(pattern, text_l):
                return section_key
        return None

    # ── Entity gathering ──────────────────────────────────────────────────────

    def _gather_chunk_entities(
        self,
        chunk: Dict[str, Any],
        elem_by_id: Dict[Any, Dict[str, Any]],
        entities_by_elem: Dict[Any, List[Dict[str, Any]]],
        entities_by_page: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []

        # Strategy 1: via element_ids stored in the chunk
        element_ids_raw = chunk.get("element_ids_json") or chunk.get("element_ids")
        if element_ids_raw:
            try:
                eids = json.loads(element_ids_raw) if isinstance(element_ids_raw, str) else element_ids_raw
            except Exception:
                eids = []
            for eid in (eids or []):
                result.extend(entities_by_elem.get(eid) or [])

        # Strategy 2: via page range
        if not result:
            start_page = chunk.get("start_page") or chunk.get("page_start")
            end_page   = chunk.get("end_page")   or chunk.get("page_end")
            if start_page is not None:
                try:
                    sp = int(start_page)
                    ep = int(end_page) if end_page is not None else sp
                    for pn in range(sp, ep + 1):
                        result.extend(entities_by_page.get(pn) or [])
                except (TypeError, ValueError):
                    pass

        # Strategy 3: text substring match (last resort)
        if not result:
            chunk_text_l = lower_clean(chunk.get("text") or chunk.get("text_preview") or "")
            for ent_list in entities_by_elem.values():
                for ent in ent_list:
                    ent_text_l = lower_clean(str(ent.get("text") or ""))
                    if ent_text_l and len(ent_text_l) >= 5 and ent_text_l in chunk_text_l:
                        result.append(ent)

        return result

    # ── Section page map ──────────────────────────────────────────────────────

    @staticmethod
    def _build_section_page_map(sections: List[Dict[str, Any]]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for sec in sections:
            section_key = sec.get("section_key") or sec.get("section_name")
            if not section_key:
                continue
            start_page = sec.get("start_page")
            end_page   = sec.get("end_page")
            if start_page is None:
                continue
            try:
                sp = int(start_page)
                ep = int(end_page) if end_page is not None else sp
                for pn in range(sp, ep + 1):
                    if pn not in mapping:
                        mapping[pn] = str(section_key)
            except (TypeError, ValueError):
                pass
        return mapping


# ─── Pipeline entry point ─────────────────────────────────────────────────────

def enrich_chunks(document_dir: Union[str, Path]) -> Dict[str, Any]:
    """Entry point called by the pipeline orchestrator."""
    return ChunkMetadataEnricher(document_dir).run()

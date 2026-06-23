from __future__ import annotations

import re
import json
import math
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def safe_json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        table = pa.Table.from_pylist([])
    else:
        table = pa.Table.from_pylist(records)

    pq.write_table(table, path, compression="zstd")


def read_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def lower_clean(text: str) -> str:
    return normalize_space(text).lower()


def normalize_heading(text: str) -> str:
    text = lower_clean(text)
    text = text.strip(" :-–—\t\n")
    text = re.sub(r"^\d+(\.\d+)*[\).:\-\s]+", "", text)
    return text.strip()


def first_non_empty(*values):
    for v in values:
        if v not in [None, "", [], {}]:
            return v
    return None


def line_page(row: Dict[str, Any]) -> Optional[int]:
    try:
        return int(row.get("page_number"))
    except Exception:
        return None


DOCUMENT_TYPE_RULES = {
    "Visit Note": [
        r"\bvisit note\b",
        r"\boffice visit\b",
        r"\bclinic visit\b",
        r"\bfollow[- ]?up visit\b",
    ],
    "Progress Note": [
        r"\bprogress note\b",
        r"\bsoap note\b",
    ],
    "Discharge Summary": [
        r"\bdischarge summary\b",
        r"\bdischarged\b",
        r"\bhospital course\b",
    ],
    "Consultation Note": [
        r"\bconsultation\b",
        r"\bconsult note\b",
        r"\breason for consult\b",
    ],
    "Operative Report": [
        r"\boperative report\b",
        r"\boperation performed\b",
        r"\bprocedure performed\b",
        r"\bpreoperative diagnosis\b",
        r"\bpostoperative diagnosis\b",
    ],
    "Radiology Report": [
        r"\bradiology\b",
        r"\bimpression\b",
        r"\bfindings\b",
        r"\bct\b",
        r"\bmri\b",
        r"\bx[- ]?ray\b",
        r"\bultrasound\b",
    ],
    "Lab Report": [
        r"\blab(oratory)? report\b",
        r"\btest name\b",
        r"\bresult\b",
        r"\breference range\b",
        r"\bunits?\b",
    ],
    "Emergency Department Note": [
        r"\bemergency department\b",
        r"\bed note\b",
        r"\btriage\b",
    ],
    "Telephone Encounter": [
        r"\btelephone encounter\b",
        r"\bphone call\b",
        r"\bcalled patient\b",
    ],
    "Prescription": [
        r"\brx\b",
        r"\bprescription\b",
        r"\btake \d+\b",
    ],
}


SECTION_ALIASES = {
    "chief_complaint": [
        "chief complaint", "cc", "reason for visit", "presenting complaint"
    ],
    "hpi": [
        "history of present illness", "hpi", "present illness"
    ],
    "past_medical_history": [
        "past medical history", "pmh", "medical history", "history"
    ],
    "surgical_history": [
        "past surgical history", "surgical history", "psh"
    ],
    "family_history": [
        "family history", "fhx"
    ],
    "social_history": [
        "social history", "shx"
    ],
    "allergies": [
        "allergy", "allergies", "drug allergies", "known allergies"
    ],
    "medications": [
        "medication", "medications", "current medications", "home medications",
        "discharge medications", "prescriptions", "rx"
    ],
    "vitals": [
        "vitals", "vital signs", "vital", "measurements"
    ],
    "physical_exam": [
        "physical exam", "physical examination", "exam", "examination"
    ],
    "assessment": [
        "assessment", "impression", "diagnosis", "diagnoses", "dx"
    ],
    "plan": [
        "plan", "treatment plan", "recommendations", "recommendation"
    ],
    "labs": [
        "labs", "laboratory", "lab results", "blood work", "test results"
    ],
    "imaging": [
        "imaging", "radiology", "xray", "x-ray", "ct", "mri", "ultrasound"
    ],
    "procedure": [
        "procedure", "procedures", "operation", "surgery"
    ],
    "follow_up": [
        "follow up", "follow-up", "return to clinic", "rtc"
    ],
    "ros": [
        "review of systems", "ros"
    ],
}


NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bnegative for\b",
    r"\bwithout\b",
    r"\bnot\b",
    r"\bno evidence of\b",
    r"\bruled out\b",
]


COMMON_MEDICATION_WORDS = [
    "tablet", "tab", "capsule", "cap", "injection", "inj", "syrup",
    "cream", "ointment", "drops", "solution", "suspension", "mg", "mcg",
    "g", "ml", "iu", "unit", "units", "daily", "bid", "tid", "qid",
    "od", "bd", "hs", "prn", "po", "iv", "im", "sc", "subcutaneous"
]


COMMON_LAB_NAMES = [
    "hb", "hgb", "hemoglobin", "wbc", "rbc", "platelet", "platelets",
    "creatinine", "urea", "sodium", "potassium", "chloride", "glucose",
    "hba1c", "bilirubin", "sgpt", "sgot", "alt", "ast", "crp", "esr",
    "tsh", "cholesterol", "triglycerides", "hdl", "ldl"
]


VITAL_PATTERNS = {
    "blood_pressure": r"\b(?:bp|blood pressure)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b",
    "heart_rate": r"\b(?:hr|heart rate|pulse)\s*[:\-]?\s*(\d{2,3})\b",
    "temperature": r"\b(?:temp|temperature)\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:f|c|°f|°c)?\b",
    "respiratory_rate": r"\b(?:rr|respiratory rate|resp rate)\s*[:\-]?\s*(\d{1,3})\b",
    "oxygen_saturation": r"\b(?:spo2|o2 sat|oxygen saturation)\s*[:\-]?\s*(\d{2,3})\s*%?\b",
    "weight": r"\b(?:wt|weight)\s*[:\-]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:kg|kgs|lb|lbs)?\b",
}


class ClinicalMetadataExtractor:
    """
    Clinical metadata extractor v1.

    Input:
      - layout_elements_with_sections.parquet OR layout_elements.parquet
      - layout_manifest.json

    Output:
      - clinical_document_metadata.json
      - clinical_sections.parquet
      - clinical_entities.parquet
      - clinical_element_metadata.parquet

    Design:
      - fast rule-based extraction
      - no mandatory external NLP dependency
      - metadata includes confidence/evidence/source/page
      - later you can plug medSpaCy/scispaCy/LLM into the same structure
    """

    def __init__(
        self,
        document_dir: Union[str, Path],
        min_entity_confidence: float = 0.55,
    ):
        self.document_dir = Path(document_dir)
        self.min_entity_confidence = min_entity_confidence

        self.manifest_path = self.document_dir / "layout_manifest.json"

        preferred = self.document_dir / "layout_elements_with_sections.parquet"
        fallback = self.document_dir / "layout_elements.parquet"

        if preferred.exists():
            self.elements_path = preferred
        elif fallback.exists():
            self.elements_path = fallback
        else:
            raise FileNotFoundError(
                f"No layout elements file found in {self.document_dir}"
            )

        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {}

    def run(self) -> Dict[str, Any]:
        elements = read_parquet(self.elements_path)
        line_elements = self._line_elements(elements)

        full_text = "\n".join(e.get("text") or e.get("text_preview") or "" for e in line_elements)
        first_pages_text = self._first_pages_text(line_elements, max_pages=2)

        document_metadata = self.extract_document_metadata(
            full_text=full_text,
            first_pages_text=first_pages_text,
            line_elements=line_elements,
        )

        section_rows = self.extract_sections(line_elements)
        entity_rows = self.extract_entities(line_elements)
        element_metadata_rows = self.extract_element_metadata(line_elements)

        output = {
            "document_metadata": document_metadata,
            "sections_count": len(section_rows),
            "entities_count": len(entity_rows),
            "element_metadata_count": len(element_metadata_rows),
            "created_at": utc_now_iso(),
            "input_elements_path": str(self.elements_path),
        }

        (self.document_dir / "clinical_document_metadata.json").write_text(
            json.dumps(document_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        write_parquet(section_rows, self.document_dir / "clinical_sections.parquet")
        write_parquet(entity_rows, self.document_dir / "clinical_entities.parquet")
        write_parquet(element_metadata_rows, self.document_dir / "clinical_element_metadata.parquet")

        print("Clinical metadata extraction complete.")
        print(f"Document type: {document_metadata.get('document_type')}")
        print(f"Sections: {len(section_rows)}")
        print(f"Entities: {len(entity_rows)}")
        print(f"Element metadata rows: {len(element_metadata_rows)}")

        return output

    # ------------------------------------------------------------------
    # Core document metadata
    # ------------------------------------------------------------------

    def extract_document_metadata(
        self,
        full_text: str,
        first_pages_text: str,
        line_elements: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        doc_type = self.classify_document_type(first_pages_text)

        dates = self.extract_dates_from_text(full_text)
        patient_ids = self.extract_patient_ids(full_text)
        provider_names = self.extract_provider_names(first_pages_text)
        facility_names = self.extract_facility_names(first_pages_text)

        return {
            "document_id": self.manifest.get("document_id"),
            "source_pdf_name": self.manifest.get("source_pdf_name"),
            "source_pdf_path": self.manifest.get("source_pdf_path"),
            "source_sha256": self.manifest.get("source_sha256"),
            "total_pages": self.manifest.get("total_pages"),
            "total_tokens": self.manifest.get("total_tokens"),

            "document_type": doc_type["label"],
            "document_type_confidence": doc_type["confidence"],
            "document_type_evidence": doc_type["evidence"],
            "document_type_scores_json": safe_json_dumps(doc_type["scores"]),

            "dates_json": safe_json_dumps(dates),
            "patient_ids_json": safe_json_dumps(patient_ids),
            "provider_names_json": safe_json_dumps(provider_names),
            "facility_names_json": safe_json_dumps(facility_names),

            "clinical_metadata_extractor_version": "clinical_metadata_v1_rules",
            "created_at": utc_now_iso(),
        }

    def classify_document_type(self, text: str) -> Dict[str, Any]:
        text_l = lower_clean(text)
        scores = {}

        for label, patterns in DOCUMENT_TYPE_RULES.items():
            score = 0.0
            evidence = []

            for pattern in patterns:
                matches = list(re.finditer(pattern, text_l, flags=re.I))
                if matches:
                    score += min(0.35, 0.12 * len(matches))
                    evidence.append(pattern)

            scores[label] = {
                "score": round(min(score, 1.0), 4),
                "evidence": evidence[:5],
            }

        best_label = "Unknown"
        best_score = 0.0
        best_evidence = []

        for label, obj in scores.items():
            if obj["score"] > best_score:
                best_label = label
                best_score = obj["score"]
                best_evidence = obj["evidence"]

        if best_score == 0:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "evidence": [],
                "scores": scores,
            }

        confidence = min(0.95, 0.45 + best_score)

        return {
            "label": best_label,
            "confidence": round(confidence, 4),
            "evidence": best_evidence,
            "scores": scores,
        }

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def extract_sections(self, line_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        current_section = None
        current_start_page = None
        current_start_element_id = None
        current_text_parts = []

        for e in line_elements:
            text = normalize_space(e.get("text") or e.get("text_preview") or "")
            if not text:
                continue

            section_key, confidence = self.detect_section(text, e)

            if section_key:
                if current_section is not None:
                    rows.append({
                        "document_id": self.manifest.get("document_id"),
                        "section_key": current_section,
                        "section_name": current_section,
                        "start_page": current_start_page,
                        "end_page": line_page(e),
                        "start_element_id": current_start_element_id,
                        "text_preview": normalize_space(" ".join(current_text_parts))[:1000],
                        "confidence": 0.75,
                        "source": "section_rules",
                    })

                current_section = section_key
                current_start_page = line_page(e)
                current_start_element_id = e.get("element_id")
                current_text_parts = [text]
            else:
                if current_section is not None:
                    current_text_parts.append(text)

        if current_section is not None:
            rows.append({
                "document_id": self.manifest.get("document_id"),
                "section_key": current_section,
                "section_name": current_section,
                "start_page": current_start_page,
                "end_page": line_page(line_elements[-1]) if line_elements else None,
                "start_element_id": current_start_element_id,
                "text_preview": normalize_space(" ".join(current_text_parts))[:1000],
                "confidence": 0.75,
                "source": "section_rules",
            })

        return rows

    def detect_section(
        self,
        text: str,
        element: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], float]:
        norm = normalize_heading(text)

        # If layout extractor already marks heading candidate, trust it more.
        is_heading_candidate = bool(element and element.get("is_heading_candidate"))

        for section_key, aliases in SECTION_ALIASES.items():
            for alias in aliases:
                alias_norm = normalize_heading(alias)

                if norm == alias_norm:
                    return section_key, 0.95

                if is_heading_candidate and alias_norm in norm and len(norm) <= 80:
                    return section_key, 0.85

                # Handles "Assessment:" / "Plan -"
                if re.match(rf"^{re.escape(alias_norm)}\s*[:\-–—]?$", norm):
                    return section_key, 0.9

        return None, 0.0

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def extract_entities(self, line_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities = []

        for e in line_elements:
            text = normalize_space(e.get("text") or e.get("text_preview") or "")
            if not text:
                continue

            page = line_page(e)
            element_id = e.get("element_id")
            section_name = self._element_section_name(e)

            entities.extend(self.extract_date_entities(text, page, element_id, section_name))
            entities.extend(self.extract_patient_id_entities(text, page, element_id, section_name))
            entities.extend(self.extract_vital_entities(text, page, element_id, section_name))
            entities.extend(self.extract_lab_entities(text, page, element_id, section_name))
            entities.extend(self.extract_medication_like_entities(text, page, element_id, section_name))
            entities.extend(self.extract_diagnosis_like_entities(text, page, element_id, section_name))
            entities.extend(self.extract_imaging_entities(text, page, element_id, section_name))
            entities.extend(self.extract_procedure_entities(text, page, element_id, section_name))

        # Deduplicate near-identical entities.
        entities = self._dedupe_entities(entities)

        return [
            x for x in entities
            if float(x.get("confidence") or 0) >= self.min_entity_confidence
        ]

    def _base_entity(
        self,
        entity_type: str,
        text: str,
        normalized_value: Any,
        page_number: Optional[int],
        element_id: Any,
        section_name: Optional[str],
        confidence: float,
        source: str,
        evidence: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "document_id": self.manifest.get("document_id"),
            "entity_type": entity_type,
            "text": text,
            "normalized_value": normalized_value,
            "page_number": page_number,
            "element_id": element_id,
            "section_name": section_name,
            "confidence": round(float(confidence), 4),
            "source": source,
            "is_negated": self.is_negated_context(evidence, text),
            "evidence": evidence[:1000],
            "attributes_json": safe_json_dumps(attributes or {}),
            "created_at": utc_now_iso(),
        }

    def extract_date_entities(self, text, page, element_id, section_name):
        rows = []
        for d in self.extract_dates_from_text(text):
            rows.append(self._base_entity(
                entity_type="date",
                text=d["raw"],
                normalized_value=d["normalized"],
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=d["confidence"],
                source="date_regex",
                evidence=text,
                attributes=d,
            ))
        return rows

    def extract_patient_id_entities(self, text, page, element_id, section_name):
        rows = []
        for obj in self.extract_patient_ids(text):
            rows.append(self._base_entity(
                entity_type="patient_identifier",
                text=obj["raw"],
                normalized_value=obj["value"],
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=obj["confidence"],
                source="patient_id_regex",
                evidence=text,
                attributes=obj,
            ))
        return rows

    def extract_vital_entities(self, text, page, element_id, section_name):
        rows = []

        for vital_name, pattern in VITAL_PATTERNS.items():
            for m in re.finditer(pattern, text, flags=re.I):
                rows.append(self._base_entity(
                    entity_type="vital",
                    text=m.group(0),
                    normalized_value=m.group(0),
                    page_number=page,
                    element_id=element_id,
                    section_name=section_name,
                    confidence=0.82,
                    source="vital_regex",
                    evidence=text,
                    attributes={
                        "vital_type": vital_name,
                        "groups": m.groups(),
                    },
                ))

        return rows

    def extract_lab_entities(self, text, page, element_id, section_name):
        rows = []
        text_l = lower_clean(text)

        # Example: Hb 12.3 g/dL, Creatinine: 1.1 mg/dL
        lab_name_pattern = "|".join(re.escape(x) for x in COMMON_LAB_NAMES)
        pattern = rf"\b({lab_name_pattern})\b\s*[:\-]?\s*(\d+(\.\d+)?)\s*([a-zA-Z/%µμ]+(?:/[a-zA-Z]+)?)?"

        for m in re.finditer(pattern, text_l, flags=re.I):
            rows.append(self._base_entity(
                entity_type="lab_result",
                text=m.group(0),
                normalized_value=m.group(0),
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=0.78,
                source="lab_regex",
                evidence=text,
                attributes={
                    "lab_name": m.group(1),
                    "value": m.group(2),
                    "unit": m.group(4),
                },
            ))

        return rows

    def extract_medication_like_entities(self, text, page, element_id, section_name):
        rows = []
        text_l = lower_clean(text)

        section_boost = 0.15 if section_name in ["medications", "plan"] else 0.0

        # Heuristic medication phrase:
        # Name + dose/unit, e.g. "Metformin 500 mg", "Tab Paracetamol 650mg"
        pattern = r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,2})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b"

        for m in re.finditer(pattern, text):
            med_text = m.group(0)
            confidence = min(0.9, 0.62 + section_boost)

            rows.append(self._base_entity(
                entity_type="medication_candidate",
                text=med_text,
                normalized_value=med_text,
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=confidence,
                source="medication_dose_regex",
                evidence=text,
                attributes={
                    "name_candidate": m.group(1),
                    "dose": m.group(2),
                    "unit": m.group(3),
                },
            ))

        # Medication route/frequency line cue.
        if section_name == "medications" or any(w in text_l for w in COMMON_MEDICATION_WORDS):
            if re.search(r"\b(take|tab|tablet|cap|capsule|inj|injection|apply)\b", text_l):
                rows.append(self._base_entity(
                    entity_type="medication_line_candidate",
                    text=text[:200],
                    normalized_value=text[:200],
                    page_number=page,
                    element_id=element_id,
                    section_name=section_name,
                    confidence=min(0.8, 0.6 + section_boost),
                    source="medication_line_rule",
                    evidence=text,
                    attributes={},
                ))

        return rows

    def extract_diagnosis_like_entities(self, text, page, element_id, section_name):
        rows = []

        # ICD-like code
        for m in re.finditer(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", text):
            rows.append(self._base_entity(
                entity_type="diagnosis_code_candidate",
                text=m.group(0),
                normalized_value=m.group(0),
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=0.68,
                source="icd_like_regex",
                evidence=text,
                attributes={},
            ))

        if section_name in ["assessment", "chief_complaint", "hpi"]:
            # Stronger only in clinical assessment-like sections.
            if len(text.split()) <= 30 and re.search(r"\b(pain|fever|cough|diabetes|hypertension|infection|fracture|injury|disease|syndrome|failure|asthma|copd)\b", text, flags=re.I):
                rows.append(self._base_entity(
                    entity_type="diagnosis_or_problem_candidate",
                    text=text[:250],
                    normalized_value=text[:250],
                    page_number=page,
                    element_id=element_id,
                    section_name=section_name,
                    confidence=0.65,
                    source="problem_section_rule",
                    evidence=text,
                    attributes={},
                ))

        return rows

    def extract_imaging_entities(self, text, page, element_id, section_name):
        rows = []

        pattern = r"\b(x[- ]?ray|ct|mri|ultrasound|usg|sonography|radiograph|echocardiogram|echo)\b"
        for m in re.finditer(pattern, text, flags=re.I):
            rows.append(self._base_entity(
                entity_type="imaging_candidate",
                text=m.group(0),
                normalized_value=m.group(0).lower(),
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=0.72,
                source="imaging_keyword_rule",
                evidence=text,
                attributes={},
            ))

        return rows

    def extract_procedure_entities(self, text, page, element_id, section_name):
        rows = []

        pattern = r"\b(surgery|operation|procedure|biopsy|endoscopy|colonoscopy|angiography|angioplasty|stent|incision|drainage|suturing|repair)\b"
        for m in re.finditer(pattern, text, flags=re.I):
            rows.append(self._base_entity(
                entity_type="procedure_candidate",
                text=m.group(0),
                normalized_value=m.group(0).lower(),
                page_number=page,
                element_id=element_id,
                section_name=section_name,
                confidence=0.68,
                source="procedure_keyword_rule",
                evidence=text,
                attributes={},
            ))

        return rows

    # ------------------------------------------------------------------
    # Element-level metadata
    # ------------------------------------------------------------------

    def extract_element_metadata(self, line_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []

        for e in line_elements:
            text = normalize_space(e.get("text") or e.get("text_preview") or "")
            if not text:
                continue

            section_key, section_conf = self.detect_section(text, e)
            existing_section = self._element_section_name(e)

            rows.append({
                "document_id": self.manifest.get("document_id"),
                "element_id": e.get("element_id"),
                "page_number": line_page(e),
                "text_preview": text[:500],

                "section_detected": section_key,
                "section_confidence": section_conf,
                "section_name": existing_section,

                "is_heading_candidate": bool(e.get("is_heading_candidate")),
                "heading_score": float(e.get("heading_score") or 0.0),
                "is_repeated_header_candidate": bool(e.get("is_repeated_header_candidate")),
                "is_repeated_footer_candidate": bool(e.get("is_repeated_footer_candidate")),

                "contains_date": bool(self.extract_dates_from_text(text)),
                "contains_patient_id": bool(self.extract_patient_ids(text)),
                "contains_vital": any(re.search(p, text, flags=re.I) for p in VITAL_PATTERNS.values()),
                "contains_lab_candidate": any(re.search(rf"\b{re.escape(x)}\b", text, flags=re.I) for x in COMMON_LAB_NAMES),
                "contains_medication_cue": any(w in lower_clean(text) for w in COMMON_MEDICATION_WORDS),
                "contains_negation": self.contains_negation(text),

                "clinical_metadata_json": safe_json_dumps({
                    "source": "ClinicalMetadataExtractor.v1",
                }),
            })

        return rows

    # ------------------------------------------------------------------
    # Pattern extractors
    # ------------------------------------------------------------------

    def extract_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b",
            r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b",
            r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})\b",
        ]

        out = []
        seen = set()

        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.I):
                raw = m.group(0)
                if raw in seen:
                    continue
                seen.add(raw)

                out.append({
                    "raw": raw,
                    "normalized": self._normalize_date(raw),
                    "confidence": 0.78,
                    "source": "date_regex",
                })

        return out

    def _normalize_date(self, raw: str) -> str:
        # Simple safe normalization. Keep raw if ambiguous.
        raw = raw.strip()

        # yyyy-mm-dd or yyyy/mm/dd
        m = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw)
        if m:
            y, mo, d = map(int, m.groups())
            return f"{y:04d}-{mo:02d}-{d:02d}"

        # dd-mm-yyyy OR mm-dd-yyyy ambiguous.
        # For Indian/clinical local usage, default to DD-MM-YYYY.
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", raw)
        if m:
            d, mo, y = m.groups()
            d, mo, y = int(d), int(mo), int(y)
            if y < 100:
                y += 2000
            return f"{y:04d}-{mo:02d}-{d:02d}"

        return raw

    def extract_patient_ids(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            r"\b(?:MRN|UHID|Patient\s*ID|Patient\s*No\.?|Reg(?:istration)?\s*No\.?)\s*[:#\-]?\s*([A-Za-z0-9\-\/]+)\b",
            r"\b(?:Case\s*ID|Case\s*No\.?)\s*[:#\-]?\s*([A-Za-z0-9\-\/]+)\b",
        ]

        out = []
        seen = set()

        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.I):
                raw = m.group(0)
                value = m.group(1)

                if raw in seen:
                    continue
                seen.add(raw)

                out.append({
                    "raw": raw,
                    "value": value,
                    "confidence": 0.86,
                    "source": "patient_id_regex",
                })

        return out

    def extract_provider_names(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            r"\bDr\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
            r"\bDoctor\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
            r"\bProvider\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
            r"\bConsultant\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
        ]

        out = []
        seen = set()

        for pattern in patterns:
            for m in re.finditer(pattern, text):
                raw = m.group(0)
                if raw in seen:
                    continue
                seen.add(raw)

                out.append({
                    "raw": raw,
                    "name": m.group(1),
                    "confidence": 0.72,
                    "source": "provider_regex",
                })

        return out

    def extract_facility_names(self, text: str) -> List[Dict[str, Any]]:
        lines = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]
        candidates = []

        for line in lines[:20]:
            if re.search(r"\b(hospital|clinic|medical center|healthcare|nursing home|diagnostic|lab|laboratory)\b", line, flags=re.I):
                candidates.append({
                    "raw": line,
                    "name": line,
                    "confidence": 0.65,
                    "source": "facility_keyword_top_page",
                })

        return candidates[:5]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _line_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines = [
            e for e in elements
            if e.get("element_type") == "line"
        ]

        lines.sort(key=lambda e: (
            int(e.get("page_number") or 0),
            float(e.get("y0") or 0),
            float(e.get("x0") or 0),
            int(e.get("element_id") or 0),
        ))

        return lines

    def _first_pages_text(self, lines: List[Dict[str, Any]], max_pages: int = 2) -> str:
        selected = []
        for e in lines:
            page = line_page(e)
            if page is not None and page < max_pages:
                selected.append(e.get("text") or e.get("text_preview") or "")
        return "\n".join(selected)

    def _element_section_name(self, e: Dict[str, Any]) -> Optional[str]:
        section_name = e.get("section_name")
        if section_name:
            mapped, _ = self.detect_section(section_name)
            return mapped or normalize_heading(section_name)

        heading_path_json = e.get("heading_path_json")
        hp = safe_json_loads(heading_path_json)
        if isinstance(hp, list) and hp:
            mapped, _ = self.detect_section(str(hp[-1]))
            return mapped or normalize_heading(str(hp[-1]))

        return None

    def contains_negation(self, text: str) -> bool:
        return any(re.search(p, text, flags=re.I) for p in NEGATION_PATTERNS)

    def is_negated_context(self, full_text: str, entity_text: str, window_chars: int = 80) -> bool:
        text_l = lower_clean(full_text)
        entity_l = lower_clean(entity_text)

        idx = text_l.find(entity_l)
        if idx == -1:
            return self.contains_negation(full_text)

        start = max(0, idx - window_chars)
        context = text_l[start:idx + len(entity_l)]

        return self.contains_negation(context)

    def _dedupe_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best = {}

        for e in entities:
            key = (
                e.get("entity_type"),
                lower_clean(e.get("normalized_value")),
                e.get("page_number"),
                e.get("section_name"),
            )

            old = best.get(key)
            if old is None or float(e.get("confidence") or 0) > float(old.get("confidence") or 0):
                best[key] = e

        return list(best.values())

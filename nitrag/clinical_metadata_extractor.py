"""Clinical Metadata Extractor v2 — rule-based, no mandatory external NLP.

Improvements over v1:
- 22 document types with weighted discriminative patterns
- 52 section types with fuzzy header matching (Jaccard fallback)
- 300+ drug names + drug class suffix patterns
- 120+ lab test names across all major panels
- Extended vital patterns: height, BMI, pain score, GCS, blood glucose, INR
- Medical condition terms list for diagnosis extraction
- Sentence-scoped negation (not just fixed window)
- Written-date parsing (January 15, 2020) + US-format priority
- Demographics: age, gender, name, DOB
- Allergy entity extraction
- Provider credentials: MD, DO, NP, PA, APRN, RN, PharmD
- Social history: smoking, alcohol, drugs, occupation
"""
from __future__ import annotations

import re
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_VERSION = "clinical_metadata_v2_rules"

# ─── Utilities ────────────────────────────────────────────────────────────────

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


def _coerce_value(v: Any) -> Any:
    """Convert non-primitive values to a PyArrow-safe type."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def write_parquet(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        pq.write_table(pa.table({}), path, compression="zstd")
        return

    # Step 1: coerce non-primitives
    clean = [{k: _coerce_value(v) for k, v in rec.items()} for rec in records]

    # Step 2: detect mixed-type columns (e.g. str vs int in normalized_value)
    # and convert those columns entirely to str to prevent pa.Table.from_pylist failure
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


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z]+", a.lower()))
    tb = set(re.findall(r"[a-z]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─── Document-type rules ──────────────────────────────────────────────────────
# Each rule: (pattern, weight)  weight ∈ {0.40 exclusive, 0.25 strong, 0.12 shared}

DOCUMENT_TYPE_RULES: Dict[str, List[Tuple[str, float]]] = {
    "Visit Note": [
        (r"\bvisit note\b", 0.40),
        (r"\boffice visit\b", 0.40),
        (r"\bclinic(al)? visit\b", 0.35),
        (r"\bfollow[- ]?up visit\b", 0.35),
        (r"\bprogress note\b", 0.12),
        (r"\bchief complaint\b", 0.12),
        (r"\bhistory of present illness\b", 0.12),
    ],
    "Progress Note": [
        (r"\bprogress note\b", 0.40),
        (r"\bsoap note\b", 0.40),
        (r"\bdaily note\b", 0.35),
        (r"\bhospital day\b", 0.25),
        (r"\binpatient note\b", 0.35),
    ],
    "Discharge Summary": [
        (r"\bdischarge summary\b", 0.40),
        (r"\bdischarge note\b", 0.40),
        (r"\bhospital course\b", 0.25),
        (r"\bdischarged (home|to)\b", 0.35),
        (r"\bdischarge (diagnosis|diagnoses)\b", 0.40),
        (r"\bdate of (admission|discharge)\b", 0.25),
        (r"\blength of stay\b", 0.25),
        (r"\bdischarge medications?\b", 0.25),
        (r"\bdischarge instructions?\b", 0.12),
    ],
    "Consultation Note": [
        (r"\bconsultation (note|report|request)\b", 0.40),
        (r"\bconsult note\b", 0.40),
        (r"\breason for consult(ation)?\b", 0.40),
        (r"\breferring physician\b", 0.25),
        (r"\brequest for consultation\b", 0.40),
        (r"\bconsulted (by|for)\b", 0.25),
    ],
    "Operative Report": [
        (r"\boperative report\b", 0.40),
        (r"\boperation performed\b", 0.40),
        (r"\bprocedure performed\b", 0.25),
        (r"\bpreoperative diagnosis\b", 0.40),
        (r"\bpostoperative diagnosis\b", 0.40),
        (r"\bintraoperative (finding|course)\b", 0.40),
        (r"\bsurgeon[:\s]", 0.25),
        (r"\bscrub (tech|nurse)\b", 0.40),
        (r"\bspecimen submitted\b", 0.35),
        (r"\bprocedure time\b", 0.25),
        (r"\bestimated blood loss\b", 0.40),
    ],
    "Radiology Report": [
        (r"\bradiology report\b", 0.40),
        (r"\bradiology findings\b", 0.35),
        (r"\bimaging report\b", 0.35),
        (r"\bct (scan|of)\b", 0.25),
        (r"\bmri (brain|spine|abdomen|pelvis|chest|knee|shoulder|hip|of)\b", 0.25),
        (r"\bx[- ]?ray\b", 0.12),
        (r"\bultrasound (of|abdomen|pelvis|neck|breast)\b", 0.25),
        (r"\b(pet|spect) scan\b", 0.35),
        (r"\bnuclear medicine\b", 0.35),
        (r"\bcontrast[- ]enhanced\b", 0.35),
        (r"\bno intravenous contrast\b", 0.40),
        (r"\bclinical (history|indication)\b", 0.12),
        (r"\bcomparison\b", 0.12),
        (r"\btechnique\b", 0.12),
        (r"\bfindings?\b", 0.12),
        (r"\bimpression\b", 0.12),
    ],
    "Pathology Report": [
        (r"\bpathology report\b", 0.40),
        (r"\bpathological diagnosis\b", 0.40),
        (r"\bgross (description|examination)\b", 0.40),
        (r"\bmicroscopic (description|examination)\b", 0.40),
        (r"\bhistopatholog(y|ical)\b", 0.40),
        (r"\bspecimen type\b", 0.35),
        (r"\bsynoptic report\b", 0.40),
        (r"\bclinical diagnosis\b", 0.12),
        (r"\bimmunohistochemistry\b", 0.40),
        (r"\bmargins? (clear|positive|negative)\b", 0.35),
        (r"\btumor (grade|stage|size)\b", 0.35),
    ],
    "Lab Report": [
        (r"\blaboratory report\b", 0.40),
        (r"\blab(oratory)? results?\b", 0.35),
        (r"\btest name\b", 0.35),
        (r"\breference range\b", 0.40),
        (r"\bnormal (range|value)\b", 0.25),
        (r"\bunits?\b", 0.12),
        (r"\bcollection (date|time)\b", 0.35),
        (r"\bspecimen type\b", 0.25),
        (r"\bhigh\b.*\blow\b", 0.25),
    ],
    "Emergency Department Note": [
        (r"\bemergency department\b", 0.40),
        (r"\bemergency room\b", 0.40),
        (r"\bed (note|visit|encounter)\b", 0.40),
        (r"\btriage (note|level|category)\b", 0.40),
        (r"\barrival (time|mode)\b", 0.25),
        (r"\bdisposition (from )?ed\b", 0.40),
        (r"\bems (arrival|transport|response)\b", 0.35),
    ],
    "Telephone Encounter": [
        (r"\btelephone (encounter|call|note)\b", 0.40),
        (r"\bphone (call|consultation|note)\b", 0.40),
        (r"\bcalled (patient|the patient)\b", 0.40),
        (r"\bpatient called\b", 0.35),
        (r"\bmessage (left|taken|received)\b", 0.25),
    ],
    "Prescription": [
        (r"\bprescription\b", 0.40),
        (r"\b(?:rx|℞)\b", 0.35),
        (r"\btake \d+\b", 0.25),
        (r"\bdispense\b", 0.35),
        (r"\brefills?\b", 0.35),
        (r"\bsig:\b", 0.40),
        (r"\bndc\b", 0.35),
        (r"\bquantity:?\s*\d+\b", 0.35),
    ],
    "Nursing Note": [
        (r"\bnursing (note|assessment|care plan)\b", 0.40),
        (r"\bnurse (note|assessment)\b", 0.40),
        (r"\bshift (note|assessment|summary)\b", 0.40),
        (r"\bpatient (education|teaching)\b", 0.25),
        (r"\bturning (and repositioning|schedule)\b", 0.40),
        (r"\bfall (precautions?|risk)\b", 0.35),
        (r"\bintervention\b", 0.12),
        (r"\boutcome\b", 0.12),
    ],
    "Physical Therapy Note": [
        (r"\bphysical therapy (note|evaluation|assessment|plan)\b", 0.40),
        (r"\bpt (evaluation|note|assessment|goals?)\b", 0.40),
        (r"\brange of motion\b", 0.25),
        (r"\bstrength testing\b", 0.35),
        (r"\bgait (analysis|training|pattern)\b", 0.35),
        (r"\bfunctional (mobility|assessment|goals?)\b", 0.35),
        (r"\btherapeutic (exercise|activity)\b", 0.35),
        (r"\brehabilitation\b", 0.12),
    ],
    "Occupational Therapy Note": [
        (r"\boccupational therapy (note|evaluation)\b", 0.40),
        (r"\bot (evaluation|note|assessment|goals?)\b", 0.40),
        (r"\badl (assessment|training|performance)\b", 0.40),
        (r"\bactivities of daily living\b", 0.40),
        (r"\biadl\b", 0.35),
        (r"\bfine motor\b", 0.35),
    ],
    "Mental Health Note": [
        (r"\bmental health (note|evaluation)\b", 0.40),
        (r"\bpsychiatric (note|evaluation|assessment|intake)\b", 0.40),
        (r"\bpsych(iatry)? note\b", 0.40),
        (r"\btherapy (note|session)\b", 0.35),
        (r"\bmental status (exam|examination)\b", 0.40),
        (r"\bsuicidal (ideation|risk)\b", 0.35),
        (r"\bhomicidal ideation\b", 0.35),
        (r"\bpsychosis\b", 0.25),
        (r"\bmood (and affect|assessment)\b", 0.25),
        (r"\bcognitive behavioral therapy\b", 0.35),
    ],
    "Referral Letter": [
        (r"\breferral (letter|note|request|form)\b", 0.40),
        (r"\bplease (see|evaluate|assess) this (patient|pt)\b", 0.35),
        (r"\bi am referring\b", 0.40),
        (r"\bthank you for (seeing|evaluating|consulting)\b", 0.40),
        (r"\breason for referral\b", 0.40),
        (r"\bfamiliar (physician|doctor)\b", 0.25),
    ],
    "Prior Authorization": [
        (r"\bprior auth(orization)?\b", 0.40),
        (r"\bpa (request|form|approval)\b", 0.35),
        (r"\binsurance (authorization|approval)\b", 0.35),
        (r"\bmedical necessity\b", 0.40),
        (r"\bpolicy number\b", 0.35),
        (r"\bcoverage (request|determination)\b", 0.35),
    ],
    "Anesthesia Note": [
        (r"\banesthesia (note|record|plan|evaluation)\b", 0.40),
        (r"\bpreanesthesia\b", 0.40),
        (r"\bpostanesthesia\b", 0.40),
        (r"\bgeneral anesthesia\b", 0.35),
        (r"\bregional anesthesia\b", 0.35),
        (r"\bspinal (anesthesia|block)\b", 0.35),
        (r"\bepidural\b", 0.25),
        (r"\bairway (assessment|management|class)\b", 0.35),
        (r"\bmallampati\b", 0.40),
    ],
    "Procedure Note": [
        (r"\bprocedure note\b", 0.40),
        (r"\bindications? (for procedure|for the procedure)\b", 0.35),
        (r"\bconsent obtained\b", 0.25),
        (r"\bsterile (prep|drape|technique)\b", 0.35),
        (r"\bsite (marked|confirmed|verified)\b", 0.35),
        (r"\bcomplication[s]? (none|none noted)\b", 0.25),
        (r"\bspecimen sent\b", 0.35),
    ],
    "Vaccine Record": [
        (r"\bimmunization (record|history)\b", 0.40),
        (r"\bvaccine (record|history|administered)\b", 0.40),
        (r"\bvaccination\b", 0.35),
        (r"\blot number\b", 0.40),
        (r"\bmanufacturer[:\s]", 0.35),
        (r"\bvaccine information statement\b", 0.40),
        (r"\bvis date\b", 0.40),
    ],
    "Care Plan": [
        (r"\bcare plan\b", 0.40),
        (r"\btreatment plan\b", 0.25),
        (r"\bgoals? of care\b", 0.40),
        (r"\bproblem list\b", 0.35),
        (r"\binterdisciplinary (plan|team)\b", 0.35),
        (r"\blong[- ]term (goal|objective)\b", 0.35),
        (r"\bshort[- ]term (goal|objective)\b", 0.35),
    ],
}


# ─── Section aliases ──────────────────────────────────────────────────────────

SECTION_ALIASES: Dict[str, List[str]] = {
    "chief_complaint": [
        "chief complaint", "cc", "reason for visit", "presenting complaint",
        "presenting concern", "reason for encounter", "chief complaints",
    ],
    "hpi": [
        "history of present illness", "hpi", "present illness",
        "history of presenting illness", "presenting history",
        "history of presenting complaint", "hpoc",
    ],
    "past_medical_history": [
        "past medical history", "pmh", "medical history", "history",
        "past history", "previous medical history", "medical background",
    ],
    "surgical_history": [
        "past surgical history", "surgical history", "psh",
        "prior surgeries", "past operations", "surgical background",
        "previous surgeries", "operative history",
    ],
    "family_history": [
        "family history", "fhx", "fh", "family medical history",
        "family background", "hereditary history",
    ],
    "social_history": [
        "social history", "shx", "sh", "social background",
        "lifestyle history", "tobacco history", "substance use history",
        "occupational history",
    ],
    "allergies": [
        "allergy", "allergies", "drug allergies", "known allergies",
        "nkda", "nka", "no known allergies", "adverse reactions",
        "drug reactions", "medication allergies",
    ],
    "medications": [
        "medication", "medications", "current medications", "home medications",
        "discharge medications", "prescriptions", "rx", "active medications",
        "medication list", "home meds", "medication reconciliation",
        "current meds", "outpatient medications",
    ],
    "vitals": [
        "vitals", "vital signs", "vital", "measurements",
        "objective vital signs", "triage vitals", "vitals on arrival",
        "vs", "vital signs on admission",
    ],
    "physical_exam": [
        "physical exam", "physical examination", "exam", "examination",
        "objective", "pe", "general examination", "physical findings",
    ],
    "assessment": [
        "assessment", "impression", "diagnosis", "diagnoses", "dx",
        "clinical impression", "assessment and plan", "a/p",
        "clinical diagnosis", "differential diagnosis", "ddx",
    ],
    "plan": [
        "plan", "treatment plan", "recommendations", "recommendation",
        "management plan", "disposition", "orders", "clinical plan",
        "therapeutic plan", "next steps",
    ],
    "labs": [
        "labs", "laboratory", "lab results", "blood work", "test results",
        "laboratory results", "lab data", "laboratory studies",
        "diagnostic studies", "laboratory findings",
    ],
    "imaging": [
        "imaging", "radiology", "xray", "x-ray", "ct", "mri", "ultrasound",
        "imaging studies", "imaging results", "radiological studies",
        "diagnostic imaging", "radiographic findings",
    ],
    "procedure": [
        "procedure", "procedures", "operation", "surgery",
        "intervention", "surgical procedure",
    ],
    "follow_up": [
        "follow up", "follow-up", "return to clinic", "rtc",
        "follow up instructions", "next appointment",
        "return precautions", "patient instructions",
    ],
    "ros": [
        "review of systems", "ros", "systems review",
        "review of systems positive", "review of systems negative",
    ],
    # ── Radiology-specific ──────────────────────────────────────────────
    "technique": [
        "technique", "protocol", "scan parameters", "imaging protocol",
        "acquisition", "scan technique", "technical parameters",
        "procedure description", "contrast protocol",
    ],
    "clinical_history": [
        "clinical history", "clinical information", "reason for exam",
        "clinical indication", "indication", "indications",
        "history and indication", "clinical question",
    ],
    "comparison": [
        "comparison", "prior study", "previous exam", "prior examination",
        "compared to", "comparison study", "prior imaging",
    ],
    "findings": [
        "findings", "observations", "radiological findings",
        "imaging findings", "scan findings", "mri findings", "ct findings",
    ],
    "impression": [
        "impression", "conclusion", "summary", "interpretation",
        "final interpretation", "radiologist impression",
        "diagnostic impression", "overall impression",
    ],
    # ── Operative-specific ──────────────────────────────────────────────
    "preoperative_diagnosis": [
        "preoperative diagnosis", "preop diagnosis", "pre-op dx",
        "pre-operative diagnosis", "diagnosis (pre)",
    ],
    "postoperative_diagnosis": [
        "postoperative diagnosis", "postop diagnosis", "post-op dx",
        "post-operative diagnosis", "diagnosis (post)", "final diagnosis",
    ],
    "operative_findings": [
        "operative findings", "intraoperative findings",
        "findings at surgery", "intraoperative course",
        "surgical findings",
    ],
    "procedure_performed": [
        "procedure performed", "operation performed", "surgery performed",
        "procedures performed", "operations performed",
    ],
    "anesthesia": [
        "anesthesia", "anesthetic", "anesthesia type", "anesthesia plan",
        "type of anesthesia",
    ],
    "complications": [
        "complications", "intraoperative complications", "adverse events",
        "procedure complications", "postoperative complications",
        "complications noted",
    ],
    "specimens": [
        "specimens", "specimen submitted", "pathology specimens",
        "specimen description", "tissue submitted",
    ],
    # ── Discharge-specific ──────────────────────────────────────────────
    "disposition": [
        "disposition", "discharge disposition", "patient disposition",
        "discharged to", "discharge condition",
    ],
    "discharge_instructions": [
        "discharge instructions", "patient instructions", "home instructions",
        "discharge education", "home care instructions",
    ],
    "diet_activity": [
        "diet", "diet and activity", "activity restrictions",
        "weight bearing", "activity orders", "diet orders",
        "activity status",
    ],
    "wound_care": [
        "wound care", "wound management", "dressing", "wound instructions",
        "incision care",
    ],
    # ── Administrative ──────────────────────────────────────────────────
    "code_status": [
        "code status", "full code", "dnr", "dnar", "dnr/dni",
        "advance directives", "advance care planning",
        "resuscitation status",
    ],
    "consent": [
        "consent", "informed consent", "patient consent",
        "consent obtained", "consent for procedure",
    ],
    "reason_for_referral": [
        "reason for referral", "reason for consultation",
        "purpose of referral", "referral reason",
    ],
    # ── Exam subsections ────────────────────────────────────────────────
    "neurological_exam": [
        "neurological exam", "neuro exam", "mental status", "cranial nerves",
        "neurological", "neurologic exam", "cognitive assessment",
        "mental status examination",
    ],
    "cardiovascular_exam": [
        "cardiovascular exam", "cardiac exam", "heart exam",
        "cardiovascular", "cardiac", "heart sounds",
    ],
    "respiratory_exam": [
        "respiratory exam", "pulmonary exam", "lung exam", "breath sounds",
        "respiratory", "pulmonary", "lungs",
    ],
    "musculoskeletal_exam": [
        "musculoskeletal", "extremities", "joints", "range of motion",
        "msk exam", "orthopedic exam",
    ],
    "psychiatric_exam": [
        "psychiatric exam", "mental status exam", "mood", "affect",
        "psychiatric evaluation", "behavioral assessment",
    ],
    "skin_exam": [
        "skin", "dermatological", "integument", "dermatology exam",
        "skin examination",
    ],
    "head_neck_exam": [
        "head and neck", "heent", "eyes", "ears", "nose", "throat",
        "head neck", "ent exam",
    ],
    "genitourinary_exam": [
        "genitourinary", "gu exam", "rectal", "pelvic", "abdominal exam",
        "abdomen",
    ],
    # ── Results & special ───────────────────────────────────────────────
    "pathology_results": [
        "pathology", "pathological diagnosis", "gross description",
        "microscopic", "histology", "cytology",
    ],
    "microbiology": [
        "microbiology", "cultures", "sensitivity", "sensitivities",
        "culture results", "culture data", "culture and sensitivity",
    ],
    "patient_education": [
        "patient education", "teaching provided", "patient counseled",
        "patient teaching", "education",
    ],
    "rehab_goals": [
        "rehab goals", "rehabilitation goals", "functional goals",
        "goals of rehabilitation",
    ],
    "functional_assessment": [
        "functional status", "functional assessment", "adl",
        "activities of daily living", "functional capacity",
    ],
    "past_history": [
        "past history", "past medical and surgical history",
        "medical and surgical history", "pmsh",
    ],
}


# ─── Vital patterns ───────────────────────────────────────────────────────────

VITAL_PATTERNS: Dict[str, str] = {
    "blood_pressure": r"\b(?:bp|blood pressure)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b",
    "heart_rate": r"\b(?:hr|heart rate|pulse|p)\s*[:\-]?\s*(\d{2,3})\s*(?:bpm)?\b",
    "temperature": r"\b(?:temp(?:erature)?|t)\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:f|c|°f|°c|fahrenheit|celsius)?\b",
    "respiratory_rate": r"\b(?:rr|respiratory rate|resp(?:iration)? rate|respirations?)\s*[:\-]?\s*(\d{1,3})\b",
    "oxygen_saturation": r"\b(?:spo2|o2 sat(?:uration)?|oxygen saturation|sat)\s*[:\-]?\s*(\d{2,3})\s*%?\b",
    "weight": r"\b(?:wt|weight)\s*[:\-]?\s*(\d{1,4}(?:\.\d+)?)\s*(?:kg|kgs|lbs?|pounds?)\b",
    "height": r"\b(?:ht|height)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:cm|m|meters?|ft|feet|inches?|in|'|\")",
    "bmi": r"\b(?:bmi|body mass index)\s*[:\-]?\s*(\d+(?:\.\d+)?)\b",
    "pain_score": r"\b(?:pain(?: score| level| scale)?|vas|nrs|numeric pain)\s*[:\-]?\s*(\d+)\s*(?:/\s*10)?\b",
    "gcs": r"\b(?:gcs|glasgow coma scale)\s*[:\-]?\s*(\d+)\b",
    "blood_glucose": r"\b(?:glucose|fbs|rbs|ppbs|blood sugar|cbg|fingerstick)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl|mmol/l)?\b",
    "inr": r"\b(?:inr|(?:pt)(?:/inr)?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\b",
}


# ─── Lab names ────────────────────────────────────────────────────────────────

COMMON_LAB_NAMES: List[str] = [
    # CBC
    "hb", "hgb", "hemoglobin", "hct", "hematocrit", "wbc", "rbc", "platelet", "platelets",
    "neutrophils", "neutrophil", "lymphocytes", "lymphocyte", "monocytes", "monocyte",
    "eosinophils", "eosinophil", "basophils", "basophil", "bands", "reticulocytes",
    "mcv", "mch", "mchc", "rdw",
    # BMP/CMP
    "sodium", "na", "potassium", "k", "chloride", "cl", "bicarbonate", "co2", "hco3",
    "bun", "blood urea nitrogen", "creatinine", "calcium", "magnesium", "phosphorus",
    "glucose", "albumin", "protein", "total protein",
    # Liver
    "alt", "sgpt", "ast", "sgot", "alk phos", "alkaline phosphatase", "ggt", "ldh",
    "bilirubin", "total bilirubin", "direct bilirubin", "indirect bilirubin",
    # Lipids
    "cholesterol", "total cholesterol", "triglycerides", "hdl", "ldl", "vldl",
    "non-hdl", "lipoprotein",
    # Thyroid
    "tsh", "t3", "t4", "free t3", "free t4", "ft3", "ft4",
    # Coagulation
    "pt", "ptt", "aptt", "inr", "fibrinogen", "d-dimer", "anti-xa",
    # Cardiac
    "troponin", "troponin i", "troponin t", "bnp", "nt-probnp", "ck", "creatine kinase",
    "ckmb", "ck-mb", "myoglobin", "lactic acid", "lactate",
    # Inflammatory/infection
    "crp", "c-reactive protein", "esr", "procalcitonin", "ferritin", "il-6",
    # Renal
    "gfr", "egfr", "cystatin c", "uric acid", "24h creatinine",
    # Hormones
    "hba1c", "a1c", "insulin", "c-peptide", "cortisol", "acth", "prolactin",
    "fsh", "lh", "estradiol", "testosterone", "progesterone", "hcg", "beta-hcg",
    "igf-1", "dhea",
    # Vitamins/minerals
    "vitamin d", "25-oh vitamin d", "vitamin b12", "b12", "folate", "folic acid",
    "iron", "serum iron", "tibc", "transferrin", "zinc", "copper",
    # Tumor markers
    "psa", "cea", "ca-125", "ca 125", "ca-19-9", "ca 19-9", "afp", "alpha fetoprotein",
    "ca-15-3", "her2",
    # Immunology
    "ana", "anca", "anti-dsdna", "rf", "rheumatoid factor", "anti-ccp",
    "c3", "c4", "complement", "antiphospholipid", "anti-ro", "anti-la",
    # Infectious disease
    "hiv", "hbsag", "hbs ag", "hcv", "hcv ab", "monospot", "cmv", "ebv",
    # ABG
    "ph", "pco2", "po2", "base excess", "pao2", "fio2",
    # Urinalysis
    "ua", "urinalysis", "urine protein", "urine creatinine", "microalbumin",
    "urine culture", "urine glucose", "urine ketones",
    # Microbiology
    "blood culture", "sputum culture", "wound culture", "csf",
    # Misc
    "ammonia", "lead", "ethanol", "drug screen", "urine drug screen",
]


# ─── Drug names ───────────────────────────────────────────────────────────────

COMMON_DRUG_NAMES: List[str] = [
    # Cardiovascular
    "metoprolol", "atenolol", "carvedilol", "bisoprolol", "propranolol", "labetalol",
    "lisinopril", "enalapril", "ramipril", "captopril", "benazepril",
    "losartan", "valsartan", "olmesartan", "irbesartan", "candesartan", "telmisartan",
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine",
    "furosemide", "hydrochlorothiazide", "chlorthalidone", "spironolactone",
    "eplerenone", "torsemide", "indapamide",
    "digoxin", "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
    "clopidogrel", "ticagrelor", "prasugrel", "aspirin",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
    "pitavastatin", "ezetimibe", "fenofibrate", "gemfibrozil",
    "nitroglycerin", "isosorbide mononitrate", "isosorbide dinitrate",
    "hydralazine", "clonidine", "doxazosin", "terazosin", "prazosin",
    "amiodarone", "sotalol", "flecainide", "dronedarone", "ivabradine",
    "sacubitril", "sacubitril-valsartan", "entresto",
    "ranolazine", "milrinone", "dobutamine", "dopamine", "norepinephrine",
    # Diabetes
    "metformin", "glipizide", "glimepiride", "glyburide", "glibenclamide",
    "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    "liraglutide", "semaglutide", "dulaglutide", "exenatide", "albiglutide",
    "pioglitazone", "rosiglitazone",
    "insulin glargine", "insulin detemir", "insulin degludec",
    "insulin aspart", "insulin lispro", "insulin glulisine",
    "insulin regular", "nph insulin",
    "acarbose", "miglitol",
    # Pain / Inflammation / Rheum
    "ibuprofen", "naproxen", "naproxen sodium", "celecoxib", "meloxicam",
    "ketorolac", "indomethacin", "diclofenac", "piroxicam", "etodolac",
    "acetaminophen", "paracetamol",
    "tramadol", "oxycodone", "hydrocodone", "morphine", "fentanyl",
    "buprenorphine", "methadone", "codeine", "hydromorphone", "oxymorphone",
    "naloxone", "naltrexone",
    "gabapentin", "pregabalin",
    "cyclobenzaprine", "methocarbamol", "baclofen", "tizanidine", "carisoprodol",
    "colchicine", "allopurinol", "febuxostat", "probenecid",
    "hydroxychloroquine", "methotrexate", "sulfasalazine", "leflunomide",
    "adalimumab", "etanercept", "infliximab", "certolizumab", "golimumab",
    "abatacept", "tocilizumab", "rituximab", "belimumab",
    "tofacitinib", "baricitinib", "upadacitinib",
    # Antibiotics
    "amoxicillin", "ampicillin", "amoxicillin-clavulanate", "ampicillin-sulbactam",
    "piperacillin-tazobactam", "ticarcillin-clavulanate",
    "cefazolin", "cephalexin", "cefadroxil", "cefdinir", "cefuroxime",
    "ceftriaxone", "cefotaxime", "ceftazidime", "cefepime", "ceftaroline",
    "aztreonam", "meropenem", "imipenem", "ertapenem", "doripenem",
    "azithromycin", "clarithromycin", "erythromycin",
    "doxycycline", "minocycline", "tetracycline",
    "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "trimethoprim-sulfamethoxazole", "trimethoprim",
    "metronidazole", "clindamycin", "linezolid", "vancomycin",
    "daptomycin", "colistin", "polymyxin",
    "nitrofurantoin", "fosfomycin",
    "fluconazole", "itraconazole", "voriconazole", "posaconazole",
    "micafungin", "caspofungin", "amphotericin b",
    "acyclovir", "valacyclovir", "famciclovir",
    "ganciclovir", "valganciclovir", "oseltamivir", "zanamivir",
    # GI
    "omeprazole", "pantoprazole", "lansoprazole", "esomeprazole", "rabeprazole",
    "ranitidine", "famotidine", "cimetidine",
    "metoclopramide", "ondansetron", "granisetron", "prochlorperazine",
    "promethazine", "droperidol",
    "loperamide", "bismuth", "psyllium", "methylcellulose",
    "docusate", "polyethylene glycol", "lactulose", "bisacodyl", "senna",
    "lubiprostone", "linaclotide", "plecanatide",
    "mesalamine", "balsalazide", "olsalazine",
    "budesonide", "prednisone",  # GI use
    "infliximab", "adalimumab",  # IBD - duplicate but fine for matching
    "vedolizumab", "ustekinumab",
    "ursodiol",
    # Respiratory
    "albuterol", "levalbuterol", "ipratropium", "tiotropium", "umeclidinium",
    "salmeterol", "formoterol", "olodaterol",
    "fluticasone", "budesonide", "beclomethasone", "ciclesonide", "mometasone",
    "fluticasone-salmeterol", "budesonide-formoterol",
    "montelukast", "zafirlukast", "zileuton",
    "theophylline", "aminophylline",
    "guaifenesin", "dextromethorphan", "benzonatate", "codeine",
    "roflumilast",
    "dupilumab", "mepolizumab", "benralizumab",
    # Psychiatry / Neurology
    "sertraline", "fluoxetine", "escitalopram", "citalopram", "paroxetine",
    "fluvoxamine",
    "venlafaxine", "desvenlafaxine", "duloxetine", "levomilnacipran",
    "bupropion", "mirtazapine", "trazodone", "nefazodone",
    "amitriptyline", "nortriptyline", "imipramine", "clomipramine",
    "lorazepam", "alprazolam", "clonazepam", "diazepam", "temazepam",
    "zolpidem", "eszopiclone", "zaleplon", "ramelteon", "suvorexant",
    "quetiapine", "olanzapine", "risperidone", "aripiprazole", "ziprasidone",
    "lurasidone", "asenapine", "brexpiprazole", "cariprazine", "clozapine",
    "haloperidol", "fluphenazine", "perphenazine",
    "lithium", "valproate", "valproic acid", "divalproex",
    "lamotrigine", "carbamazepine", "oxcarbazepine",
    "levetiracetam", "phenytoin", "fosphenytoin", "topiramate", "zonisamide",
    "lacosamide", "perampanel", "brivaracetam",
    "donepezil", "rivastigmine", "galantamine", "memantine",
    "methylphenidate", "amphetamine", "amphetamine salts", "lisdexamfetamine",
    "atomoxetine", "guanfacine", "clonidine",
    "naltrexone", "acamprosate", "disulfiram",
    "sumatriptan", "rizatriptan", "zolmitriptan", "eletriptan",
    # Endocrine
    "levothyroxine", "liothyronine", "methimazole", "propylthiouracil",
    "prednisone", "prednisolone", "dexamethasone", "methylprednisolone",
    "hydrocortisone", "fludrocortisone", "triamcinolone",
    "testosterone", "estradiol", "progesterone", "medroxyprogesterone",
    "norethindrone", "levonorgestrel", "ethinyl estradiol",
    "raloxifene", "tamoxifen", "letrozole", "anastrozole", "exemestane",
    # Urological
    "tamsulosin", "terazosin", "finasteride", "dutasteride",
    "oxybutynin", "tolterodine", "solifenacin", "darifenacin",
    "sildenafil", "tadalafil", "vardenafil",
    "mirabegron", "bethanechol",
    # Bone / Osteoporosis
    "alendronate", "risedronate", "ibandronate", "zoledronic acid",
    "denosumab", "teriparatide", "abaloparatide", "romosozumab",
    "calcium", "calcium carbonate", "calcium citrate",
    # Hematology
    "ferrous sulfate", "ferrous gluconate", "ferric carboxymaltose",
    "epoetin alfa", "darbepoetin alfa", "filgrastim", "pegfilgrastim",
    "enoxaparin", "heparin", "fondaparinux", "dalteparin",
    "alteplase", "tenecteplase", "streptokinase",
    "hydroxyurea", "deferasirox",
    # Vitamins / Supplements
    "vitamin d3", "cholecalciferol", "vitamin b12", "cyanocobalamin",
    "folic acid", "folate", "pyridoxine", "thiamine",
    "magnesium oxide", "magnesium", "zinc sulfate", "iron",
    # Immunosuppressants / Transplant
    "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine",
    "sirolimus", "everolimus", "belatacept",
    # Oncology (common)
    "carboplatin", "cisplatin", "oxaliplatin", "paclitaxel", "docetaxel",
    "cyclophosphamide", "doxorubicin", "gemcitabine", "pemetrexed",
    "bevacizumab", "cetuximab", "trastuzumab", "pertuzumab",
    "pembrolizumab", "nivolumab", "atezolizumab", "ipilimumab",
    "imatinib", "erlotinib", "gefitinib", "osimertinib",
    "ibuprofen",  # may appear again in cancer pain
]


# ─── Drug suffix patterns (class-level detection) ─────────────────────────────

DRUG_SUFFIX_PATTERNS: List[Tuple[str, str]] = [
    (r"[a-z]{4,}mab\b", "monoclonal antibody"),
    (r"[a-z]{4,}statin\b", "statin"),
    (r"[a-z]{4,}pril\b", "ACE inhibitor"),
    (r"[a-z]{4,}sartan\b", "ARB"),
    (r"[a-z]{4,}olol\b", "beta blocker"),
    (r"[a-z]{4,}dipine\b", "CCB-dihydropyridine"),
    (r"[a-z]{4,}floxacin\b", "fluoroquinolone"),
    (r"[a-z]{4,}cillin\b", "penicillin"),
    (r"[a-z]{4,}mycin\b", "macrolide/aminoglycoside"),
    (r"[a-z]{4,}cycline\b", "tetracycline"),
    (r"[a-z]{4,}azole\b", "azole antifungal/PPI"),
    (r"[a-z]{4,}gliptin\b", "DPP-4 inhibitor"),
    (r"[a-z]{4,}gliflozin\b", "SGLT2 inhibitor"),
    (r"[a-z]{4,}glutide\b", "GLP-1 agonist"),
    (r"[a-z]{4,}lukast\b", "leukotriene modifier"),
    (r"[a-z]{4,}tidine\b", "H2 blocker"),
    (r"[a-z]{4,}prazole\b", "proton pump inhibitor"),
    (r"[a-z]{4,}setron\b", "5-HT3 antagonist antiemetic"),
    (r"[a-z]{4,}tropin\b", "pituitary hormone"),
    (r"[a-z]{4,}parin\b", "heparin / LMWH"),
    (r"[a-z]{4,}dronate\b", "bisphosphonate"),
    (r"[a-z]{4,}vir\b", "antiviral"),
    (r"[a-z]{4,}fungin\b", "echinocandin antifungal"),
]


# ─── Medical condition terms ──────────────────────────────────────────────────

COMMON_CONDITION_TERMS: List[str] = [
    "hypertension", "htn", "high blood pressure",
    "diabetes", "diabetes mellitus", "type 2 diabetes", "type 1 diabetes", "t2dm", "t1dm",
    "hyperlipidemia", "hypercholesterolemia", "dyslipidemia",
    "coronary artery disease", "cad", "ischemic heart disease", "ihd",
    "congestive heart failure", "chf", "heart failure", "hfref", "hfpef",
    "atrial fibrillation", "afib", "a-fib", "af",
    "myocardial infarction", "mi", "heart attack", "stemi", "nstemi",
    "stroke", "cva", "cerebrovascular accident",
    "tia", "transient ischemic attack",
    "peripheral vascular disease", "pvd", "peripheral artery disease", "pad",
    "deep vein thrombosis", "dvt",
    "pulmonary embolism", "pe",
    "chronic kidney disease", "ckd", "renal failure", "renal insufficiency",
    "acute kidney injury", "aki",
    "urinary tract infection", "uti",
    "pneumonia",
    "copd", "chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis",
    "asthma",
    "sleep apnea", "obstructive sleep apnea", "osa",
    "anemia", "iron deficiency anemia",
    "thrombocytopenia",
    "hypothyroidism",
    "hyperthyroidism", "graves disease",
    "osteoporosis", "osteopenia",
    "osteoarthritis", "oa",
    "rheumatoid arthritis", "ra",
    "gout", "hyperuricemia",
    "fibromyalgia",
    "depression", "major depressive disorder", "mdd",
    "anxiety", "generalized anxiety disorder", "gad",
    "bipolar disorder", "bipolar",
    "schizophrenia",
    "ptsd", "post-traumatic stress disorder",
    "adhd", "attention deficit",
    "dementia", "cognitive impairment",
    "alzheimer", "alzheimer's disease",
    "parkinson", "parkinson's disease",
    "epilepsy", "seizure disorder",
    "migraine",
    "multiple sclerosis", "ms",
    "cellulitis",
    "sepsis", "septicemia", "bacteremia",
    "cancer", "carcinoma", "malignancy", "tumor", "malignant",
    "lymphoma", "leukemia", "myeloma",
    "breast cancer", "lung cancer", "colon cancer", "colorectal cancer",
    "prostate cancer", "ovarian cancer", "cervical cancer",
    "hepatocellular carcinoma", "hcc",
    "gerd", "gastroesophageal reflux", "acid reflux",
    "peptic ulcer", "gastric ulcer", "duodenal ulcer",
    "crohn's disease", "crohn disease", "ulcerative colitis", "inflammatory bowel disease",
    "irritable bowel syndrome", "ibs",
    "pancreatitis",
    "hepatitis", "hepatitis b", "hepatitis c", "cirrhosis", "liver failure",
    "cholelithiasis", "gallstones", "cholecystitis",
    "appendicitis",
    "diverticulitis", "diverticulosis",
    "obesity", "morbid obesity", "overweight",
    "chronic pain",
    "neuropathy", "peripheral neuropathy", "diabetic neuropathy",
    "radiculopathy",
    "herniated disc", "disc herniation", "disc bulge",
    "spinal stenosis",
    "scoliosis",
    "fracture",
    "dislocation",
    "rotator cuff", "rotator cuff tear",
    "acl tear", "meniscal tear",
    "cataracts",
    "glaucoma",
    "macular degeneration",
    "retinopathy", "diabetic retinopathy",
    "hypertriglyceridemia",
    "metabolic syndrome",
    "fatty liver", "nafld", "nash",
    "hiv", "aids",
    "covid", "covid-19",
    "influenza", "flu",
]


# ─── Negation patterns ────────────────────────────────────────────────────────

NEGATION_TERMS: List[str] = [
    r"\bno\b", r"\bnot\b", r"\bnone\b",
    r"\bdenies\b", r"\bdenied\b",
    r"\bnegative for\b", r"\bnegative\b",
    r"\bwithout\b", r"\bw/o\b",
    r"\bno evidence of\b", r"\bno sign of\b", r"\bno signs of\b",
    r"\bruled out\b", r"\bexcluded\b",
    r"\bdiscontinued\b", r"\bdiscontinue\b",
    r"\bstopped\b", r"\bstop\b",
    r"\bno longer\b",
    r"\bintolerant to\b", r"\bintolerance to\b",
    r"\bcontraindicated\b",
    r"\ballergic to\b",
    r"\bnkda\b", r"\bnka\b",
    r"\babsent\b", r"\bnormal\b",
    r"\bclear\b", r"\bclear of\b",
    r"\bfree of\b", r"\bfree from\b",
]


# ─── Provider credentials ─────────────────────────────────────────────────────

PROVIDER_LABEL_PATTERNS: List[str] = [
    r"\b(?:attending|attending physician)\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:ordering provider|ordering physician)\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:referring provider|referring physician)\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:signed by|author|provider|consultant|surgeon|radiologist|interpreted by)\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
    r"\bDr\.?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
    r"\bDoctor\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
]
PROVIDER_CREDENTIAL_RE = re.compile(
    r"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
    r",?\s*(?:MD|DO|NP|PA|PA-C|APRN|CRNA|LCSW|RN|LPN|RPh|PharmD|DPM|OD|DDS|PhD|PsyD|CNM)\b"
)


# ─── Main extractor ───────────────────────────────────────────────────────────

class ClinicalMetadataExtractor:
    """
    Clinical metadata extractor v2.

    Input:  layout_elements*.parquet + layout_manifest.json
    Output: clinical_document_metadata.json, clinical_sections.parquet,
            clinical_entities.parquet, clinical_element_metadata.parquet
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
        fallback  = self.document_dir / "layout_elements.parquet"
        if preferred.exists():
            self.elements_path = preferred
        elif fallback.exists():
            self.elements_path = fallback
        else:
            raise FileNotFoundError(f"No layout elements file in {self.document_dir}")

        self.manifest: Dict[str, Any] = {}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        # Pre-build sets for O(1) lookups
        self._drug_set = {d.lower() for d in COMMON_DRUG_NAMES}
        self._lab_set  = {l.lower() for l in COMMON_LAB_NAMES}
        self._cond_set = {c.lower() for c in COMMON_CONDITION_TERMS}

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        elements      = read_parquet(self.elements_path)
        line_elements = self._line_elements(elements)

        full_text        = "\n".join(e.get("text") or e.get("text_preview") or "" for e in line_elements)
        first_pages_text = self._first_pages_text(line_elements, max_pages=3)

        document_metadata    = self.extract_document_metadata(full_text, first_pages_text, line_elements)
        section_rows         = self.extract_sections(line_elements)
        entity_rows          = self.extract_entities(line_elements)
        element_metadata_rows= self.extract_element_metadata(line_elements)

        output = {
            "document_metadata":       document_metadata,
            "sections_count":          len(section_rows),
            "entities_count":          len(entity_rows),
            "element_metadata_count":  len(element_metadata_rows),
            "created_at":              utc_now_iso(),
            "input_elements_path":     str(self.elements_path),
        }

        (self.document_dir / "clinical_document_metadata.json").write_text(
            json.dumps(document_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_parquet(section_rows,          self.document_dir / "clinical_sections.parquet")
        write_parquet(entity_rows,           self.document_dir / "clinical_entities.parquet")
        write_parquet(element_metadata_rows, self.document_dir / "clinical_element_metadata.parquet")

        print("Clinical metadata extraction complete.")
        print(f"  Document type : {document_metadata.get('document_type')}")
        print(f"  Sections      : {len(section_rows)}")
        print(f"  Entities      : {len(entity_rows)}")
        return output

    # ── Document metadata ─────────────────────────────────────────────────────

    def extract_document_metadata(
        self,
        full_text: str,
        first_pages_text: str,
        line_elements: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        doc_type       = self.classify_document_type(first_pages_text + "\n" + full_text[:4000])
        dates          = self.extract_dates_from_text(full_text)
        patient_ids    = self.extract_patient_ids(full_text)
        provider_names = self.extract_provider_names(first_pages_text)
        facility_names = self.extract_facility_names(first_pages_text)

        return {
            "document_id":          self.manifest.get("document_id"),
            "source_pdf_name":      self.manifest.get("source_pdf_name"),
            "source_pdf_path":      self.manifest.get("source_pdf_path"),
            "source_sha256":        self.manifest.get("source_sha256"),
            "total_pages":          self.manifest.get("total_pages"),
            "total_tokens":         self.manifest.get("total_tokens"),

            "document_type":              doc_type["label"],
            "document_type_confidence":   doc_type["confidence"],
            "document_type_evidence":     doc_type["evidence"],
            "document_type_scores_json":  safe_json_dumps(doc_type["scores"]),

            "dates_json":          safe_json_dumps(dates),
            "patient_ids_json":    safe_json_dumps(patient_ids),
            "provider_names_json": safe_json_dumps(provider_names),
            "facility_names_json": safe_json_dumps(facility_names),

            "clinical_metadata_extractor_version": _VERSION,
            "created_at": utc_now_iso(),
        }

    def classify_document_type(self, text: str) -> Dict[str, Any]:
        text_l = lower_clean(text)
        scores: Dict[str, Dict[str, Any]] = {}

        for label, rules in DOCUMENT_TYPE_RULES.items():
            total_score = 0.0
            evidence: List[str] = []
            for pattern, weight in rules:
                if re.search(pattern, text_l, flags=re.I):
                    total_score += weight
                    evidence.append(pattern)
            scores[label] = {
                "score":    round(min(total_score, 1.0), 4),
                "evidence": evidence[:6],
            }

        best_label    = "Unknown"
        best_score    = 0.0
        best_evidence: List[str] = []

        for label, obj in scores.items():
            if obj["score"] > best_score:
                best_label    = label
                best_score    = obj["score"]
                best_evidence = obj["evidence"]

        if best_score == 0:
            return {"label": "Unknown", "confidence": 0.0, "evidence": [], "scores": scores}

        confidence = round(min(0.97, 0.50 + best_score), 4)
        return {
            "label":      best_label,
            "confidence": confidence,
            "evidence":   best_evidence,
            "scores":     scores,
        }

    # ── Sections ──────────────────────────────────────────────────────────────

    def extract_sections(self, line_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        current_section           = None
        current_start_page        = None
        current_start_element_id  = None
        current_text_parts: List[str] = []

        for e in line_elements:
            text = normalize_space(e.get("text") or e.get("text_preview") or "")
            if not text:
                continue

            section_key, confidence = self.detect_section(text, e)

            if section_key:
                if current_section is not None:
                    rows.append({
                        "document_id":       self.manifest.get("document_id"),
                        "section_key":       current_section,
                        "section_name":      current_section,
                        "start_page":        current_start_page,
                        "end_page":          line_page(e),
                        "start_element_id":  current_start_element_id,
                        "text_preview":      normalize_space(" ".join(current_text_parts))[:1200],
                        "confidence":        0.80,
                        "source":            "section_rules_v2",
                    })
                current_section           = section_key
                current_start_page        = line_page(e)
                current_start_element_id  = e.get("element_id")
                current_text_parts        = [text]
            elif current_section is not None:
                current_text_parts.append(text)

        if current_section is not None:
            rows.append({
                "document_id":      self.manifest.get("document_id"),
                "section_key":      current_section,
                "section_name":     current_section,
                "start_page":       current_start_page,
                "end_page":         line_page(line_elements[-1]) if line_elements else None,
                "start_element_id": current_start_element_id,
                "text_preview":     normalize_space(" ".join(current_text_parts))[:1200],
                "confidence":       0.80,
                "source":           "section_rules_v2",
            })
        return rows

    def detect_section(
        self,
        text: str,
        element: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], float]:
        norm = normalize_heading(text)
        if not norm or len(norm) > 100:
            return None, 0.0

        is_heading = bool(element and element.get("is_heading_candidate"))
        looks_like_header = (
            text.strip().endswith(":")
            or text.isupper()
            or is_heading
            or (len(norm.split()) <= 6 and text.strip()[-1:] in (":", "–", "—"))
        )

        for section_key, aliases in SECTION_ALIASES.items():
            for alias in aliases:
                alias_norm = normalize_heading(alias)

                # Exact match
                if norm == alias_norm:
                    return section_key, 0.97

                # Heading candidate contains alias
                if looks_like_header and alias_norm in norm and len(norm) <= 90:
                    return section_key, 0.88

                # Pattern: "Assessment:" or "Plan —"
                if re.match(rf"^{re.escape(alias_norm)}\s*[:\-–—]?$", norm):
                    return section_key, 0.93

        # Fuzzy fallback: Jaccard similarity for heading-like text
        if looks_like_header and len(norm.split()) <= 8:
            best_key  = None
            best_jacc = 0.0
            for section_key, aliases in SECTION_ALIASES.items():
                for alias in aliases:
                    j = _jaccard(norm, normalize_heading(alias))
                    if j > best_jacc:
                        best_jacc = j
                        best_key  = section_key
            if best_jacc >= 0.45:
                return best_key, round(0.65 + best_jacc * 0.15, 3)

        return None, 0.0

    # ── Entities ──────────────────────────────────────────────────────────────

    def extract_entities(self, line_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []

        for e in line_elements:
            text = normalize_space(e.get("text") or e.get("text_preview") or "")
            if not text:
                continue

            page         = line_page(e)
            element_id   = e.get("element_id")
            section_name = self._element_section_name(e)

            entities.extend(self.extract_date_entities(text, page, element_id, section_name))
            entities.extend(self.extract_patient_id_entities(text, page, element_id, section_name))
            entities.extend(self.extract_vital_entities(text, page, element_id, section_name))
            entities.extend(self.extract_lab_entities(text, page, element_id, section_name))
            entities.extend(self.extract_medication_entities(text, page, element_id, section_name))
            entities.extend(self.extract_diagnosis_entities(text, page, element_id, section_name))
            entities.extend(self.extract_imaging_entities(text, page, element_id, section_name))
            entities.extend(self.extract_procedure_entities(text, page, element_id, section_name))
            entities.extend(self.extract_allergy_entities(text, page, element_id, section_name))
            entities.extend(self.extract_demographics_entities(text, page, element_id, section_name))
            entities.extend(self.extract_social_history_entities(text, page, element_id, section_name))

        entities = self._dedupe_entities(entities)
        return [x for x in entities if float(x.get("confidence") or 0) >= self.min_entity_confidence]

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
            "document_id":     self.manifest.get("document_id"),
            "entity_type":     entity_type,
            "text":            text,
            "normalized_value":normalized_value,
            "page_number":     page_number,
            "element_id":      element_id,
            "section_name":    section_name,
            "confidence":      round(float(confidence), 4),
            "source":          source,
            "is_negated":      self._is_negated_scoped(evidence, text),
            "evidence":        evidence[:1200],
            "attributes_json": safe_json_dumps(attributes or {}),
            "created_at":      utc_now_iso(),
        }

    # ── Date entities ─────────────────────────────────────────────────────────

    def extract_date_entities(self, text, page, element_id, section_name):
        rows = []
        for d in self.extract_dates_from_text(text):
            rows.append(self._base_entity(
                entity_type="date", text=d["raw"], normalized_value=d["normalized"],
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=d["confidence"], source="date_regex_v2", evidence=text, attributes=d,
            ))
        return rows

    # ── Patient ID entities ───────────────────────────────────────────────────

    def extract_patient_id_entities(self, text, page, element_id, section_name):
        rows = []
        for obj in self.extract_patient_ids(text):
            rows.append(self._base_entity(
                entity_type="patient_identifier", text=obj["raw"], normalized_value=obj["value"],
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=obj["confidence"], source="patient_id_regex_v2", evidence=text, attributes=obj,
            ))
        return rows

    # ── Vital entities ────────────────────────────────────────────────────────

    def extract_vital_entities(self, text, page, element_id, section_name):
        rows = []
        for vital_name, pattern in VITAL_PATTERNS.items():
            for m in re.finditer(pattern, text, flags=re.I):
                rows.append(self._base_entity(
                    entity_type="vital", text=m.group(0), normalized_value=m.group(0),
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=0.85, source="vital_regex_v2", evidence=text,
                    attributes={"vital_type": vital_name, "groups": list(m.groups())},
                ))
        return rows

    # ── Lab entities ──────────────────────────────────────────────────────────

    def extract_lab_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        text_l = lower_clean(text)

        # Pattern: lab_name followed by optional separator and numeric value + optional unit
        lab_name_pattern = "|".join(
            re.escape(x) for x in sorted(COMMON_LAB_NAMES, key=len, reverse=True)
        )
        pattern = (
            rf"\b({lab_name_pattern})\b"
            r"\s*[:\-=]?\s*"
            r"([Hh]igh|[Ll]ow|[Hh]|[Ll])?\s*"          # abnormal flag
            r"(\d+(?:\.\d+)?)"                            # numeric value
            r"\s*([a-zA-Z/%µμ]+(?:/[a-zA-Z0-9]+)*)?"    # unit
        )

        seen_spans: set = set()
        for m in re.finditer(pattern, text_l, flags=re.I):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            lab_name  = m.group(1)
            flag      = m.group(2) or ""
            value     = m.group(3)
            unit      = m.group(4) or ""
            is_abnormal = bool(flag and flag.lower() in ("h", "l", "high", "low"))

            rows.append(self._base_entity(
                entity_type="lab_result", text=m.group(0), normalized_value=m.group(0),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.82, source="lab_regex_v2", evidence=text,
                attributes={
                    "lab_name": lab_name,
                    "value": value,
                    "unit": unit,
                    "abnormal_flag": flag,
                    "is_abnormal": is_abnormal,
                },
            ))

        # Reference range pattern: captures "X.X – Y.Y" near lab name
        ref_pattern = rf"\b({lab_name_pattern})\b.*?(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)"
        for m in re.finditer(ref_pattern, text_l, flags=re.I):
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            rows.append(self._base_entity(
                entity_type="lab_reference_range", text=m.group(0), normalized_value=m.group(0),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.72, source="lab_refrange_regex", evidence=text,
                attributes={"lab_name": m.group(1), "low": m.group(2), "high": m.group(3)},
            ))
        return rows

    # ── Medication entities ───────────────────────────────────────────────────

    def extract_medication_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        text_l = lower_clean(text)
        section_boost = 0.15 if section_name in ("medications", "plan", "discharge_instructions") else 0.0

        # 1. Known drug name dictionary match (word-boundary)
        for drug in sorted(self._drug_set, key=len, reverse=True):
            if re.search(rf"\b{re.escape(drug)}\b", text_l):
                # Try to capture dose if nearby
                dose_m = re.search(
                    rf"\b{re.escape(drug)}\b\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b",
                    text_l, flags=re.I,
                )
                rows.append(self._base_entity(
                    entity_type="medication",
                    text=drug,
                    normalized_value=drug,
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=min(0.92, 0.78 + section_boost),
                    source="drug_dict",
                    evidence=text,
                    attributes={
                        "dose": dose_m.group(1) if dose_m else None,
                        "unit": dose_m.group(2) if dose_m else None,
                    },
                ))
                break  # only first match per element to avoid explosion

        # 2. Drug class suffix patterns
        for suffix_pattern, drug_class in DRUG_SUFFIX_PATTERNS:
            for m in re.finditer(suffix_pattern, text_l):
                rows.append(self._base_entity(
                    entity_type="medication_class",
                    text=m.group(0), normalized_value=m.group(0),
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=min(0.80, 0.65 + section_boost),
                    source="drug_suffix",
                    evidence=text,
                    attributes={"drug_class": drug_class},
                ))

        # 3. "Name + dose" structural pattern (catches unlisted drugs)
        dose_pattern = (
            r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,2})"
            r"\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\b"
        )
        for m in re.finditer(dose_pattern, text):
            rows.append(self._base_entity(
                entity_type="medication_candidate",
                text=m.group(0), normalized_value=m.group(0),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=min(0.78, 0.60 + section_boost),
                source="medication_dose_regex",
                evidence=text,
                attributes={"name_candidate": m.group(1), "dose": m.group(2), "unit": m.group(3)},
            ))

        # 4. Contextual medication line (in meds section with route/frequency cues)
        route_freq_words = {"tablet", "tab", "capsule", "cap", "injection", "inj", "syrup",
                            "mg", "mcg", "daily", "bid", "tid", "qid", "prn", "po", "iv", "im",
                            "once daily", "twice daily", "sc", "subcutaneous", "inhale", "apply",
                            "patch", "drops", "sublingual", "transdermal", "topical"}
        if section_name in ("medications", "plan") or any(w in text_l for w in route_freq_words):
            if re.search(r"\b(take|tab|tablet|cap|capsule|inj|injection|apply|instil|inhale)\b", text_l):
                rows.append(self._base_entity(
                    entity_type="medication_line_candidate",
                    text=text[:220], normalized_value=text[:220],
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=min(0.72, 0.58 + section_boost),
                    source="medication_line_rule",
                    evidence=text, attributes={},
                ))
        return rows

    # ── Diagnosis entities ────────────────────────────────────────────────────

    def extract_diagnosis_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        text_l = lower_clean(text)
        section_boost = 0.15 if section_name in (
            "assessment", "chief_complaint", "hpi", "impression",
            "postoperative_diagnosis", "findings",
        ) else 0.0

        # 1. ICD-10 code pattern
        for m in re.finditer(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", text):
            rows.append(self._base_entity(
                entity_type="diagnosis_code",
                text=m.group(0), normalized_value=m.group(0).upper(),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.75, source="icd10_regex",
                evidence=text, attributes={"code": m.group(0).upper()},
            ))

        # 2. Known condition dictionary
        for cond in sorted(self._cond_set, key=len, reverse=True):
            if re.search(rf"\b{re.escape(cond)}\b", text_l):
                rows.append(self._base_entity(
                    entity_type="diagnosis_or_problem",
                    text=cond, normalized_value=cond,
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=min(0.88, 0.70 + section_boost),
                    source="condition_dict",
                    evidence=text,
                    attributes={"match": "condition_dictionary"},
                ))
                break  # one per element to avoid spam

        # 3. "History of / hx of / known case of" + condition
        hx_pattern = r"\b(?:history of|hx of|h/o|known case of|known h/o|dx of|diagnosis of|diagnosed with)\s+([A-Za-z][A-Za-z\s\-]{3,50}?)(?=[,;.\n]|$)"
        for m in re.finditer(hx_pattern, text_l, flags=re.I):
            condition = m.group(1).strip()
            if len(condition) > 6:
                rows.append(self._base_entity(
                    entity_type="past_diagnosis",
                    text=m.group(0), normalized_value=condition,
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=min(0.82, 0.68 + section_boost),
                    source="history_of_pattern",
                    evidence=text,
                    attributes={"condition": condition},
                ))

        # 4. CPT-like 5-digit codes
        for m in re.finditer(r"\b([0-9]{5})\b", text):
            code = m.group(1)
            # Valid CPT range: 10000-99999
            if 10000 <= int(code) <= 99999:
                rows.append(self._base_entity(
                    entity_type="procedure_code",
                    text=code, normalized_value=code,
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=0.65, source="cpt_like_regex",
                    evidence=text, attributes={"code": code, "type": "cpt_candidate"},
                ))
        return rows

    # ── Imaging entities ──────────────────────────────────────────────────────

    def extract_imaging_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        pattern = (
            r"\b(x[- ]?ray|radiograph|chest x[- ]?ray|cxr|"
            r"ct(?: scan)?|cat scan|computed tomography|"
            r"mri|magnetic resonance|"
            r"ultrasound|usg|sonography|echo|echocardiogram|"
            r"pet(?: scan)?|spect|nuclear medicine|"
            r"fluoroscopy|mammogram|mammography|"
            r"bone scan|dexa|densitometry|"
            r"angiogram|angiography|venogram|"
            r"eeg|emg|nerve conduction)\b"
        )
        for m in re.finditer(pattern, text, flags=re.I):
            # Try to capture body part / laterality near it
            context_start = max(0, m.start() - 30)
            context_end   = min(len(text), m.end() + 50)
            context       = text[context_start:context_end]
            rows.append(self._base_entity(
                entity_type="imaging",
                text=m.group(0), normalized_value=m.group(0).lower().strip(),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.78, source="imaging_keyword_v2",
                evidence=context,
                attributes={"modality": m.group(0).lower()},
            ))
        return rows

    # ── Procedure entities ────────────────────────────────────────────────────

    def extract_procedure_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        pattern = (
            r"\b(surgery|operation|procedure|"
            r"biopsy|excision|resection|ablation|"
            r"endoscopy|colonoscopy|bronchoscopy|cystoscopy|"
            r"angiography|angioplasty|stent(?:ing)?|catheterization|"
            r"intubation|extubation|tracheostomy|"
            r"appendectomy|cholecystectomy|colectomy|"
            r"arthroplasty|arthroscopy|laminectomy|discectomy|"
            r"debridement|amputation|"
            r"dialysis|hemodialysis|plasmapheresis|"
            r"transfusion|infusion|"
            r"incision|drainage|i&d|suturing|repair|"
            r"cpr|cardioversion|defibrillation|"
            r"injection|aspiration|paracentesis|thoracentesis|"
            r"lumbar puncture|lp|spinal tap|"
            r"pacemaker|defibrillator|icd)\b"
        )
        for m in re.finditer(pattern, text, flags=re.I):
            rows.append(self._base_entity(
                entity_type="procedure",
                text=m.group(0), normalized_value=m.group(0).lower(),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.72, source="procedure_keyword_v2",
                evidence=text, attributes={},
            ))
        return rows

    # ── Allergy entities ──────────────────────────────────────────────────────

    def extract_allergy_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        text_l = lower_clean(text)
        section_boost = 0.15 if section_name == "allergies" else 0.0

        # NKDA / NKA
        if re.search(r"\b(?:nkda|nka|no known drug allergies?|no known allergies?)\b", text_l):
            rows.append(self._base_entity(
                entity_type="allergy_nkda",
                text=text[:80], normalized_value="NKDA",
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.92, source="nkda_pattern",
                evidence=text, attributes={"type": "no_known_drug_allergies"},
            ))
            return rows

        # "Allergic to X" / "Allergy to X" / "X allergy"
        patterns = [
            r"\ball(?:ergic|ergy)\s+to\s+([A-Za-z][A-Za-z0-9\s\-]{2,40}?)(?=[,;.\n]|$)",
            r"\b([A-Za-z][A-Za-z0-9\s\-]{2,30}?)\s+allergy\b",
            r"\b([A-Za-z][A-Za-z0-9\s\-]{2,30}?)\s+intolerance\b",
            r"\breaction to\s+([A-Za-z][A-Za-z0-9\s\-]{2,30}?)(?=[,;.\n]|$)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text_l, flags=re.I):
                allergen = m.group(1).strip()
                if len(allergen) >= 3:
                    rows.append(self._base_entity(
                        entity_type="allergy",
                        text=m.group(0), normalized_value=allergen,
                        page_number=page, element_id=element_id, section_name=section_name,
                        confidence=min(0.88, 0.72 + section_boost),
                        source="allergy_pattern_v2",
                        evidence=text,
                        attributes={"allergen": allergen},
                    ))

        # In allergies section: any capitalized word/phrase on its own line is likely an allergen
        if section_name == "allergies" and re.match(r"^[A-Z][A-Za-z0-9 \-]{2,40}$", text.strip()):
            allergen = text.strip()
            if not any(e.get("normalized_value") == allergen.lower() for e in rows):
                rows.append(self._base_entity(
                    entity_type="allergy",
                    text=allergen, normalized_value=allergen.lower(),
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=0.68, source="allergy_section_capitalized",
                    evidence=text, attributes={"allergen": allergen},
                ))
        return rows

    # ── Demographics entities ─────────────────────────────────────────────────

    def extract_demographics_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        text_l = lower_clean(text)

        # Age patterns
        age_patterns = [
            (r"\b(\d{1,3})[- ]?(?:year[s]?[- ]?old|yo|y/?o|y\.o\.)\s*([MFmf](?:ale|emale)?)?", "age"),
            (r"\bage\s*[:\-]?\s*(\d{1,3})\b", "age_labeled"),
            (r"\b(\d{1,3})\s*(?:M|F)\b", "age_gender_compact"),
        ]
        for pat, subtype in age_patterns:
            for m in re.finditer(pat, text, flags=re.I):
                age_val = m.group(1)
                try:
                    age_int = int(age_val)
                    if 0 <= age_int <= 130:
                        rows.append(self._base_entity(
                            entity_type="demographics_age",
                            text=m.group(0), normalized_value=age_int,
                            page_number=page, element_id=element_id, section_name=section_name,
                            confidence=0.85, source=f"age_pattern_{subtype}",
                            evidence=text, attributes={"age": age_int},
                        ))
                except ValueError:
                    pass

        # Gender
        if re.search(r"\b(?:male|female|man|woman|boy|girl)\b|\(M\)|\(F\)|(?<!\w)[MF](?!\w)", text, flags=re.I):
            gender_m = re.search(r"\b(male|female|man|woman|boy|girl)\b|\(([MF])\)", text, flags=re.I)
            if gender_m:
                raw_g = (gender_m.group(1) or gender_m.group(2) or "").lower()
                gender = "male" if raw_g in ("male", "man", "boy", "m") else "female" if raw_g in ("female", "woman", "girl", "f") else raw_g
                rows.append(self._base_entity(
                    entity_type="demographics_gender",
                    text=gender_m.group(0), normalized_value=gender,
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=0.80, source="gender_pattern",
                    evidence=text, attributes={"gender": gender},
                ))

        # DOB
        dob_pattern = r"\b(?:dob|date of birth|birthday|d\.o\.b\.?)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})"
        for m in re.finditer(dob_pattern, text, flags=re.I):
            rows.append(self._base_entity(
                entity_type="demographics_dob",
                text=m.group(0), normalized_value=m.group(1),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=0.90, source="dob_pattern",
                evidence=text, attributes={"dob_raw": m.group(1)},
            ))

        # Patient name from labeled fields
        name_pattern = r"\b(?:patient|patient name|name|pt|pt\.)\s*[:\-]\s*([A-Z][A-Za-z]+(?:[\s,]+[A-Z][A-Za-z]+){0,4})\b"
        for m in re.finditer(name_pattern, text, flags=re.I):
            name = m.group(1).strip().rstrip(",")
            if len(name.split()) >= 2:
                rows.append(self._base_entity(
                    entity_type="demographics_name",
                    text=m.group(0), normalized_value=name,
                    page_number=page, element_id=element_id, section_name=section_name,
                    confidence=0.75, source="name_label_pattern",
                    evidence=text, attributes={"name": name},
                ))
        return rows

    # ── Social history entities ───────────────────────────────────────────────

    def extract_social_history_entities(self, text, page, element_id, section_name):
        rows: List[Dict[str, Any]] = []
        text_l = lower_clean(text)
        section_boost = 0.12 if section_name == "social_history" else 0.0

        # Smoking
        smoking_m = re.search(
            r"\b(non[- ]?smoker|never smoked|never smok|current smoker|former smoker|ex[- ]smoker"
            r"|smokes?\s+\d+\s*pack|pack[- ]?year|tobacco use|tobacco history|cigarette)\b",
            text_l, flags=re.I,
        )
        if smoking_m:
            rows.append(self._base_entity(
                entity_type="social_smoking",
                text=smoking_m.group(0), normalized_value=smoking_m.group(0),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=min(0.88, 0.76 + section_boost), source="smoking_pattern",
                evidence=text, attributes={},
            ))

        # Alcohol
        alcohol_m = re.search(
            r"\b(alcohol|etoh|drinks?\s+\w+|social drinker|non[- ]?drinker|denies alcohol|occasional drinker|heavy drinker|sober)\b",
            text_l, flags=re.I,
        )
        if alcohol_m:
            rows.append(self._base_entity(
                entity_type="social_alcohol",
                text=alcohol_m.group(0), normalized_value=alcohol_m.group(0),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=min(0.82, 0.70 + section_boost), source="alcohol_pattern",
                evidence=text, attributes={},
            ))

        # Substance use
        substance_m = re.search(
            r"\b(illicit drug|recreational drug|marijuana|cannabis|cocaine|heroin|methamphetamine|"
            r"opioid use|substance abuse|drug use|ivdu|intravenous drug)\b",
            text_l, flags=re.I,
        )
        if substance_m:
            rows.append(self._base_entity(
                entity_type="social_substance",
                text=substance_m.group(0), normalized_value=substance_m.group(0),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=min(0.85, 0.72 + section_boost), source="substance_pattern",
                evidence=text, attributes={},
            ))

        # Occupation
        occ_m = re.search(
            r"\b(?:occupation|works? as|employed as|retired|unemployed|occupation[:\-]|job[:\-])\s*([A-Za-z][A-Za-z\s]{2,40}?)(?=[,;.\n]|$)",
            text_l, flags=re.I,
        )
        if occ_m:
            rows.append(self._base_entity(
                entity_type="social_occupation",
                text=occ_m.group(0), normalized_value=occ_m.group(1).strip(),
                page_number=page, element_id=element_id, section_name=section_name,
                confidence=min(0.75, 0.62 + section_boost), source="occupation_pattern",
                evidence=text, attributes={"occupation": occ_m.group(1).strip()},
            ))
        return rows

    # ── Element-level metadata ────────────────────────────────────────────────

    def extract_element_metadata(self, line_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for e in line_elements:
            text = normalize_space(e.get("text") or e.get("text_preview") or "")
            if not text:
                continue

            section_key, section_conf = self.detect_section(text, e)
            existing_section          = self._element_section_name(e)

            rows.append({
                "document_id":   self.manifest.get("document_id"),
                "element_id":    e.get("element_id"),
                "page_number":   line_page(e),
                "text_preview":  text[:500],

                "section_detected":   section_key,
                "section_confidence": section_conf,
                "section_name":       existing_section,

                "is_heading_candidate":          bool(e.get("is_heading_candidate")),
                "heading_score":                 float(e.get("heading_score") or 0.0),
                "is_repeated_header_candidate":  bool(e.get("is_repeated_header_candidate")),
                "is_repeated_footer_candidate":  bool(e.get("is_repeated_footer_candidate")),

                "contains_date":             bool(self.extract_dates_from_text(text)),
                "contains_patient_id":       bool(self.extract_patient_ids(text)),
                "contains_vital":            any(re.search(p, text, flags=re.I) for p in VITAL_PATTERNS.values()),
                "contains_lab_candidate":    any(re.search(rf"\b{re.escape(x)}\b", text, flags=re.I) for x in COMMON_LAB_NAMES[:40]),
                "contains_medication_cue":   bool(
                    any(re.search(rf"\b{re.escape(d)}\b", lower_clean(text)) for d in list(self._drug_set)[:60])
                    or any(re.search(p, lower_clean(text)) for p, _ in DRUG_SUFFIX_PATTERNS[:6])
                ),
                "contains_negation":         self.contains_negation(text),
                "contains_allergy":          bool(re.search(r"\b(?:allerg|nkda|nka)\w*\b", text, flags=re.I)),
                "contains_diagnosis_cue":    any(re.search(rf"\b{re.escape(c)}\b", lower_clean(text)) for c in list(self._cond_set)[:60]),

                "clinical_metadata_json": safe_json_dumps({"source": _VERSION}),
            })
        return rows

    # ── Pattern extractors ────────────────────────────────────────────────────

    def extract_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            # Written month name — highest confidence
            (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+(\d{2,4})\b", 0.92),
            (r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})\b", 0.92),
            # ISO date
            (r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", 0.92),
            # US numeric MM/DD/YYYY
            (r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", 0.75),
            # DD-MM-YYYY with hyphens
            (r"\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b", 0.72),
        ]
        out: List[Dict[str, Any]] = []
        seen: set = set()

        for pattern, base_conf in patterns:
            for m in re.finditer(pattern, text, flags=re.I):
                raw = m.group(0)
                if raw in seen:
                    continue
                seen.add(raw)
                out.append({
                    "raw":        raw,
                    "normalized": self._normalize_date_v2(raw),
                    "confidence": base_conf,
                    "source":     "date_regex_v2",
                })
        return out

    def _normalize_date_v2(self, raw: str) -> str:
        raw = raw.strip()
        MONTHS = {
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
            "january":1,"february":2,"march":3,"april":4,"june":6,
            "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
        }

        # Written: "15 January 2020" or "January 15, 2020"
        m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{2,4})$", raw)
        if m:
            d, month_s, y = m.groups()
            mo = MONTHS.get(month_s.lower()[:9])
            if mo:
                y = int(y); y = y + 2000 if y < 100 else y
                return f"{y:04d}-{mo:02d}-{int(d):02d}"

        m = re.match(r"^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})$", raw)
        if m:
            month_s, d, y = m.groups()
            mo = MONTHS.get(month_s.lower()[:9])
            if mo:
                y = int(y); y = y + 2000 if y < 100 else y
                return f"{y:04d}-{mo:02d}-{int(d):02d}"

        # ISO
        m = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw)
        if m:
            y, mo, d = map(int, m.groups())
            return f"{y:04d}-{mo:02d}-{d:02d}"

        # US numeric MM/DD/YYYY (assumed US format if first number ≤ 12)
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", raw)
        if m:
            a, b, y = map(int, m.groups())
            y = y + 2000 if y < 100 else y
            if a <= 12:
                mo, d = a, b   # US: MM/DD
            else:
                mo, d = b, a   # day first
            return f"{y:04d}-{mo:02d}-{d:02d}"

        # DD-MM-YYYY
        m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$", raw)
        if m:
            d, mo, y = map(int, m.groups())
            y = y + 2000 if y < 100 else y
            return f"{y:04d}-{mo:02d}-{d:02d}"

        return raw

    def extract_patient_ids(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            r"\b(?:MRN|UHID|Patient\s*ID|Patient\s*No\.?|Reg(?:istration)?\s*No\.?|Account\s*(?:No\.?|Number)?)\s*[:#\-]?\s*([A-Za-z0-9\-\/]{4,20})\b",
            r"\b(?:Case\s*ID|Case\s*No\.?|Record\s*No\.?|Chart\s*No\.?)\s*[:#\-]?\s*([A-Za-z0-9\-\/]{4,20})\b",
            r"\b(?:DOB|Date of Birth)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
            r"\b(?:Acc(?:ession)?\s*No\.?|Acc\s*#)\s*[:#\-]?\s*([A-Za-z0-9\-\/]{4,20})\b",
        ]
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for pattern in patterns:
            for m in re.finditer(pattern, text, flags=re.I):
                raw   = m.group(0)
                value = m.group(1)
                if raw in seen:
                    continue
                seen.add(raw)
                out.append({
                    "raw":        raw,
                    "value":      value,
                    "confidence": 0.88,
                    "source":     "patient_id_regex_v2",
                })
        return out

    def extract_provider_names(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set = set()

        # Credential-suffix pattern: "John Smith, MD"
        for m in PROVIDER_CREDENTIAL_RE.finditer(text):
            raw = m.group(0)
            if raw in seen:
                continue
            seen.add(raw)
            out.append({
                "raw":        raw,
                "name":       m.group(1),
                "confidence": 0.85,
                "source":     "provider_credential_suffix",
            })

        # Label prefix patterns: "Attending: John Smith"
        for pattern in PROVIDER_LABEL_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.I):
                raw = m.group(0)
                if raw in seen:
                    continue
                seen.add(raw)
                out.append({
                    "raw":        raw,
                    "name":       m.group(1),
                    "confidence": 0.78,
                    "source":     "provider_label_pattern",
                })

        return out[:10]

    def extract_facility_names(self, text: str) -> List[Dict[str, Any]]:
        lines      = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]
        candidates = []
        for line in lines[:25]:
            if re.search(
                r"\b(hospital|clinic|medical center|healthcare|health system|nursing home|"
                r"diagnostic(s)?|laboratory|lab|imaging center|surgery center|"
                r"cancer center|rehabilitation|rehab center|ambulatory)\b",
                line, flags=re.I,
            ):
                candidates.append({
                    "raw":        line,
                    "name":       line,
                    "confidence": 0.68,
                    "source":     "facility_keyword_top_page",
                })
        return candidates[:5]

    # ── Negation helpers ──────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _is_negated_scoped(self, full_text: str, entity_text: str) -> bool:
        """Check negation within the same sentence as the entity (sentence-scoped)."""
        text_l   = lower_clean(full_text)
        entity_l = lower_clean(entity_text)

        # Find which sentence contains the entity
        for sentence in self._split_sentences(full_text):
            sent_l = lower_clean(sentence)
            if entity_l in sent_l:
                return self.contains_negation(sentence)

        # Fallback: 120-char window before entity
        idx = text_l.find(entity_l)
        if idx == -1:
            return self.contains_negation(full_text)
        context = text_l[max(0, idx - 120): idx + len(entity_l)]
        return self.contains_negation(context)

    def contains_negation(self, text: str) -> bool:
        return any(re.search(p, text, flags=re.I) for p in NEGATION_TERMS)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _line_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines = [e for e in elements if e.get("element_type") == "line"]
        lines.sort(key=lambda e: (
            int(e.get("page_number") or 0),
            float(e.get("y0") or 0),
            float(e.get("x0") or 0),
            int(e.get("element_id") or 0),
        ))
        return lines

    def _first_pages_text(self, lines: List[Dict[str, Any]], max_pages: int = 3) -> str:
        selected = [
            e.get("text") or e.get("text_preview") or ""
            for e in lines
            if (line_page(e) or 0) < max_pages
        ]
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

    def _dedupe_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[tuple, Dict[str, Any]] = {}
        for e in entities:
            key = (
                e.get("entity_type"),
                lower_clean(str(e.get("normalized_value") or "")),
                e.get("page_number"),
                e.get("section_name"),
            )
            old = best.get(key)
            if old is None or float(e.get("confidence") or 0) > float(old.get("confidence") or 0):
                best[key] = e
        return list(best.values())

    # ── Backwards-compatible aliases ──────────────────────────────────────────

    def is_negated_context(self, full_text: str, entity_text: str, window_chars: int = 80) -> bool:
        return self._is_negated_scoped(full_text, entity_text)

    def extract_medication_like_entities(self, text, page, element_id, section_name):
        return self.extract_medication_entities(text, page, element_id, section_name)

    def extract_diagnosis_like_entities(self, text, page, element_id, section_name):
        return self.extract_diagnosis_entities(text, page, element_id, section_name)

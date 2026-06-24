"""Query understanding for medical RAG.

Responsibilities
----------------
- Medical abbreviation expansion (HTN → hypertension, DM → diabetes mellitus …)
- Query type classification (FACTUAL, SYNTHESIS, MEDICATION, DIAGNOSTIC, TEMPORAL, COMPARISON)
- Medical entity extraction from free-text queries
- HyDE: generate a hypothetical document that answers the query, for better embedding alignment
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LLMConfig

# ─────────────────────────────────────────────────────────────────────────────
# Medical abbreviation dictionary (90+ common clinical abbreviations)
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    # Diagnoses
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "DM2": "type 2 diabetes mellitus",
    "DM1": "type 1 diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "CAD": "coronary artery disease",
    "CHF": "congestive heart failure",
    "HF": "heart failure",
    "AF": "atrial fibrillation",
    "AFib": "atrial fibrillation",
    "AFIB": "atrial fibrillation",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
    "ESRD": "end-stage renal disease",
    "CVA": "cerebrovascular accident stroke",
    "TIA": "transient ischemic attack",
    "MI": "myocardial infarction heart attack",
    "STEMI": "ST-elevation myocardial infarction",
    "NSTEMI": "non-ST-elevation myocardial infarction",
    "PE": "pulmonary embolism",
    "DVT": "deep vein thrombosis",
    "RA": "rheumatoid arthritis",
    "SLE": "systemic lupus erythematosus",
    "IBD": "inflammatory bowel disease",
    "UC": "ulcerative colitis",
    "GERD": "gastroesophageal reflux disease",
    "UTI": "urinary tract infection",
    "PNA": "pneumonia",
    "SOB": "shortness of breath dyspnea",
    "CP": "chest pain",
    "HA": "headache",
    "Fx": "fracture",
    # Labs and vitals
    "BG": "blood glucose",
    "BS": "blood sugar glucose",
    "HbA1c": "hemoglobin A1c glycated hemoglobin",
    "A1C": "hemoglobin A1c glycated hemoglobin",
    "Cr": "creatinine",
    "BUN": "blood urea nitrogen",
    "GFR": "glomerular filtration rate",
    "eGFR": "estimated glomerular filtration rate",
    "K": "potassium",
    "Na": "sodium",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "WBC": "white blood cell count",
    "RBC": "red blood cell count",
    "Plt": "platelet count",
    "INR": "international normalized ratio coagulation",
    "PT": "prothrombin time",
    "aPTT": "activated partial thromboplastin time",
    "TSH": "thyroid stimulating hormone",
    "LFTs": "liver function tests",
    "AST": "aspartate aminotransferase liver",
    "ALT": "alanine aminotransferase liver",
    "ALP": "alkaline phosphatase",
    "TBili": "total bilirubin",
    "LDL": "low-density lipoprotein cholesterol",
    "HDL": "high-density lipoprotein cholesterol",
    "TG": "triglycerides",
    "BP": "blood pressure",
    "HR": "heart rate pulse",
    "RR": "respiratory rate",
    "O2sat": "oxygen saturation",
    "SpO2": "oxygen saturation",
    "BMI": "body mass index",
    # Medications
    "ACEI": "ACE inhibitor angiotensin converting enzyme inhibitor",
    "ARB": "angiotensin receptor blocker",
    "BB": "beta blocker",
    "CCB": "calcium channel blocker",
    "ASA": "aspirin",
    "NSAID": "nonsteroidal anti-inflammatory drug",
    "PPI": "proton pump inhibitor",
    "SSRI": "selective serotonin reuptake inhibitor",
    "TCA": "tricyclic antidepressant",
    "Abx": "antibiotics",
    "abx": "antibiotics",
    "IV": "intravenous",
    "PO": "oral by mouth",
    "IM": "intramuscular",
    "SQ": "subcutaneous",
    "PRN": "as needed",
    "QD": "once daily",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    # Clinical context
    "HPI": "history of present illness",
    "PMH": "past medical history",
    "FH": "family history",
    "SH": "social history",
    "ROS": "review of systems",
    "PE": "physical examination",
    "Dx": "diagnosis",
    "Rx": "prescription treatment",
    "Tx": "treatment",
    "Sx": "symptoms",
    "CC": "chief complaint",
    "H&P": "history and physical",
    "DC": "discharge",
    "ED": "emergency department",
    "ICU": "intensive care unit",
    "OR": "operating room",
    "OT": "occupational therapy",
    "PT": "physical therapy",
    "f/u": "follow-up",
    "F/U": "follow-up",
}


# ─────────────────────────────────────────────────────────────────────────────
# Query type
# ─────────────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    FACTUAL = "factual"         # "What was the patient's blood pressure?"
    MEDICATION = "medication"   # "What medications were prescribed?"
    DIAGNOSTIC = "diagnostic"   # "What is the primary diagnosis?"
    LAB = "lab"                 # "What were the lab results?"
    TEMPORAL = "temporal"       # "When was the patient last seen?"
    SYNTHESIS = "synthesis"     # "Summarise the patient's clinical picture."
    COMPARISON = "comparison"   # "How did the patient's condition change?"


_MEDICATION_PATTERNS = re.compile(
    r"\b(medic\w+|drug|dose|dosage|prescri\w+|mg|mcg|tablet|capsule|pill|inject\w+|"
    r"formulat\w+|pharmac\w+|treat\w+|therap\w+)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_PATTERNS = re.compile(
    r"\b(diagnos\w+|impression|assessment|condition|disease|disorder|finding|"
    r"patholog\w+|abnormal\w*|etiol\w+)\b",
    re.IGNORECASE,
)
_LAB_PATTERNS = re.compile(
    r"\b(lab\w*|result\w*|level\w*|value\w*|test\w*|panel\w*|culture\w*|"
    r"biopsy|imaging|x-ray|ct\b|mri\b|ultrasound|ecg|ekg)\b",
    re.IGNORECASE,
)
_TEMPORAL_PATTERNS = re.compile(
    r"\b(when|date|time|yesterday|today|last|prior|previous|history|ago|"
    r"admission|discharge|visit\w*)\b",
    re.IGNORECASE,
)
_SYNTHESIS_PATTERNS = re.compile(
    r"\b(summar\w+|overview|overall|explain|descri\w+|tell me about|"
    r"what is the|clinical picture|status)\b",
    re.IGNORECASE,
)
_COMPARISON_PATTERNS = re.compile(
    r"\b(compar\w+|chang\w+|differ\w+|better|worse|improv\w+|deterior\w+|"
    r"progress\w+|trend\w*|vs\.?\b|versus)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# QueryManager
# ─────────────────────────────────────────────────────────────────────────────

class QueryManager:
    """Medical query understanding: expansion, classification, and HyDE."""

    def __init__(self, llm_config: Optional["LLMConfig"] = None) -> None:
        self._llm_config = llm_config
        self._llm: Optional[object] = None     # lazy-loaded LLM provider

    # ------------------------------------------------------------------
    # Abbreviation expansion
    # ------------------------------------------------------------------

    def expand(self, query: str) -> List[str]:
        """Return [original_query] + expanded variants (abbreviations resolved).

        The first element is always the original query unchanged.
        """
        expanded = self._expand_abbreviations(query)
        variants = [query]
        if expanded != query:
            variants.append(expanded)
        return variants

    def _expand_abbreviations(self, text: str) -> str:
        """Replace known medical abbreviations with full terms."""
        # Word-boundary aware replacement, case-sensitive keys
        result = text
        for abbrev, full in MEDICAL_ABBREVIATIONS.items():
            pattern = r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)"
            result = re.sub(pattern, full, result)
        return result

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def classify(self, query: str) -> QueryType:
        """Heuristic classification of the medical query type."""
        q = query.lower()
        if _MEDICATION_PATTERNS.search(q):
            return QueryType.MEDICATION
        if _LAB_PATTERNS.search(q):
            return QueryType.LAB
        if _DIAGNOSTIC_PATTERNS.search(q):
            return QueryType.DIAGNOSTIC
        if _TEMPORAL_PATTERNS.search(q):
            return QueryType.TEMPORAL
        if _COMPARISON_PATTERNS.search(q):
            return QueryType.COMPARISON
        if _SYNTHESIS_PATTERNS.search(q):
            return QueryType.SYNTHESIS
        return QueryType.FACTUAL

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract medication names, diagnoses, lab values, and sections from query."""
        entities: Dict[str, List[str]] = {
            "medications": [],
            "diagnoses": [],
            "labs": [],
            "sections": [],
            "abbreviations_found": [],
        }

        # Find abbreviations in original query
        for abbrev in MEDICAL_ABBREVIATIONS:
            pattern = r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)"
            if re.search(pattern, query):
                entities["abbreviations_found"].append(abbrev)

        # Known medication keywords
        med_keywords = re.findall(
            r"\b(metformin|lisinopril|aspirin|atorvastatin|amlodipine|omeprazole|"
            r"metoprolol|losartan|enalapril|warfarin|heparin|insulin|prednisone|"
            r"amoxicillin|azithromycin|hydrochlorothiazide|furosemide|simvastatin)\b",
            query, re.IGNORECASE,
        )
        entities["medications"] = [m.lower() for m in med_keywords]

        # Known section keywords
        section_keywords = re.findall(
            r"\b(assessment|plan|history|examination|medications|labs|"
            r"vitals|diagnosis|impression|hpi|pmh|ros)\b",
            query, re.IGNORECASE,
        )
        entities["sections"] = list({s.lower() for s in section_keywords})

        return entities

    # ------------------------------------------------------------------
    # HyDE (Hypothetical Document Embedding)
    # ------------------------------------------------------------------

    def generate_hyde(self, query: str) -> str:
        """Generate a hypothetical medical document passage that answers the query.

        This passage is then embedded and used for semantic retrieval instead of
        (or alongside) the original query embedding — often improves recall for
        clinical note queries.

        Requires an LLM config to be provided at construction time.
        """
        if self._llm_config is None:
            raise RuntimeError(
                "LLMConfig required for HyDE generation. "
                "Provide llm_config= when constructing QueryManager."
            )
        llm = self._get_llm()
        hyde_prompt = (
            "Write a brief clinical note passage (2–4 sentences) that would directly "
            f"answer the following question. Use medical terminology:\n\nQuestion: {query}"
        )
        messages = [
            {"role": "system", "content": "You are a clinical documentation assistant."},
            {"role": "user", "content": hyde_prompt},
        ]
        response = llm.chat.completions.create(
            model=self._llm_config.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content or ""

    def _get_llm(self):
        if self._llm is None:
            self._llm = _build_llm_client(self._llm_config)  # type: ignore[arg-type]
        return self._llm

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        query: str,
        use_hyde: bool = False,
    ) -> Dict[str, object]:
        """Run full query understanding pipeline.

        Returns a dict with: original, expanded, query_type, entities, hyde_passage.
        """
        expanded_variants = self.expand(query)
        query_type = self.classify(query)
        entities = self.extract_entities(query)
        hyde_passage: Optional[str] = None

        if use_hyde and self._llm_config is not None:
            try:
                hyde_passage = self.generate_hyde(query)
            except Exception:
                pass

        return {
            "original": query,
            "expanded": expanded_variants,
            "query_type": query_type.value,
            "entities": entities,
            "hyde_passage": hyde_passage,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build OpenAI-compatible LLM client
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm_client(config: "LLMConfig"):
    import os
    try:
        import openai
    except ImportError as e:
        raise ImportError("openai is required for HyDE. Install with: uv pip install openai") from e
    api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    kwargs: Dict[str, object] = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return openai.OpenAI(**kwargs)

"""Tests for nitrag/query_manager.py — QueryManager, MEDICAL_ABBREVIATIONS, QueryType."""
from __future__ import annotations

import pytest

from nitrag.query_manager import (
    MEDICAL_ABBREVIATIONS,
    QueryManager,
    QueryType,
)


# ---------------------------------------------------------------------------
# MEDICAL_ABBREVIATIONS dictionary
# ---------------------------------------------------------------------------

class TestMedicalAbbreviations:
    def test_has_at_least_60_entries(self):
        assert len(MEDICAL_ABBREVIATIONS) >= 60

    def test_htn_maps_to_hypertension(self):
        assert MEDICAL_ABBREVIATIONS["HTN"] == "hypertension"

    def test_dm2_contains_diabetes(self):
        assert "diabetes" in MEDICAL_ABBREVIATIONS["DM2"].lower()

    def test_dm_maps_to_diabetes_mellitus(self):
        assert "diabetes mellitus" in MEDICAL_ABBREVIATIONS["DM"].lower()

    def test_bid_maps_to_twice_daily(self):
        assert "twice daily" in MEDICAL_ABBREVIATIONS["BID"].lower()

    def test_chf_is_heart_failure(self):
        assert "heart failure" in MEDICAL_ABBREVIATIONS["CHF"].lower()

    def test_mi_is_myocardial_infarction(self):
        assert "myocardial infarction" in MEDICAL_ABBREVIATIONS["MI"].lower()

    def test_all_values_are_strings(self):
        for key, value in MEDICAL_ABBREVIATIONS.items():
            assert isinstance(key, str), f"Key {key!r} is not str"
            assert isinstance(value, str), f"Value for {key!r} is not str"
            assert len(value) > 0, f"Value for {key!r} is empty"


# ---------------------------------------------------------------------------
# QueryType enum
# ---------------------------------------------------------------------------

class TestQueryType:
    def test_factual_exists(self):
        assert QueryType.FACTUAL

    def test_medication_exists(self):
        assert QueryType.MEDICATION

    def test_diagnostic_exists(self):
        assert QueryType.DIAGNOSTIC

    def test_temporal_exists(self):
        assert QueryType.TEMPORAL

    def test_synthesis_exists(self):
        assert QueryType.SYNTHESIS

    def test_comparison_exists(self):
        assert QueryType.COMPARISON

    def test_all_are_string_values(self):
        # QueryType(str, Enum) — all values should be strings
        for qt in QueryType:
            assert isinstance(qt.value, str)


# ---------------------------------------------------------------------------
# QueryManager.expand
# ---------------------------------------------------------------------------

class TestQueryManagerExpand:
    def setup_method(self):
        self.qm = QueryManager()

    def test_expand_htn_contains_hypertension(self):
        result = self.qm.expand("HTN")
        # result[0] is original, result[1] (if any) is expanded
        all_text = " ".join(result)
        assert "hypertension" in all_text.lower()

    def test_expand_htn_first_element_is_original(self):
        result = self.qm.expand("HTN")
        assert result[0] == "HTN"

    def test_expand_dm_and_htn_expands_both(self):
        result = self.qm.expand("DM and HTN")
        assert len(result) >= 2
        expanded = result[1]
        assert "diabetes" in expanded.lower()
        assert "hypertension" in expanded.lower()

    def test_expand_no_abbreviations_returns_original_only(self):
        query = "no abbreviations here whatsoever"
        result = self.qm.expand(query)
        # Original is always first
        assert result[0] == query
        # No expansion needed
        assert result == [query]

    def test_expand_bid_contains_twice_daily(self):
        result = self.qm.expand("Metformin BID")
        all_text = " ".join(result)
        assert "twice daily" in all_text.lower()

    def test_expand_returns_list(self):
        assert isinstance(self.qm.expand("HTN"), list)

    def test_expand_preserves_non_abbreviation_words(self):
        result = self.qm.expand("Patient has HTN")
        # The expanded form should still contain "Patient has"
        assert any("Patient has" in v for v in result)


# ---------------------------------------------------------------------------
# QueryManager.classify
# ---------------------------------------------------------------------------

class TestQueryManagerClassify:
    def setup_method(self):
        self.qm = QueryManager()

    def test_medication_query_classified_as_medication(self):
        qt = self.qm.classify("What medications were prescribed?")
        assert qt == QueryType.MEDICATION

    def test_diagnosis_query_classified_as_diagnostic(self):
        qt = self.qm.classify("What is the diagnosis?")
        assert qt == QueryType.DIAGNOSTIC

    def test_comparison_query_classified_as_comparison(self):
        # Use a query with comparison keywords but no medication/diagnostic/temporal words
        qt = self.qm.classify("Is the patient better or worse compared to before?")
        assert qt == QueryType.COMPARISON

    def test_temporal_query_classified_as_temporal(self):
        qt = self.qm.classify("When was the procedure done?")
        assert qt == QueryType.TEMPORAL

    def test_treatment_query_classified_as_medication(self):
        qt = self.qm.classify("What treatment was given at what dose?")
        assert qt == QueryType.MEDICATION

    def test_lab_query_classified_as_lab(self):
        qt = self.qm.classify("What were the lab results for HbA1c?")
        assert qt == QueryType.LAB

    def test_summary_query_classified_as_synthesis(self):
        qt = self.qm.classify("Summarise the patient's clinical status")
        assert qt == QueryType.SYNTHESIS

    def test_unknown_query_falls_back_to_factual(self):
        qt = self.qm.classify("purple elephant seven")
        assert qt == QueryType.FACTUAL


# ---------------------------------------------------------------------------
# QueryManager.process
# ---------------------------------------------------------------------------

class TestQueryManagerProcess:
    def setup_method(self):
        self.qm = QueryManager()

    def test_process_returns_expanded_list(self):
        result = self.qm.process("HTN meds BID")
        assert "expanded" in result
        assert isinstance(result["expanded"], list)

    def test_process_expanded_contains_hypertension(self):
        result = self.qm.process("HTN meds BID")
        all_text = " ".join(result["expanded"])
        assert "hypertension" in all_text.lower()

    def test_process_returns_query_type(self):
        result = self.qm.process("What medications were prescribed?")
        assert "query_type" in result
        assert result["query_type"] == QueryType.MEDICATION.value

    def test_process_without_hyde_hyde_passage_is_none(self):
        result = self.qm.process("What is the blood pressure?", use_hyde=False)
        assert result.get("hyde_passage") is None

    def test_process_original_matches_input(self):
        query = "What is the diagnosis for this patient?"
        result = self.qm.process(query)
        assert result["original"] == query

    def test_process_returns_entities_dict(self):
        result = self.qm.process("What metformin dose was prescribed?")
        assert "entities" in result
        assert isinstance(result["entities"], dict)

    def test_process_entities_medications_found(self):
        result = self.qm.process("What metformin dose was prescribed?")
        entities = result["entities"]
        assert "metformin" in [m.lower() for m in entities.get("medications", [])]

    def test_process_entities_abbreviations_found(self):
        result = self.qm.process("HTN and DM management")
        entities = result["entities"]
        found_abbrevs = entities.get("abbreviations_found", [])
        assert "HTN" in found_abbrevs or "DM" in found_abbrevs

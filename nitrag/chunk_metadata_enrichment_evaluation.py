from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq


def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def read_parquet_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_id_list(value: Any) -> List[int]:
    parsed = safe_json_loads(value)
    if not isinstance(parsed, list):
        return []

    out = []
    for item in parsed:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def entity_key(entity: Dict[str, Any]) -> Tuple[str, str, int, int]:
    page = entity.get("page_number", entity.get("page"))
    return (
        str(entity.get("entity_type") or entity.get("type") or ""),
        str(entity.get("normalized_value") or entity.get("text") or "").lower(),
        safe_int(page),
        safe_int(entity.get("element_id")),
    )


class ChunkMetadataEnrichmentEvaluationManager:
    """
    Focused evaluator for ChunkMetadataEnricher outputs.

    It reads rag_store/<doc_id>/chunks_enriched/*.parquet and checks whether the
    enrichment stage added complete, internally consistent, and source-backed
    metadata. It does not evaluate retrieval or generation quality.
    """

    FLAG_COLUMNS = [
        "contains_date",
        "contains_patient_id",
        "contains_vital",
        "contains_lab",
        "contains_medication",
        "contains_diagnosis",
        "contains_imaging",
        "contains_procedure",
        "contains_negation",
    ]

    REQUIRED_COLUMNS = [
        "metadata_json",
        "document_type",
        "primary_section",
        "section_names_json",
        "source_element_ids_json",
        "overlap_line_count",
        "entity_count",
        "entity_type_counts_json",
        "entities_json",
        "clinical_quality_score",
        *FLAG_COLUMNS,
    ]

    ENTITY_FLAG_TYPES = {
        "contains_date": {"date"},
        "contains_patient_id": {"patient_identifier"},
        "contains_vital": {"vital"},
        "contains_lab": {"lab_result"},
        "contains_medication": {"medication_candidate", "medication_line_candidate"},
        "contains_diagnosis": {"diagnosis_code_candidate", "diagnosis_or_problem_candidate"},
        "contains_imaging": {"imaging_candidate"},
        "contains_procedure": {"procedure_candidate"},
    }

    ELEMENT_FLAG_COLUMNS = {
        "contains_date": ["contains_date"],
        "contains_patient_id": ["contains_patient_id"],
        "contains_vital": ["contains_vital"],
        "contains_lab": ["contains_lab_candidate"],
        "contains_medication": ["contains_medication_cue"],
        "contains_negation": ["contains_negation"],
    }

    def __init__(
        self,
        store_or_document_dir: Any,
        enriched_chunk_dir: Optional[Union[str, Path]] = None,
        report_dir: Optional[Union[str, Path]] = None,
    ):
        if hasattr(store_or_document_dir, "paths"):
            self.document_dir = Path(store_or_document_dir.paths.document_dir)
        else:
            self.document_dir = Path(store_or_document_dir)

        self.enriched_chunk_dir = Path(enriched_chunk_dir or self.document_dir / "chunks_enriched")
        self.report_dir = Path(report_dir or self.document_dir / "reports" / "chunk_metadata_enrichment_evaluation")
        self.metrics_dir = self.report_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.clinical_entities = read_parquet_df(self.document_dir / "clinical_entities.parquet")
        self.clinical_element_metadata = read_parquet_df(self.document_dir / "clinical_element_metadata.parquet")

        self.source_entity_keys = self._source_entity_keys()
        self.element_flags_by_id = self._element_flags_by_id()

    def list_strategies(self) -> List[str]:
        if not self.enriched_chunk_dir.exists():
            return []
        return sorted(p.stem for p in self.enriched_chunk_dir.glob("*.parquet"))

    def load_chunks(self, strategy: str) -> pd.DataFrame:
        path = self.enriched_chunk_dir / f"{strategy}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return read_parquet_df(path)

    def compute_schema_completeness(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            for column in self.REQUIRED_COLUMNS:
                present = column in df.columns
                if present and len(df) > 0:
                    non_null = int(df[column].notna().sum())
                    coverage = float(non_null / len(df) * 100)
                else:
                    non_null = 0
                    coverage = 0.0

                rows.append({
                    "strategy": strategy,
                    "column": column,
                    "present": bool(present),
                    "non_null_count": non_null,
                    "chunk_count": int(len(df)),
                    "coverage_pct": coverage,
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "schema_completeness.csv")
        return out

    def compute_enrichment_metrics(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            rows.append(self._strategy_metrics(strategy, df))

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "enrichment_metrics.csv")
        return out

    def compute_flag_consistency(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            for flag in self.FLAG_COLUMNS:
                mismatches = 0
                positives = 0
                expected_positives = 0

                for row in df.to_dict("records"):
                    actual = bool(row.get(flag))
                    expected = self._expected_flag(row, flag)
                    positives += int(actual)
                    expected_positives += int(expected)
                    mismatches += int(actual != expected)

                rows.append({
                    "strategy": strategy,
                    "flag": flag,
                    "chunk_count": int(len(df)),
                    "actual_positive_count": positives,
                    "expected_positive_count": expected_positives,
                    "mismatch_count": mismatches,
                    "match_pct": float((1 - mismatches / max(1, len(df))) * 100),
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "flag_consistency.csv")
        return out

    def find_suspicious_chunks(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            for row in df.to_dict("records"):
                reasons = self._suspicious_reasons(row)
                if reasons:
                    rows.append({
                        "strategy": strategy,
                        "chunk_id": row.get("chunk_id"),
                        "page_start": row.get("page_start"),
                        "page_end": row.get("page_end"),
                        "token_length": row.get("token_length"),
                        "primary_section": row.get("primary_section"),
                        "entity_count": row.get("entity_count"),
                        "clinical_quality_score": row.get("clinical_quality_score"),
                        "reasons": "; ".join(reasons),
                    })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "suspicious_chunks.csv")
        return out

    def generate_report(self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        strategies = strategies or self.list_strategies()
        schema_df = self.compute_schema_completeness(strategies)
        metrics_df = self.compute_enrichment_metrics(strategies)
        flags_df = self.compute_flag_consistency(strategies)
        suspicious_df = self.find_suspicious_chunks(strategies)

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "strategy_count": int(len(strategies)),
            "source_entity_count": int(len(self.source_entity_keys)),
            "metric_files": [
                str(self.metrics_dir / "schema_completeness.csv"),
                str(self.metrics_dir / "enrichment_metrics.csv"),
                str(self.metrics_dir / "flag_consistency.csv"),
                str(self.metrics_dir / "suspicious_chunks.csv"),
            ],
            "schema_rows": int(len(schema_df)),
            "metric_rows": int(len(metrics_df)),
            "flag_rows": int(len(flags_df)),
            "suspicious_chunk_rows": int(len(suspicious_df)),
        }
        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def _strategy_metrics(self, strategy: str, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {
                "strategy": strategy,
                "chunk_count": 0,
                "section_coverage_pct": 0.0,
                "source_element_coverage_pct": 0.0,
                "entity_chunk_coverage_pct": 0.0,
                "avg_entity_count": 0.0,
                "avg_quality_score": 0.0,
                "source_entity_recall_pct": 0.0,
                "valid_metadata_json_pct": 0.0,
                "valid_entities_json_pct": 0.0,
                "entity_count_mismatch_chunks": 0,
                "avg_positive_flags_per_chunk": 0.0,
            }

        entity_counts = pd.to_numeric(df.get("entity_count", pd.Series(index=df.index)), errors="coerce").fillna(0)
        quality = pd.to_numeric(df.get("clinical_quality_score", pd.Series(index=df.index)), errors="coerce").fillna(0)

        valid_metadata = 0
        valid_entities = 0
        entity_count_mismatches = 0
        assigned_entity_keys: Set[Tuple[str, str, int, int]] = set()
        source_element_rows = 0
        positive_flag_counts = []

        for row in df.to_dict("records"):
            metadata = safe_json_loads(row.get("metadata_json"))
            valid_metadata += int(isinstance(metadata, dict) and isinstance(metadata.get("clinical"), dict))

            entities = safe_json_loads(row.get("entities_json"))
            if isinstance(entities, list):
                valid_entities += 1
                entity_count_mismatches += int(int(row.get("entity_count") or 0) != len(entities))
                assigned_entity_keys.update(entity_key(e) for e in entities if isinstance(e, dict))
            else:
                entity_count_mismatches += int(int(row.get("entity_count") or 0) != 0)

            source_element_rows += int(len(parse_id_list(row.get("source_element_ids_json"))) > 0)
            positive_flag_counts.append(sum(int(bool(row.get(flag))) for flag in self.FLAG_COLUMNS))

        return {
            "strategy": strategy,
            "chunk_count": int(len(df)),
            "section_coverage_pct": float(df["primary_section"].notna().mean() * 100) if "primary_section" in df.columns else 0.0,
            "source_element_coverage_pct": float(source_element_rows / max(1, len(df)) * 100),
            "entity_chunk_coverage_pct": float((entity_counts > 0).mean() * 100),
            "avg_entity_count": float(entity_counts.mean()),
            "avg_quality_score": float(quality.mean()),
            "source_entity_recall_pct": float(len(self.source_entity_keys & assigned_entity_keys) / max(1, len(self.source_entity_keys)) * 100),
            "valid_metadata_json_pct": float(valid_metadata / max(1, len(df)) * 100),
            "valid_entities_json_pct": float(valid_entities / max(1, len(df)) * 100),
            "entity_count_mismatch_chunks": int(entity_count_mismatches),
            "avg_positive_flags_per_chunk": float(sum(positive_flag_counts) / max(1, len(positive_flag_counts))),
        }

    def _source_entity_keys(self) -> Set[Tuple[str, str, int, int]]:
        if self.clinical_entities.empty:
            return set()

        out = set()
        for row in self.clinical_entities.to_dict("records"):
            out.add(entity_key(row))
        return out

    def _element_flags_by_id(self) -> Dict[int, Dict[str, bool]]:
        out: Dict[int, Dict[str, bool]] = defaultdict(dict)
        if self.clinical_element_metadata.empty or "element_id" not in self.clinical_element_metadata.columns:
            return out

        for row in self.clinical_element_metadata.to_dict("records"):
            try:
                element_id = int(row.get("element_id"))
            except Exception:
                continue

            for flag, source_cols in self.ELEMENT_FLAG_COLUMNS.items():
                out[element_id][flag] = any(bool(row.get(col)) for col in source_cols)

        return out

    def _expected_flag(self, row: Dict[str, Any], flag: str) -> bool:
        entities = safe_json_loads(row.get("entities_json"))
        if isinstance(entities, list):
            entity_types = {str(e.get("type") or e.get("entity_type")) for e in entities if isinstance(e, dict)}
            if entity_types & self.ENTITY_FLAG_TYPES.get(flag, set()):
                return True
            if flag == "contains_negation" and any(bool(e.get("negated") or e.get("is_negated")) for e in entities if isinstance(e, dict)):
                return True

        for element_id in parse_id_list(row.get("source_element_ids_json")):
            if self.element_flags_by_id.get(element_id, {}).get(flag):
                return True

        return False

    def _suspicious_reasons(self, row: Dict[str, Any]) -> List[str]:
        reasons = []

        source_element_ids = parse_id_list(row.get("source_element_ids_json"))
        entities = safe_json_loads(row.get("entities_json"))
        entity_count = int(row.get("entity_count") or 0)
        quality = float(row.get("clinical_quality_score") or 0.0)
        overlap_line_count = int(row.get("overlap_line_count") or 0)

        if not source_element_ids:
            reasons.append("no source elements")
        if overlap_line_count == 0:
            reasons.append("no overlapping lines")
        if entity_count > 0 and not isinstance(entities, list):
            reasons.append("entity_count without valid entities_json")
        if isinstance(entities, list) and entity_count != len(entities):
            reasons.append("entity_count does not match entities_json")
        if quality >= 0.65 and not row.get("primary_section") and entity_count == 0:
            reasons.append("high quality score without section or entities")

        flag_counter = Counter()
        for flag in self.FLAG_COLUMNS:
            if bool(row.get(flag)) != self._expected_flag(row, flag):
                flag_counter[flag] += 1
        if flag_counter:
            reasons.append("flag mismatches: " + ", ".join(sorted(flag_counter)))

        return reasons

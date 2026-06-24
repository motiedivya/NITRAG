from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _cap_figsize(w: float, h: float, max_w: float = 15.0, max_h: float = 12.0):
    return (min(max_w, w), min(max_h, h))


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

    Reads rag_store/<doc_id>/chunks_enriched/*.parquet and checks whether the
    enrichment stage added complete, internally consistent, and source-backed
    metadata.

    Metrics computed
    ────────────────
    Schema completeness  : per-column non-null coverage %
    Enrichment quality   : section/element/entity coverage, quality score stats,
                           source entity recall
    Flag consistency     : actual vs expected clinical flags, mismatch rates
    Entity distribution  : per-type entity counts across strategies
    Quality distribution : min/median/max/std of clinical_quality_score
    Suspicious chunks    : no source elements, count mismatches, flag mismatches

    Plots generated
    ───────────────
    01 quality score distribution (violin + box per strategy)
    02 enrichment metrics grouped bar (entity coverage, section coverage, recall)
    03 clinical flag prevalence heatmap (flag × strategy)
    04 schema completeness heatmap (column × strategy)
    05 entity type distribution stacked bar (strategy × entity type %)
    06 quality score vs entity count scatter
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
        self.plots_dir   = self.report_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.clinical_entities         = read_parquet_df(self.document_dir / "clinical_entities.parquet")
        self.clinical_element_metadata = read_parquet_df(self.document_dir / "clinical_element_metadata.parquet")

        self.source_entity_keys = self._source_entity_keys()
        self.element_flags_by_id = self._element_flags_by_id()

    # ------------------------------------------------------------------
    # Listing / loading
    # ------------------------------------------------------------------

    def list_strategies(self) -> List[str]:
        if not self.enriched_chunk_dir.exists():
            return []
        return sorted(p.stem for p in self.enriched_chunk_dir.glob("*.parquet"))

    def load_chunks(self, strategy: str) -> pd.DataFrame:
        path = self.enriched_chunk_dir / f"{strategy}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return read_parquet_df(path)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_schema_completeness(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            for column in self.REQUIRED_COLUMNS:
                present = column in df.columns
                non_null = int(df[column].notna().sum()) if present and len(df) > 0 else 0
                rows.append({
                    "strategy": strategy,
                    "column": column,
                    "present": bool(present),
                    "non_null_count": non_null,
                    "chunk_count": int(len(df)),
                    "coverage_pct": float(non_null / len(df) * 100) if len(df) > 0 else 0.0,
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

    def compute_flag_prevalence(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns (strategy, flag, prevalence_pct) rows — used for the heatmap."""
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            n = max(1, len(df))
            for flag in self.FLAG_COLUMNS:
                if flag in df.columns:
                    pct = float(df[flag].fillna(False).astype(bool).mean() * 100)
                else:
                    pct = 0.0
                rows.append({"strategy": strategy, "flag": flag, "prevalence_pct": pct})
        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "flag_prevalence.csv")
        return out

    def compute_entity_type_distribution(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns (strategy, entity_type, count) rows."""
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            type_counts: Counter = Counter()
            for row in df.to_dict("records"):
                entities = safe_json_loads(row.get("entities_json"))
                if isinstance(entities, list):
                    for e in entities:
                        if isinstance(e, dict):
                            etype = str(e.get("type") or e.get("entity_type") or "unknown")
                            type_counts[etype] += 1
            for etype, cnt in type_counts.items():
                rows.append({"strategy": strategy, "entity_type": etype, "count": cnt})
        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "entity_type_distribution.csv")
        return out

    def compute_flag_consistency(self, strategies: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        for strategy in strategies or self.list_strategies():
            df = self.load_chunks(strategy)
            for flag in self.FLAG_COLUMNS:
                mismatches = 0
                positives  = 0
                expected_positives = 0
                for row in df.to_dict("records"):
                    actual   = bool(row.get(flag))
                    expected = self._expected_flag(row, flag)
                    positives          += int(actual)
                    expected_positives += int(expected)
                    mismatches         += int(actual != expected)
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

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_all(
        self,
        metrics_df: Optional[pd.DataFrame] = None,
        schema_df: Optional[pd.DataFrame] = None,
        prevalence_df: Optional[pd.DataFrame] = None,
        entity_type_df: Optional[pd.DataFrame] = None,
        strategies: Optional[List[str]] = None,
    ) -> List[Path]:
        strategies = strategies or self.list_strategies()
        if not strategies:
            return []

        if metrics_df is None:
            metrics_df = self.compute_enrichment_metrics(strategies)
        if schema_df is None:
            schema_df = self.compute_schema_completeness(strategies)
        if prevalence_df is None:
            prevalence_df = self.compute_flag_prevalence(strategies)
        if entity_type_df is None:
            entity_type_df = self.compute_entity_type_distribution(strategies)

        paths: List[Path] = []

        # 01 — quality score distribution (violin per strategy)
        paths.extend(self._plot_quality_violin(strategies))

        # 02 — enrichment metrics grouped bar
        if not metrics_df.empty:
            paths.extend(self._plot_enrichment_bar(metrics_df))

        # 03 — clinical flag prevalence heatmap
        if not prevalence_df.empty:
            paths.extend(self._plot_flag_heatmap(prevalence_df))

        # 04 — schema completeness heatmap
        if not schema_df.empty:
            paths.extend(self._plot_schema_heatmap(schema_df))

        # 05 — entity type distribution stacked bar
        if not entity_type_df.empty:
            paths.extend(self._plot_entity_type_bar(entity_type_df))

        # 06 — quality score vs entity count scatter
        if not metrics_df.empty:
            paths.extend(self._plot_quality_vs_entity_scatter(metrics_df))

        return paths

    def _plot_quality_violin(self, strategies: List[str]) -> List[Path]:
        data   = []
        labels = []
        for s in strategies:
            try:
                df = self.load_chunks(s)
                if "clinical_quality_score" in df.columns:
                    vals = pd.to_numeric(df["clinical_quality_score"], errors="coerce").dropna().to_numpy(float)
                    if len(vals) > 0:
                        data.append(vals)
                        labels.append(s)
            except Exception:
                pass

        if not data:
            return []

        path = self.plots_dir / "01_quality_score_distribution.png"
        fig, axes = plt.subplots(1, 2, figsize=_cap_figsize(14, max(5, len(labels) * 0.5)))

        # Violin
        ax = axes[0]
        parts = ax.violinplot(data, positions=range(1, len(data) + 1), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#0f766e")
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Clinical quality score")
        ax.set_title("Quality score distribution (violin)")

        # Box
        ax = axes[1]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Clinical quality score")
        ax.set_title("Quality score distribution (box)")

        fig.suptitle("Chunk clinical quality score by strategy", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_enrichment_bar(self, metrics_df: pd.DataFrame) -> List[Path]:
        metrics = [
            ("entity_chunk_coverage_pct", "Entity-bearing chunks %"),
            ("section_coverage_pct",       "Chunks with section %"),
            ("source_entity_recall_pct",   "Source entity recall %"),
            ("avg_quality_score",          "Avg quality score"),
        ]
        path = self.plots_dir / "02_enrichment_metrics_bar.png"
        valid = [(col, lbl) for col, lbl in metrics if col in metrics_df.columns]
        if not valid:
            return []

        ncols = 2
        nrows = (len(valid) + 1) // 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=_cap_figsize(14, max(4, nrows * 3.5)))
        axes_flat = np.asarray(axes).reshape(-1)

        for i, (col, lbl) in enumerate(valid):
            ax = axes_flat[i]
            d = metrics_df.sort_values(col, ascending=True)
            scale = 100.0 if col != "avg_quality_score" else 1.0
            ax.barh(d["strategy"], d[col] / scale, color="#2563eb")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Fraction" if col != "avg_quality_score" else "Score")
            ax.set_title(lbl)

        for ax in axes_flat[len(valid):]:
            ax.axis("off")

        fig.suptitle("Enrichment metric comparison by strategy")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_flag_heatmap(self, prevalence_df: pd.DataFrame) -> List[Path]:
        pivot = prevalence_df.pivot_table(
            index="strategy", columns="flag", values="prevalence_pct", aggfunc="mean", fill_value=0,
        )
        if pivot.empty:
            return []

        path = self.plots_dir / "03_flag_prevalence_heatmap.png"
        fig, ax = plt.subplots(figsize=_cap_figsize(max(11, pivot.shape[1] * 1.1), max(4.5, pivot.shape[0] * 0.5)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        fig.colorbar(im, ax=ax, label="% of chunks with flag=True")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title("Clinical flag prevalence per strategy (% of chunks)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if pivot.values[i, j] > 60 else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_schema_heatmap(self, schema_df: pd.DataFrame) -> List[Path]:
        pivot = schema_df.pivot_table(
            index="strategy", columns="column", values="coverage_pct", aggfunc="mean", fill_value=0,
        )
        if pivot.empty:
            return []

        path = self.plots_dir / "04_schema_completeness_heatmap.png"
        fig, ax = plt.subplots(figsize=_cap_figsize(max(14, pivot.shape[1] * 0.8), max(4.5, pivot.shape[0] * 0.5)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        fig.colorbar(im, ax=ax, label="Non-null coverage %")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title("Schema completeness: % non-null per column per strategy")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if pivot.values[i, j] < 40 else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_entity_type_bar(self, entity_type_df: pd.DataFrame) -> List[Path]:
        if entity_type_df.empty:
            return []

        pivot = entity_type_df.pivot_table(
            index="strategy", columns="entity_type", values="count", aggfunc="sum", fill_value=0,
        )
        totals = pivot.sum(axis=1).replace(0, 1)
        pivot_pct = pivot.div(totals, axis=0) * 100

        path = self.plots_dir / "05_entity_type_distribution.png"
        cmap = plt.get_cmap("tab20", len(pivot_pct.columns))
        fig, ax = plt.subplots(figsize=_cap_figsize(11, max(4.5, len(pivot_pct) * 0.5)))
        left = np.zeros(len(pivot_pct))
        for i, col in enumerate(pivot_pct.columns):
            ax.barh(pivot_pct.index, pivot_pct[col], left=left, color=cmap(i), label=col)
            left += pivot_pct[col].values
        ax.set_xlabel("% of total entities")
        ax.set_title("Entity type distribution per strategy")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_quality_vs_entity_scatter(self, metrics_df: pd.DataFrame) -> List[Path]:
        needed = {"avg_quality_score", "avg_entity_count"}
        if not needed.issubset(metrics_df.columns):
            return []

        path = self.plots_dir / "06_quality_vs_entity_count_scatter.png"
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(
            metrics_df["avg_entity_count"],
            metrics_df["avg_quality_score"],
            c=metrics_df.get("entity_chunk_coverage_pct", pd.Series([50] * len(metrics_df))),
            cmap="viridis", s=80,
        )
        plt.colorbar(sc, ax=ax, label="Entity-bearing chunk %")
        for _, row in metrics_df.iterrows():
            ax.annotate(
                row["strategy"],
                (row["avg_entity_count"], row["avg_quality_score"]),
                fontsize=7, ha="left", xytext=(4, 4), textcoords="offset points",
            )
        ax.set_xlabel("Avg entities per chunk")
        ax.set_ylabel("Avg clinical quality score")
        ax.set_title("Quality score vs entity density\n(colour = entity-bearing chunk %)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        strategies     = strategies or self.list_strategies()
        schema_df      = self.compute_schema_completeness(strategies)
        metrics_df     = self.compute_enrichment_metrics(strategies)
        flags_df       = self.compute_flag_consistency(strategies)
        prevalence_df  = self.compute_flag_prevalence(strategies)
        entity_type_df = self.compute_entity_type_distribution(strategies)
        suspicious_df  = self.find_suspicious_chunks(strategies)
        plot_paths     = self.plot_all(
            metrics_df=metrics_df,
            schema_df=schema_df,
            prevalence_df=prevalence_df,
            entity_type_df=entity_type_df,
            strategies=strategies,
        )

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "plots_dir": str(self.plots_dir),
            "strategy_count": int(len(strategies)),
            "source_entity_count": int(len(self.source_entity_keys)),
            "metric_files": [
                str(self.metrics_dir / "schema_completeness.csv"),
                str(self.metrics_dir / "enrichment_metrics.csv"),
                str(self.metrics_dir / "flag_consistency.csv"),
                str(self.metrics_dir / "flag_prevalence.csv"),
                str(self.metrics_dir / "entity_type_distribution.csv"),
                str(self.metrics_dir / "suspicious_chunks.csv"),
            ],
            "plots": [str(p) for p in plot_paths],
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

    # ------------------------------------------------------------------
    # Internal metric helpers
    # ------------------------------------------------------------------

    def _strategy_metrics(self, strategy: str, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {
                "strategy": strategy,
                "chunk_count": 0,
                "section_coverage_pct": 0.0,
                "source_element_coverage_pct": 0.0,
                "entity_chunk_coverage_pct": 0.0,
                "avg_entity_count": 0.0,
                "median_entity_count": 0.0,
                "avg_quality_score": 0.0,
                "median_quality_score": 0.0,
                "min_quality_score": 0.0,
                "max_quality_score": 0.0,
                "std_quality_score": 0.0,
                "source_entity_recall_pct": 0.0,
                "valid_metadata_json_pct": 0.0,
                "valid_entities_json_pct": 0.0,
                "entity_count_mismatch_chunks": 0,
                "avg_positive_flags_per_chunk": 0.0,
            }

        entity_counts = pd.to_numeric(df.get("entity_count", pd.Series(index=df.index)), errors="coerce").fillna(0)
        quality       = pd.to_numeric(df.get("clinical_quality_score", pd.Series(index=df.index)), errors="coerce").fillna(0)

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
            "median_entity_count": float(entity_counts.median()),
            "avg_quality_score": float(quality.mean()),
            "median_quality_score": float(quality.median()),
            "min_quality_score": float(quality.min()),
            "max_quality_score": float(quality.max()),
            "std_quality_score": float(quality.std(ddof=0)),
            "source_entity_recall_pct": float(len(self.source_entity_keys & assigned_entity_keys) / max(1, len(self.source_entity_keys)) * 100),
            "valid_metadata_json_pct": float(valid_metadata / max(1, len(df)) * 100),
            "valid_entities_json_pct": float(valid_entities / max(1, len(df)) * 100),
            "entity_count_mismatch_chunks": int(entity_count_mismatches),
            "avg_positive_flags_per_chunk": float(sum(positive_flag_counts) / max(1, len(positive_flag_counts))),
        }

    def _source_entity_keys(self) -> Set[Tuple[str, str, int, int]]:
        if self.clinical_entities.empty:
            return set()
        return {entity_key(row) for row in self.clinical_entities.to_dict("records")}

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
        entities      = safe_json_loads(row.get("entities_json"))
        entity_count  = int(row.get("entity_count") or 0)
        quality       = float(row.get("clinical_quality_score") or 0.0)
        overlap_count = int(row.get("overlap_line_count") or 0)

        if not source_element_ids:
            reasons.append("no source elements")
        if overlap_count == 0:
            reasons.append("no overlapping lines")
        if entity_count > 0 and not isinstance(entities, list):
            reasons.append("entity_count without valid entities_json")
        if isinstance(entities, list) and entity_count != len(entities):
            reasons.append("entity_count does not match entities_json")
        if quality >= 0.65 and not row.get("primary_section") and entity_count == 0:
            reasons.append("high quality score without section or entities")

        flag_counter: Counter = Counter()
        for flag in self.FLAG_COLUMNS:
            if bool(row.get(flag)) != self._expected_flag(row, flag):
                flag_counter[flag] += 1
        if flag_counter:
            reasons.append("flag mismatches: " + ", ".join(sorted(flag_counter)))

        return reasons

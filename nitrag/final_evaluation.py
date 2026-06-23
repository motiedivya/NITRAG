from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .rag_diagnostics_manager import RAGDiagnosticsManager
from .reranking_evaluation import RerankingEvaluationManager


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_if_exists(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo = values.min()
    hi = values.max()
    if hi <= lo:
        out = pd.Series([1.0 if hi > 0 else 0.0] * len(values), index=values.index)
    else:
        out = (values - lo) / (hi - lo)
    return 1.0 - out if invert else out


class FinalEvaluationManager:
    """
    Final cross-stage evaluator and pipeline ranker.

    This is the one place that compares full RAG pipelines. It combines:
      - chunking evaluation
      - chunk metadata enrichment evaluation
      - indexing evaluation
      - retrieval benchmark
      - reranking benchmark

    The resulting pipeline key is:
      chunk_strategy + retriever + reranker
    where reranker="baseline" means retrieval-only without post-reranking.
    """

    def __init__(
        self,
        store,
        retriever_manager=None,
        reranker_manager=None,
        report_dir: Optional[Union[str, Path]] = None,
    ):
        self.store = store
        self.retriever_manager = retriever_manager
        self.reranker_manager = reranker_manager
        self.document_dir = Path(store.paths.document_dir)
        self.report_dir = ensure_dir(report_dir or self.document_dir / "reports" / "final_evaluation")
        self.metrics_dir = ensure_dir(self.report_dir / "metrics")
        self.plots_dir = ensure_dir(self.report_dir / "plots")

    def generate_report(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: List[str],
        chunk_strategy_names: List[str],
        reranker_names: Optional[List[str]] = None,
        candidate_k: int = 20,
        top_k: int = 10,
        run_retrieval_benchmark: bool = True,
        run_reranking_benchmark: bool = True,
    ) -> Dict[str, Any]:
        retrieval_df = self._get_retrieval_benchmark(
            query_suite=query_suite,
            retriever_names=retriever_names,
            chunk_strategy_names=chunk_strategy_names,
            top_k=top_k,
            run=run_retrieval_benchmark,
        )
        reranking_df = self._get_reranking_benchmark(
            query_suite=query_suite,
            retriever_names=retriever_names,
            chunk_strategy_names=chunk_strategy_names,
            reranker_names=reranker_names,
            candidate_k=candidate_k,
            top_k=top_k,
            run=run_reranking_benchmark,
        )

        component_df = self.compute_component_scorecard(chunk_strategy_names)
        pipeline_df = self.compute_pipeline_rankings(
            retrieval_df=retrieval_df,
            reranking_df=reranking_df,
            component_df=component_df,
        )
        stage_summary_df = self.compute_stage_summary(component_df, retrieval_df, reranking_df, pipeline_df)
        plot_paths = self.plot_all(component_df, pipeline_df, stage_summary_df)

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "plots_dir": str(self.plots_dir),
            "component_rows": int(len(component_df)),
            "retrieval_rows": int(len(retrieval_df)),
            "reranking_rows": int(len(reranking_df)),
            "pipeline_rows": int(len(pipeline_df)),
            "top_pipeline": pipeline_df.iloc[0].to_dict() if not pipeline_df.empty else None,
            "metric_files": [
                str(self.metrics_dir / "component_scorecard.csv"),
                str(self.metrics_dir / "retrieval_benchmark.csv"),
                str(self.metrics_dir / "reranking_benchmark.csv"),
                str(self.metrics_dir / "pipeline_rankings.csv"),
                str(self.metrics_dir / "stage_summary.csv"),
            ],
            "plots": [str(p) for p in plot_paths],
        }
        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return summary

    def compute_component_scorecard(self, chunk_strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
        chunk_df = read_csv_if_exists(self.document_dir / "reports" / "chunking_evaluation" / "metrics" / "chunking_metrics.csv")
        enrich_df = read_csv_if_exists(self.document_dir / "reports" / "chunk_metadata_enrichment_evaluation" / "metrics" / "enrichment_metrics.csv")
        index_df = read_csv_if_exists(self.document_dir / "reports" / "indexing_evaluation" / "metrics" / "index_scorecard.csv")

        strategies = chunk_strategy_names or sorted(set(chunk_df.get("strategy", pd.Series(dtype=str)).dropna().tolist()))
        rows = []

        index_by_strategy = pd.DataFrame()
        if not index_df.empty and {"chunk_strategy", "health_score"}.issubset(index_df.columns):
            index_by_strategy = index_df.groupby("chunk_strategy").agg(
                index_health_score=("health_score", "mean"),
                index_error_count=("error_count", "sum"),
                index_warning_count=("warning_count", "sum"),
                index_info_count=("info_count", "sum"),
            ).reset_index()

        for strategy in strategies:
            row = {"chunk_strategy": strategy}

            c = chunk_df[chunk_df.get("strategy") == strategy] if not chunk_df.empty and "strategy" in chunk_df.columns else pd.DataFrame()
            if not c.empty:
                c0 = c.iloc[0]
                coverage = float(c0.get("coverage_pct", 0.0)) / 100.0
                redundancy = float(c0.get("redundancy_factor", 0.0))
                overlap = float(c0.get("overlap_pct_of_chunk_tokens", 0.0)) / 100.0
                page_cross = float(c0.get("page_crossing_pct", 0.0)) / 100.0
                median_tokens = float(c0.get("median_tokens", 0.0))
                length_score = 1.0 if 128 <= median_tokens <= 1200 else max(0.0, min(median_tokens / 128.0, 1200.0 / max(1.0, median_tokens)))
                chunking_score = (
                    0.40 * coverage
                    + 0.20 * max(0.0, 1.0 - max(0.0, redundancy - 1.0))
                    + 0.15 * max(0.0, 1.0 - overlap)
                    + 0.10 * max(0.0, 1.0 - page_cross)
                    + 0.15 * length_score
                )
                row.update({
                    "chunking_score": round(chunking_score, 6),
                    "chunk_coverage_pct": c0.get("coverage_pct", 0.0),
                    "chunk_redundancy_factor": redundancy,
                    "chunk_median_tokens": median_tokens,
                })
            else:
                row.update({"chunking_score": 0.0, "chunk_coverage_pct": 0.0, "chunk_redundancy_factor": 0.0, "chunk_median_tokens": 0.0})

            e = enrich_df[enrich_df.get("strategy") == strategy] if not enrich_df.empty and "strategy" in enrich_df.columns else pd.DataFrame()
            if not e.empty:
                e0 = e.iloc[0]
                enrichment_score = (
                    0.25 * float(e0.get("section_coverage_pct", 0.0)) / 100.0
                    + 0.20 * float(e0.get("source_element_coverage_pct", 0.0)) / 100.0
                    + 0.20 * float(e0.get("entity_chunk_coverage_pct", 0.0)) / 100.0
                    + 0.20 * float(e0.get("avg_quality_score", 0.0))
                    + 0.15 * float(e0.get("source_entity_recall_pct", 0.0)) / 100.0
                )
                row.update({
                    "enrichment_score": round(enrichment_score, 6),
                    "section_coverage_pct": e0.get("section_coverage_pct", 0.0),
                    "entity_chunk_coverage_pct": e0.get("entity_chunk_coverage_pct", 0.0),
                    "source_entity_recall_pct": e0.get("source_entity_recall_pct", 0.0),
                    "avg_clinical_quality_score": e0.get("avg_quality_score", 0.0),
                })
            else:
                row.update({
                    "enrichment_score": 0.0,
                    "section_coverage_pct": 0.0,
                    "entity_chunk_coverage_pct": 0.0,
                    "source_entity_recall_pct": 0.0,
                    "avg_clinical_quality_score": 0.0,
                })

            i = index_by_strategy[index_by_strategy["chunk_strategy"] == strategy] if not index_by_strategy.empty else pd.DataFrame()
            if not i.empty:
                i0 = i.iloc[0]
                row.update({
                    "indexing_score": float(i0.get("index_health_score", 0.0)),
                    "index_error_count": int(i0.get("index_error_count", 0)),
                    "index_warning_count": int(i0.get("index_warning_count", 0)),
                    "index_info_count": int(i0.get("index_info_count", 0)),
                })
            else:
                row.update({"indexing_score": 0.0, "index_error_count": 0, "index_warning_count": 0, "index_info_count": 0})

            row["foundation_score"] = round(
                0.35 * row["chunking_score"]
                + 0.35 * row["enrichment_score"]
                + 0.30 * row["indexing_score"],
                6,
            )
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("foundation_score", ascending=False)
        save_df(out, self.metrics_dir / "component_scorecard.csv")
        return out

    def compute_pipeline_rankings(
        self,
        retrieval_df: pd.DataFrame,
        reranking_df: pd.DataFrame,
        component_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        component_lookup = component_df.set_index("chunk_strategy").to_dict("index") if not component_df.empty else {}

        retrieval_clean = retrieval_df[retrieval_df.get("error").isna()].copy() if not retrieval_df.empty and "error" in retrieval_df.columns else retrieval_df.copy()
        if not retrieval_clean.empty:
            retrieval_grouped = retrieval_clean.groupby(["chunk_strategy", "retriever"], dropna=False).agg(
                runs=("query", "size"),
                retrieval_result_count=("result_count", "mean"),
                retrieval_keyword_hit_rate=("keyword_hit_rate", "mean"),
                retrieval_expected_page_hit_at_k=("expected_page_hit_at_k", "mean"),
                retrieval_mrr_page=("mrr_page", "mean"),
                retrieval_duplicate_text_ratio=("duplicate_text_ratio", "mean"),
                retrieval_page_diversity=("page_diversity", "mean"),
                retrieval_latency_ms=("latency_ms", "mean"),
            ).reset_index()
            for r in retrieval_grouped.to_dict("records"):
                base = component_lookup.get(r["chunk_strategy"], {})
                row = dict(r)
                row["reranker"] = "none"
                row.update(self._component_fields(base))
                row.update(self._pipeline_scores(row, prefix="retrieval"))
                rows.append(row)

        rerank_clean = reranking_df[reranking_df.get("error").isna()].copy() if not reranking_df.empty and "error" in reranking_df.columns else reranking_df.copy()
        if not rerank_clean.empty:
            rerank_grouped = rerank_clean.groupby(["chunk_strategy", "retriever", "reranker"], dropna=False).agg(
                runs=("query", "size"),
                rerank_result_count=("result_count", "mean"),
                rerank_keyword_hit_rate=("keyword_hit_rate", "mean"),
                rerank_expected_page_hit_at_k=("expected_page_hit_at_k", "mean"),
                rerank_mrr_page=("mrr_page", "mean"),
                rerank_duplicate_text_ratio=("duplicate_text_ratio", "mean"),
                rerank_page_diversity=("page_diversity", "mean"),
                rerank_latency_ms=("rerank_latency_ms", "mean"),
                retrieval_latency_ms=("retrieval_latency_ms", "mean"),
                mean_abs_rank_delta=("mean_abs_rank_delta", "mean"),
                top_result_changed_rate=("top_result_changed", "mean"),
            ).reset_index()
            for r in rerank_grouped.to_dict("records"):
                base = component_lookup.get(r["chunk_strategy"], {})
                row = dict(r)
                row.update(self._component_fields(base))
                row.update(self._pipeline_scores(row, prefix="rerank"))
                rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            numeric_cols = [
                "retrieval_result_count",
                "retrieval_keyword_hit_rate",
                "retrieval_expected_page_hit_at_k",
                "retrieval_mrr_page",
                "retrieval_duplicate_text_ratio",
                "retrieval_page_diversity",
                "retrieval_latency_ms",
                "rerank_result_count",
                "rerank_keyword_hit_rate",
                "rerank_expected_page_hit_at_k",
                "rerank_mrr_page",
                "rerank_duplicate_text_ratio",
                "rerank_page_diversity",
                "rerank_latency_ms",
                "mean_abs_rank_delta",
                "top_result_changed_rate",
            ]
            for col in numeric_cols:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            out = out.sort_values(
                ["final_pipeline_score", "answer_quality_score", "foundation_score"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            out.insert(0, "pipeline_rank", range(1, len(out) + 1))

        save_df(out, self.metrics_dir / "pipeline_rankings.csv")
        return out

    def compute_stage_summary(
        self,
        component_df: pd.DataFrame,
        retrieval_df: pd.DataFrame,
        reranking_df: pd.DataFrame,
        pipeline_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        rows.append({"stage": "foundation", "rows": len(component_df), "best_score": component_df["foundation_score"].max() if not component_df.empty else 0.0})
        rows.append({"stage": "retrieval", "rows": len(retrieval_df), "best_score": retrieval_df["keyword_hit_rate"].max() if not retrieval_df.empty and "keyword_hit_rate" in retrieval_df.columns else 0.0})
        rows.append({"stage": "reranking", "rows": len(reranking_df), "best_score": reranking_df["keyword_hit_rate"].max() if not reranking_df.empty and "keyword_hit_rate" in reranking_df.columns else 0.0})
        rows.append({"stage": "full_pipeline", "rows": len(pipeline_df), "best_score": pipeline_df["final_pipeline_score"].max() if not pipeline_df.empty else 0.0})
        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "stage_summary.csv")
        return out

    def plot_all(self, component_df: pd.DataFrame, pipeline_df: pd.DataFrame, stage_summary_df: pd.DataFrame) -> List[Path]:
        paths = []
        if not component_df.empty:
            paths.append(self._plot_bar(component_df.head(20), "chunk_strategy", "foundation_score", "01_foundation_score.png", "Foundation score by chunk strategy"))
        if not pipeline_df.empty:
            top = pipeline_df.head(25).copy()
            top["pipeline"] = top["chunk_strategy"].astype(str) + " / " + top["retriever"].astype(str) + " / " + top["reranker"].astype(str)
            paths.append(self._plot_bar(top, "pipeline", "final_pipeline_score", "02_top_pipeline_rankings.png", "Top full pipeline rankings"))

            pivot = pipeline_df.pivot_table(
                index="chunk_strategy",
                columns="retriever",
                values="final_pipeline_score",
                aggfunc="max",
                fill_value=0,
            )
            path = self.plots_dir / "03_best_pipeline_score_heatmap.png"
            plt.figure(figsize=(max(10, pivot.shape[1] * 1.2), max(4.5, pivot.shape[0] * 0.45)))
            plt.imshow(pivot.values, aspect="auto", cmap="viridis")
            plt.colorbar(label="Best final score")
            plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
            plt.yticks(range(pivot.shape[0]), pivot.index)
            plt.title("Best pipeline score by chunk strategy and retriever")
            plt.tight_layout()
            plt.savefig(path, dpi=170)
            plt.close()
            paths.append(path)

        if not stage_summary_df.empty:
            paths.append(self._plot_bar(stage_summary_df, "stage", "best_score", "04_stage_best_scores.png", "Best score by stage"))

        return paths

    def _get_retrieval_benchmark(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: List[str],
        chunk_strategy_names: List[str],
        top_k: int,
        run: bool,
    ) -> pd.DataFrame:
        path = self.metrics_dir / "retrieval_benchmark.csv"
        if run:
            if self.retriever_manager is None:
                raise ValueError("retriever_manager is required to run retrieval benchmark")
            diag = RAGDiagnosticsManager(
                store=self.store,
                retriever_manager=self.retriever_manager,
                report_dir=self.report_dir / "retrieval_diagnostics",
                use_enriched_chunks=True,
            )
            df = diag.run_retrieval_benchmark(
                query_suite=query_suite,
                retriever_names=retriever_names,
                chunk_strategy_names=chunk_strategy_names,
                top_k=top_k,
            )
            diag.plot_retrieval_benchmark(df)
        else:
            df = read_csv_if_exists(path)
        save_df(df, path)
        return df

    def _get_reranking_benchmark(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: List[str],
        chunk_strategy_names: List[str],
        reranker_names: Optional[List[str]],
        candidate_k: int,
        top_k: int,
        run: bool,
    ) -> pd.DataFrame:
        path = self.metrics_dir / "reranking_benchmark.csv"
        if run:
            if self.retriever_manager is None or self.reranker_manager is None:
                raise ValueError("retriever_manager and reranker_manager are required to run reranking benchmark")
            rerank_eval = RerankingEvaluationManager(
                store=self.store,
                retriever_manager=self.retriever_manager,
                reranker_manager=self.reranker_manager,
                report_dir=self.report_dir / "reranking",
            )
            rerank_eval.generate_report(
                query_suite=query_suite,
                retriever_names=retriever_names,
                chunk_strategy_names=chunk_strategy_names,
                reranker_names=[r for r in (reranker_names or []) if r not in {"baseline", "none"}] or None,
                candidate_k=candidate_k,
                top_k=top_k,
            )
            df = read_csv_if_exists(rerank_eval.metrics_dir / "reranking_benchmark.csv")
        else:
            df = read_csv_if_exists(path)
        save_df(df, path)
        return df

    def _component_fields(self, base: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chunking_score": float(base.get("chunking_score", 0.0)),
            "enrichment_score": float(base.get("enrichment_score", 0.0)),
            "indexing_score": float(base.get("indexing_score", 0.0)),
            "foundation_score": float(base.get("foundation_score", 0.0)),
        }

    def _pipeline_scores(self, row: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        keyword = float(row.get(f"{prefix}_keyword_hit_rate", 0.0))
        page_hit = float(row.get(f"{prefix}_expected_page_hit_at_k", 0.0))
        mrr = float(row.get(f"{prefix}_mrr_page", 0.0))
        dup = float(row.get(f"{prefix}_duplicate_text_ratio", 0.0))
        diversity = min(1.0, float(row.get(f"{prefix}_page_diversity", 0.0)) / 5.0)
        latency = float(row.get("retrieval_latency_ms", 0.0)) + float(row.get("rerank_latency_ms", 0.0))
        latency_score = 1.0 / (1.0 + max(0.0, latency) / 100.0)

        answer_quality = (
            0.35 * keyword
            + 0.20 * page_hit
            + 0.20 * mrr
            + 0.15 * max(0.0, 1.0 - dup)
            + 0.10 * diversity
        )
        final_score = (
            0.25 * float(row.get("foundation_score", 0.0))
            + 0.65 * answer_quality
            + 0.10 * latency_score
        )
        return {
            "answer_quality_score": round(answer_quality, 6),
            "latency_score": round(latency_score, 6),
            "total_latency_ms": round(latency, 6),
            "final_pipeline_score": round(final_score, 6),
        }

    def _plot_bar(self, df: pd.DataFrame, label_col: str, value_col: str, filename: str, title: str) -> Path:
        d = df.sort_values(value_col, ascending=True)
        path = self.plots_dir / filename
        plt.figure(figsize=(11, max(4.5, len(d) * 0.35)))
        plt.barh(d[label_col], d[value_col], color="#2563eb")
        plt.xlabel(value_col)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()
        return path

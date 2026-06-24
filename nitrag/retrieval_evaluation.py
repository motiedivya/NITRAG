"""
RetrievalEvaluationManager
==========================

Benchmarks retriever strategies against a labelled query suite, measuring:

  Per-query metrics
  -----------------
  - MRR (page hit)           reciprocal rank of the first page-matching result
  - Precision@1/3/5          fraction of relevant results in first k slots
  - keyword_hit_rate         fraction of expected keywords found across top-k
  - result_count             how many results were returned
  - latency_ms               wall-clock time for the retrieve() call
  - error                    non-empty if the retriever raised

  Aggregate metrics (per retriever × optional chunk_strategy)
  -------------------------------------------------------------
  - avg_mrr, avg_p1/3/5, avg_keyword_hit_rate, avg_latency_ms
  - p50_latency_ms, p95_latency_ms
  - error_rate

  Cross-retriever agreement
  -------------------------
  - Jaccard similarity matrix: how often two retrievers return the same top-k chunks

Plots
-----
  01_mrr_bar.png                 MRR by retriever
  02_latency_cdf.png             CDF of latency per retriever
  03_precision_at_k_heatmap.png  P@1, P@3, P@5 heatmap (retriever × metric)
  04_agreement_matrix.png        Jaccard similarity matrix between retrievers
  05_keyword_hit_rate_bar.png    avg keyword hit rate by retriever
  06_result_count_boxplot.png    distribution of result_count per retriever
  07_mrr_vs_latency_scatter.png  avg_mrr vs avg_latency scatter (coloured by retriever)
  08_error_rate_bar.png          fraction of errored queries per retriever
"""
from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", str(text or "").lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def result_chunk_keys(results: List[Dict[str, Any]]) -> Set[Tuple[str, str, int]]:
    """Return a frozen set of (chunk_strategy_name, document_id, chunk_id) tuples."""
    return {
        (
            str(r.get("chunk_strategy_name", "")),
            str(r.get("document_id", "")),
            int(r.get("chunk_id") or 0),
        )
        for r in results
    }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RetrievalEvaluationManager:
    """
    Evaluate retriever strategies defined in a RetrieverManager.

    Parameters
    ----------
    store :
        DocumentStore (or compatible object with .paths.document_dir).
    retriever_manager :
        A RetrieverManager instance with strategies registered.
    report_dir :
        Where to write CSVs, JSON summary, and plots. Defaults to
        <document_dir>/reports/retrieval_evaluation.
    """

    def __init__(
        self,
        store,
        retriever_manager,
        report_dir: Optional[Union[str, Path]] = None,
    ):
        self.store = store
        self.retriever_manager = retriever_manager
        self.document_dir = Path(store.paths.document_dir)
        self.report_dir = ensure_dir(
            report_dir or self.document_dir / "reports" / "retrieval_evaluation"
        )
        self.metrics_dir = ensure_dir(self.report_dir / "metrics")
        self.plots_dir = ensure_dir(self.report_dir / "plots")

    # ------------------------------------------------------------------
    # Benchmark runner
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: Optional[List[str]] = None,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Run every retriever against every query and collect per-query metrics.

        Each element in query_suite should be a dict with keys:
          - query (str)                         the search query
          - expected_keywords (list[str])       optional keywords to check in results
          - expected_pages (list[int])          optional page numbers that should appear
          - description (str, optional)         human-readable label

        Returns a DataFrame with one row per (retriever, query).
        """
        if retriever_names is None:
            retriever_names = self.retriever_manager.list_retrievers()

        rows: List[Dict[str, Any]] = []

        for q_spec in query_suite:
            query = q_spec.get("query", "")
            expected_keywords: List[str] = q_spec.get("expected_keywords", [])
            expected_pages: Set[int] = set(q_spec.get("expected_pages", []))
            description: str = q_spec.get("description", query[:60])
            q_filters = q_spec.get("filters", filters)
            q_top_k = q_spec.get("top_k", top_k)
            q_strategy = q_spec.get("chunk_strategy_name", chunk_strategy_name)

            for retriever_name in retriever_names:
                row: Dict[str, Any] = {
                    "retriever_name": retriever_name,
                    "query": query,
                    "description": description,
                    "chunk_strategy_name": q_strategy or "",
                    "top_k": q_top_k,
                    "error": "",
                }

                t0 = time.perf_counter()
                try:
                    results = self.retriever_manager.retrieve(
                        retriever_name=retriever_name,
                        query=query,
                        chunk_strategy_name=q_strategy,
                        top_k=q_top_k,
                        filters=q_filters,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    metrics = self._score_results(
                        results=results,
                        expected_keywords=expected_keywords,
                        expected_pages=expected_pages,
                        top_k=q_top_k,
                    )
                    row.update(metrics)
                    row["latency_ms"] = round(latency_ms, 3)
                    row["result_count"] = len(results)
                    # store top-k chunk keys as JSON for agreement computation
                    row["_top_chunk_keys"] = json.dumps(
                        [list(k) for k in result_chunk_keys(results)]
                    )
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    error_msg = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    row["error"] = error_msg
                    row.update(self._empty_metrics())
                    row["latency_ms"] = round(latency_ms, 3)
                    row["result_count"] = 0
                    row["_top_chunk_keys"] = "[]"

                rows.append(row)

        df = pd.DataFrame(rows)
        if save and len(df):
            save_df(df.drop(columns=["_top_chunk_keys"], errors="ignore"),
                    self.metrics_dir / "benchmark.csv")
        return df

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compute_summary(
        self,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate per-query benchmark rows into per-retriever summary statistics.

        Returns a DataFrame with one row per retriever (and optionally per
        chunk_strategy_name if the benchmark covered multiple strategies).
        """
        if benchmark_df is None:
            p = self.metrics_dir / "benchmark.csv"
            if not p.exists():
                return pd.DataFrame()
            benchmark_df = pd.read_csv(p)

        if benchmark_df.empty:
            return pd.DataFrame()

        group_cols = ["retriever_name"]
        if benchmark_df["chunk_strategy_name"].nunique() > 1:
            group_cols.append("chunk_strategy_name")

        numeric_metrics = [
            "mrr_page", "precision_at_1", "precision_at_3", "precision_at_5",
            "keyword_hit_rate", "latency_ms", "result_count",
        ]
        agg_spec: Dict[str, Any] = {}
        for m in numeric_metrics:
            if m in benchmark_df.columns:
                agg_spec[m] = "mean"

        if "latency_ms" in benchmark_df.columns:
            agg_spec["latency_ms_p50"] = pd.NamedAgg(column="latency_ms", aggfunc=lambda x: float(np.percentile(x, 50)))
            agg_spec["latency_ms_p95"] = pd.NamedAgg(column="latency_ms", aggfunc=lambda x: float(np.percentile(x, 95)))

        if "error" in benchmark_df.columns:
            benchmark_df = benchmark_df.copy()
            benchmark_df["_has_error"] = benchmark_df["error"].fillna("").apply(lambda x: 1 if str(x).strip() else 0)
            agg_spec["error_rate"] = pd.NamedAgg(column="_has_error", aggfunc="mean")

        query_count = benchmark_df.groupby(group_cols)["query"].count().rename("query_count")

        # Separate NamedAgg from plain string aggs (pandas requires uniform style in one call)
        named_aggs = {k: v for k, v in agg_spec.items() if isinstance(v, pd.NamedAgg)}
        plain_aggs = {k: v for k, v in agg_spec.items() if not isinstance(v, pd.NamedAgg)}

        grouped = benchmark_df.groupby(group_cols)
        summary_parts = []

        if plain_aggs:
            part = grouped[list({v_: v_ for k_, v_ in plain_aggs.items() if v_ == "mean"
                                  for v_ in [k_]}.keys())].agg(plain_aggs)
            summary_parts.append(part)

        if named_aggs:
            part2 = grouped.agg(**named_aggs)
            summary_parts.append(part2)

        if not summary_parts:
            return pd.DataFrame()

        summary = pd.concat(summary_parts, axis=1)
        summary = summary.join(query_count)
        summary = summary.reset_index()

        # Rename mean columns with avg_ prefix for clarity
        rename_map = {m: f"avg_{m}" for m in plain_aggs if m != "result_count"}
        rename_map["result_count"] = "avg_result_count"
        summary = summary.rename(columns=rename_map)

        save_df(summary, self.metrics_dir / "summary.csv")
        return summary

    def compute_retriever_agreement(
        self,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute Jaccard similarity between every pair of retrievers.

        For each query, compare the set of top-k chunk keys each retriever
        returned and compute Jaccard.  Average across queries to get a per-pair
        similarity score.

        Returns a square DataFrame indexed and columned by retriever_name.
        """
        if "_top_chunk_keys" not in benchmark_df.columns:
            return pd.DataFrame()

        retriever_names = sorted(benchmark_df["retriever_name"].unique())
        queries = benchmark_df["query"].unique()

        # retriever → query → set of chunk keys
        result_sets: Dict[str, Dict[str, Set]] = {r: {} for r in retriever_names}
        for _, row in benchmark_df.iterrows():
            r_name = row["retriever_name"]
            q = row["query"]
            try:
                keys = frozenset(tuple(k) for k in json.loads(row["_top_chunk_keys"]))
            except Exception:
                keys = frozenset()
            result_sets[r_name][q] = keys

        n = len(retriever_names)
        matrix = np.zeros((n, n))

        for i, ra in enumerate(retriever_names):
            for j, rb in enumerate(retriever_names):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                scores = []
                for q in queries:
                    sa = result_sets[ra].get(q, frozenset())
                    sb = result_sets[rb].get(q, frozenset())
                    if not sa and not sb:
                        scores.append(1.0)
                    elif not sa or not sb:
                        scores.append(0.0)
                    else:
                        scores.append(len(sa & sb) / len(sa | sb))
                matrix[i][j] = float(np.mean(scores)) if scores else 0.0

        agreement_df = pd.DataFrame(matrix, index=retriever_names, columns=retriever_names)
        agreement_df.to_csv(self.metrics_dir / "retriever_agreement.csv")
        return agreement_df

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        query_suite: Optional[List[Dict[str, Any]]] = None,
        retriever_names: Optional[List[str]] = None,
        chunk_strategy_name: Optional[str] = None,
        top_k: int = 10,
        benchmark_df: Optional[pd.DataFrame] = None,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline and save everything to report_dir.

        If benchmark_df is given, skip the benchmark run and use it directly.
        """
        if benchmark_df is None:
            if query_suite is None:
                raise ValueError("Provide either query_suite or benchmark_df.")
            benchmark_df = self.run_benchmark(
                query_suite=query_suite,
                retriever_names=retriever_names,
                chunk_strategy_name=chunk_strategy_name,
                top_k=top_k,
                save=True,
            )

        summary_df = self.compute_summary(benchmark_df)
        agreement_df = self.compute_retriever_agreement(benchmark_df)

        plots = []
        if save_plots and not benchmark_df.empty:
            plots = self.plot_all(benchmark_df, summary_df, agreement_df)

        report: Dict[str, Any] = {
            "n_queries": int(benchmark_df["query"].nunique()) if not benchmark_df.empty else 0,
            "n_retrievers": int(benchmark_df["retriever_name"].nunique()) if not benchmark_df.empty else 0,
            "plots": [str(p) for p in plots],
        }

        if not summary_df.empty:
            best_mrr_col = "avg_mrr_page" if "avg_mrr_page" in summary_df.columns else None
            if best_mrr_col and best_mrr_col in summary_df.columns:
                best_idx = summary_df[best_mrr_col].idxmax()
                report["best_retriever_by_mrr"] = str(summary_df.loc[best_idx, "retriever_name"])
                report["best_mrr"] = float(summary_df.loc[best_idx, best_mrr_col])

            if "avg_latency_ms" in summary_df.columns:
                report["median_latency_ms"] = float(summary_df["avg_latency_ms"].median())
            if "error_rate" in summary_df.columns:
                report["overall_error_rate"] = float(summary_df["error_rate"].mean())

        (self.report_dir / "summary.json").write_text(json.dumps(report, indent=2))
        return report

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_all(
        self,
        benchmark_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        agreement_df: Optional[pd.DataFrame] = None,
    ) -> List[Path]:
        plots: List[Path] = []
        try:
            plots += self._plot_mrr_bar(summary_df)
        except Exception:
            pass
        try:
            plots += self._plot_latency_cdf(benchmark_df)
        except Exception:
            pass
        try:
            plots += self._plot_precision_at_k_heatmap(summary_df)
        except Exception:
            pass
        if agreement_df is not None and not agreement_df.empty:
            try:
                plots += self._plot_agreement_matrix(agreement_df)
            except Exception:
                pass
        try:
            plots += self._plot_keyword_hit_rate_bar(summary_df)
        except Exception:
            pass
        try:
            plots += self._plot_result_count_boxplot(benchmark_df)
        except Exception:
            pass
        try:
            plots += self._plot_mrr_vs_latency_scatter(summary_df)
        except Exception:
            pass
        try:
            plots += self._plot_error_rate_bar(summary_df)
        except Exception:
            pass
        return plots

    def _plot_mrr_bar(self, summary_df: pd.DataFrame) -> List[Path]:
        col = "avg_mrr_page"
        if col not in summary_df.columns or summary_df.empty:
            return []
        df = summary_df.sort_values(col, ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6 + 2), 5))
        colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(df)))
        ax.barh(df["retriever_name"], df[col], color=colors[::-1])
        ax.set_xlabel("Mean Reciprocal Rank (MRR)")
        ax.set_title("MRR by Retriever")
        ax.set_xlim(0, 1.05)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row[col] + 0.01, i, f"{row[col]:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        path = self.plots_dir / "01_mrr_bar.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_latency_cdf(self, benchmark_df: pd.DataFrame) -> List[Path]:
        if "latency_ms" not in benchmark_df.columns or benchmark_df.empty:
            return []
        fig, ax = plt.subplots(figsize=(10, 6))
        retrievers = sorted(benchmark_df["retriever_name"].unique())
        cmap = plt.cm.tab10
        for i, r_name in enumerate(retrievers[:10]):
            data = np.sort(benchmark_df.loc[
                (benchmark_df["retriever_name"] == r_name) & (benchmark_df["latency_ms"] > 0),
                "latency_ms"
            ].dropna().values)
            if not len(data):
                continue
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, label=r_name, color=cmap(i / 10), linewidth=1.5)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Latency CDF by Retriever")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xscale("symlog", linthresh=1.0)
        fig.tight_layout()
        path = self.plots_dir / "02_latency_cdf.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_precision_at_k_heatmap(self, summary_df: pd.DataFrame) -> List[Path]:
        p_cols = [c for c in ["avg_precision_at_1", "avg_precision_at_3", "avg_precision_at_5"]
                  if c in summary_df.columns]
        if not p_cols or summary_df.empty:
            return []
        heatmap_data = summary_df.set_index("retriever_name")[p_cols]
        fig, ax = plt.subplots(figsize=(len(p_cols) * 2 + 2, max(4, len(heatmap_data) * 0.5 + 1)))
        im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(p_cols)))
        ax.set_xticklabels([c.replace("avg_precision_at_", "P@") for c in p_cols])
        ax.set_yticks(range(len(heatmap_data)))
        ax.set_yticklabels(heatmap_data.index, fontsize=8)
        for i in range(len(heatmap_data)):
            for j in range(len(p_cols)):
                v = heatmap_data.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if v < 0.6 else "white")
        plt.colorbar(im, ax=ax)
        ax.set_title("Precision@k by Retriever")
        fig.tight_layout()
        path = self.plots_dir / "03_precision_at_k_heatmap.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_agreement_matrix(self, agreement_df: pd.DataFrame) -> List[Path]:
        if agreement_df.empty:
            return []
        n = len(agreement_df)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.6 + 2), max(5, n * 0.6 + 1)))
        im = ax.imshow(agreement_df.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(agreement_df.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(agreement_df.index, fontsize=7)
        for i in range(n):
            for j in range(n):
                v = agreement_df.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if v < 0.7 else "white")
        plt.colorbar(im, ax=ax, label="Jaccard similarity")
        ax.set_title("Retriever Agreement Matrix (avg Jaccard on top-k)")
        fig.tight_layout()
        path = self.plots_dir / "04_agreement_matrix.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_keyword_hit_rate_bar(self, summary_df: pd.DataFrame) -> List[Path]:
        col = "avg_keyword_hit_rate"
        if col not in summary_df.columns or summary_df.empty:
            return []
        df = summary_df.sort_values(col, ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6 + 2), 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
        ax.barh(df["retriever_name"], df[col], color=colors[::-1])
        ax.set_xlabel("Avg Keyword Hit Rate")
        ax.set_title("Keyword Hit Rate by Retriever")
        ax.set_xlim(0, 1.1)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(row[col] + 0.01, i, f"{row[col]:.3f}", va="center", fontsize=8)
        fig.tight_layout()
        path = self.plots_dir / "05_keyword_hit_rate_bar.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_result_count_boxplot(self, benchmark_df: pd.DataFrame) -> List[Path]:
        if "result_count" not in benchmark_df.columns or benchmark_df.empty:
            return []
        retrievers = sorted(benchmark_df["retriever_name"].unique())
        data = [
            benchmark_df.loc[benchmark_df["retriever_name"] == r, "result_count"].dropna().values
            for r in retrievers
        ]
        data = [d for d in data if len(d)]
        if not data:
            return []
        fig, ax = plt.subplots(figsize=(max(8, len(retrievers) * 0.7 + 2), 5))
        ax.boxplot(data, tick_labels=retrievers[:len(data)], vert=True)
        ax.set_ylabel("Result Count")
        ax.set_title("Result Count Distribution by Retriever")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        path = self.plots_dir / "06_result_count_boxplot.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_mrr_vs_latency_scatter(self, summary_df: pd.DataFrame) -> List[Path]:
        mrr_col = "avg_mrr_page"
        lat_col = "avg_latency_ms"
        if mrr_col not in summary_df.columns or lat_col not in summary_df.columns:
            return []
        if summary_df.empty:
            return []
        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.cm.tab20
        for i, (_, row) in enumerate(summary_df.iterrows()):
            ax.scatter(row[lat_col], row[mrr_col], s=80, color=cmap(i / max(len(summary_df), 1)),
                       label=row["retriever_name"], zorder=3)
            ax.annotate(row["retriever_name"], (row[lat_col], row[mrr_col]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax.set_xlabel("Avg Latency (ms)")
        ax.set_ylabel("Avg MRR")
        ax.set_title("MRR vs. Latency Trade-off")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.plots_dir / "07_mrr_vs_latency_scatter.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_error_rate_bar(self, summary_df: pd.DataFrame) -> List[Path]:
        col = "error_rate"
        if col not in summary_df.columns or summary_df.empty:
            return []
        df = summary_df.sort_values(col, ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6 + 2), 4))
        colors = ["#d73027" if v > 0.1 else "#fc8d59" if v > 0 else "#91cf60"
                  for v in df[col]]
        ax.barh(df["retriever_name"], df[col] * 100, color=colors)
        ax.set_xlabel("Error Rate (%)")
        ax.set_title("Query Error Rate by Retriever")
        ax.set_xlim(0, max(df[col].max() * 100 + 5, 5))
        fig.tight_layout()
        path = self.plots_dir / "08_error_rate_bar.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _score_results(
        self,
        results: List[Dict[str, Any]],
        expected_keywords: List[str],
        expected_pages: Set[int],
        top_k: int,
    ) -> Dict[str, float]:
        """Compute per-query quality metrics from a list of retrieved results."""
        if not results:
            return self._empty_metrics()

        truncated = results[:top_k]

        # MRR by page
        mrr_page = 0.0
        if expected_pages:
            for rank, r in enumerate(truncated, start=1):
                page = r.get("page_start") or r.get("page_end")
                if page is not None and int(page) in expected_pages:
                    mrr_page = 1.0 / rank
                    break

        # Keyword hit rate — fraction of expected_keywords found in any result text
        keyword_hit_rate = 0.0
        if expected_keywords:
            all_text = " ".join(
                str(r.get("text_preview", "")) for r in truncated
            ).lower()
            hits = sum(1 for kw in expected_keywords if str(kw).lower() in all_text)
            keyword_hit_rate = hits / len(expected_keywords)

        # Precision@k — count results whose page is in expected_pages
        def _precision_at(k: int) -> float:
            if not expected_pages:
                return 0.0
            hits = sum(
                1 for r in truncated[:k]
                if (r.get("page_start") or r.get("page_end")) is not None
                and int(r.get("page_start") or r.get("page_end") or -1) in expected_pages
            )
            return hits / min(k, len(truncated))

        precision_at_1 = _precision_at(1)
        precision_at_3 = _precision_at(3)
        precision_at_5 = _precision_at(5)

        # Coverage (unique chunks returned / top_k)
        unique_chunks = len(result_chunk_keys(truncated))
        coverage = unique_chunks / max(top_k, 1)

        return {
            "mrr_page": mrr_page,
            "keyword_hit_rate": keyword_hit_rate,
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_5": precision_at_5,
            "coverage_ratio": min(coverage, 1.0),
        }

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "mrr_page": 0.0,
            "keyword_hit_rate": 0.0,
            "precision_at_1": 0.0,
            "precision_at_3": 0.0,
            "precision_at_5": 0.0,
            "coverage_ratio": 0.0,
        }

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def simple_keyword_hit(text: str, keywords: List[str]) -> int:
    text_l = str(text or "").lower()
    return sum(1 for keyword in keywords if str(keyword).lower() in text_l)


def token_set(text: str) -> Set[str]:
    import re
    return set(re.findall(r"[a-zA-Z0-9]+", str(text or "").lower()))


def jaccard_text(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def result_identity(result: Dict[str, Any]) -> tuple:
    return (
        str(result.get("chunk_strategy_name")),
        str(result.get("document_id")),
        result.get("chunk_id"),
        result.get("start_index"),
        result.get("end_index"),
    )


class RerankingEvaluationManager:
    """
    Evaluates rerankers over common retrieved candidate pools.

    It runs one or more retrievers to fetch candidates, then applies each
    reranker to the same candidates and measures rank movement, keyword/page
    quality, diversity, duplicate ratio, and latency.
    """

    def __init__(
        self,
        store,
        retriever_manager,
        reranker_manager,
        report_dir: Optional[Union[str, Path]] = None,
    ):
        self.store = store
        self.retriever_manager = retriever_manager
        self.reranker_manager = reranker_manager
        self.document_dir = Path(store.paths.document_dir)
        self.report_dir = ensure_dir(report_dir or self.document_dir / "reports" / "reranking_evaluation")
        self.metrics_dir = ensure_dir(self.report_dir / "metrics")
        self.plots_dir = ensure_dir(self.report_dir / "plots")

    def run_benchmark(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: List[str],
        chunk_strategy_names: List[str],
        reranker_names: Optional[List[str]] = None,
        candidate_k: int = 20,
        top_k: int = 10,
        continue_on_error: bool = True,
    ) -> pd.DataFrame:
        rows = []
        reranker_names = reranker_names or self.reranker_manager.list_rerankers()

        for query_id, qobj in enumerate(query_suite):
            query = qobj["query"]
            expected_keywords = qobj.get("expected_keywords", [])
            expected_pages = set(qobj.get("expected_pages", []))
            filters = qobj.get("filters")
            retriever_kwargs = {
                "preferred_flags": qobj.get("preferred_flags"),
                "preferred_sections": qobj.get("preferred_sections"),
            }
            retriever_kwargs = {k: v for k, v in retriever_kwargs.items() if v is not None}

            for chunk_strategy in chunk_strategy_names:
                for retriever_name in retriever_names:
                    try:
                        started = time.perf_counter()
                        candidates = self.retriever_manager.retrieve(
                            retriever_name=retriever_name,
                            query=query,
                            chunk_strategy_name=chunk_strategy,
                            top_k=candidate_k,
                            filters=filters,
                            **retriever_kwargs,
                        )
                        retrieval_latency_ms = (time.perf_counter() - started) * 1000

                        baseline = candidates[:top_k]
                        baseline_metrics = self._score_results(
                            results=baseline,
                            expected_keywords=expected_keywords,
                            expected_pages=expected_pages,
                            top_k=top_k,
                        )
                        rows.append({
                            "query_id": query_id,
                            "query": query,
                            "chunk_strategy": chunk_strategy,
                            "retriever": retriever_name,
                            "reranker": "baseline",
                            "candidate_count": len(candidates),
                            "result_count": len(baseline),
                            "retrieval_latency_ms": retrieval_latency_ms,
                            "rerank_latency_ms": 0.0,
                            "error": None,
                            "mean_abs_rank_delta": 0.0,
                            "top_result_changed": 0,
                            **baseline_metrics,
                        })

                        for reranker_name in reranker_names:
                            started = time.perf_counter()
                            try:
                                reranked = self.reranker_manager.rerank(
                                    reranker_name=reranker_name,
                                    query=query,
                                    results=candidates,
                                    top_k=top_k,
                                )
                                rerank_latency_ms = (time.perf_counter() - started) * 1000
                                metrics = self._score_results(
                                    results=reranked,
                                    expected_keywords=expected_keywords,
                                    expected_pages=expected_pages,
                                    top_k=top_k,
                                )
                                movement = self._rank_movement(candidates, reranked)
                                rows.append({
                                    "query_id": query_id,
                                    "query": query,
                                    "chunk_strategy": chunk_strategy,
                                    "retriever": retriever_name,
                                    "reranker": reranker_name,
                                    "candidate_count": len(candidates),
                                    "result_count": len(reranked),
                                    "retrieval_latency_ms": retrieval_latency_ms,
                                    "rerank_latency_ms": rerank_latency_ms,
                                    "error": None,
                                    **movement,
                                    **metrics,
                                })
                            except Exception as e:
                                rerank_latency_ms = (time.perf_counter() - started) * 1000
                                rows.append({
                                    "query_id": query_id,
                                    "query": query,
                                    "chunk_strategy": chunk_strategy,
                                    "retriever": retriever_name,
                                    "reranker": reranker_name,
                                    "candidate_count": len(candidates),
                                    "result_count": 0,
                                    "retrieval_latency_ms": retrieval_latency_ms,
                                    "rerank_latency_ms": rerank_latency_ms,
                                    "error": repr(e),
                                    "mean_abs_rank_delta": 0.0,
                                    "top_result_changed": 0,
                                    **self._empty_metrics(),
                                })
                                if not continue_on_error:
                                    raise

                    except Exception as e:
                        rows.append({
                            "query_id": query_id,
                            "query": query,
                            "chunk_strategy": chunk_strategy,
                            "retriever": retriever_name,
                            "reranker": "candidate_retrieval_failed",
                            "candidate_count": 0,
                            "result_count": 0,
                            "retrieval_latency_ms": 0.0,
                            "rerank_latency_ms": 0.0,
                            "error": repr(e),
                            "mean_abs_rank_delta": 0.0,
                            "top_result_changed": 0,
                            **self._empty_metrics(),
                        })
                        if not continue_on_error:
                            raise

        out = pd.DataFrame(rows)
        self._add_baseline_deltas(out)
        save_df(out, self.metrics_dir / "reranking_benchmark.csv")
        return out

    def compute_summary(self, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = benchmark_df
        if df is None:
            path = self.metrics_dir / "reranking_benchmark.csv"
            df = pd.read_csv(path) if path.exists() else pd.DataFrame()

        if df.empty:
            out = pd.DataFrame()
            save_df(out, self.metrics_dir / "reranking_summary.csv")
            return out

        d = df[df["error"].isna()].copy()
        grouped = d.groupby("reranker", dropna=False).agg(
            runs=("reranker", "size"),
            avg_result_count=("result_count", "mean"),
            avg_keyword_hit_rate=("keyword_hit_rate", "mean"),
            avg_expected_page_hit_at_k=("expected_page_hit_at_k", "mean"),
            avg_mrr_page=("mrr_page", "mean"),
            avg_duplicate_text_ratio=("duplicate_text_ratio", "mean"),
            avg_page_diversity=("page_diversity", "mean"),
            avg_section_diversity=("section_diversity", "mean"),
            avg_rerank_latency_ms=("rerank_latency_ms", "mean"),
            avg_mean_abs_rank_delta=("mean_abs_rank_delta", "mean"),
            top_result_changed_rate=("top_result_changed", "mean"),
            avg_delta_keyword_hit_rate=("delta_keyword_hit_rate", "mean"),
            avg_delta_mrr_page=("delta_mrr_page", "mean"),
            avg_delta_duplicate_text_ratio=("delta_duplicate_text_ratio", "mean"),
        ).reset_index()

        grouped["quality_score"] = (
            grouped["avg_keyword_hit_rate"].fillna(0) * 0.35
            + grouped["avg_mrr_page"].fillna(0) * 0.25
            + grouped["avg_expected_page_hit_at_k"].fillna(0) * 0.15
            + (1 - grouped["avg_duplicate_text_ratio"].fillna(0)).clip(0, 1) * 0.15
            + grouped["avg_page_diversity"].fillna(0).clip(0, 5) / 5 * 0.10
        )
        grouped = grouped.sort_values("quality_score", ascending=False)
        save_df(grouped, self.metrics_dir / "reranking_summary.csv")
        return grouped

    def generate_report(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: List[str],
        chunk_strategy_names: List[str],
        reranker_names: Optional[List[str]] = None,
        candidate_k: int = 20,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        benchmark = self.run_benchmark(
            query_suite=query_suite,
            retriever_names=retriever_names,
            chunk_strategy_names=chunk_strategy_names,
            reranker_names=reranker_names,
            candidate_k=candidate_k,
            top_k=top_k,
        )
        summary_df = self.compute_summary(benchmark)
        plot_paths = self.plot_all(benchmark, summary_df)

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "plots_dir": str(self.plots_dir),
            "benchmark_rows": int(len(benchmark)),
            "summary_rows": int(len(summary_df)),
            "error_rows": int(benchmark["error"].notna().sum()) if "error" in benchmark.columns else 0,
            "candidate_k": int(candidate_k),
            "top_k": int(top_k),
            "metric_files": [
                str(self.metrics_dir / "reranking_benchmark.csv"),
                str(self.metrics_dir / "reranking_summary.csv"),
            ],
            "plots": [str(p) for p in plot_paths],
        }
        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def plot_all(self, benchmark_df: pd.DataFrame, summary_df: pd.DataFrame) -> List[Path]:
        paths = []
        if summary_df.empty:
            return paths

        paths.append(self._plot_bar(summary_df, "quality_score", "01_quality_score_by_reranker.png", "Reranker quality score"))
        paths.append(self._plot_bar(summary_df, "avg_delta_keyword_hit_rate", "02_delta_keyword_hit_rate.png", "Delta keyword hit rate vs baseline"))
        paths.append(self._plot_bar(summary_df, "avg_delta_mrr_page", "03_delta_mrr_page.png", "Delta MRR page vs baseline"))
        paths.append(self._plot_bar(summary_df, "avg_rerank_latency_ms", "04_latency_by_reranker.png", "Average rerank latency ms"))

        if not benchmark_df.empty:
            pivot = benchmark_df.pivot_table(
                index="reranker",
                columns="retriever",
                values="keyword_hit_rate",
                aggfunc="mean",
                fill_value=0,
            )
            path = self.plots_dir / "05_keyword_hit_heatmap.png"
            plt.figure(figsize=(max(10, pivot.shape[1] * 1.2), max(4.5, pivot.shape[0] * 0.4)))
            plt.imshow(pivot.values, aspect="auto", cmap="viridis")
            plt.colorbar(label="Keyword hit rate")
            plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
            plt.yticks(range(pivot.shape[0]), pivot.index)
            plt.title("Keyword hit rate by retriever and reranker")
            plt.tight_layout()
            plt.savefig(path, dpi=170)
            plt.close()
            paths.append(path)

        return paths

    def _plot_bar(self, df: pd.DataFrame, metric: str, filename: str, title: str) -> Path:
        d = df.sort_values(metric, ascending=True)
        path = self.plots_dir / filename
        plt.figure(figsize=(10, max(4.5, len(d) * 0.35)))
        plt.barh(d["reranker"], d[metric].fillna(0), color="#2563eb")
        plt.xlabel(metric)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()
        return path

    def _score_results(
        self,
        results: List[Dict[str, Any]],
        expected_keywords: List[str],
        expected_pages: Set[int],
        top_k: int,
    ) -> Dict[str, Any]:
        if not results:
            return self._empty_metrics()

        scores = [float(r.get("rerank_score", r.get("score", 0)) or 0) for r in results]
        pages = set()
        sections = set()
        previews = []

        for r in results[:top_k]:
            previews.append(r.get("text_preview", ""))
            if r.get("primary_section"):
                sections.add(r.get("primary_section"))

            ps = r.get("page_start")
            pe = r.get("page_end")
            if ps is None:
                continue
            if pe is None:
                pe = ps
            for page in range(int(ps), int(pe) + 1):
                pages.add(page)

        dup_pairs = 0
        total_pairs = 0
        for i in range(len(previews)):
            for j in range(i + 1, len(previews)):
                total_pairs += 1
                if jaccard_text(previews[i], previews[j]) >= 0.80:
                    dup_pairs += 1

        keyword_hit_rate = 0.0
        if expected_keywords:
            total_hits = sum(simple_keyword_hit(r.get("text_preview", ""), expected_keywords) for r in results[:top_k])
            keyword_hit_rate = total_hits / max(1, len(expected_keywords) * len(results[:top_k]))

        expected_page_hit = 0
        mrr = 0.0
        if expected_pages:
            for rank, r in enumerate(results[:top_k], start=1):
                ps = r.get("page_start")
                pe = r.get("page_end")
                if ps is None:
                    continue
                if pe is None:
                    pe = ps
                result_pages = set(range(int(ps), int(pe) + 1))
                if result_pages & expected_pages:
                    expected_page_hit = 1
                    mrr = 1.0 / rank
                    break

        return {
            "top_score": float(max(scores)),
            "avg_score": float(np.mean(scores)),
            "page_diversity": int(len(pages)),
            "section_diversity": int(len(sections)),
            "duplicate_text_ratio": float(dup_pairs / max(1, total_pairs)),
            "keyword_hit_rate": float(keyword_hit_rate),
            "expected_page_hit_at_k": int(expected_page_hit),
            "mrr_page": float(mrr),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "top_score": 0.0,
            "avg_score": 0.0,
            "page_diversity": 0,
            "section_diversity": 0,
            "duplicate_text_ratio": 0.0,
            "keyword_hit_rate": 0.0,
            "expected_page_hit_at_k": 0,
            "mrr_page": 0.0,
        }

    def _rank_movement(self, original: List[Dict[str, Any]], reranked: List[Dict[str, Any]]) -> Dict[str, Any]:
        original_rank = {result_identity(r): i for i, r in enumerate(original, start=1)}
        deltas = []
        for rank, r in enumerate(reranked, start=1):
            key = result_identity(r)
            if key in original_rank:
                deltas.append(abs(original_rank[key] - rank))

        top_changed = 0
        if original and reranked:
            top_changed = int(result_identity(original[0]) != result_identity(reranked[0]))

        return {
            "mean_abs_rank_delta": float(np.mean(deltas)) if deltas else 0.0,
            "top_result_changed": int(top_changed),
        }

    def _add_baseline_deltas(self, df: pd.DataFrame) -> None:
        for col in ["keyword_hit_rate", "mrr_page", "duplicate_text_ratio"]:
            df[f"delta_{col}"] = 0.0

        if df.empty:
            return

        keys = ["query_id", "chunk_strategy", "retriever"]
        baseline = df[df["reranker"] == "baseline"].set_index(keys)
        for idx, row in df.iterrows():
            key = tuple(row[k] for k in keys)
            if key not in baseline.index:
                continue
            base_row = baseline.loc[key]
            for col in ["keyword_hit_rate", "mrr_page", "duplicate_text_ratio"]:
                df.at[idx, f"delta_{col}"] = float(row.get(col, 0.0) or 0.0) - float(base_row.get(col, 0.0) or 0.0)

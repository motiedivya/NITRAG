from __future__ import annotations

import json
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


def safe_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def read_parquet_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def jaccard_text(a: str, b: str) -> float:
    import re
    ta = set(re.findall(r"[a-zA-Z0-9]+", str(a).lower()))
    tb = set(re.findall(r"[a-zA-Z0-9]+", str(b).lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def simple_keyword_hit(text: str, keywords: List[str]) -> int:
    text_l = str(text or "").lower()
    return sum(1 for k in keywords if str(k).lower() in text_l)


class RAGDiagnosticsManager:
    """
    Generates metrics and plots for RAG pipeline before LLM generation.

    Evaluates:
      - chunk strategies
      - enriched metadata coverage
      - index stats
      - retrieval behavior
      - optional golden query quality
    """

    def __init__(
        self,
        store,
        retriever_manager=None,
        report_dir: Optional[Union[str, Path]] = None,
        use_enriched_chunks: bool = True,
    ):
        self.store = store
        self.retriever_manager = retriever_manager

        self.document_dir = Path(store.paths.document_dir)
        self.chunk_dir = self.document_dir / ("chunks_enriched" if use_enriched_chunks else "chunks")
        self.raw_chunk_dir = self.document_dir / "chunks"
        self.index_root_dir = self.document_dir / "indexes"

        self.report_dir = ensure_dir(
            report_dir or self.document_dir / "reports" / "rag_diagnostics"
        )

        self.metrics_dir = ensure_dir(self.report_dir / "metrics")
        self.plots_dir = ensure_dir(self.report_dir / "plots")

        self.doc_manifest = read_json(self.document_dir / "manifest.json")
        self.layout_manifest = read_json(self.document_dir / "layout_manifest.json")
        self.clinical_metadata = read_json(self.document_dir / "clinical_document_metadata.json")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_chunk_strategies(self) -> List[str]:
        if not self.chunk_dir.exists():
            return []
        return sorted(p.stem for p in self.chunk_dir.glob("*.parquet"))

    def list_index_strategies(self) -> List[Dict[str, str]]:
        rows = []

        if not self.index_root_dir.exists():
            return rows

        for chunk_strategy_dir in sorted(self.index_root_dir.iterdir()):
            if not chunk_strategy_dir.is_dir():
                continue

            for index_dir in sorted(chunk_strategy_dir.iterdir()):
                if not index_dir.is_dir():
                    continue

                rows.append({
                    "chunk_strategy": chunk_strategy_dir.name,
                    "index_strategy": index_dir.name,
                    "index_dir": str(index_dir),
                })

        return rows

    def load_chunks(self, strategy_name: str) -> pd.DataFrame:
        path = self.chunk_dir / f"{strategy_name}.parquet"
        return read_parquet_df(path)

    # ------------------------------------------------------------------
    # Chunk metrics
    # ------------------------------------------------------------------

    def compute_chunk_metrics(self) -> pd.DataFrame:
        rows = []

        for strategy in self.list_chunk_strategies():
            df = self.load_chunks(strategy)

            if df.empty:
                rows.append({
                    "strategy": strategy,
                    "chunk_count": 0,
                    "avg_tokens": 0,
                    "median_tokens": 0,
                    "min_tokens": 0,
                    "max_tokens": 0,
                    "p95_tokens": 0,
                    "avg_pages_per_chunk": 0,
                    "section_coverage_pct": 0,
                    "entity_coverage_pct": 0,
                    "avg_entity_count": 0,
                    "avg_quality_score": 0,
                    "zero_token_chunks": 0,
                    "oversized_chunks_1500": 0,
                    "tiny_chunks_50": 0,
                })
                continue

            token_lengths = pd.to_numeric(df.get("token_length", pd.Series(dtype=float)), errors="coerce").fillna(0)
            page_start = pd.to_numeric(df.get("page_start", pd.Series(dtype=float)), errors="coerce")
            page_end = pd.to_numeric(df.get("page_end", pd.Series(dtype=float)), errors="coerce")

            if "primary_section" in df.columns:
                section_coverage = df["primary_section"].notna().mean() * 100
            else:
                section_coverage = 0

            if "entity_count" in df.columns:
                entity_count = pd.to_numeric(df["entity_count"], errors="coerce").fillna(0)
                entity_coverage = (entity_count > 0).mean() * 100
                avg_entity_count = entity_count.mean()
            else:
                entity_coverage = 0
                avg_entity_count = 0

            if "clinical_quality_score" in df.columns:
                quality = pd.to_numeric(df["clinical_quality_score"], errors="coerce").fillna(0)
                avg_quality = quality.mean()
            else:
                avg_quality = 0

            pages_per_chunk = (page_end - page_start + 1).fillna(0)

            rows.append({
                "strategy": strategy,
                "chunk_count": int(len(df)),
                "avg_tokens": float(token_lengths.mean()),
                "median_tokens": float(token_lengths.median()),
                "min_tokens": float(token_lengths.min()),
                "max_tokens": float(token_lengths.max()),
                "p95_tokens": float(token_lengths.quantile(0.95)),
                "avg_pages_per_chunk": float(pages_per_chunk.mean()),
                "section_coverage_pct": float(section_coverage),
                "entity_coverage_pct": float(entity_coverage),
                "avg_entity_count": float(avg_entity_count),
                "avg_quality_score": float(avg_quality),
                "zero_token_chunks": int((token_lengths <= 0).sum()),
                "oversized_chunks_1500": int((token_lengths > 1500).sum()),
                "tiny_chunks_50": int((token_lengths < 50).sum()),
            })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "chunk_metrics.csv")
        return out

    # ------------------------------------------------------------------
    # Metadata coverage metrics
    # ------------------------------------------------------------------

    def compute_metadata_coverage(self) -> pd.DataFrame:
        flag_cols = [
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

        rows = []

        for strategy in self.list_chunk_strategies():
            df = self.load_chunks(strategy)

            if df.empty:
                for col in flag_cols:
                    rows.append({
                        "strategy": strategy,
                        "metadata_field": col,
                        "coverage_pct": 0.0,
                        "count": 0,
                        "chunk_count": 0,
                    })
                continue

            for col in flag_cols:
                if col in df.columns:
                    vals = df[col].fillna(False).astype(bool)
                    count = int(vals.sum())
                    pct = float(vals.mean() * 100)
                else:
                    count = 0
                    pct = 0.0

                rows.append({
                    "strategy": strategy,
                    "metadata_field": col,
                    "coverage_pct": pct,
                    "count": count,
                    "chunk_count": int(len(df)),
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "metadata_coverage.csv")
        return out

    # ------------------------------------------------------------------
    # Section/entity distribution
    # ------------------------------------------------------------------

    def compute_section_distribution(self) -> pd.DataFrame:
        rows = []

        for strategy in self.list_chunk_strategies():
            df = self.load_chunks(strategy)

            if df.empty or "primary_section" not in df.columns:
                continue

            counts = df["primary_section"].fillna("NO_SECTION").value_counts()

            for section, count in counts.items():
                rows.append({
                    "strategy": strategy,
                    "section": section,
                    "count": int(count),
                    "pct": float(count / max(1, len(df)) * 100),
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "section_distribution.csv")
        return out

    def compute_entity_type_distribution(self) -> pd.DataFrame:
        rows = []

        for strategy in self.list_chunk_strategies():
            df = self.load_chunks(strategy)

            if df.empty or "entity_type_counts_json" not in df.columns:
                continue

            counter = Counter()

            for raw in df["entity_type_counts_json"].dropna():
                obj = safe_json_loads(raw)
                if isinstance(obj, dict):
                    for entity_type, count in obj.items():
                        counter[entity_type] += int(count or 0)

            total = sum(counter.values())

            for entity_type, count in counter.items():
                rows.append({
                    "strategy": strategy,
                    "entity_type": entity_type,
                    "count": int(count),
                    "pct": float(count / max(1, total) * 100),
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "entity_type_distribution.csv")
        return out

    # ------------------------------------------------------------------
    # Index metrics
    # ------------------------------------------------------------------

    def compute_index_metrics(self) -> pd.DataFrame:
        rows = []

        for item in self.list_index_strategies():
            index_dir = Path(item["index_dir"])
            manifest = read_json(index_dir / "manifest.json")

            row = {
                "chunk_strategy": item["chunk_strategy"],
                "index_strategy": item["index_strategy"],
                "index_dir": str(index_dir),
            }

            row.update(manifest)
            rows.append(row)

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "index_metrics.csv")
        return out

    # ------------------------------------------------------------------
    # Retrieval benchmark
    # ------------------------------------------------------------------

    def run_retrieval_benchmark(
        self,
        query_suite: List[Dict[str, Any]],
        retriever_names: List[str],
        chunk_strategy_names: Optional[List[str]] = None,
        top_k: int = 10,
        continue_on_error: bool = True,
    ) -> pd.DataFrame:
        """
        query_suite format:
        [
            {
                "query": "medication dose pain",
                "expected_keywords": ["medication", "pain"],
                "expected_pages": [0, 1],
                "filters": {"contains_medication": True},
                "preferred_flags": ["contains_medication"]
            }
        ]

        expected fields are optional.
        """

        if self.retriever_manager is None:
            raise ValueError("retriever_manager is required for retrieval benchmark.")

        chunk_strategy_names = chunk_strategy_names or self.retriever_manager.list_chunk_strategies()

        rows = []

        for qid, qobj in enumerate(query_suite):
            query = qobj["query"]
            expected_keywords = qobj.get("expected_keywords", [])
            expected_pages = set(qobj.get("expected_pages", []))
            filters = qobj.get("filters")
            preferred_flags = qobj.get("preferred_flags")

            for chunk_strategy in chunk_strategy_names:
                for retriever_name in retriever_names:
                    started = time.perf_counter()

                    try:
                        results = self.retriever_manager.retrieve(
                            retriever_name=retriever_name,
                            query=query,
                            chunk_strategy_name=chunk_strategy,
                            top_k=top_k,
                            filters=filters,
                            preferred_flags=preferred_flags,
                        )

                        latency_ms = (time.perf_counter() - started) * 1000

                        metrics = self._score_retrieval_results(
                            results=results,
                            expected_keywords=expected_keywords,
                            expected_pages=expected_pages,
                            top_k=top_k,
                        )

                        rows.append({
                            "query_id": qid,
                            "query": query,
                            "retriever": retriever_name,
                            "chunk_strategy": chunk_strategy,
                            "result_count": len(results),
                            "latency_ms": latency_ms,
                            "error": None,
                            **metrics,
                        })

                    except Exception as e:
                        latency_ms = (time.perf_counter() - started) * 1000

                        rows.append({
                            "query_id": qid,
                            "query": query,
                            "retriever": retriever_name,
                            "chunk_strategy": chunk_strategy,
                            "result_count": 0,
                            "latency_ms": latency_ms,
                            "error": repr(e),
                            "top_score": 0,
                            "avg_score": 0,
                            "page_diversity": 0,
                            "section_diversity": 0,
                            "duplicate_text_ratio": 0,
                            "keyword_hit_rate": 0,
                            "expected_page_hit_at_k": 0,
                            "mrr_page": 0,
                        })

                        if not continue_on_error:
                            raise

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "retrieval_benchmark.csv")
        return out

    def _score_retrieval_results(
        self,
        results: List[Dict[str, Any]],
        expected_keywords: List[str],
        expected_pages: set,
        top_k: int,
    ) -> Dict[str, Any]:
        if not results:
            return {
                "top_score": 0,
                "avg_score": 0,
                "page_diversity": 0,
                "section_diversity": 0,
                "duplicate_text_ratio": 0,
                "keyword_hit_rate": 0,
                "expected_page_hit_at_k": 0,
                "mrr_page": 0,
            }

        scores = [float(r.get("score") or 0) for r in results]
        pages = set()

        for r in results:
            ps = r.get("page_start")
            pe = r.get("page_end")
            if ps is None:
                continue
            if pe is None:
                pe = ps
            for p in range(int(ps), int(pe) + 1):
                pages.add(p)

        sections = {
            r.get("primary_section")
            for r in results
            if r.get("primary_section")
        }

        # Duplicate ratio: how many pairs are very similar.
        dup_pairs = 0
        total_pairs = 0
        previews = [r.get("text_preview", "") for r in results]

        for i in range(len(previews)):
            for j in range(i + 1, len(previews)):
                total_pairs += 1
                if jaccard_text(previews[i], previews[j]) >= 0.80:
                    dup_pairs += 1

        duplicate_text_ratio = dup_pairs / max(1, total_pairs)

        # Keyword hit rate.
        if expected_keywords:
            total_hits = 0
            possible_hits = len(expected_keywords) * len(results)

            for r in results:
                total_hits += simple_keyword_hit(r.get("text_preview", ""), expected_keywords)

            keyword_hit_rate = total_hits / max(1, possible_hits)
        else:
            keyword_hit_rate = 0

        # Expected page hit + MRR.
        expected_page_hit = 0
        mrr = 0

        if expected_pages:
            for rank, r in enumerate(results, start=1):
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
            "duplicate_text_ratio": float(duplicate_text_ratio),
            "keyword_hit_rate": float(keyword_hit_rate),
            "expected_page_hit_at_k": int(expected_page_hit),
            "mrr_page": float(mrr),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_chunk_counts(self, chunk_metrics: Optional[pd.DataFrame] = None) -> Path:
        df = chunk_metrics if chunk_metrics is not None else self.compute_chunk_metrics()
        path = self.plots_dir / "chunk_counts.png"

        if df.empty:
            return path

        d = df.sort_values("chunk_count", ascending=True)

        plt.figure(figsize=(10, max(4, len(d) * 0.45)))
        plt.barh(d["strategy"], d["chunk_count"])
        plt.xlabel("Chunk count")
        plt.ylabel("Chunk strategy")
        plt.title("Chunk count by strategy")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_token_length_boxplot(self) -> Path:
        path = self.plots_dir / "token_length_boxplot.png"

        data = []
        labels = []

        for strategy in self.list_chunk_strategies():
            df = self.load_chunks(strategy)
            if df.empty or "token_length" not in df.columns:
                continue

            vals = pd.to_numeric(df["token_length"], errors="coerce").dropna().values
            if len(vals) == 0:
                continue

            data.append(vals)
            labels.append(strategy)

        if not data:
            return path

        plt.figure(figsize=(max(10, len(labels) * 0.7), 6))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Token length")
        plt.title("Token length distribution by chunk strategy")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_metadata_coverage_heatmap(self, metadata_df: Optional[pd.DataFrame] = None) -> Path:
        df = metadata_df if metadata_df is not None else self.compute_metadata_coverage()
        path = self.plots_dir / "metadata_coverage_heatmap.png"

        if df.empty:
            return path

        pivot = df.pivot_table(
            index="strategy",
            columns="metadata_field",
            values="coverage_pct",
            aggfunc="mean",
            fill_value=0,
        )

        plt.figure(figsize=(max(10, pivot.shape[1] * 0.9), max(5, pivot.shape[0] * 0.45)))
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="Coverage %")
        plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(pivot.shape[0]), pivot.index)
        plt.title("Metadata coverage by chunk strategy")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                plt.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_quality_histograms(self) -> Path:
        path = self.plots_dir / "clinical_quality_histograms.png"

        strategies = self.list_chunk_strategies()
        plotted = False

        plt.figure(figsize=(10, 6))

        for strategy in strategies:
            df = self.load_chunks(strategy)
            if df.empty or "clinical_quality_score" not in df.columns:
                continue

            vals = pd.to_numeric(df["clinical_quality_score"], errors="coerce").dropna()
            if vals.empty:
                continue

            plt.hist(vals, bins=20, alpha=0.35, label=strategy)
            plotted = True

        if not plotted:
            plt.close()
            return path

        plt.xlabel("Clinical quality score")
        plt.ylabel("Chunk count")
        plt.title("Clinical quality score distribution")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_index_metrics(self, index_df: Optional[pd.DataFrame] = None) -> List[Path]:
        df = index_df if index_df is not None else self.compute_index_metrics()
        paths = []

        if df.empty:
            return paths

        if "vocab_size" in df.columns:
            d = df[df["vocab_size"].notna()].copy()
            if not d.empty:
                d["label"] = d["chunk_strategy"].astype(str) + " / " + d["index_strategy"].astype(str)
                d = d.sort_values("vocab_size", ascending=True)

                path = self.plots_dir / "index_vocab_size.png"
                plt.figure(figsize=(10, max(4, len(d) * 0.45)))
                plt.barh(d["label"], d["vocab_size"])
                plt.xlabel("Vocabulary size")
                plt.title("Index vocabulary size")
                plt.tight_layout()
                plt.savefig(path, dpi=160)
                plt.close()
                paths.append(path)

        if "postings_count" in df.columns:
            d = df[df["postings_count"].notna()].copy()
            if not d.empty:
                d["label"] = d["chunk_strategy"].astype(str) + " / " + d["index_strategy"].astype(str)
                d = d.sort_values("postings_count", ascending=True)

                path = self.plots_dir / "index_postings_count.png"
                plt.figure(figsize=(10, max(4, len(d) * 0.45)))
                plt.barh(d["label"], d["postings_count"])
                plt.xlabel("Postings count")
                plt.title("Index postings count")
                plt.tight_layout()
                plt.savefig(path, dpi=160)
                plt.close()
                paths.append(path)

        return paths

    def plot_retrieval_benchmark(self, retrieval_df: pd.DataFrame) -> List[Path]:
        paths = []

        if retrieval_df.empty:
            return paths

        # Average score heatmap.
        for metric in [
            "top_score",
            "keyword_hit_rate",
            "expected_page_hit_at_k",
            "mrr_page",
            "latency_ms",
            "duplicate_text_ratio",
            "page_diversity",
        ]:
            if metric not in retrieval_df.columns:
                continue

            pivot = retrieval_df.pivot_table(
                index="chunk_strategy",
                columns="retriever",
                values=metric,
                aggfunc="mean",
                fill_value=0,
            )

            if pivot.empty:
                continue

            path = self.plots_dir / f"retrieval_{metric}_heatmap.png"

            plt.figure(figsize=(max(10, pivot.shape[1] * 1.2), max(5, pivot.shape[0] * 0.45)))
            plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(label=metric)
            plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
            plt.yticks(range(pivot.shape[0]), pivot.index)
            plt.title(f"Retrieval benchmark: {metric}")

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()

            paths.append(path)

        return paths

    # ------------------------------------------------------------------
    # Master report
    # ------------------------------------------------------------------

    def generate_static_report(self) -> Dict[str, Any]:
        chunk_metrics = self.compute_chunk_metrics()
        metadata_coverage = self.compute_metadata_coverage()
        section_dist = self.compute_section_distribution()
        entity_dist = self.compute_entity_type_distribution()
        index_metrics = self.compute_index_metrics()

        plot_paths = []

        plot_paths.append(self.plot_chunk_counts(chunk_metrics))
        plot_paths.append(self.plot_token_length_boxplot())
        plot_paths.append(self.plot_metadata_coverage_heatmap(metadata_coverage))
        plot_paths.append(self.plot_quality_histograms())
        plot_paths.extend(self.plot_index_metrics(index_metrics))

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "plots_dir": str(self.plots_dir),
            "chunk_strategy_count": len(self.list_chunk_strategies()),
            "index_count": len(self.list_index_strategies()),
            "chunk_metrics_rows": len(chunk_metrics),
            "metadata_coverage_rows": len(metadata_coverage),
            "section_distribution_rows": len(section_dist),
            "entity_distribution_rows": len(entity_dist),
            "index_metrics_rows": len(index_metrics),
            "plots": [str(p) for p in plot_paths],
        }

        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Static RAG diagnostics report generated.")
        print(f"Report dir: {self.report_dir}")

        return summary

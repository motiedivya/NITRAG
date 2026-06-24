from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


def _cap_figsize(w: float, h: float, max_w: float = 15.0, max_h: float = 12.0):
    return (min(w, max_w), min(h, max_h))


class ChunkingEvaluationManager:
    """
    First-stage chunking evaluator.

    Reads raw chunks from rag_store/<doc_id>/chunks so the report measures
    only chunking behaviour, before enrichment, indexing, or retrieval.

    Metrics computed
    ────────────────
    Coverage & gaps      : coverage_pct, missing_tokens, gap_count, max_gap_tokens
    Overlap / redundancy : overlap_tokens, overlap_pct_of_chunk_tokens, redundancy_factor
    Length distribution  : min/p10/median/mean/p90/p95/max/std/cv/iqr_tokens
    Distribution shape   : gini_coefficient, entropy_bits
    Size buckets         : empty_lt_10, tiny_lt_50, small_lt_128, large_gt_1024, oversized_gt_1500
    Page-crossing        : mean/max_pages_per_chunk, page_crossing_pct
    Boundary alignment   : start/end on page/block/line boundary %,
                           starts/ends inside block/line %
    Structural integrity : duplicate_span_count, monotonic_start_breaks
    """

    def __init__(
        self,
        store,
        chunk_dir: Optional[Union[str, Path]] = None,
        report_dir: Optional[Union[str, Path]] = None,
    ):
        self.store = store
        self.document_dir = Path(store.paths.document_dir)
        self.chunk_dir = Path(chunk_dir or store.paths.chunks_dir)
        self.report_dir = Path(report_dir or self.document_dir / "reports" / "chunking_evaluation")
        self.metrics_dir = self.report_dir / "metrics"
        self.plots_dir = self.report_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.pages_df = pd.DataFrame(store.pages())
        self.elements_df = pd.DataFrame(store.elements())
        self.total_tokens = int(store.total_tokens)

    # ------------------------------------------------------------------
    # Listing / loading
    # ------------------------------------------------------------------

    def list_strategies(self) -> List[str]:
        if not self.chunk_dir.exists():
            return []
        return sorted(p.stem for p in self.chunk_dir.glob("*.parquet"))

    def load_chunks(self, strategy: str) -> pd.DataFrame:
        path = self.chunk_dir / f"{strategy}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return pq.read_table(path).to_pandas()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        intervals = sorted((int(s), int(e)) for s, e in intervals if e > s)
        if not intervals:
            return []
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _covered_length(intervals: List[Tuple[int, int]]) -> int:
        return int(sum(end - start for start, end in intervals))

    @staticmethod
    def _gap_lengths(merged: List[Tuple[int, int]], total_tokens: int) -> List[int]:
        gaps = []
        cursor = 0
        for start, end in merged:
            if start > cursor:
                gaps.append(start - cursor)
            cursor = max(cursor, end)
        if cursor < total_tokens:
            gaps.append(total_tokens - cursor)
        return [int(g) for g in gaps if g > 0]

    @staticmethod
    def _pct_aligned(values: pd.Series, valid_boundaries: set) -> float:
        vals = pd.to_numeric(values, errors="coerce").dropna().astype(int)
        if vals.empty:
            return 0.0
        return float(vals.isin(valid_boundaries).mean() * 100)

    @staticmethod
    def _count_points_inside_intervals(points: pd.Series, intervals: List[Tuple[int, int]]) -> int:
        pts = pd.to_numeric(points, errors="coerce").dropna().astype(int).tolist()
        count = 0
        for point in pts:
            for start, end in intervals:
                if start < point < end:
                    count += 1
                    break
        return int(count)

    @staticmethod
    def _gini_coefficient(lengths: np.ndarray) -> float:
        """
        Gini coefficient of chunk lengths.
        0 = all chunks identical length, 1 = one chunk holds all tokens.
        """
        if len(lengths) == 0 or lengths.sum() == 0:
            return 0.0
        x = np.sort(lengths.astype(float))
        n = len(x)
        cumsum = np.cumsum(x)
        return float((2 * np.dot(np.arange(1, n + 1), x) / (n * cumsum[-1])) - (n + 1) / n)

    @staticmethod
    def _entropy_bits(lengths: np.ndarray, bins: int = 20) -> float:
        """
        Shannon entropy (bits) of the discretised chunk-length distribution.
        High entropy = diverse mix of sizes; low = concentrated at one size.
        """
        if len(lengths) == 0:
            return 0.0
        counts, _ = np.histogram(lengths, bins=max(2, bins))
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(max(0.0, -np.sum(probs * np.log2(probs))))

    # ------------------------------------------------------------------
    # Element helpers
    # ------------------------------------------------------------------

    def _element_intervals(self, element_type: str) -> List[Tuple[int, int]]:
        if self.elements_df.empty or "element_type" not in self.elements_df.columns:
            return []
        d = self.elements_df[self.elements_df["element_type"] == element_type]
        return [
            (int(r.start_index), int(r.end_index))
            for r in d.itertuples()
            if pd.notna(r.start_index) and pd.notna(r.end_index) and int(r.end_index) > int(r.start_index)
        ]

    def _boundary_sets(self, element_type: str) -> Tuple[set, set, set]:
        intervals = self._element_intervals(element_type)
        starts = {s for s, _ in intervals}
        ends = {e for _, e in intervals}
        return starts, ends, starts | ends

    def _page_coverage_rows(self, strategy: str, merged_intervals: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        rows = []
        for page in self.pages_df.itertuples():
            page_start = int(page.start_index)
            page_end = int(page.end_index)
            page_len = max(0, page_end - page_start)
            covered = sum(
                max(0, min(end, page_end) - max(start, page_start))
                for start, end in merged_intervals
            )
            rows.append({
                "strategy": strategy,
                "page_number": int(page.page_number),
                "page_tokens": page_len,
                "covered_tokens": int(covered),
                "coverage_pct": float(covered / max(1, page_len) * 100),
            })
        return rows

    # ------------------------------------------------------------------
    # Empty row helper
    # ------------------------------------------------------------------

    def _empty_metric_row(self, strategy: str) -> Dict[str, Any]:
        return {
            "strategy": strategy,
            "chunk_count": 0,
            "total_doc_tokens": int(self.total_tokens),
            "total_chunk_tokens": 0,
            "covered_tokens_unique": 0,
            "coverage_pct": 0.0,
            "missing_tokens": int(self.total_tokens),
            "gap_count": 1 if self.total_tokens > 0 else 0,
            "max_gap_tokens": int(self.total_tokens),
            "overlap_tokens": 0,
            "overlap_pct_of_chunk_tokens": 0.0,
            "redundancy_factor": 0.0,
            "duplicate_span_count": 0,
            "monotonic_start_breaks": 0,
            "min_tokens": 0.0,
            "p10_tokens": 0.0,
            "median_tokens": 0.0,
            "mean_tokens": 0.0,
            "p90_tokens": 0.0,
            "p95_tokens": 0.0,
            "max_tokens": 0.0,
            "std_tokens": 0.0,
            "cv_tokens": 0.0,
            "iqr_tokens": 0.0,
            "gini_coefficient": 0.0,
            "entropy_bits": 0.0,
            "empty_chunks_lt_10": 0,
            "tiny_chunks_lt_50": 0,
            "small_chunks_lt_128": 0,
            "medium_chunks_128_to_511": 0,
            "large_chunks_512_to_1024": 0,
            "oversized_chunks_gt_1024": 0,
            "very_oversized_chunks_gt_1500": 0,
            "mean_pages_per_chunk": 0.0,
            "max_pages_per_chunk": 0.0,
            "page_crossing_chunks": 0,
            "page_crossing_pct": 0.0,
            "start_on_page_boundary_pct": 0.0,
            "end_on_page_boundary_pct": 0.0,
            "start_on_block_boundary_pct": 0.0,
            "end_on_block_boundary_pct": 0.0,
            "start_on_line_boundary_pct": 0.0,
            "end_on_line_boundary_pct": 0.0,
            "starts_inside_block_pct": 0.0,
            "ends_inside_block_pct": 0.0,
            "starts_inside_line_pct": 0.0,
            "ends_inside_line_pct": 0.0,
        }

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self, strategies: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        strategies = strategies or self.list_strategies()
        metric_rows: List[Dict[str, Any]] = []
        length_rows: List[Dict[str, Any]] = []
        page_rows: List[Dict[str, Any]] = []

        page_starts, page_ends, page_boundaries = self._boundary_sets("page")
        block_starts, block_ends, block_boundaries = self._boundary_sets("block")
        line_starts, line_ends, line_boundaries = self._boundary_sets("line")
        block_intervals = self._element_intervals("block")
        line_intervals = self._element_intervals("line")

        for strategy in strategies:
            df = self.load_chunks(strategy).copy()
            if df.empty:
                metric_rows.append(self._empty_metric_row(strategy))
                page_rows.extend(self._page_coverage_rows(strategy, []))
                continue

            for col in ["start_index", "end_index", "token_length", "page_start", "page_end"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            intervals = [
                (max(0, int(r.start_index)), min(self.total_tokens, int(r.end_index)))
                for r in df.itertuples()
                if pd.notna(r.start_index) and pd.notna(r.end_index) and int(r.end_index) > int(r.start_index)
            ]
            merged = self._merge_intervals(intervals)
            covered_tokens = self._covered_length(merged)
            total_chunk_tokens = int(sum(e - s for s, e in intervals))
            overlap_tokens = max(0, total_chunk_tokens - covered_tokens)
            gaps = self._gap_lengths(merged, self.total_tokens)

            lengths = pd.to_numeric(
                df.get("token_length", df["end_index"] - df["start_index"]),
                errors="coerce",
            ).fillna(0)
            lengths_arr = lengths.to_numpy(dtype=float)

            pages_per_chunk = (
                pd.to_numeric(df.get("page_end", pd.Series(index=df.index)), errors="coerce")
                - pd.to_numeric(df.get("page_start", pd.Series(index=df.index)), errors="coerce")
                + 1
            ).fillna(0)

            duplicate_spans = int(df.duplicated(subset=["start_index", "end_index"]).sum()) if {"start_index", "end_index"}.issubset(df.columns) else 0
            monotonic_breaks = int((df["start_index"].diff().fillna(0) < 0).sum()) if "start_index" in df.columns else 0

            starts_inside_block = self._count_points_inside_intervals(df["start_index"], block_intervals)
            ends_inside_block   = self._count_points_inside_intervals(df["end_index"],   block_intervals)
            starts_inside_line  = self._count_points_inside_intervals(df["start_index"], line_intervals)
            ends_inside_line    = self._count_points_inside_intervals(df["end_index"],   line_intervals)

            iqr = float(np.percentile(lengths_arr, 75) - np.percentile(lengths_arr, 25))

            metric_rows.append({
                "strategy": strategy,
                "chunk_count": int(len(df)),
                "total_doc_tokens": int(self.total_tokens),
                "total_chunk_tokens": total_chunk_tokens,
                "covered_tokens_unique": int(covered_tokens),
                "coverage_pct": float(covered_tokens / max(1, self.total_tokens) * 100),
                "missing_tokens": int(max(0, self.total_tokens - covered_tokens)),
                "gap_count": int(len(gaps)),
                "max_gap_tokens": int(max(gaps) if gaps else 0),
                "overlap_tokens": int(overlap_tokens),
                "overlap_pct_of_chunk_tokens": float(overlap_tokens / max(1, total_chunk_tokens) * 100),
                "redundancy_factor": float(total_chunk_tokens / max(1, covered_tokens)),
                "duplicate_span_count": duplicate_spans,
                "monotonic_start_breaks": monotonic_breaks,
                "min_tokens": float(lengths.min()),
                "p10_tokens": float(lengths.quantile(0.10)),
                "median_tokens": float(lengths.median()),
                "mean_tokens": float(lengths.mean()),
                "p90_tokens": float(lengths.quantile(0.90)),
                "p95_tokens": float(lengths.quantile(0.95)),
                "max_tokens": float(lengths.max()),
                "std_tokens": float(lengths.std(ddof=0)),
                "cv_tokens": float(lengths.std(ddof=0) / max(1e-9, lengths.mean())),
                "iqr_tokens": iqr,
                "gini_coefficient": self._gini_coefficient(lengths_arr),
                "entropy_bits": self._entropy_bits(lengths_arr),
                "empty_chunks_lt_10": int((lengths < 10).sum()),
                "tiny_chunks_lt_50": int((lengths < 50).sum()),
                "small_chunks_lt_128": int((lengths < 128).sum()),
                "medium_chunks_128_to_511": int(((lengths >= 128) & (lengths < 512)).sum()),
                "large_chunks_512_to_1024": int(((lengths >= 512) & (lengths <= 1024)).sum()),
                "oversized_chunks_gt_1024": int((lengths > 1024).sum()),
                "very_oversized_chunks_gt_1500": int((lengths > 1500).sum()),
                "mean_pages_per_chunk": float(pages_per_chunk.mean()),
                "max_pages_per_chunk": float(pages_per_chunk.max()),
                "page_crossing_chunks": int((pages_per_chunk > 1).sum()),
                "page_crossing_pct": float((pages_per_chunk > 1).mean() * 100),
                "start_on_page_boundary_pct": self._pct_aligned(df["start_index"], page_starts),
                "end_on_page_boundary_pct": self._pct_aligned(df["end_index"], page_ends),
                "start_on_block_boundary_pct": self._pct_aligned(df["start_index"], block_boundaries),
                "end_on_block_boundary_pct": self._pct_aligned(df["end_index"], block_boundaries),
                "start_on_line_boundary_pct": self._pct_aligned(df["start_index"], line_boundaries),
                "end_on_line_boundary_pct": self._pct_aligned(df["end_index"], line_boundaries),
                "starts_inside_block_pct": float(starts_inside_block / max(1, len(df)) * 100),
                "ends_inside_block_pct": float(ends_inside_block / max(1, len(df)) * 100),
                "starts_inside_line_pct": float(starts_inside_line / max(1, len(df)) * 100),
                "ends_inside_line_pct": float(ends_inside_line / max(1, len(df)) * 100),
            })

            for i, length in enumerate(lengths.tolist()):
                length_rows.append({"strategy": strategy, "chunk_row": i, "token_length": float(length)})
            page_rows.extend(self._page_coverage_rows(strategy, merged))

        metrics_df       = pd.DataFrame(metric_rows)
        lengths_df       = pd.DataFrame(length_rows)
        page_coverage_df = pd.DataFrame(page_rows)

        metrics_df.to_csv(self.metrics_dir / "chunking_metrics.csv", index=False)
        lengths_df.to_csv(self.metrics_dir / "chunk_token_lengths_long.csv", index=False)
        page_coverage_df.to_csv(self.metrics_dir / "chunk_page_coverage.csv", index=False)

        return metrics_df, lengths_df, page_coverage_df

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_all(
        self,
        metrics_df: Optional[pd.DataFrame] = None,
        lengths_df: Optional[pd.DataFrame] = None,
        page_coverage_df: Optional[pd.DataFrame] = None,
    ) -> List[Path]:
        if metrics_df is None or lengths_df is None or page_coverage_df is None:
            metrics_df, lengths_df, page_coverage_df = self.compute_metrics()

        paths: List[Path] = []
        if metrics_df.empty:
            return paths

        # 01 — chunk count
        d = metrics_df.sort_values("chunk_count", ascending=True)
        path = self.plots_dir / "01_chunk_count_by_strategy.png"
        plt.figure(figsize=(11, max(4.5, len(d) * 0.45)))
        plt.barh(d["strategy"], d["chunk_count"], color="#3b82f6")
        plt.xlabel("Chunk count")
        plt.title("Chunk count by strategy")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        paths.append(path)

        if not lengths_df.empty:
            ordered = metrics_df.sort_values("median_tokens")["strategy"].tolist()
            data    = [lengths_df.loc[lengths_df["strategy"] == s, "token_length"].dropna().values for s in ordered]
            labels  = [s for s, x in zip(ordered, data) if len(x) > 0]
            data    = [x for x in data if len(x) > 0]

            # 02 — boxplot
            path = self.plots_dir / "02_token_length_boxplot.png"
            plt.figure(figsize=_cap_figsize(max(11, len(labels) * 0.75), 6))
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Tokens per chunk")
            plt.title("Token length distribution by strategy (box)")
            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close()
            paths.append(path)

            # 03 — per-strategy histograms
            strategies = sorted(lengths_df["strategy"].unique())
            ncols = 2
            nrows = math.ceil(len(strategies) / ncols)
            path = self.plots_dir / "03_token_length_histograms.png"
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=_cap_figsize(12, max(4, nrows * 3.2)))
            axes = np.asarray(axes).reshape(-1)
            for ax, s in zip(axes, strategies):
                vals = lengths_df.loc[lengths_df["strategy"] == s, "token_length"].dropna()
                ax.hist(vals, bins=min(30, max(5, int(np.sqrt(max(1, len(vals)))))), color="#0f766e", alpha=0.85)
                ax.axvline(vals.median(), color="#dc2626", linestyle="--", linewidth=1, label="median")
                ax.set_title(s, fontsize=10)
                ax.set_xlabel("Tokens")
                ax.set_ylabel("Chunks")
            for ax in axes[len(strategies):]:
                ax.axis("off")
            fig.suptitle("Token length histograms", y=1.0)
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)

            # 10 — violin plot
            if len(data) > 0:
                path = self.plots_dir / "10_token_length_violin.png"
                fig, ax = plt.subplots(figsize=_cap_figsize(max(11, len(labels) * 0.75), 6))
                parts = ax.violinplot(data, positions=range(1, len(data) + 1), showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#6366f1")
                    pc.set_alpha(0.7)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_ylabel("Tokens per chunk")
                ax.set_title("Token length distribution by strategy (violin)")
                fig.tight_layout()
                fig.savefig(path, dpi=120)
                plt.close(fig)
                paths.append(path)

            # 09 — cumulative distribution (CDF)
            path = self.plots_dir / "09_token_length_cdf.png"
            fig, ax = plt.subplots(figsize=(11, 6))
            cmap = plt.get_cmap("tab20", len(labels))
            for i, (s, arr) in enumerate(zip(labels, data)):
                sorted_arr = np.sort(arr)
                cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
                ax.plot(sorted_arr, cdf * 100, label=s, color=cmap(i), linewidth=1.5)
            ax.axvline(128,  color="#f59e0b", linestyle=":", linewidth=1, alpha=0.7, label="128 tok")
            ax.axvline(512,  color="#ef4444", linestyle=":", linewidth=1, alpha=0.7, label="512 tok")
            ax.axvline(1024, color="#991b1b", linestyle=":", linewidth=1, alpha=0.7, label="1024 tok")
            ax.set_xlabel("Token count")
            ax.set_ylabel("Cumulative % of chunks")
            ax.set_title("Chunk-length CDF by strategy")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)

        # 04 — coverage / overlap / missing
        path = self.plots_dir / "04_coverage_overlap_missing.png"
        d = metrics_df.sort_values("coverage_pct", ascending=True)
        y = np.arange(len(d))
        plt.figure(figsize=(11, max(4.5, len(d) * 0.45)))
        plt.barh(y, d["coverage_pct"], label="unique coverage %", color="#16a34a")
        plt.barh(y, d["overlap_pct_of_chunk_tokens"], left=d["coverage_pct"], label="overlap %", color="#f59e0b")
        plt.barh(y, 100 - d["coverage_pct"], left=d["coverage_pct"] + d["overlap_pct_of_chunk_tokens"], label="missing %", color="#ef4444")
        plt.yticks(y, d["strategy"])
        plt.xlabel("Percent")
        plt.title("Coverage, overlap, and missing-token profile")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        paths.append(path)

        # 05 — boundary alignment heatmap
        boundary_cols = [
            "start_on_page_boundary_pct", "end_on_page_boundary_pct",
            "start_on_block_boundary_pct", "end_on_block_boundary_pct",
            "start_on_line_boundary_pct", "end_on_line_boundary_pct",
            "starts_inside_block_pct", "ends_inside_block_pct",
        ]
        existing = [c for c in boundary_cols if c in metrics_df.columns]
        if existing:
            heat = metrics_df.set_index("strategy")[existing].fillna(0)
            path = self.plots_dir / "05_boundary_alignment_heatmap.png"
            plt.figure(figsize=_cap_figsize(max(11, len(existing) * 1.25), max(4.5, len(heat) * 0.45)))
            plt.imshow(heat.values, aspect="auto", cmap="viridis")
            plt.colorbar(label="Percent")
            plt.xticks(range(heat.shape[1]), heat.columns, rotation=45, ha="right")
            plt.yticks(range(heat.shape[0]), heat.index)
            plt.title("Boundary alignment")
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    plt.text(j, i, f"{heat.values[i, j]:.0f}", ha="center", va="center",
                             fontsize=8, color="white" if heat.values[i, j] < 55 else "black")
            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close()
            paths.append(path)

        # 06 — redundancy and page crossing
        path = self.plots_dir / "06_page_crossing_redundancy.png"
        d = metrics_df.sort_values("redundancy_factor", ascending=True)
        fig, ax1 = plt.subplots(figsize=(11, max(4.5, len(d) * 0.45)))
        y = np.arange(len(d))
        ax1.barh(y, d["redundancy_factor"], color="#6366f1", alpha=0.85, label="redundancy factor")
        ax1.set_yticks(y)
        ax1.set_yticklabels(d["strategy"])
        ax1.set_xlabel("Redundancy factor")
        ax2 = ax1.twiny()
        ax2.plot(d["page_crossing_pct"], y, "o-", color="#dc2626", label="page crossing %")
        ax2.set_xlabel("Page-crossing chunks (%)")
        ax1.set_title("Redundancy and page crossing")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)

        # 07 — per-page coverage heatmap
        if not page_coverage_df.empty:
            pivot = page_coverage_df.pivot_table(
                index="strategy", columns="page_number", values="coverage_pct", aggfunc="mean", fill_value=0,
            )
            path = self.plots_dir / "07_page_coverage_heatmap.png"
            plt.figure(figsize=_cap_figsize(max(10, pivot.shape[1] * 0.5), max(4.5, pivot.shape[0] * 0.45)))
            plt.imshow(pivot.values, aspect="auto", cmap="magma", vmin=0, vmax=100)
            plt.colorbar(label="Coverage %")
            plt.xticks(range(pivot.shape[1]), pivot.columns)
            plt.yticks(range(pivot.shape[0]), pivot.index)
            plt.xlabel("Page number")
            plt.title("Per-page token coverage by strategy")
            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close()
            paths.append(path)

        # 08 — chunk-size category stacked bar
        cat_cols = ["empty_chunks_lt_10", "tiny_chunks_lt_50", "small_chunks_lt_128",
                    "medium_chunks_128_to_511", "large_chunks_512_to_1024", "oversized_chunks_gt_1024"]
        cat_cols = [c for c in cat_cols if c in metrics_df.columns]
        if cat_cols:
            path = self.plots_dir / "08_chunk_size_categories_stacked.png"
            d = metrics_df.set_index("strategy")[cat_cols].copy()
            totals = d.sum(axis=1).replace(0, 1)
            d_pct = d.div(totals, axis=0) * 100

            colors = ["#94a3b8", "#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#7c3aed"]
            labels = ["< 10", "10–49", "50–127", "128–511", "512–1024", "> 1024"]

            fig, ax = plt.subplots(figsize=(11, max(4.5, len(d_pct) * 0.45)))
            left = np.zeros(len(d_pct))
            for col, color, lbl in zip(cat_cols, colors, labels):
                ax.barh(d_pct.index, d_pct[col], left=left, color=color, label=lbl)
                left += d_pct[col].values
            ax.set_xlabel("% of chunks")
            ax.set_title("Chunk size category distribution by strategy")
            ax.legend(title="Token range", fontsize=8, loc="lower right")
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)

        # 11 — Gini vs entropy scatter
        if "gini_coefficient" in metrics_df.columns and "entropy_bits" in metrics_df.columns:
            path = self.plots_dir / "11_gini_vs_entropy_scatter.png"
            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(
                metrics_df["gini_coefficient"],
                metrics_df["entropy_bits"],
                c=metrics_df["coverage_pct"],
                cmap="viridis", s=80,
            )
            plt.colorbar(sc, ax=ax, label="Coverage %")
            for _, row in metrics_df.iterrows():
                ax.annotate(
                    row["strategy"],
                    (row["gini_coefficient"], row["entropy_bits"]),
                    fontsize=7, ha="left", xytext=(4, 4), textcoords="offset points",
                )
            ax.set_xlabel("Gini coefficient (length inequality)")
            ax.set_ylabel("Shannon entropy (bits)")
            ax.set_title("Length diversity: Gini vs entropy\n(colour = coverage %)")
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)

        # 12 — IQR bar
        if "iqr_tokens" in metrics_df.columns:
            path = self.plots_dir / "12_iqr_and_std_comparison.png"
            d = metrics_df.sort_values("iqr_tokens", ascending=True)
            fig, ax = plt.subplots(figsize=(11, max(4.5, len(d) * 0.45)))
            y = np.arange(len(d))
            ax.barh(y - 0.2, d["iqr_tokens"], height=0.4, color="#3b82f6", label="IQR (P75−P25)")
            ax.barh(y + 0.2, d["std_tokens"],  height=0.4, color="#f59e0b", label="Std dev")
            ax.set_yticks(y)
            ax.set_yticklabels(d["strategy"])
            ax.set_xlabel("Tokens")
            ax.set_title("Spread of chunk lengths: IQR vs Std Dev")
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)

        return paths

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        metrics_df, lengths_df, page_coverage_df = self.compute_metrics(strategies=strategies)
        plot_paths = self.plot_all(metrics_df, lengths_df, page_coverage_df)

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "plots_dir": str(self.plots_dir),
            "strategy_count": int(len(metrics_df)),
            "metric_files": [
                str(self.metrics_dir / "chunking_metrics.csv"),
                str(self.metrics_dir / "chunk_token_lengths_long.csv"),
                str(self.metrics_dir / "chunk_page_coverage.csv"),
            ],
            "plots": [str(p) for p in plot_paths],
        }
        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

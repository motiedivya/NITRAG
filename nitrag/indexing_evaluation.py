from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _cap_figsize(w: float, h: float, max_w: float = 15.0, max_h: float = 12.0):
    return (min(max_w, w), min(max_h, h))


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_parquet_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def file_size(path: Union[str, Path]) -> int:
    path = Path(path)
    return int(path.stat().st_size) if path.exists() else 0


class IndexingEvaluationManager:
    """
    Focused evaluator for persisted indexing outputs.

    It checks rag_store/<doc_id>/indexes/<chunk_strategy>/<index_name>/ for:
      - expected default index coverage
      - required files and manifests
      - docs row consistency with source chunks
      - postings/vocab/edge/signature scale metrics
      - suspicious empty or inconsistent indexes
    """

    EXPECTED_FILES = {
        "bm25": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "keyword_inverted": ["manifest.json", "docs.parquet", "postings.parquet"],
        "metadata_inverted": ["manifest.json", "docs.parquet", "postings.parquet"],
        "tfidf": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "phrase_ngram": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "char_ngram": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "fielded_lexical": ["manifest.json", "docs.parquet", "postings.parquet"],
        "entity": ["manifest.json", "docs.parquet", "postings.parquet", "entity_types.parquet"],
        "section_page": ["manifest.json", "docs.parquet", "postings.parquet"],
        "chunk_graph": ["manifest.json", "docs.parquet", "edges.parquet"],
        "positional": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "boolean_set": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "temporal": ["manifest.json", "docs.parquet", "postings.parquet"],
        "layout_spatial": ["manifest.json", "docs.parquet", "postings.parquet"],
        "minhash_lsh": ["manifest.json", "docs.parquet", "signatures.parquet", "buckets.parquet", "candidate_pairs.parquet"],
        # new indexes
        "sentence_inverted": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
        "numeric_range": ["manifest.json", "docs.parquet", "postings.parquet"],
        "concept_cooccurrence": ["manifest.json", "docs.parquet", "postings.parquet", "vocab.parquet"],
    }

    NON_EMPTY_BY_TYPE = {
        "lexical": {
            "bm25", "keyword_inverted", "tfidf", "phrase_ngram", "char_ngram",
            "fielded_lexical", "positional", "boolean_set", "sentence_inverted",
        },
        "structural": {"metadata_inverted", "section_page", "chunk_graph", "layout_spatial", "minhash_lsh"},
        "optional_signal": {"entity", "temporal", "numeric_range", "concept_cooccurrence"},
    }

    def __init__(
        self,
        store_or_document_dir: Any,
        index_root_dir: Optional[Union[str, Path]] = None,
        chunk_dir: Optional[Union[str, Path]] = None,
        report_dir: Optional[Union[str, Path]] = None,
        expected_indexers: Optional[List[str]] = None,
    ):
        if hasattr(store_or_document_dir, "paths"):
            self.document_dir = Path(store_or_document_dir.paths.document_dir)
        else:
            self.document_dir = Path(store_or_document_dir)

        self.index_root_dir = Path(index_root_dir or self.document_dir / "indexes")
        self.chunk_dir = Path(chunk_dir or self.document_dir / "chunks_enriched")
        self.report_dir = Path(report_dir or self.document_dir / "reports" / "indexing_evaluation")
        self.metrics_dir = self.report_dir / "metrics"
        self.plots_dir = self.report_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.expected_indexers = expected_indexers or list(self.EXPECTED_FILES.keys())

    def list_chunk_strategies(self) -> List[str]:
        if self.chunk_dir.exists():
            return sorted(p.stem for p in self.chunk_dir.glob("*.parquet"))
        if self.index_root_dir.exists():
            return sorted(p.name for p in self.index_root_dir.iterdir() if p.is_dir())
        return []

    def list_index_dirs(self) -> List[Dict[str, Any]]:
        rows = []
        if not self.index_root_dir.exists():
            return rows

        for strategy_dir in sorted(p for p in self.index_root_dir.iterdir() if p.is_dir()):
            for index_dir in sorted(p for p in strategy_dir.iterdir() if p.is_dir()):
                rows.append({
                    "chunk_strategy": strategy_dir.name,
                    "index_name": index_dir.name,
                    "index_dir": index_dir,
                })
        return rows

    def compute_inventory(self) -> pd.DataFrame:
        existing = {
            (item["chunk_strategy"], item["index_name"]): item["index_dir"]
            for item in self.list_index_dirs()
        }

        rows = []
        for chunk_strategy in self.list_chunk_strategies():
            for index_name in self.expected_indexers:
                index_dir = existing.get((chunk_strategy, index_name))
                rows.append({
                    "chunk_strategy": chunk_strategy,
                    "index_name": index_name,
                    "exists": bool(index_dir),
                    "index_dir": str(index_dir) if index_dir else "",
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "index_inventory.csv")
        return out

    def compute_file_validation(self) -> pd.DataFrame:
        rows = []
        for item in self.list_index_dirs():
            index_dir = Path(item["index_dir"])
            index_name = item["index_name"]
            expected = self.EXPECTED_FILES.get(index_name, ["manifest.json", "docs.parquet"])

            for filename in expected:
                path = index_dir / filename
                rows.append({
                    "chunk_strategy": item["chunk_strategy"],
                    "index_name": index_name,
                    "file": filename,
                    "exists": path.exists(),
                    "size_bytes": file_size(path),
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "index_file_validation.csv")
        return out

    def compute_index_metrics(self) -> pd.DataFrame:
        rows = []
        for item in self.list_index_dirs():
            rows.append(self._index_metric_row(item["chunk_strategy"], item["index_name"], Path(item["index_dir"])))

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "index_metrics.csv")
        return out

    def find_suspicious_indexes(self) -> pd.DataFrame:
        inventory = self.compute_inventory()
        files = self.compute_file_validation()
        metrics = self.compute_index_metrics()
        rows = []

        for r in inventory.to_dict("records"):
            if not r["exists"]:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": r["index_name"],
                    "severity": "error",
                    "reason": "missing index directory",
                })

        for r in files.to_dict("records"):
            if not r["exists"]:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": r["index_name"],
                    "severity": "error",
                    "reason": f"missing file: {r['file']}",
                })
            elif int(r["size_bytes"] or 0) == 0:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": r["index_name"],
                    "severity": "error",
                    "reason": f"empty file: {r['file']}",
                })

        for r in metrics.to_dict("records"):
            index_name = r["index_name"]
            n_docs = int(r.get("docs_rows") or 0)
            source_chunks = int(r.get("source_chunk_rows") or 0)
            postings = int(r.get("postings_rows") or 0)
            edges = int(r.get("edges_rows") or 0)

            if source_chunks and n_docs != source_chunks:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": index_name,
                    "severity": "error",
                    "reason": f"docs row mismatch: docs={n_docs}, chunks={source_chunks}",
                })

            if index_name in self.NON_EMPTY_BY_TYPE["lexical"] and n_docs > 0 and postings == 0:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": index_name,
                    "severity": "error",
                    "reason": "lexical index has no postings",
                })

            if index_name == "chunk_graph" and n_docs > 1 and edges == 0:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": index_name,
                    "severity": "warning",
                    "reason": "chunk graph has no edges",
                })

            if index_name == "layout_spatial" and n_docs > 0 and postings == 0:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": index_name,
                    "severity": "warning",
                    "reason": "layout spatial index has no layout postings",
                })

            if index_name in self.NON_EMPTY_BY_TYPE["optional_signal"] and postings == 0:
                rows.append({
                    "chunk_strategy": r["chunk_strategy"],
                    "index_name": index_name,
                    "severity": "info",
                    "reason": "optional signal index has no postings",
                })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "suspicious_indexes.csv")
        return out

    def compute_scorecard(self) -> pd.DataFrame:
        inventory = self.compute_inventory()
        files = self.compute_file_validation()
        metrics = self.compute_index_metrics()
        suspicious = self.find_suspicious_indexes()

        suspicious_counts = (
            suspicious.groupby(["chunk_strategy", "index_name", "severity"])
            .size()
            .unstack(fill_value=0)
            if not suspicious.empty
            else pd.DataFrame()
        )

        rows = []
        for r in metrics.to_dict("records"):
            chunk_strategy = r["chunk_strategy"]
            index_name = r["index_name"]

            inv = inventory[
                (inventory["chunk_strategy"] == chunk_strategy)
                & (inventory["index_name"] == index_name)
            ]
            exists = bool(inv["exists"].iloc[0]) if not inv.empty else False

            file_slice = files[
                (files["chunk_strategy"] == chunk_strategy)
                & (files["index_name"] == index_name)
            ]
            required_file_count = int(len(file_slice))
            present_file_count = int(file_slice["exists"].sum()) if not file_slice.empty else 0
            file_score = present_file_count / max(1, required_file_count)

            docs_rows = int(r.get("docs_rows") or 0)
            source_rows = int(r.get("source_chunk_rows") or 0)
            docs_score = 1.0 if source_rows == 0 else max(0.0, 1.0 - abs(docs_rows - source_rows) / max(1, source_rows))

            postings_rows = int(r.get("postings_rows") or 0)
            edges_rows = int(r.get("edges_rows") or 0)
            signatures_rows = int(r.get("signatures_rows") or 0)
            signal_score = self._signal_score(index_name, docs_rows, postings_rows, edges_rows, signatures_rows)

            counts = suspicious_counts.loc[(chunk_strategy, index_name)].to_dict() if (
                not suspicious_counts.empty and (chunk_strategy, index_name) in suspicious_counts.index
            ) else {}
            error_count = int(counts.get("error", 0))
            warning_count = int(counts.get("warning", 0))
            info_count = int(counts.get("info", 0))

            penalty = min(1.0, error_count * 0.45 + warning_count * 0.15 + info_count * 0.03)
            score = max(0.0, min(1.0, 0.20 * float(exists) + 0.25 * file_score + 0.25 * docs_score + 0.30 * signal_score - penalty))

            rows.append({
                "chunk_strategy": chunk_strategy,
                "index_name": index_name,
                "health_score": round(float(score), 4),
                "exists": exists,
                "required_file_count": required_file_count,
                "present_file_count": present_file_count,
                "file_score": round(float(file_score), 4),
                "docs_score": round(float(docs_score), 4),
                "signal_score": round(float(signal_score), 4),
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "docs_rows": docs_rows,
                "source_chunk_rows": source_rows,
                "postings_rows": postings_rows,
                "edges_rows": edges_rows,
                "signatures_rows": signatures_rows,
                "total_size_bytes": int(r.get("total_size_bytes") or 0),
                "bytes_per_doc": float(r.get("bytes_per_doc") or 0.0),
            })

        out = pd.DataFrame(rows)
        save_df(out, self.metrics_dir / "index_scorecard.csv")
        return out

    def plot_all(
        self,
        inventory: Optional[pd.DataFrame] = None,
        metrics: Optional[pd.DataFrame] = None,
        scorecard: Optional[pd.DataFrame] = None,
        suspicious: Optional[pd.DataFrame] = None,
    ) -> List[Path]:
        inventory = inventory if inventory is not None else self.compute_inventory()
        metrics = metrics if metrics is not None else self.compute_index_metrics()
        scorecard = scorecard if scorecard is not None else self.compute_scorecard()
        suspicious = suspicious if suspicious is not None else self.find_suspicious_indexes()

        paths: List[Path] = []
        paths.extend(self._plot_inventory_heatmap(inventory))
        paths.extend(self._plot_metric_heatmap(metrics, "total_size_bytes", "02_index_size_heatmap.png", "Index size bytes"))
        paths.extend(self._plot_metric_heatmap(metrics, "postings_per_doc", "03_postings_per_doc_heatmap.png", "Postings per doc"))
        paths.extend(self._plot_scorecard(scorecard))
        paths.extend(self._plot_suspicious_counts(suspicious))
        paths.extend(self._plot_vocab_size_heatmap(metrics))
        paths.extend(self._plot_bytes_per_doc_bar(metrics))
        paths.extend(self._plot_vocab_fan_out_bar(metrics))
        return paths

    def generate_report(self) -> Dict[str, Any]:
        inventory = self.compute_inventory()
        files = self.compute_file_validation()
        metrics = self.compute_index_metrics()
        suspicious = self.find_suspicious_indexes()
        scorecard = self.compute_scorecard()
        plot_paths = self.plot_all(
            inventory=inventory,
            metrics=metrics,
            scorecard=scorecard,
            suspicious=suspicious,
        )

        summary = {
            "report_dir": str(self.report_dir),
            "metrics_dir": str(self.metrics_dir),
            "plots_dir": str(self.plots_dir),
            "chunk_strategy_count": int(len(self.list_chunk_strategies())),
            "expected_indexer_count": int(len(self.expected_indexers)),
            "existing_index_count": int(inventory["exists"].sum()) if not inventory.empty else 0,
            "expected_index_count": int(len(inventory)),
            "missing_index_count": int((~inventory["exists"]).sum()) if not inventory.empty else 0,
            "missing_required_file_count": int((~files["exists"]).sum()) if not files.empty else 0,
            "suspicious_index_rows": int(len(suspicious)),
            "mean_health_score": float(scorecard["health_score"].mean()) if not scorecard.empty else 0.0,
            "metric_files": [
                str(self.metrics_dir / "index_inventory.csv"),
                str(self.metrics_dir / "index_file_validation.csv"),
                str(self.metrics_dir / "index_metrics.csv"),
                str(self.metrics_dir / "suspicious_indexes.csv"),
                str(self.metrics_dir / "index_scorecard.csv"),
            ],
            "plots": [str(p) for p in plot_paths],
        }

        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def _signal_score(
        self,
        index_name: str,
        docs_rows: int,
        postings_rows: int,
        edges_rows: int,
        signatures_rows: int,
    ) -> float:
        if docs_rows == 0:
            return 0.0
        if index_name in self.NON_EMPTY_BY_TYPE["optional_signal"]:
            return 1.0 if postings_rows > 0 else 0.65
        if index_name == "chunk_graph":
            return 1.0 if docs_rows <= 1 or edges_rows > 0 else 0.0
        if index_name == "minhash_lsh":
            return 1.0 if signatures_rows == docs_rows else 0.0
        return 1.0 if postings_rows > 0 else 0.0

    def _plot_inventory_heatmap(self, inventory: pd.DataFrame) -> List[Path]:
        if inventory.empty:
            return []

        pivot = inventory.pivot_table(
            index="chunk_strategy",
            columns="index_name",
            values="exists",
            aggfunc="max",
            fill_value=False,
        ).astype(float)

        path = self.plots_dir / "01_index_inventory_heatmap.png"
        plt.figure(figsize=_cap_figsize(max(11, pivot.shape[1] * 0.7), max(4.5, pivot.shape[0] * 0.45)))
        plt.imshow(pivot.values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        plt.colorbar(label="Exists")
        plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(pivot.shape[0]), pivot.index)
        plt.title("Index inventory coverage")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return [path]

    def _plot_metric_heatmap(self, metrics: pd.DataFrame, metric: str, filename: str, title: str) -> List[Path]:
        if metrics.empty or metric not in metrics.columns:
            return []

        pivot = metrics.pivot_table(
            index="chunk_strategy",
            columns="index_name",
            values=metric,
            aggfunc="mean",
            fill_value=0,
        )

        values = np.log1p(pivot.values.astype(float)) if metric == "total_size_bytes" else pivot.values.astype(float)

        path = self.plots_dir / filename
        plt.figure(figsize=_cap_figsize(max(11, pivot.shape[1] * 0.7), max(4.5, pivot.shape[0] * 0.45)))
        plt.imshow(values, aspect="auto", cmap="viridis")
        plt.colorbar(label=("log1p(value)" if metric == "total_size_bytes" else "value"))
        plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(pivot.shape[0]), pivot.index)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return [path]

    def _plot_scorecard(self, scorecard: pd.DataFrame) -> List[Path]:
        if scorecard.empty:
            return []

        pivot = scorecard.pivot_table(
            index="chunk_strategy",
            columns="index_name",
            values="health_score",
            aggfunc="mean",
            fill_value=0,
        )

        path = self.plots_dir / "04_index_health_score_heatmap.png"
        plt.figure(figsize=_cap_figsize(max(11, pivot.shape[1] * 0.7), max(4.5, pivot.shape[0] * 0.45)))
        plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(label="Health score")
        plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(pivot.shape[0]), pivot.index)
        plt.title("Index health score")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()

        by_index = scorecard.groupby("index_name")["health_score"].mean().sort_values()
        path2 = self.plots_dir / "05_mean_health_by_index.png"
        plt.figure(figsize=_cap_figsize(10, max(4.5, len(by_index) * 0.35)))
        plt.barh(by_index.index, by_index.values, color="#2563eb")
        plt.xlim(0, 1)
        plt.xlabel("Mean health score")
        plt.title("Mean index health by index type")
        plt.tight_layout()
        plt.savefig(path2, dpi=120)
        plt.close()

        return [path, path2]

    def _plot_suspicious_counts(self, suspicious: pd.DataFrame) -> List[Path]:
        if suspicious.empty:
            return []

        counts = suspicious.groupby(["index_name", "severity"]).size().unstack(fill_value=0)
        for col in ["error", "warning", "info"]:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts[["error", "warning", "info"]].sort_values(["error", "warning", "info"])

        path = self.plots_dir / "06_suspicious_counts_by_index.png"
        plt.figure(figsize=_cap_figsize(10, max(4.5, len(counts) * 0.35)))
        y = np.arange(len(counts))
        left = np.zeros(len(counts))
        colors = {"error": "#dc2626", "warning": "#f59e0b", "info": "#64748b"}
        for severity in ["error", "warning", "info"]:
            vals = counts[severity].values
            plt.barh(y, vals, left=left, label=severity, color=colors[severity])
            left += vals
        plt.yticks(y, counts.index)
        plt.xlabel("Suspicious rows")
        plt.title("Suspicious index findings")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return [path]

    def _plot_vocab_size_heatmap(self, metrics: pd.DataFrame) -> List[Path]:
        if metrics.empty or "vocab_rows" not in metrics.columns:
            return []
        pivot = metrics.pivot_table(
            index="chunk_strategy", columns="index_name", values="vocab_rows", aggfunc="mean", fill_value=0,
        )
        if pivot.empty:
            return []
        path = self.plots_dir / "07_vocab_size_heatmap.png"
        fig, ax = plt.subplots(figsize=_cap_figsize(max(11, pivot.shape[1] * 0.7), max(4.5, pivot.shape[0] * 0.45)))
        values = np.log1p(pivot.values.astype(float))
        im = ax.imshow(values, aspect="auto", cmap="Blues")
        fig.colorbar(im, ax=ax, label="log1p(vocab_rows)")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = int(pivot.values[i, j])
                if v > 0:
                    ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6,
                            color="white" if values[i, j] > values.max() * 0.7 else "black")
        ax.set_title("Vocabulary size (unique terms) per index per strategy")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_bytes_per_doc_bar(self, metrics: pd.DataFrame) -> List[Path]:
        if metrics.empty or "bytes_per_doc" not in metrics.columns:
            return []
        avg = metrics.groupby("index_name")["bytes_per_doc"].mean().sort_values()
        if avg.empty:
            return []
        path = self.plots_dir / "08_bytes_per_doc_by_index.png"
        fig, ax = plt.subplots(figsize=_cap_figsize(10, max(4.5, len(avg) * 0.35)))
        ax.barh(avg.index, avg.values / 1024, color="#7c3aed")
        ax.set_xlabel("Avg KB per document")
        ax.set_title("Storage cost per document per index type")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _plot_vocab_fan_out_bar(self, metrics: pd.DataFrame) -> List[Path]:
        """vocab_fan_out = postings_rows / vocab_rows — how many (doc,pos) entries per unique term on average."""
        if metrics.empty or "vocab_fan_out" not in metrics.columns:
            return []
        m = metrics[metrics["vocab_rows"] > 0]
        if m.empty:
            return []
        avg = m.groupby("index_name")["vocab_fan_out"].mean().sort_values()
        path = self.plots_dir / "09_vocab_fan_out_by_index.png"
        fig, axes = plt.subplots(1, 2, figsize=_cap_figsize(14, max(4.5, len(avg) * 0.35)))

        axes[0].barh(avg.index, avg.values, color="#0891b2")
        axes[0].set_xlabel("Avg postings per vocab term")
        axes[0].set_title("Vocabulary fan-out (postings / vocab terms)")

        if "bytes_per_posting" in metrics.columns:
            bp = metrics[metrics["postings_rows"] > 0].groupby("index_name")["bytes_per_posting"].mean().sort_values()
            axes[1].barh(bp.index, bp.values, color="#dc2626")
            axes[1].set_xlabel("Avg bytes per posting row")
            axes[1].set_title("Storage density (bytes per posting)")
        else:
            axes[1].axis("off")

        fig.suptitle("Index compactness metrics")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return [path]

    def _source_chunk_count(self, chunk_strategy: str) -> int:
        path = self.chunk_dir / f"{chunk_strategy}.parquet"
        if not path.exists():
            return 0
        return int(pq.read_table(path, columns=[]).num_rows)

    def _index_metric_row(self, chunk_strategy: str, index_name: str, index_dir: Path) -> Dict[str, Any]:
        manifest = read_json(index_dir / "manifest.json")
        docs = read_parquet_df(index_dir / "docs.parquet")
        postings = read_parquet_df(index_dir / "postings.parquet")
        vocab = read_parquet_df(index_dir / "vocab.parquet")
        edges = read_parquet_df(index_dir / "edges.parquet")
        signatures = read_parquet_df(index_dir / "signatures.parquet")
        buckets = read_parquet_df(index_dir / "buckets.parquet")
        candidate_pairs = read_parquet_df(index_dir / "candidate_pairs.parquet")

        total_bytes = sum(file_size(p) for p in index_dir.glob("*") if p.is_file())
        docs_rows = int(len(docs))
        postings_rows = int(len(postings))

        vocab_rows = int(len(vocab))
        row = {
            "chunk_strategy": chunk_strategy,
            "index_name": index_name,
            "index_dir": str(index_dir),
            "source_chunk_rows": self._source_chunk_count(chunk_strategy),
            "docs_rows": docs_rows,
            "postings_rows": postings_rows,
            "vocab_rows": vocab_rows,
            "edges_rows": int(len(edges)),
            "signatures_rows": int(len(signatures)),
            "buckets_rows": int(len(buckets)),
            "candidate_pair_rows": int(len(candidate_pairs)),
            "total_size_bytes": int(total_bytes),
            "bytes_per_doc": float(total_bytes / max(1, docs_rows)),
            "bytes_per_posting": float(total_bytes / max(1, postings_rows)),
            "postings_per_doc": float(postings_rows / max(1, docs_rows)),
            "vocab_fan_out": float(postings_rows / max(1, vocab_rows)),
            "manifest_present": bool(manifest),
        }

        for key, value in manifest.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"manifest_{key}"] = value
            else:
                row[f"manifest_{key}"] = json.dumps(value, ensure_ascii=False, default=str)

        return row

#!/usr/bin/env python3
"""Benchmark runner for the Supplier Clustering Engine.

Reads all CSV/XLSX files from benchmarks/input/ (or --input-dir), runs the
full deterministic clustering pipeline on each, writes per-file outputs to
benchmarks/outputs/<file_stem>/, and writes a summary to
benchmarks/reports/benchmark_summary.csv.

No LLM calls are made (mode=disabled).  No answer sheets are required.
If a matching expected file exists in benchmarks/expected/, a simple
row-count and score-distribution comparison is logged; full diff is skipped.

Usage
-----
    python -m scripts.run_benchmark_pack
    python -m scripts.run_benchmark_pack --input-dir path/to/csvs
    python -m scripts.run_benchmark_pack --help
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the repo root to sys.path so src/ and scripts/ are importable when
# run as a module (python -m scripts.run_benchmark_pack).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import polars as pl

from src.config import ClusteringConfig
from src.input_reader import read_supplier_file
from src.main import cluster_suppliers
from src.run_metadata import build_run_metadata, save_run_metadata
from scripts.run_cli import auto_detect_columns


# ---------------------------------------------------------------------------
# Summary row helpers
# ---------------------------------------------------------------------------

def _score_distribution(df: pl.DataFrame) -> Dict[str, int]:
    """Return counts for 98%, 85%, 70%, and blank Match Percentage values."""
    if "Match Percentage" not in df.columns:
        return {}
    counts: Dict[str, int] = {"98%": 0, "85%": 0, "70%": 0, "blank": 0, "other": 0}
    for val in df["Match Percentage"].to_list():
        v = str(val or "").strip()
        if v in counts:
            counts[v] += 1
        elif v == "":
            counts["blank"] += 1
        else:
            counts["other"] += 1
    return counts


def _cluster_count(df: pl.DataFrame) -> int:
    if "Cluster Number" not in df.columns:
        return 0
    return df["Cluster Number"].drop_nulls().n_unique()


def _compare_expected(
    output_df: pl.DataFrame,
    expected_path: Path,
) -> str:
    """Light-touch comparison: just check row count and score distribution match.

    Returns a short status string ('ok', 'row_count_mismatch', 'score_drift',
    'expected_missing', 'error').  Full diff is intentionally not implemented here;
    flag discrepancies for the team to investigate manually.
    """
    if not expected_path.exists():
        return "expected_missing"
    try:
        expected_df = pl.read_csv(expected_path, infer_schema_length=0)
        if len(expected_df) != len(output_df):
            return f"row_count_mismatch(expected={len(expected_df)},got={len(output_df)})"
        if "Match Percentage" in expected_df.columns and "Match Percentage" in output_df.columns:
            e_dist = _score_distribution(expected_df)
            o_dist = _score_distribution(output_df)
            if e_dist != o_dist:
                return f"score_drift(expected={e_dist},got={o_dist})"
        return "ok"
    except Exception as exc:
        return f"error({exc})"


def _run_one_file(
    input_path: Path,
    output_dir: Path,
    expected_dir: Path,
) -> Dict[str, Any]:
    """Run clustering on one input file and return a summary row dict."""
    stem = input_path.stem
    file_output_dir = output_dir / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    summary: Dict[str, Any] = {
        "file": input_path.name,
        "status": "ok",
        "error": "",
        "input_rows": 0,
        "cluster_count": 0,
        "score_98": 0,
        "score_85": 0,
        "score_70": 0,
        "score_blank": 0,
        "score_other": 0,
        "candidate_pairs": 0,
        "candidate_cap_hit": False,
        "review_pairs_count": 0,
        "cluster_audit_count": 0,
        "runtime_seconds": 0.0,
        "expected_comparison": "expected_missing",
        "mapped_supplier_name": "",
        "mapped_address": "",
        "mapped_country": "",
    }

    try:
        df = read_supplier_file(str(input_path), str(input_path))
        summary["input_rows"] = len(df)

        mapping = auto_detect_columns(df.columns, sample_df=df)
        summary["mapped_supplier_name"] = mapping.get("supplier_name", "")
        summary["mapped_address"] = mapping.get("address", "")
        summary["mapped_country"] = mapping.get("country", "")

        config = ClusteringConfig()
        config.llm_enabled = False
        config.llm_execution_mode = "disabled"

        result = cluster_suppliers(df, mapping, config)
        main_df = result["main_df"]

        # Score distribution
        dist = _score_distribution(main_df)
        summary["score_98"] = dist.get("98%", 0)
        summary["score_85"] = dist.get("85%", 0)
        summary["score_70"] = dist.get("70%", 0)
        summary["score_blank"] = dist.get("blank", 0)
        summary["score_other"] = dist.get("other", 0)
        summary["cluster_count"] = _cluster_count(main_df)

        stats = result.get("stats", {})
        summary["candidate_pairs"] = stats.get("candidate_pairs", 0)
        summary["candidate_cap_hit"] = bool(stats.get("candidate_pairs_capped", False))

        review_cands = result.get("review_candidates") or []
        summary["review_pairs_count"] = len(review_cands)

        cluster_audit_df = result.get("cluster_audit_df")
        summary["cluster_audit_count"] = len(cluster_audit_df) if cluster_audit_df is not None else 0

        # Write outputs
        final_output_path = str(file_output_dir / "final_supplier_clustered.csv")
        main_df.write_csv(final_output_path)

        # run_metadata.json
        meta = build_run_metadata(
            mapping=mapping,
            input_row_count=len(df),
            config=config,
        )
        save_run_metadata(meta, str(file_output_dir / "run_metadata.json"))

        # Compare against expected
        expected_path = expected_dir / input_path.name
        summary["expected_comparison"] = _compare_expected(main_df, expected_path)

    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = str(exc)

    summary["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return summary


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "file", "status", "error",
    "input_rows", "cluster_count",
    "score_98", "score_85", "score_70", "score_blank", "score_other",
    "candidate_pairs", "candidate_cap_hit",
    "review_pairs_count", "cluster_audit_count",
    "runtime_seconds",
    "expected_comparison",
    "mapped_supplier_name", "mapped_address", "mapped_country",
]


def _write_summary(summaries: List[Dict[str, Any]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the clustering benchmark pack across all files in benchmarks/input/.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(_REPO_ROOT / "benchmarks" / "input"),
        help="Directory containing benchmark input CSV/XLSX files (default: benchmarks/input/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "benchmarks" / "outputs"),
        help="Root directory for per-file output folders (default: benchmarks/outputs/)",
    )
    parser.add_argument(
        "--expected-dir",
        default=str(_REPO_ROOT / "benchmarks" / "expected"),
        help="Directory containing optional expected-output CSV files (default: benchmarks/expected/)",
    )
    parser.add_argument(
        "--report-dir",
        default=str(_REPO_ROOT / "benchmarks" / "reports"),
        help="Directory for the benchmark_summary.csv report (default: benchmarks/reports/)",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    expected_dir = Path(args.expected_dir)
    report_dir = Path(args.report_dir)
    report_path = report_dir / "benchmark_summary.csv"

    # Collect input files
    input_files: List[Path] = []
    if input_dir.exists():
        for p in sorted(input_dir.iterdir()):
            if p.suffix.lower() in {".csv", ".xlsx"} and not p.name.startswith("."):
                input_files.append(p)

    if not input_files:
        print(
            f"No CSV/XLSX files found in {input_dir}. "
            "Add benchmark files to run comparisons.",
            flush=True,
        )
        # Still write an empty summary so CI/tests can verify the report path.
        _write_summary([], report_path)
        print(f"Empty benchmark summary written to {report_path}", flush=True)
        return 0

    print(f"Found {len(input_files)} benchmark file(s) in {input_dir}", flush=True)
    summaries: List[Dict[str, Any]] = []

    for i, f in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] {f.name} ...", flush=True)
        row = _run_one_file(f, output_dir, expected_dir)
        status_emoji = "✅" if row["status"] == "ok" else "❌"
        print(
            f"  {status_emoji} rows={row['input_rows']:,}  clusters={row['cluster_count']:,}  "
            f"98%={row['score_98']}  85%={row['score_85']}  70%={row['score_70']}  "
            f"blank={row['score_blank']}  pairs={row['candidate_pairs']:,}  "
            f"time={row['runtime_seconds']}s  expected={row['expected_comparison']}",
            flush=True,
        )
        if row["error"]:
            print(f"  ERROR: {row['error']}", flush=True)
        summaries.append(row)

    _write_summary(summaries, report_path)
    print(f"\n✅ Benchmark summary written to {report_path}", flush=True)

    errors = [r for r in summaries if r["status"] == "error"]
    if errors:
        print(f"\n⚠️  {len(errors)} file(s) failed:", flush=True)
        for r in errors:
            print(f"  {r['file']}: {r['error']}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

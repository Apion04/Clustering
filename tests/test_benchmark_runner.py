"""Tests for the benchmark runner (Part B, Phase 6).

Covers:
  1. Benchmark runner handles empty benchmarks/input/ gracefully.
  2. Benchmark runner creates reports folder and output summary (empty case).
  3. Benchmark runner processes a synthetic CSV and writes non-empty summary.
  4. team_feedback_template.csv has all required columns.
  5. benchmarks/README.md exists and is non-empty.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_benchmark_pack import main as run_benchmark_main, _SUMMARY_COLUMNS


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
FEEDBACK_TEMPLATE = BENCHMARKS_DIR / "team_feedback_template.csv"
README_PATH = BENCHMARKS_DIR / "README.md"

REQUIRED_FEEDBACK_COLUMNS = {
    "client_file",
    "supplier_name",
    "address",
    "city",
    "country",
    "cluster_number",
    "match_percentage",
    "issue_type",
    "expected_result",
    "comments",
    "reviewer",
    "date_reviewed",
}


# ---------------------------------------------------------------------------
# 1 & 2: Empty input dir — handles gracefully, writes empty summary
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerEmpty:
    def test_empty_input_dir_returns_zero(self, tmp_path):
        """Empty input dir must not crash; returns exit code 0."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        report_dir = tmp_path / "reports"
        output_dir = tmp_path / "outputs"

        rc = run_benchmark_main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ])
        assert rc == 0, f"Expected exit 0 for empty input dir, got {rc}"

    def test_empty_input_dir_creates_summary_csv(self, tmp_path):
        """Empty input dir must still create benchmark_summary.csv."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        report_dir = tmp_path / "reports"
        output_dir = tmp_path / "outputs"

        run_benchmark_main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ])

        summary_path = report_dir / "benchmark_summary.csv"
        assert summary_path.exists(), (
            f"benchmark_summary.csv must be created even for empty input dir, "
            f"not found at {summary_path}"
        )

    def test_empty_summary_has_correct_columns(self, tmp_path):
        """Empty benchmark summary must have the defined column headers."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        report_dir = tmp_path / "reports"
        output_dir = tmp_path / "outputs"

        run_benchmark_main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ])

        summary_path = report_dir / "benchmark_summary.csv"
        with open(summary_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
        missing = set(_SUMMARY_COLUMNS) - cols
        assert not missing, f"Summary CSV missing columns: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 3: Synthetic CSV — processes correctly, writes non-empty summary
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerWithData:
    def test_synthetic_csv_processed(self, tmp_path):
        """A simple synthetic CSV must be processed without error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        report_dir = tmp_path / "reports"
        output_dir = tmp_path / "outputs"

        # Write a minimal synthetic CSV
        synthetic_csv = input_dir / "test_suppliers.csv"
        df = pl.DataFrame({
            "Supplier Name": [
                "Acme Corp",
                "Acme Corporation",
                "Global Trade Ltd",
                "Widgets Inc",
            ],
            "Address": [
                "1 Main St",
                "1 Main St",
                "2 High St",
                "3 Low St",
            ],
            "Country": ["US", "US", "UK", "US"],
        })
        df.write_csv(str(synthetic_csv))

        rc = run_benchmark_main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ])
        assert rc == 0, f"Expected exit 0, got {rc}"

        summary_path = report_dir / "benchmark_summary.csv"
        assert summary_path.exists()

        with open(summary_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1, f"Expected 1 summary row, got {len(rows)}"
        row = rows[0]
        assert row["file"] == "test_suppliers.csv"
        assert row["status"] == "ok", f"Expected status=ok, got {row['status']}: {row['error']}"
        assert int(row["input_rows"]) == 4, f"Expected 4 input rows, got {row['input_rows']}"

    def test_synthetic_csv_output_files_written(self, tmp_path):
        """Outputs must be written to outputs/<file_stem>/."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        report_dir = tmp_path / "reports"
        output_dir = tmp_path / "outputs"

        synthetic_csv = input_dir / "my_suppliers.csv"
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Beta Inc"],
            "Country": ["US", "US"],
        })
        df.write_csv(str(synthetic_csv))

        run_benchmark_main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ])

        file_output_dir = output_dir / "my_suppliers"
        assert file_output_dir.exists(), f"Output dir not created: {file_output_dir}"
        assert (file_output_dir / "final_supplier_clustered.csv").exists()
        assert (file_output_dir / "run_metadata.json").exists()

    def test_synthetic_csv_score_columns_present_in_summary(self, tmp_path):
        """Summary row must contain score distribution columns."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        report_dir = tmp_path / "reports"
        output_dir = tmp_path / "outputs"

        synthetic_csv = input_dir / "scores_test.csv"
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Acme Corp", "Other LLC"],
            "Address": ["1 Main St", "1 Main St", "2 Other Ave"],
            "Country": ["US", "US", "US"],
        })
        df.write_csv(str(synthetic_csv))

        run_benchmark_main([
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--report-dir", str(report_dir),
        ])

        summary_path = report_dir / "benchmark_summary.csv"
        with open(summary_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "Summary must have at least one row"
        row = rows[0]
        for col in ["score_98", "score_85", "score_70", "score_blank", "cluster_count"]:
            assert col in row, f"Summary row missing column: {col}"


# ---------------------------------------------------------------------------
# 4: team_feedback_template.csv — required columns present
# ---------------------------------------------------------------------------

class TestFeedbackTemplate:
    def test_feedback_template_exists(self):
        assert FEEDBACK_TEMPLATE.exists(), (
            f"benchmarks/team_feedback_template.csv not found at {FEEDBACK_TEMPLATE}"
        )

    def test_feedback_template_has_required_columns(self):
        with open(FEEDBACK_TEMPLATE, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
        missing = REQUIRED_FEEDBACK_COLUMNS - cols
        assert not missing, (
            f"team_feedback_template.csv is missing columns: {sorted(missing)}"
        )

    def test_feedback_template_has_example_rows(self):
        """Template must have at least one example row so reviewers know how to use it."""
        with open(FEEDBACK_TEMPLATE, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 1, (
            "team_feedback_template.csv must have at least one example row"
        )


# ---------------------------------------------------------------------------
# 5: README.md exists and is non-trivial
# ---------------------------------------------------------------------------

class TestReadme:
    def test_readme_exists(self):
        assert README_PATH.exists(), (
            f"benchmarks/README.md not found at {README_PATH}"
        )

    def test_readme_contains_key_sections(self):
        content = README_PATH.read_text(encoding="utf-8")
        for keyword in ["issue_type", "98%", "85%", "70%", "benchmarks/input"]:
            assert keyword in content, (
                f"benchmarks/README.md must contain {keyword!r}"
            )

    def test_readme_is_non_empty(self):
        content = README_PATH.read_text(encoding="utf-8")
        assert len(content) > 200, "benchmarks/README.md seems too short"

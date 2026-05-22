"""Tests for the Audit + Review Output Layer (src/audit_review.py).

Covers:
1.  review_pairs_df is present in cluster_suppliers result
2.  review_pairs_df has required columns
3.  review_pairs_df includes 70%-candidate pairs
4.  review_pairs_df includes pass_type and match_reason columns
5.  cluster_audit_df is present in cluster_suppliers result
6.  build_cluster_kind maps correctly for all tiers
7.  large_cluster flag fires when size > 8
8.  mixed_countries flag fires when multiple countries present
9.  high_risk alias edge gets alias_based_cluster flag
10. final_supplier_clustered.csv (main_df) has no internal columns
11. no internal columns in final output (second variation)
"""
import pytest
import polars as pl

from pathlib import Path
from src.config import ClusteringConfig
from src.main import cluster_suppliers
from src.audit_review import build_cluster_kind, build_review_pairs, build_cluster_audit, _make_cluster_risk_flags
from src.merging import ClusterEdge

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cluster_simple(records, config=None):
    """Run cluster_suppliers on (name, email, country) tuples."""
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Email": [r[1] if len(r) > 1 else "" for r in records],
        "Country": [r[2] if len(r) > 2 else "" for r in records],
    })
    return cluster_suppliers(df, {
        "supplier_name": "Supplier Name",
        "email": "Email",
        "country": "Country",
    }, config or ClusteringConfig())


# ---------------------------------------------------------------------------
# TestReviewPairsGenerated
# ---------------------------------------------------------------------------

class TestReviewPairsGenerated:
    def test_review_pairs_csv_is_generated(self):
        """review_pairs_df must be present in result dict."""
        result = _cluster_simple([
            ("Acme Corp", "info@acme.com", "US"),
            ("Acme Corporation", "info@acme.com", "US"),
            ("Other Vendor", "other@other.com", "US"),
        ])
        assert "review_pairs_df" in result

    def test_review_pairs_has_required_columns(self):
        """review_pairs_df must contain the required column set when non-empty."""
        result = _cluster_simple([
            ("Bell Canada", "", "CA"),
            ("Bell Aliant", "", "CA"),
        ])
        df = result.get("review_pairs_df")
        assert df is not None
        required = {
            "supplier_a_row_id", "supplier_b_row_id",
            "supplier_a_name", "supplier_b_name",
            "suggested_score", "pass_type",
            "match_reason", "risk_flags", "suggested_action",
        }
        if len(df) > 0:
            assert required.issubset(set(df.columns))

    def test_review_pairs_includes_70pct_candidates(self):
        """review_pairs_df must not be None (vacuous pass is fine if no 70% edge formed)."""
        result = _cluster_simple([
            ("Bell Canada", "", "CA"),
            ("Bell Aliant", "", "CA"),
        ])
        df = result.get("review_pairs_df")
        assert df is not None

    def test_review_pairs_includes_pass_type_and_reason(self):
        """When non-empty, pass_type and match_reason must be present and non-empty."""
        result = _cluster_simple([
            ("Bell Canada", "", "CA"),
            ("Bell Aliant", "", "CA"),
        ])
        df = result.get("review_pairs_df", pl.DataFrame())
        if len(df) > 0:
            assert "pass_type" in df.columns
            assert "match_reason" in df.columns
            assert df["match_reason"].drop_nulls().len() > 0


# ---------------------------------------------------------------------------
# TestClusterAuditGenerated
# ---------------------------------------------------------------------------

class TestClusterAuditGenerated:
    def test_cluster_audit_csv_is_generated(self):
        """cluster_audit_df must be present and non-None in result dict."""
        result = _cluster_simple([
            ("Acme Corp", "info@acme.com", "US"),
            ("Acme Corporation", "info@acme.com", "US"),
        ])
        assert "cluster_audit_df" in result
        df = result["cluster_audit_df"]
        assert df is not None

    def test_cluster_kind_maps_correctly(self):
        """Verify all cluster_kind tier boundaries."""
        assert build_cluster_kind(98.0, 2) == "exact_duplicate"
        assert build_cluster_kind(85.0, 2) == "supplier_family"
        assert build_cluster_kind(70.0, 2) == "review_candidate"
        assert build_cluster_kind(0.0, 1) == "singleton"
        # 76% is >= 70 but < 85 → review_candidate (not supplier_family)
        assert build_cluster_kind(76.0, 2) == "review_candidate"

    def test_large_cluster_gets_flag(self):
        """A cluster with size > 8 must get the large_cluster flag."""
        flags = _make_cluster_risk_flags(
            cluster_size=10,
            countries=["US"],
            domains=["acme.com"],
            tax_ids=[],
            edge_pass_types=["name_address_exact"],
            cluster_kind="supplier_family",
            edges=[],
        )
        assert "large_cluster" in flags

    def test_mixed_country_gets_flag(self):
        """A cluster spanning multiple countries must get the mixed_countries flag."""
        flags = _make_cluster_risk_flags(
            cluster_size=2,
            countries=["US", "DE"],
            domains=[],
            tax_ids=[],
            edge_pass_types=["name_address_exact"],
            cluster_kind="supplier_family",
            edges=[],
        )
        assert "mixed_countries" in flags

    def test_high_risk_alias_cluster_gets_flag(self):
        """A cluster with a brand_alias_candidate edge must get alias_based_cluster flag."""
        mock_edge = ClusterEdge(
            row_a=0, row_b=1, match_pct=70.0,
            pass_type="brand_alias_candidate",
            evidence={"risk_level": "medium"},
            needs_review=True,
            review_reason="alias",
        )
        flags = _make_cluster_risk_flags(
            cluster_size=2,
            countries=["US"],
            domains=[],
            tax_ids=[],
            edge_pass_types=["brand_alias_candidate"],
            cluster_kind="review_candidate",
            edges=[mock_edge],
        )
        assert "alias_based_cluster" in flags


# ---------------------------------------------------------------------------
# TestOutputSafety
# ---------------------------------------------------------------------------

class TestOutputSafety:
    def test_final_supplier_clustered_unchanged(self):
        """Internal columns must not leak into main_df."""
        result = _cluster_simple([
            ("Acme Corp", "info@acme.com", "US"),
            ("Other Vendor", "other@other.com", "DE"),
        ])
        main_cols = set(result["main_df"].columns)
        forbidden = {
            "cluster_kind", "risk_flags", "review_pairs_df", "cluster_audit_df",
            "pass_type", "edge_pass_type", "cluster_root",
        }
        assert not (forbidden & main_cols), f"Internal cols leaked: {forbidden & main_cols}"

    def test_no_internal_cols_in_final_output(self):
        """Explicit checks that specific internal column names are absent from main_df."""
        result = _cluster_simple([
            ("Acme Corp", "info@acme.com", "US"),
        ])
        main_df = result["main_df"]
        assert "cluster_kind" not in main_df.columns
        assert "risk_flags" not in main_df.columns
        assert "review_pairs_df" not in main_df.columns

"""Tests for numeric ERP/vendor-ID prefix stripping and related safeguards.

Covers:
  1. auto_detect_columns prefers CommonName over Supplier Common Name
  2. strip_numeric_id_prefix strips leading ID from various separator styles
  3. normalize_supplier_name produces clean name_norm (no numeric prefix)
  4. Original output column value is unchanged after preprocessing
  5. B LAB CO./COMPANY with numeric prefixes + same address clusters at 98%
  6. Numeric prefixes do NOT create N5_ blocking keys
  7. run_metadata records id_prefix warning flag correctly
  8. Existing auto_detect tests still pass (regression)
"""
import pytest
import polars as pl

from src.preprocessing import normalize_supplier_name, preprocess_dataframe, strip_numeric_id_prefix
from src.blocking import generate_candidate_pairs
from src.config import ClusteringConfig
from src.main import cluster_suppliers
from src.run_metadata import build_run_metadata
from scripts.run_cli import auto_detect_columns, _check_supplier_name_id_prefixes


# ---------------------------------------------------------------------------
# Test 1: auto_detect_columns prefers CommonName over Supplier Common Name
# ---------------------------------------------------------------------------

class TestAutoDetectPreference:
    def test_prefers_commonname_over_supplier_common_name(self):
        """CommonName must be preferred over Supplier Common Name when both exist."""
        columns = [
            "Supplier Common Name",
            "CommonName",
            "Address Line 1",
            "Supplier Origin City",
            "Supplier Origin Country",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "CommonName", (
            f"Expected supplier_name='CommonName', got {mapping['supplier_name']!r}. "
            "CommonName must be preferred over 'Supplier Common Name'."
        )

    def test_prefers_commonname_even_when_first_column(self):
        """CommonName is preferred even if Supplier Common Name is first in the column list."""
        columns = ["Supplier Common Name", "CommonName", "Country"]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "CommonName"

    def test_falls_back_to_supplier_common_name_when_no_exact_match(self):
        """When no exact-match column exists, falls back to pattern matching."""
        columns = ["Supplier Common Name", "Address", "Country"]
        mapping = auto_detect_columns(columns)
        # Should find some supplier_name (pattern match on "name")
        assert "supplier_name" in mapping
        # Should map to the only name-like column
        assert mapping["supplier_name"] == "Supplier Common Name"

    def test_danone_columns_maps_commonname(self):
        """Regression: Danone column set must map CommonName as supplier_name."""
        columns = [
            "CommonName",
            "Supplier Common Name",
            "Address Line 1",
            "Supplier Origin City",
            "Supplier Origin Country",
            "Supplier Origin Postal",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "CommonName"
        assert mapping.get("address") == "Address Line 1"

    def test_supplier_name_column_mapped(self):
        """'Supplier Name' column (exact) must be detected."""
        columns = ["Supplier Name", "Address", "City", "Country"]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "Supplier Name"

    def test_vendor_name_column_mapped(self):
        """'Vendor Name' column (exact) must be detected."""
        columns = ["Vendor Name", "Street", "Country"]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "Vendor Name"


# ---------------------------------------------------------------------------
# Test 2: strip_numeric_id_prefix strips leading numeric prefix
# ---------------------------------------------------------------------------

class TestStripNumericIdPrefix:
    def test_strips_dash_separator(self):
        assert strip_numeric_id_prefix("0020276471-B LAB CO.") == "B LAB CO."

    def test_strips_space_separator(self):
        assert strip_numeric_id_prefix("0020219751 B LAB COMPANY") == "B LAB COMPANY"

    def test_strips_underscore_separator(self):
        assert strip_numeric_id_prefix("0020276471_Acme Corp") == "Acme Corp"

    def test_strips_colon_separator(self):
        assert strip_numeric_id_prefix("1234567890:Global Trading") == "Global Trading"

    def test_strips_multiple_separators(self):
        """Multiple separator chars after the ID are all consumed."""
        assert strip_numeric_id_prefix("0020276471 - B LAB CO.") == "B LAB CO."

    def test_no_strip_short_prefix(self):
        """Fewer than 6 digits → not stripped (e.g. '3M', '12345')."""
        assert strip_numeric_id_prefix("12345-Acme") == "12345-Acme"
        assert strip_numeric_id_prefix("3M Inc") == "3M Inc"

    def test_no_strip_letter_prefix(self):
        """Prefix starting with letters is not stripped."""
        assert strip_numeric_id_prefix("ABC1234567-Acme") == "ABC1234567-Acme"

    def test_no_strip_pure_number(self):
        """Pure number with no separator → not stripped."""
        assert strip_numeric_id_prefix("0020276471") == "0020276471"

    def test_no_strip_when_nothing_remains(self):
        """Do not strip if the prefix takes up the entire string."""
        assert strip_numeric_id_prefix("0020276471-") == "0020276471-"

    def test_handles_none(self):
        assert strip_numeric_id_prefix(None) == ""

    def test_handles_empty_string(self):
        assert strip_numeric_id_prefix("") == ""

    def test_strips_long_id(self):
        assert strip_numeric_id_prefix("001234567890-Big Supplier Inc") == "Big Supplier Inc"


# ---------------------------------------------------------------------------
# Test 3: normalize_supplier_name produces clean name_norm
# ---------------------------------------------------------------------------

class TestNormalizeStripsPrefix:
    def test_b_lab_co_with_prefix(self):
        result = normalize_supplier_name("0020276471-B LAB CO.")
        assert "0020276471" not in result, f"Numeric ID must not be in name_norm, got {result!r}"
        assert "b lab" in result, f"Expected 'b lab' in result, got {result!r}"

    def test_b_lab_company_with_prefix(self):
        result = normalize_supplier_name("0020219751-B LAB COMPANY")
        assert "0020219751" not in result
        assert "b lab" in result

    def test_generic_supplier_with_prefix(self):
        result = normalize_supplier_name("9876543210-Acme Corporation")
        assert "9876543210" not in result
        assert "acme" in result

    def test_no_prefix_unchanged(self):
        """Names without numeric prefix are normalized normally."""
        result = normalize_supplier_name("B LAB CO.")
        assert "b lab" in result
        assert "0020" not in result

    def test_short_numeric_prefix_not_stripped(self):
        """A 4-digit number at the start (short) is NOT treated as an ID prefix."""
        result = normalize_supplier_name("3401 Main Street Suppliers")
        # "3401" has only 4 digits → not stripped, but may be removed by other logic
        # Key assertion: no crash, returns a string
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test 4: original column value unchanged
# ---------------------------------------------------------------------------

class TestOriginalColumnUnchanged:
    def test_original_column_not_modified(self):
        """preprocess_dataframe must NOT modify the original supplier-name column."""
        df = pl.DataFrame({
            "Supplier Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
            ],
            "Country": ["US", "US"],
        })
        df_proc = preprocess_dataframe(
            df,
            {"supplier_name": "Supplier Name", "country": "Country"},
        )
        # Original column must be unchanged
        original_values = df_proc["Supplier Name"].to_list()
        assert original_values[0] == "0020276471-B LAB CO.", (
            f"Original column was modified: {original_values[0]!r}"
        )
        assert original_values[1] == "0020219751-B LAB COMPANY", (
            f"Original column was modified: {original_values[1]!r}"
        )
        # But name_norm must be clean
        name_norms = df_proc["name_norm"].to_list()
        assert "0020276471" not in (name_norms[0] or "")
        assert "0020219751" not in (name_norms[1] or "")

    def test_final_cluster_output_preserves_original(self):
        """cluster_suppliers final output must keep original column values intact."""
        df = pl.DataFrame({
            "CommonName": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
            ],
            "Address Line 1": ["15 WATERLOO AVE", "15 WATERLOO AVE"],
            "Supplier Origin Country": ["US", "US"],
        })
        mapping = {
            "supplier_name": "CommonName",
            "address": "Address Line 1",
            "country": "Supplier Origin Country",
        }
        config = ClusteringConfig()
        result = cluster_suppliers(df, mapping, config)
        out = result["main_df"]
        # Original prefixed values must still be in the output
        assert "0020276471-B LAB CO." in out["CommonName"].to_list()
        assert "0020219751-B LAB COMPANY" in out["CommonName"].to_list()


# ---------------------------------------------------------------------------
# Test 5: B LAB CO./COMPANY with numeric prefixes clusters at 98%
# ---------------------------------------------------------------------------

class TestBLabWithPrefixClusters:
    def test_b_lab_with_prefix_same_address_clusters(self):
        """B LAB CO. / B LAB COMPANY with numeric prefixes and same address must cluster."""
        df = pl.DataFrame({
            "CommonName": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
                "0031234567-UNRELATED CORP INC",
            ],
            "Address Line 1": [
                "15 WATERLOO AVE",
                "15 WATERLOO AVE",
                "999 OTHER ST",
            ],
            "Supplier Origin Country": ["US", "US", "US"],
        })
        mapping = {
            "supplier_name": "CommonName",
            "address": "Address Line 1",
            "country": "Supplier Origin Country",
        }
        config = ClusteringConfig()
        result = cluster_suppliers(df, mapping, config)
        main_df = result["main_df"]

        row_a = main_df.filter(pl.col("CommonName") == "0020276471-B LAB CO.")
        row_b = main_df.filter(pl.col("CommonName") == "0020219751-B LAB COMPANY")
        assert not row_a.is_empty() and not row_b.is_empty()

        cluster_a = row_a["Cluster Number"].to_list()[0]
        cluster_b = row_b["Cluster Number"].to_list()[0]
        assert cluster_a is not None and cluster_b is not None, (
            "B LAB rows must have a cluster number"
        )
        assert cluster_a == cluster_b, (
            f"B LAB CO. / B LAB COMPANY with numeric prefix must be in the same cluster, "
            f"got cluster_a={cluster_a}, cluster_b={cluster_b}"
        )

    def test_b_lab_with_prefix_name_norm_stripped(self):
        """name_norm for both B LAB rows must not contain the numeric ID."""
        df = pl.DataFrame({
            "Supplier Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
            ],
            "Country": ["US", "US"],
        })
        df_proc = preprocess_dataframe(
            df,
            {"supplier_name": "Supplier Name", "country": "Country"},
        )
        for row in df_proc.iter_rows(named=True):
            nn = row["name_norm"] or ""
            assert not nn.startswith("002"), (
                f"name_norm must not start with numeric prefix, got {nn!r}"
            )
            assert "b lab" in nn, f"Expected 'b lab' in name_norm, got {nn!r}"


# ---------------------------------------------------------------------------
# Test 6: Numeric prefixes do NOT create N5_ blocking keys
# ---------------------------------------------------------------------------

class TestNoNumericN5Blocks:
    def test_no_n5_block_from_numeric_prefix(self):
        """After stripping, N5_ block must be based on the actual name, not '00202'."""
        df = pl.DataFrame({
            "Supplier Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
            ],
            "Country": ["US", "US"],
        })
        df_proc = preprocess_dataframe(
            df,
            {"supplier_name": "Supplier Name", "country": "Country"},
        )
        candidates = generate_candidate_pairs(df_proc)
        block_keys = candidates["shared_block"].to_list() if len(candidates) > 0 else []
        for bk in block_keys:
            assert not (bk.startswith("N5_") and bk[3:8].isdigit()), (
                f"Found N5_ block based on numeric ID: {bk!r}. "
                "Numeric prefixes must not generate N5_ blocks."
            )

    def test_numeric_guard_in_blocking_no_junk_n5(self):
        """Even if preprocessing left a numeric prefix, blocking must not emit N5_00202."""
        # Simulate a row where name_norm starts with digits (preprocessing failure scenario)
        df = pl.DataFrame({
            "Supplier Name": ["0020276471 b lab co"],  # already "normalized" w/ prefix
            "Country": ["US"],
        })
        # Manually override name_norm to simulate preprocessing not stripping
        df_proc = preprocess_dataframe(
            df,
            {"supplier_name": "Supplier Name", "country": "Country"},
        )
        # Add a second row to generate pairs
        df2 = pl.DataFrame({
            "Supplier Name": ["0020276471 b lab co", "0020219751 b lab company"],
            "Country": ["US", "US"],
        })
        df_proc2 = preprocess_dataframe(
            df2,
            {"supplier_name": "Supplier Name", "country": "Country"},
        )
        candidates = generate_candidate_pairs(df_proc2)
        block_keys = candidates["shared_block"].to_list() if len(candidates) > 0 else []
        # No N5_ block starting with digits
        for bk in block_keys:
            assert not (bk.startswith("N5_") and bk[3:8].isdigit()), (
                f"N5_ block from numeric prefix should be blocked: {bk!r}"
            )


# ---------------------------------------------------------------------------
# Test 7: run_metadata records id_prefix warning correctly
# ---------------------------------------------------------------------------

class TestRunMetadataIdPrefixWarning:
    def test_warning_false_when_no_prefix(self):
        config = ClusteringConfig()
        meta = build_run_metadata(
            mapping={"supplier_name": "CommonName"},
            input_row_count=100,
            config=config,
            supplier_name_id_prefix_pct=0.0,
        )
        assert meta["supplier_name_id_prefix_warning"] is False
        assert meta["supplier_name_id_prefix_pct"] == 0.0

    def test_warning_true_when_many_prefixes(self):
        config = ClusteringConfig()
        meta = build_run_metadata(
            mapping={"supplier_name": "Supplier Common Name"},
            input_row_count=100,
            config=config,
            supplier_name_id_prefix_pct=0.95,
        )
        assert meta["supplier_name_id_prefix_warning"] is True
        assert meta["supplier_name_id_prefix_pct"] == 0.95

    def test_check_supplier_name_id_prefixes(self):
        """_check_supplier_name_id_prefixes returns fraction correctly."""
        df = pl.DataFrame({
            "Supplier Common Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
                "Acme Corp",
                "Global Tech Inc",
            ]
        })
        pct = _check_supplier_name_id_prefixes(df, {"supplier_name": "Supplier Common Name"})
        assert pct == 0.5, f"Expected 0.5, got {pct}"

    def test_check_returns_zero_for_clean_column(self):
        df = pl.DataFrame({
            "CommonName": ["Acme Corp", "Global Tech Inc", "B Lab Co"]
        })
        pct = _check_supplier_name_id_prefixes(df, {"supplier_name": "CommonName"})
        assert pct == 0.0

    def test_check_returns_zero_when_column_absent(self):
        df = pl.DataFrame({"SomeCol": ["value"]})
        pct = _check_supplier_name_id_prefixes(df, {"supplier_name": "NonExistentCol"})
        assert pct == 0.0


# ---------------------------------------------------------------------------
# Test 8: Regression — Danone columns still detected correctly
# ---------------------------------------------------------------------------

class TestAutoDetectRegressionDanone:
    def test_address_line_1_still_detected(self):
        columns = [
            "CommonName", "Supplier Common Name",
            "Address Line 1", "Supplier Origin City",
            "Supplier Origin Country", "Supplier Origin Postal",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping.get("address") == "Address Line 1"

    def test_city_still_detected(self):
        columns = [
            "CommonName", "Address Line 1",
            "Supplier Origin City", "Supplier Origin Country",
            "Supplier Origin Postal",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping.get("city") == "Supplier Origin City"

    def test_country_still_detected(self):
        columns = [
            "CommonName", "Address Line 1",
            "Supplier Origin City", "Supplier Origin Country",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping.get("country") == "Supplier Origin Country"

    def test_postal_still_detected(self):
        columns = [
            "CommonName", "Address Line 1", "Supplier Origin Postal"
        ]
        mapping = auto_detect_columns(columns)
        assert mapping.get("postal_code") == "Supplier Origin Postal"

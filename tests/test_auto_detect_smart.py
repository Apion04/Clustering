"""Tests for data-quality-aware auto_detect_columns (Phase 5d).

Covers:
  1.  Danone: CommonName chosen over Supplier Common Name because SCN has
      numeric ERP/vendor-ID prefixes
  2.  Vendor Name chosen when it is the cleanest column
  3.  Supplier Name chosen when it is the cleanest column
  4.  Supplier Common Name chosen when clean and CommonName absent/worse
  5.  Company Name chosen when it is the cleanest column
  6.  Supplier ID / Vendor ID / legalEntityId / clusterId — never chosen
  7.  Mostly-numeric column not chosen as supplier_name
  8.  Mostly-blank column not chosen as supplier_name
  9.  Numeric prefix stripping still works when polluted column is chosen
 10.  Original output column values are unchanged after full cluster_suppliers call
 11.  Column-name-only fallback (no sample_df) still uses priority list
"""
import polars as pl
import pytest

from scripts.run_cli import (
    auto_detect_columns,
    _is_id_column,
    _find_supplier_name_candidates,
    _score_name_column,
)
from src.preprocessing import preprocess_dataframe
from src.main import cluster_suppliers
from src.config import ClusteringConfig


# ---------------------------------------------------------------------------
# 1.  Danone-shaped data: CommonName beats Supplier Common Name
# ---------------------------------------------------------------------------

class TestDanonePreference:
    """When sample_df is provided, CommonName must win because Supplier Common
    Name has numeric ERP/vendor-ID prefixes that heavily penalise its score."""

    def _make_df(self):
        return pl.DataFrame({
            "CommonName": [
                "B LAB CO.",
                "B LAB COMPANY",
                "UNRELATED CORP",
                "GLOBAL TRADE LTD",
                "ACME INC",
            ],
            "Supplier Common Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
                "0031234567-UNRELATED CORP",
                "0041234567-GLOBAL TRADE LTD",
                "0051234567-ACME INC",
            ],
            "Address Line 1": [
                "15 WATERLOO AVE", "15 WATERLOO AVE",
                "999 OTHER ST", "1 MAIN ST", "2 BROAD ST",
            ],
            "Supplier Origin City": [
                "LONDON", "LONDON", "PARIS", "BERLIN", "ROME",
            ],
            "Supplier Origin Country": [
                "GB", "GB", "FR", "DE", "IT",
            ],
            "Supplier Origin Postal": [
                "WC2N 5DU", "WC2N 5DU", "75001", "10115", "00100",
            ],
        })

    def test_commonname_chosen_over_supplier_common_name(self):
        df = self._make_df()
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "CommonName", (
            f"Expected 'CommonName' (clean), got {mapping['supplier_name']!r}. "
            "Supplier Common Name has numeric ERP prefixes so must score lower."
        )

    def test_address_detected_correctly(self):
        df = self._make_df()
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping.get("address") == "Address Line 1"

    def test_country_detected_correctly(self):
        df = self._make_df()
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping.get("country") == "Supplier Origin Country"

    def test_postal_detected_correctly(self):
        df = self._make_df()
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping.get("postal_code") == "Supplier Origin Postal"

    def test_city_detected_correctly(self):
        df = self._make_df()
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping.get("city") == "Supplier Origin City"

    def test_metadata_snapshot(self):
        """Metadata shape: mapped_supplier_name must be CommonName."""
        df = self._make_df()
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "CommonName"
        assert mapping.get("address") == "Address Line 1"
        assert mapping.get("country") == "Supplier Origin Country"


# ---------------------------------------------------------------------------
# 2.  Vendor Name chosen when cleanest
# ---------------------------------------------------------------------------

class TestVendorNameChosen:
    def test_vendor_name_chosen_when_clean(self):
        df = pl.DataFrame({
            "Vendor Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Vendor ID": ["V001", "V002", "V003"],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Vendor Name"

    def test_vendor_name_beats_numeric_supplier_common_name(self):
        df = pl.DataFrame({
            "Vendor Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Supplier Common Name": [
                "0010001001-Acme Corp",
                "0010001002-Global Trade Ltd",
                "0010001003-B Lab Co",
            ],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Vendor Name"

    def test_vendor_name_in_metadata(self):
        """When Vendor Name is selected, mapped_supplier_name reflects it."""
        df = pl.DataFrame({
            "Vendor Name": ["Acme Corp", "Global Trade Ltd"],
            "Vendor ID": ["V001", "V002"],
            "Country": ["US", "UK"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Vendor Name"


# ---------------------------------------------------------------------------
# 3.  Supplier Name chosen when cleanest
# ---------------------------------------------------------------------------

class TestSupplierNameChosen:
    def test_supplier_name_chosen_when_clean(self):
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Supplier ID": ["S001", "S002", "S003"],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name"

    def test_supplier_name_beats_mostly_blank(self):
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Company Name": [None, None, None],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name"


# ---------------------------------------------------------------------------
# 4.  Supplier Common Name wins when it IS clean and CommonName absent/worse
# ---------------------------------------------------------------------------

class TestSupplierCommonNameWinsWhenClean:
    def test_supplier_common_name_chosen_when_no_prefix(self):
        """When SCN values are clean (no numeric prefix) and CommonName absent,
        Supplier Common Name should be selected."""
        df = pl.DataFrame({
            "Supplier Common Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Common Name"

    def test_supplier_common_name_beats_blank_commonname(self):
        df = pl.DataFrame({
            "Supplier Common Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "CommonName": [None, None, None],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Common Name"


# ---------------------------------------------------------------------------
# 5.  Company Name chosen when cleanest
# ---------------------------------------------------------------------------

class TestCompanyNameChosen:
    def test_company_name_chosen_when_clean(self):
        df = pl.DataFrame({
            "Company Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Company ID": ["C001", "C002", "C003"],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Company Name"


# ---------------------------------------------------------------------------
# 6.  ID/code columns are never chosen
# ---------------------------------------------------------------------------

class TestIdColumnsNeverChosen:
    @pytest.mark.parametrize("col", [
        "Supplier ID", "Vendor ID", "Internal ID", "Entity ID",
        "legalEntityId", "clusterId", "Supplier Number",
        "Vendor Code", "Company Number",
    ])
    def test_id_column_is_rejected(self, col):
        assert _is_id_column(col.lower()), (
            f"_is_id_column should reject {col!r} but did not"
        )

    def test_supplier_id_not_chosen(self):
        df = pl.DataFrame({
            "Supplier ID": ["S001", "S002", "S003"],
            "Supplier Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name"

    def test_vendor_id_not_chosen(self):
        df = pl.DataFrame({
            "Vendor ID": ["V001", "V002"],
            "Vendor Name": ["Acme Corp", "Global Trade Ltd"],
            "Country": ["US", "UK"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Vendor Name"

    def test_legal_entity_id_camel_not_chosen(self):
        df = pl.DataFrame({
            "legalEntityId": ["LE001", "LE002"],
            "Supplier Name": ["Acme Corp", "Global Trade Ltd"],
            "Country": ["US", "UK"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name"

    def test_cluster_id_camel_not_chosen(self):
        df = pl.DataFrame({
            "clusterId": ["C001", "C002"],
            "Supplier Name": ["Acme Corp", "Global Trade Ltd"],
            "Country": ["US", "UK"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name"

    def test_id_columns_not_in_candidates(self):
        col_lower = {
            "supplier id": "Supplier ID",
            "vendor id": "Vendor ID",
            "supplier name": "Supplier Name",
            "country": "Country",
        }
        candidates = _find_supplier_name_candidates(col_lower)
        assert "Supplier ID" not in candidates
        assert "Vendor ID" not in candidates
        assert "Supplier Name" in candidates


# ---------------------------------------------------------------------------
# 7.  Mostly-numeric column not chosen
# ---------------------------------------------------------------------------

class TestMostlyNumericNotChosen:
    def test_mostly_numeric_column_not_chosen(self):
        """A column full of plain integers (stored IDs) must not win."""
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "Company Name": ["10001", "20002", "30003"],  # purely numeric
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name", (
            "Mostly-numeric 'Company Name' should not beat clean 'Supplier Name'"
        )

    def test_purely_numeric_column_scores_lower(self):
        """_score_name_column assigns a lower score to pure-number columns."""
        clean_score = _score_name_column("Supplier Name", ["Acme Corp", "B Lab Co", "Global"])
        numeric_score = _score_name_column("Company Name", ["10001", "20002", "30003"])
        assert clean_score > numeric_score, (
            f"Expected clean_score ({clean_score:.2f}) > numeric_score ({numeric_score:.2f})"
        )


# ---------------------------------------------------------------------------
# 8.  Mostly-blank column not chosen
# ---------------------------------------------------------------------------

class TestMostlyBlankNotChosen:
    def test_mostly_blank_column_not_chosen(self):
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Global Trade Ltd", "B Lab Co"],
            "CommonName": [None, None, None],
            "Country": ["US", "UK", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        assert mapping["supplier_name"] == "Supplier Name", (
            "All-null 'CommonName' should not beat clean 'Supplier Name'"
        )

    def test_blank_column_scores_lower(self):
        clean_score = _score_name_column("Supplier Name", ["Acme Corp", "B Lab Co"])
        blank_score = _score_name_column("CommonName", [None, None, None])
        assert clean_score > blank_score


# ---------------------------------------------------------------------------
# 9.  Numeric prefix stripping still works if polluted column is chosen
# ---------------------------------------------------------------------------

class TestNumericPrefixStrippingFallback:
    def test_prefix_stripped_in_name_norm_even_when_polluted_column_chosen(self):
        """Even if auto_detect picks the prefixed column (e.g. no cleaner alternative),
        preprocessing must strip the numeric prefix from name_norm."""
        df = pl.DataFrame({
            "Supplier Common Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
            ],
            "Country": ["US", "US"],
        })
        # Force the polluted column via explicit mapping
        mapping = {"supplier_name": "Supplier Common Name", "country": "Country"}
        df_proc = preprocess_dataframe(df, mapping)
        for row in df_proc.iter_rows(named=True):
            nn = row["name_norm"] or ""
            assert not nn.startswith("002"), (
                f"name_norm must not start with numeric prefix, got {nn!r}"
            )

    def test_original_column_unchanged_when_polluted_column_chosen(self):
        df = pl.DataFrame({
            "Supplier Common Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
            ],
            "Country": ["US", "US"],
        })
        mapping = {"supplier_name": "Supplier Common Name", "country": "Country"}
        df_proc = preprocess_dataframe(df, mapping)
        orig = df_proc["Supplier Common Name"].to_list()
        assert orig[0] == "0020276471-B LAB CO."
        assert orig[1] == "0020219751-B LAB COMPANY"


# ---------------------------------------------------------------------------
# 10.  Original output values unchanged through full pipeline
# ---------------------------------------------------------------------------

class TestOriginalColumnsPreservedInOutput:
    def test_output_preserves_original_column_values_commonname(self):
        df = pl.DataFrame({
            "CommonName": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
                "Acme Corp",
            ],
            "Supplier Common Name": [
                "0020276471-B LAB CO.",
                "0020219751-B LAB COMPANY",
                "0030001111-Acme Corp",
            ],
            "Address Line 1": ["15 WATERLOO AVE", "15 WATERLOO AVE", "1 MAIN ST"],
            "Supplier Origin Country": ["US", "US", "US"],
        })
        mapping = auto_detect_columns(df.columns, sample_df=df)
        config = ClusteringConfig()
        result = cluster_suppliers(df, mapping, config)
        out = result["main_df"]
        # All original column values must be intact
        assert set(out["CommonName"].to_list()) >= {"0020276471-B LAB CO.", "0020219751-B LAB COMPANY", "Acme Corp"}
        assert set(out["Supplier Common Name"].to_list()) >= {
            "0020276471-B LAB CO.", "0020219751-B LAB COMPANY", "0030001111-Acme Corp"
        }
        # Only Cluster Number and Match Percentage are added
        added_cols = set(out.columns) - set(df.columns)
        assert added_cols <= {"Cluster Number", "Match Percentage"}


# ---------------------------------------------------------------------------
# 11.  Column-name-only fallback (no sample_df) still uses priority list
# ---------------------------------------------------------------------------

class TestNoSampleFallbackPriorityList:
    def test_commonname_beats_supplier_common_name_no_data(self):
        """Without sample_df, priority list must still prefer CommonName."""
        columns = [
            "Supplier Common Name",
            "CommonName",
            "Address Line 1",
            "Supplier Origin Country",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "CommonName", (
            f"Without sample data, priority list must prefer CommonName; got {mapping['supplier_name']!r}"
        )

    def test_supplier_name_chosen_no_data(self):
        columns = ["Supplier Name", "Address", "Country"]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "Supplier Name"

    def test_vendor_name_chosen_no_data(self):
        columns = ["Vendor Name", "Street", "Country"]
        mapping = auto_detect_columns(columns)
        assert mapping["supplier_name"] == "Vendor Name"

    def test_no_sample_df_columns_only_still_detects_address(self):
        columns = ["Supplier Name", "Address Line 1", "City", "Country"]
        mapping = auto_detect_columns(columns)
        assert mapping.get("address") == "Address Line 1"

    def test_fallback_selects_first_candidate_when_nothing_matches_priority(self):
        """When no priority-list entry matches, first candidate is used."""
        columns = ["Legal Entity Name", "Address", "Country"]
        mapping = auto_detect_columns(columns)
        # "Legal Entity Name" contains "name" keyword and is a valid candidate
        assert mapping["supplier_name"] == "Legal Entity Name"

"""Pipeline-level regression test: same-name + same-address recovery with Danone column names.

Tests the full cluster_suppliers pipeline (blocking → evaluate_pair → merge → output)
using the exact column names found in Danone export files:
  CommonName, Address Line 1, Supplier Origin City,
  Supplier Origin Country, Supplier Origin Postal

Verifies:
  1. B LAB CO. / B LAB COMPANY at same address → same cluster (PASS 0B via LSUF_ block)
  2. ERM FRANCE / ERM FRANCE SAS at same address → same cluster
  3. GNT USA INC / GNT USA LLC at same address → same cluster
  4. IBM NEDERLAND / IBM NEDERLAND BV at same address → same cluster
  5. Unrelated names at same address → NOT in same cluster (precision guard)
  6. auto_detect_columns correctly maps 'Address Line 1' column
  7. LSUF_ block is generated for root_is_weak legal-variant rows
"""
import pytest
import polars as pl

from src.main import cluster_suppliers
from src.config import ClusteringConfig
from src.blocking import generate_candidate_pairs
from src.preprocessing import preprocess_dataframe
from scripts.run_cli import auto_detect_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DANONE_MAPPING = {
    "supplier_name": "CommonName",
    "address": "Address Line 1",
    "city": "Supplier Origin City",
    "country": "Supplier Origin Country",
    "postal_code": "Supplier Origin Postal",
}


def _make_danone_df(records: list[dict]) -> pl.DataFrame:
    """Build a minimal Danone-format DataFrame from a list of row dicts."""
    keys = ["CommonName", "Address Line 1", "Supplier Origin City",
            "Supplier Origin Country", "Supplier Origin Postal"]
    rows = {k: [] for k in keys}
    for rec in records:
        for k in keys:
            rows[k].append(rec.get(k, ""))
    return pl.DataFrame(rows)


def _cluster(records: list[dict], config=None) -> pl.DataFrame:
    """Run full pipeline and return the annotated main_df."""
    df = _make_danone_df(records)
    result = cluster_suppliers(df, DANONE_MAPPING, config)
    return result["main_df"]


def _same_cluster(df: pl.DataFrame, name_a: str, name_b: str) -> bool:
    """Return True iff name_a and name_b ended up in the same cluster."""
    row_a = df.filter(pl.col("CommonName") == name_a)
    row_b = df.filter(pl.col("CommonName") == name_b)
    if row_a.is_empty() or row_b.is_empty():
        return False
    cluster_a = row_a["Cluster Number"].to_list()[0]
    cluster_b = row_b["Cluster Number"].to_list()[0]
    # Both must have a cluster number and it must match
    return (
        cluster_a is not None
        and cluster_b is not None
        and cluster_a == cluster_b
    )


# ---------------------------------------------------------------------------
# Test 1: B LAB CO. / B LAB COMPANY at same address
# ---------------------------------------------------------------------------

class TestBLabPipelineCluster:
    def test_b_lab_co_company_same_address_clusters(self):
        """Full pipeline: B LAB CO. / B LAB COMPANY at 15 WATERLOO AVE must end up in same cluster."""
        df = _cluster([
            {"CommonName": "B LAB CO.",
             "Address Line 1": "15 WATERLOO AVE",
             "Supplier Origin City": "BERWYN",
             "Supplier Origin Country": "US",
             "Supplier Origin Postal": "19312"},
            {"CommonName": "B LAB COMPANY",
             "Address Line 1": "15 WATERLOO AVE",
             "Supplier Origin City": "BERWYN",
             "Supplier Origin Country": "US",
             "Supplier Origin Postal": "19312"},
            {"CommonName": "UNRELATED CORP INC",
             "Address Line 1": "999 OTHER ST",
             "Supplier Origin City": "CHICAGO",
             "Supplier Origin Country": "US",
             "Supplier Origin Postal": "60601"},
        ])
        assert _same_cluster(df, "B LAB CO.", "B LAB COMPANY"), (
            "B LAB CO. / B LAB COMPANY at same address must be in the same cluster in the "
            "full pipeline. Check that LSUF_ blocking key is generated and PASS 0B fires."
        )

    def test_b_lab_does_not_merge_with_unrelated(self):
        """B LAB CO. must NOT cluster with an unrelated supplier at a different address."""
        df = _cluster([
            {"CommonName": "B LAB CO.",
             "Address Line 1": "15 WATERLOO AVE",
             "Supplier Origin City": "BERWYN",
             "Supplier Origin Country": "US",
             "Supplier Origin Postal": "19312"},
            {"CommonName": "B LAB COMPANY",
             "Address Line 1": "15 WATERLOO AVE",
             "Supplier Origin City": "BERWYN",
             "Supplier Origin Country": "US",
             "Supplier Origin Postal": "19312"},
            {"CommonName": "UNRELATED CORP INC",
             "Address Line 1": "999 OTHER ST",
             "Supplier Origin City": "CHICAGO",
             "Supplier Origin Country": "US",
             "Supplier Origin Postal": "60601"},
        ])
        assert not _same_cluster(df, "B LAB CO.", "UNRELATED CORP INC"), (
            "B LAB CO. must not cluster with UNRELATED CORP INC at a different address."
        )


# ---------------------------------------------------------------------------
# Test 2: ERM FRANCE / ERM FRANCE SAS at same address
# ---------------------------------------------------------------------------

class TestErmFrancePipelineCluster:
    def test_erm_france_sas_same_address_clusters(self):
        """Full pipeline: ERM FRANCE / ERM FRANCE SAS at same address must cluster."""
        df = _cluster([
            {"CommonName": "ERM FRANCE",
             "Address Line 1": "13 RUE FAIDHERBE",
             "Supplier Origin City": "PARIS",
             "Supplier Origin Country": "FR"},
            {"CommonName": "ERM FRANCE SAS",
             "Address Line 1": "13 RUE FAIDHERBE",
             "Supplier Origin City": "PARIS",
             "Supplier Origin Country": "FR"},
        ])
        assert _same_cluster(df, "ERM FRANCE", "ERM FRANCE SAS"), (
            "ERM FRANCE / ERM FRANCE SAS at same address must cluster in the full pipeline."
        )


# ---------------------------------------------------------------------------
# Test 3: GNT USA INC / GNT USA LLC at same address
# ---------------------------------------------------------------------------

class TestGntUsaPipelineCluster:
    def test_gnt_usa_inc_llc_same_address_clusters(self):
        """Full pipeline: GNT USA INC / GNT USA LLC at same address must cluster."""
        df = _cluster([
            {"CommonName": "GNT USA INC",
             "Address Line 1": "ONE EXBERRY DRIVE",
             "Supplier Origin City": "MORRIS PLAINS",
             "Supplier Origin Country": "US"},
            {"CommonName": "GNT USA LLC",
             "Address Line 1": "ONE EXBERRY DRIVE",
             "Supplier Origin City": "MORRIS PLAINS",
             "Supplier Origin Country": "US"},
        ])
        assert _same_cluster(df, "GNT USA INC", "GNT USA LLC"), (
            "GNT USA INC / GNT USA LLC at same address must cluster in the full pipeline."
        )


# ---------------------------------------------------------------------------
# Test 4: IBM NEDERLAND / IBM NEDERLAND BV at same address
# ---------------------------------------------------------------------------

class TestIbmNederlandPipelineCluster:
    def test_ibm_nederland_bv_same_address_clusters(self):
        """Full pipeline: IBM NEDERLAND / IBM NEDERLAND BV at same address must cluster."""
        df = _cluster([
            {"CommonName": "IBM NEDERLAND",
             "Address Line 1": "JOHAN HUIZINGALAAN 765",
             "Supplier Origin City": "AMSTERDAM",
             "Supplier Origin Country": "NL"},
            {"CommonName": "IBM NEDERLAND BV",
             "Address Line 1": "JOHAN HUIZINGALAAN 765",
             "Supplier Origin City": "AMSTERDAM",
             "Supplier Origin Country": "NL"},
        ])
        assert _same_cluster(df, "IBM NEDERLAND", "IBM NEDERLAND BV"), (
            "IBM NEDERLAND / IBM NEDERLAND BV at same address must cluster in the full pipeline."
        )


# ---------------------------------------------------------------------------
# Test 5: Precision guard — unrelated names at same address do NOT cluster
# ---------------------------------------------------------------------------

class TestPrecisionGuardPipeline:
    def test_unrelated_names_same_address_not_clustered(self):
        """Full pipeline: ACME CONSULTING INC / GLOBAL TRADING LLC at same address must NOT cluster."""
        df = _cluster([
            {"CommonName": "ACME CONSULTING INC",
             "Address Line 1": "100 MAIN STREET",
             "Supplier Origin City": "NEW YORK",
             "Supplier Origin Country": "US"},
            {"CommonName": "GLOBAL TRADING LLC",
             "Address Line 1": "100 MAIN STREET",
             "Supplier Origin City": "NEW YORK",
             "Supplier Origin Country": "US"},
        ])
        assert not _same_cluster(df, "ACME CONSULTING INC", "GLOBAL TRADING LLC"), (
            "Unrelated names at same address must NOT cluster in the full pipeline."
        )


# ---------------------------------------------------------------------------
# Test 6: auto_detect_columns maps 'Address Line 1' correctly
# ---------------------------------------------------------------------------

class TestAutoDetectDanoneColumns:
    def test_address_line_1_detected(self):
        """auto_detect_columns must map 'Address Line 1' to the address field."""
        columns = [
            "CommonName",
            "Address Line 1",
            "Supplier Origin City",
            "Supplier Origin Country",
            "Supplier Origin Postal",
        ]
        mapping = auto_detect_columns(columns)
        assert mapping.get("address") == "Address Line 1", (
            f"Expected address='Address Line 1', got {mapping.get('address')!r}. "
            "Check auto_detect_columns logic for 'address' in column name."
        )

    def test_supplier_name_detected(self):
        """auto_detect_columns must map some column to supplier_name."""
        columns = [
            "CommonName",
            "Address Line 1",
            "Supplier Origin City",
            "Supplier Origin Country",
            "Supplier Origin Postal",
        ]
        mapping = auto_detect_columns(columns)
        assert "supplier_name" in mapping, (
            "auto_detect_columns must detect a supplier_name column from Danone headers."
        )


# ---------------------------------------------------------------------------
# Test 7: LSUF_ block generated for root_is_weak legal-variant rows
# ---------------------------------------------------------------------------

class TestLsufBlockGeneration:
    def test_lsuf_block_generated_for_b_lab(self):
        """Blocking must generate a LSUF_ key for 'B LAB CO.' so it can pair with 'B LAB COMPANY'."""
        df_raw = pl.DataFrame({
            "Supplier Name": ["B LAB CO.", "B LAB COMPANY"],
            "Address": ["15 WATERLOO AVE", "15 WATERLOO AVE"],
            "City": ["BERWYN", "BERWYN"],
            "Country": ["US", "US"],
        })
        df_proc = preprocess_dataframe(
            df_raw,
            {"supplier_name": "Supplier Name", "address": "Address",
             "city": "City", "country": "Country"},
        )
        candidates = generate_candidate_pairs(df_proc)
        # There must be at least one pair
        assert len(candidates) > 0, "No candidate pairs generated for B LAB CO. / B LAB COMPANY"
        # At least one pair should come from a LSUF_ or ADR_ block
        block_types = candidates["block_type"].to_list()
        assert any(bt in ("legal_suffix_variant", "address") for bt in block_types), (
            f"Expected legal_suffix_variant or address block, got block_types={block_types}"
        )

    def test_lsuf_block_generated_without_address(self):
        """LSUF_ block must fire even when addr_norm is empty (no address column mapped)."""
        df_raw = pl.DataFrame({
            "Supplier Name": ["B LAB CO.", "B LAB COMPANY"],
            "Country": ["US", "US"],
        })
        df_proc = preprocess_dataframe(
            df_raw,
            {"supplier_name": "Supplier Name", "country": "Country"},
        )
        candidates = generate_candidate_pairs(df_proc)
        assert len(candidates) > 0, (
            "No candidate pairs for B LAB CO. / B LAB COMPANY even without address — "
            "LSUF_ block must generate pairs independent of address mapping."
        )
        block_types = candidates["block_type"].to_list()
        assert "legal_suffix_variant" in block_types, (
            f"Expected legal_suffix_variant block when no address, got {block_types}"
        )

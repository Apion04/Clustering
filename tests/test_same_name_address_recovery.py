"""Tests for same-name + same-address recovery (PASS 0B).

Verifies that:
  1. B LAB CO. / B LAB COMPANY with same address clusters (legal suffix variant)
  2. ERM FRANCE / ERM FRANCE SAS with same address clusters (already 98 via PASS 0)
  3. IBM NEDERLAND / IBM NEDERLAND BV with same address clusters
  4. GNT USA INC / GNT USA LLC with same address clusters
  5. QED ADVANCED SYSTEMS LTD repeated at same address clusters
  6. Unrelated names at same address do NOT cluster at 85/98
  7. Single generic word + same address does NOT cluster at 85/98
  8. Person-name / company-name at same address is NOT forced to 98/85 without evidence
  9. Exact same name + same address remains 98 (PASS 0 unaffected)
 10. All existing tests pass (implicit via full suite)
"""
import pytest
import polars as pl

from src.config import ClusteringConfig
from src.preprocessing import preprocess_dataframe
from src.matching import evaluate_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval(
    name_a: str, name_b: str,
    addr_a: str = "", addr_b: str = "",
    city_a: str = "", city_b: str = "",
    country_a: str = "US", country_b: str = "US",
) -> "MatchResult":
    df = pl.DataFrame({
        "Supplier Name": [name_a, name_b],
        "Address": [addr_a, addr_b],
        "City": [city_a, city_b],
        "Country": [country_a, country_b],
    })
    rows = list(preprocess_dataframe(
        df,
        {"supplier_name": "Supplier Name", "address": "Address",
         "city": "City", "country": "Country"},
    ).iter_rows(named=True))
    return evaluate_pair(rows[0], rows[1], {}, ClusteringConfig())


# ---------------------------------------------------------------------------
# Tests 1-5: same-name/address cases that MUST cluster
# ---------------------------------------------------------------------------

class TestSameAddressLegalVariantClusters:
    def test_b_lab_co_vs_b_lab_company_same_address(self):
        """B LAB CO. / B LAB COMPANY at same address must cluster (legal suffix variant)."""
        result = _eval(
            "B LAB CO.", "B LAB COMPANY",
            addr_a="15 WATERLOO AVE", addr_b="15 WATERLOO AVE",
        )
        assert result.is_match, (
            f"B LAB CO. / B LAB COMPANY at same address must cluster, "
            f"got is_match={result.is_match} {result.match_pct}% [{result.pass_type}]"
        )
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_erm_france_vs_erm_france_sas_same_address(self):
        """ERM FRANCE / ERM FRANCE SAS at same address must cluster."""
        result = _eval(
            "ERM FRANCE", "ERM FRANCE SAS",
            addr_a="13 RUE FAIDHERBE", addr_b="13 RUE FAIDHERBE",
            country_a="FR", country_b="FR",
        )
        assert result.is_match
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_ibm_nederland_vs_ibm_nederland_bv_same_address(self):
        """IBM NEDERLAND / IBM NEDERLAND BV at same address must cluster."""
        result = _eval(
            "IBM NEDERLAND", "IBM NEDERLAND BV",
            addr_a="JOHAN HUIZINGALAAN 765", addr_b="JOHAN HUIZINGALAAN 765",
            country_a="NL", country_b="NL",
        )
        assert result.is_match
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_gnt_usa_inc_vs_gnt_usa_llc_same_address(self):
        """GNT USA INC / GNT USA LLC at same address must cluster."""
        result = _eval(
            "GNT USA INC", "GNT USA LLC",
            addr_a="ONE EXBERRY DRIVE", addr_b="ONE EXBERRY DRIVE",
        )
        assert result.is_match
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_qed_advanced_systems_repeated_same_address(self):
        """QED ADVANCED SYSTEMS LTD repeated at same address must cluster."""
        result = _eval(
            "QED ADVANCED SYSTEMS LTD", "QED ADVANCED SYSTEMS LTD",
            addr_a="100 TECH PARK DR", addr_b="100 TECH PARK DR",
        )
        assert result.is_match
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Tests 6-8: cases that must NOT cluster at 85/98 (precision guards)
# ---------------------------------------------------------------------------

class TestSameAddressPrecisionGuards:
    def test_unrelated_names_same_address_not_85(self):
        """Unrelated names at same address must NOT cluster at 85/98."""
        result = _eval(
            "ACME CONSULTING INC", "GLOBAL TRADING LLC",
            addr_a="100 MAIN STREET", addr_b="100 MAIN STREET",
        )
        assert result.match_pct < 85.0 or not result.is_match, (
            f"Unrelated names at same address must not be 85%+, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )

    def test_single_generic_word_same_address_not_85(self):
        """'Services Inc' vs 'Services LLC' at same address (single generic core) must not be 85/98."""
        result = _eval(
            "SERVICES INC", "SERVICES LLC",
            addr_a="100 MAIN STREET", addr_b="100 MAIN STREET",
        )
        assert result.match_pct < 85.0 or not result.is_match, (
            f"Single generic word + same address must not cluster at 85%+, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )

    def test_person_company_same_address_not_forced_98(self):
        """A likely person name and a company name at same address must not be forced to 98%."""
        result = _eval(
            "JOHN SMITH", "GLOBAL TECH CORP",
            addr_a="100 MAIN STREET", addr_b="100 MAIN STREET",
        )
        # May be review candidate at 70-80%, but must NOT be 98% forced cluster
        assert result.match_pct < 98.0, (
            f"Person vs company at same address must not be 98%, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Test 9: exact duplicate (PASS 0) still 98%
# ---------------------------------------------------------------------------

class TestExactDuplicateUnaffected:
    def test_exact_name_address_still_98(self):
        """Exact same name + same address must still be 98% (PASS 0 unaffected)."""
        result = _eval(
            "BLACK BOX ENGINEERING LLC", "BLACK BOX ENGINEERING LLC",
            addr_a="100 ENTERPRISE DRIVE", addr_b="100 ENTERPRISE DRIVE",
        )
        assert result.match_pct >= 98.0, (
            f"Exact duplicate must still be 98%, got {result.match_pct}% [{result.pass_type}]"
        )
        assert result.pass_type == "exact_duplicate"

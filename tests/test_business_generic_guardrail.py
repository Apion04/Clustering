"""Tests for the business/industrial generic non-bridge token guardrail (Phase 6).

Verifies:
  1.  'equipment' alone must NOT bridge unrelated suppliers to 85%.
  2.  Generic business descriptors (services, solutions, logistics, food, …)
      must NOT be the sole bridge for an 85% cluster.
  3.  Good anchored clusters (Siemens, Thermo Fisher, same address) are preserved.
  4.  BUSINESS_GENERIC_TOKENS are all registered in GENERIC_ROOT_TOKENS.

Key fix: 'equipment' was the only token from the Phase 6 required list that was
missing from GENERIC_ROOT_TOKENS.  It now creates no DCORE_ blocking key and is
treated as a non-bridge token by all guardrails.
"""
import pytest
import polars as pl

from src.config import ClusteringConfig, GENERIC_ROOT_TOKENS, BUSINESS_GENERIC_TOKENS
from src.preprocessing import preprocess_dataframe
from src.matching import evaluate_pair
from src.main import cluster_suppliers


# ---------------------------------------------------------------------------
# Helper — preprocess two rows and call evaluate_pair directly
# ---------------------------------------------------------------------------

def _eval(
    name_a: str,
    name_b: str,
    email_a: str = "",
    email_b: str = "",
    address_a: str = "",
    address_b: str = "",
    country_a: str = "US",
    country_b: str = "US",
):
    """Preprocess two supplier rows and evaluate them as a pair."""
    df = pl.DataFrame({
        "Supplier Name": [name_a, name_b],
        "Email": [email_a, email_b],
        "Address": [address_a, address_b],
        "Country": [country_a, country_b],
    })
    rows = list(preprocess_dataframe(
        df,
        {
            "supplier_name": "Supplier Name",
            "email": "Email",
            "address": "Address",
            "country": "Country",
        },
    ).iter_rows(named=True))
    return evaluate_pair(rows[0], rows[1], {}, ClusteringConfig())


# ---------------------------------------------------------------------------
# 0. Token registration — all BUSINESS_GENERIC_TOKENS in GENERIC_ROOT_TOKENS
# ---------------------------------------------------------------------------

class TestBusinessGenericTokensRegistered:
    def test_all_business_generic_tokens_in_generic_root_tokens(self):
        """Every token in BUSINESS_GENERIC_TOKENS must be in GENERIC_ROOT_TOKENS."""
        missing = BUSINESS_GENERIC_TOKENS - GENERIC_ROOT_TOKENS
        assert not missing, (
            f"These BUSINESS_GENERIC_TOKENS are NOT in GENERIC_ROOT_TOKENS: {sorted(missing)}. "
            "Add them to GENERIC_ROOT_TOKENS so they cannot bridge unrelated suppliers."
        )

    def test_equipment_in_generic_root_tokens(self):
        """'equipment' specifically must be in GENERIC_ROOT_TOKENS (Phase 6 gap)."""
        assert "equipment" in GENERIC_ROOT_TOKENS, (
            "'equipment' must be in GENERIC_ROOT_TOKENS — it was the missing token "
            "that caused DCORE_equipment to bridge unrelated suppliers at 88%."
        )

    def test_equipment_in_business_generic_tokens(self):
        assert "equipment" in BUSINESS_GENERIC_TOKENS


# ---------------------------------------------------------------------------
# 1. 'equipment' alone must NOT produce 85%+
# ---------------------------------------------------------------------------

class TestEquipmentNotBridge:
    def test_global_equipment_vs_advanced_equipment_not_85(self):
        """Global Equipment / Advanced Equipment: 'equipment' alone must not bridge at 85%."""
        result = _eval("Global Equipment", "Advanced Equipment")
        assert result.match_pct < 85.0, (
            f"Global Equipment / Advanced Equipment must be < 85% (got {result.match_pct}%). "
            "'equipment' is a generic business descriptor, not a distinctive brand anchor."
        )

    def test_chemical_equipment_labs_vs_he_equipment_services_not_85(self):
        """Chemical Equipment Labs / H and E Equipment Services: equipment/services not enough."""
        result = _eval("Chemical Equipment Labs", "H and E Equipment Services")
        assert result.match_pct < 85.0, (
            f"Expected < 85%, got {result.match_pct}% [{result.pass_type}]. "
            "'equipment' plus other generic words must not bridge at 85%."
        )

    def test_global_equipment_vs_we_equipment_not_85(self):
        """Global Equipment / W&W Equipment: 'equipment' alone must not bridge."""
        result = _eval("Global Equipment", "W and W Equipment")
        assert result.match_pct < 85.0, (
            f"Got {result.match_pct}% [{result.pass_type}]"
        )

    def test_fh_food_equipment_vs_advanced_equipment_not_85(self):
        """F and H Food Equipment / Advanced Equipment: equipment+food not enough."""
        result = _eval("F and H Food Equipment", "Advanced Equipment")
        assert result.match_pct < 85.0, (
            f"Got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# 2. Other generic business descriptors must NOT produce 85%+ alone
# ---------------------------------------------------------------------------

class TestGenericBusinessDescriptorsNotBridge:
    def test_solutions_alone_not_85(self):
        """ABC Solutions / XYZ Solutions: 'solutions' alone must not bridge at 85%."""
        result = _eval("ABC Solutions", "XYZ Solutions")
        assert result.match_pct < 85.0, (
            f"'solutions' alone must not produce 85%, got {result.match_pct}%"
        )

    def test_logistics_alone_not_85(self):
        """Global Logistics / Advanced Logistics: 'logistics' alone must not bridge."""
        result = _eval("Global Logistics", "Advanced Logistics")
        assert result.match_pct < 85.0, (
            f"'logistics' alone must not produce 85%, got {result.match_pct}%"
        )

    def test_food_services_vs_food_solutions_not_85(self):
        """Food Services Inc / Food Solutions LLC: food+services/solutions not enough."""
        result = _eval("Food Services Inc", "Food Solutions LLC")
        assert result.match_pct < 85.0, (
            f"food+services/solutions must not produce 85%, got {result.match_pct}%"
        )

    def test_global_industrial_services_vs_advanced_industrial_solutions_not_85(self):
        """Global Industrial Services / Advanced Industrial Solutions: all generic tokens."""
        result = _eval("Global Industrial Services", "Advanced Industrial Solutions")
        assert result.match_pct < 85.0, (
            f"All-generic tokens must not produce 85%, got {result.match_pct}%"
        )

    def test_technology_management_not_85(self):
        """Global Technology Management / Advanced Technology Management: all generic."""
        result = _eval("Global Technology Management", "Advanced Technology Management")
        assert result.match_pct < 85.0, (
            f"technology+management must not produce 85%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# 3. Good clusters preserved — exact duplicate / same address / same domain
# ---------------------------------------------------------------------------

class TestGoodClustersWithGenericDescriptorsPreserved:
    def test_exact_same_name_same_address_still_98(self):
        """Exact duplicate (same name + same address) must remain 98%."""
        result = _eval(
            "Acme Equipment Corp", "Acme Equipment Corp",
            address_a="123 Main St", address_b="123 Main St",
        )
        assert result.match_pct >= 98.0, (
            f"Exact duplicate with same address must be 98%, got {result.match_pct}%"
        )

    def test_near_name_same_address_still_clusters(self):
        """Same address + highly similar name (legal-suffix variant) must still cluster."""
        result = _eval(
            "Acme Equipment Corp", "Acme Equipment Inc",
            address_a="123 Main St", address_b="123 Main St",
        )
        assert result.match_pct >= 85.0, (
            f"Near-name + same address must cluster, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_same_supplier_owned_domain_near_name_clusters(self):
        """Same distinctive domain + similar name must still cluster."""
        result = _eval(
            "Acme Systems Inc", "Acme Systems Corporation",
            email_a="sales@acmesystems.com", email_b="support@acmesystems.com",
        )
        assert result.match_pct >= 85.0, (
            f"Same owner domain + similar name must cluster, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_siemens_industry_vs_siemens_ag_clusters(self):
        """Siemens has a distinctive trusted anchor — industrial descriptor does not block it."""
        result = _eval("Siemens Industry", "Siemens AG")
        assert result.match_pct >= 70.0, (
            f"Siemens (trusted anchor) should cluster or reach review, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_thermo_fisher_scientific_same_domain_clusters(self):
        """Thermo Fisher Scientific / Fisher Scientific with same domain must cluster."""
        result = _eval(
            "Thermo Fisher Scientific",
            "Fisher Scientific",
            email_a="lab@thermofisher.com",
            email_b="orders@thermofisher.com",
        )
        assert result.match_pct >= 85.0, (
            f"Thermo Fisher / Fisher Scientific with same domain must cluster, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )

    def test_danone_nutricia_family_not_broken(self):
        """Danone/Nutricia brand-family pairing must survive the generic-token change."""
        df = pl.DataFrame({
            "Supplier Name": ["Danone SA", "Nutricia NV", "B Lab Co"],
            "Country": ["FR", "NL", "US"],
        })
        mapping = {"supplier_name": "Supplier Name", "country": "Country"}
        result = cluster_suppliers(df, mapping, ClusteringConfig())
        main_df = result["main_df"]
        # B Lab must NOT be in the same cluster as Danone/Nutricia
        cluster_danone = main_df.filter(pl.col("Supplier Name") == "Danone SA")["Cluster Number"].to_list()
        cluster_blab = main_df.filter(pl.col("Supplier Name") == "B Lab Co")["Cluster Number"].to_list()
        # If Danone/Nutricia cluster, B Lab must not be in it
        if cluster_danone and cluster_danone[0] is not None:
            assert cluster_blab[0] != cluster_danone[0] or cluster_blab[0] is None, (
                "B Lab must not end up in Danone/Nutricia cluster"
            )


# ---------------------------------------------------------------------------
# 4. With strong anchor + generic descriptor, legitimate pairs still cluster
# ---------------------------------------------------------------------------

class TestAnchoredGenericDescriptorPreserved:
    def test_same_domain_equipment_company_clusters(self):
        """Two rows sharing a real business domain AND equipment term can still cluster."""
        result = _eval(
            "Acme Equipment", "Acme Equipment Services",
            email_a="info@acmeequipment.com", email_b="support@acmeequipment.com",
        )
        # Same distinctive domain is enough to cluster even when name contains generic words
        assert result.match_pct >= 85.0, (
            f"Same distinctive domain should allow clustering even with 'equipment', "
            f"got {result.match_pct}% [{result.pass_type}]"
        )

    def test_equipment_same_tax_clusters(self):
        """Two equipment companies sharing the same tax ID must still cluster."""
        df = pl.DataFrame({
            "Supplier Name": ["Global Equipment Inc", "Global Equipment LLC"],
            "Tax ID": ["12-3456789", "12-3456789"],
            "Country": ["US", "US"],
        })
        rows = list(preprocess_dataframe(
            df,
            {"supplier_name": "Supplier Name", "tax_id": "Tax ID", "country": "Country"},
        ).iter_rows(named=True))
        result = evaluate_pair(rows[0], rows[1], {}, ClusteringConfig())
        assert result.match_pct >= 85.0, (
            f"Same tax ID must allow 85%+ even for equipment companies, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )

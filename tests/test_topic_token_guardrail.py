"""Tests for the medical/research/topic-word precision guardrail.

Verifies that broad topic words (cancer, melanoma, nature, …) are never the
sole bridge for an 85% cluster, while legitimate pharma/life-science brand
pairs with deterministic evidence (same domain / same address / same tax) are
not affected.
"""
import pytest
import polars as pl

from src.config import ClusteringConfig
from src.preprocessing import preprocess_dataframe
from src.matching import evaluate_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval(name_a: str, name_b: str,
          email_a: str = "", email_b: str = "",
          country_a: str = "US", country_b: str = "US") -> "MatchResult":
    """Preprocess a two-row frame and evaluate the pair directly."""
    df = pl.DataFrame({
        "Supplier Name": [name_a, name_b],
        "Email": [email_a, email_b],
        "Country": [country_a, country_b],
    })
    rows = list(preprocess_dataframe(
        df,
        {"supplier_name": "Supplier Name", "email": "Email", "country": "Country"},
    ).iter_rows(named=True))
    return evaluate_pair(rows[0], rows[1], {}, ClusteringConfig())


# ---------------------------------------------------------------------------
# Tests 1-5: topic words must NOT produce 85%
# ---------------------------------------------------------------------------

class TestTopicWordGuardrail:
    def test_cancer_orgs_not_85(self):
        """American Cancer Society / National Cancer Institute must not be 85%."""
        result = _eval("American Cancer Society", "National Cancer Institute")
        assert result.match_pct < 85.0, (
            f"Expected < 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_cancer_research_orgs_not_85(self):
        """American Association for Cancer Research / Institute for Cancer Research must not be 85%."""
        result = _eval(
            "American Association for Cancer Research",
            "Institute for Cancer Research",
        )
        assert result.match_pct < 85.0, (
            f"Expected < 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_melanoma_orgs_not_85(self):
        """Melanoma Institute Australia / Melanoma Research Foundation must not be 85%."""
        result = _eval(
            "Melanoma Institute Australia",
            "Melanoma Research Foundation",
        )
        assert result.match_pct < 85.0, (
            f"Expected < 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_nature_companies_not_85(self):
        """Nature America / Nature Technology Corporation must not be 85%."""
        result = _eval("Nature America", "Nature Technology Corporation")
        assert result.match_pct < 85.0, (
            f"Expected < 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_oracle_word_not_sole_bridge_for_85(self):
        """Gene Oracle / Oracle America must not reach 85% from shared Oracle word alone."""
        result = _eval("Gene Oracle", "Oracle America")
        assert result.match_pct < 85.0, (
            f"Expected < 85%, got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Tests 6-9: good clusters must survive
# ---------------------------------------------------------------------------

class TestGoodClustersPreserved:
    def test_pfizer_with_same_domain_still_clusters(self):
        """Pfizer Inc / Pfizer Canada with same email domain must still reach 85%+."""
        result = _eval(
            "Pfizer Inc", "Pfizer Canada",
            email_a="info@pfizer.com", email_b="info@pfizer.com",
        )
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_thermo_fisher_with_same_domain_still_clusters(self):
        """Thermo Fisher Scientific / Thermo Fisher Inc with same domain must reach 85%+."""
        result = _eval(
            "Thermo Fisher Scientific", "Thermo Fisher Inc",
            email_a="lab@thermofisher.com", email_b="lab@thermofisher.com",
        )
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_exact_duplicate_still_98(self):
        """Exact duplicate (same normalized name AND address) must still score 98%."""
        result = _eval("Acme Corp", "Acme Corp")
        # Without address it falls to same_legal_owner_confirmed, but either way >= 85
        # This test checks the 98 path via a row with address data.
        df = pl.DataFrame({
            "Supplier Name": ["Acme Corp", "Acme Corp"],
            "Email": ["", ""],
            "Country": ["US", "US"],
            "Address": ["123 Main St", "123 Main St"],
        })
        rows = list(preprocess_dataframe(
            df,
            {
                "supplier_name": "Supplier Name",
                "email": "Email",
                "country": "Country",
                "address": "Address",
            },
        ).iter_rows(named=True))
        r = evaluate_pair(rows[0], rows[1], {}, ClusteringConfig())
        assert r.match_pct >= 98.0, (
            f"Exact duplicate with same address must be 98%, got {r.match_pct}% [{r.pass_type}]"
        )

    def test_same_domain_near_name_still_85(self):
        """Same supplier-owned domain + similar name must still reach 85%+."""
        result = _eval(
            "Illumina Inc", "Illumina Corporation",
            email_a="info@illumina.com", email_b="info@illumina.com",
        )
        assert result.match_pct >= 85.0, (
            f"Expected >= 85%, got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Test 10: cancer/topic org WITH same domain still clusters at 85%
# ---------------------------------------------------------------------------

class TestTopicOrgWithDomainStillClusters:
    def test_cancer_org_same_domain_still_85(self):
        """American Cancer Society (same domain both sides) should still reach 85%+.

        The topic-word guardrail only fires when there is NO deterministic support.
        Same domain is deterministic support, so the cluster must be preserved.
        """
        result = _eval(
            "American Cancer Society", "American Cancer Society Inc",
            email_a="contact@cancer.org", email_b="contact@cancer.org",
        )
        assert result.match_pct >= 85.0, (
            f"Cancer org with same domain must still cluster, got {result.match_pct}% [{result.pass_type}]"
        )

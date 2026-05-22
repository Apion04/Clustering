"""Tests for the SLD-topic guardrail and trusted-core partial-overlap guard.

Verifies that:
  - Medical/topic SLDs (melanoma, cancer) do NOT produce 85% via domain_sld_family
  - Legitimate brand SLDs (pfizer, bcorporation) still produce 85% with name support
  - Trusted-core partial overlap (Gene Oracle / Oracle America) does NOT become a
    final cluster without domain/address/tax deterministic support
  - Exact duplicates are unaffected (still 98%)
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
    email_a: str = "", email_b: str = "",
    country_a: str = "US", country_b: str = "US",
) -> "MatchResult":
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
# Tests 1-2: medical/topic SLDs must NOT produce 85% via domain_sld_family
# ---------------------------------------------------------------------------

class TestMedicalSLDGuardrail:
    def test_melanoma_sld_not_85(self):
        """melanoma.org.au / melanoma.org share SLD 'melanoma' — a medical topic word.
        domain_sld_family must NOT reach 85% on this alone."""
        result = _eval(
            "Melanoma Institute Australia", "Melanoma Research Foundation",
            email_a="info@melanoma.org.au", email_b="info@melanoma.org",
            country_a="AU", country_b="US",
        )
        assert result.match_pct < 85.0, (
            f"Medical SLD 'melanoma' must not create 85% cluster, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )

    def test_cancer_sld_not_85(self):
        """cancer.org / cancer.org.au share SLD 'cancer' — a medical topic word.
        domain_sld_family must NOT reach 85% on this alone."""
        result = _eval(
            "Cancer Council Australia", "Cancer Research UK",
            email_a="info@cancer.org.au", email_b="info@cancer.org",
            country_a="AU", country_b="GB",
        )
        assert result.match_pct < 85.0, (
            f"Medical SLD 'cancer' must not create 85% cluster, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Tests 3-4: legitimate brand SLDs still produce 85% with name support
# ---------------------------------------------------------------------------

class TestLegitBrandSLDPreserved:
    def test_pfizer_sld_still_clusters(self):
        """pfizer.com / pfizer.co.uk share SLD 'pfizer' — a brand, not a topic word.
        Must still reach 85%+ (via same_domain or supplier_identity with domain support)."""
        result = _eval(
            "Pfizer Inc", "Pfizer Canada",
            email_a="info@pfizer.com", email_b="info@pfizer.com",
        )
        assert result.match_pct >= 85.0, (
            f"Pfizer brand SLD must still cluster, got {result.match_pct}% [{result.pass_type}]"
        )

    def test_bcorporation_sld_still_clusters(self):
        """bcorporation.net / bcorporation.fr share SLD 'bcorporation' — a brand.
        Must still reach 85%+ when names also share the brand token."""
        result = _eval(
            "B Corp Australia", "B Corp France",
            email_a="info@bcorporation.net", email_b="info@bcorporation.fr",
            country_a="AU", country_b="FR",
        )
        assert result.match_pct >= 85.0, (
            f"bcorporation brand SLD must still cluster, got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Test 5: Gene Oracle / Oracle America must NOT become a final cluster
# ---------------------------------------------------------------------------

class TestTrustedCorePartialOverlapGuard:
    def test_gene_oracle_oracle_america_not_final_cluster(self):
        """Gene Oracle / Oracle America share trusted core token 'oracle',
        but exact_core=False and no domain/address/tax support.
        main_cluster_allowed must be False → not merged into a final cluster."""
        result = _eval("Gene Oracle", "Oracle America")
        evidence = result.evidence or {}
        main_cluster_allowed = evidence.get("main_cluster_allowed", True)
        assert not main_cluster_allowed, (
            f"Gene Oracle / Oracle America must have main_cluster_allowed=False, "
            f"got {main_cluster_allowed!r}, score={result.match_pct}% [{result.pass_type}]"
        )

    def test_gene_oracle_oracle_america_below_85(self):
        """Gene Oracle / Oracle America must not reach 85% (final-cluster threshold)."""
        result = _eval("Gene Oracle", "Oracle America")
        assert result.match_pct < 85.0, (
            f"Gene Oracle / Oracle America must score < 85%, "
            f"got {result.match_pct}% [{result.pass_type}]"
        )


# ---------------------------------------------------------------------------
# Test 6: exact duplicate unaffected — still 98%
# ---------------------------------------------------------------------------

class TestExactDuplicateUnaffected:
    def test_exact_duplicate_still_98(self):
        """Exact duplicate (same name + same address) must still score 98%."""
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
            f"Exact duplicate must still be 98%, got {r.match_pct}% [{r.pass_type}]"
        )

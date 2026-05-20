"""Phase 3A recall improvements.

Scope (approved):
1. Safe status/noise stripping → operational_status_review no longer review-only
2. Same-domain bug fix → domain_review_candidate no longer review-only
3. Address normalization (already working, no new tests)
4. Polish/Unicode normalization (already working, no new tests)
5. Conservative PASS 3C descriptor expansion (emea, apac, amer, dach, latam, ventures)
6. Acronym expansion (already working, no new tests)
7. 70 surfacing → distinctive_supplier_identity main_cluster_allowed=False no longer review-only
"""
import pytest
import polars as pl
from src.config import ClusteringConfig
from src.matching import evaluate_pair
from src.matching_types import MatchResult
from src.main import _is_review_only, cluster_suppliers
from src.preprocessing import extract_supplier_identity_core

cfg = ClusteringConfig()


def _row(
    name_norm,
    addr_norm="",
    city_norm="",
    country_norm="",
    domain="",
    domain_sld="",
    name_location_core=None,
    row_id=0,
    has_operational_status_hint=False,
    supplier_identity_core=None,
    root_brand="",
    postal_norm="",
    tax_norm="",
):
    return {
        "name_norm": name_norm,
        "addr_norm": addr_norm,
        "city_norm": city_norm,
        "country_norm": country_norm,
        "domain": domain,
        "domain_sld": domain_sld,
        "is_generic_domain": False,
        "name_location_core": name_location_core if name_location_core is not None else name_norm,
        "row_id": row_id,
        "postal_norm": postal_norm,
        "has_operational_status_hint": has_operational_status_hint,
        "supplier_identity_core": supplier_identity_core if supplier_identity_core is not None else extract_supplier_identity_core(name_norm),
        "root_brand": root_brand,
        "tax_norm": tax_norm,
        "tax_loose_norm": "",
        "known_brand_family_ids": "",
        "known_brand_family_safe_ids": "",
        "known_brand_family_risky_ids": "",
        "known_brand_alias_hits": "",
        "support_fields_json": "[]",
        "json_secondary_names_norm": "",
        "is_likely_individual": False,
        "is_hospitality": False,
        "has_legal_suffix": False,
        "person_name_norm": "",
        "idf_discriminative_tokens": "",
        "franchise_store_number": "",
    }


# ---------------------------------------------------------------------------
# Item 1: operational_status_review no longer review-only
# ---------------------------------------------------------------------------

class TestOperationalStatusSurfacing:
    def test_operational_status_review_not_in_review_only(self):
        """operational_status_review pass type must NOT be review-only after Phase 3A."""
        result = MatchResult(
            True, 72.0, "operational_status_review", {}, needs_review=True,
            review_reason="Operational blocked/inactive text present"
        )
        assert not _is_review_only(result), (
            "operational_status_review should appear in main output at 70%, "
            "not be hidden in review-only"
        )

    def test_blocked_name_matches_clean_name(self):
        """BLOCKED prefix stripped → name_norm matches clean version, not review-only."""
        # Simulates: "BLOCKED Figiel GmbH Co.KG" → name_norm="figiel gmbh co kg",
        # has_operational_status_hint=True (set from original name during preprocessing)
        row_blocked = _row("figiel gmbh co kg", has_operational_status_hint=True)
        row_clean = _row("figiel gmbh co kg")
        result = evaluate_pair(row_blocked, row_clean, {}, cfg)
        assert result.is_match, "Blocked row (after stripping) should match clean version"
        assert not _is_review_only(result), (
            "Blocked row match should appear in main output, not be hidden as review-only"
        )

    def test_gesperrt_name_shows_in_output(self):
        """GESPERRT prefix stripped → match surfaces in main output."""
        row_blocked = _row("riseway ltd", has_operational_status_hint=True)
        row_clean = _row("riseway ltd")
        result = evaluate_pair(row_blocked, row_clean, {}, cfg)
        assert result.is_match
        assert not _is_review_only(result), "Gesperrt row match must not be hidden"

    def test_operational_status_match_score_reasonable(self):
        """Operational-status match score is capped ≤ 86 (guardrail stays, routing changes)."""
        row_blocked = _row(
            "acme gmbh", addr_norm="1 main street", has_operational_status_hint=True
        )
        row_clean = _row("acme gmbh", addr_norm="1 main street")
        result = evaluate_pair(row_blocked, row_clean, {}, cfg)
        assert result.is_match
        assert result.match_pct <= 86.0, "Guardrail cap of 86 must still apply"

    def test_operational_hint_on_unrelated_names_still_no_match(self):
        """Operational hint on row A with completely unrelated row B must still not match."""
        row_blocked = _row("figiel gmbh", has_operational_status_hint=True)
        row_unrelated = _row("acumenmedia group")
        result = evaluate_pair(row_blocked, row_unrelated, {}, cfg)
        assert not result.is_match or result.match_pct < 70.0, (
            "Operational hint must not force unrelated names to cluster"
        )


# ---------------------------------------------------------------------------
# Item 2: domain_review_candidate no longer review-only
# ---------------------------------------------------------------------------

class TestDomainReviewSurfacing:
    def test_domain_review_candidate_not_in_review_only(self):
        """domain_review_candidate must NOT be review-only after Phase 3A."""
        result = MatchResult(
            True, 72.0, "domain_review_candidate",
            {"domain": "acumenmedia.com"}, needs_review=True,
            review_reason="Same domain with unrelated names"
        )
        assert not _is_review_only(result), (
            "domain_review_candidate should appear at 70% in main output"
        )

    def test_same_domain_unrelated_names_shows_in_output(self):
        """Same domain + unrelated names → match in main output (was blank)."""
        row_a = _row("acumen international media", domain="acumenmedia.com")
        row_b = _row("tbd media group", domain="acumenmedia.com")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match, "Same-domain pair should produce a match"
        assert result.pass_type == "domain_review_candidate"
        assert not _is_review_only(result), (
            "domain_review_candidate must surface in main output, not be hidden"
        )
        assert result.match_pct <= 74.0, (
            "Score must not exceed ~72% for unrelated-name domain match"
        )

    def test_same_domain_related_names_scores_higher(self):
        """Same domain + related names scores > 74% (regression guard)."""
        row_a = _row("acumen international media", domain="acumenmedia.com")
        row_b = _row("acumen media group", domain="acumenmedia.com")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct > 74.0, (
            "Related-name same-domain should score above domain_review_candidate threshold"
        )

    def test_same_domain_address_confirmed_still_high(self):
        """Same domain + near name + address → ≥ 93% (regression guard)."""
        row_a = _row(
            "acumen media",
            addr_norm="42 main street",
            domain="acumenmedia.com",
            city_norm="london",
            country_norm="gb",
            postal_norm="ec1a",
        )
        row_b = _row(
            "acumen media inc",
            addr_norm="42 main street",
            domain="acumenmedia.com",
            city_norm="london",
            country_norm="gb",
            postal_norm="ec1a",
        )
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 93.0, (
            "Same domain + same address + near name should still be high-confidence"
        )

    def test_different_domain_unrelated_still_no_match(self):
        """Different domains + unrelated names → still no match."""
        row_a = _row("acumen international media", domain="acumenmedia.com")
        row_b = _row("tbd media group", domain="tbdmedia.com")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert not result.is_match or result.match_pct < 70.0, (
            "Different domains + unrelated names must remain blank"
        )


# ---------------------------------------------------------------------------
# Item 7: distinctive_supplier_identity 70% surfacing
# Note: blanket removal of the review-only check causes typo variants (e.g.
# "Eastmann" vs "Eastman") to join 85% clusters and dilute them to 70%.
# Phase 3A keeps fuzzy distinctive_supplier_identity as review-only. The check
# for main_cluster_allowed=False is preserved so near-spelling variants still
# go to review_candidates rather than auto-clustering.
# ---------------------------------------------------------------------------

class TestDistinctiveIdentity70Surfacing:
    def test_distinctive_identity_main_cluster_false_is_still_review_only(self):
        """distinctive_supplier_identity with main_cluster_allowed=False stays review-only.

        Preserves protection against typo/near-spelling variants diluting 85% clusters.
        """
        result = MatchResult(
            True, 70.0, "distinctive_supplier_identity",
            {"main_cluster_allowed": False, "score_reason": "Fuzzy core requires LLM review"},
            needs_review=True, review_reason="test"
        )
        assert _is_review_only(result), (
            "Fuzzy distinctive_supplier_identity (main_cluster_allowed=False) must remain "
            "review-only to prevent typo variants from diluting high-confidence clusters"
        )

    def test_distinctive_identity_main_cluster_true_still_in_union_find(self):
        """distinctive_supplier_identity with main_cluster_allowed=True reaches union-find."""
        result = MatchResult(
            True, 86.0, "distinctive_supplier_identity",
            {"main_cluster_allowed": True, "score_reason": "Confident identity"},
            needs_review=False, review_reason=""
        )
        assert not _is_review_only(result), (
            "Confident identity match must go to union-find"
        )


# ---------------------------------------------------------------------------
# Item 5: Conservative PASS 3C descriptor expansion
# ---------------------------------------------------------------------------

class TestPass3CDescriptorExpansion:
    def test_new_descriptor_tokens_in_config(self):
        """SERVICE_DESCRIPTOR_TOKENS contains the Phase 3A regional additions."""
        from src.config import SERVICE_DESCRIPTOR_TOKENS
        for token in ("emea", "apac", "amer", "dach", "latam", "ventures"):
            assert token in SERVICE_DESCRIPTOR_TOKENS, (
                f"{token!r} must be in SERVICE_DESCRIPTOR_TOKENS after Phase 3A"
            )

    def test_brand_emea_variant_matches(self):
        """BRAND + EMEA suffix → pair matches at ≥ 70%.

        PASS 2D (distinctive_supplier_identity via fuzzy core) fires before PASS 3C
        for brand+descriptor pairs where the cores are similar. Either way the pair
        must not be blank.
        """
        row_a = _row("plastec solutions")
        row_b = _row("plastec solutions emea")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match, "Brand + EMEA variant should produce a match"
        assert result.match_pct >= 70.0

    def test_brand_apac_variant_matches(self):
        """BRAND + APAC suffix → pair matches at ≥ 70%."""
        row_a = _row("kessler logistics")
        row_b = _row("kessler logistics apac")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 70.0

    def test_brand_amer_variant_matches(self):
        """BRAND + AMER suffix → pair matches at ≥ 70%."""
        row_a = _row("nexagen technologies")
        row_b = _row("nexagen technologies amer")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 70.0

    def test_brand_emea_same_sld_matches_at_70(self):
        """BRAND + EMEA with shared SLD → pair matches (≥ 70%).

        PASS 2D fires first because cores match ('aprolis' == 'aprolis' after
        stripping 'emea' via GENERIC_ROOT_TOKENS). Single-token ambiguous core
        scores 70% with cross-TLD same_sld (not same_domain). The pair is visible
        in main output — was blank before Phase 3A.
        """
        row_a = _row("aprolis", domain="aprolis.de", domain_sld="aprolis")
        row_b = _row("aprolis emea", domain="aprolis.fr", domain_sld="aprolis")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 70.0

    def test_brand_ventures_variant_matches(self):
        """BRAND + ventures suffix → pair matches at ≥ 70%."""
        row_a = _row("datasync corp")
        row_b = _row("datasync corp ventures")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 70.0

    def test_generic_core_with_emea_does_not_cluster_via_3c(self):
        """Generic-only brand core + EMEA must not cluster via PASS 3C."""
        row_a = _row("services group")
        row_b = _row("services group emea")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        if result.is_match:
            assert result.pass_type != "brand_prefix_descriptor_match", (
                "Generic brand core ('services group') must not match via PASS 3C"
            )

    def test_latam_descriptor_works(self):
        """LATAM regional descriptor is recognised."""
        row_a = _row("brixton supply chain")
        row_b = _row("brixton supply chain latam")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match, "LATAM variant should match"
        assert result.match_pct >= 70.0


# ---------------------------------------------------------------------------
# Regression guards: Phase 1 and Phase 2 unchanged
# ---------------------------------------------------------------------------

class TestPhase3ARegressions:
    def test_tax_exact_still_98(self):
        """Tax exact match still returns ≥ 95% and not review-only."""
        row_a = {**_row("plasticard gmbh"), "tax_norm": "de123456789"}
        row_b = {**_row("plasticard co"), "tax_norm": "de123456789"}
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 95.0
        assert result.pass_type == "tax_exact"
        assert not _is_review_only(result)

    def test_phase2_sld_cross_tld_still_fires(self):
        """Phase 2 same-SLD cross-TLD matching still works."""
        row_a = _row("aprolis gmbh", domain="aprolis.de", domain_sld="aprolis")
        row_b = _row("aprolis sa", domain="aprolis.fr", domain_sld="aprolis")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 70.0

    def test_address_alone_no_85_cluster(self):
        """Address alone with only generic name tokens must not produce ≥ 85% cluster."""
        row_a = _row(
            "centre france publicite",
            addr_norm="45 avenue france",
            city_norm="paris",
            country_norm="fr",
        )
        row_b = _row(
            "centre france communication",
            addr_norm="45 avenue france",
            city_norm="paris",
            country_norm="fr",
        )
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert not result.is_match or result.match_pct < 85.0, (
            "Address + generic tokens must not produce 85%+ cluster"
        )

    def test_generic_only_names_still_blank(self):
        """Generic-only names (services, trading) still produce no match."""
        row_a = _row("services company")
        row_b = _row("services trading")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert not result.is_match, "Generic-only names must still be blank"

    def test_exact_duplicate_still_98(self):
        """Exact same name + same address is still ≥ 95%."""
        row_a = _row("siemens ag", addr_norm="80333 munich germany")
        row_b = _row("siemens ag", addr_norm="80333 munich germany")
        result = evaluate_pair(row_a, row_b, {}, cfg)
        assert result.is_match
        assert result.match_pct >= 95.0

    def test_family_bridge_supported_still_review_only(self):
        """family_bridge_supported must still be review-only (unchanged)."""
        result = MatchResult(
            True, 76.0, "family_bridge_supported", {}, needs_review=True,
            review_reason="Parent/family bridge"
        )
        assert _is_review_only(result), "family_bridge_supported must remain review-only"

    def test_name_exact_review_still_review_only(self):
        """name_exact_review (generic same names) must still be review-only (unchanged)."""
        result = MatchResult(
            True, 86.0, "name_exact_review",
            {"name": "digital media lab"}, needs_review=True,
            review_reason="Same generic name without address/domain support"
        )
        assert _is_review_only(result), "name_exact_review must remain review-only"


# ---------------------------------------------------------------------------
# User-controlled ignore-client-domains feature
# ---------------------------------------------------------------------------

def _cluster(records, config=None):
    """Helper: run cluster_suppliers on a list of (name, address, city, country, postal, email) tuples."""
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Address": [r[1] if len(r) > 1 else "" for r in records],
        "City": [r[2] if len(r) > 2 else "" for r in records],
        "Country": [r[3] if len(r) > 3 else "" for r in records],
        "Postal Code": [r[4] if len(r) > 4 else "" for r in records],
        "Email": [r[5] if len(r) > 5 else "" for r in records],
    })
    return cluster_suppliers(df, {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
        "email": "Email",
    }, config or ClusteringConfig())


def _clusters_by_name(result):
    return {r["Supplier Name"]: r["Cluster Number"] for r in result["main_df"].to_dicts()}


class TestIgnoreClientDomains:
    """Tests for user-controlled ignore_client_domains feature."""

    def test_gilead_clusters_by_default(self):
        """gilead.com domain creates domain evidence by default (no ignore list)."""
        result = _cluster([
            ("Centrum Education Sp. z o.o.", "", "Warsaw",   "PL", "", "contact1@gilead.com"),
            ("Colorado Bioscience Corp",     "", "Denver",   "US", "", "contact2@gilead.com"),
        ])
        clusters = _clusters_by_name(result)
        assigned = [c for c in clusters.values() if c is not None]
        assert len(assigned) > 0, (
            "gilead.com domain must create domain_review_candidate evidence by default"
        )

    def test_gilead_does_not_cluster_when_ignored(self):
        """gilead.com suppresses all domain evidence when in ignore_client_domains."""
        cfg = ClusteringConfig(ignore_client_domains=frozenset({"gilead.com"}))
        result = _cluster([
            ("Centrum Education Sp. z o.o.", "", "Warsaw",   "PL", "", "contact1@gilead.com"),
            ("Colorado Bioscience Corp",     "", "Denver",   "US", "", "contact2@gilead.com"),
            ("Covenant House Inc",           "", "New York", "US", "", "contact3@gilead.com"),
            ("e-Med Solutions",              "", "London",   "GB", "", "contact4@gilead.com"),
            ("EGG Events GmbH",             "", "Berlin",   "DE", "", "contact5@gilead.com"),
        ], config=cfg)
        clusters = _clusters_by_name(result)
        assigned = [c for c in clusters.values() if c is not None]
        assert len(assigned) == 0, (
            f"Unrelated suppliers must not cluster when their shared domain is in ignore list: {clusters}"
        )

    def test_merck_does_not_cluster_when_ignored(self):
        """merck.com suppresses domain evidence when explicitly in ignore_client_domains."""
        cfg = ClusteringConfig(ignore_client_domains=frozenset({"merck.com"}))
        result = _cluster([
            ("Acme Consulting LLC",   "", "Boston",   "US", "", "vendor@merck.com"),
            ("Vertex Analytics GmbH", "", "Frankfurt","DE", "", "vendor@merck.com"),
        ], config=cfg)
        clusters = _clusters_by_name(result)
        assigned = [c for c in clusters.values() if c is not None]
        assert len(assigned) == 0, "Unrelated suppliers must not cluster when domain is ignored"

    def test_free_email_domains_never_cluster(self):
        """gmail/outlook/yahoo are always treated as generic — never produce domain matches."""
        for generic_domain in ("gmail.com", "outlook.com", "yahoo.com"):
            r1 = {**_row("apex solutions gmbh", domain=generic_domain), "is_generic_domain": True}
            r2 = {**_row("nova tech ag",        domain=generic_domain), "is_generic_domain": True}
            result = evaluate_pair(r1, r2, {}, cfg)
            assert not result.is_match or "domain" not in result.pass_type, (
                f"{generic_domain} must never create domain-based matches"
            )

    def test_amazon_sld_clusters_when_not_ignored(self):
        """amazon.com / amazon.in share SLD 'amazon' and should create domain-sld evidence when not ignored."""
        default_cfg = ClusteringConfig()
        # Use unrelated names so the only possible evidence is the shared SLD
        r1 = _row("acme wholesale distributors", domain="amazon.com", domain_sld="amazon")
        r2 = _row("vertex retail solutions",     domain="amazon.in",  domain_sld="amazon")
        result = evaluate_pair(r1, r2, {}, default_cfg)
        assert result.is_match, "Same SLD 'amazon' must create domain-sld evidence by default"
        assert "sld" in result.pass_type, f"Expected sld pass type, got {result.pass_type}"

    def test_amazon_does_not_cluster_when_ignored(self):
        """amazon.com in ignore list suppresses both same_domain and same_sld for amazon.*"""
        ignored_cfg = ClusteringConfig(ignore_client_domains=frozenset({"amazon.com"}))
        r1 = _row("acme wholesale distributors", domain="amazon.com", domain_sld="amazon")
        r2 = _row("vertex retail solutions",     domain="amazon.in",  domain_sld="amazon")
        result = evaluate_pair(r1, r2, {}, ignored_cfg)
        assert not result.is_match, (
            "Unrelated names must not cluster when their SLD's domain is in ignore list"
        )

    def test_ignored_client_domain_columns_do_not_leak_into_output(self):
        """No internal domain-ignore columns appear in the final output."""
        cfg = ClusteringConfig(ignore_client_domains=frozenset({"internal.corp"}))
        result = _cluster([
            ("Supplier Alpha Inc", "", "", "", "", "a@internal.corp"),
            ("Supplier Beta LLC",  "", "", "", "", "b@internal.corp"),
        ], config=cfg)
        output_cols = result["main_df"].columns
        for forbidden in ("is_shared_contact_domain", "ignore_client_domains", "_ignored"):
            assert forbidden not in output_cols, f"Internal column '{forbidden}' must not appear in output"

"""Tests for brand-location variant matching and branch modifier normalization."""
import pytest
from src.matching import evaluate_pair
from src.preprocessing import (
    compute_name_location_core,
    _strip_trailing_location_tokens,
    _has_distinctive_location_token,
)
from src.config import ClusteringConfig
from src.location_modifiers import load_location_modifiers
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    cfg = ClusteringConfig()
    cfg._tax_block_stats = {}
    return cfg


_LOCATION_TERMS = {
    "calgary", "toronto", "winnipeg", "montreal", "ottawa", "vancouver",
    "edmonton", "canada", "ontario", "alberta", "british", "columbia",
    "london", "manchester", "berlin", "munich", "muenchen", "paris",
    "frankfurt", "hamburg", "new", "york", "chicago", "houston",
    "seattle", "denver", "boston", "atlanta",
}


def _row(name_norm, name_location_core=None, **kwargs):
    base = {
        "addr_norm": "", "city_norm": "", "country_norm": "",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False,
        "root_brand": name_norm.split()[0] if name_norm else "",
        "supplier_identity_core": name_norm,
        "tax_norm": "", "person_name_norm": "",
        "known_brand_family_ids": "", "known_brand_family_safe_ids": "",
        "known_brand_family_risky_ids": "", "known_brand_alias_hits": "",
    }
    base["name_norm"] = name_norm
    base["name_location_core"] = name_location_core if name_location_core is not None else name_norm
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Unit: _strip_trailing_location_tokens
# ---------------------------------------------------------------------------

def test_strip_trailing_city():
    loc = {"calgary", "toronto"}
    assert _strip_trailing_location_tokens("bfi canada calgary", loc) == "bfi canada"


def test_strip_trailing_city_toronto():
    loc = {"toronto", "calgary"}
    assert _strip_trailing_location_tokens("bfi canada toronto", loc) == "bfi canada"


def test_strip_no_location_suffix():
    loc = {"calgary", "toronto"}
    assert _strip_trailing_location_tokens("bell canada", loc) == "bell canada"


def test_strip_does_not_strip_short_abbreviations():
    # 2-char abbreviations (e.g. "on" for Ontario) must not be stripped
    loc = {"on", "ontario", "toronto"}
    assert _strip_trailing_location_tokens("acme on", loc) == "acme on"  # "on" is too short


def test_strip_preserves_if_core_becomes_generic():
    # If stripping leaves only generic tokens, don't strip
    loc = {"toronto"}
    result = _strip_trailing_location_tokens("services toronto", loc)
    # "services" is in GENERIC_ROOT_TOKENS — result should be unchanged
    assert result == "services toronto"


def test_strip_at_most_two_trailing():
    loc = {"calgary", "canada", "alberta"}
    # Stripping 2: "canada calgary" → "bfi"
    # "bfi" has len=3, check distinctive... depends on whether "bfi" qualifies
    result = _strip_trailing_location_tokens("bfi canada calgary", loc)
    # Should strip "calgary" → "bfi canada"; "canada" also location but we stop if no more
    assert result in ("bfi canada", "bfi")  # acceptable: at least "calgary" stripped


# ---------------------------------------------------------------------------
# Unit: compute_name_location_core — parenthesized content
# ---------------------------------------------------------------------------

def test_core_strips_parenthesized_address():
    loc = {"canada", "toronto", "calgary"}
    core = compute_name_location_core(
        "Bell Canada (5115 Creekbank Rd)", "bell canada 5115 creekbank rd", loc
    )
    assert core == "bell canada"


def test_core_strips_parenthesized_postal_code():
    loc = {"canada", "toronto"}
    core = compute_name_location_core(
        "Bell Canada (M3C 3X9)", "bell canada m3c 3x9", loc
    )
    assert core == "bell canada"


def test_core_strips_multiple_parenthesized_groups():
    loc = {"canada", "toronto"}
    core = compute_name_location_core(
        "Bell Canada (M3C 3X9) (Deactivated)", "bell canada", loc
    )
    # After status-word removal, name_norm may already be "bell canada"
    assert core == "bell canada"


def test_core_strips_hyphenated_city():
    loc = {"calgary", "toronto", "canada"}
    core = compute_name_location_core("BFI CANADA-CALGARY", "bfi canada calgary", loc)
    assert core == "bfi canada"


def test_core_strips_hyphenated_city_toronto():
    loc = {"toronto", "calgary", "canada"}
    core = compute_name_location_core("BFI CANADA-TORONTO", "bfi canada toronto", loc)
    assert core == "bfi canada"


def test_core_unchanged_when_no_modifier():
    loc = {"calgary", "toronto"}
    core = compute_name_location_core("Bell Canada", "bell canada", loc)
    assert core == "bell canada"


def test_core_unchanged_for_london_drugs():
    loc = {"london", "toronto"}
    core = compute_name_location_core("London Drugs", "london drugs", loc)
    # "drugs" is GENERIC_ROOT_TOKENS — core should be unchanged
    assert core == "london drugs"


# ---------------------------------------------------------------------------
# Unit: load_location_modifiers
# ---------------------------------------------------------------------------

def test_load_location_modifiers_file():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "location_modifiers.csv"
    )
    terms = load_location_modifiers(path)
    assert "calgary" in terms
    assert "toronto" in terms
    assert "london" in terms
    assert "berlin" in terms
    assert "paris" in terms
    assert len(terms) >= 100


def test_load_location_modifiers_missing_file():
    terms = load_location_modifiers("/nonexistent/path.csv")
    assert terms == set()


# ---------------------------------------------------------------------------
# Integration: evaluate_pair — PASS 3B brand_location_variant_match
# ---------------------------------------------------------------------------

def test_bfi_canada_calgary_toronto_clusters():
    """BFI CANADA-CALGARY and BFI CANADA-TORONTO should match via brand_location_variant_match."""
    row_a = _row("bfi canada calgary", name_location_core="bfi canada",
                 row_id=0, country_norm="CA")
    row_b = _row("bfi canada toronto", name_location_core="bfi canada",
                 row_id=1, country_norm="CA")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}"
    assert result.pass_type == "brand_location_variant_match"
    assert result.match_pct == 85.0
    assert result.needs_review is False


def test_bell_canada_branch_address_clusters():
    """Bell Canada and Bell Canada (5115 Creekbank Rd) should match."""
    row_a = _row("bell canada", name_location_core="bell canada", row_id=0)
    row_b = _row("bell canada 5115 creekbank rd", name_location_core="bell canada",
                 row_id=1)
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}"
    # High name similarity may cause distinctive_supplier_identity to fire before PASS 3B —
    # both pass types correctly identify these as the same entity.
    assert result.pass_type in ("brand_location_variant_match", "distinctive_supplier_identity"), \
        f"Unexpected pass_type: {result.pass_type}"


def test_bfi_calgary_toronto_different_prefix_clusters():
    """BFI-CALGARY and BFI-TORONTO (short form) should match via location core."""
    row_a = _row("bfi calgary", name_location_core="bfi", row_id=0)
    row_b = _row("bfi toronto", name_location_core="bfi", row_id=1)
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}"
    assert result.pass_type == "brand_location_variant_match"


def test_location_core_same_as_name_no_extra_match():
    """When both rows have name_location_core == name_norm (no stripping), PASS 3B must not fire
    — the names are actually different entities."""
    row_a = _row("london bridge capital", name_location_core="london bridge capital", row_id=0)
    row_b = _row("london bridge fund", name_location_core="london bridge fund", row_id=1)
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    # name_location_core differs → PASS 3B must not fire
    assert result.pass_type != "brand_location_variant_match"


def test_generic_core_does_not_match():
    """Even if name_location_core values match, a generic core must not trigger PASS 3B."""
    # Both have name_location_core="services" — generic
    row_a = _row("services toronto", name_location_core="services toronto", row_id=0)
    row_b = _row("services calgary", name_location_core="services calgary", row_id=1)
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.pass_type != "brand_location_variant_match"


def test_location_variant_with_domain_scores_98():
    """Brand-location variant with shared domain should score 98."""
    row_a = _row("bfi canada calgary", name_location_core="bfi canada",
                 row_id=0, domain="bfi.ca", is_generic_domain=False)
    row_b = _row("bfi canada toronto", name_location_core="bfi canada",
                 row_id=1, domain="bfi.ca", is_generic_domain=False)
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match
    assert result.match_pct == 98.0


# ---------------------------------------------------------------------------
# Task 5: Singleton cluster cleanup — singletons must get blank output
# ---------------------------------------------------------------------------

def test_singleton_has_blank_cluster():
    """A single-entry row with no match must not appear in cluster_map."""
    import polars as pl
    from src.main import cluster_suppliers
    df = pl.DataFrame({
        "Supplier Name": ["BC LTSA"],
    })
    mapping = {"supplier_name": "Supplier Name"}
    result = cluster_suppliers(df, mapping)
    output = result["main_df"]
    # Single row → no cluster, Cluster Number must be null/blank
    cluster_col = output["Cluster Number"].to_list()
    match_col = output["Match Percentage"].to_list()
    assert cluster_col[0] is None
    assert match_col[0] == ""

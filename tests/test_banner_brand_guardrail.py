"""Tests: franchise/banner brand guardrail prevents banner-only 85% clustering.

The core bug: "BARRIE NISSAN" and "KELOWNA NISSAN" both have
name_location_core="nissan" (city stripped). When their NLCs match, PASS 2D-pre
fires → 85% (auto-cluster). They're different dealerships — 70% at most (LLM review).

The same pattern applies to Dairy Queen, Burger King, Marriott, Esso, etc.
Distinctive operator tokens (ALBI, GABRIEL) still allow 85%.
"""

import pytest
from src.matching import evaluate_pair
from src.config import ClusteringConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    cfg = ClusteringConfig()
    cfg._tax_block_stats = {}
    return cfg


def _row(name_norm, name_location_core=None, domain="", address="", tax="", country="CA", **kwargs):
    base = {
        "addr_norm": address.lower(),
        "city_norm": "",
        "country_norm": country,
        "postal_norm": "",
        "domain": domain,
        "is_generic_domain": False,
        "is_hospitality": False,
        "is_likely_individual": False,
        "has_legal_suffix": False,
        "has_company_keyword": True,
        "has_operational_status_hint": False,
        "root_brand": name_norm.split()[0] if name_norm else "",
        "supplier_identity_core": name_norm,
        "tax_norm": tax,
        "tax_loose_norm": tax,
        "person_name_norm": "",
        "known_brand_family_ids": "",
        "known_brand_family_safe_ids": "",
        "known_brand_family_risky_ids": "",
        "known_brand_alias_hits": "",
    }
    base["name_norm"] = name_norm
    base["name_location_core"] = name_location_core if name_location_core is not None else name_norm
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Test 1: Primary bug — Nissan dealers in different cities must not score 85%
# ---------------------------------------------------------------------------

def test_nissan_different_city_dealers_not_85():
    """APPLEWOOD NISSAN vs KELOWNA NISSAN share only the banner brand 'nissan'.
    Without operator/domain/address support this must be ≤70%, not 85%."""
    row_a = _row("applewood nissan", name_location_core="nissan")
    row_b = _row("kelowna nissan", name_location_core="nissan")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"applewood nissan vs kelowna nissan: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 2: Original bug report — Barrie Nissan vs Kelowna Nissan
# ---------------------------------------------------------------------------

def test_barrie_nissan_vs_kelowna_nissan_not_85():
    """BARRIE NISSAN and KELOWNA NISSAN — the original reported bug.
    Two separate dealerships must not auto-cluster at 85%."""
    row_a = _row("barrie nissan", name_location_core="nissan")
    row_b = _row("kelowna nissan", name_location_core="nissan")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"barrie nissan vs kelowna nissan: expected ≤70%, got {result.match_pct}% "
            f"(pass_type={result.pass_type})"
        )


# ---------------------------------------------------------------------------
# Test 3: Distinctive operator token (ALBI) preserves 85%
# ---------------------------------------------------------------------------

def test_albi_nissan_locations_still_85():
    """ALBI NISSAN MASCOUCHE vs ALBI NISSAN MONT-TREMBLANT share both 'albi' and 'nissan'.
    'albi' is a distinctive operator token → these are different branches of the same
    operator group and must still score ≥80%."""
    row_a = _row("albi nissan mascouche", name_location_core="albi nissan")
    row_b = _row("albi nissan mont tremblant", name_location_core="albi nissan")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, (
        f"albi nissan locations should match; got pass_type={result.pass_type}"
    )
    assert result.match_pct >= 80.0, (
        f"albi nissan locations: expected ≥80% (distinctive operator), got {result.match_pct}%"
    )


# ---------------------------------------------------------------------------
# Test 4: Distinctive operator token (GABRIEL) preserves 85%
# ---------------------------------------------------------------------------

def test_gabriel_nissan_locations_still_85():
    """NISSAN GABRIEL ANJOU vs NISSAN GABRIEL ST-LEONARD share 'nissan gabriel'.
    'gabriel' is a distinctive non-banner operator → must score ≥80%."""
    row_a = _row("nissan gabriel anjou", name_location_core="nissan gabriel")
    row_b = _row("nissan gabriel st leonard", name_location_core="nissan gabriel")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, (
        f"nissan gabriel locations should match; got pass_type={result.pass_type}"
    )
    assert result.match_pct >= 80.0, (
        f"nissan gabriel locations: expected ≥80% (distinctive operator), got {result.match_pct}%"
    )


# ---------------------------------------------------------------------------
# Test 5: Nissan with shared domain scores high
# ---------------------------------------------------------------------------

def test_nissan_banner_with_shared_domain_scores_high():
    """If both Nissan rows share a business domain (operator's site), they
    should score ≥85% — the domain proves same operator."""
    row_a = _row("barrie nissan", name_location_core="nissan", domain="barrie-nissan.example")
    row_b = _row("kelowna nissan", name_location_core="nissan", domain="barrie-nissan.example")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match
    assert result.match_pct >= 85.0, (
        f"nissan with shared domain should be ≥85%, got {result.match_pct}%"
    )


# ---------------------------------------------------------------------------
# Test 6: Dairy Queen city variants must not score 85% without support
# ---------------------------------------------------------------------------

def test_dairy_queen_different_locations_not_85():
    """DAIRY QUEEN NORTH vs DAIRY QUEEN SOUTH — banner-only, no operator evidence.
    Must be ≤70%."""
    row_a = _row("dairy queen north", name_location_core="dairy queen")
    row_b = _row("dairy queen south", name_location_core="dairy queen")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"dairy queen north/south: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 7: Burger King city variants must not score 85% without support
# ---------------------------------------------------------------------------

def test_burger_king_different_cities_not_85():
    """BURGER KING TORONTO vs BURGER KING CALGARY — banner-only.
    Must be ≤70% without operator/domain/address support."""
    row_a = _row("burger king toronto", name_location_core="burger king")
    row_b = _row("burger king calgary", name_location_core="burger king")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"burger king different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 8: Marriott banner-only rows must not score 85% without support
# ---------------------------------------------------------------------------

def test_marriott_different_cities_not_85():
    """MARRIOTT TORONTO vs MARRIOTT CALGARY — hotel brand only.
    Without operator evidence must be ≤70%."""
    row_a = _row("marriott toronto", name_location_core="marriott")
    row_b = _row("marriott calgary", name_location_core="marriott")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"marriott different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 9: Esso fuel banner-only rows must not score 85%
# ---------------------------------------------------------------------------

def test_esso_banner_only_not_85():
    """ESSO STATION TORONTO vs ESSO STATION CALGARY — fuel brand only.
    Without operator/address support must be ≤70%."""
    row_a = _row("esso station toronto", name_location_core="esso")
    row_b = _row("esso station calgary", name_location_core="esso")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"esso different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 10: Fido (telecom reseller) banner-only must not score 85%
# ---------------------------------------------------------------------------

def test_fido_banner_only_not_85():
    """FIDO STORE TORONTO vs FIDO STORE CALGARY — telecom reseller brand only.
    Without operator/address support must be ≤70%."""
    row_a = _row("fido store toronto", name_location_core="fido")
    row_b = _row("fido store calgary", name_location_core="fido")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"fido different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 11: Toyota dealers, different cities — banner-only must not score 85%
# ---------------------------------------------------------------------------

def test_toyota_dealer_different_cities_not_85():
    """TORONTO TOYOTA vs BARRIE TOYOTA — auto dealer banner only.
    Must be ≤70% without operator/address support."""
    row_a = _row("toronto toyota", name_location_core="toyota")
    row_b = _row("barrie toyota", name_location_core="toyota")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"toyota different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 12: Honda dealers with shared tax ID should still cluster
# ---------------------------------------------------------------------------

def test_honda_dealers_with_same_tax_can_cluster():
    """Two Honda dealer rows sharing the same tax ID are likely same legal entity
    → should still match (tax support overrides banner-only guard)."""
    row_a = _row("honda north", name_location_core="honda", tax="CA123456789")
    row_b = _row("honda south", name_location_core="honda", tax="CA123456789")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, (
        "Honda rows with same tax ID should still match despite banner-only guard"
    )


# ---------------------------------------------------------------------------
# Test 13: Nissan with address support scores high
# ---------------------------------------------------------------------------

def test_nissan_with_address_support_scores_high():
    """Nissan rows sharing the same address (branch/parking-lot variant) should
    still score ≥85% — same physical location evidence overrides the banner guard."""
    row_a = _row("nissan downtown", name_location_core="nissan",
                 address="100 auto mile road toronto on")
    row_b = _row("nissan downtown service", name_location_core="nissan",
                 address="100 auto mile road toronto on")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match
    assert result.match_pct >= 85.0, (
        f"nissan with same address should be ≥85%, got {result.match_pct}%"
    )


# ---------------------------------------------------------------------------
# Test 14: Starbucks city variants must not score 85%
# ---------------------------------------------------------------------------

def test_starbucks_banner_only_not_85():
    """STARBUCKS TORONTO vs STARBUCKS EDMONTON — food/café brand only.
    Must be ≤70% without operator support."""
    row_a = _row("starbucks toronto", name_location_core="starbucks")
    row_b = _row("starbucks edmonton", name_location_core="starbucks")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"starbucks different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 15: Hilton banner-only must not score 85%
# ---------------------------------------------------------------------------

def test_hilton_banner_only_not_85():
    """HILTON TORONTO vs HILTON VANCOUVER — hotel brand only.
    Must be ≤70% without operator/address support."""
    row_a = _row("hilton toronto", name_location_core="hilton")
    row_b = _row("hilton vancouver", name_location_core="hilton")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"hilton different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 16: BFI Canada recall guard — non-banner brand still gets 85%
# ---------------------------------------------------------------------------

def test_bfi_canada_location_variants_still_85():
    """BFI CANADA-CALGARY vs BFI CANADA-TORONTO — 'bfi' is not a banner brand.
    Must still score 85% (location variant of same distinctive brand)."""
    row_a = _row("bfi canada calgary", name_location_core="bfi canada")
    row_b = _row("bfi canada toronto", name_location_core="bfi canada")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, (
        f"bfi canada location variants should still match; pass_type={result.pass_type}"
    )
    assert result.match_pct >= 85.0, (
        f"bfi canada location variants: expected ≥85%, got {result.match_pct}%"
    )


# ---------------------------------------------------------------------------
# Test 17: Purolator service franchise — banner-only must not score 85%
# ---------------------------------------------------------------------------

def test_purolator_banner_only_not_85():
    """PUROLATOR TORONTO vs PUROLATOR CALGARY — courier franchise banner only.
    Must be ≤70% without operator/address support."""
    row_a = _row("purolator toronto", name_location_core="purolator")
    row_b = _row("purolator calgary", name_location_core="purolator")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"purolator different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 18: Ford dealer — banner-only must not score 85%
# ---------------------------------------------------------------------------

def test_ford_dealer_different_cities_not_85():
    """FORD TORONTO vs FORD BARRIE — auto dealer banner only.
    Must be ≤70% without operator/address support."""
    row_a = _row("ford toronto", name_location_core="ford")
    row_b = _row("ford barrie", name_location_core="ford")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.match_pct <= 70.0, (
            f"ford different cities: banner-only must be ≤70%, got {result.match_pct}%"
        )


# ---------------------------------------------------------------------------
# Test 19: Needs_review flag is set when banner-only downgrade fires
# ---------------------------------------------------------------------------

def test_nissan_banner_only_sets_needs_review():
    """When the banner-only guard downgrades to 70%, needs_review must be True
    so the pair is routed to the LLM review queue."""
    row_a = _row("barrie nissan", name_location_core="nissan")
    row_b = _row("kelowna nissan", name_location_core="nissan")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match and result.match_pct <= 70.0:
        assert result.needs_review is True, (
            "Banner-only 70% match must set needs_review=True for LLM routing"
        )

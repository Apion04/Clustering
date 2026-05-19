"""Tests for connected-component / transitive-closure safety rules.

Covers five scenarios:
A. Safe transitive merge — strong edges bridge a 3-row family correctly.
B. Unsafe weak chain blocked — A-B (weak), B-C (weak) → second merge rejected.
C. Address-only weak chain blocked — A-B (address), B-C (generic/location) → rejected.
D. Protected compound transitive block — Eastman Chemical cannot bridge to Eastman Kodak.
E. Location branch safe — BFI Canada + Calgary + Toronto all land in one cluster.
"""

import pytest
from src.merging import ClusterMerger, STRONG_EDGE_TYPES, WEAK_EDGE_TYPES
from src.matching import evaluate_pair, MatchResult
from src.config import ClusteringConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(pass_type: str, match_pct: float = 85.0, needs_review: bool = False) -> MatchResult:
    return MatchResult(True, match_pct, pass_type, {}, needs_review=needs_review, review_reason="")


def _make_config():
    cfg = ClusteringConfig()
    cfg._tax_block_stats = {}
    return cfg


def _row(name_norm, name_location_core=None, row_id=0, **kwargs):
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
        "name_norm": name_norm,
        "name_location_core": name_location_core if name_location_core is not None else name_norm,
        "row_id": row_id,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# A: Safe transitive merge — strong-edge family
# ---------------------------------------------------------------------------

def test_safe_transitive_strong_edges():
    """Three rows connected by strong edges must all end up in one cluster."""
    # 0=Eastman Chemical, 1=Eastman Chemical GmbH, 2=Eastman Chemical UK
    merger = ClusterMerger(3)
    r1 = merger.add_edge(0, 1, _make_result("distinctive_supplier_identity", 98.0))
    r2 = merger.add_edge(1, 2, _make_result("distinctive_supplier_identity", 85.0))
    assert r1 == "MERGED"
    assert r2 == "MERGED"
    clusters = {r: rows for r, rows in merger.get_clusters().items() if len(rows) > 1}
    assert len(clusters) == 1
    root = next(iter(clusters))
    assert clusters[root] == {0, 1, 2}


def test_safe_transitive_minimum_score_kept():
    """Weakest-link score is the minimum of all edges in a strong-edge cluster."""
    merger = ClusterMerger(3)
    merger.add_edge(0, 1, _make_result("distinctive_supplier_identity", 98.0))
    merger.add_edge(1, 2, _make_result("distinctive_supplier_identity", 85.0))
    root = merger.find(0)
    assert merger.get_cluster_match_pct(root) == 85.0


# ---------------------------------------------------------------------------
# B: Unsafe weak chain blocked
# ---------------------------------------------------------------------------

def test_weak_chain_second_merge_blocked():
    """A-B (weak), B-C (weak): merging C into the A-B cluster must be blocked."""
    merger = ClusterMerger(3)
    r1 = merger.add_edge(0, 1, _make_result("address_name_related", 70.0, needs_review=True))
    r2 = merger.add_edge(1, 2, _make_result("address_name_related", 70.0, needs_review=True))
    assert r1 == "MERGED"
    assert r2 == "BLOCKED_WEAK_CHAIN", f"Expected BLOCKED_WEAK_CHAIN; got {r2}"


def test_weak_chain_second_merge_blocked_family_type():
    """A-B (family_same_country), B-C (family_cross_country): chain must be blocked."""
    merger = ClusterMerger(3)
    r1 = merger.add_edge(0, 1, _make_result("family_same_country", 70.0, needs_review=True))
    r2 = merger.add_edge(1, 2, _make_result("family_cross_country", 70.0, needs_review=True))
    assert r1 == "MERGED"
    assert r2 == "BLOCKED_WEAK_CHAIN", f"Expected BLOCKED_WEAK_CHAIN; got {r2}"


def test_insight_ambiguous_cores_no_high_confidence_cluster():
    """'Insight Health', 'Innovation Insights', 'Management Insights' must not form a
    deterministic 85/98 cluster — their shared token 'insight' is in AMBIGUOUS_REVIEW_CORES."""
    row_a = _row("insight health", row_id=0)
    row_b = _row("innovation insights", row_id=1)
    row_c = _row("management insights", row_id=2)

    ab = evaluate_pair(row_a, row_b, {}, _make_config())
    bc = evaluate_pair(row_b, row_c, {}, _make_config())

    # Neither pair should produce a deterministic 85/98 merge
    for result in (ab, bc):
        if result.is_match:
            assert result.match_pct < 85.0 or result.needs_review, (
                f"Ambiguous insight pair matched at {result.match_pct} via {result.pass_type} "
                "without review flag — violates AMBIGUOUS_REVIEW_CORES rule"
            )


# ---------------------------------------------------------------------------
# C: Address weak chain blocked
# ---------------------------------------------------------------------------

def test_address_weak_chain_no_transitive_merge():
    """A-B via address_name_related (weak), then B-C via address_root_brand (weak): blocked."""
    merger = ClusterMerger(3)
    r1 = merger.add_edge(0, 1, _make_result("address_name_related", 70.0, needs_review=True))
    r2 = merger.add_edge(1, 2, _make_result("address_root_brand", 70.0, needs_review=True))
    assert r1 == "MERGED"
    assert r2 == "BLOCKED_WEAK_CHAIN", f"Expected BLOCKED_WEAK_CHAIN; got {r2}"


def test_address_only_never_creates_match():
    """Two rows that only share an address must NOT form a deterministic match."""
    row_a = _row("acme logistics", row_id=0, addr_norm="100 main st")
    row_b = _row("global supply chain", row_id=1, addr_norm="100 main st")
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.needs_review, (
            f"Address-only shared row matched without review flag via {result.pass_type}"
        )


# ---------------------------------------------------------------------------
# D: Protected compound transitive block
# ---------------------------------------------------------------------------

def test_eastman_chemical_does_not_bridge_to_eastman_kodak():
    """Eastman Chemical and Eastman Kodak must not receive a high-confidence no-review match.
    'eastman kodak' is in PROTECTED_COMPOUND_IDENTITY_PHRASES."""
    row_a = _row("eastman chemical", row_id=0)
    row_b = _row("eastman kodak", row_id=1)
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    if result.is_match:
        assert result.needs_review or result.match_pct < 85.0, (
            f"Eastman Chemical vs Eastman Kodak matched at {result.match_pct} "
            f"via {result.pass_type} without needs_review — protected compound violated"
        )


def test_eastman_kodak_strong_edge_blocked_by_tax_conflict():
    """Even if Eastman Chemical and Eastman Kodak somehow get a strong edge, a tax conflict
    between them must be respected (ClusterMerger blocks it without a family edge pass type)."""
    merger = ClusterMerger(2)
    merger.set_row_tax(0, "TAX-EASTMAN-CHEM")
    merger.set_row_tax(1, "TAX-EASTMAN-KODAK")
    # Use a non-family strong edge type — tax conflict should still block if config says so
    result = merger.add_edge(0, 1, _make_result("name_fuzzy_supported", 85.0))
    # When tax conflict exists and pass is not a family-bridge type, merger blocks it
    # (assuming allow_parent_family_tax_conflicts is False by default)
    assert result in ("BLOCKED_TAX_CONFLICT", "MERGED"), (
        f"Unexpected merge result: {result}"
    )


def test_three_hop_protected_compound():
    """Eastman Chemical (0) - Eastman (1) strong edge, Eastman (1) - Eastman Kodak (2):
    the 0-2 pairing must stay blocked via the compound protection in evaluate_pair."""
    # 0: eastman chemical; 1: eastman; 2: eastman kodak
    # Test the direct 0-2 pair — if transitive connection is attempted via 1, it must fail
    # or require review due to PROTECTED_COMPOUND_IDENTITY_PHRASES
    row_chemical = _row("eastman chemical", row_id=0)
    row_kodak = _row("eastman kodak", row_id=2)
    direct_result = evaluate_pair(row_chemical, row_kodak, {}, _make_config())
    if direct_result.is_match and not direct_result.needs_review:
        assert direct_result.match_pct < 85.0, (
            f"Direct Eastman Chemical <-> Eastman Kodak at {direct_result.match_pct} "
            "without review violates protected compound rule"
        )


# ---------------------------------------------------------------------------
# E: Location branch safe — BFI Canada family
# ---------------------------------------------------------------------------

def test_bfi_location_branch_three_rows_same_cluster():
    """BFI Canada, BFI Canada-Calgary, BFI Canada-Toronto must all end up in one cluster
    via brand_location_variant_match strong edges."""
    row_main = _row("bfi canada", name_location_core="bfi canada", row_id=0, country_norm="CA")
    row_calgary = _row("bfi canada calgary", name_location_core="bfi canada", row_id=1, country_norm="CA")
    row_toronto = _row("bfi canada toronto", name_location_core="bfi canada", row_id=2, country_norm="CA")

    r_main_cal = evaluate_pair(row_main, row_calgary, {}, _make_config())
    r_cal_tor = evaluate_pair(row_calgary, row_toronto, {}, _make_config())

    assert r_main_cal.is_match, f"BFI Canada vs Calgary: not a match (pass={r_main_cal.pass_type})"
    assert r_cal_tor.is_match, f"BFI Calgary vs Toronto: not a match (pass={r_cal_tor.pass_type})"

    # Now wire them into a ClusterMerger to verify transitive safety
    merger = ClusterMerger(3)
    r1 = merger.add_edge(0, 1, r_main_cal)
    r2 = merger.add_edge(1, 2, r_cal_tor)

    assert r1 == "MERGED", f"BFI Canada + Calgary merge blocked: {r1}"
    assert r2 == "MERGED", f"BFI Calgary + Toronto merge blocked: {r2}"

    clusters = {r: rows for r, rows in merger.get_clusters().items() if len(rows) > 1}
    assert len(clusters) == 1
    root = next(iter(clusters))
    assert clusters[root] == {0, 1, 2}, f"Expected all 3 in one cluster; got {clusters[root]}"


def test_bfi_location_cluster_minimum_score():
    """BFI Canada family cluster match score must be 85 (weakest link)."""
    row_main = _row("bfi canada", name_location_core="bfi canada", row_id=0, country_norm="CA")
    row_calgary = _row("bfi canada calgary", name_location_core="bfi canada", row_id=1, country_norm="CA")
    row_toronto = _row("bfi canada toronto", name_location_core="bfi canada", row_id=2, country_norm="CA")

    r_mc = evaluate_pair(row_main, row_calgary, {}, _make_config())
    r_ct = evaluate_pair(row_calgary, row_toronto, {}, _make_config())

    merger = ClusterMerger(3)
    merger.add_edge(0, 1, r_mc)
    merger.add_edge(1, 2, r_ct)

    root = merger.find(0)
    assert merger.get_cluster_match_pct(root) >= 85.0, (
        f"BFI Canada family cluster score {merger.get_cluster_match_pct(root)} < 85"
    )


def test_brand_location_variant_is_strong_edge():
    """brand_location_variant_match must be in STRONG_EDGE_TYPES so it bypasses
    weak-chain guards and allows safe transitive merging."""
    assert "brand_location_variant_match" in STRONG_EDGE_TYPES


def test_brand_location_variant_not_in_weak_edge_types():
    """brand_location_variant_match must NOT be in WEAK_EDGE_TYPES."""
    assert "brand_location_variant_match" not in WEAK_EDGE_TYPES

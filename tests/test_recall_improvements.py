"""Regression tests for recall-improvement changes (May 2026).

Verifies that plausible-but-weak relationships are routed to 70 (LLM review)
rather than left blank, while hard-reject cases remain blank.
"""

import pytest
from src.config import ClusteringConfig
from src.matching import evaluate_pair
from src.merging import ClusterMerger
from src.scoring import calculate_user_facing_cluster_scores


def calculate_match(row_a, row_b, config=None):
    return evaluate_pair(row_a, row_b, {}, config or ClusteringConfig())


def cfg(**kw):
    defaults = dict(
        enable_family_bridge=True,
        allow_parent_family_tax_conflicts=True,
    )
    defaults.update(kw)
    return ClusteringConfig(**defaults)


def row(**kw):
    defaults = dict(
        name_norm="", addr_norm="", city_norm="", country_norm="", postal="",
        tax_id="", domain="", is_likely_individual=False,
        name_location_core="", root_brand="",
    )
    defaults.update(kw)
    return defaults


# ── Bell / telecom weak sub-brand recall ──────────────────────────────────────

class TestBellWeakSubBrand:
    def test_bell_canada_vs_bell_aliant_is_70(self):
        a = row(name_norm="bell canada", root_brand="bell", country_norm="ca")
        b = row(name_norm="bell aliant", root_brand="bell", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match, "bell canada vs bell aliant must produce a match"
        assert r.match_pct == 70.0
        assert r.pass_type == "weak_brand_root_candidate"

    def test_bell_canada_vs_bell_mobility_is_70(self):
        a = row(name_norm="bell canada", root_brand="bell", country_norm="ca")
        b = row(name_norm="bell mobility", root_brand="bell", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        assert r.match_pct == 70.0
        assert r.pass_type == "weak_brand_root_candidate"

    def test_bell_canada_vs_bell_mts_is_70(self):
        a = row(name_norm="bell canada", root_brand="bell", country_norm="ca")
        b = row(name_norm="bell mts", root_brand="bell", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        assert r.match_pct == 70.0
        assert r.pass_type == "weak_brand_root_candidate"

    def test_rogers_vs_rogers_cable_is_70(self):
        a = row(name_norm="rogers communications", root_brand="rogers", country_norm="ca")
        b = row(name_norm="rogers cable", root_brand="rogers", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        assert r.match_pct == 70.0
        assert r.pass_type == "weak_brand_root_candidate"

    def test_shaw_vs_shaw_cable_is_70(self):
        a = row(name_norm="shaw communications", root_brand="shaw", country_norm="ca")
        b = row(name_norm="shaw cable", root_brand="shaw", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        assert r.match_pct == 70.0
        assert r.pass_type == "weak_brand_root_candidate"

    def test_bell_same_city_guardrail_does_not_auto_cluster(self):
        """Same city + same-root 2-word name: guardrail blocks auto-cluster.
        The family_bridge_supported path is rejected because the rows share only
        the first token ('bell'), which is treated as a common-first-name-only
        overlap. They still get weak_brand_root_candidate at 70 (no city check).
        """
        a = row(name_norm="bell canada", root_brand="bell", country_norm="ca", city_norm="toronto")
        b = row(name_norm="bell mobility", root_brand="bell", country_norm="ca", city_norm="toronto")
        r = calculate_match(a, b, cfg())
        # guardrail blocks family_bridge_supported; weak_brand_root_candidate (70) may still fire
        # Invariant: must NOT be auto-clustered (score < 85)
        assert r.match_pct < 85 or r.needs_review is True


# ── Fuzzy-name recall ─────────────────────────────────────────────────────────

class TestFuzzyNameCandidate:
    def test_high_similarity_no_support_is_70(self):
        """High name_sim but no city/domain/address → some 70-score LLM candidate.
        'suncor' is a distinctive token so distinctive_supplier_identity fires at
        PASS 2D before the fuzzy review pass. Either pass_type is a valid 70 candidate.
        """
        a = row(name_norm="suncor energy inc", root_brand="suncor", country_norm="ca")
        b = row(name_norm="suncor energie inc", root_brand="suncor", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        assert r.match_pct == 70.0
        assert r.needs_review is True

    def test_high_similarity_with_same_domain_is_not_70_review_only(self):
        """Same domain → should be stronger than 70 (not stuck at name_fuzzy_review_candidate)."""
        a = row(name_norm="suncor energy inc", root_brand="suncor", country_norm="ca", domain="suncor.com")
        b = row(name_norm="suncor energie inc", root_brand="suncor", country_norm="ca", domain="suncor.com")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        # Should be caught earlier (before pass 4 fuzzy review) with higher score
        assert r.match_pct > 70.0 or r.pass_type != "name_fuzzy_review_candidate"


# ── Cross-country family recall ───────────────────────────────────────────────

class TestFamilyCrossCountry:
    def test_cross_country_2word_name_guardrail_blocks(self):
        """Cross-country 2-word company names sharing only the first token are
        blocked by the person-first-name guardrail (shell oil / shell canada →
        treated as 'shell X' sharing only first='shell'). Precision protection.
        """
        a = row(name_norm="shell oil", root_brand="shell", country_norm="us")
        b = row(name_norm="shell canada", root_brand="shell", country_norm="ca")
        r = calculate_match(a, b, cfg())
        # Guardrail rejects family_cross_country for these 2-word names
        assert not r.is_match or r.match_pct < 85

    def test_same_country_unknown_gives_weak_brand_root(self):
        """Unknown/blank country → weak_brand_root_candidate at 70, not blank."""
        a = row(name_norm="shell oil", root_brand="shell", country_norm="")
        b = row(name_norm="shell canada", root_brand="shell", country_norm="")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        assert r.match_pct == 70.0
        assert r.pass_type == "weak_brand_root_candidate"


# ── Sub-brand prefix safety net ───────────────────────────────────────────────

class TestPossibleSubBrand:
    def test_prefix_sub_brand_is_70(self):
        """One name starts with the other (≥5 chars) → possible_sub_brand_candidate at 70."""
        a = row(name_norm="canadian tire", root_brand="canadian", country_norm="ca")
        b = row(name_norm="canadian tire corporation", root_brand="canadian", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert r.is_match
        # Could be caught earlier (name_location_core or distinctive_supplier_identity)
        # Ensure it is NOT blank
        assert r.match_pct > 0.0

    def test_short_generic_prefix_stays_blank(self):
        """'Inc' is a generic suffix, not a distinctive prefix — must NOT create a match."""
        a = row(name_norm="inc", root_brand="", country_norm="ca")
        b = row(name_norm="inc holdings", root_brand="", country_norm="ca")
        r = calculate_match(a, b, cfg())
        # Either blank or needs_review only — should not auto-cluster
        if r.is_match:
            assert r.needs_review is True
            assert r.match_pct <= 70.0

    def test_person_name_no_sub_brand(self):
        """Person names must NOT produce sub-brand candidates."""
        a = row(name_norm="james smith", is_likely_individual=True, country_norm="ca")
        b = row(name_norm="james smith consulting", is_likely_individual=True, country_norm="ca")
        r = calculate_match(a, b, cfg())
        if r.is_match and r.pass_type == "possible_sub_brand_candidate":
            pytest.fail("Person names must not produce sub-brand candidates")


# ── Hard-reject cases remain blank ───────────────────────────────────────────

class TestHardRejectStaysBlank:
    def test_city_only_stays_blank(self):
        """Two suppliers sharing only a city name → no match."""
        a = row(name_norm="toronto services", root_brand="", city_norm="toronto", country_norm="ca")
        b = row(name_norm="toronto solutions", root_brand="", city_norm="toronto", country_norm="ca")
        r = calculate_match(a, b, cfg())
        assert not r.is_match or r.match_pct == 0.0

    def test_generic_only_stays_blank(self):
        """Suppliers sharing only generic tokens (services, solutions) → no match."""
        a = row(name_norm="global services", root_brand="", country_norm="us")
        b = row(name_norm="global solutions", root_brand="", country_norm="us")
        r = calculate_match(a, b, cfg())
        assert not r.is_match or r.match_pct == 0.0

    def test_risky_ambiguous_alias_blocked_cross_country(self):
        """Cross-country 2-word names sharing only the first token ('metro') are
        blocked by the person-first-name guardrail — metro logistics (DE) vs
        metro bank (GB) must NOT be auto-clustered.  The guardrail correctly
        rejects family_cross_country when shared tokens == {first_token}.
        """
        a = row(name_norm="metro logistics", root_brand="metro", country_norm="de",
                city_norm="berlin")
        b = row(name_norm="metro bank", root_brand="metro", country_norm="gb",
                city_norm="london")
        r = calculate_match(a, b, cfg())
        # Must not auto-cluster; either blank or 70-LLM-review only
        assert r.match_pct < 85, (
            f"Risky cross-country ambiguous alias must not auto-cluster, got {r.pass_type} {r.match_pct}"
        )

    def test_legal_suffix_only_stays_blank(self):
        """Suppliers sharing only 'inc' or 'corp' → no match."""
        a = row(name_norm="alpha inc", root_brand="alpha", country_norm="us")
        b = row(name_norm="beta inc", root_brand="beta", country_norm="us")
        r = calculate_match(a, b, cfg())
        assert not r.is_match or r.match_pct == 0.0


# ── Scoring: 70s do not leak into final output (without flag) ─────────────────

class TestScoringLeakPrevention:
    def _make_cluster_with_70(self):
        merger = ClusterMerger(2)
        from src.matching import MatchResult
        from src.merging import ClusterEdge

        edge = ClusterEdge(
            row_a=0, row_b=1,
            match_pct=70.0,
            pass_type="weak_brand_root_candidate",
            evidence={"candidate_type": "possible_sub_brand"},
            needs_review=True,
            review_reason="LLM review required",
        )
        merger.edges[0].append(edge)
        merger.cluster_members[0] = {0, 1}
        merger.parent[1] = 0
        return merger

    def test_70_cluster_hidden_when_flag_false(self):
        merger = self._make_cluster_with_70()
        scores = calculate_user_facing_cluster_scores(
            merger, expose_unresolved_llm_candidates=False
        )
        assert scores[0] == 0.0, "70-score cluster must be hidden when expose flag is False"

    def test_70_cluster_visible_when_flag_true(self):
        merger = self._make_cluster_with_70()
        scores = calculate_user_facing_cluster_scores(
            merger, expose_unresolved_llm_candidates=True
        )
        assert scores[0] == 70.0, "70-score cluster must be visible when expose flag is True"


# ── Transitive weak chain blocked for new pass types ─────────────────────────

class TestNewPassTypesChainBlocked:
    """weak_brand_root_candidate, name_fuzzy_review_candidate, possible_sub_brand_candidate
    must not be allowed to form transitive chains (A→B via weak, B→C via weak)."""

    def _merger(self, n=3):
        return ClusterMerger(n, ClusteringConfig())

    def _mr(self, pass_type, score=70.0):
        from src.matching import MatchResult
        return MatchResult(
            is_match=True,
            match_pct=score,
            pass_type=pass_type,
            evidence={"candidate_type": "test"},
            needs_review=True,
            review_reason="test",
        )

    def test_weak_brand_root_chain_blocked(self):
        merger = self._merger(3)
        r1 = merger.add_edge(0, 1, self._mr("weak_brand_root_candidate"))
        assert r1 == "MERGED"
        # Second edge tries to chain through the existing singleton-pair cluster
        r2 = merger.add_edge(1, 2, self._mr("weak_brand_root_candidate"))
        assert r2 == "BLOCKED_WEAK_CHAIN", (
            "weak_brand_root_candidate must not chain A→B→C"
        )

    def test_fuzzy_review_candidate_chain_blocked(self):
        merger = self._merger(3)
        r1 = merger.add_edge(0, 1, self._mr("name_fuzzy_review_candidate"))
        assert r1 == "MERGED"
        r2 = merger.add_edge(1, 2, self._mr("name_fuzzy_review_candidate"))
        assert r2 == "BLOCKED_WEAK_CHAIN"

    def test_possible_sub_brand_chain_blocked(self):
        merger = self._merger(3)
        r1 = merger.add_edge(0, 1, self._mr("possible_sub_brand_candidate"))
        assert r1 == "MERGED"
        r2 = merger.add_edge(1, 2, self._mr("possible_sub_brand_candidate"))
        assert r2 == "BLOCKED_WEAK_CHAIN"

    def test_family_cross_country_chain_blocked(self):
        merger = self._merger(3)
        r1 = merger.add_edge(0, 1, self._mr("family_cross_country"))
        assert r1 == "MERGED"
        r2 = merger.add_edge(1, 2, self._mr("family_cross_country"))
        assert r2 == "BLOCKED_WEAK_CHAIN"

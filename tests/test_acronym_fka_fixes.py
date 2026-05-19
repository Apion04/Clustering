"""Regression tests for acronym/FKA/spacing/franchise/same-owner clustering fixes."""
import pytest
from src.matching import (
    evaluate_pair,
    _generate_acronym,
    _acronym_bridge_full,
    _compact_names_match,
    _full_name_for_acronym,
    secondary_name_match,
    _distinctive_address_tokens,
)
from src.guardrails import apply_guardrails
from src.matching_types import MatchResult
from src.config import ClusteringConfig


# ---------------------------------------------------------------------------
# Unit: acronym generation
# ---------------------------------------------------------------------------

def test_generate_acronym_naacp():
    assert _generate_acronym("national association for the advancement of colored people") == "naacp"


def test_generate_acronym_ficci():
    assert _generate_acronym("the federation of indian chambers of commerce industry") == "ficci"


def test_generate_acronym_rmhc():
    assert _generate_acronym("ronald mcdonald house charities of") == "rmhc"


def test_generate_acronym_too_few_tokens():
    # Only 2 significant tokens → should return "" (need ≥ 3)
    assert _generate_acronym("national association") == ""


def test_generate_acronym_strips_legal():
    # Legal suffix "inc" should be excluded
    assert _generate_acronym("alpha beta gamma inc") == "abg"


# ---------------------------------------------------------------------------
# Unit: acronym bridge (multi-field)
# ---------------------------------------------------------------------------

def test_acronym_bridge_naacp_vs_full_name():
    row_a = {"name_norm": "naacp"}
    row_b = {
        "name_norm": "national association for the",
        "name2_norm": "advancement of colored people",
    }
    assert _acronym_bridge_full(row_a, row_b)


def test_acronym_bridge_ficci_vs_split_name():
    row_a = {"name_norm": "ficci"}
    row_b = {
        "name_norm": "the federation of indian chambers",
        "name2_norm": "of commerce industry",
    }
    assert _acronym_bridge_full(row_a, row_b)


def test_acronym_bridge_rmhc_vs_ronald():
    row_a = {"name_norm": "rmhc of ohio valley"}
    row_b = {
        "name_norm": "ronald mcdonald house charities of",
        "name2_norm": "ohio valley",
    }
    assert _acronym_bridge_full(row_a, row_b)


def test_acronym_bridge_no_false_positive_short():
    # "abc" does not reliably identify any full name here
    row_a = {"name_norm": "abc"}
    row_b = {"name_norm": "alpha beta company services"}
    # "abc" vs "alpha beta company" = a-b-c = "abc" but "services" is also a token
    # generate_acronym("alpha beta company services") with len>=2 tokens = a-b-c-s = "abcs" ≠ "abc"
    assert not _acronym_bridge_full(row_a, row_b)


# ---------------------------------------------------------------------------
# Unit: compact name match (spacing normalization)
# ---------------------------------------------------------------------------

def test_compact_match_merjan():
    row_a = {"name_norm": "merjan"}
    row_b = {"name_norm": "mer jan"}
    assert _compact_names_match(row_a, row_b)


def test_compact_match_no_false_positive_long():
    # Long names should not trigger compact match (> 12 chars compact)
    row_a = {"name_norm": "some very long company name"}
    row_b = {"name_norm": "somevery long companyname"}
    assert not _compact_names_match(row_a, row_b)


def test_compact_match_identical_names_skipped():
    # Same string → skip (already handled by direct equality)
    row_a = {"name_norm": "acme"}
    row_b = {"name_norm": "acme"}
    assert not _compact_names_match(row_a, row_b)


# ---------------------------------------------------------------------------
# Unit: secondary name match
# ---------------------------------------------------------------------------

def test_secondary_match_reworld_covanta():
    row_reworld = {"name_norm": "reworld", "name2_norm": "fka covanta"}
    row_covanta = {"name_norm": "covanta"}
    assert secondary_name_match(row_reworld, row_covanta)


def test_secondary_match_seminole_tribe():
    row_hrp = {"name_norm": "human resource programs of the", "name2_norm": "seminole tribe of florida"}
    row_seminole = {"name_norm": "seminole tribe of florida"}
    assert secondary_name_match(row_hrp, row_seminole)


# ---------------------------------------------------------------------------
# Integration: evaluate_pair — PASS 1 low-similarity bypass
# ---------------------------------------------------------------------------

def _make_config():
    cfg = ClusteringConfig()
    cfg._tax_block_stats = {}
    return cfg


def test_naacp_tax_exact_clusters():
    """NAACP (acronym) and full name share tax → must not go to review-only."""
    row_a = {
        "row_id": 0, "name_norm": "naacp", "tax_norm": "131084135",
        "addr_norm": "", "city_norm": "baltimore", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "naacp",
        "supplier_identity_core": "naacp",
    }
    row_b = {
        "row_id": 1,
        "name_norm": "national association for the",
        "name2_norm": "advancement of colored people",
        "tax_norm": "131084135",
        "addr_norm": "44 wall street new york", "city_norm": "new york", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "national",
        "supplier_identity_core": "national association",
    }
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}"
    assert result.pass_type not in {"tax_exact_low_similarity_review"}, (
        f"Should not go to low_sim_review; got {result.pass_type}"
    )
    assert result.match_pct >= 90.0


def test_merjan_tax_exact_clusters():
    """MERJAN and MER JAN share tax → compact name match should bypass low_sim_review."""
    row_a = {
        "row_id": 0, "name_norm": "merjan", "tax_norm": "352137156",
        "addr_norm": "3201 wabash avenue terre haute",
        "city_norm": "terre haute", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": False,
        "has_operational_status_hint": False, "root_brand": "merjan",
        "supplier_identity_core": "merjan",
    }
    row_b = {
        "row_id": 1, "name_norm": "mer jan", "tax_norm": "352137156",
        "addr_norm": "880 east 1375 south clinton",
        "city_norm": "clinton", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": False,
        "has_operational_status_hint": False, "root_brand": "mer",
        "supplier_identity_core": "mer jan",
    }
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}"
    assert result.pass_type != "tax_exact_low_similarity_review"
    assert result.match_pct >= 90.0


def test_reworld_covanta_tax_exact_via_secondary():
    """REWORLD (with FKA COVANTA in name2) must cluster with COVANTA via tax+secondary."""
    row_a = {
        "row_id": 0, "name_norm": "reworld", "name2_norm": "fka covanta",
        "tax_norm": "956021257",
        "addr_norm": "445 south morristown nj",
        "city_norm": "morristown", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "reworld",
        "supplier_identity_core": "reworld",
    }
    row_b = {
        "row_id": 1, "name_norm": "covanta", "tax_norm": "956021257",
        "addr_norm": "445 south morristown nj",
        "city_norm": "morristown", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "covanta",
        "supplier_identity_core": "covanta",
    }
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match
    assert result.pass_type not in {"tax_exact_low_similarity_review"}
    assert result.match_pct >= 90.0


def test_seminole_tribe_human_resource_programs():
    """HUMAN RESOURCE PROGRAMS OF THE SEMINOLE TRIBE must cluster with SEMINOLE TRIBE."""
    row_hrp = {
        "row_id": 0,
        "name_norm": "human resource programs of the",
        "name2_norm": "seminole tribe of florida",
        "tax_norm": "591415030",
        "addr_norm": "6300 stirling road hollywood fl",
        "city_norm": "hollywood", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "human",
        "supplier_identity_core": "human resource programs",
    }
    row_semi = {
        "row_id": 1, "name_norm": "seminole tribe of florida",
        "tax_norm": "591415030",
        "addr_norm": "6300 stirling road hollywood fl",
        "city_norm": "hollywood", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "seminole",
        "supplier_identity_core": "seminole tribe",
    }
    result = evaluate_pair(row_hrp, row_semi, {}, _make_config())
    assert result.is_match
    assert result.pass_type not in {"tax_exact_low_similarity_review"}
    assert result.match_pct >= 90.0


# ---------------------------------------------------------------------------
# Integration: PASS 3 same_legal_owner_confirmed
# ---------------------------------------------------------------------------

def test_same_legal_owner_non_generic():
    """Exact non-generic name without tax/address support → same_legal_owner_confirmed."""
    row_a = {
        "row_id": 0, "name_norm": "lynk restaurant",
        "tax_norm": "825905813",
        "addr_norm": "317 chemin lucerne mont royal qc",
        "city_norm": "mont royal", "country_norm": "CA",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "lynk",
        "supplier_identity_core": "lynk",
    }
    row_b = {
        "row_id": 1, "name_norm": "lynk restaurant",
        "tax_norm": "711290296",
        "addr_norm": "1 rue john kennedy saint jerome qc",
        "city_norm": "saint jerome", "country_norm": "CA",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "lynk",
        "supplier_identity_core": "lynk",
    }
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}, is_match={result.is_match}"
    assert result.pass_type == "same_legal_owner_confirmed", f"Expected same_legal_owner_confirmed; got {result.pass_type}"
    assert result.match_pct == 85.0


def test_same_legal_owner_generic_still_review():
    """Exact generic name (e.g. 'services') stays as name_exact_review."""
    row_a = {
        "row_id": 0, "name_norm": "services",
        "tax_norm": "111111111",
        "addr_norm": "", "city_norm": "new york", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": False,
        "has_operational_status_hint": False, "root_brand": "",
        "supplier_identity_core": "",
    }
    row_b = {**row_a, "row_id": 1, "tax_norm": "222222222"}
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    # Generic name must not produce same_legal_owner_confirmed
    assert result.pass_type != "same_legal_owner_confirmed"


# ---------------------------------------------------------------------------
# Integration: franchise guardrail bypass for same legal owner
# ---------------------------------------------------------------------------

def test_franchise_guardrail_allows_same_legal_owner():
    """same_legal_owner_confirmed should bypass the franchise/hospitality block."""
    base = {
        "row_id": 0, "name_norm": "lynk restaurant",
        "addr_norm": "", "city_norm": "montreal", "country_norm": "CA",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "lynk",
        "supplier_identity_core": "lynk",
    }
    result_in = MatchResult(
        is_match=True, match_pct=85.0, pass_type="same_legal_owner_confirmed",
        evidence={"name_sim": 1.0, "addr_sim": 0.0},
    )
    guarded = apply_guardrails({**base, "row_id": 0}, {**base, "row_id": 1}, result_in)
    assert guarded.is_match, "Franchise guardrail must not block same_legal_owner_confirmed"
    assert guarded.match_pct >= 85.0


def test_franchise_guardrail_blocks_brand_only():
    """Brand-only hospitality match without address/domain/tax remains blocked."""
    row_a = {
        "row_id": 0, "name_norm": "hilton downtown",
        "addr_norm": "100 main st springfield",
        "city_norm": "springfield", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "hilton",
        "supplier_identity_core": "hilton",
    }
    row_b = {
        "row_id": 1, "name_norm": "hilton airport",
        "addr_norm": "200 airport road chicago",
        "city_norm": "chicago", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": True, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "hilton",
        "supplier_identity_core": "hilton",
    }
    result_in = MatchResult(
        is_match=True, match_pct=85.0, pass_type="name_fuzzy_supported",
        evidence={"name_sim": 0.75, "addr_sim": 0.0},
    )
    guarded = apply_guardrails(row_a, row_b, result_in)
    assert not guarded.is_match, "Brand-only hospitality match should be rejected"


# ---------------------------------------------------------------------------
# Integration: guardrail ordering — tax_exact before operational_status
# ---------------------------------------------------------------------------

def test_tax_exact_not_blocked_by_operational_hint():
    """tax_exact pass must return even when one row has has_operational_status_hint."""
    row_a = {
        "row_id": 0, "name_norm": "acme corp",
        "addr_norm": "", "city_norm": "", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": True, "has_company_keyword": True,
        "has_operational_status_hint": True,  # would downgrade non-tax passes
        "root_brand": "acme", "supplier_identity_core": "acme",
    }
    row_b = {**row_a, "row_id": 1, "has_operational_status_hint": False}
    result_in = MatchResult(
        is_match=True, match_pct=98.0, pass_type="tax_exact",
        evidence={"name_sim": 0.9, "addr_sim": 0.0},
    )
    guarded = apply_guardrails(row_a, row_b, result_in)
    assert guarded.is_match
    assert guarded.pass_type == "tax_exact"
    assert guarded.match_pct >= 98.0


# ---------------------------------------------------------------------------
# Integration: address_distinctive_shared
# ---------------------------------------------------------------------------

def test_address_distinctive_shared_seminole():
    """SEMINOLE HARD ROCK and SEMINOLE HR at same address → address_distinctive_shared.

    Uses realistic supplier_identity_core values (different full phrases) so
    distinctive_supplier_identity does NOT fire and the pair reaches PASS 6.
    """
    row_a = {
        "row_id": 0, "name_norm": "seminole hard rock entertainment",
        "tax_norm": "208347464",
        "addr_norm": "5701 davie fl stirling",
        "city_norm": "davie", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "seminole",
        # Full core phrase — different from row_b so distinctive_supplier_identity won't fire
        "supplier_identity_core": "seminole hard rock entertainment",
    }
    row_b = {
        "row_id": 1, "name_norm": "seminole hr",
        "tax_norm": "660690602",
        "addr_norm": "5701 davie fl stirling",
        "city_norm": "davie", "country_norm": "US",
        "postal_norm": "", "domain": "", "is_generic_domain": False,
        "is_hospitality": False, "is_likely_individual": False,
        "has_legal_suffix": False, "has_company_keyword": True,
        "has_operational_status_hint": False, "root_brand": "seminole",
        "supplier_identity_core": "seminole hr",
    }
    result = evaluate_pair(row_a, row_b, {}, _make_config())
    assert result.is_match, f"Expected match; got pass_type={result.pass_type}, is_match={result.is_match}"
    assert result.pass_type in {"address_distinctive_shared", "address_name_related"}, (
        f"Expected address_distinctive_shared or address_name_related; got {result.pass_type}"
    )
    assert result.match_pct >= 78.0

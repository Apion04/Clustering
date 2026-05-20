"""RC6 — generic/common root tokens must not bridge unrelated suppliers at 85."""
import pytest
from src.config import ClusteringConfig, GENERIC_ROOT_TOKENS, SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS
from src.matching import evaluate_pair

cfg = ClusteringConfig()

def _row(name_norm, country="us", domain="", row_id=0):
    return {
        "name_norm": name_norm, "addr_norm": "", "city_norm": "", "country_norm": country,
        "domain": domain, "is_generic_domain": False, "name_location_core": name_norm,
        "row_id": row_id, "postal_norm": "",
    }

def test_active_food_vs_active_technology_not_85():
    """'active' is generic — should not bridge at 85."""
    a = _row("active food")
    b = _row("active technology", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert not r.is_match or r.match_pct <= 70.0, f"Expected no match or <=70, got {r.match_pct}"

def test_acta_conseils_vs_acta_laboratories_not_85():
    """'acta' is a risky short root — should not bridge distinct entities at 85."""
    a = _row("acta conseils", country="fr")
    b = _row("acta", country="us", row_id=1)  # acta laboratories in US
    r = evaluate_pair(a, b, {}, cfg)
    assert not r.is_match or r.match_pct <= 70.0, f"Expected no match or <=70, got {r.match_pct}"

def test_adams_reese_vs_adams_design_not_85():
    """'adams' is a common surname — should not bridge at 85."""
    a = _row("adams reese")
    b = _row("adams design", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert not r.is_match or r.match_pct <= 70.0, f"Expected <=70, got {r.match_pct}"

def test_advance_ag_vs_advanceanalytics_not_85():
    """'advance' is too generic to bridge."""
    a = _row("advance", country="de")
    b = _row("advanceanalytics", country="us", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert not r.is_match or r.match_pct <= 70.0

def test_active_in_generic_root_tokens():
    assert "active" in GENERIC_ROOT_TOKENS

def test_advance_in_generic_root_tokens():
    assert "advance" in GENERIC_ROOT_TOKENS or "advance" in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS

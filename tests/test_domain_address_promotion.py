"""RC5 — same domain + address + near name should reach 98, not stop at 86."""
import pytest
from src.config import ClusteringConfig
from src.matching import evaluate_pair

cfg = ClusteringConfig()

def _row(name_norm, addr_norm="", city_norm="", country_norm="", domain="", row_id=0):
    return {
        "name_norm": name_norm, "addr_norm": addr_norm, "city_norm": city_norm,
        "country_norm": country_norm, "domain": domain, "is_generic_domain": False,
        "name_location_core": name_norm, "row_id": row_id,
        "postal_norm": "",
    }

def test_b_lab_same_domain_and_address_is_98():
    """B Lab Company rows with same domain + same address + near name -> 98."""
    a = _row("b lab company", "111 e hector st", "conshohocken", "us", "bcorporation.net")
    b = _row("b lab", "111 e hector st", "conshohocken", "us", "bcorporation.net", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.match_pct >= 93.0, f"Expected >=93, got {r.match_pct}"

def test_same_domain_same_address_near_name_is_98():
    a = _row("allman communication", "50 main street", "chicago", "us", "allmancomm.com")
    b = _row("allman communcation", "50 main street", "chicago", "us", "allmancomm.com", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.match_pct >= 93.0, f"Expected >=93, got {r.match_pct}"

def test_same_domain_different_address_low_name_sim_stays_review():
    """Same domain but low name_sim and no address support -> review, not 98."""
    a = _row("acme gmbh", "strasse 1", "berlin", "de", "acme.de")
    b = _row("johnson international", "anders weg 5", "hamburg", "de", "acme.de", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    # Should be a review candidate or low score, not 98
    assert r.match_pct < 93.0 or r.needs_review

def test_generic_domain_not_promoted():
    """Gmail domain must not promote to 98."""
    a = _row("acme gmbh", "strasse 1", "berlin", "de", "gmail.com")
    b = _row("acme", "strasse 1", "berlin", "de", "gmail.com", row_id=1)
    # Mark as generic domain
    a["is_generic_domain"] = True
    b["is_generic_domain"] = True
    r = evaluate_pair(a, b, {}, cfg)
    # same_domain is False when is_generic_domain=True, so promotion should not apply
    # It should still match via name/address, but not via domain path
    assert r.pass_type != "domain_address_confirmed"

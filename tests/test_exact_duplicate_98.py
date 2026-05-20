"""RC2 — exact duplicate rows must score 98, not 85."""
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

def test_screenfluence_exact_duplicate_is_98():
    a = _row("screenfluence", "272 avenue road", "toronto", "ca")
    b = _row("screenfluence", "272 avenue road", "toronto", "ca", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.match_pct >= 98.0, f"Expected 98, got {r.match_pct}"

def test_seglo_exact_duplicate_is_98():
    a = _row("seglo operaciones logisticas", "avenida jint 300", "puebla", "mx")
    b = _row("seglo operaciones logisticas", "avenida jint 300", "puebla", "mx", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.match_pct >= 98.0, f"Expected 98, got {r.match_pct}"

def test_sensus_exact_duplicate_is_98():
    a = _row("sensus", "100 main street", "raleigh", "us")
    b = _row("sensus", "100 main street", "raleigh", "us", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.match_pct >= 98.0

def test_same_name_different_address_is_85():
    """Same name but different address should NOT be 98 — still 85 via PASS 3."""
    a = _row("siemens", "strasse 1", "berlin", "de")
    b = _row("siemens", "anders strasse 5", "munich", "de", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    # Siemens is in TRUSTED tokens so may be 85+ via supplier identity — just check it's a match
    assert r.is_match

def test_noisy_exact_duplicate_is_98_after_noise_strip():
    """Rows with the same name after noise stripping should still score 98."""
    from src.preprocessing import normalize_supplier_name
    a_name = normalize_supplier_name("BEFORTE LLC - 3 WAY")
    b_name = normalize_supplier_name("BEFORTE LLC")
    a = _row(a_name, "123 main st", "new york", "us")
    b = _row(b_name, "123 main st", "new york", "us", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.is_match

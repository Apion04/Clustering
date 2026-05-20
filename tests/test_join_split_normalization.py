"""RC3 — join/split and punctuation normalization."""
import pytest
from src.preprocessing import normalize_supplier_name
from src.config import ClusteringConfig
from src.matching import evaluate_pair

cfg = ClusteringConfig()

def _row(name_norm, addr_norm="", city_norm="springfield", country_norm="us", domain="", row_id=0):
    return {
        "name_norm": name_norm, "addr_norm": addr_norm, "city_norm": city_norm,
        "country_norm": country_norm, "domain": domain, "is_generic_domain": False,
        "name_location_core": name_norm, "row_id": row_id, "postal_norm": "",
    }

def test_tbd_pizza_normalizes_same():
    """T.B.D. and TBD must produce same or compact-matching name_norm."""
    a = normalize_supplier_name("T.B.D. PIZZA INC")
    b = normalize_supplier_name("TBD PIZZA INC")
    # Either exact match or compact match
    assert a == b or a.replace(" ", "") == b.replace(" ", ""), f"a={a!r}, b={b!r}"

def test_jsjv_normalizes_compact():
    """J.S.J.V. and J S J V must compact-match."""
    a = normalize_supplier_name("J.S.J.V. HOLDINGS LTD")
    b = normalize_supplier_name("J S J V HOLDINGS LTD")
    assert a.replace(" ", "") == b.replace(" ", ""), f"a={a!r}, b={b!r}"

def test_3carp_scores_98_same_address():
    from src.preprocessing import normalize_supplier_name
    a_nn = normalize_supplier_name("3 CARP ENTERPRISES LLC")
    b_nn = normalize_supplier_name("3CARP ENTERPRISES LLC")
    a = _row(a_nn, addr_norm="1604 woodcrest ave", city_norm="corsicana")
    b = _row(b_nn, addr_norm="1604 woodcrest ave", city_norm="corsicana", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.is_match, f"3CARP/3 CARP should match; got pass_type={r.pass_type}"
    assert r.match_pct >= 85.0

def test_tbd_pizza_scores_high_same_address():
    a_nn = normalize_supplier_name("T.B.D. PIZZA INC")
    b_nn = normalize_supplier_name("TBD PIZZA INC")
    a = _row(a_nn, addr_norm="9 fairway dr", city_norm="andover")
    b = _row(b_nn, addr_norm="9 fairway dr", city_norm="andover", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.is_match, f"TBD Pizza should match; score={r.match_pct}"
    assert r.match_pct >= 85.0

def test_pie_the_vs_the_pie_same_address():
    a_nn = normalize_supplier_name("PIE THE")
    b_nn = normalize_supplier_name("THE PIE")
    a = _row(a_nn, addr_norm="216 main st", city_norm="port jefferson")
    b = _row(b_nn, addr_norm="216 main st", city_norm="port jefferson", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.is_match, f"PIE THE vs THE PIE same address should match"

def test_healthcare_normalizes_same():
    a = normalize_supplier_name("5P Health Care")
    b = normalize_supplier_name("5P Healthcare")
    assert a == b or a.replace(" ", "") == b.replace(" ", ""), f"a={a!r}, b={b!r}"

def test_rhl_subways_compact_match():
    """R H L and RHL should compact-match."""
    a_nn = normalize_supplier_name("R H L SUBWAYS")
    b_nn = normalize_supplier_name("RHL SUBWAYS")
    # RHL is an operator — compact match should fire
    a = _row(a_nn, city_norm="springfield")
    b = _row(b_nn, city_norm="springfield", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.is_match, f"R H L / RHL should match; score={r.match_pct}"

def test_merjan_compact_match():
    a_nn = normalize_supplier_name("MER JAN LLC")
    b_nn = normalize_supplier_name("MERJAN LLC")
    a = _row(a_nn, city_norm="clinton", country_norm="us")
    b = _row(b_nn, city_norm="clinton", country_norm="us", row_id=1)
    r = evaluate_pair(a, b, {}, cfg)
    assert r.is_match, f"MER JAN / MERJAN should match; score={r.match_pct}"

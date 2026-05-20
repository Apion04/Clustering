"""RC4 — PT prefix, OOO suffix, CEDEX, GmbH & Co KG normalization."""
import pytest
from src.preprocessing import normalize_supplier_name

def test_pt_prefix_stripped():
    """Indonesian PT prefix should be stripped."""
    a = normalize_supplier_name("PT Arga Bangun Bangsa")
    b = normalize_supplier_name("Arga Bangun Bangsa")
    assert a == b, f"PT prefix should be stripped; a={a!r}, b={b!r}"

def test_pt_suffix_stripped():
    """PT as suffix (Krisbow PT) should be stripped."""
    a = normalize_supplier_name("Krisbow PT")
    b = normalize_supplier_name("Krisbow")
    assert a == b, f"PT suffix should be stripped; a={a!r}, b={b!r}"

def test_pt_not_stripped_from_short_name():
    """PT alone should not be stripped to empty."""
    result = normalize_supplier_name("PT")
    assert result != ""

def test_ooo_suffix_stripped():
    """Russian OOO suffix should be stripped like a legal form."""
    a = normalize_supplier_name("OOO Ankor")
    b = normalize_supplier_name("Ankor")
    assert a == b, f"OOO suffix should be stripped; a={a!r}, b={b!r}"

def test_gmbh_co_kg_normalized():
    """GmbH & Co KG compound should normalize cleanly."""
    a = normalize_supplier_name("Mustermann GmbH & Co KG")
    b = normalize_supplier_name("Mustermann GmbH")
    # Both should normalize to same root (company name without legal form)
    assert a == b, f"GmbH & Co KG should normalize same as GmbH; a={a!r}, b={b!r}"

def test_polska_not_bridge_token():
    """'Polska' should not bridge two different Polish companies."""
    from src.config import GENERIC_ROOT_TOKENS, LOCATION_ROOT_TOKENS
    assert "polska" in GENERIC_ROOT_TOKENS or "polska" in LOCATION_ROOT_TOKENS

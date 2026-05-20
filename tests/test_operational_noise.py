"""RC1 — operational noise stripping: noisy vendor names must normalize to clean cores."""
import pytest
from src.preprocessing import normalize_supplier_name

def test_sap_vendor_number_stripped():
    assert normalize_supplier_name("Absolute Handling Systems SAP vendor number 1181451") == normalize_supplier_name("Absolute Handling Systems")

def test_payment_term_stripped():
    result = normalize_supplier_name("CESAR CASTILLO LLC FOR MC IDST03 AND PAYMENT TERM 0033")
    expected = normalize_supplier_name("CESAR CASTILLO LLC")
    assert result == expected

def test_web_edi_stripped():
    result = normalize_supplier_name("ACCELERA ALTERNATE TO WEB EDI VENDOR 12345")
    expected = normalize_supplier_name("ACCELERA")
    assert result == expected

def test_use_vendor_stripped():
    result = normalize_supplier_name("Donnelly Gillen Law USE VENDOR X12345")
    expected = normalize_supplier_name("Donnelly Gillen Law")
    assert result == expected

def test_blocked_prefix_stripped():
    result = normalize_supplier_name("BLOCKED Plasticard GmbH")
    expected = normalize_supplier_name("Plasticard GmbH")
    assert result == expected

def test_gesperrt_prefix_stripped():
    result = normalize_supplier_name("GESPERRT Riseway Ltd")
    expected = normalize_supplier_name("Riseway Ltd")
    assert result == expected

def test_three_way_suffix_stripped():
    result = normalize_supplier_name("BEFORTE LLC - 3 WAY")
    expected = normalize_supplier_name("BEFORTE LLC")
    assert result == expected

def test_masked_x_prefix_stripped():
    result = normalize_supplier_name("XXXXXXXXXXConcat AG")
    expected = normalize_supplier_name("Concat AG")
    assert result == expected

def test_noise_strip_length_guard():
    # "USE VENDOR" alone must not strip to empty
    result = normalize_supplier_name("USE VENDOR")
    assert result != ""  # should not be empty — guard prevents it

def test_accugenomics_not_same_as_accelera_after_noise_strip():
    """After stripping WEB EDI noise, ACCELERA and ACCUGENOMICS must have different name_norms."""
    a = normalize_supplier_name("ACCELERA ALTERNATE TO WEB EDI VENDOR")
    b = normalize_supplier_name("ACCUGENOMICS ALTERNATE TO WEB EDI VENDOR")
    assert a != b

def test_clean_name_unchanged():
    """Noise stripping must not alter clean names."""
    result = normalize_supplier_name("Siemens AG")
    assert "siemens" in result

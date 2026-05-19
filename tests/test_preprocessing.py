"""Tests for preprocessing module."""

import pytest
import polars as pl
from src.preprocessing import (
    normalize_text, normalize_supplier_name, normalize_address,
    normalize_tax_id, normalize_tax_id_loose, extract_domain, is_generic_domain,
    normalize_city, normalize_country, remove_legal_suffixes,
    remove_status_words, token_sort_fingerprint, phonetic_key, has_legal_suffix_hint,
    preprocess_dataframe,
)
from src.legal_keywords import find_legal_keyword_matches, load_legal_keywords


class TestNormalizeText:
    def test_basic_lowercase(self):
        assert normalize_text("ABC Corp") == "abc corp"

    def test_accent_removal(self):
        assert normalize_text("CZEPczyński") == "czepczynski"

    def test_special_chars(self):
        assert normalize_text("A&B Corp") == "a b corp"
        assert normalize_text("427/QEW Kia") == "427 qew kia"

    def test_multiple_spaces(self):
        assert normalize_text("ABC    Corp") == "abc corp"


class TestNormalizeSupplierName:
    def test_case_variation(self):
        assert normalize_supplier_name("ABS Safety GmbH") == "abs safety"
        assert normalize_supplier_name("ABS SAFETY GMBH") == "abs safety"

    def test_punctuation(self):
        assert normalize_supplier_name("427/QEW KIA") == "427 qew kia"
        assert normalize_supplier_name("H. Y. Louie") == "h y louie"

    def test_ampersand(self):
        assert normalize_supplier_name("A & B ONE GmbH") == "a b one"
        assert normalize_supplier_name("A&B ONE GmbH") == "a b one"

    def test_legal_suffixes(self):
        assert normalize_supplier_name("ABC Inc") == "abc"
        assert normalize_supplier_name("ABC Ltd") == "abc"
        assert normalize_supplier_name("ABC LLC") == "abc"
        assert normalize_supplier_name("ABC GmbH") == "abc"
        assert normalize_supplier_name("ABC S.A.") == "abc"

    def test_loaded_global_legal_suffixes(self, tmp_path):
        legal_file = tmp_path / "legal.csv"
        legal_file.write_text(
            "legal keywords\n"
            "Aktiengesellschaft\n"
            "Aktiebolag (Limited Company)\n"
            "Aksjeselskap (Private Limited)\n"
            "akciová společnost (Joint-Stock Company)\n"
            "akcionarsko društvo (JSC)\n"
            "Pvt Ltd\n"
            "A/S\n"
            "AB\n"
            "SA\n",
            encoding="utf-8",
        )
        legal = load_legal_keywords(str(legal_file))
        assert normalize_supplier_name("ABC Aktiengesellschaft", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC Aktiebolag", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC Aksjeselskap", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC akciová společnost", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC akcionarsko društvo", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC Pvt Ltd", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC A/S", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC AB", legal_keywords=legal) == "abc"
        assert normalize_supplier_name("ABC S.A.", legal_keywords=legal) == "abc"

    def test_short_legal_forms_are_boundary_only(self, tmp_path):
        legal_file = tmp_path / "legal.csv"
        legal_file.write_text("legal keywords\nAG\nAB\nSA\nCO\n", encoding="utf-8")
        legal = load_legal_keywords(str(legal_file))
        assert find_legal_keyword_matches("Agilent", legal) == []
        assert find_legal_keyword_matches("Abbott", legal) == []
        assert find_legal_keyword_matches("Sartorius", legal) == []
        assert find_legal_keyword_matches("Cognis", legal) == []
        assert normalize_supplier_name("Agilent AG", legal_keywords=legal) == "agilent"
        assert normalize_supplier_name("Abbott AB", legal_keywords=legal) == "abbott"
        assert normalize_supplier_name("Sartorius SA", legal_keywords=legal) == "sartorius"
        assert normalize_supplier_name("Cognis Co", legal_keywords=legal) == "cognis"

    def test_loaded_legal_keywords_drive_entity_hint_not_person_detection(self, tmp_path):
        legal_file = tmp_path / "legal.csv"
        legal_file.write_text("legal keywords\nGmbH\nLtd\n", encoding="utf-8")
        legal = load_legal_keywords(str(legal_file))
        df = pl.DataFrame({
            "Name": [
                "Robert Bosch GmbH",
                "Alfred L. Wolff GmbH",
                "Dr. Reddy's Laboratories Ltd",
                "Wilhelm Schmidt",
            ]
        })
        out = preprocess_dataframe(df, {"supplier_name": "Name"}, legal_keywords=legal)
        rows = {r["orig_supplier_name"]: r for r in out.iter_rows(named=True)}
        assert rows["Robert Bosch GmbH"]["has_legal_suffix"] is True
        assert rows["Robert Bosch GmbH"]["is_likely_individual"] is False
        assert rows["Alfred L. Wolff GmbH"]["is_likely_individual"] is False
        assert rows["Dr. Reddy's Laboratories Ltd"]["is_likely_individual"] is False
        assert rows["Wilhelm Schmidt"]["has_legal_suffix"] is False

    def test_status_words(self):
        assert normalize_supplier_name("BELL CANADA (Deactivated)") == "bell canada"
        assert normalize_supplier_name("1-800-GOT-JUNK? (DEACTIVATED)") == "1 800 got junk"

    def test_store_numbers(self):
        assert normalize_supplier_name("Shoppers Drug Mart #313") == "shoppers drug mart"
        assert normalize_supplier_name("Subway 25542") == "subway"


class TestNormalizeAddress:
    def test_abbreviations(self):
        assert "street" in normalize_address("123 Main St")
        assert "avenue" in normalize_address("456 First Ave")
        assert "road" in normalize_address("789 Telge Rd")

    def test_german_street(self):
        assert "street" in normalize_address("Ludwig-Erhard-Straße 10")


class TestNormalizeTaxID:
    def test_german_vat(self):
        assert normalize_tax_id("DE233002380", "DE") == "de233002380"
        assert normalize_tax_id_loose("DE233002380", "DE") == "233002380"

    def test_us_ein(self):
        assert normalize_tax_id("95-2499371", "US") == "952499371"

    def test_canada_bn(self):
        assert normalize_tax_id("123456789RC0001", "CA") == "123456789rc0001"


class TestExtractDomain:
    def test_email(self):
        assert extract_domain("contact@abs-safety.de") == "abs-safety.de"

    def test_website(self):
        assert extract_domain("https://www.greenpharma.fr") == "greenpharma.fr"

    def test_generic(self):
        assert is_generic_domain("gmail.com") is True
        assert is_generic_domain("abs-safety.de") is False


class TestNormalizeCountry:
    def test_variations(self):
        assert normalize_country("USA") == "US"
        assert normalize_country("United States") == "US"
        assert normalize_country("Deutschland") == "DE"
        assert normalize_country("FR") == "FR"


class TestTokenSortFingerprint:
    def test_word_order(self):
        fp1 = token_sort_fingerprint("abc czepczynski")
        fp2 = token_sort_fingerprint("czepczynski abc")
        assert fp1 == fp2

class TestExpandedGenericDomainsAndTax:
    def test_expanded_generic_domains(self):
        assert is_generic_domain("mail.gmail.com") is True
        assert is_generic_domain("yahoo.co.in") is True
        assert is_generic_domain("mail.ru") is True
        assert is_generic_domain("yopmail.com") is True
        assert is_generic_domain("supplier-company.example") is False

    def test_worldwide_tax_placeholders_ignored(self):
        assert normalize_tax_id("NOT PROVIDED", "GB") == ""
        assert normalize_tax_id("000000000", "DE") == ""
        assert normalize_tax_id("SUBJECT TO TAX", "US") == ""
        assert normalize_tax_id("FR33775642853", "FR") == "fr33775642853"
        assert normalize_tax_id_loose("FR33775642853", "FR") == "33775642853"

    def test_domain_falls_back_when_mapped_domain_blank(self):
        df = pl.DataFrame({
            "Name": ["ABC Ltd"],
            "Domain": [""],
            "Website": ["https://www.abc.example"],
            "Email": ["contact@fallback.example"],
        })
        out = preprocess_dataframe(df, {
            "supplier_name": "Name",
            "domain": "Domain",
            "website": "Website",
            "email": "Email",
        })
        assert out["domain"].to_list() == ["abc.example"]

    def test_json_family_name_is_extracted_as_secondary_name(self):
        df = pl.DataFrame({
            "Name": ["Spiess"],
            "CustomExtensionFormJSON": ['{"familyName":"DR SPIESS CHEMISCHE FABRIK GMBH","taxNumber":"","vatNumber":""}'],
        })
        out = preprocess_dataframe(df, {
            "supplier_name": "Name",
            "metadata_json_columns": ["CustomExtensionFormJSON"],
        })
        assert out["json_secondary_names_norm"].to_list() == ["dr spiess chemische fabrik"]
        assert out["orig_secondary_names"].to_list() == ["DR SPIESS CHEMISCHE FABRIK GMBH"]

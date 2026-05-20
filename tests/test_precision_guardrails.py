"""Regression tests from Rohit's real review examples."""
import json

import polars as pl

from src.main import cluster_suppliers
from src.config import ClusteringConfig
from src.matching import evaluate_pair
from src.preprocessing import normalize_supplier_name, normalize_address, normalize_city, normalize_country, normalize_tax_id, normalize_tax_id_loose, extract_root_brand
from src.person import clean_person_name, is_likely_individual_name
from src.brand_families import load_known_brand_families


def row(name, address="", city="", country="DE", tax="", domain="", legal=False, company=False, hospitality=False):
    nn = normalize_supplier_name(name)
    person_name = clean_person_name(nn)
    has_legal = legal or any(x in name.lower() for x in ["gmbh", " inc", " ltd", " ag", " sas", " s.a", " llc", " corp"])
    has_company = company
    is_hospitality = hospitality or any(x in nn.split() for x in ["hotel", "parkhotel", "hyatt", "restaurant"])
    return {
        "name_norm": nn,
        "person_name_norm": person_name,
        "addr_norm": normalize_address(address),
        "city_norm": normalize_city(city),
        "country_norm": normalize_country(country),
        "postal_norm": "",
        "tax_norm": normalize_tax_id(tax, country),
        "tax_loose_norm": normalize_tax_id_loose(tax, country),
        "domain": domain,
        "is_generic_domain": False,
        "root_brand": extract_root_brand(nn),
        "has_legal_suffix": has_legal,
        "has_company_keyword": has_company,
        "is_hospitality": is_hospitality,
        "is_likely_individual": is_likely_individual_name(person_name, has_legal_suffix=has_legal, has_company_keyword=has_company, is_hospitality=is_hospitality),
    }


def clustered_names(records):
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Address": [r[1] for r in records],
        "City": [r[2] for r in records],
        "Country": [r[3] for r in records],
        "Postal Code": [r[4] if len(r) > 4 else "" for r in records],
        "Tax ID": [r[5] if len(r) > 5 else "" for r in records],
        "Email": [r[6] if len(r) > 6 else "" for r in records],
    })
    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
        "tax_id": "Tax ID",
        "email": "Email",
    }
    return cluster_suppliers(df, mapping)


def clustered_names_with_brand_file(records, brand_file):
    return clustered_names_with_files(records, brand_file=brand_file)


def clustered_names_with_files(records, brand_file=None, legal_file=None):
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Address": [r[1] if len(r) > 1 else "" for r in records],
        "City": [r[2] if len(r) > 2 else "" for r in records],
        "Country": [r[3] if len(r) > 3 else "" for r in records],
        "Postal Code": [r[4] if len(r) > 4 else "" for r in records],
        "Email": [r[5] if len(r) > 5 else "" for r in records],
    })
    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
        "email": "Email",
    }
    config = ClusteringConfig()
    if brand_file:
        config.known_brand_families_file = str(brand_file)
    if legal_file:
        config.legal_keywords_file = str(legal_file)
    return cluster_suppliers(df, mapping, config)


def clustered_names_with_metadata(records):
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Address": [r[1] for r in records],
        "City": [r[2] for r in records],
        "Country": [r[3] for r in records],
        "Postal Code": [r[4] if len(r) > 4 else "" for r in records],
        "Meta": [json.dumps(r[5]) if len(r) > 5 else "{}" for r in records],
    })
    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
        "metadata_json_columns": ["Meta"],
        "json_secondary_name_keys": ["familyName", "parentName", "groupName", "tradeName", "legalName"],
    }
    return cluster_suppliers(df, mapping)


def test_person_same_name_different_address_rejected():
    a = row("Peter Kraft", "ALTE BAHNHOFSTR. 10", "KARLSTADT")
    b = row("Peter Kraft", "Hauptstr. 87", "Rossdorf")
    assert not evaluate_pair(a, b, {}).is_match


def test_common_first_name_only_rejected():
    a = row("Robert Andre", "A STR 1", "Berlin")
    b = row("Robert Bauer", "B STR 2", "Berlin")
    assert not evaluate_pair(a, b, {}).is_match


def test_reported_aaron_people_do_not_cluster():
    result = clustered_names([
        ("Aaron Lackner", "Fritz-Salm-Str 9A", "Mannheim", "DE", ""),
        ("Aaron Lawson McLean", "Am Klinikum 1", "Jena", "DE", ""),
        ("Aaron Tan", "11 National Cancer Centre, Hospital", "Singapore", "SG", ""),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_reported_alexandre_people_do_not_cluster():
    result = clustered_names([
        ("Alexandre Guiraud", "1 Rue A", "Paris", "FR", ""),
        ("Alexandre Prat", "2 Rue B", "Outremont", "CA", ""),
        ("Alexandre Varnek", "3 Rue C", "Strasbourg", "FR", ""),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_reported_alice_same_city_people_do_not_cluster():
    result = clustered_names([
        ("Alice Antonello", "A Street 1", "Darmstadt", "DE", "64293"),
        ("Alice Hoffmann-Ziegler", "B Street 2", "Darmstadt", "DE", "64293"),
        ("Alice Lichtenberg", "C Street 3", "Darmstadt", "DE", "64293"),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_different_people_same_or_similar_address_rejected():
    bad_pairs = [
        ("Alexander Kehl", "Kurt-Schumacher Strasse 5", "Waldemar Kehl", "Kurt-Schumacher Str. 5"),
        ("Aline Knoll", "Schulstr. 8", "Kevin Knoll", "Schulstr. 8"),
        ("Andrea Daechert", "Korellweg 20", "Juergen Daechert", "Korellweg 20"),
    ]
    for a_name, a_addr, b_name, b_addr in bad_pairs:
        assert not evaluate_pair(row(a_name, a_addr, "Darmstadt"), row(b_name, b_addr, "Darmstadt"), {}).is_match


def test_same_full_individual_name_different_address_rejected():
    assert not evaluate_pair(
        row("Dr. Anita Nair", "1 Lake Road", "Mumbai", "IN"),
        row("Anita Nair", "99 Other Road", "Mumbai", "IN"),
        {},
    ).is_match


def test_professional_title_person_company_same_address_clusters():
    result = clustered_names([
        ("Wilhelm Schmidt", "Breslauer Strasse 14", "Seeheim-Jugenheim", "DE", ""),
        ("Wilhelm Schmidt", "Breslauer Strasse 14", "Seeheim-Jugenheim", "DE", ""),
        ("Dipl.-Ing. Wilhelm Schmidt", "Breslauer Strasse 14 14", "Seeheim-Jugenheim", "DE", ""),
        ("Dipl.-Ing. Wilhelm Schmidt GmbH", "Breslauer Straße 14", "Seeheim-Jugenheim", "DE", ""),
        ("Wilhelm Schmidt Dipl.-Ing.", "Breslauer Str 14", "Seeheim-Jugenheim", "DE", ""),
    ])
    clusters = [c for c in result["main_df"]["Cluster Number"].to_list() if c is not None]
    assert len(set(clusters)) == 1
    assert len(clusters) == 5


def test_professional_title_same_person_different_address_rejected():
    assert not evaluate_pair(
        row("Dipl.-Ing. Wilhelm Schmidt", "Breslauer Strasse 14", "Seeheim-Jugenheim", "DE"),
        row("Wilhelm Schmidt Dipl.-Ing.", "Other Strasse 99", "Seeheim-Jugenheim", "DE"),
        {},
    ).is_match


def test_initial_variation_same_address_allowed_for_manual_review():
    pairs = [
        ("Rahul Mehta", "12 Lake View Road", "R. Mehta", "12 Lake View Rd"),
        ("Rahul Mehta", "12 Lake View Road", "Rahul M.", "12 Lake View Rd"),
        ("Dr. Anita Nair", "12 Lake View Road", "Anita Nair", "12 Lake View Rd"),
    ]
    for a_name, a_addr, b_name, b_addr in pairs:
        res = evaluate_pair(row(a_name, a_addr, "Mumbai", "IN"), row(b_name, b_addr, "Mumbai", "IN"), {})
        assert res.is_match
        assert res.match_pct <= 90


def test_middle_initial_does_not_allow_different_surname_person_cluster():
    res = evaluate_pair(
        row("John Baldoni", "9549 Siracusa Court", "Naples", "US"),
        row("John M. Bandoni", "9549 Siracusa Court", "Naples", "US"),
        {},
    )
    assert not res.is_match


def test_ra_initial_same_address_is_review_not_high_confidence():
    res = evaluate_pair(
        row("RA A. Huber", "Bahnhofstr. 7", "Berlin", "DE"),
        row("Angela Huber", "Bahnhofstrasse 7", "Berlin", "DE"),
        {},
    )
    assert res.is_match
    assert res.needs_review
    assert res.match_pct <= 80


def test_company_with_first_name_hint_not_blocked_when_legal_suffix():
    a = row("Robert Bosch GmbH", "Robert-Bosch-Str. 1", "Stuttgart", legal=True)
    b = row("ROBERT BOSCH GMBH", "Robert Bosch Strasse 1", "Stuttgart", legal=True)
    assert evaluate_pair(a, b, {}).is_match


def test_hotel_different_properties_rejected():
    a = row("Hyatt Regency Mumbai", "Airport Road", "Mumbai", "IN", hospitality=True)
    b = row("Hyatt Regency Pune", "Kalyani Nagar", "Pune", "IN", hospitality=True)
    assert not evaluate_pair(a, b, {}).is_match


def test_hotel_same_property_allowed_with_address():
    a = row("Hyatt Regency Hotel", "57 Block Airport Road", "Nagpur", "IN", hospitality=True)
    b = row("Hotel Hyatt Regency", "57 Block Airport Road", "Nagpur", "IN", hospitality=True)
    assert evaluate_pair(a, b, {}).is_match


def test_invalid_tax_placeholder_removed():
    assert normalize_tax_id("SUBJECT TO TAX", "DE") == ""


def test_loose_tax_requires_support():
    a = row("ABC GmbH", "Main Str 1", "Berlin", "DE", "DE233002380", legal=True)
    b = row("ABC GmbH", "Main Str 1", "Berlin", "DE", "233002380", legal=True)
    res = evaluate_pair(a, b, {})
    assert res.is_match and res.pass_type in {"tax_loose_supported", "name_address_exact", "name_exact", "exact_duplicate"}

def test_person_common_street_token_different_city_rejected():
    a = row("Robert Boehme", "MARTIN-RIESENBURGER-STR. 38", "BERLIN")
    b = row("Robert Hennig", "Martinstrasse 68", "Darmstadt")
    assert not evaluate_pair(a, b, {}).is_match


def test_generic_business_word_name_only_rejected():
    a = row("Services LLC", legal=True, company=True)
    b = row("Services Inc", legal=True, company=True)
    assert not evaluate_pair(a, b, {}).is_match


def test_new_generic_industry_token_false_positives_rejected():
    bad_pairs = [
        ("202 Production", "FIT Production GmbH"),
        ("30 Point Strategies, LLC", "CHEMICAL POINT LTD."),
        ("3B Scientific Corporation", "CJSC Scientific Center of Drug"),
        ("3WAY PHARM INC.", "Alps Pharm. Ind.Co.,Ltd."),
        ("Thermo Fisher Scientific", "Thermoquest Scientific"),
        ("Divis Laboratories", "Neuland Laboratories"),
        ("Chemie Pharm", "JPN Pharm"),
        ("Advanced Clinical Services", "Clinical Practice Research"),
        ("1-Material Inc.", "Grp Material Supplies"),
        ("Association GEMINI", "CAP GEMINI FRANCE"),
        ("Association Jasmin de Riche Lieu", "Jasmin Adler"),
        ("ASSOCIATION OF COMMUNITY", "BCN Brand Community Network GmbH"),
        ("ATLAS PORTAGE", "Tele Atlas Navigation ServicCenter"),
        ("ATMI Packaging Inc.", "AVI Packaging GmbH"),
        ("ATS Automation Tooling Systems GmbH", "TAP The Automation Partnership"),
        ("AVG Trucks GmbH", "DAF Trucks Frankfurt GmbH"),
        ("Avin Electronics", "EKC Advanced Electronics Switzerland"),
        ("AVIS Autoverhuur B.V.", "Merck B.V."),
        ("HANGZHOU FLUORO PHARMACEUTICAL CO.", "HANGZHOU TIGERMED CONSULTING CO."),
        ("EP-EXPRESS", "Service Express, LLC"),
    ]
    for left, right in bad_pairs:
        assert not evaluate_pair(row(left, city="Berlin", country="DE", legal=True, company=True), row(right, city="Berlin", country="DE", legal=True, company=True), {}).is_match


def test_location_and_publishing_tokens_do_not_bridge_clusters():
    bad_pairs = [
        ("WUHAN 3B SCIENTIFIC CORP", "Wuhan Hezhong Chemical", "Wuhan", "CN"),
        ("Wuhan Sunshine Optoelectronics Tech", "Wuhan Yeasen Biotechnology Co., Ltd", "Wuhan", "CN"),
        ("Nanjing GenScript Biotech", "NANJING YICHEN TRADING CO.,LTD", "Nanjing", "CN"),
        ("Inabata Europe S.A.", "Management Centre Europe", "Brussels", "BE"),
        ("CCT EUROPE BV", "CFS Europe S.p.A.", "Berlin", "DE"),
        ("BUND-VERLAG GMBH", "TUEV -Verlag GmbH", "Koeln", "DE"),
        ("China National Chemical", "China National Pharmaceutical", "Shanghai", "CN"),
        ("Duesseldorf Congress GmbH", "Düsseldorf Tourismus GmbH", "Duesseldorf", "DE"),
        ("GEO Net solution GmbH", "Lab Science Solution Sdn Bhd", "Leipzig", "DE"),
        ("ASIA OPTICAL CO., INC.", "Asia Pacific Gas Ent. Co. Ltd.", "Taipei", "TW"),
        ("Sanitaetshaus Armin Kunze GmbH", "Sanitätshaus Fritsch GmbH", "Darmstadt", "DE"),
        ("Gebrüder Schmitt GmbH", "Gebrueder Weiss GmbH", "Hockenheim", "DE"),
    ]
    for left, right, city, country in bad_pairs:
        assert not evaluate_pair(row(left, city=city, country=country, legal=True, company=True), row(right, city=city, country=country, legal=True, company=True), {}).is_match


def test_generic_industry_tokens_can_support_with_strong_evidence():
    assert evaluate_pair(
        row("Chemical Point UG", "Main Str. 1", "Berlin", "DE", legal=True, company=True),
        row("Chemical Point Handel UG", "Main Strasse 1", "Berlin", "DE", legal=True, company=True),
        {},
    ).is_match
    assert evaluate_pair(
        row("Chemical Point UG", city="Berlin", country="DE", tax="DE123456789", legal=True, company=True),
        row("Chemical Point Handel UG", city="Berlin", country="DE", tax="DE123456789", legal=True, company=True),
        {},
    ).is_match
    assert evaluate_pair(
        row("Chemical Point UG", city="Berlin", country="DE", domain="chemical-point.example", legal=True, company=True),
        row("Chemical Point Handel UG", city="Berlin", country="DE", domain="chemical-point.example", legal=True, company=True),
        {},
    ).is_match
    assert evaluate_pair(
        row("ATLAS Material Testing Technology G", "1 Lab Road", "Linsengericht", "DE", legal=True, company=True),
        row("ATLAS Material Testing Technology G", "1 Lab Road", "Linsengericht", "DE", legal=True, company=True),
        {},
    ).is_match


def test_same_address_generic_token_only_company_names_rejected():
    bad_pairs = [
        ("Aberle Software GmbH", "Körber Supply Chain Software GmbH"),
        ("Chlor Chemicals", "ICI Chemicals & Polymers"),
        ("BIZINT Solutions, Inc.", "Syndio Solutions, Inc."),
    ]
    for left, right in bad_pairs:
        assert not evaluate_pair(
            row(left, "100 Shared Road", "Berlin", "DE", legal=True, company=True),
            row(right, "100 Shared Road", "Berlin", "DE", legal=True, company=True),
            {},
        ).is_match


def test_sigma_broad_tax_does_not_pull_unrelated_suppliers():
    config = ClusteringConfig()
    config._tax_block_stats = {
        "de811968975": {"row_count": 17, "distinct_roots": 4, "distinct_names": 5}
    }
    unrelated = [
        "ASL Cargo GmbH",
        "Integrated DNA Technologies Germany",
        "SIGMA - ARK GmbH",
    ]
    sigma = row("Sigma-Aldrich Chemie GmbH", "Riedstrasse 2", "Steinheim", "DE", tax="DE811968975", legal=True, company=True)
    for name in unrelated:
        assert not evaluate_pair(
            row(name, "Other Road 1", "Darmstadt", "DE", tax="DE811968975", legal=True, company=True),
            sigma,
            {},
            config,
        ).is_match
    assert evaluate_pair(
        row("SIGMA-ALDRICH Chemie GmbH", "Riedstrasse 2", "Steinheim", "DE", tax="DE811968975", legal=True, company=True),
        sigma,
        {},
        config,
    ).is_match


def test_autohaus_weeber_address_range_clusters():
    res = evaluate_pair(
        row("Autohaus Weeber GmbH", "GLEMSECKSTR 39", "Leonberg", "DE", legal=True, company=True),
        row("Autohaus Weeber GmbH & Co.KG", "Glemseckstrasse 39/49", "Leonberg", "DE", legal=True, company=True),
        {},
    )
    assert res.is_match


def test_autohaus_weeber_address_range_clusters_despite_legal_tax_split():
    result = clustered_names([
        ("Autohaus Weeber GmbH", "GLEMSECKSTR 39", "Leonberg", "DE", "71229", "DE811460983"),
        ("Autohaus Weeber GmbH & Co.KG", "Glemseckstrasse 39/49", "Leonberg", "DE", "71229", "DE146004045"),
    ])
    assert result["stats"]["clusters_found"] == 1


def test_fluoropharm_distinctive_prefix_same_address_clusters():
    res = evaluate_pair(
        row("HANGZHOU FLUORO PHARMACEUTICAL CO.", "4028 Joinhands Science Park", "Hangzhou", "CN", legal=True, company=True),
        row("Fluoropharm Co., Ltd.", "4028 Joinhands Science Park", "Nanhuan RD, Hangzhou", "CN", legal=True, company=True),
        {},
    )
    assert res.is_match
    assert res.needs_review


def test_same_address_secondary_family_evidence_clusters():
    service = row("Service Express, LLC", "3855 Sparks Dr. SE", "Grand Rapids", "US", legal=True, company=True)
    top_gun = row("TOP GUN TECHNOLOGY, INC.", "3855 Sparks Dr", "Grand Rapids", "US", legal=True, company=True)
    res = evaluate_pair(service, top_gun, {})
    assert res.is_match
    assert res.pass_type == "address_secondary_or_acronym"


def test_weka_family_and_turnus_address_bridge_clusters_for_review():
    # Round 2: family/review edges are routed to review_candidates and do not
    # enter the main union-find merger.
    result = clustered_names_with_metadata([
        ("TURNUS GMBH", "Römerstraße 4", "Kissing", "DE", "86438", {}),
        ("Weka Business Medien GmbH", "Julius-Reiber-Str. 15", "Darmstadt", "DE", "64293", {"familyName": "WEKA BUSINESS MEDIEN GMBH"}),
        ("WEKA MEDIA GMBH & CO. KG", "Römerstraße 4", "Kissing", "DE", "86438", {"familyName": "WEKA BUSINESS MEDIEN GMBH"}),
        ("WEKA Media GmbH & Co.KG", "Roemerstrasse 4", "Kissing", "DE", "86438", {"familyName": "WEKA BUSINESS MEDIEN GMBH"}),
        ("WEKA MEDIA PUBLISHING GMBH", "Richard-Reitzner-Allee 2", "Haar bei München", "DE", "85540", {}),
        ("WEKA VERLAG AUGSBURG", "Augsburg", "Augsburg", "DE", "86150", {}),
    ])
    rows = result["main_df"].to_dicts()
    cluster_by_name = {r["Supplier Name"]: r["Cluster Number"] for r in rows}
    assert cluster_by_name.get("TURNUS GMBH") is None
    review_candidates = result.get("review_candidates", [])
    assert len(review_candidates) >= 1
    review_pass_types = {rc["pass_type"] for rc in review_candidates}
    assert review_pass_types & {"known_family_bridge", "family_bridge_supported", "secondary_or_acronym_bridge", "name_exact_review"}


def test_turnus_address_bridge_requires_known_family_pair():
    assert not evaluate_pair(
        row("Random GmbH", "Römerstraße 4", "Kissing", "DE", legal=True, company=True),
        row("WEKA MEDIA GMBH & CO. KG", "Römerstraße 4", "Kissing", "DE", legal=True, company=True),
        {},
    ).is_match


def test_media_publishing_verlag_business_are_not_family_roots():
    result = clustered_names([
        ("Media Publishing GmbH", "1 Main Str", "Berlin", "DE", "10115"),
        ("Business Verlag GmbH", "2 Main Str", "Berlin", "DE", "10115"),
        ("Publishing Business Media GmbH", "3 Main Str", "Berlin", "DE", "10115"),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_known_brand_family_loader_classifies_aliases(tmp_path):
    path = tmp_path / "known_brand_families.csv"
    path.write_text(
        "keyword\n"
        "Fed Ex;FedEx;Federal Express;Express\n"
        "Cap gemini;Capgemini;Gemini\n"
        "Metro;Metro\n"
        "Too Short;AA\n"
        "Deutsche Post;DHL Group;DPDHL;DHL\n",
        encoding="utf-8",
    )
    loaded = load_known_brand_families(str(path))
    report = loaded.report
    assert report["rows_loaded"] == 5
    assert report["aliases_accepted_safe"] >= 5
    assert report["aliases_skipped_generic"] >= 2  # Express, Gemini
    assert report["aliases_marked_risky"] >= 1  # Metro
    assert report["aliases_skipped_short"] >= 1  # AA


def test_known_brand_family_aliases_cluster_safe_phrases(tmp_path):
    # Round 2: known_brand_family_alias edges below 88 are review-only unless
    # stronger address/domain/name support raises confidence.
    path = tmp_path / "known_brand_families.csv"
    path.write_text(
        "keyword\n"
        "Fed Ex;FedEx;Federal Express\n"
        "Deutsche Post;DHL Group;DPDHL;DHL\n"
        "Cap gemini;Capgemini\n"
        "AkzoNobel;Akzo Nobel;AKZA\n"
        "Perkin Elmer;Revvity;Perkin-Elmer\n",
        encoding="utf-8",
    )

    fedex = clustered_names_with_brand_file([
        ("FedEx Express Germany", "", "Frankfurt", "DE", ""),
        ("Federal Express", "", "Memphis", "US", ""),
        ("FEDEX UK LTD", "", "London", "GB", ""),
    ], path)
    assert fedex["stats"]["clusters_found"] >= 1 or len(fedex.get("review_candidates", [])) >= 1

    dhl = clustered_names_with_brand_file([
        ("DHL Express", "", "Bonn", "DE", ""),
        ("Deutsche Post", "", "Bonn", "DE", ""),
        ("DHL Group", "", "Bonn", "DE", ""),
    ], path)
    assert dhl["stats"]["clusters_found"] == 0 or len(dhl.get("review_candidates", [])) >= 1

    cap = clustered_names_with_brand_file([
        ("CAP GEMINI FRANCE", "", "Paris", "FR", ""),
        ("Capgemini Technology Services", "", "Mumbai", "IN", ""),
        ("Association GEMINI", "", "Bratislava", "SK", ""),
    ], path)
    rows = cap["main_df"].sort("Supplier Name").to_dicts()
    clusters = {r["Supplier Name"]: r["Cluster Number"] for r in rows}
    assert clusters["Association GEMINI"] is None
    cap_clustered_together = (
        clusters["CAP GEMINI FRANCE"] is not None
        and clusters["CAP GEMINI FRANCE"] == clusters["Capgemini Technology Services"]
    )
    assert cap_clustered_together or len(cap.get("review_candidates", [])) >= 1

    akzo = clustered_names_with_brand_file([
        ("AkzoNobel Chemicals GmbH", "", "Koeln", "DE", ""),
        ("Akzo Nobel Coatings", "", "Amsterdam", "NL", ""),
    ], path)
    assert akzo["stats"]["clusters_found"] >= 1 or len(akzo.get("review_candidates", [])) >= 1

    revvity = clustered_names_with_brand_file([
        ("Perkin Elmer Inc", "", "Boston", "US", ""),
        ("Revvity Health Sciences", "", "Waltham", "US", ""),
    ], path)
    assert revvity["stats"]["clusters_found"] == 0 or len(revvity.get("review_candidates", [])) >= 1


def test_known_brand_aliases_do_not_use_generic_partial_tokens(tmp_path):
    path = tmp_path / "known_brand_families.csv"
    path.write_text(
        "keyword\n"
        "Fed Ex;FedEx;Federal Express\n"
        "Cap gemini;Capgemini\n"
        "Generic;Group;Bank;Energy;Services;Technology;Chemical;International;Global;Express\n"
        "Green;Green\n",
        encoding="utf-8",
    )

    express = clustered_names_with_brand_file([
        ("Service Express", "", "Grand Rapids", "US", ""),
        ("EP-EXPRESS", "", "Unterankenreute", "DE", ""),
    ], path)
    assert express["stats"]["clusters_found"] == 0

    gemini = clustered_names_with_brand_file([
        ("Association GEMINI", "", "Bratislava", "SK", ""),
        ("CAP GEMINI FRANCE", "", "Villeurbanne", "FR", ""),
    ], path)
    assert gemini["stats"]["clusters_found"] == 0

    generic = clustered_names_with_brand_file([
        ("Global Energy Group", "", "Berlin", "DE", ""),
        ("International Chemical Services", "", "Berlin", "DE", ""),
        ("Technology Solutions Bank", "", "Berlin", "DE", ""),
    ], path)
    assert generic["stats"]["clusters_found"] == 0

    green = clustered_names_with_brand_file([
        ("FEEL GREEN", "", "Berlin", "DE", ""),
        ("green-ivf", "", "Berlin", "DE", ""),
    ], path)
    assert green["stats"]["clusters_found"] == 0


def test_legal_keyword_file_does_not_create_bridge_clusters(tmp_path):
    legal_file = tmp_path / "legal_keywords.csv"
    legal_file.write_text("legal keywords\nGmbH\nAG\nLtd\nLimited\nS.A.\nBV\n", encoding="utf-8")
    result = clustered_names_with_files([
        ("ABC GmbH", "", "Berlin", "DE", ""),
        ("XYZ GmbH", "", "Berlin", "DE", ""),
        ("MNO AG", "", "Munich", "DE", ""),
        ("PQR AG", "", "Munich", "DE", ""),
    ], legal_file=legal_file)
    assert result["stats"]["clusters_found"] == 0


def test_multilingual_generic_non_bridge_words_do_not_cluster():
    cases = [
        ("Alpha Technology GmbH", "Beta Technology GmbH"),
        ("Alpha Technologie GmbH", "Beta Technologie GmbH"),
        ("Alpha Tecnologia Ltd", "Beta Tecnologia Ltd"),
        ("Alpha Society", "Beta Society"),
        ("Alpha Association", "Beta Association"),
        ("Alpha Verein", "Beta Verein"),
        ("Alpha Stiftung", "Beta Stiftung"),
        ("Alpha Fondation", "Beta Fondation"),
        ("Alpha Institut", "Beta Institut"),
        ("Alpha Services GmbH", "Beta Services GmbH"),
        ("Alpha Dienstleistungen GmbH", "Beta Dienstleistungen GmbH"),
        ("Alpha Servicios SA", "Beta Servicios SA"),
        ("Alpha Servizi SpA", "Beta Servizi SpA"),
        ("Alpha Logistics Ltd", "Beta Logistics Ltd"),
        ("Alpha Logistik GmbH", "Beta Logistik GmbH"),
        ("Alpha Logistique SAS", "Beta Logistique SAS"),
    ]
    for left, right in cases:
        result = clustered_names([
            (left, "", "Berlin", "DE", ""),
            (right, "", "Berlin", "DE", ""),
        ])
        assert result["stats"]["clusters_found"] == 0, f"{left} / {right} clustered"


def test_brand_alias_file_and_legal_file_are_separate(tmp_path):
    # Round 2: no-address brand alias family matches are review candidates,
    # not main auto-clusters.
    brand_file = tmp_path / "known_brand_families.csv"
    brand_file.write_text(
        "keyword\n"
        "Fed Ex;FedEx;Federal Express\n"
        "Deutsche Post;DHL Group;DPDHL;DHL\n"
        "Cap gemini;Capgemini\n"
        "AkzoNobel;Akzo Nobel;Nouryon\n",
        encoding="utf-8",
    )
    legal_file = tmp_path / "legal_keywords.csv"
    legal_file.write_text("legal keywords\nGmbH\nLtd\nLimited\nInc\nLLC\n", encoding="utf-8")

    fedex = clustered_names_with_files([
        ("FedEx Express Germany GmbH", "", "Bonn", "DE", ""),
        ("Federal Express Ltd", "", "London", "GB", ""),
        ("FEDEX UK LTD", "", "London", "GB", ""),
    ], brand_file=brand_file, legal_file=legal_file)
    assert fedex["stats"]["clusters_found"] == 0 or len(fedex.get("review_candidates", [])) >= 1

    dhl = clustered_names_with_files([
        ("DHL Express GmbH", "", "Bonn", "DE", ""),
        ("Deutsche Post AG", "", "Bonn", "DE", ""),
    ], brand_file=brand_file, legal_file=legal_file)
    assert dhl["stats"]["clusters_found"] == 0 or len(dhl.get("review_candidates", [])) >= 1

    capgemini = clustered_names_with_files([
        ("Cap Gemini France", "", "Paris", "FR", ""),
        ("Capgemini Technology Services Ltd", "", "Mumbai", "IN", ""),
    ], brand_file=brand_file, legal_file=legal_file)
    assert capgemini["stats"]["clusters_found"] == 0 or len(capgemini.get("review_candidates", [])) >= 1

    bad_express = clustered_names_with_files([
        ("EP-EXPRESS", "", "Unterankenreute", "DE", ""),
        ("Service Express LLC", "", "Grand Rapids", "US", ""),
    ], brand_file=brand_file, legal_file=legal_file)
    assert bad_express["stats"]["clusters_found"] == 0

    bad_gemini = clustered_names_with_files([
        ("Association GEMINI", "", "Bratislava", "SK", ""),
        ("CAP GEMINI FRANCE", "", "Villeurbanne", "FR", ""),
    ], brand_file=brand_file, legal_file=legal_file)
    assert bad_gemini["stats"]["clusters_found"] == 0


def test_legal_keyword_person_company_bridge_requires_address_support(tmp_path):
    legal_file = tmp_path / "legal_keywords.csv"
    legal_file.write_text("legal keywords\nGmbH\n", encoding="utf-8")
    same_address = clustered_names_with_files([
        ("Wilhelm Schmidt", "Breslauer Strasse 14", "Seeheim-Jugenheim", "DE", ""),
        ("Wilhelm Schmidt GmbH", "Breslauer Straße 14", "Seeheim-Jugenheim", "DE", ""),
    ], legal_file=legal_file)
    assert same_address["stats"]["clusters_found"] == 1

    different_address = clustered_names_with_files([
        ("Wilhelm Schmidt", "Breslauer Strasse 14", "Seeheim-Jugenheim", "DE", ""),
        ("Wilhelm Schmidt GmbH", "Other Strasse 99", "Seeheim-Jugenheim", "DE", ""),
    ], legal_file=legal_file)
    assert different_address["stats"]["clusters_found"] == 0


def test_risky_known_brand_alias_requires_extra_evidence(tmp_path):
    path = tmp_path / "known_brand_families.csv"
    path.write_text("keyword\nMetro;Metro\n", encoding="utf-8")
    no_support = clustered_names_with_brand_file([
        ("Metro Logistics", "", "Berlin", "DE", ""),
        ("Metro Bank", "", "London", "GB", ""),
    ], path)
    assert no_support["stats"]["clusters_found"] == 0

    address_support = clustered_names_with_brand_file([
        ("Metro Logistics", "1 Same Road", "Berlin", "DE", ""),
        ("Metro Bank", "1 Same Road", "Berlin", "DE", ""),
    ], path)
    assert address_support["stats"]["clusters_found"] == 1


def test_numeric_code_prefix_requires_strong_support():
    broad_identity = evaluate_pair(
        row("3B Scientific Corporation", city="Wuhan", country="CN", legal=True, company=True),
        row("WUHAN 3B SCIENTIFIC CORP", city="Wuhan", country="CN", legal=True, company=True),
        {},
    )
    assert broad_identity.is_match
    assert broad_identity.pass_type == "distinctive_supplier_identity"
    assert 80 <= broad_identity.match_pct <= 89
    assert broad_identity.needs_review
    assert evaluate_pair(
        row("3B Scientific Corporation", "1 Science Road", "Wuhan", "CN", legal=True, company=True),
        row("WUHAN 3B SCIENTIFIC CORP", "1 Science Rd", "Wuhan", "CN", legal=True, company=True),
        {},
    ).is_match
    assert evaluate_pair(
        row("3B Scientific Corporation", city="Wuhan", country="CN", domain="3bscientific.example", legal=True, company=True),
        row("WUHAN 3B SCIENTIFIC CORP", city="Wuhan", country="CN", domain="3bscientific.example", legal=True, company=True),
        {},
    ).is_match


def test_restaurant_chain_different_addresses_rejected():
    a = row("Subway 100", "1 Main St", "Austin", "US", hospitality=True)
    b = row("Subway 200", "2 Main St", "Austin", "US", hospitality=True)
    assert not evaluate_pair(a, b, {}).is_match


def test_acronym_same_city_only_rejected():
    a = row("ABC", "", "Toronto", "CA")
    b = row("Alpha Beta Consulting", "", "Toronto", "CA", company=True)
    assert not evaluate_pair(a, b, {}).is_match


def test_acronym_allowed_with_business_domain_support():
    a = row("ABC", "", "Toronto", "CA", domain="abc.example")
    b = row("Alpha Beta Consulting", "", "Toronto", "CA", domain="abc.example", company=True)
    assert evaluate_pair(a, b, {}).is_match


def test_access_open_consulting_words_do_not_cluster():
    result = clustered_names([
        ("Access BIO, L.C.", "1 Main St", "Boston", "US", "02101"),
        ("Access Events International", "2 Main St", "Boston", "US", "02101"),
        ("OPEN Access Consulting", "3 Main St", "Boston", "US", "02101"),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_cologne_location_and_publishing_words_do_not_cluster():
    result = clustered_names([
        ("AdCoach Marketing & Publishing Serv", "1 Media Str", "Cologne", "DE", "50667"),
        ("Cologne Energy- Loesungen", "2 Energy Str", "Cologne", "DE", "50667"),
        ("Cologne Publishing Group GmbH", "3 Print Str", "Cologne", "DE", "50667"),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_beijing_technology_and_bridge_name_do_not_pull_unrelated_supplier():
    result = clustered_names([
        ("Beijing Skywing Technology Co., Ltd", "1 Skywing Road", "Beijing", "CN", "100000"),
        ("BEIJING WONDER UNION TECHNOLOGY", "2 Wonder Road", "Beijing", "CN", "100000"),
        ("Merck Millipore Beijing Skywing", "1 Skywing Road", "Beijing", "CN", "100000"),
    ])
    out = result["main_df"].sort("Supplier Name")
    cluster_by_name = dict(zip(out["Supplier Name"].to_list(), out["Cluster Number"].to_list()))
    wonder_cluster = cluster_by_name["BEIJING WONDER UNION TECHNOLOGY"]
    assert wonder_cluster is None


def test_weak_edges_do_not_chain_into_giant_cluster():
    result = clustered_names([
        ("Acme Alpha GmbH", "1 Shared Road", "Berlin", "DE", "10115"),
        ("Acme Beta GmbH", "2 Shared Road", "Berlin", "DE", "10115"),
        ("Beta Gamma GmbH", "3 Shared Road", "Berlin", "DE", "10115"),
        ("Gamma Delta GmbH", "4 Shared Road", "Berlin", "DE", "10115"),
    ])
    clusters = [
        c for c in result["main_df"]["Cluster Number"].to_list()
        if c is not None
    ]
    assert result["stats"]["largest_cluster_size"] <= 2
    assert len(set(clusters)) <= 2


def test_known_family_bridge_is_low_confidence_review_candidate():
    res = evaluate_pair(
        row("Merck KGaA", "Frankfurter Str. 250", "Darmstadt", "DE", legal=True),
        row("Millipore AB", "Frosundaviks All 1", "Solna", "SE", legal=True),
        {},
    )
    assert res.is_match
    assert res.pass_type == "known_family_bridge"
    assert res.needs_review
    assert res.match_pct <= 70


def test_akzo_nouryon_known_family_bridge_is_low_confidence():
    # Known-family bridges are no longer exempt from the generic/non-bridge guard.
    # Without address/domain/tax support, the two names only share generic industry
    # and legal-form language, so this should remain audit/review-only.
    res = evaluate_pair(
        row("Akzo Nobel Chemicals GmbH", "1 Paint Road", "Hamburg", "DE", legal=True),
        row("Nouryon Chemicals GmbH", "2 Paint Road", "Amsterdam", "NL", legal=True),
        {},
    )
    assert not res.is_match
    assert res.needs_review


def test_broad_exact_tax_requires_name_address_or_domain_support():
    config = ClusteringConfig()
    config._tax_block_stats = {
        "de811850788": {"row_count": 39, "distinct_roots": 30, "distinct_names": 35}
    }
    a = row("Akzo Nobel Chemicals GmbH", "1 Paint Road", "Koeln", "DE", tax="DE811850788", legal=True)
    b = row("VIDEC Data Engineering GmbH", "9 Data Road", "Bremen", "DE", tax="DE811850788", legal=True)
    assert not evaluate_pair(a, b, {}, config).is_match

    c = row("Akzo Nobel Chemicals GmbH", "1 Paint Road", "Koeln", "DE", tax="DE811850788", legal=True)
    d = row("AKZO NOBEL CHEMICALS LTD", "2 Paint Road", "Surrey", "GB", tax="DE811850788", legal=True)
    assert evaluate_pair(c, d, {}, config).is_match


def test_reused_small_exact_tax_block_requires_name_address_or_domain_support():
    config = ClusteringConfig()
    config._tax_block_stats = {
        "nl004702426b01": {"row_count": 4, "distinct_roots": 2, "distinct_names": 2}
    }
    avis = row("AVIS Autoverhuur B.V.", "Louis Armstrongweg 4", "Almere", "NL", tax="NL004702426B01", legal=True)
    merck = row("Merck B.V.", "Tupolevlaan 41-61", "Schiphol-Rijk", "NL", tax="NL004702426B01", legal=True)
    assert not evaluate_pair(avis, merck, {}, config).is_match
    assert evaluate_pair(
        row("Merck B.V.", "Tupolevlaan 41-61", "Schiphol-Rijk", "NL", tax="NL004702426B01", legal=True),
        merck,
        {},
        config,
    ).is_match


# ---------------------------------------------------------------------------
# Precision guardrails: generic/location root tokens must not score 85%
# ---------------------------------------------------------------------------

def test_bank_only_does_not_bridge_distinct_banks_at_85():
    """Bank of Canada / Bank of Montreal share only "bank" — a generic word.
    They must not be scored 85% (MANUAL_REVIEW). Either blank or ≤70% only."""
    pairs = [
        ("Bank of Canada", "Bank of Montreal"),
        ("Bank of Canada", "Bank of Nova Scotia"),
        ("Bank of Montreal", "Bank of Nova Scotia"),
        ("TD Bank", "Bank of Canada"),
    ]
    for name_a, name_b in pairs:
        a = row(name_a, "", "Ottawa", "CA")
        b = row(name_b, "", "Montreal", "CA")
        res = evaluate_pair(a, b, {})
        assert not res.is_match or res.match_pct <= 70, (
            f"{name_a!r} vs {name_b!r}: expected blank or ≤70%, got {res.match_pct}% "
            f"(pass_type={res.pass_type})"
        )


def test_canadian_adjective_does_not_bridge_unrelated_suppliers():
    """'Canadian' is a geographic adjective and must never be the root bridge."""
    pairs = [
        ("Canadian Black Book", "Canadian Linen Supply"),
        ("Canadian Tire", "Canadian Pacific Railway"),
        ("Canadian Broadcasting Corporation", "Canadian Utilities"),
    ]
    for name_a, name_b in pairs:
        a = row(name_a, "", "Toronto", "CA")
        b = row(name_b, "", "Toronto", "CA")
        res = evaluate_pair(a, b, {})
        assert not res.is_match or res.match_pct <= 70, (
            f"{name_a!r} vs {name_b!r}: should not score 85%+, got {res.match_pct}% "
            f"(pass_type={res.pass_type})"
        )


def test_plus_suffix_does_not_bridge_unrelated_suppliers():
    """'Plus' is too generic to bridge distinct suppliers (Facility Plus vs Food Plus)."""
    pairs = [
        ("Facility Plus", "Food Plus"),
        ("Office Plus", "Trucks Plus"),
        ("Facility Plus", "Office Plus"),
    ]
    for name_a, name_b in pairs:
        a = row(name_a, "", "Toronto", "CA")
        b = row(name_b, "", "Toronto", "CA")
        res = evaluate_pair(a, b, {})
        assert not res.is_match or res.match_pct <= 70, (
            f"{name_a!r} vs {name_b!r}: 'plus'-only should not score 85%+, got {res.match_pct}% "
            f"(pass_type={res.pass_type})"
        )


def test_auto_prefix_does_not_bridge_unrelated_suppliers():
    """'Auto' alone is too generic to bridge distinct automotive suppliers."""
    pairs = [
        ("AUTO A LA CARTE", "AUTO VALUE"),
        ("AUTO A LA CARTE", "AUTO MOTION"),
        ("AUTO VALUE", "AUTO PRO"),
    ]
    for name_a, name_b in pairs:
        a = row(name_a, "", "Toronto", "CA")
        b = row(name_b, "", "Toronto", "CA")
        res = evaluate_pair(a, b, {})
        assert not res.is_match or res.match_pct <= 70, (
            f"{name_a!r} vs {name_b!r}: 'auto'-only should not score 85%+, got {res.match_pct}% "
            f"(pass_type={res.pass_type})"
        )


def test_telus_bare_entity_connected_to_division_at_most_70():
    """TELUS (standalone) vs TELUS Health must be ≤70% — bare single-token hub.

    Without domain/address/tax, a standalone brand name connecting to a
    multi-word division is ambiguous and must route to LLM review (70%)."""
    telus_bare = row("TELUS", "", "Vancouver", "CA")
    telus_health = row("TELUS Health", "", "Vancouver", "CA")
    res = evaluate_pair(telus_bare, telus_health, {})
    if res.is_match:
        assert res.match_pct <= 70, (
            f"TELUS vs TELUS Health: bare single-token hub must be ≤70%, got {res.match_pct}%"
        )


def test_telus_divisions_peer_to_peer_not_85():
    """TELUS Health vs TELUS Communications should not score 85% (two-word name guardrail)."""
    a = row("TELUS Health", "", "Vancouver", "CA")
    b = row("TELUS Communications", "", "Vancouver", "CA")
    res = evaluate_pair(a, b, {})
    assert not res.is_match or res.match_pct <= 70, (
        f"TELUS Health vs TELUS Communications: must be blank or ≤70%, got {res.match_pct}%"
    )


def test_telus_multi_word_with_domain_support_can_still_cluster():
    """Multi-word TELUS entities sharing a business domain should still connect."""
    a = row("TELUS Corporation", "", "Vancouver", "CA", domain="telus.example")
    b = row("TELUS Health Inc", "", "Vancouver", "CA", domain="telus.example", legal=True)
    res = evaluate_pair(a, b, {})
    assert res.is_match, "TELUS Corporation / TELUS Health sharing domain must still cluster"


def test_scotiabank_and_bank_of_nova_scotia_same_address_clusters():
    """Scotia / Bank of Nova Scotia with same address must remain captured."""
    a = row("Scotiabank", "44 King St W", "Toronto", "CA")
    b = row("Bank of Nova Scotia", "44 King St W", "Toronto", "CA")
    res = evaluate_pair(a, b, {})
    assert res.is_match, (
        "Scotiabank vs Bank of Nova Scotia at same address must still cluster — "
        "do not break this recall case"
    )


def test_bfi_base_and_branch_still_cluster():
    """BFI Canada and BFI Canada Ltd must remain captured (not broken by generic-token changes)."""
    result = clustered_names([
        ("BFI Canada", "1 Industrial Rd", "Calgary", "CA", "T2E 7C4"),
        ("BFI Canada Ltd", "1 Industrial Rd", "Calgary", "CA", "T2E 7C4"),
    ])
    clusters = [c for c in result["main_df"]["Cluster Number"].to_list() if c is not None]
    assert len(clusters) == 2 and len(set(clusters)) == 1, (
        "BFI Canada / BFI Canada Ltd must still cluster together"
    )


def test_hy_louie_variants_still_cluster():
    """H. Y. Louie and H.Y. Louie Co. must remain captured (recall regression guard)."""
    result = clustered_names([
        ("H. Y. Louie Co.", "1450 Foreshore Walk", "Vancouver", "CA", "V6H 3X6"),
        ("H.Y. Louie Company Limited", "1450 Foreshore Walk", "Vancouver", "CA", "V6H 3X6"),
    ])
    clusters = [c for c in result["main_df"]["Cluster Number"].to_list() if c is not None]
    assert len(clusters) == 2 and len(set(clusters)) == 1, (
        "H. Y. Louie variants must still cluster together"
    )


def test_rogers_and_rogers_cable_without_support_not_85():
    """Rogers vs Rogers Cable without domain/address/tax must be ≤70% (weak single-token)."""
    a = row("Rogers", "", "Toronto", "CA")
    b = row("Rogers Cable", "", "Toronto", "CA")
    res = evaluate_pair(a, b, {})
    if res.is_match:
        assert res.match_pct <= 70, (
            f"Rogers vs Rogers Cable (bare) must be ≤70% without support, got {res.match_pct}%"
        )

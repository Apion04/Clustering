"""Supplier brand/group identity policy tests."""

import polars as pl

from src.main import cluster_suppliers
from src.matching import evaluate_pair
from tests.test_precision_guardrails import row


def _cluster(records):
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Address": [r[1] if len(r) > 1 else "" for r in records],
        "City": [r[2] if len(r) > 2 else "" for r in records],
        "Country": [r[3] if len(r) > 3 else "" for r in records],
        "Postal Code": [r[4] if len(r) > 4 else "" for r in records],
        "Tax ID": [r[5] if len(r) > 5 else "" for r in records],
    })
    return cluster_suppliers(df, {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
        "tax_id": "Tax ID",
    })


def _cluster_numbers(result):
    return [r["Cluster Number"] for r in result["main_df"].to_dicts()]


def _assert_one_main_cluster(result, expected_rows=None):
    clusters = [c for c in _cluster_numbers(result) if c is not None]
    assert clusters
    assert len(set(clusters)) == 1
    if expected_rows is not None:
        assert len(clusters) == expected_rows


def test_cognis_group_identity_not_missed():
    result = _cluster([
        ("COGNIS", "LITTLE ISLAND", "CORK", "IT", "99999"),
        ("COGNIS CORPORATION", "CINCINNATI", "US", "45232-1446"),
        ("Cognis Deutschland GmbH & Co.KG", "Henkelstrasse 67", "Duesseldorf", "DE", "40589"),
        ("Cognis France", "185 Av. de Fontebleau", "St-Fargeau Ponthierry", "FR", "77981"),
        ("Cognis GmbH", "Robert-Hansen-Str. 1", "Illertissen", "DE", "89257"),
        ("Cognis GmbH", "Rheinpromenade 1", "Monheim am Rhein", "DE", "40789"),
        ("Cognis GmbH", "Rheinpromenade 1", "Monheim", "DE", "40789"),
        ("Cognis Performance Chem. UK Ltd.", "Charleston Road", "Southhampton", "GB", "SO45 3ZG"),
        ("Cognis UK Ltd", "Charleston Road", "Southampton", "GB", "SO45 3ZG"),
    ])
    _assert_one_main_cluster(result, expected_rows=9)
    assert result["stats"]["pass_type_counts"].get("distinctive_supplier_identity", 0) >= 1


def test_cognizant_group_identity_not_missed():
    result = _cluster([
        ("Cognizant Technology Solutions", "", "Dublin", "IE"),
        ("Cognizant Technology Solutions GmbH", "", "Frankfurt", "DE"),
        ("Cognizant Technology Solutions US C", "", "Teaneck", "US"),
        ("Cognizant Worldwide Limited", "", "London", "GB"),
        ("COGNIZANT WORLDWIDE LIMITED", "", "London", "GB"),
    ])
    _assert_one_main_cluster(result, expected_rows=5)


def test_computershare_group_identity_not_missed():
    result = _cluster([
        ("Computershare Communication", "", "Munich", "DE"),
        ("Computershare Deutschland GmbH & Co", "", "Munich", "DE"),
        ("Computershare Inc.", "", "Palatine", "US"),
        ("Computershare Inc.", "", "Palatine", "US"),
    ])
    _assert_one_main_cluster(result, expected_rows=4)


def test_colonial_metals_exact_core_not_missed():
    result = _cluster([
        ("Colonial Metals Inc", "South Wirral", "South Wirral", "GB"),
        ("Colonial Metals Inc.", "Elkton", "Elkton", "US"),
        ("Colonial Metals, Inc.", "Elkton", "Elkton", "US"),
    ])
    _assert_one_main_cluster(result, expected_rows=3)


def test_punctuation_hyphen_and_regional_variants_not_missed():
    result = _cluster([
        ("A. Hartrodt (GmbH & Co) KG", "", "Hamburg", "DE"),
        ("A. Hartrodt B.V.", "", "Rotterdam", "NL"),
        ("A. Hartrodt Deutschland (GmbH & Co)", "", "Hamburg", "DE"),
        ("A.Hartrodt GmbH & Co.", "", "Hamburg", "DE"),
        ("a.hartrodt GmbH & Co.KG", "", "Hamburg", "DE"),
    ])
    _assert_one_main_cluster(result, expected_rows=5)


def test_near_spelling_supplier_identity_goes_to_review_when_not_exact_core():
    result = _cluster([
        ("Abu-Ghazahleh Intellectual Property", "Street A", "Amman", "JO"),
        ("ABU-GHAZALEH INTELLECTUAL PROPERTY", "Street B", "Amman", "JO"),
    ])
    assert result["stats"]["clusters_found"] == 0
    assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))


def test_exact_supplier_identity_different_house_number_not_missed():
    result = _cluster([
        ("ADEKA Europe GmbH", "Berliner Allee 22", "Duesseldorf", "DE", "40212"),
        ("ADEKA Europe GmbH", "Berliner Allee 48", "Duesseldorf", "DE", "40212"),
        ("ADVANCE RESEARCH CHEMICALS LLC", "1110 W KEYSTONE AVE", "Catoosa", "US"),
        ("ADVANCE RESEARCH CHEMICALS, INC.", "110 W KEYSTONE AVE", "Catoosa", "US"),
    ])
    rows = result["main_df"].to_dicts()
    adeka = [r for r in rows if "ADEKA" in r["Supplier Name"]]
    advance = [r for r in rows if "ADVANCE" in r["Supplier Name"]]
    assert adeka[0]["Cluster Number"] == adeka[1]["Cluster Number"]
    assert advance[0]["Cluster Number"] == advance[1]["Cluster Number"]


def test_short_exact_supplier_brand_phrase_not_missed():
    result = _cluster([
        ("aescolab ApS", "", "Berlin", "DE"),
        ("Aescolab ApS", "", "Copenhagen", "DK"),
        ("AGA Gas GmbH", "", "Bad Vilbel", "DE"),
        ("Aga Gas GmbH", "", "Hamburg", "DE"),
    ])
    rows = result["main_df"].to_dicts()
    aes = [r for r in rows if "Aescolab" in r["Supplier Name"] or "aescolab" in r["Supplier Name"]]
    aga = [r for r in rows if "Gas" in r["Supplier Name"]]
    assert aes[0]["Cluster Number"] == aes[1]["Cluster Number"]
    assert aga[0]["Cluster Number"] == aga[1]["Cluster Number"]


def test_regulatory_task_force_related_is_review_only_not_high_confidence():
    res = evaluate_pair(
        row("Acrylate REACH Task Force", "", "Berlin", "DE", legal=False, company=True),
        row("METHACRYLATE REACH TASK FORCE", "", "Berlin", "DE", legal=False, company=True),
        {},
    )
    assert res.is_match
    assert res.pass_type == "regulatory_or_task_force_related"
    assert 70 <= res.match_pct <= 79

    result = _cluster([
        ("Acrylate REACH Task Force", "", "Berlin", "DE"),
        ("METHACRYLATE REACH TASK FORCE", "", "Berlin", "DE"),
    ])
    assert result["stats"]["clusters_found"] == 0
    assert {c["pass_type"] for c in result.get("review_candidates", [])} <= {"regulatory_or_task_force_related"}


def test_old_false_positive_pairs_still_blocked():
    bad_pairs = [
        ("4Tune GmbH", "ValGenesis GmbH"),
        ("A & A Gastronomie GmbH", "DB Gastronomie GmbH"),
        ("ABCR GmbH", "Guntermann & Drunck GmbH"),
        ("IMEC Messtechnik GmbH", "Tempmate GmbH"),
        ("Alfa Medical GmbH", "Alfa Logistics GmbH"),
        ("Meta Qingdao Trading Co Ltd", "Meta Analytics GmbH"),
        ("ADM Executive Ingredients", "ADMIRAL Logistics"),
        ("Jana Zimmer", "Zimmer Medizin GmbH"),
    ]
    for left, right in bad_pairs:
        result = _cluster([
            (left, "1 Shared Road", "Berlin", "DE"),
            (right, "2 Other Road", "Berlin", "DE"),
        ])
        assert result["stats"]["clusters_found"] == 0


def test_low_similarity_exact_tax_goes_to_review_not_main_cluster():
    result = _cluster([
        ("ABCR GmbH", "Im Schlehert 10", "Karlsruhe", "DE", "76187", "DE126575222"),
        ("Guntermann & Drunck GmbH", "Obere Leimbach 9", "Siegen", "DE", "57074", "DE126575222"),
        ("AC medienhaus GmbH", "Mainzer Strasse 1", "Wiesbaden", "DE", "65185", "DE113839238"),
        ("Druckerei Chmielorz GmbH", "Mainzer Strasse 1", "Wiesbaden", "DE", "65185", "DE113839238"),
    ])
    rows = result["main_df"].to_dicts()
    assert not rows[0]["Cluster Number"] or rows[0]["Cluster Number"] != rows[1]["Cluster Number"]
    assert not rows[2]["Cluster Number"] or rows[2]["Cluster Number"] != rows[3]["Cluster Number"]
    assert {c["pass_type"] for c in result.get("review_candidates", [])} >= {"tax_exact_low_similarity_review"}


def test_same_address_and_different_names_remain_not_main_clustered():
    result = _cluster([
        ("AC medienhaus GmbH", "Mainzer Strasse 1", "Wiesbaden", "DE"),
        ("Druckerei Chmielorz GmbH", "Mainzer Strasse 1", "Wiesbaden", "DE"),
        ("Ilona Rosebrock", "Hauptstrasse 8", "Darmstadt", "DE"),
        ("Jonas Rosebrock", "Hauptstrasse 8", "Darmstadt", "DE"),
    ])
    assert result["stats"]["clusters_found"] == 0

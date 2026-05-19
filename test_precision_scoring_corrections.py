"""Precision/scoring regression tests after supplier identity broadening."""

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


def _cluster_number_by_name(result):
    return {r["Supplier Name"]: r["Cluster Number"] for r in result["main_df"].to_dicts()}


def _rows_by_name(result):
    return {r["Supplier Name"]: r for r in result["main_df"].to_dicts()}


def test_chemistry_generic_token_does_not_bridge_supplier_identities():
    result = _cluster([
        ("Aaron Chemistry", "", "Berlin", "DE"),
        ("Advanced Chemistry", "", "Berlin", "DE"),
        ("Alfa Chemistry", "", "Berlin", "DE"),
        ("American Chemistry Council", "", "Berlin", "DE"),
        ("Congress Support International GmbH", "", "Berlin", "DE"),
        ("Science Support", "", "Berlin", "DE"),
        ("Factory Software GmbH", "", "Berlin", "DE"),
        ("Prime Factory GmbH", "", "Berlin", "DE"),
        ("Molecular Products Limited", "", "London", "GB"),
        ("Open Molecular Software Foundation", "", "London", "GB"),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_abbott_trusted_core_not_silently_missed_but_broad_group_is_review():
    result = _cluster([
        ("ABBOTT Biologicals B.V.", "", "Amsterdam", "NL"),
        ("Abbott Laboratories Int.", "", "Chicago", "US"),
        ("ABBOTT Liestal AG", "", "Liestal", "CH"),
    ])
    rows = _rows_by_name(result)
    clusters = {rows[name]["Cluster Number"] for name in rows}
    if any(clusters):
        assert len({c for c in clusters if c}) == 1
        pct = int(str(next(r["Match Percentage"] for r in rows.values() if r["Match Percentage"])).rstrip("%"))
        assert 80 <= pct <= 89
    else:
        assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))


def test_merck_broad_group_without_support_is_below_90_review_only():
    res = evaluate_pair(
        row("Merck KGaA", "", "Darmstadt", "DE", legal=True),
        row("Merck Inc", "", "Rahway", "US", legal=True),
        {},
    )
    assert res.is_match
    assert res.pass_type == "distinctive_supplier_identity"
    assert res.match_pct < 90
    assert res.needs_review


def test_person_suffix_variant_same_location_is_high_priority_review_not_main():
    result = _cluster([
        ("REITER, LEOPOLD JUN.", "", "Vienna", "AT", "1010"),
        ("REITER, LEOPOLD", "", "Vienna", "AT", "1010"),
        ("Jonas Rosebrock", "Hauptstrasse 8", "Darmstadt", "DE", "64283"),
        ("Ilona Rosebrock", "Hauptstrasse 8", "Darmstadt", "DE", "64283"),
    ])
    clusters = _cluster_number_by_name(result)
    assert not clusters["REITER, LEOPOLD JUN."]
    assert not clusters["REITER, LEOPOLD"]
    assert not clusters["Jonas Rosebrock"] or clusters["Jonas Rosebrock"] != clusters["Ilona Rosebrock"]
    assert any(c["pass_type"] == "person_same_name_location_review" for c in result.get("review_candidates", []))


def test_multilingual_transliteration_matches_when_otherwise_supported():
    result = _cluster([
        ("Müller GmbH", "Straße 10", "Berlin", "DE"),
        ("Mueller GmbH", "Strasse 10", "Berlin", "DE"),
        ("Muller GmbH", "Str. 10", "Berlin", "DE"),
        ("Kühne GmbH", "Hafenstraße 1", "Hamburg", "DE"),
        ("Kuehne GmbH", "Hafenstrasse 1", "Hamburg", "DE"),
        ("Ölmühlen GmbH", "Werkstrasse 5", "Hamburg", "DE"),
        ("Oelmuehlen GmbH", "Werkstraße 5", "Hamburg", "DE"),
    ])
    clusters = _cluster_number_by_name(result)
    assert clusters["Müller GmbH"] == clusters["Mueller GmbH"] == clusters["Muller GmbH"]
    assert clusters["Kühne GmbH"] == clusters["Kuehne GmbH"]
    assert clusters["Ölmühlen GmbH"] == clusters["Oelmuehlen GmbH"]


def test_operational_status_names_route_to_review_not_main_cluster():
    result = _cluster([
        ("BLOCKED - ACME GmbH", "Main Street 1", "Berlin", "DE"),
        ("ACME GmbH", "Main Str. 1", "Berlin", "DE"),
        ("DO NOT USE Beta Ltd", "High Street 5", "London", "GB"),
        ("Beta Ltd", "High St 5", "London", "GB"),
    ])
    clusters = _cluster_number_by_name(result)
    assert not clusters["BLOCKED - ACME GmbH"] or clusters["BLOCKED - ACME GmbH"] != clusters["ACME GmbH"]
    assert not clusters["DO NOT USE Beta Ltd"] or clusters["DO NOT USE Beta Ltd"] != clusters["Beta Ltd"]
    assert any(c["pass_type"] == "operational_status_review" for c in result.get("review_candidates", []))


def test_millipore_broad_identity_is_not_blank_and_below_90():
    result = _cluster([
        ("Millipore AB, Sweden", "Frösundaviks All 1", "Solna", "SE", "169 70"),
        ("Millipore Sigma", "5485 Country Road V", "Sheboygan Falls WI", "US", "53085-2814"),
    ])
    rows = _cluster_number_by_name(result)
    if rows["Millipore AB, Sweden"] and rows["Millipore Sigma"]:
        assert rows["Millipore AB, Sweden"] == rows["Millipore Sigma"]
        pct = result["main_df"].to_dicts()[0]["Match Percentage"]
        assert 80 <= int(str(pct).rstrip("%")) <= 89
    else:
        assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))


def test_same_address_different_name_not_high_confidence():
    result = _cluster([
        ("TBWA\\WorldHealth London Limited", "2 Bankside, 90-100 Southwark Street", "London", "GB", "SE1 0SW"),
        ("Weave Health Ltd", "2 Bankside, 90-100 Southwark Street", "London", "GB", "SE1 0SW"),
    ])
    rows = _rows_by_name(result)
    c1 = rows["TBWA\\WorldHealth London Limited"]["Cluster Number"]
    c2 = rows["Weave Health Ltd"]["Cluster Number"]
    if c1 and c2 and c1 == c2:
        assert int(str(rows["TBWA\\WorldHealth London Limited"]["Match Percentage"]).rstrip("%")) < 90


def test_avisor_prefix_does_not_bridge_weber_and_schaefer_suppliers():
    result = _cluster([
        ("Adam Weber KG", "", "Berlin", "DE"),
        ("AVISOR-ADAM WEBER-669844", "", "Berlin", "DE"),
        ("AVISOR-SCHAEFER 536664", "", "Berlin", "DE"),
        ("AVISOR-Schaefer 656069", "", "Berlin", "DE"),
        ("AVISOR-WEBER 610518", "", "Berlin", "DE"),
        ("Schaefer GmbH", "", "Berlin", "DE"),
        ("Schäfer GmbH", "", "Berlin", "DE"),
        ("WEBER Industrieller Rohrleitungsbau", "", "Berlin", "DE"),
    ])
    rows = _cluster_number_by_name(result)
    assigned = [c for c in rows.values() if c]
    assert len(set(assigned)) != 1
    assert not rows["AVISOR-SCHAEFER 536664"] or rows["AVISOR-SCHAEFER 536664"] != rows["AVISOR-WEBER 610518"]


def test_air_liquide_and_air_products_are_separate_and_not_blank():
    result = _cluster([
        ("Air Liquide Germany GmbH", "", "Duesseldorf", "DE"),
        ("Air Liquide France", "", "Paris", "FR"),
        ("Air Products", "", "Allentown", "US"),
        ("Air Products Chemicals Group Europe", "", "Brussels", "BE"),
    ])
    rows = _cluster_number_by_name(result)
    assert rows["Air Liquide Germany GmbH"] == rows["Air Liquide France"]
    assert rows["Air Products"] == rows["Air Products Chemicals Group Europe"]
    assert rows["Air Liquide Germany GmbH"] != rows["Air Products"]


def test_aircom_pneumatic_pneumatik_same_address_clusters_high():
    result = _cluster([
        ("AirCom Pneumatic GmbH", "Siemensstrasse 18", "Ratingen", "DE", "40885"),
        ("AirCom Pneumatik GmbH", "Siemensstraße 18", "Ratingen", "DE", "40885"),
    ])
    rows = _rows_by_name(result)
    assert rows["AirCom Pneumatic GmbH"]["Cluster Number"] == rows["AirCom Pneumatik GmbH"]["Cluster Number"]
    assert 95 <= int(str(rows["AirCom Pneumatic GmbH"]["Match Percentage"]).rstrip("%")) <= 98


def test_airtec_pneumatic_regional_descriptor_is_broad_identity():
    result = _cluster([
        ("AIRTEC Pneumatic GmbH", "", "Kronberg", "DE"),
        ("AIRTEC Pneumatic Sweden AB", "", "OSBY", "SE"),
    ])
    rows = _rows_by_name(result)
    if rows["AIRTEC Pneumatic GmbH"]["Cluster Number"] and rows["AIRTEC Pneumatic Sweden AB"]["Cluster Number"]:
        assert rows["AIRTEC Pneumatic GmbH"]["Cluster Number"] == rows["AIRTEC Pneumatic Sweden AB"]["Cluster Number"]
        assert 80 <= int(str(rows["AIRTEC Pneumatic GmbH"]["Match Percentage"]).rstrip("%")) <= 89
    else:
        assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))


def test_ajinomoto_global_group_not_silently_missed():
    result = _cluster([
        ("Ajinomoto Co.", "", "Tokyo", "JP"),
        ("Ajinomoto Co. Inc.", "", "Tokyo", "JP"),
        ("Ajinomoto Europe Sales GmbH", "", "Hamburg", "DE"),
        ("Ajinomoto Health & Nutrition", "", "Chicago", "US"),
        ("AJINOMOTO NORTH AMERICA, INC.", "", "Raleigh", "US"),
        ("Ajinomoto Do Brasil Industria", "", "Sao Paulo", "BR"),
        ("Ajinomoto Food Europe", "", "Paris", "FR"),
        ("Ajinomoto Foods Europe S.A.S.", "1 Rue A", "Paris", "FR"),
        ("Ajinomoto Foods Europe S.A.S.", "1 Rue A", "Paris", "FR"),
        ("Ajinomoto Omnichem", "", "Wetteren", "BE"),
    ])
    rows = _rows_by_name(result)
    assert rows["Ajinomoto Do Brasil Industria"]["Cluster Number"] or any("Ajinomoto Do Brasil" in c["evidence"].get("supplier_identity_core_a", "") for c in result.get("review_candidates", []))
    assert rows["Ajinomoto Food Europe"]["Cluster Number"]
    assert rows["Ajinomoto Foods Europe S.A.S."]["Cluster Number"]
    assert 80 <= int(str(rows["Ajinomoto Food Europe"]["Match Percentage"]).rstrip("%")) <= 89


def test_3bl_media_alphanumeric_core_is_not_missed():
    result = _cluster([
        ("3BL MEDIA LLC", "2222 SEDWICK ROAD", "DURHAM", "US", "27713"),
        ("3BL Media, inc", "136 Suite 104, West Street", "Northampton", "GB", "NN5 4DP"),
    ])
    rows = _rows_by_name(result)
    c1 = rows["3BL MEDIA LLC"]["Cluster Number"]
    c2 = rows["3BL Media, inc"]["Cluster Number"]
    if c1 and c2:
        assert c1 == c2
        assert 80 <= int(str(rows["3BL MEDIA LLC"]["Match Percentage"]).rstrip("%")) <= 89
    else:
        assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))


def test_4titude_regional_descriptor_alphanumeric_core_is_not_missed():
    result = _cluster([
        ("4titude Deutschland", "Sickingenstr. 26", "Berlin", "DE", "10553"),
        ("4titude Ltd", "Surrey Hills Business Park", "Surrey", "GB", "RH5 6QT"),
    ])
    rows = _rows_by_name(result)
    c1 = rows["4titude Deutschland"]["Cluster Number"]
    c2 = rows["4titude Ltd"]["Cluster Number"]
    if c1 and c2:
        assert c1 == c2
        assert 80 <= int(str(rows["4titude Deutschland"]["Match Percentage"]).rstrip("%")) <= 89
    else:
        assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))


def test_3b_scientific_location_descriptor_alphanumeric_core_is_not_missed():
    result = _cluster([
        ("3B Scientific Corporation", "1840 Industrial Drive, Unit 160", "Libertyville", "US"),
        ("WUHAN 3B SCIENTIFIC CORP", "XU DONG ROAD", "WUHAN", "CN"),
    ])
    rows = _rows_by_name(result)
    c1 = rows["3B Scientific Corporation"]["Cluster Number"]
    c2 = rows["WUHAN 3B SCIENTIFIC CORP"]["Cluster Number"]
    if c1 and c2:
        assert c1 == c2
        assert 80 <= int(str(rows["3B Scientific Corporation"]["Match Percentage"]).rstrip("%")) <= 89
    else:
        assert any(c["pass_type"] == "distinctive_supplier_identity" for c in result.get("review_candidates", []))

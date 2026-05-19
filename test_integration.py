"""Integration test: end-to-end clustering pipeline."""

import pytest
import polars as pl
from src.main import cluster_suppliers
from src.config import ClusteringConfig
from src.output import save_audit_file, save_review_candidates
from scripts.run_cli import auto_detect_columns


def _run_supplier_records(records, config=None):
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in records],
        "Address": [r[1] if len(r) > 1 else "" for r in records],
        "City": [r[2] if len(r) > 2 else "" for r in records],
        "Country": [r[3] if len(r) > 3 else "" for r in records],
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
    return cluster_suppliers(df, mapping, config)


def _assert_no_main_clusters(result):
    assert result["main_df"]["Cluster Number"].drop_nulls().len() == 0


def test_basic_clustering():
    """Test with a small synthetic dataset."""
    df = pl.DataFrame({
        "Supplier Name": [
            "ABS Safety GmbH",
            "ABS SAFETY GMBH",
            "GreenPharma S.A.",
            "Unrelated Company Inc",
        ],
        "Address": [
            "Gewerbering 3",
            "GEWERBERING 3",
            "3 Allée du Titane",
            "999 Unknown St",
        ],
        "City": ["Kevelaer", "KEVELAER", "Orléans", "Nowhere"],
        "Country": ["DE", "DE", "FR", "US"],
        "Tax ID": ["DE233002380", "", "", ""],
        "Email": ["", "", "", ""],
    })

    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "tax_id": "Tax ID",
        "email": "Email",
    }

    result = cluster_suppliers(df, mapping)

    # Should find at least 1 cluster (the two ABS Safety records)
    assert result["stats"]["clusters_found"] >= 1

    # The two ABS records should be in same cluster
    main_df = result["main_df"]
    abs_rows = main_df.filter(pl.col("Supplier Name").str.contains("ABS"))
    clusters = abs_rows["Cluster Number"].to_list()
    assert clusters[0] is not None
    assert clusters[0] == clusters[1]


def test_tax_driven_cluster():
    """Test that same tax ID clusters even with different names."""
    df = pl.DataFrame({
        "Supplier Name": ["ICEE CO", "ICEE COMPANY", "Random Corp"],
        "Address": ["1000 Corp Ave", "2000 Other St", "999 Unknown"],
        "City": ["LA", "NY", "Nowhere"],
        "Country": ["US", "US", "US"],
        "Tax ID": ["95-2499371", "95-2499371", ""],
        "Email": ["", "", ""],
    })

    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "tax_id": "Tax ID",
        "email": "Email",
    }

    result = cluster_suppliers(df, mapping)

    # ICEE CO and ICEE COMPANY should cluster via tax
    assert result["stats"]["clusters_found"] >= 1


def test_singletons_at_bottom():
    """Test that unclustered rows keep original order under anchor-based sorting."""
    df = pl.DataFrame({
        "Supplier Name": ["Unique A", "Unique B", "Unique C"],
        "Address": ["111 St", "222 Ave", "333 Rd"],
        "City": ["A", "B", "C"],
        "Country": ["US", "US", "US"],
        "Tax ID": ["", "", ""],
        "Email": ["", "", ""],
    })

    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "tax_id": "Tax ID",
        "email": "Email",
    }

    result = cluster_suppliers(df, mapping)

    # All should be singletons
    assert result["stats"]["clusters_found"] == 0
    assert result["stats"]["singleton_rows"] == 3


def test_main_output_contract_and_anchor_ordering():
    df = pl.DataFrame({
        "Supplier Name": ["Anchor Co LLC", "Unique Vendor", "Anchor Company LLC", "Another Vendor"],
        "Address": ["100 King St", "1 Side St", "100 King Street", "2 Side St"],
        "City": ["Toronto", "Ottawa", "Toronto", "Montreal"],
        "Country": ["CA", "CA", "CA", "CA"],
        "Tax ID": ["123456789", "", "123456789", ""],
    })
    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "tax_id": "Tax ID",
    }

    result = cluster_suppliers(df, mapping)
    out = result["main_df"]

    assert out.columns == df.columns + ["Cluster Number", "Match Percentage"]
    assert out["Supplier Name"].to_list() == [
        "Anchor Co LLC",
        "Anchor Company LLC",
        "Unique Vendor",
        "Another Vendor",
    ]


def test_auto_detect_real_client_address_columns():
    mapping = auto_detect_columns([
        "ExternalId",
        "CustomExtensionFormJSON",
        "Name",
        "Address_Line1",
        "Address_City",
        "Address_CountryCode",
        "Address_PostalCode",
        "PostingBlocked",
        "Address_UnitNumber",
    ])

    assert mapping["supplier_name"] == "Name"
    assert mapping["address"] == "Address_Line1"
    assert mapping["city"] == "Address_City"
    assert mapping["country"] == "Address_CountryCode"
    assert mapping["postal_code"] == "Address_PostalCode"
    assert mapping["metadata_json_columns"] == ["CustomExtensionFormJSON"]
    assert "tax_id" not in mapping
    assert "tax_ids" not in mapping


def test_regression_last_first_people_and_surname_roots_do_not_cluster():
    """BUG 3/8: LAST,FIRST people and surname aliases must not bridge records."""
    result = _run_supplier_records([
        ("MUELLER, GUSTAV", "A Strasse 1", "Berlin", "DE", ""),
        ("MÜLLER, KARIN", "B Strasse 2", "Hamburg", "DE", ""),
        ("THUM, CINDY", "C Strasse 3", "Darmstadt", "DE", ""),
        ("Cindy Baur", "D Strasse 4", "Darmstadt", "DE", ""),
        ("Jana Zimmer", "E Strasse 5", "Munich", "DE", ""),
        ("Jill Zimmer", "F Strasse 6", "Cologne", "DE", ""),
    ])

    _assert_no_main_clusters(result)


def test_regression_new_generic_roots_do_not_bridge_main_clusters():
    """BUG 1/7: newly generic business/location terms are non-bridge tokens."""
    records = []
    for idx, token in enumerate([
        "Private",
        "Partner",
        "Industrie",
        "Product",
        "Biological",
        "Medizin",
        "Excellence",
        "Europa",
        "Medicine",
        "Executive Ingredients",
    ], 1):
        records.append((f"Alpha {token} GmbH", f"{idx} Alpha Road", "Berlin", "DE", ""))
        records.append((f"Beta {token} GmbH", f"{idx} Beta Road", "Berlin", "DE", ""))

    result = _run_supplier_records(records)

    _assert_no_main_clusters(result)


def test_regression_compact_single_token_aliases_are_not_main_bridges(tmp_path):
    """BUG 2/8: compact one-word aliases such as Alfa/Meta/Zimmer are risky."""
    brand_file = tmp_path / "known_brand_families.csv"
    brand_file.write_text(
        "keyword\n"
        "Alfa\n"
        "Meta\n"
        "Zimmer\n"
        "red\n"
        "virgin\n"
        "Dominion\n"
        "Next\n"
        "Berry\n"
        "Dollar\n",
        encoding="utf-8",
    )
    config = ClusteringConfig(known_brand_families_file=str(brand_file))
    result = _run_supplier_records([
        ("Alfa Medical GmbH", "1 Clinic Road", "Berlin", "DE", ""),
        ("Alfa Logistics GmbH", "2 Cargo Road", "Hamburg", "DE", ""),
        ("Meta Qingdao Trading Co Ltd", "3 Port Road", "Qingdao", "CN", ""),
        ("Meta Analytics GmbH", "4 Data Road", "Munich", "DE", ""),
        ("Jana Zimmer", "5 Person Road", "Frankfurt", "DE", ""),
        ("Zimmer Medizin GmbH", "6 Device Road", "Cologne", "DE", ""),
        ("Red Technologies Ltd", "7 Tech Road", "London", "GB", ""),
        ("Red Pharma Ltd", "8 Pharma Road", "Oxford", "GB", ""),
    ], config)

    _assert_no_main_clusters(result)


def test_regression_weak_review_edges_do_not_get_main_cluster_numbers():
    """BUG 6: weak/review-only clusters are audit candidates, not main clusters."""
    result = _run_supplier_records([
        ("Cap Gemini France", "", "Paris", "FR", ""),
        ("Capgemini Technology Services", "", "Mumbai", "IN", ""),
    ])

    _assert_no_main_clusters(result)
    assert result["stats"]["clusters_found"] == 0
    assert len(result.get("review_candidates", [])) >= 1


def test_exact_same_distinctive_supplier_name_different_address_can_cluster_as_group_identity():
    """Broad same-core identity now clusters below 90 unless strong evidence exists."""
    result = _run_supplier_records([
        ("Acme Precision GmbH", "1 North Road", "Berlin", "DE", ""),
        ("ACME PRECISION GMBH", "99 South Road", "Munich", "DE", ""),
    ])

    assert result["stats"]["clusters_found"] == 1
    rows = result["main_df"].to_dicts()
    assert rows[0]["Cluster Number"] == rows[1]["Cluster Number"]
    assert rows[0]["Match Percentage"] in {"80%", "81%", "82%", "83%", "84%", "85%", "86%", "87%", "88%", "89%"}


def test_exact_same_normalized_name_same_address_still_clusters():
    result = _run_supplier_records([
        ("Acme Precision GmbH", "1 North Road", "Berlin", "DE", ""),
        ("ACME PRECISION AG", "1 North Rd", "Berlin", "DE", ""),
    ])

    assert result["stats"]["clusters_found"] == 1
    clusters = result["main_df"]["Cluster Number"].drop_nulls().unique().to_list()
    assert len(clusters) == 1


def test_distinctive_core_now_main_clusters_instead_of_rare_token_only_review():
    result = _run_supplier_records([
        ("Quorvanta Services GmbH", "", "Berlin", "DE", ""),
        ("Quorvanta Solutions GmbH", "", "Munich", "DE", ""),
    ], ClusteringConfig(rare_token_max_document_fraction=1.0))

    assert result["stats"]["clusters_found"] == 1
    assert result["stats"]["pass_type_counts"].get("distinctive_supplier_identity", 0) == 1


def test_review_candidate_output_is_priority_sorted_and_enriched(tmp_path):
    result = _run_supplier_records([
        ("Cap Gemini France", "", "Paris", "FR", ""),
        ("Capgemini Technology Services", "", "Mumbai", "IN", ""),
        ("Quorvanta Services GmbH", "", "Berlin", "DE", ""),
        ("Quorvanta Solutions GmbH", "", "Munich", "DE", ""),
    ])
    rows_dict = {row["row_id"]: row for row in result["preprocessed_df"].iter_rows(named=True)}
    review_path = tmp_path / "review.csv"

    save_review_candidates(result["review_candidates"], rows_dict, str(review_path))
    review_df = pl.read_csv(review_path)

    expected_columns = {
        "priority",
        "row_id_1",
        "row_id_2",
        "supplier_name_1",
        "supplier_name_2",
        "reason",
        "score",
        "suggested_action",
        "evidence_json",
        "why_not_auto_clustered",
        "shared_discriminative_token",
        "risky_token",
        "shared_support_field",
        "support_field_value",
        "support_field_strength",
        "review_group_key",
    }
    assert expected_columns.issubset(set(review_df.columns))
    assert review_df.sort(["review_group_key", "priority"], descending=[False, True]).height == review_df.height


def test_audit_output_contains_edge_evidence_for_main_clusters(tmp_path):
    result = _run_supplier_records([
        ("Anchor Co LLC", "100 King St", "Toronto", "CA", "", "123456789"),
        ("Anchor Company LLC", "100 King Street", "Toronto", "CA", "", "123456789"),
    ])
    audit_path = tmp_path / "audit.csv"

    save_audit_file(
        result["audit_data"],
        str(audit_path),
        result["preprocessed_df"],
        result["cluster_map"],
        result["merger"],
    )
    audit_df = pl.read_csv(audit_path)

    assert {"cluster_number", "row_id_1", "row_id_2", "edge_pass_type", "evidence_json", "guardrail_applied"}.issubset(set(audit_df.columns))


def _run_support_records(records, config=None):
    df = pl.DataFrame({
        "Supplier Name": [r.get("name", "") for r in records],
        "Address": [r.get("address", "") for r in records],
        "City": [r.get("city", "") for r in records],
        "Country": [r.get("country", "") for r in records],
        "Postal Code": [r.get("postal", "") for r in records],
        "Tax ID": [r.get("tax", "") for r in records],
        "Website": [r.get("website", "") for r in records],
        "Canonical": [r.get("canonical", "") for r in records],
        "Family": [r.get("family", "") for r in records],
    })
    mapping = {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
        "tax_id": "Tax ID",
        "website": "Website",
        "canonical_name": "Canonical",
        "family_name": "Family",
    }
    return cluster_suppliers(df, mapping, config)


def _review_passes(result):
    return {candidate["pass_type"] for candidate in result.get("review_candidates", [])}


def test_person_same_street_city_mismatch_is_high_priority_review_not_main():
    result = _run_support_records([
        {"name": "Dr. Lisa Francesca Licitra", "address": "Via Sismondi 6", "city": "Milan", "country": "IT"},
        {"name": "LISA FRANCESCA LINDA LICITRA", "address": "VIA SISMONDI 6", "city": "Albettone", "country": "IT"},
    ])

    _assert_no_main_clusters(result)
    assert "person_address_city_mismatch_review" in _review_passes(result)


def test_3b_scientific_cross_country_alphanumeric_core_can_main_cluster():
    result = _run_support_records([
        {
            "name": "3B Scientific Corporation",
            "address": "1840 Industrial Drive, Unit 160",
            "city": "Libertyville",
            "country": "US",
            "canonical": "3B Scientific",
        },
        {
            "name": "WUHAN 3B SCIENTIFIC CORP",
            "address": "XU DONG ROAD",
            "city": "Wuhan",
            "country": "CN",
            "canonical": "3B Scientific",
        },
    ])

    rows = {r["Supplier Name"]: r for r in result["main_df"].to_dicts()}
    assert rows["3B Scientific Corporation"]["Cluster Number"] == rows["WUHAN 3B SCIENTIFIC CORP"]["Cluster Number"]
    assert 80 <= int(str(rows["3B Scientific Corporation"]["Match Percentage"]).rstrip("%")) <= 89


def test_meca_industrial_engraving_canonical_is_review_unless_trusted():
    records = [
        {
            "name": "Industrial Engraving",
            "address": "5324 Kunesh Rd",
            "city": "Pulaski",
            "country": "US",
            "canonical": "MECA SOLUTIONS LLC",
        },
        {
            "name": "MECA SOLUTIONS LLC",
            "address": "1281 Parkview Road",
            "city": "Green Bay",
            "country": "US",
            "canonical": "MECA SOLUTIONS LLC",
        },
    ]
    result = _run_support_records(records)
    _assert_no_main_clusters(result)
    assert "support_field_review" in _review_passes(result)

    trusted = ClusteringConfig()
    trusted.support_field_strengths = {**trusted.support_field_strengths, "canonical_name": "same_entity_name"}
    trusted_result = _run_support_records(records, trusted)
    assert trusted_result["stats"]["clusters_found"] == 1


def test_brunswick_duplicates_main_same_name_different_address_review_family():
    result = _run_support_records([
        {"name": "Brunswick Group GmbH", "address": "Mainzer Landstr 1", "city": "Frankfurt", "country": "DE", "postal": "60311", "canonical": "Brunswick Group"},
        {"name": "Brunswick Group GmbH", "address": "Mainzer Landstr. 1", "city": "Frankfurt", "country": "DE", "postal": "60311", "canonical": "Brunswick Group"},
        {"name": "Brunswick Group GmbH", "address": "Unter den Linden 2", "city": "Berlin", "country": "DE", "postal": "10117", "canonical": "Brunswick Group"},
        {"name": "Brunswick SARL", "address": "12 Rue Auber", "city": "Paris", "country": "FR", "postal": "75009", "canonical": "Brunswick Group"},
    ])
    rows = result["main_df"].sort("Supplier Name").to_dicts()
    frankfurt = [row for row in rows if row["City"] == "Frankfurt"]
    assert frankfurt[0]["Cluster Number"] is not None
    assert frankfurt[0]["Cluster Number"] == frankfurt[1]["Cluster Number"]

    berlin = [row for row in rows if row["City"] == "Berlin"][0]
    sarl = [row for row in rows if "SARL" in row["Supplier Name"]][0]
    assert berlin["Cluster Number"] is not None
    assert sarl["Cluster Number"] is not None
    assert berlin["Cluster Number"] == frankfurt[0]["Cluster Number"]
    assert sarl["Cluster Number"] == frankfurt[0]["Cluster Number"]


def test_escriba_family_field_is_review_support_not_main_by_default():
    result = _run_support_records([
        {"name": "E&E information consultants AG", "address": "Berlin Office", "city": "Berlin", "country": "DE", "canonical": "ESCRIBA AG"},
        {"name": "ESCRIBA AG", "address": "Other Berlin Office", "city": "Berlin", "country": "DE", "tax": "DE198267160", "canonical": "ESCRIBA AG"},
    ])

    _assert_no_main_clusters(result)
    assert "support_field_review" in _review_passes(result)

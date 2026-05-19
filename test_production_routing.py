"""Production routing and final user-facing score tests."""

import polars as pl

from src.llm_workflow import apply_llm_decisions
from src.main import cluster_suppliers
from src.matching import MatchResult
from src.merging import ClusterMerger
from src.scoring import calculate_user_facing_cluster_scores


def _cluster(names):
    df = pl.DataFrame({
        "Supplier Name": [r[0] for r in names],
        "Address": [r[1] if len(r) > 1 else "" for r in names],
        "City": [r[2] if len(r) > 2 else "" for r in names],
        "Country": [r[3] if len(r) > 3 else "" for r in names],
        "Postal Code": [r[4] if len(r) > 4 else "" for r in names],
    })
    return cluster_suppliers(df, {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "postal_code": "Postal Code",
    })


def _rows(result):
    return {r["Supplier Name"]: r for r in result["main_df"].to_dicts()}


def _scores(result):
    return {v for v in result["main_df"]["Match Percentage"].to_list() if v}


def test_final_output_scores_are_normalized_to_allowed_values():
    result = _cluster([
        ("Acme GmbH", "Main Strasse 1", "Berlin", "DE"),
        ("ACME GmbH", "Main Straße 1", "Berlin", "DE"),
        ("3BL MEDIA LLC", "2222 SEDWICK ROAD", "DURHAM", "US"),
        ("3BL Media, inc", "136 Suite 104, West Street", "Northampton", "GB"),
    ])
    assert _scores(result) <= {"98%", "85%", "70%"}
    assert result["main_df"].columns == [
        "Supplier Name", "Address", "City", "Country", "Postal Code",
        "Cluster Number", "Match Percentage",
    ]


def test_eastman_family_routes_to_85_and_typo_routes_to_llm_review():
    result = _cluster([
        ("EASTMAN", "", "Kingsport", "US"),
        ("Eastman Chemical (uk) Ltd", "", "Newport", "GB"),
        ("Eastman Chemical B.V.", "", "Rotterdam", "NL"),
        ("Eastman Chemical GmbH", "", "Cologne", "DE"),
        ("Eastman Chemicals", "", "Kingsport", "US"),
        ("Eastman Company", "", "Kingsport", "US"),
        ("Eastman Company UK Limited", "", "London", "GB"),
        ("Eastman Chemical Jaeger GmbH", "", "Bremen", "DE"),
        ("Eastman Fine Chem.", "", "Kingsport", "US"),
        ("Eastmann Chemical B.V.", "", "Rotterdam", "NL"),
        ("Eastman Kodak Company", "", "Rochester", "US"),
    ])
    rows = _rows(result)
    family_cluster = rows["EASTMAN"]["Cluster Number"]
    assert family_cluster
    for name in [
        "Eastman Chemical (uk) Ltd", "Eastman Chemical B.V.", "Eastman Chemical GmbH",
        "Eastman Chemicals", "Eastman Company", "Eastman Company UK Limited",
        "Eastman Chemical Jaeger GmbH", "Eastman Fine Chem.",
    ]:
        assert rows[name]["Cluster Number"] == family_cluster
        assert rows[name]["Match Percentage"] == "85%"
    assert rows["Eastman Kodak Company"]["Cluster Number"] != family_cluster
    assert any(c["pass_type"] == "distinctive_supplier_identity" and c["match_pct"] == 70.0 for c in result.get("review_candidates", []))


def test_protected_eastman_kodak_does_not_merge_with_eastman_company_by_address():
    result = _cluster([
        ("Eastman Company", "343 State Street", "Rochester", "US"),
        ("Eastman Kodak Company", "343 State Street", "Rochester", "US"),
    ])
    rows = _rows(result)
    assert not rows["Eastman Company"]["Cluster Number"] or rows["Eastman Company"]["Cluster Number"] != rows["Eastman Kodak Company"]["Cluster Number"]


def test_edi_country_slot_pattern_is_consistently_captured_not_98():
    result = _cluster([
        ("EDI Denmark to Germany", "", "Copenhagen", "DK"),
        ("EDI France to Germany", "", "Paris", "FR"),
        ("EDI Sweden to Germany", "", "Stockholm", "SE"),
        ("EDI Switzerland to Germany", "", "Zurich", "CH"),
        ("EDI Great Britain to Germany", "", "London", "GB"),
    ])
    rows = _rows(result)
    clusters = {row["Cluster Number"] for row in rows.values()}
    assert len(clusters) == 1
    assert next(iter(clusters))
    assert {row["Match Percentage"] for row in rows.values()} == {"85%"}


def test_edinburgh_city_token_alone_does_not_cluster():
    result = _cluster([
        ("Edinburgh Innovations Limited", "", "Edinburgh", "GB"),
        ("University of Edinburgh", "", "Edinburgh", "GB"),
    ])
    assert result["stats"]["clusters_found"] == 0


def test_edinburgh_institutional_shared_tax_routes_to_review_not_final_98_or_85():
    df = pl.DataFrame({
        "Supplier Name": ["Edinburgh Innovations Limited", "University of Edinburgh"],
        "Address": ["", ""],
        "City": ["Edinburgh", "Edinburgh"],
        "Country": ["GB", "GB"],
        "Tax": ["GB592950700", "GB592950700"],
    })
    result = cluster_suppliers(df, {
        "supplier_name": "Supplier Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
        "tax_id": "Tax",
    })
    rows = _rows(result)
    assert not rows["Edinburgh Innovations Limited"]["Cluster Number"]
    assert any(c["pass_type"] == "tax_exact_institutional_ecosystem_review" for c in result["review_candidates"])
    assert any(
        "tax_exact_institutional_ecosystem_review" in group.get("deterministic_pass_types", [])
        for group in result["llm_review_groups"]
    )


def test_elrig_distinctive_acronym_family_is_captured_at_85():
    result = _cluster([
        ("ELRIG (UK) Limited", "", "London", "GB"),
        ("ELRIG.de e.V.", "", "Hamburg", "DE"),
    ])
    rows = _rows(result)
    assert rows["ELRIG (UK) Limited"]["Cluster Number"] == rows["ELRIG.de e.V."]["Cluster Number"]
    assert rows["ELRIG (UK) Limited"]["Match Percentage"] == "85%"


def test_insight_and_springer_ambiguous_cores_do_not_auto_cluster_85_or_98():
    insight = _cluster([
        ("INSIGHT Health GmbH", "", "Berlin", "DE"),
        ("Insight Technology Solutions GmbH", "", "Frankfurt", "DE"),
        ("Innovation Insights", "", "London", "GB"),
        ("Management Insights AG", "", "Zurich", "CH"),
    ])
    assert not (_scores(insight) & {"98%", "85%"})

    springer = _cluster([
        ("Axel Springer SE", "", "Berlin", "DE"),
        ("Bio Springer S.A.", "", "Paris", "FR"),
        ("Springer Nature Limited", "", "London", "GB"),
        ("Springer Medizin Verlag GmbH", "", "Berlin", "DE"),
    ])
    assert not (_scores(springer) & {"98%", "85%"})


def test_tuebingen_and_gastro_do_not_cluster_by_location_or_generic_core():
    tuebingen = _cluster([
        ("Finanzamt Tuebingen", "", "Tuebingen", "DE"),
        ("Universität Tübingen", "", "Tübingen", "DE"),
        ("Universitätsklinikum Tübingen", "", "Tübingen", "DE"),
    ])
    assert tuebingen["stats"]["clusters_found"] == 0

    gastro = _cluster([
        ("GastroHero GmbH", "", "Berlin", "DE"),
        ("Gastro 24 GmbH", "", "Berlin", "DE"),
        ("Gastro Total Deutschland GmbH", "", "Berlin", "DE"),
    ])
    assert gastro["stats"]["clusters_found"] == 0


def test_llm_decision_application_approve_reject_split_merge_and_promote():
    base_map = {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4, 6: 6, 7: 6, 8: 8, 9: 8}
    scores = {0: 70.0, 2: 70.0, 4: 70.0, 6: 85.0, 8: 70.0}
    decisions = [
        {"decision": "approve", "row_ids": [0, 1], "match_percentage": 85},
        {"decision": "reject", "row_ids": [2, 3]},
        {"decision": "split", "row_ids": [4, 5], "clusters": [{"row_ids": [4, 5], "match_percentage": 85}]},
        {"decision": "merge_with_existing", "target_cluster_number": 6, "row_ids": [8, 9], "match_percentage": 85},
        {"decision": "promote_score", "row_ids": [6, 7], "match_percentage": 98},
    ]
    final_map, final_scores, counts = apply_llm_decisions(10, base_map, scores, decisions)
    assert counts["approve"] == counts["reject"] == counts["split"] == counts["merge_with_existing"] == counts["promote_score"] == 1
    assert 2 not in final_map and 3 not in final_map
    assert final_map[0] == final_map[1]
    assert final_map[4] == final_map[5]
    assert final_map[8] == final_map[9] == final_map[6] == final_map[7]
    assert set(final_scores.values()) <= {98.0, 85.0, 70.0}
    assert sorted(final_scores) == list(range(1, len(final_scores) + 1))


def test_unresolved_llm_candidates_are_hidden_when_final_output_disallows_them():
    merger = ClusterMerger(2)
    merger.add_edge(0, 1, MatchResult(
        True,
        70.0,
        "distinctive_supplier_identity",
        {"route": "LLM_REVIEW", "shared_core_tokens": ["insight"]},
        needs_review=True,
        review_reason="Ambiguous supplier core requires LLM review",
    ))
    exposed = calculate_user_facing_cluster_scores(merger, expose_unresolved_llm_candidates=True)
    hidden = calculate_user_facing_cluster_scores(merger, expose_unresolved_llm_candidates=False)
    root = next(iter(exposed))
    assert exposed[root] == 70.0
    assert hidden[root] == 0.0


def test_generic_only_accepted_edge_is_downgraded_to_llm_not_85_or_98():
    merger = ClusterMerger(2)
    merger.add_edge(0, 1, MatchResult(
        True,
        86.0,
        "distinctive_supplier_identity",
        {
            "route": "MANUAL_REVIEW",
            "shared_core_tokens": ["technology"],
            "shared_supplier_identity_core": "technology",
        },
        needs_review=True,
        review_reason="Synthetic generic-only edge",
    ))
    scores = calculate_user_facing_cluster_scores(merger, expose_unresolved_llm_candidates=True)
    assert set(scores.values()) == {70.0}

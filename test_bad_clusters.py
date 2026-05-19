"""Regression tests for known bad cluster examples.

Each test verifies that a pair of suppliers does NOT auto-cluster together
(i.e., they must be in different main clusters or remain singletons).
Two positive tests ensure valid clustering continues to work.
"""

from tests.test_precision_guardrails import row, clustered_names
from src.main import REVIEW_ONLY_PASS_TYPES
from src.matching import evaluate_pair


def _different_clusters(result, name_a: str, name_b: str) -> bool:
    """Return True if name_a and name_b ended up in different (or no) clusters."""
    rows = result["main_df"].to_dicts()
    cluster_by_name = {r["Supplier Name"]: r["Cluster Number"] for r in rows}
    ca = cluster_by_name.get(name_a)
    cb = cluster_by_name.get(name_b)
    # Both singletons — definitely separate
    if ca is None and cb is None:
        return True
    # One singleton, one clustered — separate
    if ca is None or cb is None:
        return True
    # Both clustered but in different clusters
    return ca != cb


def _same_cluster(result, name_a: str, name_b: str) -> bool:
    """Return True if name_a and name_b are in the same cluster."""
    rows = result["main_df"].to_dicts()
    cluster_by_name = {r["Supplier Name"]: r["Cluster Number"] for r in rows}
    ca = cluster_by_name.get(name_a)
    cb = cluster_by_name.get(name_b)
    if ca is None or cb is None:
        return False
    return ca == cb


# ---------------------------------------------------------------------------
# Negative tests: pairs that must NOT auto-cluster
# ---------------------------------------------------------------------------

def test_mueller_person_bridge_rejected():
    """MUELLER THOMAS + MUELLER AUTOMOTIVE GMBH must not cluster (person name bridge)."""
    result = clustered_names([
        ("MUELLER THOMAS", "", "Munich", "DE", ""),
        ("MUELLER AUTOMOTIVE GMBH", "", "Munich", "DE", ""),
    ])
    assert _different_clusters(result, "MUELLER THOMAS", "MUELLER AUTOMOTIVE GMBH"), (
        "MUELLER THOMAS and MUELLER AUTOMOTIVE GMBH should NOT be in the same cluster"
    )


def test_cindy_thum_person_bridge_rejected():
    """CINDY THUM + THUM LOGISTIK GMBH must not cluster (first name detected)."""
    result = clustered_names([
        ("CINDY THUM", "", "Berlin", "DE", ""),
        ("THUM LOGISTIK GMBH", "", "Berlin", "DE", ""),
    ])
    assert _different_clusters(result, "CINDY THUM", "THUM LOGISTIK GMBH"), (
        "CINDY THUM and THUM LOGISTIK GMBH should NOT be in the same cluster"
    )


def test_private_generic_bridge_rejected():
    """'private' is a generic token — PRIVATE EQUITY PARTNERS and PRIVATE LABEL SOLUTIONS must not cluster."""
    result = clustered_names([
        ("PRIVATE EQUITY PARTNERS", "", "London", "GB", ""),
        ("PRIVATE LABEL SOLUTIONS", "", "London", "GB", ""),
    ])
    assert _different_clusters(result, "PRIVATE EQUITY PARTNERS", "PRIVATE LABEL SOLUTIONS"), (
        "PRIVATE EQUITY PARTNERS and PRIVATE LABEL SOLUTIONS should NOT be in the same cluster"
    )


def test_partner_generic_bridge_rejected():
    """'partner' is a generic token — PARTNER CONSULTING GROUP and PARTNER SOLUTIONS GMBH must not cluster."""
    result = clustered_names([
        ("PARTNER CONSULTING GROUP", "", "Berlin", "DE", ""),
        ("PARTNER SOLUTIONS GMBH", "", "Berlin", "DE", ""),
    ])
    assert _different_clusters(result, "PARTNER CONSULTING GROUP", "PARTNER SOLUTIONS GMBH"), (
        "PARTNER CONSULTING GROUP and PARTNER SOLUTIONS GMBH should NOT be in the same cluster"
    )


def test_industrie_generic_bridge_rejected():
    """'industrie' is a generic token — INDUSTRIE TECHNIK AG and INDUSTRIE LOGISTIK GMBH must not cluster."""
    result = clustered_names([
        ("INDUSTRIE TECHNIK AG", "", "Hamburg", "DE", ""),
        ("INDUSTRIE LOGISTIK GMBH", "", "Hamburg", "DE", ""),
    ])
    assert _different_clusters(result, "INDUSTRIE TECHNIK AG", "INDUSTRIE LOGISTIK GMBH"), (
        "INDUSTRIE TECHNIK AG and INDUSTRIE LOGISTIK GMBH should NOT be in the same cluster"
    )


def test_product_generic_bridge_rejected():
    """'product' is a generic token — PRODUCT INNOVATIONS INC and PRODUCT SERVICES LTD must not cluster."""
    result = clustered_names([
        ("PRODUCT INNOVATIONS INC", "", "New York", "US", ""),
        ("PRODUCT SERVICES LTD", "", "New York", "US", ""),
    ])
    assert _different_clusters(result, "PRODUCT INNOVATIONS INC", "PRODUCT SERVICES LTD"), (
        "PRODUCT INNOVATIONS INC and PRODUCT SERVICES LTD should NOT be in the same cluster"
    )


def test_biological_generic_bridge_rejected():
    """'biological' is a generic token — BIOLOGICAL RESEARCH LABS and BIOLOGICAL SCIENCES CORP must not cluster."""
    result = clustered_names([
        ("BIOLOGICAL RESEARCH LABS", "", "Boston", "US", ""),
        ("BIOLOGICAL SCIENCES CORP", "", "Boston", "US", ""),
    ])
    assert _different_clusters(result, "BIOLOGICAL RESEARCH LABS", "BIOLOGICAL SCIENCES CORP"), (
        "BIOLOGICAL RESEARCH LABS and BIOLOGICAL SCIENCES CORP should NOT be in the same cluster"
    )


def test_medizin_generic_bridge_rejected():
    """'medizin' is a generic token — MEDIZIN TECHNIK GMBH and MEDIZIN SOLUTIONS AG must not cluster."""
    result = clustered_names([
        ("MEDIZIN TECHNIK GMBH", "", "Munich", "DE", ""),
        ("MEDIZIN SOLUTIONS AG", "", "Munich", "DE", ""),
    ])
    assert _different_clusters(result, "MEDIZIN TECHNIK GMBH", "MEDIZIN SOLUTIONS AG"), (
        "MEDIZIN TECHNIK GMBH and MEDIZIN SOLUTIONS AG should NOT be in the same cluster"
    )


def test_excellence_generic_bridge_rejected():
    """'excellence' is a generic token — EXCELLENCE CONSULTING and EXCELLENCE SERVICES GMBH must not cluster."""
    result = clustered_names([
        ("EXCELLENCE CONSULTING", "", "Zurich", "CH", ""),
        ("EXCELLENCE SERVICES GMBH", "", "Zurich", "CH", ""),
    ])
    assert _different_clusters(result, "EXCELLENCE CONSULTING", "EXCELLENCE SERVICES GMBH"), (
        "EXCELLENCE CONSULTING and EXCELLENCE SERVICES GMBH should NOT be in the same cluster"
    )


def test_europa_location_bridge_rejected():
    """'europa' is a location token — EUROPA LOGISTICS AG and EUROPA TRANSPORT GMBH must not cluster."""
    result = clustered_names([
        ("EUROPA LOGISTICS AG", "", "Vienna", "AT", ""),
        ("EUROPA TRANSPORT GMBH", "", "Vienna", "AT", ""),
    ])
    assert _different_clusters(result, "EUROPA LOGISTICS AG", "EUROPA TRANSPORT GMBH"), (
        "EUROPA LOGISTICS AG and EUROPA TRANSPORT GMBH should NOT be in the same cluster"
    )


def test_medicine_generic_bridge_rejected():
    """'medicine' is a generic token — MEDICINE RESEARCH INC and MEDICINE SOLUTIONS CORP must not cluster."""
    result = clustered_names([
        ("MEDICINE RESEARCH INC", "", "Chicago", "US", ""),
        ("MEDICINE SOLUTIONS CORP", "", "Chicago", "US", ""),
    ])
    assert _different_clusters(result, "MEDICINE RESEARCH INC", "MEDICINE SOLUTIONS CORP"), (
        "MEDICINE RESEARCH INC and MEDICINE SOLUTIONS CORP should NOT be in the same cluster"
    )


def test_alfa_broad_unsafe_alias_rejected():
    """'alfa' is a risky/broad alias — ALFA CHEMICALS and ALFA ROMEO DEALER must not auto-cluster."""
    result = clustered_names([
        ("ALFA CHEMICALS", "", "Milan", "IT", ""),
        ("ALFA ROMEO DEALER", "", "Milan", "IT", ""),
    ])
    assert _different_clusters(result, "ALFA CHEMICALS", "ALFA ROMEO DEALER"), (
        "ALFA CHEMICALS and ALFA ROMEO DEALER should NOT be in the same cluster"
    )


def test_meta_broad_unsafe_alias_rejected():
    """'meta' is a risky/broad alias — META PLATFORMS INC and META ANALYTICS GMBH must not auto-cluster."""
    result = clustered_names([
        ("META PLATFORMS INC", "", "Menlo Park", "US", ""),
        ("META ANALYTICS GMBH", "", "Berlin", "DE", ""),
    ])
    assert _different_clusters(result, "META PLATFORMS INC", "META ANALYTICS GMBH"), (
        "META PLATFORMS INC and META ANALYTICS GMBH should NOT be in the same cluster"
    )


def test_adm_acronym_is_review_only():
    """ADM LOGISTICS and ADM SUPPLY CHAIN: acronym match must be review-only, not auto-clustered."""
    result = clustered_names([
        ("ADM LOGISTICS", "", "Chicago", "US", ""),
        ("ADM SUPPLY CHAIN", "", "Chicago", "US", ""),
    ])
    # Either not clustered at all, or if matched it must be via a review-only pass type
    rows = result["main_df"].to_dicts()
    cluster_by_name = {r["Supplier Name"]: r["Cluster Number"] for r in rows}
    ca = cluster_by_name.get("ADM LOGISTICS")
    cb = cluster_by_name.get("ADM SUPPLY CHAIN")
    # Must NOT be auto-clustered together in main output
    assert ca != cb or ca is None or cb is None, (
        "ADM LOGISTICS and ADM SUPPLY CHAIN should NOT be auto-clustered in the main output"
    )


def test_zimmer_unsafe_alias_rejected():
    """'zimmer' is a risky alias — ZIMMER BIOMET and ZIMMER GMBH must not auto-cluster."""
    result = clustered_names([
        ("ZIMMER BIOMET", "", "Warsaw", "US", ""),
        ("ZIMMER GMBH", "", "Winterthur", "CH", ""),
    ])
    assert _different_clusters(result, "ZIMMER BIOMET", "ZIMMER GMBH"), (
        "ZIMMER BIOMET and ZIMMER GMBH should NOT be in the same cluster"
    )


# ---------------------------------------------------------------------------
# Positive tests: pairs that SHOULD cluster correctly
# ---------------------------------------------------------------------------

def test_same_name_same_address_clusters_correctly():
    """ACME GMBH at exact same address should cluster via name_address_exact."""
    result = clustered_names([
        ("ACME GMBH", "Hauptstrasse 1", "Berlin", "DE", ""),
        ("ACME GMBH", "Hauptstrasse 1", "Berlin", "DE", ""),
    ])
    assert _same_cluster(result, "ACME GMBH", "ACME GMBH"), (
        "ACME GMBH at the same address should be in the same cluster"
    )
    assert result["stats"]["clusters_found"] == 1


def test_tax_exact_clusters_correctly():
    """Two rows sharing the same tax ID must cluster via tax_exact."""
    result = clustered_names([
        # (name, address, city, country, postal, tax)
        ("Alpha GmbH", "Street A 1", "Frankfurt", "DE", "60311", "DE123456789"),
        ("Alpha GmbH Zweigstelle", "Street B 2", "Frankfurt", "DE", "60311", "DE123456789"),
    ])
    assert _same_cluster(result, "Alpha GmbH", "Alpha GmbH Zweigstelle"), (
        "Rows sharing a tax ID should be in the same cluster"
    )
    assert result["stats"]["clusters_found"] == 1

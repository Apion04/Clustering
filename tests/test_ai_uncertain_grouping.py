import polars as pl
from src.main import cluster_suppliers
from src.config import ClusteringConfig
from src.ai_review import AIReviewResult


def test_ai_uncertain_goes_to_review_not_main_cluster(monkeypatch):
    df = pl.DataFrame({
        "Name": ["PPC", "Potasse et Produits Chimiques SAS"],
        "Website": ["ppc-example.com", "ppc-example.com"],
        "Address": ["", ""],
        "Country": ["FR", "FR"],
    })
    def fake_ai(row_a, row_b, config=None):
        return AIReviewResult("UNCERTAIN", 67.0, "Plausible acronym/family bridge but not proven", "family")
    monkeypatch.setattr("src.main.ai_review_pair", fake_ai)
    cfg = ClusteringConfig(ai_review_enabled=True, ai_uncertain_cluster_enabled=True, ai_uncertain_match_pct=68.0)
    result = cluster_suppliers(df, {"supplier_name": "Name", "website": "Website", "address": "Address", "country": "Country"}, cfg)
    stats = result["stats"]
    assert stats["clusters_found"] == 0
    assert stats["review_candidate_pairs"] == 1
    assert result["review_candidates"][0]["evidence"]["ai_decision"] == "UNCERTAIN_GROUPED"


def test_ai_unavailable_does_not_group_llm_only_acronym_candidate(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = pl.DataFrame({
        "Name": ["ABC", "Alpha Beta Consulting"],
        "Address": ["", ""],
        "City": ["Toronto", "Toronto"],
        "Country": ["CA", "CA"],
    })
    cfg = ClusteringConfig(ai_review_enabled=True)

    result = cluster_suppliers(df, {
        "supplier_name": "Name",
        "address": "Address",
        "city": "City",
        "country": "Country",
    }, cfg)

    assert result["stats"]["clusters_found"] == 0

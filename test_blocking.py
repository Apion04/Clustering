import polars as pl

from src.blocking import generate_candidate_pairs
from src.config import ClusteringConfig


def test_large_exact_tax_block_uses_linear_star_pairs():
    df = pl.DataFrame({
        "row_id": [0, 1, 2, 3],
        "tax_norm": ["de123456789"] * 4,
        "tax_loose_norm": ["123456789"] * 4,
        "name_norm": ["supplier a", "supplier b", "supplier c", "supplier d"],
        "domain": [""] * 4,
        "is_generic_domain": [False] * 4,
        "addr_norm": [""] * 4,
        "postal_norm": [""] * 4,
        "country_norm": ["DE"] * 4,
        "city_norm": [""] * 4,
        "root_brand": [""] * 4,
        "name_phonetic": [""] * 4,
        "name_token_sort": [""] * 4,
    })
    cfg = ClusteringConfig(max_candidates_per_block=1, max_total_candidate_pairs=10)

    pairs = generate_candidate_pairs(df, cfg)

    tax_pairs = pairs.filter(pl.col("block_type") == "tax")
    assert tax_pairs.height == 3
    assert set(tax_pairs["row_a"].to_list()) == {0}
    assert set(tax_pairs["row_b"].to_list()) == {1, 2, 3}


def test_blank_tax_values_do_not_create_shared_tax_block():
    df = pl.DataFrame({
        "row_id": [0, 1, 2],
        "tax_norm": ["", "", ""],
        "tax_loose_norm": ["", "", ""],
        "name_norm": ["alpha", "bravo", "charlie"],
        "domain": ["", "", ""],
        "is_generic_domain": [False, False, False],
        "addr_norm": ["", "", ""],
        "postal_norm": ["", "", ""],
        "country_norm": ["US", "US", "US"],
        "city_norm": ["", "", ""],
        "root_brand": ["alpha", "bravo", "charlie"],
        "name_phonetic": ["", "", ""],
        "name_token_sort": ["alpha", "bravo", "charlie"],
    })

    pairs = generate_candidate_pairs(df, ClusteringConfig())

    assert pairs.filter(pl.col("block_type").is_in(["tax", "tax_loose"])).height == 0

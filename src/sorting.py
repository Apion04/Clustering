"""Output ordering for reviewer files.

Anchor-based ordering keeps the first occurrence of each cluster in its original
position and pulls later matching rows directly below that first occurrence.
It avoids dumping every cluster at the top, preserving reviewer context.
"""

from __future__ import annotations

from typing import Dict, List, Set
from collections import defaultdict
import polars as pl


def sort_by_clusters(df: pl.DataFrame, cluster_map: Dict[int, int], match_pcts: Dict[int, float]) -> pl.DataFrame:
    """Apply anchor-based output ordering.

    Rules:
    1. First occurrence of each cluster stays where it originally appeared.
    2. Other rows in that cluster are moved immediately under the anchor.
    3. Rows moved under an anchor are removed from their later positions.
    4. Non-clustered rows keep their original relative order.
    5. Main output keeps only original client columns + Cluster Number + Match Percentage.
    """
    unique_roots = sorted(set(cluster_map.values()), key=lambda r: min([rid for rid, root in cluster_map.items() if root == r]))
    cluster_number_map = {root: i + 1 for i, root in enumerate(unique_roots)}

    cluster_rows: Dict[int, List[int]] = defaultdict(list)
    for row_id, root in cluster_map.items():
        cluster_rows[root].append(row_id)
    for root in cluster_rows:
        cluster_rows[root].sort()

    row_to_cluster_number = {rid: cluster_number_map[root] for rid, root in cluster_map.items()}
    row_to_match_pct = {rid: match_pcts.get(root, 0.0) for rid, root in cluster_map.items()}
    anchor_for_root = {root: rows[0] for root, rows in cluster_rows.items() if rows}
    anchor_rows = set(anchor_for_root.values())
    moved_rows = {rid for rows in cluster_rows.values() for rid in rows[1:]}

    ordered_row_ids: List[int] = []
    emitted: Set[int] = set()
    all_row_ids = df.get_column("row_id").to_list()

    for rid in all_row_ids:
        if rid in emitted:
            continue
        if rid in moved_rows:
            # This row will be emitted under its cluster anchor.
            continue
        ordered_row_ids.append(rid)
        emitted.add(rid)
        if rid in anchor_rows:
            root = cluster_map.get(rid)
            for other in cluster_rows.get(root, [])[1:]:
                if other not in emitted:
                    ordered_row_ids.append(other)
                    emitted.add(other)

    # Safety: append any row not emitted, preserving original order.
    for rid in all_row_ids:
        if rid not in emitted:
            ordered_row_ids.append(rid)
            emitted.add(rid)

    order_df = pl.DataFrame({"row_id": ordered_row_ids, "_anchor_order": list(range(len(ordered_row_ids)))})
    out = df.join(order_df, on="row_id", how="left").sort("_anchor_order")

    out = out.with_columns([
        pl.col("row_id").map_elements(lambda x: row_to_cluster_number.get(x, None), return_dtype=pl.Int64).alias("Cluster Number"),
        pl.col("row_id").map_elements(lambda x: row_to_match_pct.get(x, 0.0), return_dtype=pl.Float64).alias("Match Percentage"),
    ])

    out = out.with_columns([
        pl.when(pl.col("Cluster Number").is_not_null())
        .then(pl.col("Match Percentage").round(0).cast(pl.Int64).cast(pl.Utf8) + "%")
        .otherwise(pl.lit(""))
        .alias("Match Percentage")
    ])

    internal_cols = [
        "row_id", "_anchor_order", "orig_supplier_name", "orig_address", "orig_city", "orig_country", "orig_email", "orig_website", "orig_domain", "orig_secondary_names",
        "orig_json_secondary_names", "json_secondary_names_norm",
        "name_norm", "name2_norm", "name3_norm", "name4_norm", "name5_norm", "name6_norm", "name7_norm",
        "addr_norm", "city_norm", "country_norm", "postal_norm", "tax_norm", "tax_loose_norm",
        "domain_from_domain", "domain_from_email", "domain_from_website", "domain", "is_generic_domain",
        "name_token_sort", "name_phonetic", "root_brand", "supplier_identity_core", "has_legal_suffix", "has_company_keyword", "is_hospitality",
        "person_name_norm", "is_likely_individual", "legal_suffixes_found",
        "known_brand_family_ids", "known_brand_family_safe_ids", "known_brand_family_risky_ids", "known_brand_alias_hits",
        "idf_discriminative_tokens",
        "support_fields_json", "support_same_entity_id_values", "support_same_entity_name_values",
        "support_family_values", "support_domain_values", "support_review_values", "orig_support_fields",
        "has_operational_status_hint",
    ]
    return out.drop([c for c in internal_cols if c in out.columns])

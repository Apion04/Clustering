"""Blocking module: generate candidate pairs without O(n²) comparison.

The implementation uses Polars group_by/explode style logic and max block-size
filtering. It intentionally skips broad blocks that create too many candidate
pairs, because those blocks usually generate false positives and timeouts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from itertools import combinations
from collections import defaultdict
import time
import polars as pl

from src.config import (
    ClusteringConfig,
    COMMON_FIRST_NAMES,
    GENERIC_ROOT_TOKENS,
    LOCATION_ROOT_TOKENS,
    SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS,
    SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS,
    TRUSTED_SUPPLIER_IDENTITY_CORES,
)
from src.legal_keywords import strip_legal_suffixes as _lsuf_strip


def _add_block(blocks: List[Tuple[int, str, str]], row_id: int, key: str, block_type: str):
    if key and key not in {"_", "", "None"}:
        blocks.append((int(row_id), key, block_type))


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(schema={"row_a": pl.Int64, "row_b": pl.Int64, "shared_block": pl.Utf8, "block_type": pl.Utf8})


def _pair_count(n: int) -> int:
    return n * (n - 1) // 2


def _block_preview(block_key: str) -> str:
    if block_key.startswith(("TAX_", "TAXLOOSE_")):
        prefix, _, value = block_key.partition("_")
        return f"{prefix}_{value[:4]}...{value[-3:]}" if len(value) > 8 else f"{prefix}_***"
    return block_key[:100]


def _set_blocking_diagnostics(config: ClusteringConfig, diagnostics: Dict[str, Any]) -> None:
    setattr(config, "_blocking_diagnostics", diagnostics)


def _safe_distinctive_core_for_block(core: str, *, is_individual: bool, is_hospitality: bool) -> bool:
    if is_individual or is_hospitality:
        return False
    tokens = [t for t in str(core or "").split() if t]
    if not tokens:
        return False
    if all(t in GENERIC_ROOT_TOKENS or t in LOCATION_ROOT_TOKENS or t in COMMON_FIRST_NAMES for t in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS:
            return True
        if token in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and token not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS:
            return False
        if token.isdigit():
            return False
        if any(ch.isdigit() for ch in token):
            # Stable alphanumeric supplier cores such as 3B, 3BL, 4titude,
            # 4flow, or 7layers are valid brand identities. Pure numeric
            # references and operational IDs are removed upstream and stay
            # blocked here by the digit-only check above.
            return any(ch.isalpha() for ch in token) and len(token) >= 2
        return len(token) >= 4
    return True


def generate_candidate_pairs(df: pl.DataFrame, config: ClusteringConfig = None) -> pl.DataFrame:
    """Generate candidate pairs using bounded multi-key blocking.

    Returns: row_a, row_b, shared_block, block_type
    """
    if config is None:
        config = ClusteringConfig()

    key_start = time.perf_counter()
    blocks: List[Tuple[int, str, str]] = []
    for row in df.iter_rows(named=True):
        row_id = row["row_id"]
        name_norm = row.get("name_norm", "") or ""
        domain = row.get("domain", "") or ""
        is_generic = bool(row.get("is_generic_domain", False))
        is_hospitality = bool(row.get("is_hospitality", False))
        addr_norm = row.get("addr_norm", "") or ""
        postal_norm = row.get("postal_norm", "") or ""
        country_norm = row.get("country_norm", "") or ""
        city_norm = row.get("city_norm", "") or ""
        root_brand = row.get("root_brand", "") or ""
        supplier_identity_core = row.get("supplier_identity_core", "") or ""
        name_phonetic = row.get("name_phonetic", "") or ""
        name_token_sort = row.get("name_token_sort", "") or ""
        first_token = name_norm.split()[0] if name_norm else ""
        is_individual = bool(row.get("is_likely_individual", False))
        person_name_norm = row.get("person_name_norm", "") or ""
        known_brand_family_ids = row.get("known_brand_family_ids", "") or ""
        support_same_entity_id_values = row.get("support_same_entity_id_values", "") or ""
        support_same_entity_name_values = row.get("support_same_entity_name_values", "") or ""
        support_family_values = row.get("support_family_values", "") or ""
        support_review_values = row.get("support_review_values", "") or ""

        for t in str(row.get("tax_norm", "") or "").split("|"):
            if t:
                _add_block(blocks, row_id, f"TAX_{t}", "tax")
        for t in str(row.get("tax_loose_norm", "") or "").split("|"):
            if t:
                _add_block(blocks, row_id, f"TAXLOOSE_{t}", "tax_loose")
        for value in str(support_same_entity_id_values).split("|"):
            if value:
                _add_block(blocks, row_id, f"SUPID_{value}", "support_same_entity_id")
        for value in str(support_same_entity_name_values).split("|"):
            if value:
                _add_block(blocks, row_id, f"SUPNAME_{value}", "support_same_entity_name")
        for value in str(support_family_values).split("|"):
            if value:
                _add_block(blocks, row_id, f"SUPFAM_{value}", "support_family")
        for value in str(support_review_values).split("|"):
            if value:
                _add_block(blocks, row_id, f"SUPREV_{value}", "support_review")

        if is_individual and person_name_norm:
            # Person records are intentionally strict: compare exact cleaned full names
            # and same-address candidates, but do not create broad first-name/root/phonetic blocks.
            _add_block(blocks, row_id, f"PER_{person_name_norm}", "person_full_name")

        first_token_is_weak = first_token in COMMON_FIRST_NAMES or first_token in GENERIC_ROOT_TOKENS or first_token in LOCATION_ROOT_TOKENS
        root_is_weak = not root_brand or root_brand in COMMON_FIRST_NAMES or root_brand in GENERIC_ROOT_TOKENS or root_brand in LOCATION_ROOT_TOKENS
        allow_brand_name_blocks = not is_individual and not is_hospitality and not first_token_is_weak and not root_is_weak
        if name_norm and allow_brand_name_blocks:
            _add_block(blocks, row_id, f"N5_{name_norm[:5]}", "name_prefix")
            if len(name_norm) >= 8:
                _add_block(blocks, row_id, f"N8_{name_norm[:8]}", "name_prefix")
        if name_phonetic and len(name_phonetic) >= 4 and allow_brand_name_blocks:
            _add_block(blocks, row_id, f"PHO_{name_phonetic}", "phonetic")
        if name_token_sort and len(name_token_sort) >= 4 and allow_brand_name_blocks:
            _add_block(blocks, row_id, f"TOK_{name_token_sort}", "token_sort")
        if domain and not is_generic:
            _add_block(blocks, row_id, f"DOM_{domain}", "domain")
        if addr_norm:
            _add_block(blocks, row_id, f"ADR_{addr_norm}", "address")
        # LSUF_ block: ensures legal-suffix variants of weak-root names (e.g.
        # "B LAB CO." / "B LAB COMPANY") become candidate pairs even when
        # root_is_weak suppresses N5/N8/phonetic/token_sort blocks.  The block
        # key is the legal-stripped form (capped to 12 chars) so that PASS 0B
        # in evaluate_pair can confirm address identity before clustering.
        if root_is_weak and name_norm and not is_individual and not is_hospitality:
            _lsuf_name = _lsuf_strip(name_norm)[0] or ""
            if _lsuf_name and _lsuf_name != name_norm and len(_lsuf_name.split()) >= 2:
                _add_block(blocks, row_id, f"LSUF_{_lsuf_name[:12]}", "legal_suffix_variant")
        if postal_norm and country_norm and not is_individual:
            _add_block(blocks, row_id, f"PC_{postal_norm}_{country_norm}", "postal_country")
        # City-country alone is too broad. Use it only with a non-generic, non-hospitality root.
        root_is_safe = bool(
            root_brand
            and root_brand not in GENERIC_ROOT_TOKENS
            and root_brand not in LOCATION_ROOT_TOKENS
            and root_brand not in COMMON_FIRST_NAMES
            and not is_hospitality
            and not is_individual
        )
        if city_norm and country_norm and root_is_safe and len(root_brand) >= 4:
            _add_block(blocks, row_id, f"CCR_{city_norm}_{country_norm}_{root_brand}", "city_country_root")
        if root_is_safe and len(root_brand) >= 4:
            _add_block(blocks, row_id, f"ROOT_{root_brand}", "root_brand")
        if _safe_distinctive_core_for_block(supplier_identity_core, is_individual=is_individual, is_hospitality=is_hospitality):
            _add_block(blocks, row_id, f"DCORE_{supplier_identity_core}", "distinctive_core")
            for token in supplier_identity_core.split():
                if token in TRUSTED_SUPPLIER_IDENTITY_CORES:
                    _add_block(blocks, row_id, f"TCORE_{token}", "trusted_core")
        for family_id in str(known_brand_family_ids).split("|"):
            if family_id:
                _add_block(blocks, row_id, f"KBF_{family_id}", "known_brand_family")
        # Location-core block: groups rows that share the same brand core after
        # stripping trailing location/branch modifiers (e.g. "bfi canada" for both
        # "BFI CANADA-CALGARY" and "BFI CANADA-TORONTO").
        name_location_core = row.get("name_location_core", "") or ""
        if (
            name_location_core
            and name_location_core != name_norm
            and not is_individual
            and not is_hospitality
        ):
            nlc_tokens = [t for t in name_location_core.split() if t]
            nlc_distinctive = any(
                len(t) >= 3
                and t not in GENERIC_ROOT_TOKENS
                and t not in LOCATION_ROOT_TOKENS
                and t not in COMMON_FIRST_NAMES
                for t in nlc_tokens
            )
            if nlc_distinctive:
                _add_block(blocks, row_id, f"NLOC_{name_location_core}", "location_core")

    key_seconds = time.perf_counter() - key_start

    if not blocks:
        _set_blocking_diagnostics(config, {
            "blocking_key_generation_seconds": key_seconds,
            "candidate_pair_generation_seconds": 0.0,
            "total_blocking_keys": 0,
            "unique_block_keys": 0,
            "top_20_largest_blocks": [],
            "candidate_pairs_before_caps": 0,
            "candidate_pairs_after_caps": 0,
            "candidate_pair_cap_reached": False,
            "skipped_oversized_blocks": 0,
            "skipped_blocks_by_type": {},
            "skipped_blocks_by_reason": {},
            "top_skipped_blocks": [],
            "exact_tax_star_blocks": 0,
            "exact_tax_star_edges": 0,
        })
        return _empty_candidates()

    pair_start = time.perf_counter()
    by_key: Dict[str, Dict[str, object]] = {}
    for row_id, block_key, block_type in blocks:
        if block_key not in by_key:
            by_key[block_key] = {"type": block_type, "rows": set()}
        by_key[block_key]["rows"].add(row_id)

    pairs = {}
    max_block = int(config.max_candidates_per_block)
    max_total_pairs = int(getattr(config, "max_total_candidate_pairs", 1000000))
    weak_block_max = int(getattr(config, "max_weak_block_size", 300))
    exact_tax_star_threshold = int(getattr(config, "exact_tax_star_threshold", 100))
    # Hard row cap for very broad blocks. Candidate pairs grow n*(n-1)/2.
    max_rows_by_type = {
        "tax": int(getattr(config, "max_tax_block_size", 5000)),
        "tax_loose": min(1000, weak_block_max),
        "domain": min(2000, int(getattr(config, "max_domain_block_size", 2000))),
        "address": min(1000, int(getattr(config, "max_address_block_size", 1000))),
        "token_sort": weak_block_max,
        "person_full_name": min(150, weak_block_max),
        "phonetic": weak_block_max,
        "name_prefix": weak_block_max,
        "postal_country": min(500, weak_block_max),
        "city_country_root": min(300, weak_block_max),
        "root_brand": min(150, weak_block_max),
        "distinctive_core": min(200, weak_block_max),
        "trusted_core": min(300, weak_block_max),
        "known_brand_family": min(int(getattr(config, "max_known_brand_family_block_size", 300)), weak_block_max),
        "location_core": min(150, weak_block_max),
        "legal_suffix_variant": min(50, weak_block_max),
        "support_same_entity_id": int(getattr(config, "max_tax_block_size", 5000)),
        "support_same_entity_name": min(300, weak_block_max),
        "support_family": min(300, weak_block_max),
        "support_review": min(300, weak_block_max),
    }

    block_priority = {
        "tax": 0,
        "tax_loose": 1,
        "domain": 2,
        "token_sort": 3,
        "person_full_name": 4,
        "address": 5,
        "postal_country": 6,
        "city_country_root": 7,
        "phonetic": 8,
        "name_prefix": 9,
        "root_brand": 10,
        "distinctive_core": 10,
        "trusted_core": 10,
        "known_brand_family": 11,
        "support_same_entity_id": 1,
        "support_same_entity_name": 4,
        "support_family": 11,
        "support_review": 12,
        "location_core": 10,
        "legal_suffix_variant": 6,
    }
    ordered_blocks = sorted(
        by_key.items(),
        key=lambda item: (
            block_priority.get(str(item[1]["type"]), 99),
            len(item[1]["rows"]),
            item[0],
        ),
    )
    block_summaries = [
        {
            "block_key_preview": _block_preview(block_key),
            "block_type": str(info["type"]),
            "row_count": len(info["rows"]),
            "raw_pair_count": _pair_count(len(info["rows"])),
        }
        for block_key, info in by_key.items()
        if len(info["rows"]) >= 2
    ]
    top_20_largest_blocks = sorted(block_summaries, key=lambda x: x["row_count"], reverse=True)[:20]
    candidate_pairs_before_caps = sum(item["raw_pair_count"] for item in block_summaries)
    skipped_by_type: Dict[str, int] = defaultdict(int)
    skipped_by_reason: Dict[str, int] = defaultdict(int)
    top_skipped_blocks: List[Dict[str, Any]] = []
    exact_tax_star_blocks = 0
    exact_tax_star_edges = 0
    candidate_pair_cap_reached = False

    for block_key, info in ordered_blocks:
        rows = sorted(info["rows"])
        block_type = info["type"]
        n = len(rows)
        if n < 2:
            continue
        raw_pairs = _pair_count(n)
        if block_type == "tax" and (n >= exact_tax_star_threshold or raw_pairs > max_block):
            # Exact valid tax ID is the strongest signal. For very large exact-tax
            # blocks, emit a linear star instead of skipping the block or creating
            # n² pairs.
            anchor = rows[0]
            exact_tax_star_blocks += 1
            for b in rows[1:]:
                pairs.setdefault((anchor, b), (block_key, block_type))
                exact_tax_star_edges += 1
                if len(pairs) >= max_total_pairs:
                    candidate_pair_cap_reached = True
                    break
            if len(pairs) >= max_total_pairs:
                break
            continue
        if n > max_rows_by_type.get(block_type, weak_block_max):
            skipped_by_type[str(block_type)] += 1
            skipped_by_reason["block_row_limit"] += 1
            top_skipped_blocks.append({
                "block_key_preview": _block_preview(block_key),
                "block_type": str(block_type),
                "row_count": n,
                "raw_pair_count": raw_pairs,
                "reason": "block_row_limit",
            })
            continue
        if raw_pairs > max_block:
            skipped_by_type[str(block_type)] += 1
            skipped_by_reason["block_pair_limit"] += 1
            top_skipped_blocks.append({
                "block_key_preview": _block_preview(block_key),
                "block_type": str(block_type),
                "row_count": n,
                "raw_pair_count": raw_pairs,
                "reason": "block_pair_limit",
            })
            continue
        for a, b in combinations(rows, 2):
            # Keep first shared block only. Matching function will evaluate full row evidence.
            pairs.setdefault((a, b), (block_key, block_type))
            # Global safety cap for large files (100k-200k rows). This prevents memory/time blowups
            # when client data contains unexpectedly broad blocking keys.
            if len(pairs) >= max_total_pairs:
                candidate_pair_cap_reached = True
                break
        if len(pairs) >= max_total_pairs:
            break

    pair_seconds = time.perf_counter() - pair_start
    top_skipped_blocks = sorted(top_skipped_blocks, key=lambda x: x["row_count"], reverse=True)[:20]
    diagnostics = {
        "blocking_key_generation_seconds": key_seconds,
        "candidate_pair_generation_seconds": pair_seconds,
        "total_blocking_keys": len(blocks),
        "unique_block_keys": len(by_key),
        "top_20_largest_blocks": top_20_largest_blocks,
        "candidate_pairs_before_caps": candidate_pairs_before_caps,
        "candidate_pairs_after_caps": len(pairs),
        "candidate_pair_cap_reached": candidate_pair_cap_reached,
        "skipped_oversized_blocks": sum(skipped_by_type.values()),
        "skipped_blocks_by_type": dict(skipped_by_type),
        "skipped_blocks_by_reason": dict(skipped_by_reason),
        "top_skipped_blocks": top_skipped_blocks,
        "exact_tax_star_blocks": exact_tax_star_blocks,
        "exact_tax_star_edges": exact_tax_star_edges,
        "max_rows_by_type": max_rows_by_type,
        "max_candidates_per_block": max_block,
        "max_total_candidate_pairs": max_total_pairs,
        "exact_tax_star_threshold": exact_tax_star_threshold,
    }
    _set_blocking_diagnostics(config, diagnostics)

    if not pairs:
        return _empty_candidates()

    data = [(a, b, blk, typ) for (a, b), (blk, typ) in pairs.items()]
    return pl.DataFrame(data, schema=["row_a", "row_b", "shared_block", "block_type"], orient="row")


def get_address_company_counts(df: pl.DataFrame) -> Dict[str, int]:
    """Count distinct normalized names at each address for shared-address risk."""
    if "addr_norm" not in df.columns or "name_norm" not in df.columns:
        return {}
    out = {}
    grouped = (
        df.filter(pl.col("addr_norm") != "")
        .group_by("addr_norm")
        .agg(pl.col("name_norm").n_unique().alias("company_count"))
    )
    for row in grouped.iter_rows(named=True):
        out[row["addr_norm"]] = int(row["company_count"])
    return out

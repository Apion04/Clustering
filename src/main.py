"""Main clustering orchestrator: ties all modules together."""

import time
import polars as pl
from typing import Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

from src.config import ClusteringConfig
from src.brand_families import (
    encode_hits,
    find_known_brand_family_hits,
    load_known_brand_families,
)
from src.generic_keywords import apply_generic_non_bridge_keywords, load_generic_non_bridge_keywords
from src.legal_keywords import load_legal_keywords
from src.location_modifiers import load_location_modifiers
from src.idf_tokens import annotate_rare_tokens, build_rare_token_index, shared_rare_tokens
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidate_pairs, get_address_company_counts
from src.matching import evaluate_pair
from src.merging import ClusterMerger
from src.scoring import calculate_cluster_scores, calculate_user_facing_cluster_scores, get_cluster_metadata
from src.sorting import sort_by_clusters
from src.output import save_main_output, save_audit_file, generate_processing_report
from src.ai_review import should_ai_review, ai_review_pair
from src.llm_workflow import apply_llm_decisions, build_llm_review_groups, build_llm_review_groups_from_review_candidates


REVIEW_ONLY_PASS_TYPES = frozenset({
    "family_bridge_supported",
    "family_cross_country",
    "family_cross_country_rejected",
    "known_family_bridge",
    "acronym_review_candidate",
    "secondary_or_acronym_bridge",
    "name_exact_review",
    "rare_token_review_candidate",
    "support_field_review",
    "domain_review_candidate",
    "person_address_city_mismatch_review",
    "person_same_name_location_review",
    "regulatory_or_task_force_related",
    "tax_exact_low_similarity_review",
    "tax_exact_institutional_ecosystem_review",
    "operational_status_review",
    # Recall-improvement LLM candidates enter union-find as 70-score clusters
    # rather than review_candidates so they appear in output when
    # allow_unresolved_llm_candidates_in_final_output=True.
    # They are NOT in REVIEW_ONLY_PASS_TYPES by design.
    # Types kept here only to document explicitly what is NOT here:
    #   weak_brand_root_candidate, name_fuzzy_review_candidate,
    #   known_brand_family_weak_candidate, possible_sub_brand_candidate
})


def _is_review_only(result, config: Optional[ClusteringConfig] = None) -> bool:
    """Return true when an edge belongs in review_candidates, not union-find."""
    if (
        config is not None
        and not bool(getattr(config, "llm_can_auto_cluster", False))
        and result.evidence.get("ai_decision")
    ):
        return True
    if result.pass_type in REVIEW_ONLY_PASS_TYPES:
        return True
    if result.pass_type == "distinctive_supplier_identity" and not result.evidence.get("main_cluster_allowed", False):
        return True
    # Known brand/family aliases are review-only unless address/domain/name
    # support raised them into a high-confidence same-entity candidate.
    if result.pass_type == "known_brand_family_alias" and result.match_pct < 88.0:
        return True
    return False


def _pair_key(row_a: int, row_b: int) -> tuple[int, int]:
    return (row_a, row_b) if row_a <= row_b else (row_b, row_a)


def _filter_cluster_map_by_scores(cluster_map: Dict[int, int], scores: Dict[int, float]) -> Dict[int, int]:
    """Drop rows whose cluster root has no user-facing final score."""
    return {
        int(row_id): int(root)
        for row_id, root in cluster_map.items()
        if float(scores.get(root, 0.0) or 0.0) > 0.0
    }


def _augment_with_rare_token_evidence(row_a: Dict[str, Any], row_b: Dict[str, Any], result) -> None:
    shared = sorted(shared_rare_tokens(row_a, row_b))
    if not shared:
        return
    result.evidence["shared_discriminative_tokens"] = shared[:20]
    result.evidence["shared_discriminative_token_count"] = len(shared)


def _log(message: str):
    print(message, flush=True)


def _build_tax_block_stats(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Summarize exact tax values so broad/shared client-side IDs do not over-cluster."""
    stats: Dict[str, Dict[str, Any]] = {}
    if "tax_norm" not in df.columns:
        return stats
    for row in df.iter_rows(named=True):
        roots = stats
        root = row.get("root_brand") or row.get("name_norm") or ""
        name = row.get("name_norm") or ""
        for tax in str(row.get("tax_norm", "") or "").split("|"):
            if not tax:
                continue
            item = roots.setdefault(tax, {"row_count": 0, "roots": set(), "names": set()})
            item["row_count"] += 1
            if root:
                item["roots"].add(root)
            if name:
                item["names"].add(name)
    return {
        tax: {
            "row_count": item["row_count"],
            "distinct_roots": len(item["roots"]),
            "distinct_names": len(item["names"]),
        }
        for tax, item in stats.items()
    }


def _resolve_project_path(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(__file__).resolve().parents[1] / p)


def _annotate_known_brand_families(df: pl.DataFrame, config: ClusteringConfig) -> pl.DataFrame:
    """Attach phrase-level known brand/family alias hits to preprocessed rows."""
    family_file = _resolve_project_path(getattr(config, "known_brand_families_file", ""))
    index = load_known_brand_families(
        family_file,
        default_confidence=float(getattr(config, "known_brand_family_default_confidence", 76.0)),
    )
    setattr(config, "_known_brand_family_index", index)
    setattr(config, "_known_brand_family_report", index.report)
    if index.report.get("missing_file"):
        _log(f"      Known brand families file not found: {family_file}")
    elif index.report.get("rows_loaded", 0):
        _log(
            "      Known brand families loaded: "
            f"rows={index.report.get('rows_loaded', 0):,}, "
            f"families={index.report.get('families_loaded', 0):,}, "
            f"aliases={index.report.get('alias_phrases_loaded', 0):,}, "
            f"safe={index.report.get('aliases_accepted_safe', 0):,}, "
            f"risky={index.report.get('aliases_marked_risky', 0):,}, "
            f"skipped_generic={index.report.get('aliases_skipped_generic', 0):,}"
        )
    if not index.enabled:
        return df.with_columns([
            pl.lit("").alias("known_brand_family_ids"),
            pl.lit("").alias("known_brand_family_safe_ids"),
            pl.lit("").alias("known_brand_family_risky_ids"),
            pl.lit("").alias("known_brand_alias_hits"),
        ])

    family_ids = []
    safe_ids = []
    risky_ids = []
    alias_hits = []
    for row in df.iter_rows(named=True):
        values = [
            row.get("name_norm", ""),
            row.get("orig_supplier_name", ""),
            row.get("json_secondary_names_norm", ""),
            row.get("orig_secondary_names", ""),
            row.get("root_brand", ""),
        ]
        for i in range(2, 8):
            values.append(row.get(f"name{i}_norm", ""))
        encoded = encode_hits(find_known_brand_family_hits(values, index))
        family_ids.append(encoded["known_brand_family_ids"])
        safe_ids.append(encoded["known_brand_family_safe_ids"])
        risky_ids.append(encoded["known_brand_family_risky_ids"])
        alias_hits.append(encoded["known_brand_alias_hits"])
    return df.with_columns([
        pl.Series("known_brand_family_ids", family_ids),
        pl.Series("known_brand_family_safe_ids", safe_ids),
        pl.Series("known_brand_family_risky_ids", risky_ids),
        pl.Series("known_brand_alias_hits", alias_hits),
    ])


def _load_legal_keywords(config: ClusteringConfig):
    """Load legal suffix/company-form keywords for normalization only."""
    legal_file = _resolve_project_path(getattr(config, "legal_keywords_file", ""))
    index = load_legal_keywords(legal_file, include_defaults=True)
    setattr(config, "_legal_keywords_index", index)
    setattr(config, "_legal_keywords_report", index.report)
    if index.report.get("missing_file"):
        _log(f"      Legal keyword file not found, using built-in defaults: {legal_file}")
    else:
        _log(
            "      Legal keywords loaded: "
            f"file_rows={index.report.get('rows_loaded', 0):,}, "
            f"keywords={index.report.get('legal_keywords_loaded', 0):,}, "
            f"duplicates={index.report.get('duplicates_skipped', 0):,}, "
            f"boundary_only={len(index.report.get('boundary_only_suffixes', [])):,}"
        )
    return index


def _load_generic_non_bridge_keywords(config: ClusteringConfig):
    """Load multilingual generic/non-bridge words for suppression only."""
    generic_file = _resolve_project_path(getattr(config, "generic_non_bridge_file", ""))
    index = load_generic_non_bridge_keywords(generic_file, include_defaults=True)
    apply_generic_non_bridge_keywords(index)
    setattr(config, "_generic_non_bridge_index", index)
    setattr(config, "_generic_non_bridge_report", index.report)
    if index.report.get("missing_file"):
        _log(f"      Generic non-bridge file not found, using built-in defaults: {generic_file}")
    else:
        _log(
            "      Generic non-bridge keywords loaded: "
            f"file_rows={index.report.get('rows_loaded', 0):,}, "
            f"keywords={index.report.get('total_generic_words_loaded', 0):,}, "
            f"duplicates={index.report.get('duplicates_skipped', 0):,}"
        )
    return index


def _load_location_modifiers_terms(config: ClusteringConfig):
    """Load location modifier terms for branch/location suffix stripping."""
    loc_file = _resolve_project_path(getattr(config, "location_modifiers_file", ""))
    terms = load_location_modifiers(loc_file) if loc_file else set()
    setattr(config, "_location_modifier_terms", terms)
    if terms:
        _log(f"      Location modifiers loaded: {len(terms):,} terms from {loc_file}")
    else:
        _log(f"      Location modifiers: file not found or empty, using LOCATION_ROOT_TOKENS only")
    return terms


def cluster_suppliers(
    input_df: pl.DataFrame,
    column_mapping: Dict[str, str],
    config: Optional[ClusteringConfig] = None
) -> Dict[str, Any]:
    """
    Main clustering pipeline.

    Args:
        input_df: Raw supplier DataFrame from client
        column_mapping: Maps standard field names to client column names
        config: Processing configuration

    Returns:
        Dict with:
        - main_df: DataFrame with Cluster Number + Match Percentage
        - audit_data: Cluster metadata for debugging
        - stats: Processing statistics
    """
    start_time = time.perf_counter()

    if config is None:
        config = ClusteringConfig()
    config._runtime_timing = {"guardrails_seconds": 0.0}
    stage_timings: Dict[str, float] = {}

    n_rows = len(input_df)
    _log(f"[1/8] Starting clustering for {n_rows:,} rows...")

    # Step 1: Preprocessing
    t0 = time.perf_counter()
    _load_generic_non_bridge_keywords(config)
    legal_index = _load_legal_keywords(config)
    location_terms = _load_location_modifiers_terms(config)
    df = preprocess_dataframe(
        input_df,
        column_mapping,
        legal_keywords=legal_index,
        support_field_strengths=getattr(config, "support_field_strengths", {}),
        location_terms=location_terms,
    )
    df = _annotate_known_brand_families(df, config)
    rare_token_index = build_rare_token_index(df, config, legal_index)
    df = annotate_rare_tokens(df, rare_token_index)
    config._tax_block_stats = _build_tax_block_stats(df)
    stage_timings["preprocessing_seconds"] = time.perf_counter() - t0
    _log(f"[2/8] Preprocessing complete: {stage_timings['preprocessing_seconds']:.2f}s")

    # Step 2: Blocking & Candidate Generation
    t0 = time.perf_counter()
    candidates_df = generate_candidate_pairs(df, config)
    n_candidates = len(candidates_df)
    stage_timings["blocking_total_seconds"] = time.perf_counter() - t0
    blocking_diagnostics = getattr(config, "_blocking_diagnostics", {})
    stage_timings["blocking_key_generation_seconds"] = float(blocking_diagnostics.get("blocking_key_generation_seconds", 0.0))
    stage_timings["candidate_pair_generation_seconds"] = float(blocking_diagnostics.get("candidate_pair_generation_seconds", 0.0))
    _log(f"[3/8] Blocking complete: {n_candidates:,} candidate pairs ({stage_timings['blocking_total_seconds']:.2f}s)")
    _log(
        "      Blocking diagnostics: "
        f"keys={blocking_diagnostics.get('total_blocking_keys', 0):,}, "
        f"unique_blocks={blocking_diagnostics.get('unique_block_keys', 0):,}, "
        f"raw_pairs={blocking_diagnostics.get('candidate_pairs_before_caps', 0):,}, "
        f"after_caps={blocking_diagnostics.get('candidate_pairs_after_caps', 0):,}, "
        f"skipped_blocks={blocking_diagnostics.get('skipped_oversized_blocks', 0):,}"
    )
    _log(
        "      Blocking timing: "
        f"key_generation={stage_timings['blocking_key_generation_seconds']:.2f}s, "
        f"candidate_generation={stage_timings['candidate_pair_generation_seconds']:.2f}s"
    )
    if blocking_diagnostics.get("skipped_blocks_by_type"):
        _log(f"      Skipped by type: {blocking_diagnostics.get('skipped_blocks_by_type')}")
    for idx, block in enumerate(blocking_diagnostics.get("top_20_largest_blocks", [])[:20], 1):
        _log(
            "      Top block "
            f"{idx}: type={block.get('block_type')} rows={block.get('row_count'):,} "
            f"raw_pairs={block.get('raw_pair_count'):,} key={block.get('block_key_preview')}"
        )
    candidate_pairs_capped = bool(blocking_diagnostics.get("candidate_pair_cap_reached")) or n_candidates >= int(getattr(config, "max_total_candidate_pairs", 1000000))
    if candidate_pairs_capped:
        _log(f"      WARNING: candidate pair cap reached at {n_candidates:,}; broad candidate blocks were truncated.")

    # Step 3: Address risk analysis
    t0 = time.perf_counter()
    address_counts = get_address_company_counts(df)
    stage_timings["address_risk_seconds"] = time.perf_counter() - t0
    _log(f"[4/8] Address risk analysis complete ({stage_timings['address_risk_seconds']:.2f}s)")

    # Step 4: Convert to dicts for fast row lookup
    t0 = time.perf_counter()
    rows_dict = {}
    for row in df.iter_rows(named=True):
        rows_dict[row["row_id"]] = row
    stage_timings["row_dictionary_seconds"] = time.perf_counter() - t0
    _log(f"[5/8] Row dictionary built ({stage_timings['row_dictionary_seconds']:.2f}s)")

    # Step 5: Initialize merger and set tax IDs
    t0 = time.perf_counter()
    merger = ClusterMerger(n_rows, config)
    for row_id, row in rows_dict.items():
        tax = row.get("tax_norm", "")
        if tax:
            merger.set_row_tax(row_id, tax)
    stage_timings["cluster_merger_init_seconds"] = time.perf_counter() - t0
    _log(f"[6/9] Cluster merger initialized ({stage_timings['cluster_merger_init_seconds']:.2f}s)")

    # Step 6: Evaluate all candidate pairs
    t0 = time.perf_counter()
    match_results = []
    review_candidates = []
    rejected_pair_blocklist: Dict[tuple[int, int], str] = {}
    pass_type_counts = defaultdict(int)
    pair_evaluation_seconds = 0.0
    cluster_merge_seconds = 0.0
    progress_interval = max(100000, n_candidates // 10) if n_candidates else 100000

    for idx, candidate in enumerate(candidates_df.iter_rows(named=True), 1):
        row_a = candidate["row_a"]
        row_b = candidate["row_b"]

        if row_a not in rows_dict or row_b not in rows_dict:
            continue
        pair_key = _pair_key(row_a, row_b)
        if pair_key in rejected_pair_blocklist:
            continue

        eval_t0 = time.perf_counter()
        result = evaluate_pair(rows_dict[row_a], rows_dict[row_b], address_counts, config)
        _augment_with_rare_token_evidence(rows_dict[row_a], rows_dict[row_b], result)
        pair_evaluation_seconds += time.perf_counter() - eval_t0

        if result.is_match:
            # Optional: AI review for ambiguous cases
            if config.ai_review_enabled and should_ai_review(result):
                ai_result = ai_review_pair(rows_dict[row_a], rows_dict[row_b], config)
                if ai_result.verdict == "UNAVAILABLE":
                    result.evidence["ai_decision"] = "UNAVAILABLE"
                    result.evidence["ai_reason"] = ai_result.reason
                    result.evidence["ai_confidence"] = ai_result.confidence
                    if result.pass_type == "acronym_review_candidate":
                        continue
                if ai_result.verdict == "DIFFERENT":
                    continue  # Skip this match entirely.
                if ai_result.verdict in {"SAME_ENTITY", "RELATED"}:
                    # AI approval can raise a weak candidate, but keep it below deterministic exact-tax/name-address scores.
                    result.match_pct = max(result.match_pct, min(float(ai_result.confidence), 92.0))
                    result.evidence["ai_decision"] = ai_result.verdict
                    result.evidence["ai_reason"] = ai_result.reason
                    result.evidence["ai_confidence"] = ai_result.confidence
                    result.evidence["ai_relationship_type"] = ai_result.relationship_type
                if ai_result.verdict == "UNCERTAIN":
                    # Uncertain but plausible candidates stay in the separate
                    # Review output by default, with a low score for manual
                    # triage. They do not enter union-find unless explicitly
                    # allowed by llm_can_auto_cluster.
                    if getattr(config, "ai_uncertain_cluster_enabled", True):
                        result.match_pct = min(result.match_pct, float(getattr(config, "ai_uncertain_match_pct", 68.0)))
                        result.needs_review = True
                        result.review_reason = f"AI uncertain, kept for manual review: {ai_result.reason}"
                        result.evidence["ai_decision"] = "UNCERTAIN_GROUPED"
                        result.evidence["ai_reason"] = ai_result.reason
                        result.evidence["ai_confidence"] = ai_result.confidence
                        result.evidence["ai_relationship_type"] = ai_result.relationship_type
                    else:
                        continue

            if _is_review_only(result, config):
                review_candidates.append({
                    "row_a": row_a,
                    "row_b": row_b,
                    "match_pct": result.match_pct,
                    "pass_type": result.pass_type,
                    "needs_review": result.needs_review,
                    "review_reason": result.review_reason,
                    "evidence": result.evidence,
                })
            else:
                merge_t0 = time.perf_counter()
                merge_status = merger.add_edge(row_a, row_b, result)
                cluster_merge_seconds += time.perf_counter() - merge_t0

                if merge_status in ["MERGED", "ALREADY_SAME"]:
                    match_results.append({
                        "row_a": row_a,
                        "row_b": row_b,
                        "match_pct": result.match_pct,
                        "pass_type": result.pass_type,
                        "needs_review": result.needs_review,
                    })
                    pass_type_counts[result.pass_type] += 1
        else:
            reason = result.evidence.get("guardrail_reject") or result.review_reason
            if reason:
                rejected_pair_blocklist[pair_key] = reason
            elif result.evidence.get("shared_discriminative_tokens"):
                review_candidates.append({
                    "row_a": row_a,
                    "row_b": row_b,
                    "match_pct": 60.0,
                    "pass_type": "rare_token_review_candidate",
                    "needs_review": True,
                    "review_reason": "Shared input-specific rare token only; no strong duplicate support",
                    "evidence": {**result.evidence, "original_pass_type": result.pass_type},
                })
        if idx % progress_interval == 0:
            _log(
                f"      Matching progress: {idx:,}/{n_candidates:,} candidate pairs, "
                f"edges={len(match_results):,}, elapsed={time.perf_counter() - t0:.1f}s"
            )

    stage_timings["matching_seconds"] = time.perf_counter() - t0
    stage_timings["pair_evaluation_seconds"] = pair_evaluation_seconds
    stage_timings["guardrails_seconds"] = float(getattr(config, "_runtime_timing", {}).get("guardrails_seconds", 0.0))
    stage_timings["cluster_merge_seconds"] = cluster_merge_seconds
    _log(
        f"[7/9] Matching complete: {len(match_results):,} edges created "
        f"({stage_timings['matching_seconds']:.2f}s; eval={pair_evaluation_seconds:.2f}s, "
        f"guardrails={stage_timings['guardrails_seconds']:.2f}s, merge={cluster_merge_seconds:.2f}s)"
    )

    # Step 7: Calculate scores and build cluster map
    t0 = time.perf_counter()
    clusters = merger.get_clusters()
    raw_match_pcts = calculate_cluster_scores(merger)
    pre_llm_match_pcts = calculate_user_facing_cluster_scores(
        merger,
        expose_unresolved_llm_candidates=True,
    )
    audit_data = get_cluster_metadata(merger)

    # Build row_id -> cluster_root mapping
    cluster_map = {}
    for root, rows in clusters.items():
        if len(rows) > 1:  # Only real clusters
            for row_id in rows:
                cluster_map[row_id] = root

    llm_review_groups = build_llm_review_groups(cluster_map, pre_llm_match_pcts, rows_dict, merger)
    llm_review_groups.extend(build_llm_review_groups_from_review_candidates(
        review_candidates,
        rows_dict,
        start_index=len(llm_review_groups) + 1,
    ))
    llm_decision_counts = {"approve": 0, "reject": 0, "split": 0, "merge_with_existing": 0, "promote_score": 0, "errors": 0}
    allow_unresolved_llm = bool(getattr(config, "allow_unresolved_llm_candidates_in_final_output", False))
    output_cluster_map = cluster_map
    output_match_pcts = calculate_user_facing_cluster_scores(
        merger,
        expose_unresolved_llm_candidates=allow_unresolved_llm,
    )
    if getattr(config, "llm_group_decisions", None):
        output_cluster_map, output_match_pcts, llm_decision_counts = apply_llm_decisions(
            n_rows,
            cluster_map,
            pre_llm_match_pcts,
            list(getattr(config, "llm_group_decisions", []) or []),
        )
        setattr(config, "_llm_decision_counts", llm_decision_counts)
        if not allow_unresolved_llm:
            unresolved_numbers = {root for root, score in output_match_pcts.items() if int(score or 0) == 70}
            if unresolved_numbers:
                output_cluster_map = {rid: root for rid, root in output_cluster_map.items() if root not in unresolved_numbers}
                output_match_pcts = {root: score for root, score in output_match_pcts.items() if root not in unresolved_numbers}
    elif not allow_unresolved_llm:
        output_cluster_map = _filter_cluster_map_by_scores(output_cluster_map, output_match_pcts)
    setattr(config, "_llm_review_groups_sent", len(llm_review_groups) if bool(getattr(config, "ai_review_enabled", False)) else 0)
    pre_llm_cluster_70_count = sum(1 for root, score in pre_llm_match_pcts.items() if int(score or 0) == 70)
    unresolved_llm_group_count = len(llm_review_groups)
    unresolved_llm_output_count = sum(1 for root, score in output_match_pcts.items() if int(score or 0) == 70)
    production_run_status = "OK"
    if unresolved_llm_group_count and not allow_unresolved_llm and not getattr(config, "llm_group_decisions", None):
        production_run_status = "INCOMPLETE_UNRESOLVED_LLM_CANDIDATES"
    elif unresolved_llm_output_count and not allow_unresolved_llm:
        production_run_status = "INCOMPLETE_UNRESOLVED_LLM_CANDIDATES"

    stage_timings["scoring_seconds"] = time.perf_counter() - t0
    _log(f"[8/9] Scoring complete: {len([c for c in clusters.values() if len(c) > 1])} clusters ({stage_timings['scoring_seconds']:.2f}s)")

    # Step 8: Sort output
    t0 = time.perf_counter()
    output_df = sort_by_clusters(df, output_cluster_map, output_match_pcts)
    stage_timings["sorting_seconds"] = time.perf_counter() - t0
    _log(f"[9/9] Sorting complete ({stage_timings['sorting_seconds']:.2f}s)")

    # Calculate stats
    total_clusters = len([c for c in clusters.values() if len(c) > 1])
    auto_clustered = sum(len(c) for c in clusters.values() if len(c) > 1)
    singletons = n_rows - auto_clustered

    review_clusters = sum(1 for meta in audit_data.values() if meta["needs_review"])
    review_rows = sum(meta["cluster_size"] for meta in audit_data.values() if meta["needs_review"])
    review_candidate_rows = len({row_id for cand in review_candidates for row_id in (cand["row_a"], cand["row_b"])})

    processing_time = time.perf_counter() - start_time
    largest_cluster_size = max((len(c) for c in clusters.values()), default=0)
    pre_llm_score_distribution = {
        "98": sum(1 for root, rows in clusters.items() if len(rows) > 1 and int(pre_llm_match_pcts.get(root, 0)) == 98),
        "85": sum(1 for root, rows in clusters.items() if len(rows) > 1 and int(pre_llm_match_pcts.get(root, 0)) == 85),
        "70": pre_llm_cluster_70_count,
        "blank": singletons,
    }
    final_cluster_roots = set(output_cluster_map.values())
    final_score_distribution = {
        "98": sum(1 for root in final_cluster_roots if int(output_match_pcts.get(root, 0)) == 98),
        "85": sum(1 for root in final_cluster_roots if int(output_match_pcts.get(root, 0)) == 85),
        "70": sum(1 for root in final_cluster_roots if int(output_match_pcts.get(root, 0)) == 70),
        "blank": n_rows - len(output_cluster_map),
    }

    stats = {
        "total_rows": n_rows,
        "candidate_pairs": n_candidates,
        "candidate_pairs_capped": candidate_pairs_capped,
        "match_edges_created": len(match_results),
        "clusters_found": total_clusters,
        "main_clusters_found": total_clusters,
        "review_only_clusters_excluded_from_main": 0,
        "review_candidate_pairs": len(review_candidates),
        "review_candidate_rows": review_candidate_rows,
        "guardrail_rejected_pairs": len(rejected_pair_blocklist),
        "auto_clustered_rows": auto_clustered - review_rows,
        "review_queue_rows": review_rows,
        "singleton_rows": singletons,
        "processing_time_seconds": processing_time,
        "largest_cluster_size": largest_cluster_size,
        "size_2_clusters": sum(1 for c in clusters.values() if len(c) == 2),
        "size_3_clusters": sum(1 for c in clusters.values() if len(c) == 3),
        "size_4plus_clusters": sum(1 for c in clusters.values() if len(c) >= 4),
        "pass_type_counts": dict(pass_type_counts),
        "pre_llm_score_distribution": pre_llm_score_distribution,
        "final_score_distribution": final_score_distribution,
        "llm_review_candidate_clusters": unresolved_llm_group_count,
        "pre_llm_70_cluster_count": pre_llm_cluster_70_count,
        "llm_review_candidate_groups": unresolved_llm_group_count,
        "llm_review_groups_sent": int(getattr(config, "_llm_review_groups_sent", 0)),
        "unresolved_llm_groups": unresolved_llm_output_count if allow_unresolved_llm else unresolved_llm_group_count,
        "allow_unresolved_llm_candidates_in_final_output": allow_unresolved_llm,
        "unresolved_llm_candidate_mode": str(getattr(config, "unresolved_llm_candidate_mode", "exception")),
        "production_run_status": production_run_status,
        "llm_decisions": llm_decision_counts,
        "stage_timings": stage_timings,
        "blocking_diagnostics": blocking_diagnostics,
        "known_brand_families": getattr(config, "_known_brand_family_report", {}),
        "legal_keywords": getattr(config, "_legal_keywords_report", {}),
        "generic_non_bridge": getattr(config, "_generic_non_bridge_report", {}),
        "rare_tokens": getattr(config, "_rare_token_report", {}),
    }

    _log(f"\n✅ Done! {processing_time:.1f}s total")
    _log(
        f"   Clusters: {total_clusters} | Auto: {auto_clustered - review_rows} | "
        f"Review rows: {review_rows} | Review candidates: {len(review_candidates)} | Singletons: {singletons}"
    )

    return {
        "main_df": output_df,
        "preprocessed_df": df,
        "audit_data": audit_data,
        "stats": stats,
        "merger": merger,
        "cluster_map": cluster_map,
        "review_candidates": review_candidates,
        "llm_review_groups": llm_review_groups,
        "output_cluster_map": output_cluster_map,
        "output_match_pcts": output_match_pcts,
        "pre_llm_match_pcts": pre_llm_match_pcts,
    }

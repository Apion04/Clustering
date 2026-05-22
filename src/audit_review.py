"""Audit + Review Output Layer.

Produces two optional output DataFrames:
- review_pairs_df:    70%-candidate and needs_review edges, for manual triage
- cluster_audit_df:   per-cluster risk classification and quality flags

No matching or scoring logic is changed here. This module is purely additive.
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional, Any

import polars as pl

from src.merging import ClusterEdge, WEAK_EDGE_TYPES
from src.config import ClusteringConfig


# ---------------------------------------------------------------------------
# cluster_kind
# ---------------------------------------------------------------------------

def build_cluster_kind(match_pct: float, cluster_size: int) -> str:
    """Classify a cluster by its minimum match percentage and size."""
    if cluster_size <= 1 or match_pct == 0.0:
        return "singleton"
    if match_pct >= 98:
        return "exact_duplicate"
    if match_pct >= 85:
        return "supplier_family"
    if match_pct >= 70:
        return "review_candidate"
    return "singleton"


# ---------------------------------------------------------------------------
# Edge-level helpers
# ---------------------------------------------------------------------------

_PASS_TYPE_REASON_TEMPLATES: Dict[str, str] = {
    "domain_review_candidate":        "Same domain but weak name similarity ({score:.0f}%)",
    "brand_alias_candidate":          "Global brand alias match — {risk_level} risk ({score:.0f}%)",
    "weak_brand_root_candidate":      "Shared brand root without address/city/domain support ({score:.0f}%)",
    "family_bridge_supported":        "Parent/family bridge with location support ({score:.0f}%)",
    "family_cross_country":           "Cross-country brand root relation ({score:.0f}%)",
    "known_brand_family_alias":       "Known brand/family alias match ({score:.0f}%)",
    "possible_sub_brand_candidate":   "One supplier name appears to be a sub-brand prefix ({score:.0f}%)",
    "name_fuzzy_review_candidate":    "High name similarity without location/domain support ({score:.0f}%)",
}


def _make_match_reason(pass_type: str, score: float, evidence: Dict[str, Any]) -> str:
    """Return a deterministic human-readable reason string."""
    template = _PASS_TYPE_REASON_TEMPLATES.get(pass_type)
    if template:
        risk_level = evidence.get("risk_level", "medium") if evidence else "medium"
        return template.format(score=score, risk_level=risk_level)
    return f"{pass_type.replace('_', ' ').title()} ({score:.0f}%)"


def _make_edge_risk_flags(edge: ClusterEdge) -> str:
    """Return comma-separated risk flags for a single edge."""
    flags: List[str] = []
    evidence = edge.evidence or {}

    if edge.pass_type in WEAK_EDGE_TYPES:
        flags.append("weak_edge")
    if edge.pass_type == "brand_alias_candidate":
        flags.append("alias_candidate")
    if edge.pass_type == "brand_alias_candidate" and evidence.get("risk_level") == "high":
        flags.append("high_risk_alias")
    if edge.pass_type in {"weak_brand_root_candidate", "family_cross_country"}:
        flags.append("generic_root_bridge")
    if edge.pass_type == "domain_review_candidate":
        flags.append("domain_only_risk")
    if edge.pass_type == "professional_name_address":
        flags.append("person_company_mix")
    if edge.needs_review:
        flags.append("needs_review")

    return ",".join(flags)


def _make_edge_suggested_action(edge: ClusterEdge) -> str:
    """Return a suggested action for a single edge."""
    if edge.pass_type == "brand_alias_candidate":
        return "possible_alias"
    if edge.pass_type in {
        "family_bridge_supported",
        "family_cross_country",
        "known_family_bridge",
        "known_brand_family_alias",
        "secondary_or_acronym_bridge",
    }:
        return "possible_supplier_family"
    if edge.pass_type in WEAK_EDGE_TYPES:
        return "llm_review_when_available"
    if edge.needs_review:
        return "manual_review"
    return "manual_review"


# ---------------------------------------------------------------------------
# build_review_pairs
# ---------------------------------------------------------------------------

_REVIEW_PAIRS_SCHEMA = {
    "supplier_a_row_id": pl.Int64,
    "supplier_b_row_id": pl.Int64,
    "supplier_a_name": pl.Utf8,
    "supplier_b_name": pl.Utf8,
    "supplier_a_address": pl.Utf8,
    "supplier_b_address": pl.Utf8,
    "supplier_a_country": pl.Utf8,
    "supplier_b_country": pl.Utf8,
    "supplier_a_domain": pl.Utf8,
    "supplier_b_domain": pl.Utf8,
    "suggested_score": pl.Float64,
    "pass_type": pl.Utf8,
    "match_reason": pl.Utf8,
    "risk_flags": pl.Utf8,
    "suggested_action": pl.Utf8,
}


def build_review_pairs(
    all_edges: List[ClusterEdge],
    row_lookup: Dict[int, Dict],
    config: Optional[ClusteringConfig] = None,
) -> pl.DataFrame:
    """Build a review-pairs DataFrame from all cluster edges.

    Includes an edge if:
    - edge.match_pct > 0 AND
    - (edge.match_pct <= 70 OR edge.needs_review OR edge.pass_type in WEAK_EDGE_TYPES)
    """
    rows: List[Dict] = []
    for edge in all_edges:
        if edge.match_pct <= 0:
            continue
        if not (edge.match_pct <= 70 or edge.needs_review or edge.pass_type in WEAK_EDGE_TYPES):
            continue

        row_a_data = row_lookup.get(edge.row_a, {})
        row_b_data = row_lookup.get(edge.row_b, {})
        evidence = edge.evidence or {}

        rows.append({
            "supplier_a_row_id": int(edge.row_a),
            "supplier_b_row_id": int(edge.row_b),
            "supplier_a_name": str(row_a_data.get("name_norm", "") or ""),
            "supplier_b_name": str(row_b_data.get("name_norm", "") or ""),
            "supplier_a_address": str(row_a_data.get("addr_norm", "") or ""),
            "supplier_b_address": str(row_b_data.get("addr_norm", "") or ""),
            "supplier_a_country": str(row_a_data.get("country_norm", "") or ""),
            "supplier_b_country": str(row_b_data.get("country_norm", "") or ""),
            "supplier_a_domain": str(row_a_data.get("domain", "") or ""),
            "supplier_b_domain": str(row_b_data.get("domain", "") or ""),
            "suggested_score": float(edge.match_pct),
            "pass_type": str(edge.pass_type or ""),
            "match_reason": _make_match_reason(edge.pass_type, edge.match_pct, evidence),
            "risk_flags": _make_edge_risk_flags(edge),
            "suggested_action": _make_edge_suggested_action(edge),
        })

    if not rows:
        return pl.DataFrame(schema=_REVIEW_PAIRS_SCHEMA)

    return pl.DataFrame(rows, schema=_REVIEW_PAIRS_SCHEMA)


# ---------------------------------------------------------------------------
# Cluster audit helpers
# ---------------------------------------------------------------------------

def _make_cluster_risk_flags(
    cluster_size: int,
    countries: List[str],
    domains: List[str],
    tax_ids: List[str],
    edge_pass_types: List[str],
    cluster_kind: str,
    edges: List[ClusterEdge],
) -> List[str]:
    """Return list of risk flag strings for a cluster."""
    flags: List[str] = []

    if cluster_size > 8:
        flags.append("large_cluster")
    if len(countries) > 1:
        flags.append("mixed_countries")
    if len(domains) > 1:
        flags.append("multiple_domains")
    if len(tax_ids) > 1:
        flags.append("multiple_tax_ids")
    if any(pt == "brand_alias_candidate" for pt in edge_pass_types):
        flags.append("alias_based_cluster")
    if cluster_kind == "review_candidate":
        flags.append("review_candidate_cluster")
    if any(pt in WEAK_EDGE_TYPES for pt in edge_pass_types):
        flags.append("weak_edge_chain")
    if any(pt in {"weak_brand_root_candidate", "family_cross_country"} for pt in edge_pass_types):
        flags.append("generic_root_bridge")

    # possible_outlier: size >= 3 and any member country differs from majority
    if cluster_size >= 3 and len(countries) > 1:
        flags.append("possible_outlier")

    # franchise_banner_brand: alias/brand family edge with medium or high risk evidence
    for edge in edges:
        if edge.pass_type in {"known_brand_family_alias", "brand_alias_candidate"}:
            risk_level = (edge.evidence or {}).get("risk_level", "")
            if risk_level in {"high", "medium"}:
                flags.append("franchise_banner_brand")
                break

    return flags


def _make_cluster_possible_issue(flags: List[str], cluster_size: int) -> str:
    if "possible_outlier" in flags:
        return "Cluster member(s) may not belong — different country/domain from majority"
    if "large_cluster" in flags:
        return f"Large cluster ({cluster_size} members) — verify no over-clustering"
    if "weak_edge_chain" in flags and "review_candidate_cluster" in flags:
        return "Low-confidence edges only — needs review before final clustering"
    if "mixed_countries" in flags:
        return "Multiple countries in cluster — verify intended cross-border grouping"
    return ""


def _make_cluster_suggested_action(flags: List[str]) -> str:
    """Priority: possible_outlier > llm_review > manual_review > review_cluster > safe_cluster."""
    if "possible_outlier" in flags:
        return "possible_outlier"
    if "alias_based_cluster" in flags and "review_candidate_cluster" in flags:
        return "llm_review_when_available"
    if "large_cluster" in flags:
        return "manual_review"
    if "review_candidate_cluster" in flags or "weak_edge_chain" in flags:
        return "review_cluster"
    if not flags:
        return "safe_cluster"
    return "review_cluster"


def _make_cluster_explanation(cluster_kind: str, flags: List[str], cluster_score: float) -> str:
    if cluster_kind == "singleton":
        return "Singleton — no match found"
    if cluster_kind == "exact_duplicate":
        return f"{cluster_score:.0f}% exact duplicate cluster — same normalized name and address"
    if cluster_kind == "supplier_family":
        return f"{cluster_score:.0f}% supplier-family cluster based on distinctive brand across legal/location variants"
    if cluster_kind == "review_candidate":
        if "generic_root_bridge" in flags:
            return f"{cluster_score:.0f}% review candidate — weak brand root bridge without address/city support"
        return f"{cluster_score:.0f}% review candidate — requires manual or LLM validation"
    return f"{cluster_score:.0f}% cluster"


# ---------------------------------------------------------------------------
# build_cluster_audit
# ---------------------------------------------------------------------------

_CLUSTER_AUDIT_SCHEMA = {
    "cluster_number": pl.Int64,
    "cluster_size": pl.Int64,
    "cluster_score": pl.Float64,
    "cluster_kind": pl.Utf8,
    "risk_flags": pl.Utf8,
    "possible_issue": pl.Utf8,
    "suggested_action": pl.Utf8,
    "sample_supplier_names": pl.Utf8,
    "countries": pl.Utf8,
    "domains": pl.Utf8,
    "pass_types_seen": pl.Utf8,
    "explanation": pl.Utf8,
}


def build_cluster_audit(
    clusters: Dict[int, Set[int]],
    edges_by_root: Dict[int, List[ClusterEdge]],
    row_lookup: Dict[int, Dict],
    cluster_number_map: Dict[int, int],
    config: Optional[ClusteringConfig] = None,
) -> pl.DataFrame:
    """Build a per-cluster audit DataFrame.

    Args:
        clusters:           root -> set of member row_ids
        edges_by_root:      root -> list of ClusterEdge (from merger.edges)
        row_lookup:         row_id -> preprocessed row dict
        cluster_number_map: root -> 1-based display cluster number
    """
    rows: List[Dict] = []

    for root, members in clusters.items():
        member_list = sorted(members)
        size = len(member_list)
        cluster_num = cluster_number_map.get(root, 0)

        edges = edges_by_root.get(root, [])

        # cluster_score = min match_pct (0.0 if no edges / singleton)
        cluster_score = min((e.match_pct for e in edges), default=0.0) if edges else 0.0

        kind = build_cluster_kind(cluster_score, size)

        # Collect member attributes
        countries_set: Set[str] = set()
        domains_set: Set[str] = set()
        tax_ids_set: Set[str] = set()
        names_list: List[str] = []

        for rid in member_list:
            row = row_lookup.get(rid, {})
            c = str(row.get("country_norm", "") or "").strip()
            if c:
                countries_set.add(c)
            d = str(row.get("domain", "") or "").strip()
            if d:
                domains_set.add(d)
            t = str(row.get("tax_norm", "") or "").strip()
            if t:
                for part in t.split("|"):
                    part = part.strip()
                    if part:
                        tax_ids_set.add(part)
            n = str(row.get("name_norm", "") or "").strip()
            if n:
                names_list.append(n)

        countries = sorted(countries_set)
        domains = sorted(domains_set)
        tax_ids = sorted(tax_ids_set)

        edge_pass_types = sorted({e.pass_type for e in edges if e.pass_type})
        sample_names = " | ".join(names_list[:3])

        flags = _make_cluster_risk_flags(
            cluster_size=size,
            countries=countries,
            domains=domains,
            tax_ids=tax_ids,
            edge_pass_types=edge_pass_types,
            cluster_kind=kind,
            edges=edges,
        )

        possible_issue = _make_cluster_possible_issue(flags, size)
        suggested_action = _make_cluster_suggested_action(flags)
        explanation = _make_cluster_explanation(kind, flags, cluster_score)

        rows.append({
            "cluster_number": int(cluster_num),
            "cluster_size": int(size),
            "cluster_score": float(cluster_score),
            "cluster_kind": str(kind),
            "risk_flags": ",".join(flags),
            "possible_issue": str(possible_issue),
            "suggested_action": str(suggested_action),
            "sample_supplier_names": str(sample_names),
            "countries": ",".join(countries),
            "domains": ",".join(domains),
            "pass_types_seen": ",".join(edge_pass_types),
            "explanation": str(explanation),
        })

    if not rows:
        return pl.DataFrame(schema=_CLUSTER_AUDIT_SCHEMA)

    return pl.DataFrame(rows, schema=_CLUSTER_AUDIT_SCHEMA)

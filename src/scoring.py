"""Scoring module: calculate final match percentages per cluster."""

from typing import Dict, Set, List, Any
from collections import defaultdict

from src.config import COMMON_FIRST_NAMES, GENERIC_ROOT_TOKENS, LOCATION_ROOT_TOKENS
from src.merging import ClusterMerger


AUTO_CONFIDENT_SCORE = 98.0
MANUAL_REVIEW_SCORE = 85.0
LLM_REVIEW_SCORE = 70.0


AUTO_CONFIDENT_PASS_TYPES = {
    "tax_exact",
    "tax_loose_supported",
    "support_same_entity_id",
    "support_same_entity_name",
    "name_address_exact",
    "domain_name_related",
}

LLM_REVIEW_PASS_TYPES = {
    "address_name_related",
    "regulatory_or_task_force_related",
    "tax_exact_low_similarity_review",
    "operational_status_review",
}


def edge_user_route(edge) -> str:
    """Map an accepted internal edge to the production user-facing route."""
    evidence = edge.evidence or {}
    explicit = evidence.get("route")
    if explicit in {"AUTO_CONFIDENT", "MANUAL_REVIEW", "LLM_REVIEW", "REJECT"}:
        return explicit
    if edge.pass_type in AUTO_CONFIDENT_PASS_TYPES and not edge.needs_review:
        return "AUTO_CONFIDENT"
    if edge.pass_type == "distinctive_supplier_identity":
        if evidence.get("ambiguous_review_core") or float(edge.match_pct or 0.0) < 80:
            return "LLM_REVIEW"
        if evidence.get("same_domain") or evidence.get("address_supported") or float(edge.match_pct or 0.0) >= 93:
            return "AUTO_CONFIDENT"
        return "MANUAL_REVIEW"
    if edge.pass_type in LLM_REVIEW_PASS_TYPES or float(edge.match_pct or 0.0) < 80:
        return "LLM_REVIEW"
    if edge.needs_review:
        return "MANUAL_REVIEW"
    if float(edge.match_pct or 0.0) >= 90:
        return "AUTO_CONFIDENT"
    return "MANUAL_REVIEW"


def route_to_user_score(route: str, *, expose_unresolved_llm_candidates: bool = True) -> float:
    if route == "AUTO_CONFIDENT":
        return AUTO_CONFIDENT_SCORE
    if route == "MANUAL_REVIEW":
        return MANUAL_REVIEW_SCORE
    if route == "LLM_REVIEW":
        return LLM_REVIEW_SCORE if expose_unresolved_llm_candidates else 0.0
    return 0.0


def calculate_user_facing_cluster_scores(
    merger: ClusterMerger,
    *,
    expose_unresolved_llm_candidates: bool = True,
) -> Dict[int, float]:
    """Calculate final user-facing scores: only 98, 85, 70, or blank.

    Internal/raw edge scores stay available in audit metadata. The main output
    uses the lowest-confidence route in the cluster so a weak/LLM-review edge
    cannot be hidden by stronger edges elsewhere in the connected component.
    """
    clusters = merger.get_clusters()
    scores: Dict[int, float] = {}
    route_rank = {"LLM_REVIEW": 0, "MANUAL_REVIEW": 1, "AUTO_CONFIDENT": 2}
    for root, rows in clusters.items():
        if len(rows) <= 1:
            scores[root] = 0.0
            continue
        edges = merger.get_cluster_edges(root)
        if not edges:
            scores[root] = 0.0
            continue
        routes = [edge_user_route(edge) for edge in edges]
        route = min(routes, key=lambda r: route_rank.get(r, -1))
        if route in {"AUTO_CONFIDENT", "MANUAL_REVIEW"}:
            hardening_route = cluster_hardening_route(edges)
            if hardening_route:
                route = hardening_route
        scores[root] = route_to_user_score(route, expose_unresolved_llm_candidates=expose_unresolved_llm_candidates)
    return scores


def cluster_hardening_route(edges: List[Any]) -> str:
    """Downgrade accepted clusters that are only generic/location evidence.

    Pair-level guardrails should prevent these in normal operation. This
    cluster-level check is a production backstop for connected components:
    accepted 85/98 clusters that have no tax/domain/address/name-address support
    and are driven only by generic/location/common tokens are routed to LLM
    review, or rejected when the shared evidence is location-only.
    """
    if not edges:
        return ""
    if any(_edge_has_strong_support(edge) for edge in edges):
        return ""

    shared_tokens = set()
    nonbridge_tokens = set()
    location_tokens = set()
    suspicious = False
    for edge in edges:
        evidence = edge.evidence or {}
        tokens = _evidence_shared_tokens(evidence)
        shared_tokens |= tokens
        nonbridge_tokens |= tokens & (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES)
        location_tokens |= tokens & LOCATION_ROOT_TOKENS
        if evidence.get("ambiguous_review_core") or edge.pass_type in {"known_brand_family_alias", "distinctive_supplier_identity", "name_fuzzy_supported"}:
            suspicious = True
    if not suspicious or not shared_tokens:
        return ""
    if shared_tokens and shared_tokens <= LOCATION_ROOT_TOKENS:
        return "REJECT"
    if shared_tokens <= (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES):
        return "LLM_REVIEW"
    return ""


def _edge_has_strong_support(edge: Any) -> bool:
    evidence = edge.evidence or {}
    if edge.pass_type in AUTO_CONFIDENT_PASS_TYPES:
        return True
    if edge.pass_type in {"name_address_exact", "tax_exact", "tax_loose_supported", "domain_name_related"}:
        return True
    return bool(
        evidence.get("tax_overlap")
        or evidence.get("tax_loose_overlap")
        or evidence.get("same_domain")
        or evidence.get("address_supported")
        or evidence.get("addr_sim", 0) and float(evidence.get("addr_sim") or 0) >= 0.88
    )


def _evidence_shared_tokens(evidence: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    for key in (
        "shared_tokens",
        "shared_core_tokens",
        "shared_trusted_core_tokens",
        "risky_overlap",
        "regulatory_overlap",
        "known_brand_family_overlap",
    ):
        value = evidence.get(key)
        if isinstance(value, str):
            tokens |= {t for t in value.lower().replace("|", " ").split() if t}
        elif isinstance(value, (list, tuple, set)):
            tokens |= {str(t).lower() for t in value if str(t).strip()}
    core = evidence.get("shared_supplier_identity_core")
    if isinstance(core, str) and core:
        tokens |= {t for t in core.lower().split() if t}
    return tokens


def calculate_cluster_scores(merger: ClusterMerger) -> Dict[int, float]:
    """
    Calculate match percentage for each cluster.
    Uses weakest-link principle (minimum edge score).
    """
    clusters = merger.get_clusters()
    scores = {}

    for root, rows in clusters.items():
        if len(rows) <= 1:
            scores[root] = 0.0
            continue

        edges = merger.get_cluster_edges(root)
        if not edges:
            scores[root] = 0.0
            continue

        # Weakest link = minimum match percentage
        min_score = min(e.match_pct for e in edges)
        scores[root] = min_score

    return scores


def get_cluster_metadata(merger: ClusterMerger) -> Dict[int, Dict[str, Any]]:
    """Get full metadata for each cluster for audit trail."""
    clusters = merger.get_clusters()
    metadata = {}

    for root, rows in clusters.items():
        if len(rows) <= 1:
            continue

        edges = merger.get_cluster_edges(root)

        # Collect all pass types
        pass_types = list(set(e.pass_type for e in edges))

        # Check if review needed
        needs_review, review_reasons = merger.get_cluster_needs_review(root)

        # Get match percentage
        match_pct = merger.get_cluster_match_pct(root)

        metadata[root] = {
            "cluster_size": len(rows),
            "match_pct": match_pct,
            "pass_types": pass_types,
            "primary_pass": pass_types[0] if pass_types else "unknown",
            "needs_review": needs_review,
            "review_reasons": list(set(review_reasons)) if review_reasons else [],
            "edge_count": len(edges),
            "min_edge_score": min(e.match_pct for e in edges) if edges else 0,
            "max_edge_score": max(e.match_pct for e in edges) if edges else 0,
        }

    return metadata

"""Cluster merging: Union-Find with chain merge protection."""

from typing import Dict, Set, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.config import ClusteringConfig
from src.matching import MatchResult


WEAK_EDGE_TYPES = {
    "address_name_related", "address_secondary", "address_secondary_or_acronym",
    "address_root_brand", "family_same_country", "family_bridge_supported",
    "family_cross_country", "known_family_bridge", "name_fuzzy_address_weak",
    "secondary_or_acronym_bridge", "acronym_review_candidate",
    "known_brand_family_alias",
    # Recall-improvement 70-score LLM candidates — weak by design, chain-blocked.
    "weak_brand_root_candidate",       # same root_brand, no city/domain/address support
    "name_fuzzy_review_candidate",     # high name sim, no location/domain support
    "known_brand_family_weak_candidate",  # known family alias, risky, no support
    "possible_sub_brand_candidate",    # one name is a prefix of the other
    # Phase 3A: domain-only with unrelated names — safe for single pair, chain-blocked.
    # Prevents shared client-contact domains (e.g. pharma.com) from bridging unrelated suppliers.
    "domain_review_candidate",
    # Phase B alias framework: alias-only evidence is inherently weak — chain-blocked to
    # prevent A=alias=B + B=alias=C from auto-merging A with C without direct evidence.
    "brand_alias_candidate",
}

DOMAIN_SUPPORTED_EDGE_TYPES = {"address_domain", "domain_name_related"}

STRONG_EDGE_TYPES = {
    "tax_exact", "tax_loose_supported", "name_address_exact", "name_exact",
    "name_fuzzy_supported", "professional_name_address", "distinctive_supplier_identity",
    "same_legal_owner_confirmed",  # exact non-generic name, different tax/address allowed
    "brand_location_variant_match",  # same brand core, different branch/location modifier
}


@dataclass
class ClusterEdge:
    """An edge between two rows in a cluster."""
    row_a: int
    row_b: int
    match_pct: float
    pass_type: str
    evidence: Dict[str, Any]
    needs_review: bool = False
    review_reason: str = ""


class ClusterMerger:
    """
    Union-Find (Disjoint Set Union) with chain merge protection.

    Prevents weak edges from incorrectly merging strong clusters.
    """

    def __init__(self, n_rows: int, config: ClusteringConfig = None):
        self.n_rows = n_rows
        self.parent = list(range(n_rows))
        self.rank = [0] * n_rows
        self.edges: Dict[int, List[ClusterEdge]] = defaultdict(list)  # cluster_root -> edges
        self.cluster_tax_ids: Dict[int, Set[str]] = {}  # cluster_root -> set of tax IDs
        self.cluster_members: Dict[int, Set[int]] = {i: {i} for i in range(n_rows)}
        self.config = config or ClusteringConfig()

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def _get_cluster_rows(self, root: int) -> Set[int]:
        """Get all rows in a cluster."""
        return self.cluster_members.get(self.find(root), set())

    def _has_strong_evidence_between_clusters(self, root_a: int, root_b: int) -> bool:
        """
        Check if there's strong evidence (tax match or high name sim) 
        between any pair of rows across two clusters.
        """
        rows_a = self._get_cluster_rows(root_a)
        rows_b = self._get_cluster_rows(root_b)

        # Check all edges in both clusters
        all_edges = self.edges.get(root_a, []) + self.edges.get(root_b, [])

        for edge in all_edges:
            if (edge.row_a in rows_a and edge.row_b in rows_b) or                (edge.row_a in rows_b and edge.row_b in rows_a):
                if edge.pass_type in ["tax_exact", "name_address_exact"]:
                    return True
                if edge.match_pct >= 85:
                    return True

        return False

    def _cluster_has_weak_edges(self, root: int) -> bool:
        return any(edge.pass_type in WEAK_EDGE_TYPES for edge in self.edges.get(root, []))

    def cluster_has_only_weak_or_review_edges(self, root: int) -> bool:
        """Return true when a cluster is only connected by review/weak edges."""
        edges = self.edges.get(root, [])
        if not edges:
            return False
        return all(edge.pass_type in WEAK_EDGE_TYPES or edge.needs_review for edge in edges)

    def _is_weak_chain(self, root_a: int, root_b: int, new_edge: ClusterEdge) -> bool:
        """
        Detect if merging would create a risky chain:
        A-B (strong), B-C (weak), C-D (weak) → forcing A and D together
        """
        if new_edge.pass_type in STRONG_EDGE_TYPES:
            return False

        rows_a = self._get_cluster_rows(root_a)
        rows_b = self._get_cluster_rows(root_b)

        total_size = len(rows_a) + len(rows_b)

        if new_edge.pass_type in DOMAIN_SUPPORTED_EDGE_TYPES:
            return total_size > int(getattr(self.config, "max_weak_review_cluster_size", 50))

        if new_edge.pass_type in {"known_family_bridge", "known_brand_family_alias"}:
            return total_size > int(getattr(self.config, "max_known_family_cluster_size", 25))

        if new_edge.pass_type not in WEAK_EDGE_TYPES:
            return False

        if total_size > int(getattr(self.config, "max_low_confidence_cluster_size", 6)):
            return True

        # A weak edge may group two singleton rows for review, but must not
        # become a transitive bridge. This prevents A-B, B-C, C-D weak chains.
        if len(rows_a) > 1 or len(rows_b) > 1:
            return True

        if self._cluster_has_weak_edges(root_a) or self._cluster_has_weak_edges(root_b):
            return True

        return False

    def _has_tax_conflict(self, root_a: int, root_b: int) -> bool:
        """Check if two clusters have conflicting tax IDs."""
        tax_a = self.cluster_tax_ids.get(root_a, set())
        tax_b = self.cluster_tax_ids.get(root_b, set())

        # Remove empty strings
        tax_a = {t for t in tax_a if t}
        tax_b = {t for t in tax_b if t}

        if tax_a and tax_b:
            # If they share any tax ID, no conflict
            if tax_a & tax_b:
                return False
            # Different non-empty tax IDs = conflict
            return True

        return False

    def add_edge(self, row_a: int, row_b: int, match_result: MatchResult) -> str:
        """
        Add a match edge and potentially merge clusters.

        Returns: "MERGED", "BLOCKED_TAX_CONFLICT", "BLOCKED_WEAK_CHAIN", "ALREADY_SAME"
        """
        root_a = self.find(row_a)
        root_b = self.find(row_b)

        if root_a == root_b:
            # Already in same cluster, just add edge
            edge = ClusterEdge(
                row_a=row_a, row_b=row_b,
                match_pct=match_result.match_pct,
                pass_type=match_result.pass_type,
                evidence=match_result.evidence,
                needs_review=match_result.needs_review,
                review_reason=match_result.review_reason
            )
            self.edges[root_a].append(edge)
            return "ALREADY_SAME"

        # Build edge before risk checks
        new_edge = ClusterEdge(
            row_a=row_a, row_b=row_b,
            match_pct=match_result.match_pct,
            pass_type=match_result.pass_type,
            evidence=match_result.evidence,
            needs_review=match_result.needs_review,
            review_reason=match_result.review_reason
        )

        # Check tax conflict. For parent/family clustering, allow selected family edges to bridge
        # different legal identifiers when configured. This is required for cases like
        # Rain Carbon <-> Rütgers Aromatic <-> Ruetgers Chemicals.
        if self._has_tax_conflict(root_a, root_b):
            family_edge = new_edge.pass_type in {
                "family_bridge_supported", "family_cross_country",
                "secondary_or_acronym_bridge", "domain_name_related",
                "address_secondary_or_acronym", "name_address_exact",
                "known_family_bridge", "professional_name_address",
                "known_brand_family_alias", "distinctive_supplier_identity",
                "same_legal_owner_confirmed",   # same legal name, different tax records
                "address_distinctive_shared",   # distinctive token + address, different tax
                "brand_location_variant_match",  # same brand core, branch/location variant
            }
            if not (self.config.allow_parent_family_tax_conflicts and family_edge):
                return "BLOCKED_TAX_CONFLICT"

        if self._is_weak_chain(root_a, root_b, new_edge):
            return "BLOCKED_WEAK_CHAIN"

        # Merge clusters (union by rank)
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        # Merge edges
        self.edges[root_a].extend(self.edges.get(root_b, []))
        self.edges[root_a].append(new_edge)
        if root_b in self.edges:
            del self.edges[root_b]

        # Merge member sets. Keeping this cache avoids scanning every row for
        # weak-chain checks on large files.
        self.cluster_members[root_a].update(self.cluster_members.get(root_b, set()))
        if root_b in self.cluster_members:
            del self.cluster_members[root_b]

        # Merge tax IDs
        tax_a = self.cluster_tax_ids.get(root_a, set())
        tax_b = self.cluster_tax_ids.get(root_b, set())
        self.cluster_tax_ids[root_a] = tax_a | tax_b
        if root_b in self.cluster_tax_ids:
            del self.cluster_tax_ids[root_b]

        return "MERGED"

    def set_row_tax(self, row_id: int, tax_id: str):
        """Set one or more tax IDs for a row. Pipe-separated IDs are supported."""
        root = self.find(row_id)
        if root not in self.cluster_tax_ids:
            self.cluster_tax_ids[root] = set()
        if tax_id:
            for t in str(tax_id).split("|"):
                if t:
                    self.cluster_tax_ids[root].add(t)

    def get_clusters(self) -> Dict[int, Set[int]]:
        """Get final clusters as dict: root -> set of row_ids."""
        clusters: Dict[int, Set[int]] = defaultdict(set)
        for root, rows in self.cluster_members.items():
            clusters[self.find(root)].update(rows)
        return dict(clusters)

    def get_cluster_edges(self, root: int) -> List[ClusterEdge]:
        """Get all edges in a cluster."""
        return self.edges.get(root, [])

    def get_cluster_match_pct(self, root: int) -> float:
        """
        Get cluster match percentage using weakest-link principle.
        Returns minimum match_pct of all edges in cluster.
        """
        edges = self.edges.get(root, [])
        if not edges:
            return 0.0
        return min(e.match_pct for e in edges)

    def get_cluster_needs_review(self, root: int) -> Tuple[bool, List[str]]:
        """Check if cluster has any edges needing review."""
        edges = self.edges.get(root, [])
        review_reasons = []
        for e in edges:
            if e.needs_review and e.review_reason:
                review_reasons.append(e.review_reason)
        return len(review_reasons) > 0, review_reasons

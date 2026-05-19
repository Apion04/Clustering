"""Group-level LLM review request and decision application helpers.

The production backend can call these helpers after deterministic clustering:

1. Build compact 70-score review groups.
2. Send those groups to an LLM provider outside this module.
3. Apply returned approve/reject/split/merge/promote decisions.

This module does not call an LLM directly. That keeps the merge semantics easy
to unit test and prevents provider failures from corrupting deterministic
cluster state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


ALLOWED_LLM_DECISIONS = {"approve", "reject", "split", "merge_with_existing", "promote_score", "uncertain"}


class _DSU:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_llm_review_groups(
    cluster_map: Dict[int, int],
    user_scores: Dict[int, float],
    rows_dict: Dict[int, Dict[str, Any]],
    merger: Any = None,
) -> List[Dict[str, Any]]:
    """Build compact request payloads for unresolved 70-score clusters."""
    rows_by_root: Dict[int, List[int]] = defaultdict(list)
    for row_id, root in cluster_map.items():
        if int(user_scores.get(root, 0) or 0) == 70:
            rows_by_root[int(root)].append(int(row_id))

    groups = []
    for group_idx, (root, row_ids) in enumerate(sorted(rows_by_root.items(), key=lambda item: min(item[1])), 1):
        records = []
        for row_id in sorted(row_ids):
            row = rows_dict.get(row_id, {})
            records.append({
                "row_id": row_id,
                "supplier_name": row.get("orig_supplier_name") or row.get("name_norm") or "",
                "address": row.get("orig_address") or row.get("addr_norm") or "",
                "city": row.get("orig_city") or row.get("city_norm") or "",
                "country": row.get("orig_country") or row.get("country_norm") or "",
                "tax_ids": row.get("tax_norm") or "",
                "domain": row.get("domain") or "",
                "supplier_identity_core": row.get("supplier_identity_core") or "",
            })
        pass_types = []
        raw_scores = []
        if merger is not None:
            for edge in merger.get_cluster_edges(root):
                if edge.pass_type not in pass_types:
                    pass_types.append(edge.pass_type)
                raw_scores.append(edge.match_pct)
        groups.append({
            "candidate_group_id": f"llm-{group_idx}",
            "cluster_root": root,
            "row_ids": sorted(row_ids),
            "records": records,
            "deterministic_pass_types": pass_types,
            "raw_scores": raw_scores,
            "initial_user_score": 70,
            "reason_for_llm_review": "70-score weak/ambiguous supplier-family candidate requires LLM decision before production finalization",
        })
    return groups


def build_llm_review_groups_from_review_candidates(
    review_candidates: List[Dict[str, Any]],
    rows_dict: Dict[int, Dict[str, Any]],
    *,
    start_index: int = 1,
) -> List[Dict[str, Any]]:
    """Build LLM groups from pair/group candidates routed outside union-find."""
    groups = []
    seen = set()
    idx = start_index
    for cand in review_candidates or []:
        evidence = cand.get("evidence") or {}
        score = float(cand.get("match_pct") or 0.0)
        route = evidence.get("route")
        pass_type = str(cand.get("pass_type") or "")
        should_send = (
            route == "LLM_REVIEW"
            or 65.0 <= score <= 75.0
            or pass_type in {
                "tax_exact_institutional_ecosystem_review",
                "regulatory_or_task_force_related",
                "person_address_city_mismatch_review",
                "person_same_name_location_review",
            }
        )
        if not should_send:
            continue
        row_ids = sorted({int(cand.get("row_a")), int(cand.get("row_b"))})
        key = tuple(row_ids + [pass_type])
        if key in seen:
            continue
        seen.add(key)
        records = []
        for row_id in row_ids:
            row = rows_dict.get(row_id, {})
            records.append({
                "row_id": row_id,
                "supplier_name": row.get("orig_supplier_name") or row.get("name_norm") or "",
                "address": row.get("orig_address") or row.get("addr_norm") or "",
                "city": row.get("orig_city") or row.get("city_norm") or "",
                "country": row.get("orig_country") or row.get("country_norm") or "",
                "tax_ids": row.get("tax_norm") or "",
                "domain": row.get("domain") or "",
                "supplier_identity_core": row.get("supplier_identity_core") or "",
            })
        groups.append({
            "candidate_group_id": f"llm-{idx}",
            "cluster_root": "",
            "row_ids": row_ids,
            "records": records,
            "deterministic_pass_types": [pass_type],
            "raw_scores": [score],
            "initial_user_score": 70,
            "reason_for_llm_review": cand.get("review_reason") or "Review-queue weak/ambiguous candidate requires LLM decision before production finalization",
            "evidence": evidence,
        })
        idx += 1
    return groups


def apply_llm_decisions(
    n_rows: int,
    cluster_map: Dict[int, int],
    user_scores: Dict[int, float],
    decisions: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], Dict[int, float], Dict[str, int]]:
    """Apply LLM decisions to final cluster assignments.

    Returns:
        final_cluster_map: row_id -> contiguous final cluster number
        final_cluster_scores: final cluster number -> user-facing score
        counts: decision counts and errors
    """
    counts = {"approve": 0, "reject": 0, "split": 0, "merge_with_existing": 0, "promote_score": 0, "uncertain": 0, "errors": 0}

    # Start from deterministic nonblank clusters.
    row_to_group: Dict[int, int] = {}
    group_scores: Dict[int, float] = {}
    for row_id, root in cluster_map.items():
        score = float(user_scores.get(root, 0) or 0)
        if score <= 0:
            continue
        row_to_group[int(row_id)] = int(root)
        group_scores[int(root)] = score

    rejected_rows = set()
    next_group = (max(group_scores) + 1) if group_scores else n_rows + 1

    def new_group(rows: Iterable[int], score: float) -> int:
        nonlocal next_group
        gid = next_group
        next_group += 1
        for rid in rows:
            row_to_group[int(rid)] = gid
            rejected_rows.discard(int(rid))
        group_scores[gid] = score
        return gid

    for decision in decisions or []:
        action = str(decision.get("decision") or "").lower()
        if action not in ALLOWED_LLM_DECISIONS:
            counts["errors"] += 1
            continue
        counts[action] += 1
        row_ids = [int(x) for x in decision.get("row_ids", []) if str(x).strip() != ""]
        score = _decision_score(decision)

        if action in {"reject", "uncertain"}:
            for rid in row_ids:
                rejected_rows.add(rid)
                row_to_group.pop(rid, None)
            continue

        if action in {"approve", "promote_score"}:
            if not row_ids:
                counts["errors"] += 1
                continue
            base_group = row_to_group.get(row_ids[0])
            if base_group is None:
                base_group = new_group(row_ids, score)
            for rid in row_ids:
                row_to_group[rid] = base_group
                rejected_rows.discard(rid)
            group_scores[base_group] = max(score, group_scores.get(base_group, 0.0))
            continue

        if action == "split":
            for rid in row_ids:
                row_to_group.pop(rid, None)
            for subgroup in decision.get("clusters", []) or []:
                subgroup_rows = [int(x) for x in subgroup.get("row_ids", [])]
                if len(subgroup_rows) >= 2:
                    new_group(subgroup_rows, _decision_score(subgroup, default=score))
            continue

        if action == "merge_with_existing":
            target = decision.get("target_cluster_number")
            if target is None:
                target = decision.get("target_cluster")
            if target is None:
                target = decision.get("target_group")
            if target is None:
                counts["errors"] += 1
                continue
            target = int(target)
            for rid in row_ids:
                row_to_group[rid] = target
                rejected_rows.discard(rid)
            group_scores[target] = max(score, group_scores.get(target, 85.0))

    # Re-contiguize cluster numbers by first row anchor.
    grouped: Dict[int, List[int]] = defaultdict(list)
    for row_id, group in row_to_group.items():
        if row_id not in rejected_rows:
            grouped[group].append(row_id)
    ordered_groups = sorted(grouped, key=lambda group: min(grouped[group]))
    group_to_number = {group: idx + 1 for idx, group in enumerate(ordered_groups)}
    final_map = {}
    final_scores = {}
    for group in ordered_groups:
        number = group_to_number[group]
        for row_id in grouped[group]:
            final_map[row_id] = number
        final_scores[number] = _normalize_score(group_scores.get(group, 85.0))
    return final_map, final_scores, counts


def _decision_score(decision: Dict[str, Any], default: float = 85.0) -> float:
    value = decision.get("promote_score") or decision.get("match_percentage") or decision.get("score") or default
    return _normalize_score(float(value))


def _normalize_score(value: float) -> float:
    if value >= 95:
        return 98.0
    if value >= 80:
        return 85.0
    if value >= 65:
        return 70.0
    return 0.0

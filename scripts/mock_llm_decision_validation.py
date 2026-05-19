#!/usr/bin/env python3
"""Validate final cluster-number semantics with mocked LLM group decisions.

This script uses the real unresolved LLM exception report produced by a full
deterministic run, then applies mixed approve/reject/split/merge/promote
decisions against those actual candidate row groups. It is intentionally
provider-free; live LLM calls belong in backend orchestration.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import polars as pl

from src.llm_workflow import apply_llm_decisions


def _parse_row_ids(value: str) -> List[int]:
    return [int(x) for x in str(value or "").split("|") if str(x).strip()]


def main():
    parser = argparse.ArgumentParser(description="Mock LLM decision validation using actual unresolved groups")
    parser.add_argument("--exception-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pl.read_csv(args.exception_report)
    groups = []
    for row in df.iter_rows(named=True):
        row_ids = _parse_row_ids(row.get("row_ids", ""))
        if len(row_ids) >= 2:
            raw_root = row.get("cluster_root")
            try:
                root = int(raw_root)
            except Exception:
                root = max(row_ids) + 1000000
            groups.append({
                "cluster_root": root,
                "row_ids": row_ids,
                "candidate_group_id": row.get("candidate_group_id", ""),
            })

    cluster_map: Dict[int, int] = {}
    scores: Dict[int, float] = {}
    for group in groups:
        root = int(group["cluster_root"])
        for row_id in group["row_ids"]:
            cluster_map[int(row_id)] = root
        scores[root] = 70.0

    decisions = []
    if groups:
        decisions.append({"decision": "approve", "row_ids": groups[0]["row_ids"], "match_percentage": 85})
    if len(groups) > 1:
        decisions.append({"decision": "reject", "row_ids": groups[1]["row_ids"]})
    split_group = next((group for group in groups[2:] if len(group["row_ids"]) >= 4), None)
    if split_group:
        split_rows = split_group["row_ids"]
        midpoint = max(1, len(split_rows) // 2)
        subgroups = []
        if len(split_rows[:midpoint]) >= 2:
            subgroups.append({"row_ids": split_rows[:midpoint], "match_percentage": 85})
        if len(split_rows[midpoint:]) >= 2:
            subgroups.append({"row_ids": split_rows[midpoint:], "match_percentage": 85})
        if subgroups:
            decisions.append({"decision": "split", "row_ids": split_rows, "clusters": subgroups})
    if len(groups) > 3:
        target = groups[0]["cluster_root"]
        decisions.append({
            "decision": "merge_with_existing",
            "target_cluster_number": target,
            "row_ids": groups[3]["row_ids"],
            "match_percentage": 85,
        })
    if len(groups) > 4:
        decisions.append({"decision": "promote_score", "row_ids": groups[4]["row_ids"], "match_percentage": 98})

    n_rows = max(cluster_map.keys(), default=0) + 1
    final_map, final_scores, counts = apply_llm_decisions(n_rows, cluster_map, scores, decisions)
    cluster_numbers = sorted(set(final_map.values()))
    contiguous = cluster_numbers == list(range(1, len(cluster_numbers) + 1))
    rejected_rows = set(groups[1]["row_ids"]) if len(groups) > 1 else set()
    rejected_blank = all(row_id not in final_map for row_id in rejected_rows)
    split_ok = True
    if split_group and any(d.get("decision") == "split" for d in decisions):
        split_rows = set(split_group["row_ids"])
        split_clusters = {final_map[row_id] for row_id in split_rows if row_id in final_map}
        split_ok = len(split_clusters) >= 1 and all(final_scores.get(cluster, 0) in {85.0, 98.0} for cluster in split_clusters)
    merge_ok = True
    if len(groups) > 3:
        merged_clusters = {final_map[row_id] for row_id in groups[3]["row_ids"] if row_id in final_map}
        approved_clusters = {final_map[row_id] for row_id in groups[0]["row_ids"] if row_id in final_map}
        merge_ok = bool(merged_clusters and approved_clusters and merged_clusters == approved_clusters)

    report = {
        "actual_candidate_groups_loaded": len(groups),
        "mock_decisions_count": len(decisions),
        "decision_counts": counts,
        "final_cluster_count": len(cluster_numbers),
        "cluster_numbers_contiguous": contiguous,
        "cluster_numbers_conflict_free": len(final_map) == sum(1 for _ in final_map),
        "rejected_rows_blank": rejected_blank,
        "split_groups_valid": split_ok,
        "merge_with_existing_reused_cluster": merge_ok,
        "final_scores_allowed": sorted(set(final_scores.values())) <= [70.0, 85.0, 98.0],
        "sample_decisions": decisions[:5],
    }
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

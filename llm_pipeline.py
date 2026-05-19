"""Backend LLM queue, execution, validation, and final-output orchestration."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import polars as pl

from src.config import COMMON_FIRST_NAMES, GENERIC_ROOT_TOKENS, LOCATION_ROOT_TOKENS, ClusteringConfig
from src.llm_workflow import apply_llm_decisions
from src.openai_client import OpenAIReviewClient, build_openai_request, write_jsonl
from src.sorting import sort_by_clusters


def run_llm_backend_flow(result: Dict[str, Any], config: ClusteringConfig, final_output_path: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    groups = result.get("llm_review_groups") or []
    rows_dict = {int(row["row_id"]): row for row in result["preprocessed_df"].iter_rows(named=True)}
    deterministic_cluster_map = result.get("cluster_map") or {}
    pre_scores = result.get("pre_llm_match_pcts") or {}
    queue = build_llm_queue(groups, rows_dict, deterministic_cluster_map, pre_scores, config)

    queue_csv = os.path.join(output_dir, "llm_queue_breakdown.csv")
    queue_md = os.path.join(output_dir, "llm_queue_breakdown.md")
    save_llm_queue_breakdown(queue, queue_csv, queue_md, config)

    cost = estimate_llm_cost(queue["groups_to_send"], config)
    cost_path = os.path.join(output_dir, "llm_cost_estimate.json")
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost, f, ensure_ascii=False, indent=2)

    requests_path = os.path.join(output_dir, "llm_review_requests.jsonl")
    responses_path = os.path.join(output_dir, "llm_review_responses.jsonl")
    application_path = os.path.join(output_dir, "llm_decision_application_report.json")
    conflict_path = os.path.join(output_dir, "llm_conflict_resolution_report.csv")
    batch_requests_path = os.path.join(output_dir, "llm_batch_requests.jsonl")
    batch_manifest_path = os.path.join(output_dir, "llm_batch_manifest.json")

    mode = _normalize_mode(getattr(config, "llm_execution_mode", "disabled"))
    selected_model = getattr(config, "openai_model", None) or getattr(config, "ai_model", "gpt-5.5")
    write_jsonl(requests_path, [serialize_llm_request(group, selected_model) for group in queue["groups_to_send"]])

    job_status = "INCOMPLETE_UNRESOLVED_LLM_CANDIDATES"
    raw_responses: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    api_errors: List[str] = []

    if mode in {"disabled", "off", "none"}:
        raw_responses = []
    elif mode == "batch":
        write_jsonl(batch_requests_path, [serialize_batch_request(group, selected_model) for group in queue["groups_to_send"]])
        with open(batch_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "selected_openai_model": selected_model,
                "request_file": batch_requests_path,
                "response_file_expected": os.path.join(output_dir, "llm_batch_responses.jsonl"),
                "groups": len(queue["groups_to_send"]),
                "submission_note": "Submit llm_batch_requests.jsonl to OpenAI Batch API, then place completed response JSONL at response_file_expected and rerun application.",
            }, f, ensure_ascii=False, indent=2)
    elif mode == "mock":
        mock_target = _first_existing_cluster_root(deterministic_cluster_map, pre_scores)
        deterministic_98_rows_for_mock = _deterministic_98_rows(deterministic_cluster_map, pre_scores)
        raw_responses = [
            mock_llm_response(group, idx, mock_target, deterministic_98_rows_for_mock)
            for idx, group in enumerate(queue["groups_to_send"])
        ]
    elif mode == "live":
        if not cost["allowed_to_run"]:
            job_status = "INCOMPLETE_LLM_COST_CAP_EXCEEDED"
        else:
            client = OpenAIReviewClient(
                api_key=getattr(config, "ai_api_key", ""),
                model=selected_model,
                base_url=getattr(config, "ai_base_url", "https://api.openai.com/v1"),
                timeout=getattr(config, "llm_timeout_seconds", 60),
                retries=getattr(config, "llm_retry_count", 2),
            )
            for group in queue["groups_to_send"]:
                try:
                    raw_responses.append(client.review_group(group))
                except Exception as exc:
                    api_errors.append(f"{group.get('candidate_group_id')}: {exc}")
                    raw_responses.append({
                        "candidate_group_id": group.get("candidate_group_id"),
                        "decision": "uncertain",
                        "final_score": None,
                        "clusters": [],
                        "rejected_row_ids": group.get("row_ids", []),
                        "target_cluster_number": None,
                        "reasoning": f"OpenAI call failed: {exc}",
                        "confidence": 0.0,
                    })
    else:
        api_errors.append(f"Unknown LLM execution mode: {mode}")

    write_jsonl(responses_path, raw_responses)
    deterministic_98_rows = {
        int(row_id)
        for row_id, root in deterministic_cluster_map.items()
        if int(pre_scores.get(root, 0) or 0) == 98
    }
    existing_roots = set(int(root) for root in deterministic_cluster_map.values())
    decisions, validation_report = validate_llm_responses(
        raw_responses,
        queue["groups_by_id"],
        existing_roots,
        deterministic_98_rows,
        bool(getattr(config, "override_llm_can_modify_98", False)),
    )

    if decisions:
        final_map, final_scores, counts = apply_llm_decisions(
            len(result["preprocessed_df"]),
            deterministic_cluster_map,
            pre_scores,
            decisions,
        )
    else:
        final_map = result.get("output_cluster_map") or {}
        final_scores = result.get("output_match_pcts") or {}
        counts = {"approve": 0, "reject": 0, "split": 0, "merge_with_existing": 0, "promote_score": 0, "uncertain": 0, "errors": 0}

    responses_received = len(raw_responses)
    unresolved = max(0, len(queue["groups_to_send"]) - responses_received)
    if mode == "mock" and unresolved == 0:
        job_status = "COMPLETE"
    elif mode == "live" and not api_errors and unresolved == 0 and cost["allowed_to_run"]:
        job_status = "COMPLETE"
    elif mode == "batch":
        job_status = "INCOMPLETE_BATCH_PENDING"
    elif mode == "disabled":
        job_status = "INCOMPLETE_UNRESOLVED_LLM_CANDIDATES"

    final_df = sort_by_clusters(result["preprocessed_df"], final_map, final_scores)
    if job_status == "COMPLETE":
        final_df = final_df.with_columns(
            pl.when(pl.col("Match Percentage") == "70%")
            .then(pl.lit(""))
            .otherwise(pl.col("Match Percentage"))
            .alias("Match Percentage")
        )
    final_df.write_csv(final_output_path)

    unresolved_path = os.path.join(output_dir, "unresolved_llm_exception_report.csv")
    if job_status != "COMPLETE" and queue["groups_to_send"]:
        save_unresolved_llm_exception_report(queue["groups_to_send"], unresolved_path, job_status)
    else:
        save_unresolved_llm_exception_report([], unresolved_path, job_status)

    save_conflict_report(validation_report, conflict_path)
    app_report = {
        "job_status": job_status,
        "mode": mode,
        "selected_openai_model": selected_model,
        "raw_groups_generated": queue["raw_groups_generated"],
        "deduped_groups_sent": len(queue["groups_to_send"]),
        "excluded_groups": len(queue["excluded_groups"]),
        "llm_decision_counts": counts,
        "validation": validation_report,
        "unresolved_group_count": unresolved if job_status != "COMPLETE" else 0,
        "api_errors": api_errors,
        "final_score_distribution": _score_distribution(final_df),
        "cluster_numbers_contiguous": _cluster_numbers_contiguous(final_df),
        "final_output_columns": final_df.columns,
        "allowed_final_scores": _final_scores_allowed(final_df, job_status),
        "final_output_path": final_output_path,
        "cost_estimate": cost,
    }
    with open(application_path, "w", encoding="utf-8") as f:
        json.dump(app_report, f, ensure_ascii=False, indent=2)

    return {
        "job_status": job_status,
        "final_df": final_df,
        "final_output_path": final_output_path,
        "queue": queue,
        "cost_estimate": cost,
        "decision_application_report": app_report,
        "paths": {
            "llm_queue_breakdown_csv": queue_csv,
            "llm_queue_breakdown_md": queue_md,
            "llm_cost_estimate": cost_path,
            "llm_review_requests": requests_path,
            "llm_review_responses": responses_path,
            "llm_decision_application_report": application_path,
            "llm_conflict_resolution_report": conflict_path,
            "unresolved_llm_exception_report": unresolved_path,
            "llm_batch_requests": batch_requests_path if mode == "batch" else "",
            "llm_batch_manifest": batch_manifest_path if mode == "batch" else "",
        },
    }


def build_llm_queue(groups: List[Dict[str, Any]], rows_dict: Dict[int, Dict[str, Any]], cluster_map: Dict[int, int], scores: Dict[int, float], config: ClusteringConfig) -> Dict[str, Any]:
    max_rows = int(getattr(config, "max_rows_per_llm_group", 60) or 60)
    max_groups = int(getattr(config, "max_llm_groups_per_job", 0) or 0)
    max_tokens = int(getattr(config, "max_tokens_per_llm_group", 0) or 0)
    raw = list(groups or [])
    exact_seen = set()
    deduped = []
    excluded = []
    for group in raw:
        row_ids = sorted({int(x) for x in group.get("row_ids", []) if str(x).strip() != ""})
        if len(row_ids) < 2:
            excluded.append({"candidate_group_id": group.get("candidate_group_id", ""), "row_ids": row_ids, "exclusion_reason": "too_few_rows"})
            continue
        key = tuple(row_ids)
        if key in exact_seen:
            excluded.append({"candidate_group_id": group.get("candidate_group_id", ""), "row_ids": row_ids, "exclusion_reason": "duplicate_row_set"})
            continue
        exact_seen.add(key)
        if _group_is_generic_location_only(group, rows_dict):
            excluded.append({"candidate_group_id": group.get("candidate_group_id", ""), "row_ids": row_ids, "exclusion_reason": "generic_or_location_only_rejected"})
            continue
        item = dict(group)
        item["row_ids"] = row_ids
        deduped.append(item)

    # Merge overlapping pair-level candidates into components when the merged
    # group stays below the configured payload size.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in deduped:
        ids = group["row_ids"]
        if len(ids) <= max_rows:
            for rid in ids[1:]:
                union(ids[0], rid)
    components = defaultdict(list)
    for group in deduped:
        root = find(group["row_ids"][0])
        components[root].append(group)

    merged = []
    for comp_groups in components.values():
        row_ids = sorted({rid for group in comp_groups for rid in group["row_ids"]})
        if len(row_ids) > max_rows:
            for group in comp_groups:
                group["exclusion_reason"] = ""
                merged.append(group)
            continue
        records = [_record_for_row(rows_dict, rid) for rid in row_ids]
        pass_types = sorted({pt for group in comp_groups for pt in group.get("deterministic_pass_types", [])})
        priority = _priority_for_group(row_ids, comp_groups, cluster_map, scores)
        item = {
            "candidate_group_id": f"llm-q-{len(merged) + 1}",
            "source_candidate_group_ids": [g.get("candidate_group_id", "") for g in comp_groups],
            "row_ids": row_ids,
            "records": records,
            "deterministic_pass_types": pass_types,
            "raw_scores": [s for group in comp_groups for s in group.get("raw_scores", [])],
            "initial_user_score": 70,
            "priority": priority,
            "reason_for_llm_review": "; ".join(sorted({str(g.get("reason_for_llm_review") or "") for g in comp_groups if g.get("reason_for_llm_review")}))[:1000],
            "overlap_with_existing_85_98_clusters": _overlaps_existing_85_98(row_ids, cluster_map, scores),
        }
        if max_tokens > 0 and _estimate_tokens(item) > max_tokens:
            excluded.append({
                "candidate_group_id": item.get("candidate_group_id", ""),
                "row_ids": row_ids,
                "exclusion_reason": "max_tokens_per_llm_group",
            })
            continue
        merged.append(item)

    if max_groups > 0 and len(merged) > max_groups:
        for group in merged[max_groups:]:
            excluded.append({"candidate_group_id": group.get("candidate_group_id", ""), "row_ids": group.get("row_ids", []), "exclusion_reason": "max_llm_groups_per_job"})
        merged = merged[:max_groups]
    groups_by_id = {group["candidate_group_id"]: group for group in merged}
    return {
        "raw_groups_generated": len(raw),
        "deduped_groups_sent": len(merged),
        "groups_to_send": merged,
        "excluded_groups": excluded,
        "groups_by_id": groups_by_id,
        "priority_counts": dict(_count_by(merged, "priority")),
    }


def estimate_llm_cost(groups: List[Dict[str, Any]], config: ClusteringConfig) -> Dict[str, Any]:
    input_tokens = sum(_estimate_tokens(group) for group in groups)
    output_tokens = len(groups) * 350
    input_rate = float(getattr(config, "openai_input_cost_per_1m_tokens", 0.0) or 0.0)
    output_rate = float(getattr(config, "openai_output_cost_per_1m_tokens", 0.0) or 0.0)
    cap = float(getattr(config, "max_total_llm_cost_per_job", 0.0) or 0.0)
    pricing_missing = input_rate <= 0.0 or output_rate <= 0.0
    allow_unknown = bool(getattr(config, "allow_unknown_llm_cost", False))
    estimated = None if pricing_missing else (input_tokens / 1_000_000.0) * input_rate + (output_tokens / 1_000_000.0) * output_rate
    blocked_unknown_cost = bool(pricing_missing and cap > 0 and not allow_unknown)
    allowed = bool(not blocked_unknown_cost and (pricing_missing or cap <= 0 or float(estimated or 0.0) <= cap))
    if pricing_missing:
        note = (
            "Pricing is missing. Estimated cost is UNKNOWN, not zero/free. "
            "Set OPENAI_INPUT_COST_PER_1M_TOKENS and OPENAI_OUTPUT_COST_PER_1M_TOKENS before live mode. "
            "When MAX_TOTAL_LLM_COST_PER_JOB is set, live mode is blocked unless ALLOW_UNKNOWN_LLM_COST=true."
        )
    else:
        note = "Cost estimate uses configured OPENAI_INPUT_COST_PER_1M_TOKENS and OPENAI_OUTPUT_COST_PER_1M_TOKENS."
    return {
        "selected_openai_model": getattr(config, "openai_model", None) or getattr(config, "ai_model", "gpt-5.5"),
        "groups_to_send": len(groups),
        "estimated_input_tokens": int(input_tokens),
        "estimated_output_tokens": int(output_tokens),
        "estimated_cost_usd": None if estimated is None else round(float(estimated), 4),
        "cost_estimate_known": not pricing_missing,
        "pricing_missing": pricing_missing,
        "configured_cost_cap": cap,
        "allow_unknown_llm_cost": allow_unknown,
        "blocked_unknown_cost": blocked_unknown_cost,
        "allowed_to_run": allowed,
        "cost_rates_note": note,
    }


def serialize_llm_request(group: Dict[str, Any], model: str) -> Dict[str, Any]:
    return {"candidate_group_id": group.get("candidate_group_id"), "request": build_openai_request(group, model)}


def serialize_batch_request(group: Dict[str, Any], model: str) -> Dict[str, Any]:
    return {
        "custom_id": str(group.get("candidate_group_id")),
        "method": "POST",
        "url": "/v1/responses",
        "body": build_openai_request(group, model),
    }


def mock_llm_response(group: Dict[str, Any], idx: int, merge_target: int | None = None, protected_rows: set[int] | None = None) -> Dict[str, Any]:
    row_ids = [int(x) for x in group.get("row_ids", [])]
    has_protected_98 = bool((protected_rows or set()) & set(row_ids))
    if has_protected_98:
        decision, clusters, rejected, final_score = "approve", [{"row_ids": row_ids, "final_score": 85, "reasoning": "Mock leaves protected deterministic rows intact"}], [], 85
        target = None
    elif merge_target is not None and idx % 101 == 0 and idx > 0:
        decision, clusters, rejected, final_score = "merge_with_existing", [], [], 85
        target = int(merge_target)
    elif idx % 97 == 0 and len(row_ids) >= 4:
        midpoint = len(row_ids) // 2
        decision = "split"
        clusters = [
            {"row_ids": row_ids[:midpoint], "final_score": 85, "reasoning": "Mock split subgroup A"},
            {"row_ids": row_ids[midpoint:], "final_score": 85, "reasoning": "Mock split subgroup B"},
        ]
        rejected = []
        final_score = None
        target = None
    elif idx % 89 == 0 and not _group_has_no_strong_evidence(group):
        decision, clusters, rejected, final_score = "promote_score", [], [], 98
        target = None
    elif idx % 37 == 0:
        decision, clusters, rejected, final_score = "uncertain", [], row_ids, None
        target = None
    elif idx % 11 == 0:
        decision, clusters, rejected, final_score = "reject", [], row_ids, None
        target = None
    else:
        decision, clusters, rejected, final_score = "approve", [{"row_ids": row_ids, "final_score": 85, "reasoning": "Mock-approved plausible supplier relation"}], [], 85
        target = None
    return {
        "candidate_group_id": group.get("candidate_group_id"),
        "decision": decision,
        "final_score": final_score,
        "clusters": clusters,
        "rejected_row_ids": rejected,
        "target_cluster_number": target,
        "reasoning": f"Mock {decision} decision for pipeline validation",
        "confidence": 0.85 if decision in {"approve", "promote_score", "split"} else 0.45,
    }


def validate_llm_responses(responses: List[Dict[str, Any]], groups_by_id: Dict[str, Dict[str, Any]], existing_roots: set, deterministic_98_rows: set, override_98: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid = []
    report = []
    seen_rows = set()
    for response in responses:
        group_id = str(response.get("candidate_group_id") or "")
        group = groups_by_id.get(group_id)
        if not group:
            report.append(_validation_row(group_id, "reject_invalid", "candidate_group_id not found"))
            continue
        decision = str(response.get("decision") or "").lower()
        row_ids = [int(x) for x in group.get("row_ids", [])]
        final_score = response.get("final_score")
        reason = ""
        if decision not in {"approve", "reject", "split", "merge_with_existing", "promote_score", "uncertain"}:
            reason = "invalid decision"
        elif final_score not in {85, 98, None}:
            reason = "invalid final_score"
        elif not override_98 and (set(row_ids) & deterministic_98_rows) and decision in {"reject", "split", "merge_with_existing"}:
            reason = "deterministic 98 cluster protected"
        elif decision == "merge_with_existing" and response.get("target_cluster_number") not in existing_roots:
            reason = "merge target cluster does not exist"
        elif decision == "split" and _split_has_overlap_or_drops(response, set(row_ids)):
            reason = "split groups overlap or drop rows without explicit rejection"
        elif decision == "promote_score" and final_score == 98 and _group_has_no_strong_evidence(group):
            reason = "promote_score to 98 lacks deterministic support"
        elif seen_rows & set(row_ids):
            reason = "row already appeared in earlier LLM decision"
        if reason:
            report.append(_validation_row(group_id, "rejected", reason))
            continue
        seen_rows |= set(row_ids)
        decision_row = {
            "decision": "reject" if decision == "uncertain" else decision,
            "row_ids": row_ids if decision != "split" else row_ids,
            "match_percentage": final_score or 0,
            "reasoning": response.get("reasoning", ""),
        }
        if decision == "split":
            decision_row["clusters"] = [
                {"row_ids": [int(x) for x in c.get("row_ids", [])], "match_percentage": c.get("final_score") or 85}
                for c in response.get("clusters", [])
            ]
        if decision == "merge_with_existing":
            decision_row["target_cluster_number"] = int(response.get("target_cluster_number"))
        valid.append(decision_row)
        report.append(_validation_row(group_id, "accepted", ""))
    return valid, report


def save_llm_queue_breakdown(queue: Dict[str, Any], csv_path: str, md_path: str, config: ClusteringConfig) -> None:
    rows = []
    for group in queue["groups_to_send"]:
        indicators = _risk_indicators_for_records(group.get("records", []))
        rows.append({
            "candidate_group_id": group.get("candidate_group_id", ""),
            "status": "send",
            "priority": group.get("priority", "P3"),
            "row_count": len(group.get("row_ids", [])),
            "row_ids": "|".join(str(x) for x in group.get("row_ids", [])),
            "exclusion_reason": "",
            "selected_openai_model": getattr(config, "openai_model", "gpt-5.5"),
            "estimated_tokens": _estimate_tokens(group),
            "risky_generic_location_indicators": "|".join(indicators),
            "overlap_with_existing_85_98_clusters": bool(group.get("overlap_with_existing_85_98_clusters", False)),
        })
    for item in queue["excluded_groups"]:
        rows.append({
            "candidate_group_id": item.get("candidate_group_id", ""),
            "status": "excluded",
            "priority": "",
            "row_count": len(item.get("row_ids", [])),
            "row_ids": "|".join(str(x) for x in item.get("row_ids", [])),
            "exclusion_reason": item.get("exclusion_reason", ""),
            "selected_openai_model": getattr(config, "openai_model", "gpt-5.5"),
            "estimated_tokens": 0,
            "risky_generic_location_indicators": "",
            "overlap_with_existing_85_98_clusters": False,
        })
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    pl.DataFrame(rows if rows else [{"candidate_group_id": "", "status": "", "priority": "", "row_count": 0, "row_ids": "", "exclusion_reason": "", "selected_openai_model": getattr(config, "openai_model", "gpt-5.5"), "estimated_tokens": 0, "risky_generic_location_indicators": "", "overlap_with_existing_85_98_clusters": False}]).write_csv(csv_path)
    by_priority = _count_by(queue["groups_to_send"], "priority")
    sizes = [len(g.get("row_ids", [])) for g in queue["groups_to_send"]]
    lines = [
        "# LLM Queue Breakdown",
        "",
        f"- raw groups generated: {queue['raw_groups_generated']:,}",
        f"- deduped groups sent: {len(queue['groups_to_send']):,}",
        f"- excluded groups: {len(queue['excluded_groups']):,}",
        f"- selected OpenAI model: {getattr(config, 'openai_model', 'gpt-5.5')}",
        f"- groups by priority: {dict(by_priority)}",
        f"- group size min/max: {min(sizes) if sizes else 0}/{max(sizes) if sizes else 0}",
        f"- estimated tokens: {sum(_estimate_tokens(g) for g in queue['groups_to_send']):,}",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_conflict_report(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    pl.DataFrame(rows if rows else [{"candidate_group_id": "", "status": "", "reason": ""}]).write_csv(path)


def save_unresolved_llm_exception_report(groups: List[Dict[str, Any]], path: str, job_status: str) -> None:
    rows = []
    for group in groups:
        rows.append({
            "job_status": job_status,
            "candidate_group_id": group.get("candidate_group_id", ""),
            "priority": group.get("priority", ""),
            "row_count": len(group.get("row_ids", [])),
            "row_ids": "|".join(str(x) for x in group.get("row_ids", [])),
            "reason_for_llm_review": group.get("reason_for_llm_review", ""),
            "recommended_backend_action": "Resolve through OpenAI LLM review before marking final output complete.",
        })
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if not rows:
        rows = [{
            "job_status": job_status,
            "candidate_group_id": "",
            "priority": "",
            "row_count": 0,
            "row_ids": "",
            "reason_for_llm_review": "",
            "recommended_backend_action": "",
        }]
    pl.DataFrame(rows).write_csv(path)


def _record_for_row(rows_dict: Dict[int, Dict[str, Any]], row_id: int) -> Dict[str, Any]:
    row = rows_dict.get(int(row_id), {})
    return {
        "row_id": int(row_id),
        "supplier_name": row.get("orig_supplier_name") or row.get("name_norm") or "",
        "address": row.get("orig_address") or row.get("addr_norm") or "",
        "city": row.get("orig_city") or row.get("city_norm") or "",
        "country": row.get("orig_country") or row.get("country_norm") or "",
        "tax_ids": row.get("tax_norm") or "",
        "domain": row.get("domain") or "",
        "supplier_identity_core": row.get("supplier_identity_core") or "",
    }


def _priority_for_group(row_ids: List[int], groups: List[Dict[str, Any]], cluster_map: Dict[int, int], scores: Dict[int, float]) -> str:
    roots = {cluster_map.get(rid) for rid in row_ids if rid in cluster_map}
    if any(root is not None and int(scores.get(root, 0) or 0) == 70 for root in roots):
        return "P0"
    if any(root is not None and int(scores.get(root, 0) or 0) in {85, 98} for root in roots):
        return "P1"
    if any("risky" in json.dumps(g.get("evidence", {}), ensure_ascii=False).lower() for g in groups):
        return "P2"
    return "P3"


def _overlaps_existing_85_98(row_ids: List[int], cluster_map: Dict[int, int], scores: Dict[int, float]) -> bool:
    roots = {cluster_map.get(rid) for rid in row_ids if rid in cluster_map}
    return any(root is not None and int(scores.get(root, 0) or 0) in {85, 98} for root in roots)


def _group_is_generic_location_only(group: Dict[str, Any], rows_dict: Dict[int, Dict[str, Any]]) -> bool:
    evidence = group.get("evidence") or {}
    reject = str(evidence.get("guardrail_reject") or "").lower()
    if "only shared name tokens are generic" in reject:
        return True
    row_ids = group.get("row_ids", [])
    if len(row_ids) < 2:
        return True
    token_sets = []
    for rid in row_ids:
        row = rows_dict.get(int(rid), {})
        token_sets.append(set(str(row.get("name_norm") or "").split()))
    if not token_sets:
        return False
    shared = set.intersection(*token_sets)
    return bool(shared) and shared <= (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES)


def _risk_indicators_for_records(records: List[Dict[str, Any]]) -> List[str]:
    tokens = set()
    for record in records:
        tokens |= set(str(record.get("supplier_name", "")).lower().replace("-", " ").split())
        tokens |= set(str(record.get("supplier_identity_core", "")).lower().split())
    return sorted(tokens & (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES))[:30]


def _estimate_tokens(group: Dict[str, Any]) -> int:
    return max(1, len(json.dumps(group, ensure_ascii=False)) // 4 + 500)


def _count_by(items: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts = defaultdict(int)
    for item in items:
        counts[str(item.get(key, ""))] += 1
    return counts


def _split_has_overlap_or_drops(response: Dict[str, Any], expected_rows: set) -> bool:
    seen = []
    for cluster in response.get("clusters", []) or []:
        seen.extend(int(x) for x in cluster.get("row_ids", []))
    rejected = {int(x) for x in response.get("rejected_row_ids", []) or []}
    return len(seen) != len(set(seen)) or (set(seen) | rejected) != expected_rows


def _group_has_no_strong_evidence(group: Dict[str, Any]) -> bool:
    evidence = json.dumps(group.get("evidence", {}), ensure_ascii=False).lower()
    passes = set(group.get("deterministic_pass_types", []) or [])
    strong_pass = bool(passes & {"tax_exact", "name_address_exact", "domain_name_related", "support_same_entity_id"})
    return not strong_pass and not any(x in evidence for x in ["tax_overlap", "same_domain", "address_supported"])


def _validation_row(group_id: str, status: str, reason: str) -> Dict[str, Any]:
    return {"candidate_group_id": group_id, "status": status, "reason": reason}


def _score_distribution(df: pl.DataFrame) -> Dict[str, int]:
    rows = df.group_by("Match Percentage").len().to_dicts()
    return {str(r["Match Percentage"]): int(r["len"]) for r in rows}


def _cluster_numbers_contiguous(df: pl.DataFrame) -> bool:
    nums = sorted(x for x in df.get_column("Cluster Number").to_list() if x is not None)
    unique = sorted(set(nums))
    return unique == list(range(1, len(unique) + 1))


def _normalize_mode(value: Any) -> str:
    mode = str(value or "disabled").strip().lower()
    if mode == "sync":
        return "live"
    if mode in {"off", "none"}:
        return "disabled"
    return mode


def _first_existing_cluster_root(cluster_map: Dict[int, int], scores: Dict[int, float]) -> int | None:
    for root in sorted(set(cluster_map.values())):
        if int(scores.get(root, 0) or 0) in {85, 98}:
            return int(root)
    return None


def _deterministic_98_rows(cluster_map: Dict[int, int], scores: Dict[int, float]) -> set[int]:
    return {
        int(row_id)
        for row_id, root in cluster_map.items()
        if int(scores.get(root, 0) or 0) == 98
    }


def _final_scores_allowed(df: pl.DataFrame, job_status: str) -> bool:
    allowed = {"98%", "85%", ""}
    if job_status != "COMPLETE":
        allowed.add("70%")
    return set(str(x or "") for x in df.get_column("Match Percentage").to_list()) <= allowed

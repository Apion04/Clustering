"""Output generation: main reviewer file and optional audit/report files."""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Set
import polars as pl
from rapidfuzz import fuzz

from src.config import (
    BROAD_GLOBAL_SUPPLIER_CORES,
    COMMON_FIRST_NAMES,
    GENERIC_ROOT_TOKENS,
    LOCATION_ROOT_TOKENS,
    REGULATORY_REVIEW_TOKENS,
)
from src.scoring import cluster_hardening_route


_REVIEW_DEFAULT_REASON = "family/review edge type not suitable for auto-clustering"


def save_review_candidates(review_candidates: List[Dict[str, Any]], rows_dict: Dict[Any, Dict[str, Any]], output_path: str):
    """Save review candidate pairs routed outside the main auto-cluster output."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    columns = [
        "priority",
        "row_id_1", "row_id_2",
        "supplier_name_1", "supplier_name_2",
        "supplier_identity_core_1", "supplier_identity_core_2", "shared_supplier_identity_core",
        "address_1", "address_2",
        "city_1", "city_2",
        "country_1", "country_2",
        "reason", "score", "suggested_action", "pass_type",
        "evidence_json", "why_not_auto_clustered",
        "same_tax", "same_domain", "same_or_similar_address", "same_country",
        "shared_discriminative_token", "risky_token",
        "shared_support_field", "support_field_value", "support_field_strength", "review_group_key",
        "llm_verdict", "llm_confidence",
    ]

    if not review_candidates:
        pl.DataFrame({col: [] for col in columns}).write_csv(output_path)
        return output_path

    rows = []
    for cand in review_candidates:
        row_a_id = cand["row_a"]
        row_b_id = cand["row_b"]
        row_a = rows_dict.get(row_a_id, {})
        row_b = rows_dict.get(row_b_id, {})
        evidence = cand.get("evidence") or {}
        why = cand.get("review_reason") or _REVIEW_DEFAULT_REASON
        flags = _review_flags(row_a, row_b, evidence)
        priority = _review_priority(cand, flags)
        suggested_action = _suggested_review_action(cand, flags, priority)

        rows.append({
            "priority": priority,
            "row_id_1": str(row_a_id),
            "row_id_2": str(row_b_id),
            "supplier_name_1": str(row_a.get("orig_supplier_name") or row_a.get("name_norm") or ""),
            "supplier_name_2": str(row_b.get("orig_supplier_name") or row_b.get("name_norm") or ""),
            "supplier_identity_core_1": str(row_a.get("supplier_identity_core") or ""),
            "supplier_identity_core_2": str(row_b.get("supplier_identity_core") or ""),
            "shared_supplier_identity_core": str(evidence.get("shared_supplier_identity_core") or ""),
            "address_1": str(row_a.get("addr_norm") or ""),
            "address_2": str(row_b.get("addr_norm") or ""),
            "city_1": str(row_a.get("city_norm") or ""),
            "city_2": str(row_b.get("city_norm") or ""),
            "country_1": str(row_a.get("country_norm") or ""),
            "country_2": str(row_b.get("country_norm") or ""),
            "reason": why,
            "score": float(cand.get("match_pct") or 0.0),
            "suggested_action": suggested_action,
            "pass_type": str(cand.get("pass_type") or ""),
            "evidence_json": json.dumps(evidence, ensure_ascii=False),
            "why_not_auto_clustered": why,
            "same_tax": flags["same_tax"],
            "same_domain": flags["same_domain"],
            "same_or_similar_address": flags["same_or_similar_address"],
            "same_country": flags["same_country"],
            "shared_discriminative_token": "|".join(flags["shared_discriminative_tokens"]),
            "risky_token": "|".join(flags["risky_tokens"]),
            "shared_support_field": flags["shared_support_field"],
            "support_field_value": flags["support_field_value"],
            "support_field_strength": flags["support_field_strength"],
            "review_group_key": flags["review_group_key"],
            "llm_verdict": str(evidence.get("ai_decision") or ""),
            "llm_confidence": float(evidence.get("ai_confidence") or 0.0),
        })

    rows.sort(key=lambda row: (
        0 if row["review_group_key"] else 1,
        row["review_group_key"],
        -float(row["priority"]),
        row["reason"],
        row["supplier_name_1"],
        row["supplier_name_2"],
    ))
    pl.DataFrame(rows, schema={
        "priority": pl.Float64,
        "row_id_1": pl.Utf8,
        "row_id_2": pl.Utf8,
        "supplier_name_1": pl.Utf8,
        "supplier_name_2": pl.Utf8,
        "supplier_identity_core_1": pl.Utf8,
        "supplier_identity_core_2": pl.Utf8,
        "shared_supplier_identity_core": pl.Utf8,
        "address_1": pl.Utf8,
        "address_2": pl.Utf8,
        "city_1": pl.Utf8,
        "city_2": pl.Utf8,
        "country_1": pl.Utf8,
        "country_2": pl.Utf8,
        "reason": pl.Utf8,
        "score": pl.Float64,
        "suggested_action": pl.Utf8,
        "pass_type": pl.Utf8,
        "evidence_json": pl.Utf8,
        "why_not_auto_clustered": pl.Utf8,
        "same_tax": pl.Boolean,
        "same_domain": pl.Boolean,
        "same_or_similar_address": pl.Boolean,
        "same_country": pl.Boolean,
        "shared_discriminative_token": pl.Utf8,
        "risky_token": pl.Utf8,
        "shared_support_field": pl.Utf8,
        "support_field_value": pl.Utf8,
        "support_field_strength": pl.Utf8,
        "review_group_key": pl.Utf8,
        "llm_verdict": pl.Utf8,
        "llm_confidence": pl.Float64,
    }).write_csv(output_path)
    return output_path


def save_main_output(df: pl.DataFrame, output_path: str):
    """Save main output with original columns + Cluster Number + Match Percentage only."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    lower = output_path.lower()
    if lower.endswith(".xlsx"):
        try:
            df.write_excel(output_path)
        except Exception:
            # openpyxl fallback via pandas if polars excel writer is unavailable.
            df.to_pandas().to_excel(output_path, index=False)
    else:
        df.write_csv(output_path)
    return output_path


def save_audit_file(
    audit_data: Dict[int, Dict[str, Any]],
    output_path: str,
    df: pl.DataFrame,
    cluster_map: Dict[int, int],
    merger: Any = None,
):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    rows = []
    if merger is not None and "row_id" in df.columns:
        rows_dict = {int(row["row_id"]): row for row in df.iter_rows(named=True)}
        root_to_number = _root_to_cluster_number(cluster_map)
        clusters = merger.get_clusters()
        for root, meta in audit_data.items():
            member_ids = sorted(clusters.get(root, set()))
            warnings = _cluster_warnings(member_ids, rows_dict, meta)
            edges = merger.get_cluster_edges(root)
            if not edges:
                rows.append(_audit_cluster_row(root, root_to_number, meta, member_ids, rows_dict, warnings))
                continue
            for edge in edges:
                row_a = rows_dict.get(int(edge.row_a), {})
                row_b = rows_dict.get(int(edge.row_b), {})
                evidence = edge.evidence or {}
                rows.append({
                    "cluster_root": root,
                    "cluster_number": root_to_number.get(root),
                    "cluster_size": meta.get("cluster_size"),
                    "row_ids": "|".join(str(rid) for rid in member_ids),
                    "row_id_1": edge.row_a,
                    "row_id_2": edge.row_b,
                    "supplier_name_1": row_a.get("orig_supplier_name") or row_a.get("name_norm") or "",
                    "supplier_name_2": row_b.get("orig_supplier_name") or row_b.get("name_norm") or "",
                    "supplier_identity_core_1": row_a.get("supplier_identity_core") or "",
                    "supplier_identity_core_2": row_b.get("supplier_identity_core") or "",
                    "match_score": edge.match_pct,
                    "edge_pass_type": edge.pass_type,
                    "primary_pass": meta.get("primary_pass"),
                    "evidence_json": json.dumps(evidence, ensure_ascii=False),
                    "guardrail_applied": bool(evidence.get("guardrail_reject")),
                    "distinctive_supplier_identity_used": edge.pass_type == "distinctive_supplier_identity" or bool(evidence.get("distinctive_supplier_identity")),
                    "cross_address_or_country_warning": bool(evidence.get("cross_address") or evidence.get("cross_country")),
                    "score_reason": evidence.get("score_reason", ""),
                    "needs_review": edge.needs_review,
                    "review_reason": edge.review_reason,
                    "cluster_warnings": "; ".join(warnings),
                    "min_edge_score": meta.get("min_edge_score"),
                    "max_edge_score": meta.get("max_edge_score"),
                })
    else:
        for root, meta in audit_data.items():
            rows.append({
                "cluster_root": root,
                "cluster_size": meta.get("cluster_size"),
                "match_pct": meta.get("match_pct"),
                "primary_pass": meta.get("primary_pass"),
                "pass_types": ", ".join(meta.get("pass_types", [])),
                "needs_review": meta.get("needs_review"),
                "review_reasons": "; ".join(meta.get("review_reasons", [])) if meta.get("review_reasons") else "",
                "edge_count": meta.get("edge_count"),
                "min_edge_score": meta.get("min_edge_score"),
                "max_edge_score": meta.get("max_edge_score"),
            })
    audit_df = pl.DataFrame(rows) if rows else pl.DataFrame({"message": ["No audit rows"]})
    if output_path.lower().endswith(".xlsx"):
        try:
            audit_df.write_excel(output_path)
        except Exception:
            audit_df.to_pandas().to_excel(output_path, index=False)
    else:
        audit_df.write_csv(output_path)
    return output_path


def save_ai_review_clusters(
    audit_data: Dict[int, Dict[str, Any]],
    output_path: str,
    df: pl.DataFrame,
    cluster_map: Dict[int, int],
    merger: Any,
):
    """Export accepted main clusters that need AI/human validation.

    This is an export-only QA file. It does not call an AI service and does not
    modify the main clustered output.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    columns = [
        "Cluster Number",
        "Match Percentage",
        "Review Priority",
        "Review Reason",
        "Cluster Size",
        "Countries",
        "Supplier Names",
        "Addresses",
        "Pass Types",
        "Evidence Summary",
        "Risk Flags",
        "Question For AI",
    ]
    if merger is None or "row_id" not in df.columns:
        pl.DataFrame({col: [] for col in columns}).write_csv(output_path)
        return output_path

    rows_dict = {int(row["row_id"]): row for row in df.iter_rows(named=True)}
    clusters = merger.get_clusters()
    root_to_number = _root_to_cluster_number(cluster_map)
    out_rows = []
    question = (
        "Do these rows represent the same supplier brand/group identity for procurement supplier cleanup? "
        "Answer SAME_SUPPLIER, RELATED_NOT_SAME, DIFFERENT, or NEEDS_HUMAN_REVIEW, with a short reason."
    )

    for root, member_set in clusters.items():
        member_ids = sorted(member_set)
        if len(member_ids) < 2 or root not in root_to_number:
            continue
        meta = audit_data.get(root, {})
        match_pct = float(meta.get("match_pct") or 0.0)
        edges = merger.get_cluster_edges(root)
        pass_types = sorted({edge.pass_type for edge in edges if edge.pass_type})
        evidence_items = [edge.evidence or {} for edge in edges]
        warnings = _cluster_warnings(member_ids, rows_dict, meta)
        countries = sorted({str(rows_dict.get(row_id, {}).get("country_norm") or "") for row_id in member_ids if rows_dict.get(row_id, {}).get("country_norm")})
        names = [str(rows_dict.get(row_id, {}).get("orig_supplier_name") or rows_dict.get(row_id, {}).get("name_norm") or "") for row_id in member_ids]
        addresses = [str(rows_dict.get(row_id, {}).get("orig_address") or rows_dict.get(row_id, {}).get("addr_norm") or "") for row_id in member_ids]

        has_tax_support = any(edge.pass_type in {"tax_exact", "tax_loose_supported"} or e.get("tax_overlap") for edge, e in zip(edges, evidence_items))
        has_domain_support = any(edge.pass_type in {"domain_name_related", "address_domain"} or e.get("same_domain") for edge, e in zip(edges, evidence_items))
        has_address_support = any(edge.pass_type in {"name_address_exact", "professional_name_address", "address_name_related", "address_secondary_or_acronym"} or e.get("address_supported") for edge, e in zip(edges, evidence_items))
        multi_country_no_support = len(countries) > 1 and not (has_tax_support or has_domain_support or has_address_support)
        has_distinctive_identity = "distinctive_supplier_identity" in pass_types
        has_broad_brand = any(bool(e.get("broad_global_supplier_core")) for e in evidence_items)
        cluster_tokens = set()
        for row_id in member_ids:
            cluster_tokens |= set(str(rows_dict.get(row_id, {}).get("name_norm") or "").split())
            cluster_tokens |= set(str(rows_dict.get(row_id, {}).get("supplier_identity_core") or "").split())
        regulatory_tokens = sorted(cluster_tokens & REGULATORY_REVIEW_TOKENS)
        risky_tokens = sorted(cluster_tokens & (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES | BROAD_GLOBAL_SUPPLIER_CORES))
        risk_flags = []
        if match_pct < 90:
            risk_flags.append("match_pct_below_90")
        if len(member_ids) > 20:
            risk_flags.append("large_cluster")
        if has_distinctive_identity:
            risk_flags.append("distinctive_supplier_identity")
        if multi_country_no_support:
            risk_flags.append("multi_country_without_tax_domain_address_support")
        if has_broad_brand:
            risk_flags.append("broad_global_supplier_group")
        if regulatory_tokens:
            risk_flags.append("regulatory_or_association_wording")
        risk_flags.extend(warnings)
        if risky_tokens:
            risk_flags.append("risky_tokens:" + "|".join(risky_tokens[:20]))

        include = bool(
            match_pct < 90
            or len(member_ids) > 20
            or has_distinctive_identity
            or multi_country_no_support
            or has_broad_brand
            or regulatory_tokens
            or warnings
            or risky_tokens
        )
        if not include:
            continue

        score_reasons = []
        for evidence in evidence_items:
            reason = evidence.get("score_reason") or evidence.get("review_only_guardrail") or evidence.get("guardrail_reject")
            if reason and reason not in score_reasons:
                score_reasons.append(str(reason))
        review_reason = "; ".join(risk_flags[:10])
        priority = _ai_review_priority(match_pct, len(member_ids), risk_flags, has_tax_support, has_domain_support, has_address_support)
        out_rows.append({
            "Cluster Number": int(root_to_number[root]),
            "Match Percentage": f"{int(round(match_pct))}%",
            "Review Priority": priority,
            "Review Reason": review_reason,
            "Cluster Size": len(member_ids),
            "Countries": " | ".join(countries),
            "Supplier Names": " | ".join(names[:50]),
            "Addresses": " | ".join([a for a in addresses[:50] if a]),
            "Pass Types": " | ".join(pass_types),
            "Evidence Summary": " | ".join(score_reasons[:20]),
            "Risk Flags": " | ".join(risk_flags),
            "Question For AI": question,
        })

    out_rows.sort(key=lambda row: (-float(row["Review Priority"]), -int(row["Cluster Size"]), int(row["Cluster Number"])))
    if out_rows:
        pl.DataFrame(out_rows, schema={
            "Cluster Number": pl.Int64,
            "Match Percentage": pl.Utf8,
            "Review Priority": pl.Float64,
            "Review Reason": pl.Utf8,
            "Cluster Size": pl.Int64,
            "Countries": pl.Utf8,
            "Supplier Names": pl.Utf8,
            "Addresses": pl.Utf8,
            "Pass Types": pl.Utf8,
            "Evidence Summary": pl.Utf8,
            "Risk Flags": pl.Utf8,
            "Question For AI": pl.Utf8,
        }).write_csv(output_path)
    else:
        pl.DataFrame({col: [] for col in columns}).write_csv(output_path)
    return output_path


def save_risky_accepted_cluster_review(
    output_path: str,
    df: pl.DataFrame,
    cluster_map: Dict[int, int],
    match_pcts: Dict[int, float],
    merger: Any = None,
):
    """Write accepted 85/98 clusters that contain risky/generic/location terms."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    columns = [
        "cluster_number",
        "match_percentage",
        "row_count",
        "risky_tokens_found",
        "countries_present",
        "pass_types_used",
        "evidence_summary",
        "has_tax_support",
        "has_address_support",
        "has_domain_support",
        "has_only_generic_or_location_evidence",
        "sample_supplier_names",
        "recommended_action",
    ]
    rows_dict = {int(row["row_id"]): row for row in df.iter_rows(named=True)} if "row_id" in df.columns else {}
    root_to_number = _root_to_cluster_number(cluster_map)
    rows_by_root: Dict[int, List[int]] = {}
    for row_id, root in cluster_map.items():
        rows_by_root.setdefault(int(root), []).append(int(row_id))

    out_rows = []
    for root, member_ids in rows_by_root.items():
        score = float(match_pcts.get(root, 0.0) or 0.0)
        if int(score) not in {85, 98} or len(member_ids) < 2:
            continue
        edges = merger.get_cluster_edges(root) if merger is not None and hasattr(merger, "get_cluster_edges") else []
        risk = _accepted_cluster_risk(member_ids, rows_dict, edges)
        if not risk["risky_tokens_found"]:
            continue
        out_rows.append({
            "cluster_number": int(root_to_number.get(root, root)),
            "match_percentage": f"{int(round(score))}%",
            "row_count": len(member_ids),
            "risky_tokens_found": "|".join(risk["risky_tokens_found"]),
            "countries_present": "|".join(risk["countries_present"]),
            "pass_types_used": "|".join(risk["pass_types_used"]),
            "evidence_summary": risk["evidence_summary"],
            "has_tax_support": risk["has_tax_support"],
            "has_address_support": risk["has_address_support"],
            "has_domain_support": risk["has_domain_support"],
            "has_only_generic_or_location_evidence": risk["has_only_generic_or_location_evidence"],
            "sample_supplier_names": " | ".join(risk["sample_supplier_names"]),
            "recommended_action": _recommended_risky_action(score, risk),
        })
    out_rows.sort(key=lambda row: (
        {"reject": 0, "downgrade_to_70": 1, "inspect": 2, "keep": 3}.get(row["recommended_action"], 9),
        -int(row["row_count"]),
        int(row["cluster_number"]),
    ))
    if out_rows:
        pl.DataFrame(out_rows, schema={
            "cluster_number": pl.Int64,
            "match_percentage": pl.Utf8,
            "row_count": pl.Int64,
            "risky_tokens_found": pl.Utf8,
            "countries_present": pl.Utf8,
            "pass_types_used": pl.Utf8,
            "evidence_summary": pl.Utf8,
            "has_tax_support": pl.Boolean,
            "has_address_support": pl.Boolean,
            "has_domain_support": pl.Boolean,
            "has_only_generic_or_location_evidence": pl.Boolean,
            "sample_supplier_names": pl.Utf8,
            "recommended_action": pl.Utf8,
        }).write_csv(output_path)
    else:
        pl.DataFrame({col: [] for col in columns}).write_csv(output_path)
    return output_path


def save_top_suspicious_clusters_report(
    output_path: str,
    df: pl.DataFrame,
    cluster_map: Dict[int, int],
    match_pcts: Dict[int, float],
    merger: Any = None,
):
    """Write a Markdown QA report for the riskiest accepted clusters."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    rows_dict = {int(row["row_id"]): row for row in df.iter_rows(named=True)} if "row_id" in df.columns else {}
    root_to_number = _root_to_cluster_number(cluster_map)
    rows_by_root: Dict[int, List[int]] = {}
    for row_id, root in cluster_map.items():
        rows_by_root.setdefault(int(root), []).append(int(row_id))

    summaries = []
    terms = {
        "insight", "insights", "springer", "gastro", "tuebingen", "tubingen",
        "edinburgh", "apple", "green", "express", "chemical", "chemicals",
        "chemistry", "technology", "technologies", "services", "service",
        "group", "institute", "society", "university",
    }
    for root, member_ids in rows_by_root.items():
        if len(member_ids) < 2:
            continue
        score = float(match_pcts.get(root, 0.0) or 0.0)
        if score <= 0:
            continue
        edges = merger.get_cluster_edges(root) if merger is not None and hasattr(merger, "get_cluster_edges") else []
        risk = _accepted_cluster_risk(member_ids, rows_dict, edges)
        token_hit = sorted(set(risk["risky_tokens_found"]) & terms)
        summaries.append({
            "root": root,
            "cluster_number": int(root_to_number.get(root, root)),
            "score": score,
            "size": len(member_ids),
            "risk": risk,
            "token_hit": token_hit,
            "cross_country": len(risk["countries_present"]) > 1,
            "no_support": not (risk["has_tax_support"] or risk["has_address_support"] or risk["has_domain_support"]),
            "member_ids": member_ids,
        })

    def emit_section(title: str, items: List[Dict[str, Any]]) -> List[str]:
        lines = [f"## {title}", ""]
        if not items:
            lines.append("_None found._")
            lines.append("")
            return lines
        for item in items[:25]:
            risk = item["risk"]
            lines.extend([
                f"### Cluster {item['cluster_number']} ({int(round(item['score']))}%, {item['size']} rows)",
                f"- Countries: {', '.join(risk['countries_present']) or 'blank'}",
                f"- Pass types: {', '.join(risk['pass_types_used']) or 'unknown'}",
                f"- Risk tokens: {', '.join(risk['risky_tokens_found'][:20]) or 'none'}",
                f"- Support: tax={risk['has_tax_support']}, address={risk['has_address_support']}, domain={risk['has_domain_support']}",
                f"- Recommended action: {_recommended_risky_action(item['score'], risk)}",
                f"- Evidence: {risk['evidence_summary'] or 'n/a'}",
                "- Sample names:",
            ])
            for name in risk["sample_supplier_names"][:10]:
                lines.append(f"  - {name}")
            lines.append("")
        return lines

    lines = [
        "# Top Suspicious Clusters",
        "",
        "This report is an internal QA artifact. It does not change the clean user-facing output.",
        "",
    ]
    largest = sorted(summaries, key=lambda x: (-x["size"], x["cluster_number"]))
    cross_country = sorted([x for x in summaries if x["cross_country"]], key=lambda x: (-x["size"], x["cluster_number"]))
    ambiguous = sorted([x for x in summaries if {"insight", "insights", "springer", "apple"} & set(x["risk"]["risky_tokens_found"])], key=lambda x: (-x["size"], x["cluster_number"]))
    generic_location = sorted([x for x in summaries if x["risk"]["risky_tokens_found"]], key=lambda x: (-len(x["risk"]["risky_tokens_found"]), -x["size"]))
    no_support = sorted([x for x in summaries if x["no_support"]], key=lambda x: (-x["size"], x["cluster_number"]))
    above_50 = sorted([x for x in summaries if x["size"] > 50], key=lambda x: (-x["size"], x["cluster_number"]))
    term_hits = sorted([x for x in summaries if x["token_hit"]], key=lambda x: (-x["size"], x["cluster_number"]))
    lines += emit_section("Largest 25 Clusters", largest)
    lines += emit_section("Largest 25 Cross-Country Clusters", cross_country)
    lines += emit_section("Accepted Clusters With Ambiguous Cores", ambiguous)
    lines += emit_section("Accepted Clusters With Generic/Location Tokens", generic_location)
    lines += emit_section("Accepted Clusters With No Tax/Address/Domain Support", no_support)
    lines += emit_section("All Clusters Above 50 Rows", above_50)
    lines += emit_section("Clusters Containing Watched Terms", term_hits)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def save_unresolved_llm_exception_report(llm_review_groups: List[Dict[str, Any]], output_path: str):
    """Write internal exception report for unresolved 70-score groups."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    rows = []
    for group in llm_review_groups:
        rows.append({
            "candidate_group_id": group.get("candidate_group_id", ""),
            "cluster_root": str(group.get("cluster_root", "")),
            "row_count": len(group.get("row_ids", []) or []),
            "row_ids": "|".join(str(x) for x in group.get("row_ids", []) or []),
            "supplier_names": " | ".join(str(r.get("supplier_name", "")) for r in group.get("records", [])[:30]),
            "deterministic_pass_types": "|".join(group.get("deterministic_pass_types", []) or []),
            "reason_for_llm_review": group.get("reason_for_llm_review", ""),
        })
    if rows:
        pl.DataFrame(rows, schema={
            "candidate_group_id": pl.Utf8,
            "cluster_root": pl.Utf8,
            "row_count": pl.Int64,
            "row_ids": pl.Utf8,
            "supplier_names": pl.Utf8,
            "deterministic_pass_types": pl.Utf8,
            "reason_for_llm_review": pl.Utf8,
        }).write_csv(output_path)
    else:
        pl.DataFrame({
            "candidate_group_id": [],
            "cluster_root": [],
            "row_count": [],
            "row_ids": [],
            "supplier_names": [],
            "deterministic_pass_types": [],
            "reason_for_llm_review": [],
        }).write_csv(output_path)
    return output_path


def _ai_review_priority(match_pct: float, cluster_size: int, risk_flags: List[str], has_tax: bool, has_domain: bool, has_address: bool) -> float:
    priority = 50.0
    if match_pct < 90:
        priority += min(25.0, 90.0 - match_pct)
    if cluster_size > 20:
        priority += min(20.0, (cluster_size - 20) / 5.0)
    priority += min(25.0, 4.0 * len(risk_flags))
    if has_tax or has_domain or has_address:
        priority -= 8.0
    return round(max(0.0, min(100.0, priority)), 2)


def generate_processing_report(stats: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    report = f"""Supplier Clustering Report
Generated: {datetime.now().isoformat()}

Input:
  Total rows: {stats['total_rows']}

Processing:
  Candidate pairs generated: {stats['candidate_pairs']}
  Candidate pair cap reached: {stats.get('candidate_pairs_capped', False)}
  Match edges created: {stats.get('match_edges_created', 0)}
  Processing time: {stats['processing_time_seconds']:.1f} seconds
  Blocking keys: {stats.get('blocking_diagnostics', {}).get('total_blocking_keys', 0)}
  Unique blocks: {stats.get('blocking_diagnostics', {}).get('unique_block_keys', 0)}
  Raw candidate pairs before caps: {stats.get('blocking_diagnostics', {}).get('candidate_pairs_before_caps', 0)}
  Candidate pairs after caps: {stats.get('blocking_diagnostics', {}).get('candidate_pairs_after_caps', stats['candidate_pairs'])}
  Skipped oversized blocks: {stats.get('blocking_diagnostics', {}).get('skipped_oversized_blocks', 0)}
  Skipped blocks by type: {stats.get('blocking_diagnostics', {}).get('skipped_blocks_by_type', {})}

Output:
  Total clusters found: {stats['clusters_found']}
  Largest cluster size: {stats.get('largest_cluster_size', 0)}
  Auto-clustered rows: {stats['auto_clustered_rows']}
  Review queue rows: {stats['review_queue_rows']}
  Review candidate pairs: {stats.get('review_candidate_pairs', 0)}
  Review candidate rows: {stats.get('review_candidate_rows', 0)}
  Guardrail rejected pairs: {stats.get('guardrail_rejected_pairs', 0)}
  Singleton rows: {stats['singleton_rows']}
  Production run status: {stats.get('production_run_status', 'OK')}
  Pre-LLM score distribution: {stats.get('pre_llm_score_distribution', {})}
  Final score distribution: {stats.get('final_score_distribution', {})}
  LLM review candidate clusters: {stats.get('llm_review_candidate_clusters', 0)}
  Unresolved LLM groups in final mode: {stats.get('unresolved_llm_groups', 0)}
  LLM decisions: {stats.get('llm_decisions', {})}

Stage timings:
"""
    for stage, seconds in stats.get("stage_timings", {}).items():
        report += f"  {stage}: {seconds:.3f}s\n"
    report += """
Top 20 largest blocking keys:
"""
    for block in stats.get("blocking_diagnostics", {}).get("top_20_largest_blocks", []):
        report += (
            f"  type={block.get('block_type')} rows={block.get('row_count')} "
            f"raw_pairs={block.get('raw_pair_count')} key={block.get('block_key_preview')}\n"
        )
    report += """
Top skipped blocks:
"""
    for block in stats.get("blocking_diagnostics", {}).get("top_skipped_blocks", []):
        report += (
            f"  type={block.get('block_type')} rows={block.get('row_count')} "
            f"raw_pairs={block.get('raw_pair_count')} reason={block.get('reason')} "
            f"key={block.get('block_key_preview')}\n"
        )

    report += f"""
Cluster size distribution:
  Size 2: {stats.get('size_2_clusters', 0)}
  Size 3: {stats.get('size_3_clusters', 0)}
  Size 4+: {stats.get('size_4plus_clusters', 0)}

Pass type distribution:
"""
    for pass_type, count in stats.get('pass_type_counts', {}).items():
        report += f"  {pass_type}: {count}\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


def _tax_values(row: Dict[str, Any]) -> Set[str]:
    return {v for v in str(row.get("tax_norm", "") or "").split("|") if v}


def _review_flags(row_a: Dict[str, Any], row_b: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    domain_a = row_a.get("domain", "")
    domain_b = row_b.get("domain", "")
    addr_a = row_a.get("addr_norm", "")
    addr_b = row_b.get("addr_norm", "")
    country_a = row_a.get("country_norm", "")
    country_b = row_b.get("country_norm", "")
    shared_discriminative = sorted(
        {t for t in str(row_a.get("idf_discriminative_tokens", "") or "").split("|") if t}
        & {t for t in str(row_b.get("idf_discriminative_tokens", "") or "").split("|") if t}
    )
    risky_tokens = sorted(set(evidence.get("risky_overlap") or []) | (set(evidence.get("shared_tokens") or []) & (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES)))
    support_fields = evidence.get("shared_support_field") or evidence.get("support_fields") or []
    if isinstance(support_fields, str):
        support_fields = [support_fields] if support_fields else []
    support_value = str(evidence.get("support_field_value") or "")
    support_strength = str(evidence.get("support_field_strength") or "")
    review_group_key = ""
    if support_value:
        review_group_key = f"support:{support_strength}:{support_value}"
    elif domain_a and domain_b and domain_a == domain_b and not row_a.get("is_generic_domain") and not row_b.get("is_generic_domain"):
        review_group_key = f"domain:{domain_a}"
    elif shared_discriminative:
        review_group_key = f"rare:{shared_discriminative[0]}"
    return {
        "same_tax": bool(_tax_values(row_a) & _tax_values(row_b)),
        "same_domain": bool(domain_a and domain_b and domain_a == domain_b and not row_a.get("is_generic_domain") and not row_b.get("is_generic_domain")),
        "same_or_similar_address": bool(addr_a and addr_b and (addr_a == addr_b or fuzz.token_set_ratio(addr_a, addr_b) >= 88)),
        "same_country": bool(country_a and country_b and country_a == country_b),
        "shared_discriminative_tokens": shared_discriminative,
        "risky_tokens": risky_tokens,
        "shared_support_field": "|".join(str(x) for x in support_fields if x),
        "support_field_value": support_value,
        "support_field_strength": support_strength,
        "review_group_key": review_group_key,
    }


def _review_priority(cand: Dict[str, Any], flags: Dict[str, Any]) -> float:
    evidence = cand.get("evidence") or {}
    priority = float(cand.get("match_pct") or 0.0)
    if flags["same_tax"]:
        priority += 30
    if flags["same_domain"]:
        priority += 20
    if flags["same_or_similar_address"]:
        priority += 20
    if evidence.get("known_brand_family_overlap") or evidence.get("exact_alias_overlap"):
        priority += 10
    if flags["support_field_strength"] == "same_entity_id":
        priority += 35
    elif flags["support_field_strength"] == "same_entity_name":
        priority += 25
    elif flags["support_field_strength"] == "family_or_parent":
        priority += 18
    elif flags["support_field_strength"] == "review_only":
        priority += 8
    if float(evidence.get("name_sim", 0.0) or 0.0) >= 0.85:
        priority += 8
    if flags["shared_discriminative_tokens"]:
        priority += min(8, 2 * len(flags["shared_discriminative_tokens"]))
    if str(evidence.get("ai_decision") or "") in {"SAME_ENTITY", "RELATED"}:
        priority += 10
    if flags["risky_tokens"]:
        priority -= 10
    if cand.get("pass_type") in {"family_bridge_supported", "family_cross_country", "known_family_bridge"}:
        priority -= 5
    return max(0.0, min(100.0, round(priority, 2)))


def _suggested_review_action(cand: Dict[str, Any], flags: Dict[str, Any], priority: float) -> str:
    if flags["same_tax"] or flags["same_domain"] or flags["same_or_similar_address"]:
        return "review_possible_duplicate"
    if flags["support_field_value"]:
        return "review_shared_support_field"
    if cand.get("pass_type") in {"known_family_bridge", "known_brand_family_alias", "family_bridge_supported", "support_field_review"}:
        return "review_family_or_alias_candidate"
    if priority >= 85:
        return "review_high_priority"
    return "review_low_confidence"


def _root_to_cluster_number(cluster_map: Dict[int, int]) -> Dict[int, int]:
    roots = sorted(set(cluster_map.values()), key=lambda root: min(rid for rid, candidate_root in cluster_map.items() if candidate_root == root))
    return {root: idx + 1 for idx, root in enumerate(roots)}


def _audit_cluster_row(root: int, root_to_number: Dict[int, int], meta: Dict[str, Any], member_ids: List[int], rows_dict: Dict[int, Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
    names = [str(rows_dict.get(row_id, {}).get("orig_supplier_name") or rows_dict.get(row_id, {}).get("name_norm") or "") for row_id in member_ids]
    return {
        "cluster_root": root,
        "cluster_number": root_to_number.get(root),
        "cluster_size": meta.get("cluster_size"),
        "row_ids": "|".join(str(rid) for rid in member_ids),
        "supplier_names": " | ".join(names[:20]),
        "supplier_identity_cores": " | ".join(sorted({str(rows_dict.get(row_id, {}).get("supplier_identity_core") or "") for row_id in member_ids if rows_dict.get(row_id, {}).get("supplier_identity_core")})),
        "match_score": meta.get("match_pct"),
        "edge_pass_type": meta.get("primary_pass"),
        "primary_pass": meta.get("primary_pass"),
        "evidence_json": "{}",
        "guardrail_applied": False,
        "needs_review": meta.get("needs_review"),
        "review_reason": "; ".join(meta.get("review_reasons", [])) if meta.get("review_reasons") else "",
        "cluster_warnings": "; ".join(warnings),
        "min_edge_score": meta.get("min_edge_score"),
        "max_edge_score": meta.get("max_edge_score"),
    }


def _cluster_warnings(member_ids: List[int], rows_dict: Dict[int, Dict[str, Any]], meta: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if len(member_ids) >= 25:
        warnings.append("large_cluster")
    countries = {rows_dict.get(row_id, {}).get("country_norm") for row_id in member_ids if rows_dict.get(row_id, {}).get("country_norm")}
    if len(countries) > 1:
        warnings.append("multiple_countries")
    person_flags = {bool(rows_dict.get(row_id, {}).get("is_likely_individual")) for row_id in member_ids}
    if len(person_flags) > 1:
        warnings.append("person_company_mix")
    addresses = {rows_dict.get(row_id, {}).get("addr_norm") for row_id in member_ids if rows_dict.get(row_id, {}).get("addr_norm")}
    if len(addresses) > max(3, len(member_ids) // 2):
        warnings.append("multiple_addresses")
    if float(meta.get("match_pct") or 0.0) < 80:
        warnings.append("low_match_percentage")
    risky = set()
    for row_id in member_ids:
        risky |= set(str(rows_dict.get(row_id, {}).get("name_norm") or "").split()) & (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES)
    if risky:
        warnings.append("contains_risky_or_generic_tokens")
    return warnings


def _accepted_cluster_risk(member_ids: List[int], rows_dict: Dict[int, Dict[str, Any]], edges: List[Any]) -> Dict[str, Any]:
    names = [
        str(rows_dict.get(row_id, {}).get("orig_supplier_name") or rows_dict.get(row_id, {}).get("name_norm") or "")
        for row_id in sorted(member_ids)
    ]
    countries = sorted({
        str(rows_dict.get(row_id, {}).get("country_norm") or "")
        for row_id in member_ids
        if rows_dict.get(row_id, {}).get("country_norm")
    })
    cluster_tokens: Set[str] = set()
    for row_id in member_ids:
        cluster_tokens |= set(str(rows_dict.get(row_id, {}).get("name_norm") or "").split())
        cluster_tokens |= set(str(rows_dict.get(row_id, {}).get("supplier_identity_core") or "").split())
    risky_tokens = sorted(cluster_tokens & (GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES | BROAD_GLOBAL_SUPPLIER_CORES))
    pass_types = sorted({str(edge.pass_type) for edge in edges if getattr(edge, "pass_type", "")})
    evidence_items = [getattr(edge, "evidence", {}) or {} for edge in edges]
    has_tax_support = any(edge.pass_type in {"tax_exact", "tax_loose_supported"} or e.get("tax_overlap") or e.get("tax_loose_overlap") for edge, e in zip(edges, evidence_items))
    has_domain_support = any(edge.pass_type in {"domain_name_related", "address_domain"} or e.get("same_domain") for edge, e in zip(edges, evidence_items))
    has_address_support = any(edge.pass_type in {"name_address_exact", "professional_name_address", "address_name_related", "address_secondary_or_acronym"} or e.get("address_supported") for edge, e in zip(edges, evidence_items))
    hardening_route = cluster_hardening_route(edges)
    score_reasons: List[str] = []
    for evidence in evidence_items:
        for key in ("score_reason", "review_reason", "guardrail_reject", "review_only_guardrail"):
            reason = evidence.get(key)
            if reason and str(reason) not in score_reasons:
                score_reasons.append(str(reason))
    return {
        "risky_tokens_found": risky_tokens,
        "countries_present": countries,
        "pass_types_used": pass_types,
        "evidence_summary": " | ".join(score_reasons[:20]),
        "has_tax_support": has_tax_support,
        "has_address_support": has_address_support,
        "has_domain_support": has_domain_support,
        "has_only_generic_or_location_evidence": hardening_route in {"LLM_REVIEW", "REJECT"},
        "hardening_route": hardening_route,
        "sample_supplier_names": names[:50],
    }


def _recommended_risky_action(score: float, risk: Dict[str, Any]) -> str:
    if risk.get("hardening_route") == "REJECT":
        return "reject"
    if risk.get("has_only_generic_or_location_evidence"):
        return "downgrade_to_70"
    if risk.get("has_tax_support") or risk.get("has_address_support") or risk.get("has_domain_support"):
        return "keep"
    if int(score) == 98 and risk.get("risky_tokens_found"):
        return "inspect"
    return "inspect"

#!/usr/bin/env python3
"""End-to-end deterministic plus Gemini group-review QA runner.

This script is intentionally separate from the normal CLI. It proves that:
1. deterministic clustering can process the full file without LLM calls, and
2. Gemini is called only for selected low-score/ambiguous deterministic groups.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import httpx
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cli import auto_detect_columns  # noqa: E402
from src.config import (  # noqa: E402
    ClusteringConfig,
    COMMON_FIRST_NAMES,
    GENERIC_ROOT_TOKENS,
    HOSPITALITY_TERMS,
    LOCATION_ROOT_TOKENS,
)
from src.input_reader import read_supplier_file  # noqa: E402
from src.main import cluster_suppliers  # noqa: E402
from src.output import generate_processing_report, save_audit_file, save_main_output  # noqa: E402
from src.preprocessing import normalize_supplier_name, normalize_text  # noqa: E402
from src.scoring import calculate_cluster_scores  # noqa: E402
from src.sorting import sort_by_clusters  # noqa: E402


NON_BRIDGE_TOKENS = (
    set(GENERIC_ROOT_TOKENS)
    | set(LOCATION_ROOT_TOKENS)
    | set(HOSPITALITY_TERMS)
    | set(COMMON_FIRST_NAMES)
)

STRONG_PASS_TYPES = {
    "tax_exact",
    "tax_loose_supported",
    "name_address_exact",
    "name_exact",
    "name_fuzzy_supported",
    "address_domain",
    "domain_name_related",
}

REVIEW_PASS_TYPES = {
    "address_name_related",
    "address_secondary",
    "address_secondary_or_acronym",
    "family_bridge_supported",
    "family_cross_country",
    "known_family_bridge",
    "secondary_or_acronym_bridge",
    "acronym_review_candidate",
    "domain_name_related",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def safe_json(value: Any) -> Any:
    if is_dataclass(value):
        return safe_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, set):
        return sorted(safe_json(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(data), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux usually reports KiB.
    return usage / (1024 * 1024) if usage > 10_000_000 else usage / 1024


def clamp_pct(value: Any, default: float) -> float:
    try:
        pct = float(value)
    except Exception:
        return default
    if pct <= 1:
        pct *= 100
    return max(0.0, min(100.0, pct))


def tokens(text: str) -> List[str]:
    return [t for t in str(text or "").split() if t]


def name_search_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def clusters_from_map(cluster_map: Dict[int, int]) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = defaultdict(list)
    for row_id, root in cluster_map.items():
        clusters[int(root)].append(int(row_id))
    return {root: sorted(rows) for root, rows in clusters.items() if len(rows) > 1}


def cluster_number_map(cluster_map: Dict[int, int]) -> Dict[int, int]:
    clusters = clusters_from_map(cluster_map)
    ordered_roots = sorted(clusters, key=lambda root: min(clusters[root]))
    return {root: idx + 1 for idx, root in enumerate(ordered_roots)}


def compute_anchor_order(df: pl.DataFrame, cluster_map: Dict[int, int]) -> List[int]:
    clusters = clusters_from_map(cluster_map)
    moved_rows = {rid for rows in clusters.values() for rid in rows[1:]}
    anchor_rows = {rows[0] for rows in clusters.values() if rows}
    emitted: Set[int] = set()
    ordered: List[int] = []
    all_row_ids = [int(x) for x in df.get_column("row_id").to_list()]
    for row_id in all_row_ids:
        if row_id in emitted:
            continue
        if row_id in moved_rows:
            continue
        ordered.append(row_id)
        emitted.add(row_id)
        if row_id in anchor_rows:
            root = cluster_map.get(row_id)
            for other in clusters.get(root, [])[1:]:
                if other not in emitted:
                    ordered.append(other)
                    emitted.add(other)
    for row_id in all_row_ids:
        if row_id not in emitted:
            ordered.append(row_id)
            emitted.add(row_id)
    return ordered


def verify_anchor_order(df: pl.DataFrame, cluster_map: Dict[int, int]) -> Dict[str, Any]:
    original_ids = [int(x) for x in df.get_column("row_id").to_list()]
    ordered = compute_anchor_order(df, cluster_map)
    clusters = clusters_from_map(cluster_map)
    pos = {row_id: idx for idx, row_id in enumerate(ordered)}
    no_lost_or_duplicate = (
        len(ordered) == len(original_ids)
        and len(set(ordered)) == len(original_ids)
        and set(ordered) == set(original_ids)
    )
    clustered_consecutive = True
    bad_clusters = []
    for root, rows in clusters.items():
        expected = list(range(pos[rows[0]], pos[rows[0]] + len(rows)))
        actual = [pos[row_id] for row_id in rows]
        if actual != expected:
            clustered_consecutive = False
            bad_clusters.append({"root": root, "rows": rows[:20], "positions": actual[:20]})
            if len(bad_clusters) >= 10:
                break
    singleton_original = [rid for rid in original_ids if rid not in cluster_map]
    singleton_ordered = [rid for rid in ordered if rid not in cluster_map]
    singleton_relative_order = singleton_original == singleton_ordered
    return {
        "anchor_order_works": bool(no_lost_or_duplicate and clustered_consecutive and singleton_relative_order),
        "no_rows_lost_or_duplicated_by_ordering": bool(no_lost_or_duplicate),
        "clustered_rows_consecutive_after_anchor": bool(clustered_consecutive),
        "non_clustered_relative_order_preserved": bool(singleton_relative_order),
        "bad_anchor_clusters_sample": bad_clusters,
    }


def row_lookup(df: pl.DataFrame) -> Dict[int, Dict[str, Any]]:
    return {int(row["row_id"]): row for row in df.iter_rows(named=True)}


def names_for_rows(row_by_id: Dict[int, Dict[str, Any]], row_ids: Sequence[int], limit: int = 10) -> List[str]:
    return [str(row_by_id[row_id].get("orig_supplier_name", "")) for row_id in list(row_ids)[:limit] if row_id in row_by_id]


def cluster_shared_tokens(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(set(tokens(row.get("name_norm", ""))))
    return {token for token, count in counts.items() if count >= 2}


def distinctive_shared_tokens(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    out = set()
    for token in cluster_shared_tokens(rows):
        if token in NON_BRIDGE_TOKENS:
            continue
        if len(token) < 3:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        out.add(token)
    return out


def cluster_has_supporting_evidence(rows: Sequence[Dict[str, Any]], pass_types: Set[str]) -> bool:
    if pass_types & STRONG_PASS_TYPES:
        return True
    tax_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    address_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    for row in rows:
        for tax in str(row.get("tax_norm", "") or "").split("|"):
            if tax:
                tax_counts[tax] += 1
        domain = str(row.get("domain", "") or "")
        if domain and not row.get("is_generic_domain"):
            domain_counts[domain] += 1
        addr = str(row.get("addr_norm", "") or "")
        if addr:
            address_counts[addr] += 1
        name = str(row.get("name_norm", "") or "")
        if name:
            name_counts[name] += 1
    return any(c >= 2 for c in tax_counts.values()) or any(c >= 2 for c in domain_counts.values()) or any(c >= 2 for c in address_counts.values()) or any(c >= 2 for c in name_counts.values())


def is_obvious_generic_only_cluster(rows: Sequence[Dict[str, Any]], pass_types: Set[str]) -> bool:
    shared = cluster_shared_tokens(rows)
    if not shared:
        return False
    if distinctive_shared_tokens(rows):
        return False
    if cluster_has_supporting_evidence(rows, pass_types):
        return False
    return True


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "row_id": int(row.get("row_id")),
        "supplier_name": row.get("orig_supplier_name", ""),
        "address": row.get("orig_address", ""),
        "city": row.get("orig_city", ""),
        "country": row.get("orig_country", ""),
        "postal_code": row.get("postal_norm", ""),
        "tax_ids": row.get("tax_norm", ""),
        "domain": row.get("domain", ""),
        "email": row.get("orig_email", ""),
        "website": row.get("orig_website", ""),
        "secondary_names": row.get("orig_secondary_names", ""),
        "normalized_name": row.get("name_norm", ""),
        "normalized_address": row.get("addr_norm", ""),
        "is_likely_individual": bool(row.get("is_likely_individual", False)),
    }


def risk_flags_for_group(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any], score: float) -> List[str]:
    flags = []
    pass_types = set(meta.get("pass_types", []) or [])
    if score < 80:
        flags.append("low_score")
    if len(rows) >= 10:
        flags.append("large_cluster")
    if pass_types & REVIEW_PASS_TYPES:
        flags.append("review_pass_type")
    if any(row.get("is_likely_individual") for row in rows):
        flags.append("contains_individual_like_record")
    shared = cluster_shared_tokens(rows)
    if shared and not distinctive_shared_tokens(rows):
        flags.append("generic_or_location_token_overlap_only")
    if any(str(row.get("name_norm", "")).split() and str(row.get("name_norm", "")).split()[0][:1].isdigit() for row in rows):
        flags.append("numeric_or_short_code_prefix")
    return sorted(set(flags))


def select_llm_candidates(
    row_by_id: Dict[int, Dict[str, Any]],
    cluster_map: Dict[int, int],
    match_pcts: Dict[int, float],
    audit_data: Dict[int, Dict[str, Any]],
    min_score: float,
    max_score: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected = []
    skipped = []
    for root, row_ids in clusters_from_map(cluster_map).items():
        score = float(match_pcts.get(root, 0.0))
        if score < min_score or score >= max_score:
            continue
        rows = [row_by_id[row_id] for row_id in row_ids]
        meta = audit_data.get(root, {})
        pass_types = set(meta.get("pass_types", []) or [])
        if is_obvious_generic_only_cluster(rows, pass_types):
            skipped.append({
                "root": root,
                "row_ids": row_ids,
                "score": score,
                "reason": "obvious_generic_or_location_token_only",
                "shared_tokens": sorted(cluster_shared_tokens(rows)),
                "names": names_for_rows(row_by_id, row_ids, 10),
            })
            continue
        selected.append({
            "root": root,
            "row_ids": row_ids,
            "deterministic_score": score,
            "match_sources": sorted(pass_types),
            "risk_flags": risk_flags_for_group(rows, meta, score),
            "review_reasons": meta.get("review_reasons", []) or [],
            "rows": [compact_row(row) for row in rows],
        })
    selected.sort(key=lambda item: (item["deterministic_score"], -len(item["row_ids"]), item["root"]))
    return selected, skipped


def build_gemini_prompt(candidate: Dict[str, Any]) -> str:
    rules = {
        "strict_reject": [
            "Do not cluster individuals by first name only, surname only, city/postal only, or same household/address with different first names.",
            "Do not cluster generic industry-token-only matches such as Production, Point, Scientific, Pharm, Services, Consulting, Technology, Chemicals, Group.",
            "Do not cluster city/postal/location-token-only matches.",
            "Do not cluster hotel, restaurant, or franchise brand-only matches unless same property/address/tax/domain supports it.",
            "Ignore invalid tax placeholders and generic email domains.",
        ],
        "cluster_allowed_when": [
            "same valid tax/VAT ID",
            "same business domain with related names",
            "same or highly similar full normalized name plus same/highly similar address",
            "explicit secondary/family-name evidence",
            "known family, acquisition, rebrand, DBA, or parent bridge is plausible enough for manual review",
        ],
        "uncertain_policy": "If plausible but not proven, return uncertain with match_percentage 65-70 so reviewers can inspect together.",
        "partial_policy": "Use partial when only some row_ids belong together and list exact row_ids in clusters.",
    }
    payload = {
        "task": "Review this supplier clustering candidate group. Return JSON only.",
        "allowed_schema": {
            "decision": "cluster | do_not_cluster | uncertain | partial",
            "match_percentage": "0-100",
            "clusters": [{"row_ids": [], "match_percentage": "0-100", "reason": ""}],
            "exclude_row_ids": [],
            "reason": "",
        },
        "rules": rules,
        "reason_style": "Keep reasons concise, under 25 words.",
        "candidate_group": candidate,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_gemini_batch_prompt(candidates: Sequence[Dict[str, Any]]) -> str:
    rules = {
        "strict_reject": [
            "Do not cluster individuals by first name only, surname only, city/postal only, or same household/address with different first names.",
            "Do not cluster generic industry-token-only matches such as Production, Point, Scientific, Pharm, Services, Consulting, Technology, Chemicals, Group.",
            "Do not cluster city/postal/location-token-only matches.",
            "Do not cluster hotel, restaurant, or franchise brand-only matches unless same property/address/tax/domain supports it.",
            "Ignore invalid tax placeholders and generic email domains.",
        ],
        "cluster_allowed_when": [
            "same valid tax/VAT ID",
            "same business domain with related names",
            "same or highly similar full normalized name plus same/highly similar address",
            "explicit secondary/family-name evidence",
            "known family, acquisition, rebrand, DBA, or parent bridge is plausible enough for manual review",
        ],
        "uncertain_policy": "If plausible but not proven, return uncertain with match_percentage 65-70 so reviewers can inspect together.",
        "partial_policy": "Use partial when only some row_ids belong together and list exact row_ids in clusters.",
    }
    payload = {
        "task": "Review each supplier clustering candidate group independently. Return JSON only.",
        "response_schema": {
            "reviews": [
                {
                    "group_root": "integer from candidate_group.root",
                    "decision": "cluster | do_not_cluster | uncertain | partial",
                    "match_percentage": "0-100",
                    "clusters": [{"row_ids": [], "match_percentage": "0-100", "reason": ""}],
                    "exclude_row_ids": [],
                    "reason": "",
                }
            ]
        },
        "rules": rules,
        "reason_style": "Keep each reason concise, under 25 words.",
        "candidate_groups": list(candidates),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_llm_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}") + 1
    if start >= 0 and end > start:
        stripped = stripped[start:end]
    data = json.loads(stripped)
    return normalize_review_object(data)


def normalize_review_object(data: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(data.get("decision", "uncertain")).lower().strip()
    if decision not in {"cluster", "do_not_cluster", "uncertain", "partial"}:
        decision = "uncertain"
    clusters = []
    for cluster in data.get("clusters") or []:
        row_ids = []
        for row_id in cluster.get("row_ids") or []:
            try:
                row_ids.append(int(row_id))
            except Exception:
                continue
        clusters.append({
            "row_ids": sorted(set(row_ids)),
            "match_percentage": clamp_pct(cluster.get("match_percentage"), 68.0),
            "reason": str(cluster.get("reason", "")),
        })
    return {
        "decision": decision,
        "match_percentage": clamp_pct(data.get("match_percentage"), 68.0),
        "clusters": clusters,
        "exclude_row_ids": [int(x) for x in data.get("exclude_row_ids", []) if str(x).strip().isdigit()],
        "reason": str(data.get("reason", "")),
    }


def parse_llm_batch_json(text: str, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}") + 1
    if start >= 0 and end > start:
        stripped = stripped[start:end]
    data = json.loads(stripped)
    if isinstance(data, list):
        reviews = data
    elif "reviews" in data:
        reviews = data.get("reviews") or []
    elif "decision" in data and len(candidates) == 1:
        reviews = [data]
    else:
        reviews = []
    out = []
    for idx, review in enumerate(reviews):
        if not isinstance(review, dict):
            continue
        parsed = normalize_review_object(review)
        root_value = review.get("group_root", review.get("root"))
        if root_value is None and idx < len(candidates):
            root_value = candidates[idx]["root"]
        try:
            parsed["root"] = int(root_value)
        except Exception:
            continue
        out.append(parsed)
    return out


async def review_one_group(
    client: httpx.AsyncClient,
    candidate: Dict[str, Any],
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
    retry_base_seconds: float,
) -> Dict[str, Any]:
    prompt = build_gemini_prompt(candidate)
    url = base_url.rstrip("/") + f"/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 2048,
        },
    }
    started = time.perf_counter()
    try:
        response: Optional[httpx.Response] = None
        for attempt in range(max_retries + 1):
            response = await client.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                retry_after = response.headers.get("retry-after")
                try:
                    sleep_seconds = float(retry_after) if retry_after else retry_base_seconds * (2 ** attempt)
                except Exception:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
                await asyncio.sleep(min(60.0, max(1.0, sleep_seconds)))
                continue
            break
        assert response is not None
        if response.status_code in {401, 403}:
            return {
                "root": candidate["root"],
                "row_ids": candidate["row_ids"],
                "status": "api_error",
                "fatal": True,
                "http_status": response.status_code,
                "response_preview": response.text[:800],
                "error": f"Gemini authentication/authorization failed with HTTP {response.status_code}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        if response.status_code >= 400:
            return {
                "root": candidate["root"],
                "row_ids": candidate["row_ids"],
                "deterministic_score": candidate["deterministic_score"],
                "match_sources": candidate["match_sources"],
                "risk_flags": candidate["risk_flags"],
                "status": "api_error",
                "fatal": response.status_code in {400, 404},
                "http_status": response.status_code,
                "response_preview": response.text[:800],
                "error": f"Gemini HTTP {response.status_code}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        try:
            parsed = parse_llm_json(text)
            return {
                "root": candidate["root"],
                "row_ids": candidate["row_ids"],
                "deterministic_score": candidate["deterministic_score"],
                "match_sources": candidate["match_sources"],
                "risk_flags": candidate["risk_flags"],
                "status": "success",
                "fatal": False,
                "usage": data.get("usageMetadata", {}),
                "elapsed_seconds": time.perf_counter() - started,
                **parsed,
            }
        except Exception as exc:
            return {
                "root": candidate["root"],
                "row_ids": candidate["row_ids"],
                "deterministic_score": candidate["deterministic_score"],
                "match_sources": candidate["match_sources"],
                "risk_flags": candidate["risk_flags"],
                "status": "json_error",
                "fatal": False,
                "error": str(exc),
                "raw_response_preview": text[:500],
                "elapsed_seconds": time.perf_counter() - started,
            }
    except Exception as exc:
        return {
            "root": candidate["root"],
            "row_ids": candidate["row_ids"],
            "deterministic_score": candidate["deterministic_score"],
            "match_sources": candidate["match_sources"],
            "risk_flags": candidate["risk_flags"],
            "status": "api_error",
            "fatal": False,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(size))
    for idx in range(0, len(items), size):
        yield list(items[idx:idx + size])


async def review_one_batch(
    client: httpx.AsyncClient,
    candidates: List[Dict[str, Any]],
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: int,
    max_retries: int,
    retry_base_seconds: float,
) -> List[Dict[str, Any]]:
    if len(candidates) == 1:
        return [await review_one_group(
            client=client,
            candidate=candidates[0],
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
        )]

    prompt = build_gemini_batch_prompt(candidates)
    url = base_url.rstrip("/") + f"/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 65536,
        },
    }
    started = time.perf_counter()

    def error_results(status: str, error: str, *, fatal: bool = False, http_status: Optional[int] = None, response_preview: str = "") -> List[Dict[str, Any]]:
        return [
            {
                "root": candidate["root"],
                "row_ids": candidate["row_ids"],
                "deterministic_score": candidate["deterministic_score"],
                "match_sources": candidate["match_sources"],
                "risk_flags": candidate["risk_flags"],
                "status": status,
                "fatal": fatal,
                "http_status": http_status,
                "response_preview": response_preview if idx == 0 else "",
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
            }
            for idx, candidate in enumerate(candidates)
        ]

    try:
        response: Optional[httpx.Response] = None
        for attempt in range(max_retries + 1):
            response = await client.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                retry_after = response.headers.get("retry-after")
                try:
                    sleep_seconds = float(retry_after) if retry_after else retry_base_seconds * (2 ** attempt)
                except Exception:
                    sleep_seconds = retry_base_seconds * (2 ** attempt)
                await asyncio.sleep(min(60.0, max(1.0, sleep_seconds)))
                continue
            break
        assert response is not None
        if response.status_code in {401, 403}:
            return error_results(
                "api_error",
                f"Gemini authentication/authorization failed with HTTP {response.status_code}",
                fatal=True,
                http_status=response.status_code,
                response_preview=response.text[:800],
            )
        if response.status_code >= 400:
            return error_results(
                "api_error",
                f"Gemini HTTP {response.status_code}",
                fatal=response.status_code in {400, 404},
                http_status=response.status_code,
                response_preview=response.text[:800],
            )
        data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        try:
            parsed_reviews = parse_llm_batch_json(text, candidates)
            by_root = {int(item["root"]): item for item in parsed_reviews}
            usage = data.get("usageMetadata", {})
            usage_attached = False
            results = []
            for candidate in candidates:
                root = int(candidate["root"])
                parsed = by_root.get(root)
                if not parsed:
                    results.append({
                        "root": root,
                        "row_ids": candidate["row_ids"],
                        "deterministic_score": candidate["deterministic_score"],
                        "match_sources": candidate["match_sources"],
                        "risk_flags": candidate["risk_flags"],
                        "status": "json_error",
                        "fatal": False,
                        "error": "Gemini response did not include this group_root",
                        "raw_response_preview": text[:500] if not usage_attached else "",
                        "elapsed_seconds": time.perf_counter() - started,
                    })
                    usage_attached = True
                    continue
                result = {
                    "root": root,
                    "row_ids": candidate["row_ids"],
                    "deterministic_score": candidate["deterministic_score"],
                    "match_sources": candidate["match_sources"],
                    "risk_flags": candidate["risk_flags"],
                    "status": "success",
                    "fatal": False,
                    "usage": usage if not usage_attached else {},
                    "usage_shared_batch_size": len(candidates) if not usage_attached else 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    **parsed,
                }
                usage_attached = True
                results.append(result)
            return results
        except Exception as exc:
            return error_results("json_error", str(exc), response_preview=text[:800])
    except Exception as exc:
        return error_results("api_error", str(exc))


async def run_gemini_reviews(
    candidates: List[Dict[str, Any]],
    config: ClusteringConfig,
    concurrency: int,
    max_groups: int,
    batch_size: int,
    max_retries: int,
    retry_base_seconds: float,
    stop_after_consecutive_api_errors: int,
    min_seconds_between_calls: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    provider = str(config.ai_provider or "").lower()
    if not config.ai_review_enabled:
        return [], {"enabled": False, "skipped_reason": "AI_REVIEW_ENABLED is not true"}
    if provider not in {"gemini", "google", "google_gemini"}:
        return [], {"enabled": False, "skipped_reason": f"LLM provider is {config.ai_provider!r}, not gemini"}
    if not config.ai_api_key:
        return [], {"enabled": False, "skipped_reason": "Gemini API key missing"}
    if not candidates:
        return [], {"enabled": True, "skipped_reason": "No eligible low-score/ambiguous groups"}

    work = candidates[:max_groups] if max_groups and max_groups > 0 else candidates
    batches = list(chunked(work, batch_size))
    log(
        f"Gemini review enabled: sending {len(work):,} selected groups only; "
        f"calls={len(batches):,}, batch_size={max(1, int(batch_size))}, model={config.ai_model}"
    )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: List[Dict[str, Any]] = []
    fatal_seen = False
    consecutive_api_errors = 0
    started = time.perf_counter()
    rate_lock = asyncio.Lock()
    last_call_started_at = {"value": 0.0}

    async with httpx.AsyncClient() as client:
        async def run_one(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            async with semaphore:
                if min_seconds_between_calls > 0:
                    async with rate_lock:
                        elapsed = time.perf_counter() - last_call_started_at["value"]
                        wait_seconds = float(min_seconds_between_calls) - elapsed
                        if wait_seconds > 0:
                            await asyncio.sleep(wait_seconds)
                        last_call_started_at["value"] = time.perf_counter()
                return await review_one_batch(
                    client=client,
                    candidates=batch,
                    base_url=config.ai_base_url,
                    model=config.ai_model,
                    api_key=config.ai_api_key,
                    timeout_seconds=int(config.ai_timeout_seconds),
                    max_retries=int(max_retries),
                    retry_base_seconds=float(retry_base_seconds),
                )

        tasks = [asyncio.create_task(run_one(batch)) for batch in batches]
        completed = 0
        last_logged = 0
        for task in asyncio.as_completed(tasks):
            batch_results = await task
            results.extend(batch_results)
            completed += len(batch_results)
            for result in batch_results:
                if result.get("status") == "api_error":
                    consecutive_api_errors += 1
                else:
                    consecutive_api_errors = 0
            if any(result.get("fatal") for result in batch_results):
                fatal_seen = True
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                log("Gemini review stopped after a fatal API/model/config error.")
                break
            if stop_after_consecutive_api_errors and consecutive_api_errors >= stop_after_consecutive_api_errors:
                fatal_seen = True
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                log(f"Gemini review stopped after {consecutive_api_errors} consecutive API errors.")
                break
            if completed - last_logged >= 25 or completed == len(work):
                last_logged = completed
                counts = Counter(r.get("decision", r.get("status", "unknown")) for r in results)
                log(f"Gemini progress: {completed:,}/{len(work):,} groups reviewed; decisions/status={dict(counts)}")
        if fatal_seen:
            await asyncio.gather(*tasks, return_exceptions=True)

    return results, {
        "enabled": True,
        "groups_selected": len(candidates),
        "groups_planned_for_review": len(work),
        "groups_sent": len(results),
        "api_calls_planned": len(batches),
        "batch_size": max(1, int(batch_size)),
        "min_seconds_between_calls": float(min_seconds_between_calls),
        "fatal_error": fatal_seen,
        "runtime_seconds": time.perf_counter() - started,
    }


def apply_llm_decisions(
    deterministic_cluster_map: Dict[int, int],
    deterministic_match_pcts: Dict[int, float],
    llm_results: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, Dict[str, Any]]]:
    final_map = dict(deterministic_cluster_map)
    final_scores = dict(deterministic_match_pcts)
    final_sources: Dict[int, Dict[str, Any]] = {}
    deterministic_clusters = clusters_from_map(deterministic_cluster_map)

    for result in llm_results:
        if result.get("status") != "success":
            continue
        root = int(result["root"])
        original_rows = set(deterministic_clusters.get(root, []))
        if not original_rows:
            continue
        decision = result.get("decision")
        pct = clamp_pct(result.get("match_percentage"), deterministic_match_pcts.get(root, 68.0))
        if decision == "cluster":
            final_scores[root] = pct
            final_sources[root] = {"source": "gemini_cluster", "llm_result": result}
        elif decision == "uncertain":
            final_scores[root] = max(65.0, min(70.0, pct))
            final_sources[root] = {"source": "gemini_uncertain", "llm_result": result}
        elif decision == "do_not_cluster":
            for row_id in original_rows:
                final_map.pop(row_id, None)
            final_scores.pop(root, None)
            final_sources[root] = {"source": "gemini_do_not_cluster", "llm_result": result}
        elif decision == "partial":
            for row_id in original_rows:
                final_map.pop(row_id, None)
            final_scores.pop(root, None)
            final_sources[root] = {"source": "gemini_partial_parent", "llm_result": result}
            for cluster in result.get("clusters", []):
                rows = sorted(set(int(rid) for rid in cluster.get("row_ids", []) if int(rid) in original_rows))
                if len(rows) < 2:
                    continue
                new_root = min(rows)
                for row_id in rows:
                    final_map[row_id] = new_root
                final_scores[new_root] = clamp_pct(cluster.get("match_percentage"), 68.0)
                final_sources[new_root] = {
                    "source": "gemini_partial_child",
                    "parent_root": root,
                    "llm_cluster_reason": cluster.get("reason", ""),
                    "llm_result": result,
                }
    return final_map, final_scores, final_sources


def final_audit_rows(
    row_by_id: Dict[int, Dict[str, Any]],
    final_cluster_map: Dict[int, int],
    final_match_pcts: Dict[int, float],
    deterministic_audit: Dict[int, Dict[str, Any]],
    deterministic_root_by_row: Dict[int, int],
    final_sources: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    number_by_root = cluster_number_map(final_cluster_map)
    rows = []
    for root, row_ids in clusters_from_map(final_cluster_map).items():
        det_roots = sorted({deterministic_root_by_row.get(row_id) for row_id in row_ids if row_id in deterministic_root_by_row})
        meta = deterministic_audit.get(root) or (deterministic_audit.get(det_roots[0]) if det_roots else {}) or {}
        source = final_sources.get(root, {"source": "deterministic"})
        llm_result = source.get("llm_result", {})
        rows.append({
            "cluster_number": number_by_root.get(root),
            "cluster_root": root,
            "cluster_size": len(row_ids),
            "match_pct": final_match_pcts.get(root, 0.0),
            "final_source": source.get("source", "deterministic"),
            "llm_decision": llm_result.get("decision", ""),
            "llm_reason": llm_result.get("reason", ""),
            "deterministic_roots": "|".join(str(x) for x in det_roots if x is not None),
            "primary_pass": meta.get("primary_pass", ""),
            "pass_types": ", ".join(meta.get("pass_types", []) or []),
            "needs_review": meta.get("needs_review", False),
            "review_reasons": "; ".join(meta.get("review_reasons", []) or []),
            "row_ids": "|".join(str(x) for x in row_ids),
            "supplier_names_sample": " | ".join(names_for_rows(row_by_id, row_ids, 10)),
        })
    return sorted(rows, key=lambda r: (r["cluster_number"] or 0))


def find_matching_row_ids(row_by_id: Dict[int, Dict[str, Any]], query: str) -> List[int]:
    q = name_search_key(query)
    q_tokens = [t for t in q.split() if t]
    matches = []
    for row_id, row in row_by_id.items():
        raw = name_search_key(str(row.get("orig_supplier_name", "")))
        if not raw:
            continue
        if q and q in raw:
            matches.append(row_id)
            continue
        if q_tokens and all(token in raw.split() for token in q_tokens):
            matches.append(row_id)
    return sorted(matches)


def case_clusters(row_by_id: Dict[int, Dict[str, Any]], final_cluster_map: Dict[int, int], queries: Sequence[str]) -> Dict[str, Any]:
    found = {}
    for query in queries:
        ids = find_matching_row_ids(row_by_id, query)
        found[query] = [
            {
                "row_id": row_id,
                "name": row_by_id[row_id].get("orig_supplier_name", ""),
                "city": row_by_id[row_id].get("orig_city", ""),
                "country": row_by_id[row_id].get("orig_country", ""),
                "cluster_root": final_cluster_map.get(row_id),
            }
            for row_id in ids[:10]
        ]
    roots = {
        item["cluster_root"]
        for items in found.values()
        for item in items
        if item["cluster_root"] is not None
    }
    return {"queries": list(queries), "found": found, "cluster_roots_found": sorted(roots)}


def verify_bad_cases(row_by_id: Dict[int, Dict[str, Any]], final_cluster_map: Dict[int, int]) -> List[Dict[str, Any]]:
    cases = [
        ("202/FIT Production", ["202 Production", "FIT Production"], "all_distinct_or_singleton"),
        ("30 Point/Chemical Point", ["30 Point Strategies", "CHEMICAL POINT"], "all_distinct_or_singleton"),
        ("3B/CJSC Scientific", ["3B Scientific Corporation", "CJSC Scientific Center of Drug"], "all_distinct_or_singleton"),
        ("3WAY/Alps Pharm", ["3WAY PHARM", "Alps Pharm"], "all_distinct_or_singleton"),
        ("Aaron individuals", ["Aaron Lackner", "Aaron Lawson McLean", "Aaron Tan"], "all_distinct_or_singleton"),
        ("Alexandre individuals", ["Alexandre Guiraud", "Alexandre Prat", "Alexandre Varnek"], "all_distinct_or_singleton"),
        ("Alice individuals", ["Alice Antonello", "Alice Hoffmann-Ziegler", "Alice Lichtenberg"], "all_distinct_or_singleton"),
        ("Kehl household", ["Alexander Kehl", "Waldemar Kehl"], "all_distinct_or_singleton"),
        ("Knoll household", ["Aline Knoll", "Kevin Knoll"], "all_distinct_or_singleton"),
        ("Daechert household", ["Andrea Daechert", "Juergen Daechert"], "all_distinct_or_singleton"),
        ("Access false positive", ["Access BIO", "Access Events International", "OPEN Access Consulting"], "all_distinct_or_singleton"),
        ("Cologne/Publishing false positive", ["AdCoach Marketing", "Cologne Energy", "Cologne Publishing"], "all_distinct_or_singleton"),
        ("Beijing Skywing/Wonder/Merck", ["Beijing Skywing Technology", "BEIJING WONDER UNION TECHNOLOGY", "Merck Millipore Beijing Skywing"], "wonder_not_with_others"),
    ]
    results = []
    for name, queries, expected in cases:
        detail = case_clusters(row_by_id, final_cluster_map, queries)
        cluster_members_by_query = {}
        for query, found in detail["found"].items():
            cluster_members_by_query[query] = {item["cluster_root"] for item in found if item["cluster_root"] is not None}
        fixed = True
        if expected == "all_distinct_or_singleton":
            roots_seen: Dict[Any, List[str]] = defaultdict(list)
            for query, roots in cluster_members_by_query.items():
                for root in roots:
                    roots_seen[root].append(query)
            fixed = all(len(labels) <= 1 for labels in roots_seen.values())
        elif expected == "wonder_not_with_others":
            wonder_roots = cluster_members_by_query.get("BEIJING WONDER UNION TECHNOLOGY", set())
            other_roots = set()
            for query in queries:
                if query != "BEIJING WONDER UNION TECHNOLOGY":
                    other_roots |= cluster_members_by_query.get(query, set())
            fixed = not (wonder_roots & other_roots)
        detail["case"] = name
        detail["expected"] = expected
        detail["fixed"] = fixed
        results.append(detail)

    weak_terms = ["Akzo", "Merck", "Stockmeier", "Cognis", "Thermconcept", "Gaerner", "Hopf", "Schircks"]
    root_hits: Dict[int, Set[str]] = defaultdict(set)
    for term in weak_terms:
        for row_id in find_matching_row_ids(row_by_id, term):
            root = final_cluster_map.get(row_id)
            if root is not None:
                root_hits[root].add(term)
    suspicious_roots = {root: sorted(terms) for root, terms in root_hits.items() if len(terms) >= 4}
    results.append({
        "case": "Akzo/Merck/etc weak-chain issue",
        "queries": weak_terms,
        "expected": "no_cluster_mixes_four_or_more_watch_terms",
        "fixed": not suspicious_roots,
        "suspicious_roots": suspicious_roots,
    })
    return results


def top_cluster_review(
    row_by_id: Dict[int, Dict[str, Any]],
    final_cluster_map: Dict[int, int],
    final_match_pcts: Dict[int, float],
    deterministic_audit: Dict[int, Dict[str, Any]],
    deterministic_root_by_row: Dict[int, int],
    final_sources: Dict[int, Dict[str, Any]],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    number_by_root = cluster_number_map(final_cluster_map)
    reviews = []
    for root, row_ids in clusters_from_map(final_cluster_map).items():
        score = float(final_match_pcts.get(root, 0.0))
        det_roots = sorted({deterministic_root_by_row.get(row_id) for row_id in row_ids if row_id in deterministic_root_by_row})
        meta = deterministic_audit.get(root) or (deterministic_audit.get(det_roots[0]) if det_roots else {}) or {}
        pass_types = set(meta.get("pass_types", []) or [])
        rows = [row_by_id[row_id] for row_id in row_ids]
        flags = risk_flags_for_group(rows, meta, score)
        if is_obvious_generic_only_cluster(rows, pass_types):
            flags.append("suspicious_generic_only_overlap")
        if "suspicious_generic_only_overlap" in flags or (len(row_ids) >= 15 and score < 80):
            verdict = "suspicious"
        elif score >= 90 and pass_types and pass_types <= STRONG_PASS_TYPES:
            verdict = "safe"
        else:
            verdict = "review-needed"
        reviews.append({
            "cluster_number": number_by_root.get(root),
            "cluster_root": root,
            "cluster_size": len(row_ids),
            "average_match_percentage": score,
            "lowest_match_percentage": score,
            "top_10_supplier_names": names_for_rows(row_by_id, row_ids, 10),
            "main_reason_source": final_sources.get(root, {}).get("source", "deterministic"),
            "pass_types": sorted(pass_types),
            "risk_flags": sorted(set(flags)),
            "verdict": verdict,
        })
    return sorted(reviews, key=lambda r: (-r["cluster_size"], r["cluster_number"] or 0))[:limit]


def low_score_review(
    row_by_id: Dict[int, Dict[str, Any]],
    final_cluster_map: Dict[int, int],
    final_match_pcts: Dict[int, float],
    deterministic_audit: Dict[int, Dict[str, Any]],
    deterministic_root_by_row: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    number_by_root = cluster_number_map(final_cluster_map)
    examples = []
    reasons: Counter[str] = Counter()
    for root, row_ids in clusters_from_map(final_cluster_map).items():
        score = float(final_match_pcts.get(root, 0.0))
        if not (65 <= score <= 76):
            continue
        det_roots = sorted({deterministic_root_by_row.get(row_id) for row_id in row_ids if row_id in deterministic_root_by_row})
        meta = deterministic_audit.get(root) or (deterministic_audit.get(det_roots[0]) if det_roots else {}) or {}
        pass_types = sorted(meta.get("pass_types", []) or [])
        reasons.update(pass_types or ["unknown"])
        examples.append({
            "cluster_number": number_by_root.get(root),
            "cluster_root": root,
            "cluster_size": len(row_ids),
            "match_percentage": score,
            "pass_types": pass_types,
            "review_reasons": meta.get("review_reasons", []) or [],
            "supplier_names_sample": names_for_rows(row_by_id, row_ids, 10),
        })
    examples.sort(key=lambda r: (r["match_percentage"], -r["cluster_size"], r["cluster_number"] or 0))
    return examples, reasons


def suspicious_generic_scan(
    row_by_id: Dict[int, Dict[str, Any]],
    final_cluster_map: Dict[int, int],
    final_match_pcts: Dict[int, float],
    deterministic_audit: Dict[int, Dict[str, Any]],
    deterministic_root_by_row: Dict[int, int],
) -> List[Dict[str, Any]]:
    number_by_root = cluster_number_map(final_cluster_map)
    findings = []
    for root, row_ids in clusters_from_map(final_cluster_map).items():
        score = float(final_match_pcts.get(root, 0.0))
        if score > 85:
            continue
        det_roots = sorted({deterministic_root_by_row.get(row_id) for row_id in row_ids if row_id in deterministic_root_by_row})
        meta = deterministic_audit.get(root) or (deterministic_audit.get(det_roots[0]) if det_roots else {}) or {}
        rows = [row_by_id[row_id] for row_id in row_ids]
        pass_types = set(meta.get("pass_types", []) or [])
        shared = cluster_shared_tokens(rows)
        distinct = distinctive_shared_tokens(rows)
        only_generic = bool(shared and not distinct)
        if only_generic or (shared & NON_BRIDGE_TOKENS and score < 80):
            findings.append({
                "cluster_number": number_by_root.get(root),
                "cluster_root": root,
                "cluster_size": len(row_ids),
                "match_percentage": score,
                "only_shared_tokens_are_generic_or_location": only_generic,
                "shared_tokens": sorted(shared)[:30],
                "distinctive_shared_tokens": sorted(distinct)[:30],
                "pass_types": sorted(pass_types),
                "supplier_names_sample": names_for_rows(row_by_id, row_ids, 10),
            })
    return sorted(findings, key=lambda r: (not r["only_shared_tokens_are_generic_or_location"], r["match_percentage"], -r["cluster_size"]))[:100]


def markdown_bad_cases(results: List[Dict[str, Any]]) -> str:
    lines = ["# Bad-Case Verification", ""]
    for result in results:
        status = "FIXED" if result.get("fixed") else "NOT FIXED"
        lines.append(f"## {result['case']}: {status}")
        if result.get("suspicious_roots"):
            lines.append(f"- suspicious_roots: `{json.dumps(result['suspicious_roots'], ensure_ascii=False)}`")
        for query, found in (result.get("found") or {}).items():
            compact = [
                f"{item['row_id']} | {item['name']} | cluster={item['cluster_root']}"
                for item in found[:5]
            ]
            lines.append(f"- {query}: " + ("; ".join(compact) if compact else "not found"))
        lines.append("")
    return "\n".join(lines)


def markdown_top_clusters(reviews: List[Dict[str, Any]]) -> str:
    lines = ["# Top 25 Largest Cluster Review", ""]
    for item in reviews:
        lines.append(f"## Cluster {item['cluster_number']} | size {item['cluster_size']} | verdict {item['verdict']}")
        lines.append(f"- score_avg: {item['average_match_percentage']:.1f}")
        lines.append(f"- score_low: {item['lowest_match_percentage']:.1f}")
        lines.append(f"- source: {item['main_reason_source']}")
        lines.append(f"- pass_types: {', '.join(item['pass_types'])}")
        lines.append(f"- risk_flags: {', '.join(item['risk_flags'])}")
        lines.append("- names:")
        for name in item["top_10_supplier_names"]:
            lines.append(f"  - {name}")
        lines.append("")
    return "\n".join(lines)


def markdown_low_score(examples: List[Dict[str, Any]], reasons: Counter[str]) -> str:
    lines = ["# Low-Score Cluster Review", ""]
    lines.append(f"- low_score_cluster_count: {len(examples)}")
    lines.append(f"- common_reasons: {dict(reasons.most_common(20))}")
    lines.append("- usefulness: review-oriented; these are not auto-approval-ready.")
    lines.append("")
    for item in examples[:20]:
        lines.append(f"## Cluster {item['cluster_number']} | score {item['match_percentage']:.1f} | size {item['cluster_size']}")
        lines.append(f"- pass_types: {', '.join(item['pass_types'])}")
        lines.append(f"- review_reasons: {'; '.join(item['review_reasons'])}")
        lines.append("- names:")
        for name in item["supplier_names_sample"]:
            lines.append(f"  - {name}")
        lines.append("")
    return "\n".join(lines)


def markdown_suspicious_generic(findings: List[Dict[str, Any]]) -> str:
    lines = ["# Suspicious Generic Low-Score Scan", ""]
    lines.append(f"- finding_count: {len(findings)}")
    lines.append("")
    for item in findings[:50]:
        lines.append(f"## Cluster {item['cluster_number']} | score {item['match_percentage']:.1f} | size {item['cluster_size']}")
        lines.append(f"- only_generic_or_location: {item['only_shared_tokens_are_generic_or_location']}")
        lines.append(f"- shared_tokens: {', '.join(item['shared_tokens'])}")
        lines.append(f"- pass_types: {', '.join(item['pass_types'])}")
        lines.append("- names:")
        for name in item["supplier_names_sample"]:
            lines.append(f"  - {name}")
        lines.append("")
    return "\n".join(lines)


def markdown_llm_report(metrics: Dict[str, Any], examples: List[Dict[str, Any]]) -> str:
    lines = ["# Gemini LLM Review Report", ""]
    for key in [
        "enabled",
        "skipped_reason",
        "groups_selected",
        "groups_planned_for_review",
        "groups_sent",
        "total_rows_sent",
        "model_used",
        "runtime_seconds",
        "json_parsing_errors",
        "api_errors",
        "partial_decisions",
    ]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    lines.append(f"- decisions_count: {metrics.get('decisions_count', {})}")
    lines.append(f"- token_usage: {metrics.get('token_usage', {})}")
    lines.append("- cost_estimate: not calculated by this runner")
    lines.append("")
    lines.append("## Decision Examples")
    for item in examples[:20]:
        lines.append(f"- root {item.get('root')}: {item.get('decision', item.get('status'))} score={item.get('match_percentage', '')} rows={item.get('row_ids', [])[:10]} reason={item.get('reason', item.get('error', ''))[:240]}")
    lines.append("")
    return "\n".join(lines)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        flat_rows = []
        for row in rows:
            flat_rows.append({
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict, set, tuple))
                    else value
                )
                for key, value in row.items()
            })
        pl.DataFrame(flat_rows).write_csv(path)
    else:
        pl.DataFrame({"message": ["No rows"]}).write_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full deterministic + controlled Gemini end-to-end QA.")
    parser.add_argument("--input", default="data/sample_suppliers_100.csv")
    parser.add_argument("--mapping")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--prefix", default="e2e_gemini_full")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-total-candidate-pairs", type=int, default=1_000_000)
    parser.add_argument("--llm-score-min", type=float, default=65.0)
    parser.add_argument("--llm-score-max", type=float, default=80.0)
    parser.add_argument("--max-llm-groups", type=int, default=0)
    parser.add_argument("--llm-concurrency", type=int, default=3)
    parser.add_argument("--llm-batch-size", type=int, default=1)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-retry-base-seconds", type=float, default=3.0)
    parser.add_argument("--llm-stop-after-consecutive-api-errors", type=int, default=50)
    parser.add_argument("--llm-min-seconds-between-calls", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    prefix = args.prefix
    paths = {
        "deterministic_output": out_dir / f"{prefix}_deterministic_clustered.csv",
        "deterministic_audit": out_dir / f"{prefix}_deterministic_audit.csv",
        "deterministic_report": out_dir / f"{prefix}_deterministic_report.txt",
        "deterministic_metrics": out_dir / f"{prefix}_deterministic_metrics.json",
        "final_output": out_dir / f"{prefix}_final_clustered.csv",
        "final_audit": out_dir / f"{prefix}_final_audit.csv",
        "final_metrics": out_dir / f"{prefix}_final_metrics.json",
        "bad_cases": out_dir / f"{prefix}_bad_case_verification.md",
        "top25": out_dir / f"{prefix}_top25_cluster_review.md",
        "low_score": out_dir / f"{prefix}_low_score_cluster_review.md",
        "llm_report": out_dir / f"{prefix}_llm_review_report.md",
        "llm_metrics": out_dir / f"{prefix}_llm_metrics.json",
        "suspicious_generic": out_dir / f"{prefix}_suspicious_generic_low_score_scan.md",
        "suspicious_generic_csv": out_dir / f"{prefix}_suspicious_generic_low_score_scan.csv",
    }

    log("Starting end-to-end deterministic + Gemini QA.")
    log("Loading input file...")
    t0 = time.perf_counter()
    raw_df = read_supplier_file(args.input, args.input, max_rows=args.max_rows)
    file_load_seconds = time.perf_counter() - t0
    log(f"Loaded {len(raw_df):,} rows and {len(raw_df.columns)} columns in {file_load_seconds:.2f}s")

    if args.mapping:
        mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    else:
        mapping = auto_detect_columns(raw_df.columns)
    log(f"Column mapping: {mapping}")

    deterministic_config = ClusteringConfig.from_env()
    deterministic_config.ai_review_enabled = False
    deterministic_config.max_total_candidate_pairs = int(args.max_total_candidate_pairs)
    log("Running deterministic clustering with LLM disabled...")
    det_started = time.perf_counter()
    result = cluster_suppliers(raw_df, mapping, deterministic_config)
    deterministic_runtime = time.perf_counter() - det_started
    result["stats"].setdefault("stage_timings", {})["file_load_seconds"] = file_load_seconds
    result["stats"]["deterministic_wrapper_runtime_seconds"] = deterministic_runtime
    result["stats"]["max_rss_mb"] = rss_mb()

    log("Saving deterministic output/audit/report/metrics...")
    save_main_output(result["main_df"], str(paths["deterministic_output"]))
    save_audit_file(result["audit_data"], str(paths["deterministic_audit"]), result["preprocessed_df"], result["cluster_map"], result["merger"])
    generate_processing_report(result["stats"], str(paths["deterministic_report"]))

    preprocessed_df = result["preprocessed_df"]
    row_by_id = row_lookup(preprocessed_df)
    deterministic_cluster_map = {int(k): int(v) for k, v in result["cluster_map"].items()}
    deterministic_match_pcts = calculate_cluster_scores(result["merger"])
    deterministic_root_by_row = dict(deterministic_cluster_map)
    det_anchor = verify_anchor_order(preprocessed_df, deterministic_cluster_map)
    expected_columns = list(raw_df.columns) + ["Cluster Number", "Match Percentage"]
    deterministic_output_columns_correct = result["main_df"].columns == expected_columns
    write_json(paths["deterministic_metrics"], {
        "input": args.input,
        "generated_at": datetime.now().isoformat(),
        "mapping": mapping,
        "llm_enabled_for_deterministic_run": False,
        "stats": result["stats"],
        "anchor_verification": det_anchor,
        "output_row_count_preserved": len(result["main_df"]) == len(raw_df),
        "output_columns_correct": deterministic_output_columns_correct,
        "output_columns": result["main_df"].columns,
        "expected_columns": expected_columns,
    })

    candidates, skipped_llm = select_llm_candidates(
        row_by_id=row_by_id,
        cluster_map=deterministic_cluster_map,
        match_pcts=deterministic_match_pcts,
        audit_data=result["audit_data"],
        min_score=float(args.llm_score_min),
        max_score=float(args.llm_score_max),
    )
    log(f"Selected {len(candidates):,} low-score/ambiguous groups for Gemini; skipped {len(skipped_llm):,} obvious generic/location-only groups.")

    llm_config = ClusteringConfig.from_env()
    llm_config.ai_timeout_seconds = int(llm_config.ai_timeout_seconds or 45)
    llm_results, llm_run_meta = asyncio.run(run_gemini_reviews(
        candidates=candidates,
        config=llm_config,
        concurrency=int(args.llm_concurrency),
        max_groups=int(args.max_llm_groups),
        batch_size=int(args.llm_batch_size),
        max_retries=int(args.llm_max_retries),
        retry_base_seconds=float(args.llm_retry_base_seconds),
        stop_after_consecutive_api_errors=int(args.llm_stop_after_consecutive_api_errors),
        min_seconds_between_calls=float(args.llm_min_seconds_between_calls),
    ))

    final_cluster_map, final_match_pcts, final_sources = apply_llm_decisions(
        deterministic_cluster_map=deterministic_cluster_map,
        deterministic_match_pcts=deterministic_match_pcts,
        llm_results=llm_results,
    )

    log("Applying final cluster decisions and anchor-based ordering...")
    final_output_df = sort_by_clusters(preprocessed_df, final_cluster_map, final_match_pcts)
    save_main_output(final_output_df, str(paths["final_output"]))

    final_audit = final_audit_rows(
        row_by_id=row_by_id,
        final_cluster_map=final_cluster_map,
        final_match_pcts=final_match_pcts,
        deterministic_audit=result["audit_data"],
        deterministic_root_by_row=deterministic_root_by_row,
        final_sources=final_sources,
    )
    write_csv_rows(paths["final_audit"], final_audit)

    final_anchor = verify_anchor_order(preprocessed_df, final_cluster_map)
    final_clusters = clusters_from_map(final_cluster_map)
    rows_clustered = sum(len(rows) for rows in final_clusters.values())
    rows_not_clustered = len(raw_df) - rows_clustered
    final_output_columns_correct = final_output_df.columns == expected_columns
    duplicate_lost = not final_anchor["no_rows_lost_or_duplicated_by_ordering"]

    bad_case_results = verify_bad_cases(row_by_id, final_cluster_map)
    write_text(paths["bad_cases"], markdown_bad_cases(bad_case_results))

    top25 = top_cluster_review(
        row_by_id=row_by_id,
        final_cluster_map=final_cluster_map,
        final_match_pcts=final_match_pcts,
        deterministic_audit=result["audit_data"],
        deterministic_root_by_row=deterministic_root_by_row,
        final_sources=final_sources,
        limit=25,
    )
    write_text(paths["top25"], markdown_top_clusters(top25))

    low_score_examples, low_score_reasons = low_score_review(
        row_by_id=row_by_id,
        final_cluster_map=final_cluster_map,
        final_match_pcts=final_match_pcts,
        deterministic_audit=result["audit_data"],
        deterministic_root_by_row=deterministic_root_by_row,
    )
    write_text(paths["low_score"], markdown_low_score(low_score_examples, low_score_reasons))

    generic_findings = suspicious_generic_scan(
        row_by_id=row_by_id,
        final_cluster_map=final_cluster_map,
        final_match_pcts=final_match_pcts,
        deterministic_audit=result["audit_data"],
        deterministic_root_by_row=deterministic_root_by_row,
    )
    write_text(paths["suspicious_generic"], markdown_suspicious_generic(generic_findings))
    write_csv_rows(paths["suspicious_generic_csv"], generic_findings)

    decision_counts = Counter(r.get("decision", r.get("status", "unknown")) for r in llm_results)
    status_counts = Counter(r.get("status", "unknown") for r in llm_results)
    token_usage = Counter()
    for item in llm_results:
        for key, value in (item.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                token_usage[key] += value
    llm_metrics = {
        **llm_run_meta,
        "model_used": llm_config.ai_model if llm_run_meta.get("enabled") else "",
        "provider": llm_config.ai_provider if llm_run_meta.get("enabled") else "",
        "groups_selected": len(candidates),
        "groups_sent": llm_run_meta.get("groups_sent", len(llm_results)),
        "groups_skipped_obvious_generic_or_location_only": len(skipped_llm),
        "total_rows_sent": sum(len(item.get("row_ids", [])) for item in llm_results),
        "decisions_count": dict(decision_counts),
        "status_count": dict(status_counts),
        "json_parsing_errors": status_counts.get("json_error", 0),
        "api_errors": status_counts.get("api_error", 0),
        "partial_decisions": decision_counts.get("partial", 0),
        "token_usage": dict(token_usage),
        "examples": llm_results[:20],
        "skipped_llm_sample": skipped_llm[:20],
    }
    write_json(paths["llm_metrics"], llm_metrics)
    write_text(paths["llm_report"], markdown_llm_report(llm_metrics, llm_results[:20]))

    final_metrics = {
        "input": args.input,
        "generated_at": datetime.now().isoformat(),
        "rows_processed": len(raw_df),
        "deterministic_runtime_seconds": result["stats"].get("processing_time_seconds"),
        "end_to_end_runtime_seconds": file_load_seconds + deterministic_runtime + float(llm_run_meta.get("runtime_seconds", 0.0)),
        "candidate_pairs": result["stats"].get("candidate_pairs"),
        "match_edges": result["stats"].get("match_edges_created"),
        "deterministic_clusters_created": result["stats"].get("clusters_found"),
        "final_clusters_created": len(final_clusters),
        "rows_clustered": rows_clustered,
        "rows_not_clustered": rows_not_clustered,
        "largest_cluster_size": max((len(rows) for rows in final_clusters.values()), default=0),
        "top_25_largest_clusters": top25,
        "candidate_cap_hit": result["stats"].get("candidate_pairs_capped", False),
        "skipped_oversized_blocks": result["stats"].get("blocking_diagnostics", {}).get("skipped_oversized_blocks", 0),
        "memory_max_rss_mb": rss_mb(),
        "output_row_count_preserved": len(final_output_df) == len(raw_df),
        "duplicate_or_lost_rows": duplicate_lost,
        "anchor_ordering": final_anchor,
        "output_columns_correct": final_output_columns_correct,
        "output_columns": final_output_df.columns,
        "expected_columns": expected_columns,
        "bad_case_verification": bad_case_results,
        "suspicious_generic_low_score_count": len(generic_findings),
        "llm_metrics_path": str(paths["llm_metrics"]),
    }
    write_json(paths["final_metrics"], final_metrics)

    log("End-to-end QA artifacts written:")
    for key, path in paths.items():
        log(f"  {key}: {path}")
    log("Summary:")
    log(f"  rows={len(raw_df):,} candidate_pairs={result['stats'].get('candidate_pairs'):,} match_edges={result['stats'].get('match_edges_created'):,}")
    log(f"  final_clusters={len(final_clusters):,} rows_clustered={rows_clustered:,} rows_not_clustered={rows_not_clustered:,} largest_cluster={final_metrics['largest_cluster_size']:,}")
    log(f"  output_columns_correct={final_output_columns_correct} anchor_ordering={final_anchor['anchor_order_works']} duplicate_or_lost_rows={duplicate_lost}")
    log(f"  bad_cases_fixed={all(item.get('fixed') for item in bad_case_results)} llm_status={llm_metrics.get('status_count', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

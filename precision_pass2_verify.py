"""QA report for the latest precision-pass full-file output."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(os.getenv("SUPPLIER_QA_INPUT", str(ROOT / "data/sample_suppliers_100.csv")))
OUTPUT = ROOT / "output/precision_pass2_full_clustered.csv"
METRICS = ROOT / "output/precision_pass2_full_metrics.json"
REPORT_MD = ROOT / "output/precision_pass2_case_verification.md"
REPORT_JSON = ROOT / "output/precision_pass2_case_verification.json"
TOP25_MD = ROOT / "output/precision_pass2_top25_cluster_review.md"
GENERIC_SCAN_MD = ROOT / "output/precision_pass2_suspicious_generic_low_score_scan.md"
GENERIC_SCAN_CSV = ROOT / "output/precision_pass2_suspicious_generic_low_score_scan.csv"
REVIEW_CANDIDATES = ROOT / "output/precision_pass2_family_review_candidates.csv"


GENERIC_TOKENS = {
    "access", "advanced", "akademie", "academy", "airport", "analytical",
    "association", "automation", "bio", "biochem", "bioscience",
    "biosciences", "biotech", "biotechnology", "brand", "bv", "cargo",
    "center", "centre", "chemical", "chemicals", "clinical", "clinic",
    "co", "college", "community", "company", "consulting", "corp",
    "corporation", "data", "diagnostic", "distribution", "drug", "drugs",
    "electronics", "energy", "engineering", "events", "express", "gemini",
    "global", "gmbh", "green", "group", "healthcare", "hospital", "inc",
    "industrial", "industries", "institute", "instruments", "international",
    "jasmin", "lab", "laboratories", "laboratory", "labs", "life",
    "limited", "logistics", "ltd", "manufacturing", "marketing",
    "material", "materials", "medical", "network", "open", "packaging",
    "partnership", "performance", "pharm", "pharma", "pharmaceutical",
    "pharmaceuticals", "point", "production", "publishing", "research",
    "red", "sales", "science", "scientific", "service", "services", "sigma",
    "society", "software", "solutions", "standards", "strategy",
    "strategies", "supplies", "supply", "systems", "technology",
    "technologies", "terminal", "testing", "trading", "trucks",
    "university", "blue",
}

try:
    from src.generic_keywords import load_generic_non_bridge_keywords

    GENERIC_TOKENS |= load_generic_non_bridge_keywords(
        str(ROOT / "data/generic_non_bridge_keywords.csv"),
        include_defaults=True,
    ).keywords
except Exception:
    pass


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def norm(value: Any) -> str:
    value = text(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def tokens(value: Any) -> set[str]:
    return {tok for tok in norm(value).split() if len(tok) > 1}


def pct_to_float(value: Any) -> float:
    raw = text(value).strip().replace("%", "")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def row_label(row: dict[str, Any]) -> str:
    cluster = row.get("Cluster Number")
    pct = text(row.get("Match Percentage"))
    cluster_text = "None" if cluster is None else str(cluster)
    return (
        f"{text(row.get('ExternalId'))} | {text(row.get('Name'))} | "
        f"{text(row.get('Address_Line1'))} | {text(row.get('Address_City'))} | "
        f"cluster={cluster_text} | {pct}"
    )


def find_rows(rows: list[dict[str, Any]], pattern: str) -> list[dict[str, Any]]:
    needle = norm(pattern)
    return [row for row in rows if needle in norm(row.get("Name"))]


def cluster_set(rows: list[dict[str, Any]]) -> set[int]:
    return {int(row["Cluster Number"]) for row in rows if row.get("Cluster Number") is not None}


def summarize_matches(rows: list[dict[str, Any]], limit: int = 8) -> str:
    if not rows:
        return "_not found_"
    return "; ".join(row_label(row) for row in rows[:limit])


def should_not_overlap(
    name: str,
    rows: list[dict[str, Any]],
    groups: dict[str, str],
) -> dict[str, Any]:
    found = {label: find_rows(rows, pattern) for label, pattern in groups.items()}
    overlaps = []
    labels = list(found)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            shared = sorted(cluster_set(found[left]) & cluster_set(found[right]))
            if shared:
                overlaps.append({"a": left, "b": right, "shared_clusters": shared})
    return {
        "name": name,
        "type": "should_not_cluster",
        "fixed": not overlaps,
        "details": {"overlaps": overlaps},
        "groups": {label: summarize_matches(matches) for label, matches in found.items()},
    }


def should_cluster(
    name: str,
    rows: list[dict[str, Any]],
    groups: dict[str, str],
) -> dict[str, Any]:
    found = {label: find_rows(rows, pattern) for label, pattern in groups.items()}
    cluster_sets = {label: cluster_set(matches) for label, matches in found.items()}
    common: set[int] | None = None
    for clusters in cluster_sets.values():
        common = set(clusters) if common is None else common & clusters
    common = common or set()
    return {
        "name": name,
        "type": "should_cluster",
        "fixed": bool(common),
        "details": {
            "common_clusters": sorted(common),
            "cluster_sets": {label: sorted(clusters) for label, clusters in cluster_sets.items()},
        },
        "groups": {label: summarize_matches(matches) for label, matches in found.items()},
    }


def should_be_review_only(
    name: str,
    rows: list[dict[str, Any]],
    groups: dict[str, str],
) -> dict[str, Any]:
    """Low-confidence useful candidates should not get main cluster IDs."""
    found = {label: find_rows(rows, pattern) for label, pattern in groups.items()}
    found_all = all(found.values())
    any_main_cluster = any(cluster_set(matches) for matches in found.values())
    return {
        "name": name,
        "type": "review_only_expected",
        "fixed": bool(found_all and not any_main_cluster),
        "details": {
            "main_output_clusters": {label: sorted(cluster_set(matches)) for label, matches in found.items()},
            "policy": "Expected to remain audit/review-only because weak/review-only clusters are excluded from main Cluster Number.",
        },
        "groups": {label: summarize_matches(matches) for label, matches in found.items()},
    }


def review_candidate_pair_exists(review_rows: list[dict[str, Any]], left_pattern: str, right_pattern: str) -> bool:
    left = norm(left_pattern)
    right = norm(right_pattern)
    for row in review_rows:
        name_1 = norm(row.get("supplier_name_1"))
        name_2 = norm(row.get("supplier_name_2"))
        if (left in name_1 and right in name_2) or (left in name_2 and right in name_1):
            return True
    return False


def should_have_review_candidates(
    name: str,
    review_rows: list[dict[str, Any]],
    required_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    pair_results = {
        f"{left} / {right}": review_candidate_pair_exists(review_rows, left, right)
        for left, right in required_pairs
    }
    examples = []
    for row in review_rows:
        combined = f"{text(row.get('supplier_name_1'))} | {text(row.get('supplier_name_2'))}"
        if any(norm(pattern) in norm(combined) for pair in required_pairs for pattern in pair):
            examples.append(
                f"{text(row.get('supplier_name_1'))} <-> {text(row.get('supplier_name_2'))} "
                f"| {text(row.get('pass_type'))} | {text(row.get('score'))}"
            )
        if len(examples) >= 8:
            break
    return {
        "name": name,
        "type": "review_candidates_expected",
        "fixed": all(pair_results.values()),
        "details": {"required_pairs": pair_results},
        "groups": {"review_candidate_examples": "; ".join(examples) if examples else "_not found_"},
    }


def grouped_output(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster = row.get("Cluster Number")
        if cluster is not None:
            grouped[int(cluster)].append(row)
    return grouped


def top_clusters(rows: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    grouped = grouped_output(rows)
    out = []
    for cluster, members in grouped.items():
        pcts = [pct_to_float(row.get("Match Percentage")) for row in members]
        names = []
        seen = set()
        for row in members:
            name = text(row.get("Name"))
            if name not in seen:
                names.append(name)
                seen.add(name)
            if len(names) >= 10:
                break
        avg_pct = round(sum(pcts) / len(pcts), 1) if pcts else 0.0
        min_pct = min(pcts) if pcts else 0.0
        verdict = "safe"
        if len(members) >= 25 or avg_pct < 80:
            verdict = "review-needed"
        if len(members) >= 50 and avg_pct <= 80:
            verdict = "suspicious"
        out.append(
            {
                "cluster": cluster,
                "size": len(members),
                "avg_match_percentage": avg_pct,
                "lowest_match_percentage": min_pct,
                "top_names": names,
                "verdict": verdict,
            }
        )
    return sorted(out, key=lambda item: (-item["size"], item["cluster"]))[:limit]


def generic_low_score_scan(rows: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    suspicious = []
    for cluster, members in grouped_output(rows).items():
        if len(members) < 2:
            continue
        pcts = [pct_to_float(row.get("Match Percentage")) for row in members]
        avg_pct = sum(pcts) / len(pcts) if pcts else 0.0
        min_pct = min(pcts) if pcts else 0.0
        if min_pct > 80:
            continue
        token_counts: Counter[str] = Counter()
        for row in members:
            token_counts.update(tokens(row.get("Name")))
        shared = {tok for tok, count in token_counts.items() if count >= 2}
        generic_shared = sorted(shared & GENERIC_TOKENS)
        distinctive_shared = sorted(shared - GENERIC_TOKENS)
        if generic_shared and not distinctive_shared:
            suspicious.append(
                {
                    "cluster": cluster,
                    "size": len(members),
                    "avg_match_percentage": round(avg_pct, 1),
                    "lowest_match_percentage": min_pct,
                    "generic_shared_tokens": ", ".join(generic_shared[:12]),
                    "top_names": " | ".join(text(row.get("Name")) for row in members[:8]),
                }
            )
    return sorted(suspicious, key=lambda item: (-item["size"], item["lowest_match_percentage"]))[:limit]


def verify_anchor_order(raw_rows: list[dict[str, Any]], out_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [text(row.get("ExternalId")) for row in raw_rows]
    unique_external_ids = len(ids) == len(set(ids))
    if not unique_external_ids:
        return {
            "unique_external_ids": False,
            "anchor_ordering_ok": None,
            "cluster_rows_consecutive": None,
            "noncluster_relative_order_ok": None,
        }
    original_index = {text(row.get("ExternalId")): i for i, row in enumerate(raw_rows)}
    out_indices = [original_index[text(row.get("ExternalId"))] for row in out_rows]
    grouped_positions: dict[int, list[int]] = defaultdict(list)
    grouped_originals: dict[int, list[int]] = defaultdict(list)
    noncluster_originals = []
    for pos, row in enumerate(out_rows):
        cluster = row.get("Cluster Number")
        original = out_indices[pos]
        if cluster is None:
            noncluster_originals.append(original)
        else:
            grouped_positions[int(cluster)].append(pos)
            grouped_originals[int(cluster)].append(original)
    consecutive_failures = []
    anchor_failures = []
    for cluster, positions in grouped_positions.items():
        expected = list(range(min(positions), max(positions) + 1))
        if positions != expected:
            consecutive_failures.append(cluster)
        if grouped_originals[cluster][0] != min(grouped_originals[cluster]):
            anchor_failures.append(cluster)
    noncluster_ok = noncluster_originals == sorted(noncluster_originals)
    return {
        "unique_external_ids": True,
        "anchor_ordering_ok": not consecutive_failures and not anchor_failures and noncluster_ok,
        "cluster_rows_consecutive": not consecutive_failures,
        "anchor_failures": anchor_failures[:20],
        "consecutive_failures": consecutive_failures[:20],
        "noncluster_relative_order_ok": noncluster_ok,
    }


def write_markdown(
    contract: dict[str, Any],
    cases: list[dict[str, Any]],
    top25: list[dict[str, Any]],
    suspicious: list[dict[str, Any]],
) -> None:
    lines = ["# Precision Pass 2 Verification", ""]
    for key, value in contract.items():
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    lines.append("")
    lines.append("## Case Checks")
    for case in cases:
        lines.append(f"### {case['name']}: {'PASS' if case['fixed'] else 'FAIL'}")
        for key, value in case["details"].items():
            lines.append(f"- {key}: {value}")
        for label, summary in case["groups"].items():
            lines.append(f"- {label}: {summary}")
        lines.append("")
    lines.append("## Top 25 Largest Clusters")
    for item in top25:
        lines.append(
            f"### Cluster {item['cluster']} | size {item['size']} | "
            f"avg {item['avg_match_percentage']}% | low {item['lowest_match_percentage']}% | "
            f"{item['verdict']}"
        )
        for name in item["top_names"]:
            lines.append(f"- {name}")
        lines.append("")
    lines.append("## Suspicious Generic Low-Score Scan")
    if not suspicious:
        lines.append("No low-score clusters found where only shared name tokens were generic/non-bridge tokens.")
    for item in suspicious[:25]:
        lines.append(
            f"- Cluster {item['cluster']} | size {item['size']} | "
            f"low {item['lowest_match_percentage']}% | generic shared: {item['generic_shared_tokens']} | "
            f"{item['top_names']}"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")

    top_lines = ["# Top 25 Largest Cluster Review", ""]
    for item in top25:
        top_lines.append(
            f"## Cluster {item['cluster']} | size {item['size']} | "
            f"avg {item['avg_match_percentage']}% | low {item['lowest_match_percentage']}% | "
            f"{item['verdict']}"
        )
        for name in item["top_names"]:
            top_lines.append(f"- {name}")
        top_lines.append("")
    TOP25_MD.write_text("\n".join(top_lines) + "\n")

    scan_lines = ["# Suspicious Generic Low-Score Scan", ""]
    if not suspicious:
        scan_lines.append("No low-score clusters found where only shared name tokens were generic/non-bridge tokens.")
    for item in suspicious:
        scan_lines.append(
            f"- Cluster {item['cluster']} | size {item['size']} | "
            f"avg {item['avg_match_percentage']}% | low {item['lowest_match_percentage']}% | "
            f"generic shared: {item['generic_shared_tokens']} | {item['top_names']}"
        )
    GENERIC_SCAN_MD.write_text("\n".join(scan_lines) + "\n")


def main() -> None:
    raw = pl.read_csv(INPUT, infer_schema_length=10000).to_dicts()
    out_df = pl.read_csv(OUTPUT, infer_schema_length=10000)
    out = out_df.to_dicts()
    review_rows = pl.read_csv(REVIEW_CANDIDATES, infer_schema_length=10000).to_dicts() if REVIEW_CANDIDATES.exists() else []
    metrics = json.loads(METRICS.read_text())
    stats = metrics.get("stats", metrics)

    raw_df = pl.read_csv(INPUT, n_rows=1, infer_schema_length=10000)
    expected_columns = raw_df.columns + ["Cluster Number", "Match Percentage"]

    anchor = verify_anchor_order(raw, out)
    contract = {
        "rows_preserved": len(raw) == len(out),
        "external_id_multiset_preserved": Counter(text(row.get("ExternalId")) for row in raw)
        == Counter(text(row.get("ExternalId")) for row in out),
        "output_columns_correct": out_df.columns == expected_columns,
        "anchor_ordering_ok": anchor["anchor_ordering_ok"],
        "cluster_rows_consecutive": anchor["cluster_rows_consecutive"],
        "noncluster_relative_order_ok": anchor["noncluster_relative_order_ok"],
    }

    cases = [
        should_not_overlap(
            "Sigma broad-token / reused-tax false positives",
            out,
            {
                "ASL Cargo": "ASL Cargo",
                "Integrated DNA Technologies Germany": "Integrated DNA Technologies Germany",
                "SIGMA ARK": "SIGMA ARK",
                "Sigma-Aldrich": "Sigma Aldrich",
            },
        ),
        should_not_overlap("Association GEMINI / CAP GEMINI", out, {"Association GEMINI": "Association GEMINI", "CAP GEMINI": "CAP GEMINI"}),
        should_not_overlap("Association Jasmin / Jasmin Adler", out, {"Association Jasmin": "Association Jasmin de Riche Lieu", "Jasmin Adler": "Jasmin Adler"}),
        should_not_overlap("Association Community / BCN Brand Community", out, {"Association of Community": "ASSOCIATION OF COMMUNITY", "BCN Brand Community": "BCN Brand Community"}),
        should_not_overlap("ATLAS PORTAGE / Tele Atlas", out, {"ATLAS PORTAGE": "ATLAS PORTAGE", "Tele Atlas": "Tele Atlas Navigation"}),
        should_not_overlap("ATMI Packaging / AVI Packaging", out, {"ATMI Packaging": "ATMI Packaging", "AVI Packaging": "AVI Packaging"}),
        should_not_overlap("ATS Automation / TAP Automation Partnership", out, {"ATS Automation": "ATS Automation Tooling Systems", "TAP Automation": "TAP The Automation Partnership"}),
        should_not_overlap("AVG Trucks / DAF Trucks", out, {"AVG Trucks": "AVG Trucks", "DAF Trucks": "DAF Trucks Frankfurt"}),
        should_not_overlap("Avin Electronics / EKC Advanced Electronics", out, {"Avin Electronics": "Avin Electronics", "EKC Advanced Electronics": "EKC Advanced Electronics"}),
        should_not_overlap("AVIS Autoverhuur / Merck BV", out, {"AVIS Autoverhuur": "AVIS Autoverhuur", "Merck BV": "Merck B.V."}),
        should_not_overlap("Hangzhou Fluoro / Hangzhou Tigermed", out, {"Hangzhou Fluoro": "HANGZHOU FLUORO PHARMACEUTICAL", "Hangzhou Tigermed": "HANGZHOU TIGERMED CONSULTING"}),
        should_not_overlap("EP-EXPRESS / Service Express", out, {"EP-EXPRESS": "EP-EXPRESS", "Service Express": "Service Express, LLC"}),
        should_not_overlap("202 Production / FIT Production", out, {"202 Production": "202 Production", "FIT Production": "FIT Production"}),
        should_not_overlap("30 Point Strategies / Chemical Point", out, {"30 Point Strategies": "30 Point Strategies", "Chemical Point": "CHEMICAL POINT LTD"}),
        should_not_overlap("3B Scientific / CJSC Scientific Center", out, {"3B Scientific": "3B Scientific Corporation", "CJSC Scientific Center": "CJSC Scientific Center of Drug"}),
        should_not_overlap("3WAY PHARM / Alps Pharm", out, {"3WAY PHARM": "3WAY PHARM", "Alps Pharm": "Alps Pharm"}),
        should_not_overlap("Aaron first-name-only people", out, {"Aaron Lackner": "Aaron Lackner", "Aaron Lawson McLean": "Aaron Lawson McLean", "Aaron Tan": "Aaron Tan"}),
        should_not_overlap("Alexandre first-name-only people", out, {"Alexandre Guiraud": "Alexandre Guiraud", "Alexandre Prat": "Alexandre Prat", "Alexandre Varnek": "Alexandre Varnek"}),
        should_not_overlap("Alice same-city people", out, {"Alice Antonello": "Alice Antonello", "Alice Hoffmann-Ziegler": "Alice Hoffmann-Ziegler", "Alice Lichtenberg": "Alice Lichtenberg"}),
        should_not_overlap("Kehl same-address different people", out, {"Alexander Kehl": "Alexander Kehl", "Waldemar Kehl": "Waldemar Kehl"}),
        should_not_overlap("Knoll same-address different people", out, {"Aline Knoll": "Aline Knoll", "Kevin Knoll": "Kevin Knoll"}),
        should_not_overlap("Daechert same-address different people", out, {"Andrea Daechert": "Andrea Daechert", "Juergen Daechert": "Juergen Daechert"}),
        should_cluster("Sigma-Aldrich Chemie variants", out, {"Sigma Aldrich Chemie": "Sigma Aldrich Chemie", "SIGMA-ALDRICH Chemie": "SIGMA-ALDRICH Chemie"}),
        should_cluster("Sigma-Aldrich Produktions family", out, {"Sigma-Aldrich Chemie": "Sigma-Aldrich Chemie", "Sigma-Aldrich Produktions": "Sigma-Aldrich Produktions"}),
        should_cluster("ATLAS Material Testing duplicate rows", out, {"ATLAS Material A": "ATLAS Material Testing Technology", "ATLAS Material B": "ATLAS Material Testing Technology"}),
        should_cluster("Autohaus Weeber address range", out, {"Autohaus Weeber GmbH": "Autohaus Weeber GmbH", "Autohaus Weeber GmbH Co KG": "Autohaus Weeber GmbH & Co.KG"}),
        should_cluster("Hangzhou Fluoro / Fluoropharm", out, {"Hangzhou Fluoro": "HANGZHOU FLUORO PHARMACEUTICAL", "Fluoropharm": "Fluoropharm"}),
        should_cluster("Service Express / Top Gun Technology", out, {"Service Express": "Service Express, LLC", "TOP GUN TECHNOLOGY": "TOP GUN TECHNOLOGY"}),
        should_cluster(
            "Wilhelm Schmidt professional title/person-company bridge",
            out,
            {
                "Wilhelm Schmidt": "Wilhelm Schmidt",
                "Dipl.-Ing. Wilhelm Schmidt": "Dipl.-Ing. Wilhelm Schmidt",
                "Dipl.-Ing. Wilhelm Schmidt GmbH": "Dipl.-Ing. Wilhelm Schmidt GmbH",
                "Wilhelm Schmidt Dipl.-Ing.": "Wilhelm Schmidt Dipl.-Ing.",
            },
        ),
        should_have_review_candidates(
            "WEKA / TURNUS family review candidates",
            review_rows,
            [
                ("TURNUS GMBH", "WEKA MEDIA"),
                ("Weka Business Medien", "WEKA MEDIA"),
                ("Weka Business Medien", "WEKA MEDIA PUBLISHING"),
                ("WEKA MEDIA PUBLISHING", "WEKA VERLAG"),
            ],
        ),
    ]

    top25 = top_clusters(out)
    suspicious = generic_low_score_scan(out)

    contract["all_should_not_cluster_cases_fixed"] = all(
        case["fixed"] for case in cases if case["type"] == "should_not_cluster"
    )
    contract["all_should_cluster_cases_caught"] = all(
        case["fixed"] for case in cases if case["type"] == "should_cluster"
    )
    contract["all_review_candidate_cases_caught"] = all(
        case["fixed"] for case in cases if case["type"] == "review_candidates_expected"
    )
    contract["review_candidate_pairs"] = stats.get("review_candidate_pairs")
    contract["review_candidate_rows"] = stats.get("review_candidate_rows")
    contract["runtime_seconds"] = stats.get("processing_time_seconds")
    contract["candidate_pairs"] = stats.get("candidate_pairs")
    contract["match_edges"] = stats.get("match_edges_created")
    contract["clusters"] = stats.get("clusters_found")
    contract["largest_cluster"] = stats.get("largest_cluster_size")
    contract["candidate_cap_hit"] = stats.get("candidate_pairs_capped")

    write_markdown(contract, cases, top25, suspicious)
    GENERIC_SCAN_CSV.write_text(
        pl.DataFrame(suspicious).write_csv() if suspicious else "cluster,size,avg_match_percentage,lowest_match_percentage,generic_shared_tokens,top_names\n"
    )
    REPORT_JSON.write_text(
        json.dumps(
            {
                "contract": contract,
                "anchor": anchor,
                "cases": cases,
                "top25": top25,
                "suspicious_generic_low_score": suspicious,
            },
            indent=2,
        )
    )
    print(json.dumps(contract, indent=2))
    failed = [case["name"] for case in cases if not case["fixed"]]
    if failed:
        print("FAILED_CASES=" + json.dumps(failed))
    else:
        print("FAILED_CASES=[]")
    print(f"REPORT={REPORT_MD}")
    print(f"TOP25={TOP25_MD}")
    print(f"GENERIC_SCAN={GENERIC_SCAN_MD}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate multilingual generic/non-bridge keywords."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.brand_families import compact_alias, load_known_brand_families
from src.config import ClusteringConfig
from src.generic_keywords import load_generic_non_bridge_keywords
from src.legal_keywords import compact_legal_keyword, load_legal_keywords


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generic/non-bridge keyword dictionary")
    parser.add_argument("--input", help="Generic keyword CSV path. Defaults to GENERIC_NON_BRIDGE_FILE")
    parser.add_argument("--brand-file", help="Known brand family file for overlap warnings")
    parser.add_argument("--legal-file", help="Legal keyword file for overlap warnings")
    parser.add_argument("--output-md", default="output/generic_non_bridge_validation.md")
    parser.add_argument("--output-json", default="output/generic_non_bridge_validation.json")
    args = parser.parse_args()

    config = ClusteringConfig.from_env()
    path = _project_path(args.input or config.generic_non_bridge_file)
    brand_path = _project_path(args.brand_file or config.known_brand_families_file)
    legal_path = _project_path(args.legal_file or config.legal_keywords_file)

    index = load_generic_non_bridge_keywords(path, include_defaults=True)
    report = dict(index.report)
    report["overlap_with_brand_aliases"] = _brand_overlap(index.keywords, brand_path)
    report["overlap_with_legal_suffixes"] = _legal_overlap(index.keywords, legal_path)
    warnings = list(report.get("warnings", []))
    for word in report["overlap_with_brand_aliases"][:100]:
        warnings.append({"keyword": word, "warning": "appears in brand alias dictionary; only full phrase alias should win"})
    for word in report["overlap_with_legal_suffixes"][:100]:
        warnings.append({"keyword": word, "warning": "appears in legal suffix dictionary; legal forms remain normalization-only"})
    report["warnings"] = warnings

    out_md = Path(args.output_md)
    out_json = Path(args.output_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(report), encoding="utf-8")

    print(json.dumps({
        "file": report.get("file"),
        "rows_loaded": report.get("rows_loaded"),
        "total_generic_words_loaded": report.get("total_generic_words_loaded"),
        "duplicates_skipped": report.get("duplicates_skipped"),
        "risky_short_words": len(report.get("risky_short_words", [])),
        "overlap_with_brand_aliases": len(report.get("overlap_with_brand_aliases", [])),
        "overlap_with_legal_suffixes": len(report.get("overlap_with_legal_suffixes", [])),
        "output_md": str(out_md),
        "output_json": str(out_json),
    }, indent=2))


def _project_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(__file__).resolve().parents[1] / p)


def _brand_overlap(generic_words: set[str], brand_path: str) -> list[str]:
    index = load_known_brand_families(brand_path)
    brand_compacts = {compact_alias(alias) for family in index.families.values() for alias in family.normalized_aliases}
    return sorted(word for word in generic_words if compact_alias(word) in brand_compacts)


def _legal_overlap(generic_words: set[str], legal_path: str) -> list[str]:
    index = load_legal_keywords(legal_path, include_defaults=True)
    legal_compacts = {compact_legal_keyword(word) for word in index.normalized_suffixes}
    return sorted(word for word in generic_words if compact_legal_keyword(word) in legal_compacts)


def _markdown(report: dict) -> str:
    lines = [
        "# Generic Non-Bridge Validation",
        "",
        f"- file: {report.get('file', '')}",
        f"- rows loaded: {report.get('rows_loaded', 0):,}",
        f"- total generic words loaded: {report.get('total_generic_words_loaded', 0):,}",
        f"- duplicates skipped: {report.get('duplicates_skipped', 0):,}",
        f"- blank rows skipped: {report.get('blank_rows_skipped', 0):,}",
        f"- risky short words: {len(report.get('risky_short_words', [])):,}",
        f"- overlap with brand aliases: {len(report.get('overlap_with_brand_aliases', [])):,}",
        f"- overlap with legal suffixes: {len(report.get('overlap_with_legal_suffixes', [])):,}",
        "",
        "## Examples By Category",
    ]
    for category, words in sorted((report.get("examples_by_category") or {}).items()):
        lines.append(f"- {category}: {', '.join(words[:20])}")
    lines.append("")
    lines.append("## Risky Short Words")
    for word in report.get("risky_short_words", [])[:50]:
        lines.append(f"- {word}")
    lines.append("")
    lines.append("## Overlap With Brand Aliases")
    for word in report.get("overlap_with_brand_aliases", [])[:50]:
        lines.append(f"- {word}")
    lines.append("")
    lines.append("## Overlap With Legal Suffixes")
    for word in report.get("overlap_with_legal_suffixes", [])[:50]:
        lines.append(f"- {word}")
    lines.append("")
    lines.append("## Warnings")
    for item in report.get("warnings", [])[:100]:
        lines.append(f"- {item.get('keyword')}: {item.get('warning')}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

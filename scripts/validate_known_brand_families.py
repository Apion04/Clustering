#!/usr/bin/env python3
"""Validate the curated known brand/family alias file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.brand_families import load_known_brand_families
from src.config import ClusteringConfig
from src.generic_keywords import apply_generic_non_bridge_keywords, load_generic_non_bridge_keywords


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate known brand/family aliases")
    parser.add_argument("--input", help="Keyword CSV path. Defaults to KNOWN_BRAND_FAMILIES_FILE or data/known_brand_families.csv")
    parser.add_argument("--output-md", default="output/known_brand_families_validation.md")
    parser.add_argument("--output-json", default="output/known_brand_families_validation.json")
    args = parser.parse_args()

    config = ClusteringConfig.from_env()
    generic_path = config.generic_non_bridge_file
    if not Path(generic_path).is_absolute():
        generic_path = str(Path(__file__).resolve().parents[1] / generic_path)
    generic_index = load_generic_non_bridge_keywords(generic_path, include_defaults=True)
    apply_generic_non_bridge_keywords(generic_index)

    path = args.input or config.known_brand_families_file
    if not Path(path).is_absolute():
        path = str(Path(__file__).resolve().parents[1] / path)

    index = load_known_brand_families(path, default_confidence=config.known_brand_family_default_confidence)
    report = index.report

    out_md = Path(args.output_md)
    out_json = Path(args.output_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(report), encoding="utf-8")

    print(json.dumps({
        "file": report.get("file"),
        "rows_loaded": report.get("rows_loaded"),
        "alias_phrases_loaded": report.get("alias_phrases_loaded"),
        "aliases_accepted_safe": report.get("aliases_accepted_safe"),
        "aliases_marked_risky": report.get("aliases_marked_risky"),
        "aliases_skipped_generic": report.get("aliases_skipped_generic"),
        "aliases_skipped_short": report.get("aliases_skipped_short"),
        "output_md": str(out_md),
        "output_json": str(out_json),
    }, indent=2))


def _markdown(report: dict) -> str:
    lines = [
        "# Known Brand Families Validation",
        "",
        f"- file: {report.get('file', '')}",
        f"- total keyword rows loaded: {report.get('rows_loaded', 0):,}",
        f"- families loaded: {report.get('families_loaded', 0):,}",
        f"- total alias phrases loaded: {report.get('alias_phrases_loaded', 0):,}",
        f"- aliases accepted as safe: {report.get('aliases_accepted_safe', 0):,}",
        f"- aliases marked risky: {report.get('aliases_marked_risky', 0):,}",
        f"- aliases skipped as generic: {report.get('aliases_skipped_generic', 0):,}",
        f"- very short aliases skipped: {report.get('aliases_skipped_short', 0):,}",
        "",
        "## Safe Examples",
    ]
    for item in report.get("examples_safe", [])[:10]:
        lines.append(f"- {item.get('canonical')}: {', '.join(item.get('aliases', []))}")
    lines.append("")
    lines.append("## Risky Examples")
    for item in report.get("examples_risky", [])[:10]:
        lines.append(f"- {item.get('canonical')}: {', '.join(item.get('aliases', []))}")
    lines.append("")
    lines.append("## Skipped Generic Examples")
    for item in report.get("examples_skipped_generic", [])[:10]:
        lines.append(f"- {item.get('canonical')}: {', '.join(item.get('aliases', []))}")
    lines.append("")
    lines.append("## Warnings")
    for item in report.get("warnings", [])[:50]:
        lines.append(f"- {item.get('normalized')}: {item.get('warning')}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

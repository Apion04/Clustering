#!/usr/bin/env python3
"""Validate the legal suffix/company-form keyword file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import ClusteringConfig
from src.legal_keywords import load_legal_keywords


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate legal suffix/company-form keywords")
    parser.add_argument("--input", help="Legal keyword CSV path. Defaults to LEGAL_KEYWORDS_FILE or data/legal_keywords.csv")
    parser.add_argument("--output-md", default="output/legal_keywords_validation.md")
    parser.add_argument("--output-json", default="output/legal_keywords_validation.json")
    args = parser.parse_args()

    config = ClusteringConfig.from_env()
    path = args.input or config.legal_keywords_file
    if not Path(path).is_absolute():
        path = str(Path(__file__).resolve().parents[1] / path)

    index = load_legal_keywords(path, include_defaults=True)
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
        "legal_keywords_loaded": report.get("legal_keywords_loaded"),
        "duplicates_skipped": report.get("duplicates_skipped"),
        "too_short_skipped": report.get("too_short_skipped"),
        "boundary_only_suffixes": len(report.get("boundary_only_suffixes", [])),
        "risky_very_short_suffixes": len(report.get("risky_very_short_suffixes", [])),
        "output_md": str(out_md),
        "output_json": str(out_json),
    }, indent=2))


def _markdown(report: dict) -> str:
    lines = [
        "# Legal Keywords Validation",
        "",
        f"- file: {report.get('file', '')}",
        f"- legal keyword rows loaded: {report.get('rows_loaded', 0):,}",
        f"- normalized legal keywords loaded: {report.get('legal_keywords_loaded', 0):,}",
        f"- blank rows skipped: {report.get('blank_rows_skipped', 0):,}",
        f"- duplicates skipped: {report.get('duplicates_skipped', 0):,}",
        f"- too-short variants skipped: {report.get('too_short_skipped', 0):,}",
        f"- risky very-short suffixes: {len(report.get('risky_very_short_suffixes', [])):,}",
        f"- boundary-only suffix warnings: {len(report.get('boundary_only_suffixes', [])):,}",
        f"- built-in defaults included: {report.get('built_in_defaults_included', False)}",
        "",
        "## Examples Loaded",
    ]
    for item in report.get("examples_loaded", [])[:25]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Risky Very-Short Suffixes")
    for item in report.get("risky_very_short_suffixes", [])[:50]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Boundary-Only Warnings")
    for item in report.get("warnings", [])[:50]:
        lines.append(f"- {item.get('suffix')}: {item.get('warning')}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

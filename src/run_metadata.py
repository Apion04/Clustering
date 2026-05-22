"""Run metadata: capture app version, column mapping, and config for debug/audit.

Writes a lightweight run_metadata.json alongside the main outputs so that
support engineers can confirm exactly which code and mapping a run used —
without touching any scoring, matching, or merging logic.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Increment when a new phase ships.
_APP_VERSION = "5.1.0"

# Module-level cache so we only shell out to git once per process.
_cached_commit: Optional[str] = None


def get_app_commit_hash() -> str:
    """Return the current git commit hash (7 chars), or '' if unavailable."""
    global _cached_commit
    if _cached_commit is not None:
        return _cached_commit

    # 1. Try git CLI (works in Streamlit Cloud and local envs with git installed).
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=repo_root,
        )
        if res.returncode == 0:
            _cached_commit = res.stdout.strip()
            return _cached_commit
    except Exception:
        pass

    # 2. Fall back: read .git/HEAD directly (works when git CLI is absent but
    #    the repo is a normal clone, e.g. some Docker images).
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        head_file = os.path.join(repo_root, ".git", "HEAD")
        if os.path.isfile(head_file):
            with open(head_file) as fh:
                head = fh.read().strip()
            if head.startswith("ref: "):
                ref_path = os.path.join(repo_root, ".git", head[5:])
                if os.path.isfile(ref_path):
                    with open(ref_path) as fh:
                        _cached_commit = fh.read().strip()[:7]
                        return _cached_commit
            else:
                _cached_commit = head[:7]
                return _cached_commit
    except Exception:
        pass

    _cached_commit = ""
    return ""


def get_app_version() -> str:
    """Return the git commit hash if available, else a fallback version string."""
    h = get_app_commit_hash()
    return h if h else f"v{_APP_VERSION}"


def build_run_metadata(
    *,
    mapping: Dict[str, str],
    input_row_count: int,
    config: Any,
    run_timestamp: Optional[str] = None,
    supplier_name_id_prefix_pct: float = 0.0,
) -> Dict[str, Any]:
    """Build a metadata dict capturing version, mapping, and config for this run.

    Parameters
    ----------
    mapping:
        Column mapping dict produced by auto_detect_columns or --mapping JSON.
    input_row_count:
        Number of rows in the input file (after any --max-rows cap).
    config:
        ClusteringConfig instance used for this run.
    run_timestamp:
        ISO-8601 UTC timestamp; defaults to now().
    """
    commit = get_app_commit_hash()
    version = commit if commit else f"v{_APP_VERSION}"
    ts = run_timestamp or datetime.now(timezone.utc).isoformat()

    ignore_domains = getattr(config, "ignore_client_domains", None) or set()
    # tax_ids may be a list when multiple columns are detected
    tax_id_str = mapping.get("tax_id", "")
    if not tax_id_str and mapping.get("tax_ids"):
        tax_id_str = ", ".join(str(c) for c in mapping["tax_ids"])

    return {
        "app_commit_hash": commit,
        "app_version": version,
        "run_timestamp": ts,
        "input_row_count": input_row_count,
        "mapped_supplier_name": mapping.get("supplier_name", ""),
        "mapped_address": mapping.get("address", ""),
        "mapped_city": mapping.get("city", ""),
        "mapped_country": mapping.get("country", ""),
        "mapped_postal_code": mapping.get("postal_code", ""),
        "mapped_email": mapping.get("email", ""),
        "mapped_tax_id": tax_id_str,
        "mapped_website": mapping.get("website", ""),
        "ignore_client_domains": sorted(ignore_domains),
        "show_70_candidates": bool(
            getattr(config, "allow_unresolved_llm_candidates_in_final_output", False)
        ),
        "llm_mode": str(getattr(config, "llm_execution_mode", "disabled") or "disabled"),
        "review_output_enabled": True,
        "audit_output_enabled": True,
        "supplier_name_id_prefix_warning": supplier_name_id_prefix_pct > 0.10,
        "supplier_name_id_prefix_pct": round(supplier_name_id_prefix_pct, 3),
    }


def save_run_metadata(metadata: Dict[str, Any], output_path: str) -> str:
    """Write *metadata* as indented JSON to *output_path*. Returns the path."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return output_path

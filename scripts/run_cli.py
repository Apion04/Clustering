#!/usr/bin/env python3
"""CLI script to run clustering from command line."""

import argparse
import json
import sys
import os
import time
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import polars as pl
from src.main import cluster_suppliers
from src.config import ClusteringConfig, TAX_COLUMN_PATTERNS, DEFAULT_JSON_TAX_KEYS, DEFAULT_JSON_SECONDARY_NAME_KEYS
from src.input_reader import read_supplier_file
from src.llm_pipeline import run_llm_backend_flow
from src.output import (
    generate_processing_report,
    save_risky_accepted_cluster_review,
    save_ai_review_clusters,
    save_audit_file,
    save_main_output,
    save_review_candidates,
    save_top_suspicious_clusters_report,
    save_unresolved_llm_exception_report,
)
from src.run_metadata import build_run_metadata, save_run_metadata


def main():
    parser = argparse.ArgumentParser(description="Supplier Clustering Engine CLI")
    parser.add_argument("--input", "-i", required=True, help="Input CSV/XLSX file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV/XLSX file path")
    parser.add_argument("--mapping", "-m", help="JSON file with column mapping")
    parser.add_argument("--audit", "-a", help="Optional audit output path")
    parser.add_argument("--review-output", help="Optional review/family candidate output path")
    parser.add_argument("--ai-review-clusters-output", help="Optional accepted-cluster AI/human review export path")
    parser.add_argument("--risky-accepted-output", help="Optional risky accepted cluster review CSV path")
    parser.add_argument("--top-suspicious-output", help="Optional top suspicious cluster Markdown report path")
    parser.add_argument("--llm-exception-output", help="Optional unresolved LLM candidate exception CSV path")
    parser.add_argument("--threshold", "-t", type=float, default=0.90, help="Auto-cluster threshold")
    parser.add_argument("--review", "-r", type=float, default=0.50, help="Review threshold")
    parser.add_argument("--report", help="Path for text report")
    parser.add_argument("--metrics", help="Optional JSON metrics output path")
    parser.add_argument("--max-rows", type=int, help="Optional row limit for staged performance tests")
    parser.add_argument("--max-total-candidate-pairs", type=int, default=1000000, help="Hard cap for total candidate pairs")
    parser.add_argument("--llm", choices=["disabled", "mock", "live", "sync", "batch"], default=None, help="Backend LLM stage mode")
    parser.add_argument("--openai-model", help="OpenAI model override, e.g. gpt-5.5 or gpt-5.4")
    parser.add_argument("--max-llm-groups-per-job", type=int, help="Optional hard cap for groups sent to LLM")
    parser.add_argument("--max-rows-per-llm-group", type=int, help="Optional max rows per LLM group")
    parser.add_argument("--max-tokens-per-llm-group", type=int, help="Optional max estimated tokens per LLM group")
    parser.add_argument("--max-total-llm-cost-per-job", type=float, help="Optional live LLM cost cap in USD")
    parser.add_argument("--llm-timeout-seconds", type=int, help="OpenAI request timeout")
    parser.add_argument("--llm-retry-count", type=int, help="OpenAI retry count")
    parser.add_argument("--allow-unresolved-llm-candidates-in-final-output", action="store_true", help="Allow unresolved 70-score candidates to remain in final output")
    parser.add_argument("--override-llm-can-modify-98", action="store_true", help="Allow LLM decisions to modify deterministic 98 clusters")
    parser.add_argument(
        "--ignore-client-domains",
        default="",
        help="Semicolon-separated list of client/internal domains to exclude from clustering evidence, e.g. merck.com;gilead.com;pfizer.com",
    )
    parser.add_argument(
        "--review-pairs-output",
        default="",
        help="Path for review_pairs.csv (70%% candidates and review-level pairs)",
    )
    parser.add_argument(
        "--cluster-audit-output",
        default="",
        help="Path for cluster_audit.csv (cluster risk and classification audit)",
    )
    parser.add_argument(
        "--run-metadata-output",
        default="",
        help="Path for run_metadata.json (version, mapping, config snapshot for debug/audit). "
             "Defaults to run_metadata.json in the output directory.",
    )

    args = parser.parse_args()
    output_paths = _resolve_output_paths(args)

    # Read input
    print(f"Reading {args.input}...", flush=True)
    t0 = time.perf_counter()
    df = read_supplier_file(args.input, args.input, max_rows=args.max_rows)
    file_load_seconds = time.perf_counter() - t0
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns in {file_load_seconds:.2f}s", flush=True)
    if args.max_rows:
        print(f"Applied --max-rows={args.max_rows:,}", flush=True)

    # Column mapping
    if args.mapping:
        with open(args.mapping) as f:
            mapping = json.load(f)
    else:
        # Auto-detect common column names, using data-quality scoring when
        # sample rows are available to pick the cleanest supplier-name column.
        mapping = auto_detect_columns(df.columns, sample_df=df)
        print(f"Auto-detected mapping: {mapping}", flush=True)

    # Warn if the selected supplier-name column contains numeric ERP/vendor-ID
    # prefixes (e.g. "0020276471-B LAB CO.").  These are stripped during
    # normalization but the warning confirms which column was chosen.
    _id_prefix_pct = _check_supplier_name_id_prefixes(df, mapping)
    if _id_prefix_pct > 0.10:
        print(
            f"⚠️  WARNING: {_id_prefix_pct:.0%} of sampled rows in supplier-name "
            f"column {mapping.get('supplier_name')!r} start with a numeric "
            f"ERP/vendor-ID prefix (e.g. '0020276471-'). "
            f"These prefixes are stripped before name normalization. "
            f"If the wrong column was selected, use --mapping to override.",
            flush=True,
        )

    # Config
    config = ClusteringConfig.from_env()
    config.auto_cluster_threshold = args.threshold
    config.review_threshold = args.review
    config.max_total_candidate_pairs = args.max_total_candidate_pairs
    if args.llm:
        config.llm_execution_mode = "live" if args.llm == "sync" else args.llm
        config.llm_enabled = args.llm not in {"disabled"}
    if args.openai_model:
        config.openai_model = args.openai_model
        config.ai_model = args.openai_model
    if args.max_llm_groups_per_job is not None:
        config.max_llm_groups_per_job = args.max_llm_groups_per_job
    if args.max_rows_per_llm_group is not None:
        config.max_rows_per_llm_group = args.max_rows_per_llm_group
    if args.max_tokens_per_llm_group is not None:
        config.max_tokens_per_llm_group = args.max_tokens_per_llm_group
    if args.max_total_llm_cost_per_job is not None:
        config.max_total_llm_cost_per_job = args.max_total_llm_cost_per_job
    if args.llm_timeout_seconds is not None:
        config.llm_timeout_seconds = args.llm_timeout_seconds
    if args.llm_retry_count is not None:
        config.llm_retry_count = args.llm_retry_count
    if args.allow_unresolved_llm_candidates_in_final_output:
        config.allow_unresolved_llm_candidates_in_final_output = True
    if args.override_llm_can_modify_98:
        config.override_llm_can_modify_98 = True
    if args.ignore_client_domains:
        config.ignore_client_domains = frozenset(
            d.strip().lower()
            for d in args.ignore_client_domains.split(";")
            if d.strip()
        )

    # Run clustering
    result = cluster_suppliers(df, mapping, config)
    result["stats"].setdefault("stage_timings", {})["file_load_seconds"] = file_load_seconds

    # Debug smoke check: log B LAB / same-address variant pairing details.
    # Only fires when rows containing "b lab" are found; no-op otherwise.
    _smoke_check_b_lab(result, mapping)

    # Write run_metadata.json — lightweight version/mapping/config snapshot.
    metadata_path = args.run_metadata_output or output_paths["run_metadata"]
    run_meta = build_run_metadata(
        mapping=mapping,
        input_row_count=len(df),
        config=config,
        supplier_name_id_prefix_pct=_id_prefix_pct,
    )
    save_run_metadata(run_meta, metadata_path)
    print(f"✅ Run metadata saved: {metadata_path}", flush=True)
    print(
        f"   commit={run_meta['app_commit_hash'] or '(unavailable)'}  "
        f"supplier_name={run_meta['mapped_supplier_name']!r}  "
        f"address={run_meta['mapped_address']!r}  "
        f"country={run_meta['mapped_country']!r}",
        flush=True,
    )

    # Optional audit/review output files
    if args.review_pairs_output and result.get("review_pairs_df") is not None:
        result["review_pairs_df"].write_csv(args.review_pairs_output)
        print(f"✅ Review pairs saved: {args.review_pairs_output} ({len(result['review_pairs_df'])} pairs)", flush=True)

    if args.cluster_audit_output and result.get("cluster_audit_df") is not None:
        result["cluster_audit_df"].write_csv(args.cluster_audit_output)
        print(f"✅ Cluster audit saved: {args.cluster_audit_output} ({len(result['cluster_audit_df'])} clusters)", flush=True)

    # Save deterministic main output. In directory mode this is an internal
    # deterministic artifact; final_supplier_clustered.csv is written after
    # the backend LLM stage below.
    t0 = time.perf_counter()
    save_main_output(result["main_df"], output_paths["deterministic_output"])
    main_write_seconds = time.perf_counter() - t0
    result["stats"]["stage_timings"]["main_output_write_seconds"] = main_write_seconds
    print(f"\n✅ Deterministic output saved: {output_paths['deterministic_output']} ({main_write_seconds:.2f}s)", flush=True)

    # Auto-save review/family candidates alongside the main output. These are
    # intentionally separate from the simple reviewer file.
    t0 = time.perf_counter()
    review_output_dir = output_paths["output_dir"]
    review_output_path = args.review_output or output_paths["review"]
    review_cands = result.get("review_candidates") or []
    rows_dict = {row["row_id"]: row for row in result["preprocessed_df"].iter_rows(named=True)}
    save_review_candidates(review_cands, rows_dict, review_output_path)
    review_write_seconds = time.perf_counter() - t0
    result["stats"]["stage_timings"]["review_candidates_write_seconds"] = review_write_seconds
    print(f"✅ Review candidates saved: {review_output_path} ({len(review_cands):,} pairs, {review_write_seconds:.2f}s)", flush=True)

    # Optional audit
    audit_output_path = args.audit or output_paths["audit"]
    if audit_output_path:
        t0 = time.perf_counter()
        save_audit_file(result["audit_data"], audit_output_path, result["preprocessed_df"], result.get("output_cluster_map") or result["cluster_map"], result["merger"])
        audit_write_seconds = time.perf_counter() - t0
        result["stats"]["stage_timings"]["audit_output_write_seconds"] = audit_write_seconds
        print(f"✅ Audit file saved: {audit_output_path} ({audit_write_seconds:.2f}s)", flush=True)

    ai_review_clusters_path = args.ai_review_clusters_output or output_paths["ai_review_clusters"]
    if ai_review_clusters_path:
        t0 = time.perf_counter()
        save_ai_review_clusters(
            result["audit_data"],
            ai_review_clusters_path,
            result["preprocessed_df"],
            result.get("output_cluster_map") or result["cluster_map"],
            result["merger"],
        )
        ai_review_write_seconds = time.perf_counter() - t0
        result["stats"]["stage_timings"]["ai_review_clusters_write_seconds"] = ai_review_write_seconds
        print(f"✅ AI/human review clusters saved: {ai_review_clusters_path} ({ai_review_write_seconds:.2f}s)", flush=True)

    risky_accepted_path = args.risky_accepted_output or output_paths["risky_accepted"]
    if risky_accepted_path:
        t0 = time.perf_counter()
        save_risky_accepted_cluster_review(
            risky_accepted_path,
            result["preprocessed_df"],
            result.get("output_cluster_map") or result["cluster_map"],
            result.get("output_match_pcts") or {},
            result["merger"],
        )
        risky_write_seconds = time.perf_counter() - t0
        result["stats"]["stage_timings"]["risky_accepted_cluster_review_write_seconds"] = risky_write_seconds
        print(f"✅ Risky accepted cluster review saved: {risky_accepted_path} ({risky_write_seconds:.2f}s)", flush=True)

    top_suspicious_path = args.top_suspicious_output or output_paths["top_suspicious"]
    if top_suspicious_path:
        t0 = time.perf_counter()
        save_top_suspicious_clusters_report(
            top_suspicious_path,
            result["preprocessed_df"],
            result.get("output_cluster_map") or result["cluster_map"],
            result.get("output_match_pcts") or {},
            result["merger"],
        )
        suspicious_write_seconds = time.perf_counter() - t0
        result["stats"]["stage_timings"]["top_suspicious_clusters_write_seconds"] = suspicious_write_seconds
        print(f"✅ Top suspicious cluster report saved: {top_suspicious_path} ({suspicious_write_seconds:.2f}s)", flush=True)

    if args.llm_exception_output:
        t0 = time.perf_counter()
        save_unresolved_llm_exception_report(result.get("llm_review_groups") or [], args.llm_exception_output)
        exception_write_seconds = time.perf_counter() - t0
        result["stats"]["stage_timings"]["llm_exception_report_write_seconds"] = exception_write_seconds
        print(f"✅ LLM exception report saved: {args.llm_exception_output} ({exception_write_seconds:.2f}s)", flush=True)

    llm_backend = run_llm_backend_flow(
        result,
        config,
        output_paths["final_output"],
        output_paths["output_dir"],
    )
    result["stats"]["llm_backend"] = llm_backend["decision_application_report"]
    print(f"✅ Final supplier output saved: {output_paths['final_output']}", flush=True)
    print(f"✅ LLM backend job status: {llm_backend['job_status']}", flush=True)

    # Optional report
    report_path = args.report or output_paths["report"]
    if report_path:
        t0 = time.perf_counter()
        generate_processing_report(result["stats"], report_path)
        report_write_seconds = time.perf_counter() - t0
        result["stats"]["stage_timings"]["report_write_seconds"] = report_write_seconds
        print(f"✅ Report saved: {report_path} ({report_write_seconds:.2f}s)", flush=True)

    readiness_path = os.path.join(output_paths["output_dir"], "final_production_readiness_report.md")
    _write_final_production_readiness_report(readiness_path, args, result, llm_backend, output_paths)
    print(f"✅ Production readiness report saved: {readiness_path}", flush=True)

    metrics_path = args.metrics or output_paths["metrics"]
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "input": args.input,
                "output": output_paths["final_output"],
                "deterministic_output": output_paths["deterministic_output"],
                "audit": audit_output_path,
                "review_output": review_output_path,
                "ai_review_clusters_output": ai_review_clusters_path,
                "risky_accepted_output": risky_accepted_path,
                "top_suspicious_output": top_suspicious_path,
                "llm_exception_output": args.llm_exception_output,
                "report": report_path,
                "final_production_readiness_report": readiness_path,
                "max_rows": args.max_rows,
                "mapping": mapping,
                "llm_backend": llm_backend["decision_application_report"],
                "llm_paths": llm_backend["paths"],
                "stats": result["stats"],
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Metrics saved: {metrics_path}", flush=True)

    # Print summary
    stats = result["stats"]
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total rows:        {stats['total_rows']:,}")
    print(f"Candidate pairs:   {stats['candidate_pairs']:,}")
    print(f"Candidate cap hit: {stats.get('candidate_pairs_capped', False)}")
    print(f"Match edges:       {stats.get('match_edges_created', 0):,}")
    print(f"Clusters found:    {stats['clusters_found']:,}")
    print(f"Largest cluster:   {stats.get('largest_cluster_size', 0):,}")
    print(f"Skipped blocks:    {stats.get('blocking_diagnostics', {}).get('skipped_oversized_blocks', 0):,}")
    print(f"Auto-clustered:    {stats['auto_clustered_rows']:,}")
    print(f"Review queue:      {stats['review_queue_rows']:,}")
    print(f"Singletons:        {stats['singleton_rows']:,}")
    print(f"Time:              {stats['processing_time_seconds']:.1f}s")
    print(f"{'='*50}")


def _resolve_output_paths(args):
    output = os.path.abspath(args.output)
    ext = os.path.splitext(output)[1].lower()
    if ext in {".csv", ".xlsx"}:
        output_dir = os.path.dirname(output) or os.getcwd()
        deterministic_output = output
    else:
        output_dir = output
        deterministic_output = os.path.join(output_dir, "deterministic_supplier_clustered.csv")
    os.makedirs(output_dir, exist_ok=True)
    return {
        "output_dir": output_dir,
        "deterministic_output": deterministic_output,
        "final_output": os.path.join(output_dir, "final_supplier_clustered.csv"),
        "audit": os.path.join(output_dir, "audit.csv"),
        "review": os.path.join(output_dir, "family_review_candidates.csv"),
        "ai_review_clusters": os.path.join(output_dir, "oro_safe_75k_ai_review_clusters.csv"),
        "risky_accepted": os.path.join(output_dir, "risky_accepted_cluster_review.csv"),
        "top_suspicious": os.path.join(output_dir, "top_suspicious_clusters.md"),
        "report": os.path.join(output_dir, "processing_report.txt"),
        "metrics": os.path.join(output_dir, "metrics.json"),
        "run_metadata": os.path.join(output_dir, "run_metadata.json"),
    }


def _write_final_production_readiness_report(path, args, result, llm_backend, output_paths):
    stats = result.get("stats", {})
    app = llm_backend.get("decision_application_report", {})
    queue = llm_backend.get("queue", {})
    cost = llm_backend.get("cost_estimate", {})
    final_df = llm_backend.get("final_df")
    final_columns = final_df.columns if final_df is not None else []
    allowed_scores = {"98%", "85%", "70%", ""}
    final_score_values = set(str(x or "") for x in final_df["Match Percentage"].to_list()) if final_df is not None and "Match Percentage" in final_df.columns else set()
    clean_score_ok = final_score_values <= allowed_scores
    expected_columns = result["main_df"].columns if "main_df" in result else final_columns
    schema_ok = final_columns == expected_columns
    rows_in = stats.get("total_rows", 0)
    rows_out = len(final_df) if final_df is not None else 0
    lines = [
        "# Final Production Readiness Report",
        "",
        f"- input file: `{args.input}`",
        f"- deterministic output: `{output_paths['deterministic_output']}`",
        f"- final clean output: `{output_paths['final_output']}`",
        f"- job status: `{llm_backend.get('job_status')}`",
        f"- selected OpenAI model: `{app.get('selected_openai_model') or cost.get('selected_openai_model')}`",
        f"- rows in/out: {rows_in:,} / {rows_out:,}",
        f"- deterministic runtime seconds: {float(stats.get('processing_time_seconds') or 0):.2f}",
        f"- candidate pairs: {int(stats.get('candidate_pairs') or 0):,}",
        f"- accepted main edges: {int(stats.get('match_edges_created') or 0):,}",
        f"- deterministic clusters: {int(stats.get('clusters_found') or 0):,}",
        f"- largest deterministic cluster: {int(stats.get('largest_cluster_size') or 0):,}",
        f"- raw LLM/review groups: {int(queue.get('raw_groups_generated') or app.get('raw_groups_generated') or 0):,}",
        f"- deduped LLM groups sent/prepared: {int(queue.get('deduped_groups_sent') or app.get('deduped_groups_sent') or 0):,}",
        f"- excluded LLM groups: {int(len(queue.get('excluded_groups', [])) or app.get('excluded_groups') or 0):,}",
        f"- estimated GPT/OpenAI cost: {_format_cost(cost)}",
        f"- configured cost cap: ${float(cost.get('configured_cost_cap') or 0):.4f}",
        f"- cost estimate known: {bool(cost.get('cost_estimate_known'))}",
        f"- cost note: {cost.get('cost_rates_note', '')}",
        f"- mock/live LLM decisions applied: `{app.get('llm_decision_counts', {})}`",
        f"- unresolved group count: {int(app.get('unresolved_group_count') or 0):,}",
        f"- final score distribution: `{app.get('final_score_distribution', {})}`",
        f"- final output schema valid: {schema_ok}",
        f"- final output allowed scores only: {clean_score_ok}",
        f"- cluster numbers contiguous: {bool(app.get('cluster_numbers_contiguous'))}",
        f"- risky/generic accepted cluster review: `{output_paths['risky_accepted']}`",
        f"- top suspicious cluster report: `{output_paths['top_suspicious']}`",
        "",
        "## Final User-Facing Contract",
        "",
        "The final user-facing file must contain only original input columns plus `Cluster Number` and `Match Percentage`.",
        "`Match Percentage` may contain `98%`, `85%`, `70%`, or blank. "
        "`70%` means plausible but uncertain — LLM/manual review needed. "
        "If no OpenAI key is configured or LLM is disabled, 70% candidates remain visible in the output.",
        "",
        "## Readiness",
        "",
        "Create the final ZIP only after reviewing this report, the LLM decision application report, and the risky accepted cluster review.",
    ]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _smoke_check_b_lab(result: dict, mapping: dict) -> None:
    """Debug: log the full pairing path for B LAB CO. / B LAB COMPANY rows.

    Only fires when 2+ rows whose name_norm contains 'b lab' are present.
    Answers in order:
      1. Are both rows present?
      2. Is the address field non-empty after mapping?
      3. Do addr_norm values match?
      4. Is a LSUF_ blocking key generated for each row?
      5. Did they end up in the same cluster (and at what score)?

    Prints [SMOKE]-prefixed lines to stdout so they appear in the Streamlit
    processing log without affecting any output file.
    """
    from src.legal_keywords import strip_legal_suffixes

    preprocessed_df = result.get("preprocessed_df")
    if preprocessed_df is None or "name_norm" not in preprocessed_df.columns:
        return

    b_lab_rows = preprocessed_df.filter(
        pl.col("name_norm").str.contains("b lab")
    )
    if len(b_lab_rows) < 2:
        return  # No B LAB variant rows — smoke check not applicable

    addr_col_mapped = mapping.get("address", "")
    output_cluster_map = result.get("output_cluster_map") or result.get("cluster_map") or {}
    output_match_pcts = result.get("output_match_pcts") or {}

    print(
        f"[SMOKE] B LAB variant rows found: {len(b_lab_rows)}  "
        f"(address column mapped to: {addr_col_mapped!r})",
        flush=True,
    )

    row_infos: list[dict] = []
    for row in b_lab_rows.iter_rows(named=True):
        nn = row.get("name_norm", "") or ""
        addr = row.get("addr_norm", "") or ""
        rid = row.get("row_id")
        stripped, _ = strip_legal_suffixes(nn)
        has_lsuf = bool(
            stripped and stripped != nn and len(stripped.split()) >= 2
        )
        lsuf_key = f"LSUF_{stripped[:12]}" if has_lsuf else None
        cluster = output_cluster_map.get(rid)
        pct = output_match_pcts.get(cluster) if cluster is not None else None
        print(
            f"[SMOKE]   row_id={rid}  name_norm={nn!r}  addr_norm={addr!r}  "
            f"lsuf_key={lsuf_key!r}  cluster={cluster}  match_pct={pct}",
            flush=True,
        )
        row_infos.append(
            {"nn": nn, "addr": addr, "rid": rid,
             "cluster": cluster, "pct": pct, "lsuf_key": lsuf_key}
        )

    # 2. Address field check
    addrs = [r["addr"] for r in row_infos]
    non_empty_addrs = [a for a in addrs if a]
    if not non_empty_addrs:
        print(
            "[SMOKE] ⚠️  addr_norm is EMPTY for all B LAB rows — "
            "address column not mapped or address data is blank",
            flush=True,
        )
    elif len(set(non_empty_addrs)) == 1 and len(non_empty_addrs) == len(addrs):
        print(f"[SMOKE] ✅ addr_norm is consistent: {non_empty_addrs[0]!r}", flush=True)
    else:
        print(
            f"[SMOKE] ⚠️  addr_norm values differ across B LAB rows: {addrs}",
            flush=True,
        )

    # 3. LSUF_ key check
    lsuf_keys = [r["lsuf_key"] for r in row_infos]
    unique_lsuf = set(k for k in lsuf_keys if k)
    if len(unique_lsuf) == 1:
        print(f"[SMOKE] ✅ LSUF_ key matches across rows: {list(unique_lsuf)[0]!r}", flush=True)
    else:
        print(f"[SMOKE] ⚠️  LSUF_ keys: {lsuf_keys}", flush=True)

    # 4. Cluster outcome
    clusters = [r["cluster"] for r in row_infos if r["cluster"] is not None]
    unique_clusters = set(clusters)
    if clusters and len(unique_clusters) == 1:
        pcts = [r["pct"] for r in row_infos if r["pct"] is not None]
        print(
            f"[SMOKE] ✅ Clustered together: cluster={clusters[0]}  scores={pcts}",
            flush=True,
        )
    else:
        print("[SMOKE] ❌ B LAB rows NOT in the same cluster", flush=True)


# ── Supplier-name column selection helpers ────────────────────────────────
# These are module-level so they can be imported and tested independently.

_SUPPLIER_NAME_KEYWORDS = frozenset({
    "name", "vendor", "supplier", "company", "legal",
    "payee", "organisation", "organization", "partner",
})

# Shared regex for both scoring and the post-select diagnostic check.
_SUPPLIER_ID_PREFIX_RE = re.compile(r'^\d{6,}[\s\-_:]+')
_PURELY_NUMERIC_RE = re.compile(r'^\d+$')

# Column-name compacts that earn a bonus in data-quality scoring.
_CLEAN_NAME_COMPACTS = frozenset({
    "commonname", "suppliername", "vendorname", "name",
    "companyname", "legalname", "payeename",
    "businesspartnername", "organisationname", "organizationname",
    "supplierlegalname", "vendorlegalname",
    "supplierofficialname", "vendorofficialname",
})

# Ordered fallback priority used when no sample data is available.
_EXACT_NAME_PRIORITY = [
    "commonname", "common name", "common_name",
    "suppliername", "supplier name", "supplier_name",
    "vendorname", "vendor name", "vendor_name",
    "legalname", "legal name", "legal_name",
    "companyname", "company name", "company_name",
    "payeename", "payee name", "payee_name",
    "businesspartnername", "business partner name",
    "organisationname", "organisation name", "organisation_name",
    "organizationname", "organization name", "organization_name",
    "name",
]


def _is_id_column(col_lower: str) -> bool:
    """Return True if the column name suggests it is an ID/code column, not a name column.

    Rejects: "Supplier ID", "Vendor ID", "Internal ID", "Entity ID",
             "legalEntityId", "clusterId", "Vendor Code", "Supplier Number", etc.
    """
    if re.search(
        r'\b(?:id|ids|identifier|identifiers|code|codes|number|num|pk|key|ref|reference)\b',
        col_lower,
    ):
        return True
    # CamelCase endings: "legalEntityId", "clusterId", "vendorId"
    compact = re.sub(r'[\s_\-]', '', col_lower)
    if compact.endswith("id") and len(compact) > 4:
        return True
    return False


# Location/geographic column indicators — reject these even when they contain a
# supplier_name keyword (e.g. "Supplier Origin Country", "Supplier City").
_LOCATION_COLUMN_TERMS = re.compile(
    r'\b(?:country|city|region|state|province|territory|postal|zip|postcode|'
    r'origin|destination|location|site|address|street|building|floor|room)\b'
)


def _is_location_column(col_lower: str) -> bool:
    """Return True when the column name signals geographic/address data, not a supplier name."""
    return bool(_LOCATION_COLUMN_TERMS.search(col_lower))


def _find_supplier_name_candidates(col_lower: dict) -> list:
    """Return column names that could be supplier names, excluding ID/code and location columns."""
    result = []
    for lower, orig in col_lower.items():
        if not any(kw in lower for kw in _SUPPLIER_NAME_KEYWORDS):
            continue
        if _is_id_column(lower):
            continue
        if _is_location_column(lower):
            continue
        # Skip secondary/tertiary name columns (Name2, Name 3, etc.)
        if re.search(r"name\s*[23456789]", lower):
            continue
        result.append(orig)
    return result


def _score_name_column(col_name: str, sample_values: list) -> float:
    """Score a column as a supplier-name candidate.  Higher is better.

    Penalties:  numeric ERP/vendor-ID prefix values, purely-numeric values,
                high blank-row fraction, very short average value length
                (< 4 chars → likely country codes or other non-name data).
    Bonuses:    exact clean column names (CommonName, Vendor Name, …).
    """
    col_compact = re.sub(r"[\s_\-]", "", col_name.lower())
    score = 0.5 if col_compact in _CLEAN_NAME_COMPACTS else 0.0

    if not sample_values:
        return score - 2.0

    non_null = [str(v).strip() for v in sample_values if v is not None and str(v).strip()]
    if not non_null:
        return score - 2.0

    n = len(non_null)
    total = len(sample_values)

    # Penalty: blank rows
    score -= (1.0 - n / total) * 1.5

    # Heavy penalty: numeric ERP/vendor-ID prefix (breaks clustering)
    prefix_pct = sum(1 for v in non_null if _SUPPLIER_ID_PREFIX_RE.match(v)) / n
    score -= prefix_pct * 3.0

    # Penalty: purely numeric values (stored IDs, not names)
    numeric_pct = sum(1 for v in non_null if _PURELY_NUMERIC_RE.match(v)) / n
    score -= numeric_pct * 2.0

    # Penalty: very short average value length (< 4 chars → likely country codes,
    # 2-letter codes, or other non-name data that slipped through column-name filtering).
    avg_len = sum(len(v) for v in non_null) / n
    if avg_len < 4.0:
        score -= 2.5

    return score


def _check_supplier_name_id_prefixes(df: pl.DataFrame, mapping: dict) -> float:
    """Return the fraction of non-null supplier-name values that start with a
    numeric ERP/vendor-ID prefix (e.g. '0020276471-').  Samples up to 500 rows
    for speed.  Returns 0.0 if the column is absent or empty.
    """
    col = mapping.get("supplier_name", "")
    if not col or col not in df.columns:
        return 0.0
    sample = df[col].drop_nulls().head(500).to_list()
    if not sample:
        return 0.0
    matches = sum(
        1 for v in sample
        if isinstance(v, str) and _SUPPLIER_ID_PREFIX_RE.match(v.strip())
    )
    return matches / len(sample)


def auto_detect_columns(columns, sample_df=None):
    """Auto-detect common supplier column names.

    When *sample_df* is supplied, data-quality scoring is used to prefer the
    cleanest supplier-name column (penalises numeric ERP/vendor-ID prefixes,
    purely-numeric values, and high blank-row fractions).

    When *sample_df* is not available (e.g. tests that pass column names only),
    falls back to a column-name priority list — backward-compatible behaviour.
    """
    if not columns:
        return {}
    col_lower = {c.lower(): c for c in columns}
    mapping = {"supplier_name": columns[0]}  # absolute fallback

    # ── Supplier name ──────────────────────────────────────────────────────
    candidates = _find_supplier_name_candidates(col_lower)
    if not candidates:
        candidates = [columns[0]]

    if len(candidates) == 1:
        mapping["supplier_name"] = candidates[0]
    elif sample_df is not None:
        # Data-quality path: score every candidate and pick the winner.
        def _get_score(col: str) -> float:
            vals = sample_df[col].head(500).to_list() if col in sample_df.columns else []
            return _score_name_column(col, vals)
        mapping["supplier_name"] = max(candidates, key=_get_score)
    else:
        # Column-name-only fallback: use priority list, restricted to actual candidates.
        found = False
        for exact in _EXACT_NAME_PRIORITY:
            if exact in col_lower and col_lower[exact] in candidates:
                mapping["supplier_name"] = col_lower[exact]
                found = True
                break
        if not found:
            mapping["supplier_name"] = candidates[0]

    # Address
    for lower, orig in col_lower.items():
        if "address" in lower or "street" in lower:
            mapping["address"] = orig
            break

    # City
    for lower, orig in col_lower.items():
        normalized = lower.replace("_", " ").replace("-", " ")
        if lower in ["city", "town"] or normalized.endswith(" city") or "address city" in normalized:
            mapping["city"] = orig
            break

    # Country
    for lower, orig in col_lower.items():
        normalized = lower.replace("_", " ").replace("-", " ")
        compact = normalized.replace(" ", "")
        if lower in ["country", "cntry", "nation"] or "countrycode" in compact or normalized.endswith(" country"):
            mapping["country"] = orig
            break

    # Postal
    for lower, orig in col_lower.items():
        if "postal" in lower or "zip" in lower or "postcode" in lower:
            mapping["postal_code"] = orig
            break

    # Tax/VAT/legal registration IDs: support multiple worldwide identifier columns.
    # This is intentionally broad because client files use many country-specific names.
    tax_cols = []
    for lower, orig in col_lower.items():
        if _looks_like_direct_tax_column(lower):
            tax_cols.append(orig)
    if tax_cols:
        mapping["tax_id"] = tax_cols[0]
        mapping["tax_ids"] = tax_cols
        mapping.setdefault("json_tax_keys", DEFAULT_JSON_TAX_KEYS)

    # JSON metadata columns that may contain vatNumber/taxNumber
    json_cols = []
    for lower, orig in col_lower.items():
        if any(x in lower for x in ["json", "metadata", "additional", "info", "attributes"]):
            json_cols.append(orig)
    if json_cols:
        mapping["metadata_json_columns"] = json_cols
        mapping.setdefault("json_tax_keys", DEFAULT_JSON_TAX_KEYS)
        mapping.setdefault("json_secondary_name_keys", DEFAULT_JSON_SECONDARY_NAME_KEYS)

    # Email
    for lower, orig in col_lower.items():
        if "email" in lower or "e-mail" in lower:
            mapping["email"] = orig
            break

    # Website
    for lower, orig in col_lower.items():
        if "website" in lower or "web" in lower or "url" in lower:
            mapping["website"] = orig
            break

    # Optional support fields. These help review/blocking/sorting and only
    # become main-cluster evidence when configured as trusted same-entity fields.
    support_exact = {
        "orovendorid": "OROVendorId",
        "companyentityid": "CompanyEntityId",
        "familyname": "family_name",
        "family": "family_name",
        "canonicalname": "canonical_name",
        "canonical": "canonical_name",
        "parentname": "parent_name",
        "parent": "parent_name",
        "normalizedsuppliername": "normalized_supplier_name",
        "emaildomain": "email_domain",
        "domain": "domain",
    }
    used_support_cols = set()
    for lower, orig in col_lower.items():
        compact = lower.replace("_", "").replace("-", "").replace(" ", "")
        key = support_exact.get(compact)
        if key and orig not in used_support_cols:
            mapping[key] = orig
            used_support_cols.add(orig)

    # Secondary names
    secondary = []
    for lower, orig in col_lower.items():
        if any(p in lower for p in ["name 2", "name_2", "name2", "name 3", "name_3", "name3", "name 4", "name_4", "name4", "family name", "alternate name", "alt name"]):
            if orig != mapping.get("supplier_name"):
                secondary.append(orig)
    if secondary:
        mapping["secondary_names"] = secondary
        for idx, col in enumerate(secondary[:3], 2):
            mapping[f"name_{idx}"] = col

    return mapping


def _format_cost(cost):
    if not cost.get("cost_estimate_known", False):
        return "UNKNOWN (pricing env vars missing; this does not mean free)"
    return f"${float(cost.get('estimated_cost_usd') or 0):.4f}"


def _looks_like_direct_tax_column(lower_name: str) -> bool:
    """Return true for actual tax ID columns, not arbitrary substring hits.

    Short identifiers such as TIN/NIT/PAN must match whole tokens. This avoids
    false positives like PostingBlocked, CustomExtensionFormJSON, or UnitNumber.
    """
    if any(x in lower_name for x in ["json", "metadata", "additional", "attributes"]):
        return False
    tokens = re.findall(r"[a-z0-9]+", lower_name.lower())
    phrase = " ".join(tokens)
    compact = "".join(tokens)
    token_set = set(tokens)
    for pattern in TAX_COLUMN_PATTERNS:
        p_tokens = re.findall(r"[a-z0-9]+", pattern.lower())
        if not p_tokens:
            continue
        p_phrase = " ".join(p_tokens)
        p_compact = "".join(p_tokens)
        if len(p_tokens) > 1:
            if re.search(rf"\b{re.escape(p_phrase)}\b", phrase):
                return True
            continue
        if p_compact in token_set or compact == p_compact:
            return True
        if len(p_compact) >= 4 and (compact.startswith(p_compact) or compact.endswith(p_compact) or p_compact in compact):
            return True
    return False


if __name__ == "__main__":
    main()

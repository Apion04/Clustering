"""FastAPI routes for supplier clustering service."""

import os
import tempfile
import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

from src.api.models import ColumnMapping, ClusteringRequest, ClusteringResponse, ClusteringStats
from src.main import cluster_suppliers
from src.config import ClusteringConfig
from src.input_reader import read_supplier_file
from src.output import save_main_output, save_audit_file, generate_processing_report, save_review_candidates

app = FastAPI(
    title="Supplier Clustering Engine",
    description="Automated supplier deduplication and family clustering",
    version="1.0.0",
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/cluster-suppliers", response_model=ClusteringResponse)
async def cluster_suppliers_endpoint(
    file: UploadFile = File(..., description="CSV or Excel file with supplier data"),
    column_mapping: str = Form(..., description="JSON string of column mapping"),
    auto_cluster_threshold: Optional[float] = Form(0.90),
    review_threshold: Optional[float] = Form(0.50),
    generate_audit: Optional[bool] = Form(False),
    allow_parent_family_tax_conflicts: Optional[bool] = Form(True),
    ai_review_enabled: Optional[bool] = Form(False),
    ai_uncertain_cluster_enabled: Optional[bool] = Form(True),
    ai_uncertain_match_pct: Optional[float] = Form(68.0),
    max_total_candidate_pairs: Optional[int] = Form(1000000),
):
    """
    Upload a supplier file and receive clustered output.

    The output file contains all original columns plus:
    - Cluster Number: integer cluster ID (null for singletons)
    - Match Percentage: confidence score as percentage string

    Output uses anchor-based ordering: the first occurrence of a cluster stays in its original position and later matching rows move directly below it.
    """
    import json

    # Validate file type
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "File must be CSV or Excel")

    # Parse column mapping
    try:
        mapping_dict = json.loads(column_mapping)
        col_mapping = ColumnMapping(**mapping_dict)
    except Exception as e:
        raise HTTPException(400, f"Invalid column_mapping JSON: {str(e)}")

    # Read file
    try:
        df = read_supplier_file(file.file, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {str(e)}")

    # Validate required columns exist
    required = [col_mapping.supplier_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required columns: {missing}")

    # Build config
    config = ClusteringConfig.from_env()
    config.auto_cluster_threshold = auto_cluster_threshold
    config.review_threshold = review_threshold
    config.allow_parent_family_tax_conflicts = allow_parent_family_tax_conflicts
    config.ai_review_enabled = ai_review_enabled
    config.ai_uncertain_cluster_enabled = ai_uncertain_cluster_enabled
    config.ai_uncertain_match_pct = ai_uncertain_match_pct
    config.max_total_candidate_pairs = max_total_candidate_pairs

    # Run clustering
    try:
        result = cluster_suppliers(df, col_mapping.model_dump(), config)
    except Exception as e:
        raise HTTPException(500, f"Clustering failed: {str(e)}")

    # Generate unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = str(uuid.uuid4())[:8]
    base_name = f"clustered_{timestamp}_{job_id}"

    # Save main output
    main_path = os.path.join(OUTPUT_DIR, f"{base_name}.xlsx")
    save_main_output(result["main_df"], main_path)

    # Optional: save audit file
    audit_path = None
    if generate_audit:
        audit_path = os.path.join(OUTPUT_DIR, f"{base_name}_audit.csv")
        save_audit_file(
            result["audit_data"], 
            audit_path, 
            result["preprocessed_df"],
            result["cluster_map"],
            result["merger"],
        )

    review_path = os.path.join(OUTPUT_DIR, f"{base_name}_review.csv")
    rows_dict = {row["row_id"]: row for row in result["preprocessed_df"].iter_rows(named=True)}
    save_review_candidates(result.get("review_candidates") or [], rows_dict, review_path)

    # Generate report
    report_path = os.path.join(OUTPUT_DIR, f"{base_name}_report.txt")
    generate_processing_report(result["stats"], report_path)

    # Build response
    stats = result["stats"]

    return ClusteringResponse(
        success=True,
        message=f"Clustering complete. {stats['clusters_found']} clusters found.",
        stats=ClusteringStats(**stats),
        main_file_url=f"/download/{base_name}.xlsx",
        audit_file_url=f"/download/{base_name}_audit.csv" if audit_path else None,
        review_file_url=f"/download/{base_name}_review.csv",
        report_url=f"/download/{base_name}_report.txt",
    )


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, filename=filename)

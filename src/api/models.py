"""Pydantic models for API requests and responses."""
from typing import Dict, Optional, List
from pydantic import BaseModel, Field

class ColumnMapping(BaseModel):
    supplier_name: str = Field(..., description="Primary supplier name column")
    secondary_names: Optional[List[str]] = Field(default_factory=list, description="Secondary/alternate name columns")
    name_2: Optional[str] = None
    name_3: Optional[str] = None
    name_4: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    tax_id: Optional[str] = None
    tax_ids: Optional[List[str]] = Field(default_factory=list, description="All tax/VAT/PAN/GST/EIN columns")
    email: Optional[str] = None
    website: Optional[str] = None
    domain: Optional[str] = None
    email_domain: Optional[str] = None
    family_name: Optional[str] = None
    canonical_name: Optional[str] = None
    parent_name: Optional[str] = None
    normalized_supplier_name: Optional[str] = None
    OROVendorId: Optional[str] = None
    CompanyEntityId: Optional[str] = None
    support_fields: Optional[Dict[str, str]] = Field(default_factory=dict, description="Optional support field key -> source column mapping")
    metadata_json_columns: Optional[List[str]] = Field(default_factory=list, description="Columns with JSON metadata that may contain vatNumber/taxNumber")
    json_tax_keys: Optional[List[str]] = Field(default_factory=lambda: ["vatNumber", "taxNumber", "vatId", "taxId", "vatID", "taxID", "vat", "tax", "gstNumber", "gstin", "pan", "tan", "ein", "tin", "abn", "acn", "bn", "siren", "siret", "nif", "cif", "rfc", "ruc", "rut", "nit", "cuit", "cuil", "cnpj", "cpf", "trn", "registrationNumber", "businessRegistrationNumber", "companyRegistrationNumber", "legalRegistrationNumber", "taxRegistrationNumber"])
    json_secondary_name_keys: Optional[List[str]] = Field(default_factory=lambda: ["familyName", "family_name", "parentName", "parent_name", "groupName", "group_name", "tradeName", "trade_name", "dba", "doingBusinessAs", "alternateName", "alternate_name", "legalName", "legal_name"])

class ClusteringRequest(BaseModel):
    column_mapping: ColumnMapping
    auto_cluster_threshold: Optional[float] = Field(0.90, ge=0.0, le=1.0)
    review_threshold: Optional[float] = Field(0.50, ge=0.0, le=1.0)
    generate_audit: Optional[bool] = False
    allow_parent_family_tax_conflicts: Optional[bool] = True
    ai_review_enabled: Optional[bool] = False
    ai_uncertain_cluster_enabled: Optional[bool] = True
    ai_uncertain_match_pct: Optional[float] = 68.0
    max_total_candidate_pairs: Optional[int] = 1000000

class ClusteringStats(BaseModel):
    total_rows: int
    candidate_pairs: int
    candidate_pairs_capped: Optional[bool] = False
    clusters_found: int
    auto_clustered_rows: int
    review_queue_rows: int
    review_candidate_pairs: Optional[int] = 0
    review_candidate_rows: Optional[int] = 0
    guardrail_rejected_pairs: Optional[int] = 0
    singleton_rows: int
    processing_time_seconds: float
    pass_type_counts: Dict[str, int]

class ClusteringResponse(BaseModel):
    success: bool
    message: str
    stats: ClusteringStats
    main_file_url: str
    audit_file_url: Optional[str] = None
    review_file_url: Optional[str] = None
    report_url: Optional[str] = None

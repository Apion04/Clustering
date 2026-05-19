# API Documentation

## Endpoints

### POST /cluster-suppliers

Upload a supplier file and receive clustered output.

**Request:**
```bash
curl -X POST "http://localhost:8000/cluster-suppliers" \
  -F "file=@suppliers.csv" \
  -F 'column_mapping={
    "supplier_name": "Vendor Name",
    "address": "Street Address",
    "city": "City",
    "country": "Country",
    "tax_id": "Tax ID",
    "email": "Email",
    "website": "Website"
  }' \
  -F "auto_cluster_threshold=0.90" \
  -F "generate_audit=true"
```

**Response:**
```json
{
  "success": true,
  "message": "Clustering complete. 124 clusters found.",
  "stats": {
    "total_rows": 100000,
    "candidate_pairs": 2450000,
    "clusters_found": 124,
    "auto_clustered_rows": 380,
    "review_queue_rows": 120,
    "singleton_rows": 99500,
    "processing_time_seconds": 142.5,
    "pass_type_counts": {
      "tax_exact": 45,
      "name_address_exact": 30,
      "name_fuzzy_strong": 25,
      "domain_name_related": 15,
      "address_name_related": 9
    }
  },
  "main_file_url": "/download/clustered_20240513_143022_a1b2c3d4.csv",
  "audit_file_url": "/download/clustered_20240513_143022_a1b2c3d4_audit.csv",
  "report_url": "/download/clustered_20240513_143022_a1b2c3d4_report.txt"
}
```

### GET /download/{filename}

Download a processed file.

### GET /health

Health check.

## Column Mapping

Standard field names and their purpose:

| Standard Name | Required | Description |
|--------------|----------|-------------|
| `supplier_name` | ✅ | Primary supplier name |
| `name_2` | ❌ | Secondary name (DBA, trade name) |
| `name_3` | ❌ | Tertiary name |
| `name_4` | ❌ | Quaternary name |
| `address` | ❌ | Street address |
| `city` | ❌ | City |
| `country` | ❌ | Country (ISO2 or full name) |
| `postal_code` | ❌ | Postal/ZIP code |
| `tax_id` | ❌ | Tax/VAT/PAN/GST/EIN |
| `email` | ❌ | Contact email |
| `website` | ❌ | Website URL |

# Supplier Clustering Engine

Supplier brand/group identity clustering for procurement data cleanup, with strict guardrails against generic/person/location-only false positives.

This package is an engineering-review reference implementation. It is not a production deployment package or auto-approval workflow.

Latest full validation on `/Users/rohitbhojwani/Downloads/tesfile_onelakh.csv` processed 74,355 rows in 106.66s with 847,667 candidate pairs, 60,421 accepted main edges, 8,654 main clusters, 178,488 review candidates, and no candidate cap hit. The full verification report is in `output/oro_safe_75k_identity_verification.md`.

## What It Does

Takes a raw supplier file (CSV/Excel) and automatically:
1. **Normalizes** names, addresses, tax IDs, domains
2. **Blocks** records to avoid comparing all 5 billion pairs
3. **Matches** across deterministic passes (tax, name+address, domain+name, distinctive supplier identity, address+evidence, review-only family/alias candidates)
4. **Merges** clear supplier brand/group identity edges into main clusters with chain-merge protection
5. **Outputs** only 2 new columns: `Cluster Number` + `Match Percentage`
6. **Writes separate Audit and Review files** for explanations and weak/family/alias candidates
7. **Anchor-orders** output so later matching rows move directly under the first occurrence of their cluster

## Quick Start

### 1. Install

```bash
git clone <repo>
cd supplier-clustering-engine
make install
```

Or with Docker:
```bash
docker-compose up --build
```

### 2. Generate Test Data

```bash
python scripts/generate_test_data.py
```

### 3. Run CLI

```bash
python -m scripts.run_cli \
  --input data/sample_suppliers_100.csv \
  --output output/ \
  --llm disabled
```

Mock the full backend LLM decision path without an API key:

```bash
python -m scripts.run_cli \
  --input data/sample_suppliers_100.csv \
  --output output/ \
  --llm mock \
  --openai-model gpt-5.5
```

Run live OpenAI review when `OPENAI_API_KEY` is configured:

```bash
python -m scripts.run_cli \
  --input input.csv \
  --output output/ \
  --llm live \
  --openai-model gpt-5.5
```

### 4. Run API

```bash
make run-api
```

Then POST to `http://localhost:8000/cluster-suppliers` with:
- `file`: CSV upload
- `column_mapping`: JSON string mapping standard names to your column names

Example:
```bash
curl -X POST "http://localhost:8000/cluster-suppliers" \
  -F "file=@data/sample_suppliers_100.csv" \
  -F 'column_mapping={"supplier_name":"Supplier Name","address":"Address","city":"City","country":"Country","tax_id":"Tax ID","email":"Email"}'
```

## Architecture

```
supplier-clustering-engine/
├── src/
│   ├── main.py              # Orchestrator
│   ├── preprocessing.py     # Normalize names/addresses/tax/domains
│   ├── blocking.py          # DuckDB blocking to avoid O(n²)
│   ├── matching.py          # Deterministic matching and review-only pass routing
│   ├── merging.py           # Union-Find with chain protection
│   ├── scoring.py           # Match percentage calculation
│   ├── sorting.py           # Reorder output
│   ├── output.py            # Main, audit, and review file generation
│   ├── idf_tokens.py        # Input-specific rare-token support for review priority
│   ├── ai_review.py         # Optional cached LLM review for ambiguous cases
│   └── api/
│       ├── routes.py        # FastAPI endpoints
│       └── models.py        # Pydantic schemas
├── scripts/
│   ├── run_cli.py           # Command-line interface
│   ├── run_e2e_gemini_review.py # Controlled LLM review workflow
│   └── generate_test_data.py # Test data generator
├── tests/
│   ├── test_preprocessing.py
│   ├── test_matching.py
│   ├── test_integration.py
│   └── test_bad_clusters.py
├── data/                     # Test data
├── output/                   # Generated files
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Output Format

The main output file contains **only your original columns + 2 new columns**:

| Supplier Name | Address | City | ... | **Cluster Number** | **Match Percentage** |
|---------------|---------|------|-----|-------------------|---------------------|
| ABS Safety GmbH | Gewerbering 3 | Kevelaer | ... | **1** | **98%** |
| ABS SAFETY GMBH | GEWERBERING 3 | KEVELAER | ... | **1** | **98%** |
| Unrelated Corp | 999 Other St | Nowhere | ... | **—** | **—** |

- First occurrence of each cluster stays in original place; later matching rows move directly below it
- Non-clustered rows keep their original relative order
- Final user-facing Match Percentage is normalized to `98%`, `85%`, or blank after successful backend LLM processing.
- `70%` is an internal unresolved LLM-review state and should not appear in a completed final user-facing output.

Additional files are separate:

- Audit file: accepted main-cluster row pairs, pass type, score, evidence JSON, guardrail flag, and cluster warnings.
- Review file: family/review/alias/branch/ownership/weak/LLM-suggested candidates that do not get a main `Cluster Number` unless strong duplicate evidence exists.
- Metrics/report files: timings, candidate diagnostics, dictionary validation summaries, and run stats.
- LLM files: queue breakdown, request/response JSONL, cost estimate, decision application report, conflict report, and unresolved exception report.

## Matching Passes

| Signal | Main Cluster? | Notes |
|------|--------|---------|
| Same valid tax ID | Yes/Review | Broad/reused or low-similarity tax IDs require review instead of merging unrelated names. |
| Same normalized name + same/similar full address | Yes | Includes address spelling and house-number range variants. |
| Same distinctive supplier core across legal forms/countries/addresses | Yes/Review | `distinctive_supplier_identity`; exact safe cores can cluster around 90-92%, near/ambiguous cores go to Review. |
| Same business domain + clearly related name | Yes or Review | Depends on name strength and guardrails. |
| Exact duplicate rows | Yes | Main duplicate cluster. |
| Same individual full name/initial variation + same address | Yes/Review | Strict person guardrails apply. |
| Same exact normalized company name but different address and no tax/domain | Review only | Routed as `name_exact_review`. |
| Family/root/brand alias without strong duplicate support | Review only | Never unioned into main clusters by default. |
| Generic token, legal suffix, city/postal/country, first name, surname, one-token risky alias | No | Can support stronger evidence but cannot create clusters. |

Blank `Cluster Number` / `Match Percentage` means no usable relation was found or the candidate was hard-rejected/review-routed. It does not mean the supplier row is bad.

Review-only pass types bypass union-find and are written to `family_review_candidates.csv`.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
AUTO_CLUSTER_THRESHOLD=0.90    # Auto-approve clusters above this
REVIEW_THRESHOLD=0.50          # Flag for review below this
AI_REVIEW_ENABLED=false        # Enable LLM for ambiguous cases
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.5           # Can be changed to gpt-5.4 or later models without code changes
LLM_ENABLED=false
LLM_EXECUTION_MODE=disabled    # disabled, mock, live, batch (sync maps to live)
LLM_SEND_SCOPE=all_review_candidates
ALLOW_UNRESOLVED_LLM_CANDIDATES_IN_FINAL_OUTPUT=false
MAX_LLM_GROUPS_PER_JOB=0
MAX_ROWS_PER_LLM_GROUP=60
MAX_TOKENS_PER_LLM_GROUP=6000
MAX_TOTAL_LLM_COST_PER_JOB=0
LLM_TIMEOUT_SECONDS=60
LLM_RETRY_COUNT=2
OVERRIDE_LLM_CAN_MODIFY_98=false
OPENAI_INPUT_COST_PER_1M_TOKENS=
OPENAI_OUTPUT_COST_PER_1M_TOKENS=
ALLOW_UNKNOWN_LLM_COST=false
AI_MAX_CALLS=50
AI_CACHE_ENABLED=true
AI_CACHE_PATH=.cache/llm_review_cache.json
LLM_CAN_AUTO_CLUSTER=false     # Keep LLM output review-only by default
RARE_TOKEN_MAX_DOCUMENT_FRACTION=0.03
KNOWN_BRAND_FAMILIES_FILE=data/known_brand_families.csv
LEGAL_KEYWORDS_FILE=data/legal_keywords.csv
GENERIC_NON_BRIDGE_FILE=data/generic_non_bridge_keywords.csv
SUPPORT_FIELD_STRENGTHS=        # Optional JSON/comma mapping for canonical/entity support fields
```

Pricing fields must be configured before live LLM mode. If
`OPENAI_INPUT_COST_PER_1M_TOKENS` or `OPENAI_OUTPUT_COST_PER_1M_TOKENS` is
blank/zero, the cost estimate is `UNKNOWN`; it must not be interpreted as a free
run. When `MAX_TOTAL_LLM_COST_PER_JOB` is set and pricing is missing, live mode
fails closed unless `ALLOW_UNKNOWN_LLM_COST=true` is explicitly configured.

### Support Fields

Optional mapped fields such as `family_name`, `canonical_name`, `parent_name`,
`normalized_supplier_name`, `website`, `domain`, `email_domain`, `OROVendorId`,
`CompanyEntityId`, and `tax_id` can be used as support signals. Each support
field is assigned a strength:

- `same_entity_id`
- `same_entity_name`
- `family_or_parent`
- `domain`
- `review_only`

By default, family/parent/canonical text is review-only. It can help blocking,
review priority, and review grouping, but it does not assign a main `Cluster
Number` unless explicitly configured as trusted same-entity evidence. Domain
support can main-cluster only with related/similar supplier names; unrelated
same-domain cases go to Review. The main output remains original columns plus
only `Cluster Number` and `Match Percentage`.

### Known Brand Families

Optional curated alias dictionaries can be loaded with `KNOWN_BRAND_FAMILIES_FILE`.
The expected CSV has one column named `keyword`; each row is semicolon-separated:

```text
canonical_or_primary_name;alias1;alias2;alias3
```

Aliases are matched as full phrases, not split into bridge tokens. The loader
classifies aliases as safe, risky, or generic/non-bridge. Generic aliases such as
`group`, `services`, `technology`, `chemical`, `express`, and `global` are skipped
as bridge signals; risky aliases such as bank/location/common-word names require
extra tax/domain/address/secondary evidence. Known family alias matches create
review candidates unless strong duplicate evidence exists; they are not automatic exact legal-entity approval.

Validate a keyword file with:

```bash
python scripts/validate_known_brand_families.py --input data/known_brand_families.csv
```

### Legal Keywords

Optional global legal entity/company-form dictionaries can be loaded with
`LEGAL_KEYWORDS_FILE`. This file is separate from known brand families. It is
used only for legal suffix cleanup, company/entity detection, avoiding person
misclassification, and global name normalization.

Legal keywords are never used as brand/family aliases, bridge tokens, root
tokens, or clustering evidence by themselves. For example, `GmbH`, `Ltd`, `AG`,
`BV`, and `SA` can normalize `ABC GmbH` to `abc` and mark the record as
company-like, but `ABC GmbH` and `XYZ GmbH` will not cluster because they both
contain `GmbH`.

Short legal forms such as `AB`, `AG`, `AS`, `SA`, `BV`, `OY`, `NV`, and `CO`
are matched only at token boundaries and suffix positions, so `AG` in
`Agilent`, `AB` in `Abbott`, `SA` in `Sartorius`, and `Co` in `Cognis` are not
removed.

Validate a legal keyword file with:

```bash
python scripts/validate_legal_keywords.py --input data/legal_keywords.csv
```

### Generic Non-Bridge Keywords

Optional multilingual generic suppression dictionaries can be loaded with
`GENERIC_NON_BRIDGE_FILE`. These words are used only to prevent weak root,
prefix, family, and token-overlap edges. They never create clusters by
themselves and are separate from both known brand aliases and legal suffixes.

Examples include generic words such as `services`, `technology`, `association`,
`verein`, `stiftung`, `fondation`, `institut`, `societe`, `servicios`,
`servizi`, `logistik`, `logistique`, `green`, `global`, and `international`.

If a word appears in both a brand alias and the generic list, the full phrase
brand alias can still match, but the individual generic token cannot bridge
clusters. For example, `Federal Express` can support a FedEx review family while
`Express` alone remains generic.

Validate a generic suppression file with:

```bash
python scripts/validate_generic_non_bridge_keywords.py --input data/generic_non_bridge_keywords.csv
```

### IDF / Rare Tokens

For each input file, the engine computes a document-frequency table from normalized supplier-name tokens. A token is considered discriminative only when it is rare in that file and not a generic word, legal suffix, risky alias, location token, common business token, or person/surname-risky token.

Rare tokens are deliberately support/review-only:

- They can raise Review file priority.
- They can explain why a candidate is interesting.
- They can help when address/domain/tax/name similarity is already strong.
- They never create a main `Cluster Number` by themselves.

### Optional LLM Review

LLM review is optional and disabled by default. When enabled, it is used for ambiguous review/QA cases, is cached with stable pair hashes, and is capped by `AI_MAX_CALLS`. LLM output is kept in the Review/Audit surfaces unless `LLM_CAN_AUTO_CLUSTER=true` is explicitly configured. Hard guardrails still win.

## Performance

| Dataset Size | Time |
|-------------|------|
| 1,000 rows | ~3 seconds |
| 10,000 rows | ~15 seconds |
| 100,000 rows | ~2-3 minutes |

## Testing

```bash
make test
```

Latest local result for this working state:

```text
python -m compileall src scripts tests
pytest tests/ -v --tb=short
139 passed
```

## Roadmap

- **Phase 1** (Now): Deterministic supplier brand/group identity clustering + separate review queue
- **Phase 2** (Next): Business review of large brand/group clusters and review queue ergonomics
- **Phase 3** (Later): Policy-tuned LLM review for ambiguous cases only
- **Phase 4** (Future): Web research / directory lookup

## License

Internal use only.

## Final ORO Review Decisions

- Main output adds only `Cluster Number` and `Match Percentage`.
- Output uses anchor-based ordering, not full cluster-top sorting.
- Individuals require same/highly similar address in same city/country, same tax, or business-domain support.
- Hotels/restaurants/franchises are not brand-clustered unless same property/address/tax/domain exists.
- Parent/family bridge cases go to the separate Review file unless strong duplicate evidence exists.
- If LLM is uncertain, the candidate remains review-only by default.
- Canonicalization, parent rollup, `canonical_id`, and `canonical_name` are not part of the main output.
- For 100k-200k row files, use blocking caps and tune `MAX_TOTAL_CANDIDATE_PAIRS` to avoid runaway comparisons.

## Final ORO Review Notes

See `docs/FINAL_ENGINEER_NOTES.md` for the final business-rule summary, global tax/domain support, guardrails, scalability controls, and known limitations.

Main output rule: original client columns + `Cluster Number` + `Match Percentage` only.

Output ordering rule: anchor-based ordering, not all clusters at top.

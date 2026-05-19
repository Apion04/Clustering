# Engineer Handoff

## Project Purpose

This package is a working reference implementation for converting supplier/vendor files into review-ready clustered output. It is intended for engineer review and hardening, not immediate production deployment or auto-approval workflows.

The engine accepts CSV or Excel supplier files, maps client-specific columns to standard supplier fields, normalizes supplier signals, generates bounded candidate pairs, applies deterministic matching and guardrails, merges clear supplier brand/group identity matches into main clusters, and writes separate main, audit, and review outputs.

## Output Requirement

The main output file must contain all original client columns plus only:

1. `Cluster Number`
2. `Match Percentage`

Audit, debug, candidate, timing, LLM, and guardrail details belong in separate audit/report/metrics files.

Standard outputs:

1. Main clustered file: original columns + `Cluster Number` + `Match Percentage` only.
2. Audit file: accepted main-cluster edge explanations, evidence, guardrail flags, and cluster warnings.
3. Review file: family/review/alias/branch/ownership/weak/LLM-suggested candidates that must not share a main cluster number unless strong duplicate evidence exists.

## Anchor-Based Ordering

Output ordering is anchor-based:

1. The first occurrence of a cluster stays in its original input position.
2. Later matching rows from that same cluster move directly below the first occurrence.
3. Non-clustered rows retain their original relative order.

Anchor ordering is applied after final cluster decisions.

## Current Reference Dictionaries

The engine now uses three separate reference dictionary types:

- `data/known_brand_families.csv`: phrase-level known brand/family aliases for controlled review clustering.
- `data/legal_keywords.csv`: legal suffix/company-form normalization and entity detection only.
- `data/generic_non_bridge_keywords.csv`: multilingual generic non-bridge suppression only.

These dictionaries must remain separate. Legal suffixes and generic words must never create or bridge clusters by themselves.

The engine also builds an input-file-specific IDF / rare-token table from normalized supplier names. Rare tokens are supporting evidence only: they can raise review priority or explain why a pair is interesting, but they never create a main cluster by themselves.

## Business Rules Summary

- Main clusters represent supplier brand/group identity for review-ready cleanup, not only exact same legal entity duplicates.
- Same valid tax/VAT/PAN/GST/EIN ID is the strongest deterministic signal, but low-similarity exact-tax overlaps route to Review instead of merging unrelated supplier names.
- Multiple tax columns and JSON metadata tax fields are supported.
- Invalid tax placeholders such as `N/A`, `UNKNOWN`, `SUBJECT TO TAX`, `000000`, and `999999` are ignored.
- Country-prefix tax matches require supporting name/address/domain evidence.
- Same normalized name plus same or similar address can cluster.
- Same business domain plus related name can cluster.
- Generic email domains such as Gmail, Yahoo, Outlook, and Hotmail do not drive clustering.
- Same address alone should not cluster unrelated suppliers.
- Individuals do not cluster by name only.
- Same individual name with different address is rejected.
- Different people at the same address are not household-clustered.
- Hotels, restaurants, and franchises do not cluster by brand only.
- Generic roots, multilingual generic terms, city/postal/location tokens, and legal suffixes are not enough to cluster.
- Acronym matches require address, tax, domain, secondary-name, or LLM support.
- Parent/family bridge cases are review-oriented and low/medium confidence.
- Review/family edges bypass the main union-find merger by default.
- Same exact normalized company name with clearly different address and no tax/domain support is review-only unless the shared supplier core is a distinctive, safe brand/group identity.
- Same exact distinctive supplier core can cluster across legal forms/countries/addresses as `distinctive_supplier_identity` in the 80-89 band by default. Scores above 90 are reserved for same/similar address, valid tax/entity ID, business domain, exact duplicate, or other strong local-entity evidence.
- Near-equivalent or ambiguous supplier cores route to Review so they are not silently missed.
- LLM review is optional, cached, budgeted, and guardrail-bound. No provider key is hardcoded, and LLM decisions do not override hard guardrails.
- Canonical parent / ultimate-parent rollup is intentionally not part of the main output.

## What Was Tested

- Python compile checks across `src` and `scripts`.
- Full unit/integration test suite.
- CSV and Excel input paths.
- Main output schema enforcement.
- Anchor-based ordering and singleton relative order.
- Invalid tax placeholder handling.
- Multiple tax fields and JSON metadata tax extraction.
- JSON secondary/family-name extraction.
- Legal suffix normalization across global company forms.
- Brand/family alias phrase matching.
- Multilingual generic token suppression.
- Individual/person guardrails.
- Hotel/restaurant/franchise guardrails.
- Generic email-domain guardrails.
- BUG 1-8 precision fixes from latest review: expanded generic/location/person dictionaries, risky compact alias handling, LAST,FIRST individual detection, cross-country root-only rejection, known-family generic-token guard enforcement, and review-only cluster exclusion from the main output.
- Round-2 structural routing: review/family edges bypass union-find entirely and are written to `family_review_candidates.csv`; same-name/no-support matches are `name_exact_review` review candidates instead of main auto-clusters.
- Priority-sorted review output with evidence flags, suggested action, risky/discriminative token fields, and optional LLM verdict/confidence columns.
- Configurable support-field handling for canonical/family/parent/domain/entity fields. Default family/parent/canonical support is review-only; trusted same-entity fields must be explicitly configured.
- Review output grouping/sorting fields: `shared_support_field`, `support_field_value`, `support_field_strength`, `review_group_key`, and `why_not_auto_clustered`.
- Input-specific IDF / rare-token support as review/support signal only.
- Edge-level audit output for accepted main clusters.
- LLM-disabled deterministic run path.
- Brand/group identity policy pass: 3BL Media, 4titude, 3B Scientific, AirCom Pneumatic/Pneumatik, AIRTEC Pneumatic, Air Liquide, Air Products, Ajinomoto, Cognis, Cognizant, Computershare, Colonial Metals, A. Hartrodt, ADEKA, Advance Research Chemicals, Aescolab, and AGA Gas.
- Low-similarity exact-tax review routing for cases such as ABCR/Guntermann & Drunck, AC medienhaus/Druckerei Chmielorz, and 4Tune/ValGenesis.
- CSV original-column preservation: input CSV columns are read as text so boolean-looking values and leading-zero strings round-trip unchanged.
- Large-file staged performance path at 5k, 20k, 50k, and full 74,355 rows.

## Latest Test Result

- Compile check: passed with a local pycache prefix.
- Unit/integration tests: `155 passed`.

## Latest Local CLI Validation

LLM was disabled for this validation.

| Input | Runtime | Candidate Pairs | Match Edges | Main Clusters | Review Candidate Pairs | Largest Cluster | Candidate Cap Hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/Users/rohitbhojwani/Downloads/tesfile_onelakh.csv` | 106.66s | 847,667 | 60,421 | 8,654 | 178,488 | 604 | false |

Validation checks:

- Rows preserved: yes.
- ExternalId identity set preserved: yes.
- Main output columns correct: yes.
- Anchor first occurrence preserved: yes.
- Cluster rows contiguous: yes.
- Singleton relative order preserved: yes.
- Review file priority sorted: yes.
- Audit file contains edge-level evidence: yes.
- Support/canonical/family examples were routed to Review unless configured as trusted same-entity fields: yes.

## Latest Deterministic Full-File Run

LLM was disabled for this run.

| Run | Runtime | Candidate Pairs | Raw Pairs Before Caps | Match Edges | Main Clusters | Review Candidate Pairs | Largest Cluster | Candidate Cap Hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full 74,355 | 106.66s | 847,667 | 3,969,779 | 60,421 | 8,654 | 178,488 | 604 | false |

Additional full-run diagnostics:

- Blocking keys: 576,122
- Unique block keys: 360,079
- Skipped oversized blocks: 34
- Main clustered rows: 12,608
- Review queue rows inside main clusters: 12,092
- Review candidate rows: 27,804
- Singleton rows: 49,655
- Distinctive supplier identity edges: 39,738
- AI/human review export rows: 5,401
- Score distribution: 5,167 clusters at 95-100, 343 at 90-94, 2,781 at 80-89, 363 at 70-79, and 49,655 blank/singleton rows.
- Distinctive supplier identity distribution: 2,730 clusters at 80-89 and 48 at 70-79; none are 90+.
- 90+ multi-country clusters without tax/domain/address support: 0.

Latest full-file verification:

- Rows preserved: yes.
- External ID multiset preserved: yes.
- Output columns correct: yes.
- Anchor ordering correct: yes.
- Cluster rows consecutive: yes.
- Noncluster relative order preserved: yes.
- Known bad generic/person/city/tax false positives fixed: yes.
- Old bad examples verified as not sharing main Cluster Number: MUELLER/Müller Processing, Cindy, Private, Partner, Industrie, Product, Biological, Medizin/Medicine, Excellence, Europa, Alfa, Meta/Qingdao, ADM, Zimmer, 4Tune/ValGenesis, ABCR/Guntermann, AC medienhaus/Druckerei, IMEC/Tempmate, and Acrylate/Methacrylate REACH Task Force.
- New supplier identity examples verified: 3BL Media, 4titude, 3B Scientific, AirCom Pneumatic/Pneumatik, AIRTEC Pneumatic, Air Liquide, Air Products, Ajinomoto, Cognis, Cognizant, Computershare, Colonial Metals, A. Hartrodt, ADEKA, Advance Research Chemicals, Aescolab, and AGA Gas are clustered or surfaced in Review according to strength.
- Review/family candidates are excluded before union-find and written to `family_review_candidates.csv`.
- Low-score main clusters 65-76%: 0 after Round 2.

## Included Reports

- `output/known_brand_families_validation.md`
- `output/legal_keywords_validation.md`
- `output/generic_non_bridge_validation.md`
- `output/precision_pass2_full_metrics.json`
- `output/precision_pass2_full_clustered.csv`
- `output/precision_pass2_full_audit.csv`
- `output/precision_pass2_full_report.txt`
- `output/precision_pass2_family_review_candidates.csv`
- `output/precision_pass2_case_verification.md`
- `output/precision_pass2_top25_cluster_review.md`
- `output/precision_pass2_suspicious_generic_low_score_scan.md`
- `output/precision_pass2_low_score_cluster_review.md`
- `output/oro_safe_75k_clustered.csv`
- `output/oro_safe_75k_audit.csv`
- `output/oro_safe_75k_review.csv`
- `output/oro_safe_75k_metrics.json`
- `output/oro_safe_75k_report.txt`
- `output/oro_safe_75k_support_verification.md`
- `output/oro_safe_75k_identity_clustered.csv`
- `output/oro_safe_75k_identity_audit.csv`
- `output/oro_safe_75k_identity_review.csv`
- `output/oro_safe_75k_identity_metrics.json`
- `output/oro_safe_75k_identity_report.txt`
- `output/oro_safe_75k_identity_verification.md`

## How To Run CLI

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run with auto-detected columns and LLM disabled:

```bash
python -m scripts.run_cli \
  --input data/sample_suppliers_100.csv \
  --output output/ \
  --llm disabled
```

The CLI writes deterministic internals plus `final_supplier_clustered.csv`. In a successful LLM run the final user-facing file contains only original columns plus `Cluster Number` and `Match Percentage`, and scores are only `98%`, `85%`, or blank.

Run the full backend LLM decision path without an API key:

```bash
python -m scripts.run_cli \
  --input data/sample_suppliers_100.csv \
  --output output/ \
  --llm mock \
  --openai-model gpt-5.5
```

Run live OpenAI review:

```bash
python -m scripts.run_cli \
  --input path/to/suppliers.csv \
  --output output/ \
  --llm live \
  --openai-model gpt-5.5
```

The model is configurable with `OPENAI_MODEL` or CLI `--openai-model`; no key is hardcoded. Default is `gpt-5.5`, and it can be changed to `gpt-5.4` without code changes.

Before enabling live OpenAI mode, configure pricing in `.env` or deployment
config:

```bash
OPENAI_INPUT_COST_PER_1M_TOKENS=
OPENAI_OUTPUT_COST_PER_1M_TOKENS=
MAX_TOTAL_LLM_COST_PER_JOB=
ALLOW_UNKNOWN_LLM_COST=false
```

If pricing is blank/zero, the cost estimate is `UNKNOWN`, not free. When
`MAX_TOTAL_LLM_COST_PER_JOB` is set and pricing is missing, live mode fails
closed unless `ALLOW_UNKNOWN_LLM_COST=true` is explicitly configured.

If `--review-output` is omitted, the CLI auto-saves review candidates as `family_review_candidates.csv` in the output directory.

Run with an explicit mapping:

```bash
python -m scripts.run_cli \
  --input path/to/suppliers.csv \
  --mapping docs/SAMPLE_MAPPING_REAL_FILE.json \
  --output output/ \
  --llm disabled \
  --audit output/audit.csv \
  --report output/report.txt \
  --metrics output/metrics.json
```

Run a bounded sample:

```bash
python -m scripts.run_cli \
  --input path/to/suppliers.csv \
  --max-rows 5000 \
  --output output/ \
  --llm mock \
  --audit output/audit_5k.csv \
  --report output/report_5k.txt \
  --metrics output/metrics_5k.json
```

## How To Run API

```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Cluster a file:

```bash
curl -X POST "http://localhost:8000/cluster-suppliers" \
  -F "file=@data/sample_suppliers_100.csv" \
  -F 'column_mapping={"supplier_name":"Supplier Name","address":"Address","city":"City","country":"Country","tax_id":"Tax ID","email":"Email"}'
```

## How To Run Tests

```bash
pytest tests/ -v --tb=short
```

Or:

```bash
make test
```

## How To Validate Dictionaries

```bash
python scripts/validate_known_brand_families.py --input data/known_brand_families.csv
python scripts/validate_legal_keywords.py --input data/legal_keywords.csv
python scripts/validate_generic_non_bridge_keywords.py --input data/generic_non_bridge_keywords.csv
```

## Known Limitations

- Candidate caps and block-size caps may reduce recall on very broad 100k-200k files.
- Global address parsing is heuristic.
- Large family/review relationships such as Merck and Siemens are review-candidate pairs, not auto-approval-ready clusters.
- IDF / rare-token support is review-only; it can increase review output volume on broad files and should be monitored.
- Low-confidence family candidates need reviewer/LLM policy tuning.
- LLM review can only review candidates surfaced by deterministic logic.
- LLM decisions are review/audit support unless `LLM_CAN_AUTO_CLUSTER=true` is explicitly configured.
- Canonicalization, branch rollup, parent rollup, `canonical_id`, and `canonical_name` are not part of the main deliverable.
- Output is review-ready, not auto-approval-ready.
- 100k-200k files should be tested on the target server profile before production use.

## Remaining Production Concerns

- Production deployment still needs observability, queueing/background execution, access control, retention policy, and operational monitoring.
- Legal/privacy review is needed before enabling external LLM review on client data.
- Business team should confirm score thresholds for review versus auto-cluster display.
- Very large brand-family dictionaries can increase review-cluster recall and should be monitored for broad-family overreach.
- Review-only family/name/acronym candidates are intentionally not given a main output cluster number. Engineering should confirm whether `family_review_candidates.csv` is the right downstream manual-review surface.

## Ready For Engineer Review

Yes. This package is ready for engineer review as a tested reference implementation. It is not positioned as production-ready without further hardening.

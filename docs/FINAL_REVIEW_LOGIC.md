# Final Supplier Clustering Logic

## Output contract
The main reviewer file must keep all original client columns and add only:

- Cluster Number
- Match Percentage

Audit/debug details stay in separate audit files.

## Ordering contract
The output uses anchor-based ordering:

1. The first occurrence of a cluster stays where it originally appeared.
2. Later matching rows in that cluster move directly below the first occurrence.
3. Moved rows are removed from their later positions.
4. Non-clustered rows keep their original relative order.

## High-level pipeline
1. Upload CSV/XLSX file.
2. Map client columns to standard fields.
3. Normalize names, addresses, cities, countries, tax IDs, domains, and secondary names.
4. Generate candidate pairs through blocking, not full pairwise comparison.
5. Score candidate pairs through tax, name, address, domain, secondary-name, acronym, and parent/family matchers.
6. Apply precision guardrails before merging.
7. Optionally send ambiguous candidates to LLM review.
8. Merge accepted edges into clusters.
9. Assign Cluster Number and Match Percentage.
10. Write output with anchor-based ordering.

## Strong automatic cluster conditions
- Exact same tax/VAT/PAN/GST/EIN.
- Same normalized name plus same normalized address.
- Exact duplicate rows after normalization.
- Same company name and exact/highly similar address/location.

## Loose tax condition
If one record has a country-prefixed tax value and another has only the numeric part, for example DE233002380 vs 233002380, do not cluster on tax alone. Require name, address, or domain support.

## Invalid tax values
Ignore invalid placeholders such as SUBJECT TO TAX, N/A, UNKNOWN, NOT PROVIDED, 000000, and 999999.

## Address-only condition
Same address alone does not cluster. It must have support from name similarity, tax, domain, secondary names, acronym/DBA evidence, or LLM approval.

## Individual/person guardrail
Individuals cannot cluster by name only.

Allowed:
- Same/highly similar person name plus exact/highly similar address in same city/country.
- Same tax or same business domain.

Rejected:
- Same first name only.
- Same full name but different address.
- Common-name groups such as Robert/Peter names without exact address support.

## Hotel/restaurant/franchise guardrail
Avoid brand-level clustering for hotels, restaurants, and franchises unless same property/address/tax/domain exists.

Allowed:
- Same hotel/property at same address/location.
- Same tax/domain and location support.

Rejected:
- Hyatt Mumbai, Hyatt Pune, Hyatt Dubai by brand only.
- Parkhotel Krone, Parkhotel Engelsburg, Parkhotel St. Leonhard by root word only.

## Parent/family bridge condition
Parent/family bridges are allowed only for company-like records and only when plausible evidence exists.

Examples:
- Rain Carbon / Rütgers / Ruetgers family cases.
- PPC / Potasse / VYNOVA PPC cases.

These should be lower confidence and can be sent to LLM review.

## LLM review behavior
LLM review is optional and should only run on ambiguous candidate pairs, never all rows.

Send the mapped original supplier name, address, city, country, tax IDs, domain/email/website, secondary names, and normalized values.

LLM decisions:
- cluster: merge and use the LLM-supported match percentage, capped below deterministic exact matches.
- do_not_cluster: reject.
- uncertain: still group for manual review with low score, default 68%.

## Scalability guardrails
The engine uses blocking and caps to avoid full pairwise comparison:

- max_candidates_per_block
- max_total_candidate_pairs
- block-size limits by block type
- no broad city-country-only blocks
- generic root suppression

For 100k-200k rows, engineers should tune max_total_candidate_pairs and run on a server if files are large or many broad blocks exist.

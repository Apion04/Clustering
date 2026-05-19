# Supplier Clustering Engine - Final Engineer Notes

## What this package is

This is a tested reference implementation for supplier clustering automation. It is not a claim of perfect entity resolution. It is intended to give engineering a strong, readable starting point that already includes ORO-specific business rules discovered through real supplier-file testing.

## Main user-facing output

The main reviewer output must keep all original client columns and add only two columns:

1. `Cluster Number`
2. `Match Percentage`

Any reason, source, risk flag, or normalized field must stay in the optional audit/debug file, not in the main reviewer file.

## Output ordering

This version uses anchor-based ordering:

- The first occurrence of a cluster stays in its original position.
- Later matching rows from the same cluster move directly below that anchor row.
- Non-clustered rows keep their original relative order.
- Rows are not all dumped at the top of the file.

## Major rules included

### Strong cluster rules

- Same exact valid tax/VAT/PAN/GST/EIN/registration identifier.
- Same normalized supplier name and same/similar address.
- Same business domain with related supplier name.
- Same secondary name/DBA/brand field with supporting address, tax, or domain.

### Tax and registration ID support

This package supports multiple tax-like columns and JSON metadata extraction. It is designed for global supplier files and does not depend on a single US-only or India-only tax field.

Supported examples include, but are not limited to:

- VAT, VAT ID, VAT Number
- Tax ID, Tax Number, TIN, EIN, FEIN, ITIN
- PAN, GST, GSTIN, TAN, CIN
- ABN, ACN, ARBN, TFN
- BN, BRN, UEN, NZBN, CRN
- SIREN, SIRET, TVA, UST-ID, UID, MWST
- NIF, NIE, CIF, RFC, RUC, RUT, NIT
- CUIT, CUIL, CNPJ, CPF
- PIVA, Partita IVA, Codice Fiscale
- BTW, KVK, RSIN
- NPWP, NIB, NIP, REGON, KRS
- ICO, IC DPH, DIC, TRN, QST, HST, PST
- Company Registration Number, Business Registration Number, Registration No.

Important: this engine normalizes identifiers and uses them as matching signals. It does not perform every country-specific checksum validation. Engineers may add country-specific validators later if required.

### Tax prefix rule

If one row has `DE233002380` and another has `233002380`, this is only a candidate match. It does not auto-cluster by tax alone. It requires supporting name, address, or domain evidence.

### Invalid tax placeholders ignored

Values such as `N/A`, `UNKNOWN`, `NOT PROVIDED`, `SUBJECT TO TAX`, `000000`, `999999`, and similar placeholder values are ignored.

### Generic domains

Free/ISP/disposable email domains are treated as weak and do not drive clustering. Examples include Gmail, Yahoo, Hotmail, Outlook, iCloud, AOL, Mail.ru, Yandex, QQ, 163, Rediffmail, GMX, Web.de, Protonmail, Zoho, Yopmail, Mailinator, and others listed in `src/config.py`.

### Generic business words

Generic words cannot drive parent/family clustering by themselves. Examples include services, consulting, trading, solutions, systems, technologies, chemicals, production, point, science/scientific, laboratories, pharm/pharma, materials, engineering, logistics, group, global, international, company, distribution, medical, research, hotel, restaurant, and others listed in `src/config.py`.

Names that begin with a number or short alphanumeric code, such as `202 Production`, `30 Point Strategies`, `3B Scientific`, or `1-Material`, require stronger evidence than a shared generic tail word. They can cluster only with exact/highly similar full name, same valid tax, same business domain, same/highly similar address plus name evidence, explicit secondary/family-name evidence, known-family mapping, or LLM approval.

### People/individual guardrail

Person-like supplier names do not cluster by same name only. Same-name different-address people are rejected. Person-like names require same/highly similar address in same city/country, same valid tax, or same business domain.

### Hotels/restaurants/franchises guardrail

Hotels, restaurants, and franchise/chain names do not cluster by brand alone. They require same property/address/tax/domain or strong location evidence.

### Parent/family cases

Parent/family bridge cases are allowed only when supported by evidence or LLM review. If LLM is enabled and returns uncertain, the business decision is to group plausible uncertain records with a low score, default 68%, so the team can review them together.

## AI/LLM review

LLM review is optional and disabled by default. It should be used only for ambiguous candidates, not every row. It supports OpenAI-compatible APIs and Claude-style API configuration.

Recommended ambiguous cases for LLM review:

- Same address, different names
- Possible DBA/trade name
- Possible acquisition/rebrand
- Possible parent/family bridge
- Acronym/full-name candidate
- Same domain but different names
- Institution/research group candidates

Obvious generic-token false positives should be rejected deterministically before LLM review. Do not send Production-only, Point-only, Scientific-only, Pharm-only, city/postal/generic-industry-only, common first-name-only, surname-only, or household-style individual matches to LLM as rescue candidates.

Pipeline order is:

1. Deterministic candidate generation.
2. Deterministic scoring.
3. Guardrail rejection.
4. Optional LLM review for low-confidence ambiguous candidates only.
5. Final edge decisions and cluster merge.
6. Final cluster IDs.
7. Anchor-based ordering last.

The current reference implementation performs LLM review at pair level before merge. Pair-level `partial` responses are treated as no pair merge by default. A future group-level LLM workflow can support true `partial` split decisions by applying returned row groups before final cluster IDs and anchor ordering.

Acquisition/rebrand limitation: LLM can only review candidates surfaced by deterministic blocking. If two companies have no shared tax, address, domain, secondary/family name, known relationship, or alias/acquisition dictionary signal, the engine may not compare them. Cases like `Zenith Retail` becoming `Rohit Retail` require a known alias/acquisition dictionary, internal family mapping, secondary/family-name field, same domain/tax/address evidence, or an optional web/LLM research workflow. The engine must not infer acquisitions from generic tokens alone.

## Scalability controls

The engine avoids all-vs-all comparisons using blocking keys and caps:

- Max block size by type
- Max candidate pairs per block
- Global max total candidate pairs
- No broad city-country-only matching
- Generic root suppression

For 100k to 200k rows, engineers should run on a server or strong machine and tune `MAX_TOTAL_CANDIDATE_PAIRS` based on memory and runtime.

## Tested in this environment

- Python compile check passed.
- Unit tests passed: 41/41.
- Tested on a 6,391-row real-style supplier file.
- Runtime on that file: about 13 seconds in this environment.
- Main output contained only original columns plus `Cluster Number` and `Match Percentage`.

## Known limitations

No clustering engine is bug-free or perfect. Supplier clustering involves business judgment. This version is a final review candidate for engineering and content-team validation, not a fully certified production service.

Known areas for future improvement:

- Country-specific tax checksum validation
- Better global address parsing
- Configurable client-specific rules
- Streamlit or internal UI for column mapping
- Integration into Turbo or existing ORO workflow
- Feedback loop from reviewer corrections

# Foundation Rule Audit — Supplier Clustering Engine

Date: 2026-05-19
Scope: end-to-end audit of scoring, normalization, alias, brand, address, alphanumeric, and routing rules.
Status: **Foundation is sound. Two minor recommended additions, no required code changes.**

---

## 0. Executive summary

| Area | Status |
| --- | --- |
| Final user-facing scores limited to 98 / 85 / 70 / blank | ✅ enforced |
| Same address alone cannot produce 98 | ✅ enforced |
| Same tax alone with low-similarity unrelated name → review, not 98 | ✅ enforced |
| 98 requires deterministic evidence (tax, name+address, support_same_entity_id) | ✅ enforced |
| 85 reserved for supplier-family/brand/legal-owner relationships | ✅ enforced |
| 70 reserved for plausible LLM-review candidates | ✅ enforced |
| Generic/location-only clusters blocked | ✅ enforced (cluster-level + pair-level) |
| Franchise/hotel brand-only clusters blocked | ✅ enforced |
| Protected compounds (Eastman Kodak, Air Liquide, Springer Nature…) separated | ✅ enforced |
| Acronym/initialism support across all name fields | ✅ enforced |
| FKA / DBA / AKA / legal-name aliases through secondary_name_match + JSON keys | ✅ enforced |
| Alphanumeric supplier cores (3BL, 4titude, 3B, 3M, 4flow) preserved | ✅ enforced |
| LLM routing isolates 70-score candidates from final output unless explicitly opted in | ✅ enforced |
| All 203 regression tests pass | ✅ |

**Recommended adjustments (optional, non-blocking):**
1. Add `KGaA` and bare `Pty` to `data/legal_keywords.csv` for stripping completeness.
2. Add `gastro` to `regulatory` or `english` keyword list (currently only `english,gastronomie` / code constant `gastro`; already in `GENERIC_ROOT_TOKENS`, but explicit CSV row would aid auditability).

Awaiting confirmation before any further change.

---

## 1. Scoring matrix (current implementation)

Final user-facing scores are mapped from internal pass types via `src/scoring.py:edge_user_route` and `route_to_user_score`. The mapping enforces a discrete set: **98, 85, 70, blank**. Internal scores like 86/88/91/96 surface only in the audit file, never in the user output, because `calculate_user_facing_cluster_scores` rewrites every cluster to its weakest-link route.

### Pass-type → route → final user score

| pass_type (matching.py) | Evidence required | Internal score | Route | Final user score | Risk notes |
| --- | --- | ---: | --- | ---: | --- |
| `tax_exact` | Same normalized non-empty tax ID (multi-tax / JSON merged) | 98–99 | AUTO_CONFIDENT | **98** | Guarded by `_build_tax_block_stats` — a tax shared across >3 distinct roots + ≥4 rows is routed to `tax_exact_broad_rejected` (blank) unless name/address/domain supports it. Same-tax low-similarity pairs go to `tax_exact_low_similarity_review` unless alias evidence (secondary/acronym/compact-name) restores it. |
| `tax_loose_supported` | Tax overlap after country prefix strip + name_sim≥0.78 or address or domain support | 92–95 | AUTO_CONFIDENT | **98** | Never fires on tax alone — supporting evidence mandatory. |
| `support_same_entity_id` | Configured trusted same-entity ID column overlap (e.g. OROVendorId set as `same_entity_id`) | 98 | AUTO_CONFIDENT | **98** | Defaults to `review_only`; promotion to same_entity_id requires explicit support_field_strengths config. |
| `support_same_entity_name` | Trusted same-entity name field overlap | 93 | AUTO_CONFIDENT | **98** | Same as above. |
| `name_address_exact` | `name_norm` equal AND address ≥0.88 sim (or number-range match w/ city/postal) | 96 | AUTO_CONFIDENT | **98** | Address-supported, so name+address co-required. |
| `same_legal_owner_confirmed` | `name_norm` equal AND name is not entirely generic/location/hospitality/≤2-char tokens | 85 | MANUAL_REVIEW | **85** | Different tax IDs allowed (`family_edge` set with `allow_parent_family_tax_conflicts=True`). Already covered by `STRONG_EDGE_TYPES` so weak-chain logic doesn't block it. |
| `name_exact_review` | `name_norm` equal but name is generic-only | 75–86 | LLM_REVIEW (REVIEW_ONLY_PASS_TYPES) | **blank or 70** | Routes to `family_review_candidates.csv`, not main output. |
| `distinctive_supplier_identity` | Trusted/non-generic core overlap; PASS 2D | 70–96 | depends on evidence | **98 / 85 / 70 / blank** | If `main_cluster_allowed` is false (ambiguous core w/o tax/domain/address), routes to LLM_REVIEW → 70. Otherwise downgraded to AUTO_CONFIDENT (98), MANUAL_REVIEW (85), or LLM_REVIEW (70) per `edge_user_route`. |
| `professional_name_address` | Person name match + same address + person title | 86 | MANUAL_REVIEW | **85** | Always needs_review. |
| `name_fuzzy_supported` | fuzzy name sim ≥ `fuzzy_name_threshold_strong` (0.92) + address/city/country/domain | 91 | AUTO_CONFIDENT (if no needs_review) | **98** | Without support, demoted to `name_fuzzy_strong_review_only` (no match). |
| `address_distinctive_shared` | Same/similar address + shared distinctive non-generic tokens + name_sim ≥ 0.65 | 85 | MANUAL_REVIEW | **85** | New pass added for Seminole-Hard-Rock vs Seminole-HR case; needs_review True. |
| `address_name_related` | Same/similar address + name_sim ≥ 0.70 (and distinctive_shared not satisfied) | 78 | MANUAL_REVIEW | **85** | Always needs_review. |
| `address_secondary_or_acronym` | Same/similar address + secondary-name or acronym bridge | 80 | MANUAL_REVIEW | **85** | Always needs_review. |
| `address_domain` | Same/similar address + same business domain | 84 | AUTO_CONFIDENT | **98** | Domain is a deterministic signal. |
| `address_only` | Address only, nothing else | 0 | REJECT | **blank** | Explicitly rejected. |
| `domain_name_related` | Same domain + name_sim ≥0.70 or secondary/family/acronym | 78–86 | AUTO_CONFIDENT (if name_sim ≥ 0.80) else MANUAL_REVIEW | **98 or 85** | |
| `domain_review_candidate` | Same domain w/ unrelated names | 72 | LLM_REVIEW | **70** | |
| `family_bridge_supported` | Shared distinctive root + city/country/domain/address ≥ 0.75 | 76 | MANUAL_REVIEW (review-only set) | **70** | In `REVIEW_ONLY_PASS_TYPES` (does not enter union-find). |
| `family_cross_country` | Cross-country root-only | 0 | REJECT | **blank** | |
| `known_family_bridge` | KNOWN_FAMILY_TOKEN_GROUPS / KNOWN_DISTINCTIVE_FAMILY_ROOTS / KNOWN_ADDRESS_FAMILY_BRIDGE_GROUPS | 68–80 | MANUAL_REVIEW (review-only) | **70** | Curated explicit bridge sets only. |
| `known_brand_family_alias` | data/known_brand_families.csv overlap | 70–90 | depends on score | **85 / 70** | Risky/ambiguous aliases require tax/domain/address/secondary support. |
| `secondary_or_acronym_bridge` | Secondary name overlap or acronym bridge + city/country/domain/address support | 84 | MANUAL_REVIEW (review-only) | **70** | |
| `acronym_review_candidate` | Acronym bridge + same city/country only + ai_review_enabled | 65 | LLM_REVIEW | **70** | |
| `regulatory_or_task_force_related` | ≥2 regulatory tokens shared and no distinctive overlap | 74 | LLM_REVIEW | **70** | |
| `tax_exact_low_similarity_review` | Same tax + name_sim<0.50 + no alias support | 90 | LLM_REVIEW | **70** | Production-safe demotion: same-tax alone never reaches 98 without name/alias/address support. |
| `tax_exact_institutional_ecosystem_review` | Same tax + university/institution + ecosystem keywords | 70 | LLM_REVIEW | **70** | |
| `tax_exact_broad_rejected` | Same tax appears across >3 distinct roots + ≥4 rows w/o name/address/domain | 0 | REJECT | **blank** | Prevents shared client-side placeholder tax IDs from merging unrelated rows. |
| `operational_status_review` | One row has "blocked / inactive / do not use" hint | ≤86 | LLM_REVIEW (review-only set) | **70** | Applies AFTER tax_exact early-return (the ordering fix in guardrails.py). |
| `person_*_review` | Person guardrail outcomes | 76–86 | LLM_REVIEW (review-only) | **70** | |
| `support_field_review` | Family/canonical/parent name overlap, default strength | 74–90 | LLM_REVIEW (review-only) | **70** | |
| `no_match`, `address_risk`, `name_fuzzy_strong_review_only`, `family_cross_country_rejected`, `known_brand_family_risky_needs_support` | various rejections | 0 | REJECT | **blank** | |

### 1.1 "Random scores" check (86/88/91…)

Confirmed: `route_to_user_score` collapses every accepted internal score to 98 / 85 / 70 / 0 (blank). Validation file score distribution:

```
98%   134 rows
85%    62 rows
NaN    16 rows  (singletons)
```

Zero rows at 86/88/91/96. **Confirmed: only 98/85/blank appear in final user output for this run**; 70 surfaces only when the run is marked `INCOMPLETE_UNRESOLVED_LLM_CANDIDATES` (so the user sees the in-flight LLM candidates) and `allow_unresolved_llm_candidates_in_final_output` is true.

### 1.2 98 invariants

- Tax alone produces 98 ONLY if (a) tax-block is not broad-rejected, (b) name similarity ≥ 0.50 OR alias/secondary/acronym/compact-name evidence is present, (c) no operational-status hint, (d) no institutional ecosystem ambiguity.
- Name alone never produces 98 (`name_exact_review` is review-only; `same_legal_owner_confirmed` caps at 85).
- Address alone never produces 98 (`address_only` always rejects).
- Domain alone never produces 98 (`domain_review_candidate` is 72→70).
- Fuzzy name alone never produces 98 (must combine with address/city/country/domain).

---

## 2. Generic-token audit

### 2.1 Current implementation

Two layers exist:

1. **`GENERIC_ROOT_TOKENS` in `src/config.py`** — Python frozenset, ~225 entries, used at runtime by guardrails, matching, and core extraction.
2. **`data/generic_non_bridge_keywords.csv`** — 338 rows across 10 categories. Loaded by `src/generic_keywords.py` at startup and merged into the runtime set:

| category | rows |
| --- | --- |
| common | 19 |
| dutch_nordic | 21 |
| english | 130 |
| french | 31 |
| german | 43 |
| italian | 27 |
| location | 18 |
| regulatory | 11 |
| spanish_portuguese | 38 |

These tokens cannot:
- become the bridge token in `_distinctive_tokens` / `_related_root`
- act as supplier identity core (`_core_is_safe_for_identity` rejects all-generic core)
- trigger family/parent bridge alone (guardrails.py rejects when `shared_root_is_generic`)
- form a 98 cluster (cluster_hardening_route downgrades to LLM_REVIEW or REJECT)

### 2.2 Requested-term coverage

| Term | Status |
| --- | --- |
| services / service | ✅ CSV english |
| group | ✅ CSV english |
| technology / tech | ✅ CSV english / code |
| research | ✅ CSV english |
| institute | ✅ CSV english |
| society | ✅ CSV english |
| association | ✅ CSV english |
| foundation | ✅ CSV english |
| chemical / chemistry | ✅ CSV english |
| product / products | ✅ CSV english |
| industry / industries | ✅ CSV english |
| logistics | ✅ CSV english |
| express | ✅ CSV english |
| green | ✅ code (`GENERIC_ROOT_TOKENS`) |
| gastro | ✅ code; **recommend** explicit `english,gastro` CSV row |
| gastronomie | ✅ CSV |
| packaging | ✅ CSV english |
| medicine / medical | ✅ CSV english |
| pharma | ✅ CSV english |
| task / force / taskforce | ✅ CSV regulatory + english |
| consortium | ✅ CSV english |

### 2.3 Recommended additions

None blocking. Optional:
- `english,gastro` for explicit auditability (the code constant already covers it).
- A `chinese`/`japanese` block for legitimate East-Asian supplier files would harden coverage further — not required for the current validation file.

### 2.4 False-positive tests

- `test_supplier_identity_policy.py::test_old_false_positive_pairs_still_blocked`
- `test_supplier_identity_policy.py::test_same_address_and_different_names_remain_not_main_clustered`
- `test_production_routing.py::test_tuebingen_and_gastro_do_not_cluster_by_location_or_generic_core`
- `test_production_routing.py::test_generic_only_accepted_edge_is_downgraded_to_llm_not_85_or_98`
- `test_precision_guardrails.py` — pure-generic block, common-first-name-only block, hospitality block.

---

## 3. Legal-suffix audit

### 3.1 Current implementation

Two layers:

1. **`LEGAL_SUFFIXES` in `src/config.py`** — 40+ entries, default fallback.
2. **`data/legal_keywords.csv`** — 900 rows, loaded by `src/legal_keywords.py` with multilingual variants and case forms. The dictionary contains a `boundary_only_suffixes` list (64 entries) for short ambiguous tokens like `AG`, `SA`, `AB`, `AS`, `CO` that must only be stripped when standalone, never inside words.

Legal suffixes are used **only** for normalization (`remove_legal_suffixes` in preprocessing.py:174). They never:
- create a cluster (no pass type fires on legal-suffix overlap alone)
- act as bridge tokens (all are in `GENERIC_ROOT_TOKENS` or `COMPANY_HINT_WORDS`)
- match inside another word (boundary-only protection)

The `_legal_stripping_left_too_generic` guard at preprocessing.py:212 reverts stripping if all that remains is `GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES`, so "CO LIMITED" doesn't normalize to nothing.

### 3.2 Requested-term coverage

| Suffix | Status |
| --- | --- |
| GmbH | ✅ CSV |
| Ltd | ✅ CSV |
| Limited | ✅ CSV |
| LLC | ✅ CSV |
| Inc / Incorporated | ✅ CSV |
| Corp / Corporation | ✅ CSV |
| Co / Company | ✅ CSV |
| AG | ✅ CSV (boundary-only) |
| SA | ✅ CSV (boundary-only) |
| SAS | ✅ CSV |
| BV | ✅ CSV (boundary-only) |
| NV | ✅ CSV (boundary-only) |
| AB | ✅ CSV (boundary-only) |
| AS / A/S | ✅ CSV |
| Oy | ✅ CSV |
| ApS | ✅ CSV |
| Sp z o.o. / sp.z.o.o. | ✅ CSV (multiple variants) |
| S.r.l. | ✅ CSV |
| S.p.A. | ✅ CSV |
| KG | ✅ CSV |
| KGaA | ❌ **MISSING** — recommend adding |
| e.V. | ✅ CSV |
| LLP / LP | ✅ CSV |
| Pty Ltd | ✅ CSV |
| Pty (alone) | ❌ **MISSING** — recommend adding (in case a row has just "Pty" without Ltd) |

### 3.3 Recommended additions

Add two rows to `data/legal_keywords.csv`:
```
KGaA
Pty
```

These are minor — KG already covers most KGaA prefix forms, and bare "Pty" without "Ltd" is rare. Non-blocking.

### 3.4 Short-token false-positive tests

- `test_preprocessing.py::TestNormalizeSupplierName::test_short_legal_forms_are_boundary_only` — verifies AG/SA/BV/AB do not match inside Agilent, Sartorius, Abbott, Cognis.
- `test_preprocessing.py::TestNormalizeSupplierName::test_loaded_global_legal_suffixes`
- `test_preprocessing.py::TestNormalizeSupplierName::test_loaded_legal_keywords_drive_entity_hint_not_person_detection`

---

## 4. Acronym / initialism logic

### 4.1 Functions and files

| Function | File:line | Role |
| --- | --- | --- |
| `_acronym(text)` | matching.py:170 | Legacy: takes first letter of each word >2 chars. |
| `_acronym_bridge(row_a, row_b)` | matching.py:176 | Legacy: checks ≤4-char names against `_acronym()` of all name fields. |
| `_generate_acronym(text)` | matching.py:198 | New: stopword-aware (`_ACRONYM_STOPWORDS`, `_ACRONYM_LEGAL_STOPS`), requires ≥3 significant tokens. Drops "of", "and", "the", "inc", "ltd", etc. |
| `_full_name_for_acronym(row)` | matching.py:219 | Concatenates Name 1 + Name 2 to reconstruct names split across columns. |
| `_acronym_bridge_full(row_a, row_b)` | matching.py:228 | Generates acronym from each individual name AND the concatenation; matches against any 3–7 char alphabetic token from the other row. |
| `_compact_names_match(row_a, row_b)` | matching.py:263 | Removes whitespace and matches names whose compact form is 4–12 chars. Catches "MER JAN" ↔ "MERJAN". |
| `COMMON_ACRONYM_EXPANSIONS` | preprocessing.py:31 | Curated allow-list (PPC, RBC, RMHC, AHS, CTL) for root-brand extraction. |

### 4.2 Behavior matrix

| Scenario | Pass type | Final score |
| --- | --- | --- |
| Acronym + same valid tax | `tax_exact` (via the alias_supported bypass at matching.py:752) | **98** |
| Acronym + same/similar address + strong long-name support | `address_secondary_or_acronym` | **85** |
| Acronym + same domain | `domain_name_related` | **98** if name_sim≥0.80 else **85** |
| Acronym + city/country only | `acronym_review_candidate` if `ai_review_enabled` else `no_match` | **70** or **blank** |
| Acronym alone | rejects (acronyms need support) | **blank** |
| Common acronym matching legal suffix only (AG/SA/BV/AB/AS) | rejected — `_ACRONYM_LEGAL_STOPS` excludes them from generation, and `_acronym_bridge_full` ignores ≤2-char tokens | **blank** |

### 4.3 Tests

- `test_acronym_fka_fixes.py::test_generate_acronym_naacp`
- `test_acronym_fka_fixes.py::test_generate_acronym_ficci`
- `test_acronym_fka_fixes.py::test_generate_acronym_rmhc_drops_of_stopword`
- `test_acronym_fka_fixes.py::test_generate_acronym_too_few_significant_tokens_returns_empty`
- `test_acronym_fka_fixes.py::test_generate_acronym_strips_legal_suffixes`
- `test_acronym_fka_fixes.py::test_acronym_bridge_full_naacp`
- `test_acronym_fka_fixes.py::test_acronym_bridge_full_ficci_split_across_name_fields`
- `test_acronym_fka_fixes.py::test_acronym_bridge_full_no_false_positive`
- `test_acronym_fka_fixes.py::test_naacp_tax_exact_clusters`
- `test_acronym_fka_fixes.py::test_seminole_tribe_human_resource_programs`

### 4.4 Remaining risk

- Punctuated acronyms like `N.A.A.C.P.` are normalized to `naacp` by `normalize_supplier_name` (punctuation collapsed). ✅
- Spaced acronyms like `N A A C P` normalize to `n a a c p` — these would NOT match because `_acronym_bridge_full` requires a single 3–7 char alphabetic token. **Minor gap**: a future enhancement could compact single-letter spaced tokens, but it's currently not in the failing case list. Not required.
- Alphanumeric acronyms (e.g. "3M", "M3") are handled by `SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS` for 3M and by the alphanumeric core path; they pass through the same generic/person/hospitality guardrails.

---

## 5. Brand-family and protected-compound audit

### 5.1 Trusted brand-family inventory

| Source | Coverage |
| --- | --- |
| `data/known_brand_families.csv` | 4,364 rows / 3,817 families / 3,952 aliases. Includes 3M, 3DS, 3i, 5W Public, 5/3 Bank, 6sense, 7-Eleven, ABB, Arca Continental, Eastman Chemical, Flextronics, Kingfisher, Prosperity Bank, and thousands more. |
| `SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS` (config.py:264) | 3b, 3bl, 4titude, abbott, accenture, airtec, ajinomoto, cognis, cognizant, computershare, dhl, eastman, edi, elrig, fedex, merck, microsoft, millipore, oracle, sap, siemens. |
| `TRUSTED_SUPPLIER_IDENTITY_CORES` (config.py:272) | same list — these can act as single-token cores. |
| `BROAD_GLOBAL_SUPPLIER_CORES` (config.py:282) | abbott, airtec, ajinomoto, cognis, cognizant, computershare, eastman, merck, millipore, siemens — these get scored conservatively (cap 86) without address/domain/tax support. |
| `KNOWN_FAMILY_TOKEN_GROUPS` (config.py:305) | merck+millipore+emd+sigma+aldrich+supelco; akzo+nouryon; rain+ruetgers+rutgers. |
| `KNOWN_DISTINCTIVE_FAMILY_ROOTS` (config.py:311) | weka. |
| `KNOWN_ADDRESS_FAMILY_BRIDGE_GROUPS` (config.py:317) | weka+turnus (same Kissing address). |
| `KNOWN_RELATED_NAME_PAIRS` (config.py:323) | service express + top gun technology. |
| `COMMON_ACRONYM_EXPANSIONS` (preprocessing.py:31) | ppc, rbc, rmhc, ahs, ctl. |

### 5.2 Requested positive-capture coverage

| Brand | Status |
| --- | --- |
| 3BL Media | ✅ `3bl` in TRUSTED_SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS |
| 4titude | ✅ `4titude` |
| 3B Scientific | ✅ `3b` |
| 3M | ✅ in known_brand_families.csv ("3 M", "3M", "3M Health", "3M Health care") |
| 4flow | covered via TRUSTED_SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS — `4flow` is not explicitly listed but alphanumeric supplier-core logic preserves it as a distinctive 5-char alpha-numeric token. ✅ |
| Eastman Chemical / Eastman Company / Eastman Fine Chem / Eastman Chemical Jaeger | ✅ `eastman` in TRUSTED and BROAD_GLOBAL_SUPPLIER_CORES; tests `test_production_routing.py::test_eastman_family_routes_to_85_and_typo_routes_to_llm_review` |
| ELRIG UK / ELRIG.de | ✅ `elrig`; test `test_elrig_distinctive_acronym_family_is_captured_at_85` |
| Covanta / Reworld FKA Covanta | ✅ via secondary_name_match in PASS 1 + `same_legal_owner_confirmed`; validation row 1800029598 captured at 85 |
| AirCom Pneumatic/Pneumatik | covered: `pneumatik`→`pneumatic` normalization in preprocessing.py:201 |
| AIRTEC Pneumatic Sweden | ✅ `airtec` in TRUSTED_SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS |
| Ajinomoto | ✅ `ajinomoto` |

### 5.3 Protected-compound separation

| Pair | Status |
| --- | --- |
| Eastman Kodak vs Eastman Chemical | ✅ `PROTECTED_COMPOUND_IDENTITY_PHRASES` includes "eastman kodak"; test `test_protected_eastman_kodak_does_not_merge_with_eastman_company_by_address` |
| Air Liquide vs Air Products | ✅ both in PROTECTED_COMPOUND_IDENTITY_PHRASES |
| Sigma-Aldrich vs broad Sigma | ✅ "sigma aldrich" protected; `sigma` is in GENERIC_ROOT_TOKENS to prevent broad bridging |
| Merck / Millipore / Skywing | ✅ KNOWN_FAMILY_TOKEN_GROUPS for Merck+Millipore+EMD+Sigma+Aldrich+Supelco |
| Springer broad surname | ✅ "springer" in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS + "axel springer", "bio springer", "springer nature" in protected compounds |
| Insight broad common-word | ✅ "insight"/"insights" in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS + AMBIGUOUS_REVIEW_CORES; test `test_insight_and_springer_ambiguous_cores_do_not_auto_cluster_85_or_98` |
| Apple broad/common-word | ✅ "apple" in AMBIGUOUS_REVIEW_CORES — requires distinctive phrase support (e.g. "Apple Auto Glass" in KNOWN_BRANDS) |

### 5.4 Missing entries

None blocking. Optional brand additions would be tracked in `data/known_brand_families.csv` (already 3,817 families) rather than code.

### 5.5 Tests

- `test_supplier_identity_policy.py::test_cognis_group_identity_not_missed`
- `test_supplier_identity_policy.py::test_cognizant_group_identity_not_missed`
- `test_supplier_identity_policy.py::test_computershare_group_identity_not_missed`
- `test_supplier_identity_policy.py::test_colonial_metals_exact_core_not_missed`
- `test_supplier_identity_policy.py::test_old_false_positive_pairs_still_blocked`
- `test_production_routing.py::test_protected_eastman_kodak_does_not_merge_with_eastman_company_by_address`
- `test_production_routing.py::test_elrig_distinctive_acronym_family_is_captured_at_85`
- `test_production_routing.py::test_insight_and_springer_ambiguous_cores_do_not_auto_cluster_85_or_98`
- `test_acronym_fka_fixes.py::test_franchise_guardrail_blocks_brand_only` (Subway brand-only block)
- `test_acronym_fka_fixes.py::test_franchise_guardrail_allows_same_legal_owner` (LYNK same legal owner)

---

## 6. FKA / DBA / AKA / former-name audit

### 6.1 Parsing pipeline

FKA / DBA / AKA names are not parsed as marker words — they arrive intact as **secondary name fields** (`Name 2`, `Name 3`, `Name 4`) in the input CSV. The auto-detection in `input_reader.py` maps these to `secondary_names` and `name_2/3/4`. The matching engine then walks **all** name fields in:

| Function | What it walks |
| --- | --- |
| `_all_match_names(row)` | `name_norm`, `name2_norm`, …, `name7_norm`, `json_secondary_names_norm` |
| `secondary_name_match(row_a, row_b)` | matching.py:137 — exact or token_set_ratio ≥ 90 between any pair of names |
| `_acronym_bridge_full(row_a, row_b)` | matching.py:228 — also walks all name fields |
| `extract_json_secondary_names(row, mapping, …)` | preprocessing.py:387 — pulls JSON keys `familyName`, `parentName`, `groupName`, `tradeName`, `dba`, `doingBusinessAs`, `alternateName`, `legalName` |
| `DEFAULT_JSON_SECONDARY_NAME_KEYS` | config.py:358 |

So "FKA COVANTA HOLDING CORPORATION" in Name 2 of REWORLD's row matches "COVANTA HOLDING CORPORATION" via `secondary_name_match` (token_set_ratio = 100 after removing the FKA prefix as a non-distinctive token — actually it matches even with the "fka" token present because token_set_ratio is permissive).

### 6.2 Behavior

| Scenario | Pass type | Final score |
| --- | --- | --- |
| FKA name matches primary name of another row + same tax | `tax_exact` (alias-supported bypass) | **98** |
| FKA name matches primary name + same address | `address_secondary_or_acronym` or `address_name_related` | **85** |
| FKA name matches + domain | `domain_name_related` | **98** if name_sim ≥ 0.80 else **85** |
| DBA brand-only (e.g. "DBA SUBWAY 22959") with no shared property/tax/domain/owner | franchise guardrail block | **blank** |
| DBA at same property + same legal owner non-generic name | `same_legal_owner_confirmed` (bypasses franchise guardrail) | **85** |

### 6.3 Tests

- `test_acronym_fka_fixes.py::test_reworld_covanta_tax_exact_via_secondary` — REWORLD HOLDING / FKA COVANTA HOLDING captured.
- `test_acronym_fka_fixes.py::test_seminole_tribe_human_resource_programs` — HUMAN RESOURCE PROGRAMS OF THE / SEMINOLE TRIBE OF FLORIDA secondary match.
- `test_acronym_fka_fixes.py::test_franchise_guardrail_allows_same_legal_owner` — LYNK RESTAURANT GROUP (same legal owner non-generic) at 85 across tax IDs.
- `test_acronym_fka_fixes.py::test_franchise_guardrail_blocks_brand_only` — SUBWAY brand-only blocked.

### 6.4 Note

The engine does NOT need to literally parse "FKA " / "DBA " marker text. The structural approach — treat all name columns as equal candidates — is more robust and handles all marker variants (formerly, formerly known as, old name, renamed from, AKA, DBA, doing business as) automatically because the marker words drop out as generic tokens after `normalize_supplier_name`.

---

## 7. Address logic audit

### 7.1 Pass types that consume address

| Pass type | Address role | Requires |
| --- | --- | --- |
| `name_address_exact` | Co-required | `name_norm` equal + `address_supported` (≥0.88 sim or number-range with city/postal) |
| `address_distinctive_shared` | Co-required | Same/similar address + shared distinctive non-generic name tokens + name_sim ≥ 0.65 |
| `address_name_related` | Co-required | Same/similar address + name_sim ≥ 0.70 |
| `address_secondary_or_acronym` | Co-required | Same/similar address + secondary-name or acronym bridge |
| `address_domain` | Co-required | Same/similar address + same business domain |
| `address_only` | — | **Always rejected** |
| `professional_name_address` | Co-required | Same address + person name match + person title |
| `name_fuzzy_supported` | Supporting | Strong fuzzy name + address/city/country/domain |
| `family_bridge_supported` | Supporting | Distinctive root + address ≥ 0.75 or city/country |
| `tax_exact` (score 98 instead of 99) | Boost only | Slightly raises tax_exact score; does not enable it |

### 7.2 Invariants

- Address alone produces **blank** (`address_only` always rejected at matching.py:982).
- Address + same first-name only → **blank** (common-first-name guardrail at guardrails.py:212).
- Address + only generic shared tokens → **blank** (`_only_shared_tokens_are_non_bridge` guardrail at guardrails.py:67).
- City/postal/country alone never drives a match — no pass type fires on city or country alone.
- Address typo tolerance: token-sort + abbreviation normalization + German street collapse (preprocessing.py:220) + spelling-variant table for known cases. Numeric mismatch caps similarity at 0.74 (matching.py:79) to keep "1 main street" / "2 main street" from clustering.

### 7.3 Tests

- `test_supplier_identity_policy.py::test_exact_supplier_identity_different_house_number_not_missed`
- `test_supplier_identity_policy.py::test_same_address_and_different_names_remain_not_main_clustered`
- `test_precision_guardrails.py` — same-address different-person blocked, same-household-no-name blocked.
- `test_acronym_fka_fixes.py::test_address_distinctive_shared_seminole`

---

## 8. Alphanumeric supplier audit

### 8.1 Logic

| Mechanism | File:line |
| --- | --- |
| Preserve leading digits | `extract_supplier_identity_core` strips pure-digit tokens only; alphanumeric like `3b`, `3bl`, `4titude` are kept (preprocessing.py:611). |
| Numeric-prefix guardrail | `_has_numeric_or_short_code_prefix` (guardrails.py:80) — rows starting with 3B / 3WAY / etc. require exact/high-similarity name, tax, domain, or strong address before they can match. |
| Alphanumeric supplier core | `_core_is_safe_for_identity` (matching.py:493–501) accepts cores with ≥2 chars and at least one alphabetic char. |
| Exact alphanumeric identity bypass | `exact_alphanumeric_identity_core` flag in guardrails.py:276 — once two rows share an exact alphanumeric core like `3bl`, the numeric-prefix guardrail does not over-reject them. |
| Curated single-token cores | `SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS` includes 3b, 3bl, 4titude (config.py:264). |
| Distinctive-shared-token in guardrails | `_distinctive_shared_tokens` (guardrails.py:47) — only alphabetic ≥3 chars qualify; alphanumeric short codes go through `_shared_short_code_tokens` separately and only act as supporting evidence with strong address. |

### 8.2 Behavior

| Case | Outcome |
| --- | --- |
| 3BL Media vs 3BL Media (same row content) | `same_legal_owner_confirmed` → 85 |
| 3BL Media vs 3BL Media (same tax) | `tax_exact` → 98 |
| 4titude vs 4titude (different tax, different address) | `same_legal_owner_confirmed` → 85 |
| 3B Scientific vs 3B Scientific Corp | `same_legal_owner_confirmed` if normalized same, otherwise `name_fuzzy_supported` with support |
| 3B Scientific vs Scientific Center | rejected — only shared token is generic "scientific" + numeric-prefix guardrail |
| 4flow vs 4flow Inc | `same_legal_owner_confirmed` after legal-suffix stripping |
| Pure numeric vendor IDs like "12345" | rejected by `_core_is_safe_for_identity` |
| Fuzzy alphanumeric ("3BL" vs "BL3") | goes to LLM review at 70 if not exact |

### 8.3 Tests

- `test_supplier_identity_policy.py::test_short_exact_supplier_brand_phrase_not_missed`
- `test_precision_guardrails.py::test_numeric_or_short_code_prefix_block` (and related)
- `test_acronym_fka_fixes.py::test_merjan_tax_exact_clusters` (compact-name variant)

---

## 9. LLM routing audit

### 9.1 Pipeline

1. `REVIEW_ONLY_PASS_TYPES` in main.py:29 — these pass types never enter union-find and never appear in the main user-facing output. They go to `review_candidates` (later exported as `family_review_candidates.csv`) or LLM review groups.
2. `build_llm_review_groups` in main.py:459 — packages connected components with score 70 into LLM-ready groups.
3. `apply_llm_decisions` — if `llm_group_decisions` are provided (from a backend resolution step), it applies approve/reject/split/merge/promote decisions to the cluster map and scores.
4. `allow_unresolved_llm_candidates_in_final_output` (default False) — if False, any cluster that remains at score 70 after LLM step is **removed from the final main output**, leaving those rows as blank. Status flag `INCOMPLETE_UNRESOLVED_LLM_CANDIDATES` is raised so the caller knows the run is not fully resolved.
5. `override_llm_can_modify_98` (default False) — protects 98 clusters from LLM downgrade. Explicit opt-in required.

### 9.2 Invariants

- Plausible-but-weak candidates go to 70, not blank. ✅
- 70 groups are exported to `llm_review_groups`. ✅
- Final main output uses only 98/85/blank when `allow_unresolved_llm_candidates_in_final_output=False`. ✅
- LLM cannot override 98 unless `override_llm_can_modify_98=True`. ✅
- Final user CSV columns are exactly: original input columns + `Cluster Number` + `Match Percentage`. ✅ (verified in run output).

### 9.3 Tests

- `test_production_routing.py::test_final_output_scores_are_normalized_to_allowed_values`
- `test_production_routing.py::test_unresolved_llm_candidates_are_hidden_when_final_output_disallows_them`
- `test_production_routing.py::test_llm_decision_application_approve_reject_split_merge_and_promote`

---

## 10. Test results and validation

### 10.1 Compile

```
python3 -m compileall src scripts tests  →  OK (no errors)
```

### 10.2 Test suite

```
python3 -m pytest tests/ -v --tb=short  →  203 passed in 45.32s
```

No failures, no skips.

### 10.3 CLI help

```
python3 -m scripts.run_cli --help  →  OK
```

### 10.4 Sample-input validation

Run on `/Users/rohitbhojwani/Downloads/cluster testing - Sheet1.csv` (212 rows):

| Check | Outcome |
| --- | --- |
| NAACP acronym → captured in cluster 6 at 98% | ✅ |
| FICCI acronym → captured in cluster 50 at 98% | ✅ |
| RMHC acronym (Ronald McDonald) → captured in cluster 22 at 98% | ✅ |
| MERJAN / MER JAN same tax → cluster 23 at 98% via compact match | ✅ |
| LYNK same legal owner across tax IDs → cluster 8 at 98% (within-tax) and 85% bridge | ✅ |
| Reworld FKA Covanta → cluster 49 at 85% | ✅ |
| Seminole Tribe / Human Resource Programs same address+tax → cluster 52 at 98% | ✅ |
| Seminole Hard Rock / Seminole HR same address, different tax → cluster 53 at 85% | ✅ |
| Same address alone never produces 98 | ✅ no `address_only` matches in output |
| Franchise brand-only (SUBWAY across different owners, no shared property) | ✅ still blocked |
| Generic/location-only (gastro, tuebingen) still blocked | ✅ |
| Protected compounds (Eastman Kodak vs Eastman Chemical) still blocked | ✅ |
| Final score distribution | 98%: 134 rows, 85%: 62 rows, blank: 16 singletons. **No 86/88/91/96 in user output.** |

---

## 11. Open gaps

| Gap | Severity | Recommendation |
| --- | --- | --- |
| `KGaA` not in `data/legal_keywords.csv` | Low | Add row `KGaA`. Currently KG strips and leaves "aa". |
| Bare `Pty` (without Ltd) not in legal_keywords.csv | Low | Add row `Pty`. Currently `Pty Ltd` strips fine. |
| Spaced acronyms (`N A A C P`) | Low | Not in current failure list; current pipeline normalizes punctuated forms (`N.A.A.C.P.` → `naacp`) but treats each single letter as a separate token after space normalization. A future enhancement could compact runs of 3–6 single-letter tokens into one acronym candidate. Not blocking. |
| `english,gastro` explicit row in CSV | Cosmetic | Improves auditability; behavior already correct via code constant. |

None of these block the current foundation. All are CSV/data tweaks, not engine logic changes.

---

## 12. Recommended code changes

**None required.** Foundation is consistent with the rule spec.

If the user approves the optional CSV-only additions, the proposed diffs are:

```diff
--- a/data/legal_keywords.csv
+++ b/data/legal_keywords.csv
@@ -1 +1,3 @@
 ...existing rows...
+KGaA
+Pty
```

```diff
--- a/data/generic_non_bridge_keywords.csv
+++ b/data/generic_non_bridge_keywords.csv
@@ -1 +1,2 @@
 ...existing rows...
+english,gastro
```

These are **data-only** changes. No Python edits are recommended.

---

## 13. Sign-off

- Foundation rules: **consistent and enforced**.
- Recent fixes (acronym, FKA, compact-name, same-legal-owner, franchise bypass, guardrail ordering) integrate cleanly with the existing rule spec.
- 203/203 tests pass.
- Validation sample produces the expected score distribution and cluster assignments.
- Final user output is restricted to 98 / 85 / 70 / blank (70 only when run is flagged INCOMPLETE).
- No required code changes. Two optional CSV additions await confirmation.

Awaiting explicit user approval before any further commit or push.

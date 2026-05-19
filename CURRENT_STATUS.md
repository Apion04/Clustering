# Current Working State

This package is a working-state / engineering-review version of the ORO supplier clustering engine. It is not a final production deployment package, and no new final ZIP has been created in this pass.

## 1. Current Project Status

The engine now clusters at supplier brand/group identity level for review-ready supplier cleanup, while keeping strict guardrails for generic, person, location-only, legal-suffix, and risky-token matches. Main cluster numbers can be assigned to clear supplier brand/group identities across legal forms, branches, addresses, and countries when the supplier core is distinctive and no hard guardrail rejects the pair.

Main output requirement is still enforced: original input columns plus only `Cluster Number` and `Match Percentage`.

## 2. Latest Implemented Features

- Review/family/alias/weak/LLM-suggested candidates bypass union-find and stay out of the main `Cluster Number` by default.
- Same exact normalized name without address/tax/domain support routes to `name_exact_review`, not auto-clustering.
- Priority-sorted `family_review_candidates.csv` with evidence flags, suggested action, LLM fields, and why the pair was not auto-clustered.
- Edge-level audit output with cluster root/number, row ids, supplier names, score, pass type, evidence JSON, guardrail flag, and cluster warnings.
- Input-specific IDF / rare-token table from normalized supplier names. Rare tokens are support/review-only and never create main clusters by themselves.
- Stricter rare-token filtering for generic tokens, legal suffixes, risky alias words, person/surname-risky tokens, and location terms.
- Backend OpenAI LLM orchestration is now wired for `disabled`, `mock`, `live`, and `batch` modes. The default configured OpenAI model is `gpt-5.5`, overridable by `OPENAI_MODEL` or CLI `--openai-model`.
- LLM queue generation writes queue breakdown, request JSONL, response JSONL, cost estimate, decision application report, conflict report, and unresolved exception report.
- Final clean output is regenerated after LLM decision application as `final_supplier_clustered.csv`.
- In-run rejected-pair memory for hard guardrail rejections.
- Configurable support-field handling for optional mapped fields such as `family_name`, `canonical_name`, `parent_name`, `normalized_supplier_name`, `website`, `domain`, `email_domain`, `OROVendorId`, `CompanyEntityId`, and `tax_id`.
- Support fields have explicit strengths: `same_entity_id`, `same_entity_name`, `family_or_parent`, `domain`, or `review_only`. Family/parent/canonical text is review-only by default unless explicitly configured as trusted same-entity evidence.
- Review output now includes `shared_support_field`, `support_field_value`, `support_field_strength`, `review_group_key`, and `why_not_auto_clustered` so related rows can be grouped and sorted for manual inspection without changing the main output.
- Existing precision fixes remain in place: strict individuals, generic-token suppression, numeric/short-code prefix guardrails, weak-chain protection, brand alias dictionary, legal suffix dictionary, and multilingual generic non-bridge dictionary.
- Added `distinctive_supplier_identity` for clear same supplier brand/group cores across legal forms/countries/addresses. Broad brand/group identity defaults to 80-89 unless same/similar address, valid tax/entity ID, business domain, or exact duplicate/local-entity evidence supports a higher score.
- Stabilized case-insensitive alphanumeric supplier cores such as `3BL`, `4titude`, and `3B Scientific` so they are not dropped as numeric/reference IDs. Pure numeric and operational/reference IDs remain blocked.
- Low-similarity exact-tax overlaps now route to Review (`tax_exact_low_similarity_review`) instead of main-clustering unrelated names that share a suspicious/reused tax value.
- CSV input columns are read as text so original client values such as `FALSE` and leading-zero postal codes round-trip unchanged in the main output.

## 3. Current Dictionaries Integrated

- `data/known_brand_families.csv`
- `data/legal_keywords.csv`
- `data/generic_non_bridge_keywords.csv`

These remain separate:

- Brand aliases can create controlled family/review candidates at phrase level.
- Legal suffixes only normalize names and help entity detection.
- Generic words only suppress weak evidence and never create clusters.

## 4. Latest Test Result

- Compile check: passed with `env PYTHONPYCACHEPREFIX=/private/tmp/oro_pycache python -m compileall src scripts tests`.
- Unit/integration tests: `155 passed`.

## 5. Latest Validation Result

Latest deterministic CLI validation used the real 74,355-row supplier file with LLM disabled:

`/Users/rohitbhojwani/Downloads/tesfile_onelakh.csv`

| Rows | Runtime | Candidate Pairs | Edges | Main Clusters | Review Candidate Pairs | Largest Cluster | Candidate Cap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 74,355 | 106.66s | 847,667 | 60,421 | 8,654 | 178,488 | 604 | false |

Validation checks:

- Rows preserved: yes.
- ExternalId identity set preserved: yes.
- Output columns correct: yes.
- Anchor first occurrence preserved: yes.
- Cluster rows contiguous: yes.
- Singleton relative order preserved: yes.
- Review file priority sorted: yes.
- Main output has no debug columns: yes.
- New supplier identity examples were handled: 3BL Media, 4titude, 3B Scientific, AirCom Pneumatic/Pneumatik, AIRTEC Pneumatic, Ajinomoto, Air Liquide, Air Products, Cognis, Cognizant, Computershare, Colonial Metals, A. Hartrodt, ADEKA, Advance Research Chemicals, Aescolab, and AGA Gas are no longer silently missed. Near-spelling/ambiguous examples such as Abu-Ghazaleh/Ghazahleh route to Review unless exact-core evidence is strong.
- Old bad examples remained fixed: MUELLER/Müller Processing, Cindy, Private, Partner, Industrie, Product, Biological, Medizin/Medicine, Excellence, Europa, Alfa, Meta/Qingdao, ADM, Zimmer, 4Tune/ValGenesis, ABCR/Guntermann, AC medienhaus/Druckerei, IMEC/Tempmate, and individual same-address household cases do not share main Cluster Number.
- Score distribution on the latest full run: 5,167 clusters at 95-100, 343 at 90-94, 2,781 at 80-89, 363 at 70-79, and 49,655 blank/singleton rows.
- `distinctive_supplier_identity` distribution: 2,730 clusters at 80-89 and 48 at 70-79; none are 90+.
- 90+ multi-country clusters without tax/domain/address support: 0.
- AI/human review cluster export rows: 5,401.

## 6. Known Remaining Concerns

- Rare-token support is intentionally conservative and review-only; it should be monitored on broad 100k-200k files for review queue volume.
- Large brand/group clusters such as Merck/Millipore/Sigma/Aldrich, Siemens, Evonik, Schenker, and Kuehne + Nagel now appear as low/medium confidence main review clusters under the updated brand/group identity policy. They are review-ready, not auto-approval-ready.
- `family_review_candidates.csv` can be large on brand-heavy files and needs downstream workflow review.
- Candidate caps and block-size caps may reduce recall on very broad files.
- Global address parsing is heuristic.
- LLM live mode still needs privacy review, target OpenAI model/cost settings, and rate-limit testing on the deployment environment before production use. Mock mode exercises the full decision application path without an API key.
- Output is review-ready, not auto-approval-ready.

## 7. Final ZIP Status

No final production ZIP was created in this pass. An engineering-review ZIP may be created for handoff, but it must not be labeled production-ready.

## 8. Recommended Next Step

Review `output/oro_safe_75k_identity_verification.md` and `output/oro_safe_75k_ai_review_clusters.csv`, especially large low-confidence brand/group clusters and all 70-79 clusters, then decide whether any brand/group families should be demoted to Review-only before production hardening.

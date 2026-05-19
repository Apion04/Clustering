# Engineer Handoff Notes

This package is a reference implementation, not final production code. It combines Kimi's project structure with ORO-specific rule fixes found during real-file testing.

## What to review first

1. `src/preprocessing.py` for normalization, multiple tax fields, and JSON tax extraction.
2. `src/matching.py` for name/address/tax/domain/acronym/family matching.
3. `src/merging.py` for Union-Find and parent/family tax conflict handling.
4. `src/ai_review.py` for Kimi/OpenAI-compatible LLM review.
5. `src/sorting.py` and `src/output.py` for final output format.

## Main product rule

The main output file must keep all original columns and add only:

- `Cluster Number`
- `Match Percentage`

No extra technical/debug columns in the reviewer output.

## Suggested first engineering tests

Run this on the real 6,391-row test file and verify these cases:

1. PPC / Potasse / VYNOVA PPC cluster together.
2. ELEKTROPHYSIK / ELEKTRO-PHYSIK cluster together.
3. NAEGELE / Nägele Waagenservice cluster together.
4. Rain Carbon / Rütgers / Ruetgers family bridge clusters together in parent-level mode.

## Important config

For Rohit's current requirement, parent/family-level clustering is expected:

```env
ALLOW_PARENT_FAMILY_TAX_CONFLICTS=true
```

If the business later wants strict legal entity clustering, set it to false and adjust thresholds.

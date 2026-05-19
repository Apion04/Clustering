# Architecture Documentation

## Design Principles

1. **Minimal output**: Only 2 columns added to client file
2. **Explainable**: Every cluster has a reason (in audit file)
3. **Safe**: Chain-merge protection prevents false positives
4. **Fast**: Blocking avoids O(n²) comparisons
5. **Modular**: Each pass is independent and testable

## Data Flow

```
Input CSV (100K rows)
    ↓
Preprocessing (normalize all fields)
    ↓
Blocking (generate 9 blocking keys per row)
    ↓
Candidate Pairs (~2M pairs via DuckDB join)
    ↓
6-Pass Matching (evaluate each pair)
    ↓
Union-Find Merge (with chain protection)
    ↓
Score Calculation (weakest-link principle)
    ↓
Anchor Sort & Output (first occurrence stays, later matches move below it)
```

## Blocking Keys

1. `TAX_<tax_id>` — Exact tax ID
2. `N4_<name_prefix>` — First 4 chars of normalized name
3. `N5_<name_prefix>` — First 5 chars
4. `PHO_<metaphone>` — Phonetic fingerprint
5. `TOK_<token_sort>` — Sorted tokens
6. `DOM_<domain>` — Business domain
7. `ADR_<address_hash>` — Normalized address
8. `PC_<postal>_<country>` — Postal + country
9. `CC_<city>_<country>` — City + country
10. `ROOT_<brand>` — Root brand name

## Chain Merge Protection

Problem: A matches B (tax), B matches C (address), C matches D (name fuzzy). Should A-D be one cluster?

Solution:
- Track all edges in each cluster
- Before merging via weak edge, check if there's strong evidence between the two clusters
- If no strong evidence exists and cluster size > 3, block the merge
- Tax conflicts always block merges

## Match Percentage Calculation

Weakest-link principle: Cluster match % = minimum edge match % in the cluster.

This ensures that if one edge in a cluster is weak (e.g., address-only), the entire cluster is flagged with low confidence.

## AI Risk Module (Phase 3)

Trigger conditions:
- Address-only match with different names
- Domain match with low name similarity
- Cross-country family detection
- Possible DBA/acquisition patterns

AI returns: verdict, confidence, reason, relationship_type.
Does NOT appear in main output.

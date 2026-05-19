# ORO Real-File Rule Additions

This version merges Kimi's engineering structure with business rules found from Rohit's real supplier clustering test files.

## Must-have additions over generic clustering

1. **Multiple tax fields**
   - Support `vatid`, `taxid`, PAN, GST, EIN, and any mapped tax columns.
   - `tax_norm` may contain multiple pipe-separated normalized IDs.

2. **Tax/VAT from JSON metadata**
   - Extract `vatNumber`, `taxNumber`, `vat`, `tax`, `ein`, `gst`, `pan` from metadata JSON columns.
   - Real files contain metadata like `{ "vatNumber": "DE122812081", "taxNumber": "" }`.

3. **German address normalization**
   - `PASTEURSTR. 15` = `PASTEUR-STR. 15`
   - `BUCHENBACHSTR. 13` = `Buchenbachstraße 13`
   - `Kekulstrasse 30` ≈ `Kekulestraße 30`

4. **City cleanup**
   - `KOELN 60` = `KOELN`
   - `THANNXXXXX` = `THANN`
   - `WüSTENROT-NEULAUTER` ≈ `Wüstenrot-Neulautern`

5. **Germanic/accent name bridge**
   - `Rütgers` = `Ruetgers`
   - `Nägele` = `Naegele`

6. **Acronym and secondary-name bridge**
   - `PPC` can connect to `Potasse Et Produits Chimiques` / `VYNOVA PPC` when VAT/address/family evidence supports it.

7. **Parent/family bridge**
   - Parent-level clustering is required, not only strict legal entity matching.
   - Example: `Rain Carbon Germany GmbH`, `Rütgers Aromatic Chemicals GmbH`, `RUETGERS Chemicals AG`, `Ruetgers Chemicals AG` should become a family-level cluster.

8. **Tax conflict handling**
   - Strict legal entity mode should block different tax IDs.
   - Parent/family mode can allow selected family/acronym/domain bridge edges with different tax IDs.
   - Config: `ALLOW_PARENT_FAMILY_TAX_CONFLICTS=true`.

9. **Output rule**
   - Main reviewer output must only add `Cluster Number` and `Match Percentage` to original client columns.
   - All reasons/sources/risk flags belong in optional audit files only.

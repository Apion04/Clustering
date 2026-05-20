#!/usr/bin/env python3
"""
Phase 3: Classify all 273 blank rows from Phase 2 output.
Generates:
  /private/tmp/phase3_blank_row_classification.csv
  /private/tmp/phase3_blank_row_classification.md
"""

import pandas as pd
import re
from collections import defaultdict

# ── Input files ──────────────────────────────────────────────────────────────
PHASE2_CSV   = "/private/tmp/phase2_output.csv"
REFERENCE_CSV = "/Users/rohitbhojwani/Downloads/all_raw_missed_examples_combined_v2_reference.csv"
OUT_CSV      = "/private/tmp/phase3_blank_row_classification.csv"
OUT_MD       = "/private/tmp/phase3_blank_row_classification.md"

# ── Load data ─────────────────────────────────────────────────────────────────
p2  = pd.read_csv(PHASE2_CSV)
ref = pd.read_csv(REFERENCE_CSV)

blank = p2[p2["Match Percentage"].isna() | (p2["Match Percentage"].astype(str).str.strip() == "")].copy()
assert len(blank) == 273, f"Expected 273 blank rows, got {len(blank)}"

# Build a lookup from reference: (source_file, source_row) → (cluster, match%)
ref_lookup = {}
for _, r in ref.iterrows():
    key = (r["Source File"], int(r["Source Row"]))
    ref_lookup[key] = {
        "ref_cluster": r["Original Cluster Number"],
        "ref_match":   r["Original Match Percentage"],
    }

# ── Classification table ──────────────────────────────────────────────────────
# Each entry: (source_file, source_row, classification, fix_type, notes, target_cluster)
# We'll build this row-by-row using the patterns described in the task spec.

ROWS = []   # list of dicts

def add(sf, sr, classification, fix_type, notes, target_cluster=""):
    ROWS.append({
        "source_file":      sf,
        "source_row":       sr,
        "classification":   classification,
        "fix_type":         fix_type,
        "notes":            notes,
        "target_cluster":   target_cluster,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# F1 — 179 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F1_first_missed_file"

# Row 1-2: (JU) THE BOX - KRW / (JU)THE BOX — same city/postal, "KRW" is internal
add(SF,  1, "should_be_70", "name_noise_strip",
    "(JU) THE BOX - KRW vs (JU)THE BOX — same city/postal Icheon-si KR 17384; 'KRW' likely internal cost-centre suffix; moderate confidence due to different email domains")
add(SF,  2, "should_be_70", "name_noise_strip",
    "(JU)THE BOX — see row 1; slightly different email; classify 70 rather than 98 due to uncertainty about KRW suffix meaning")

# Row 3-4: A + A / A+A — punctuation only
add(SF,  3, "should_be_98", "name_noise_strip",
    "A + A vs A+A — identical name, punctuation/spacing variant only")
add(SF,  4, "should_be_98", "name_noise_strip",
    "A+A vs A + A — identical name, punctuation/spacing variant only")

# Row 5-6: AC COMPACTING EQUIPMENT LLC / AC COMPACTING LLC — descriptor token
add(SF,  5, "should_be_85", "pass_3c_extension",
    "AC COMPACTING EQUIPMENT LLC vs AC COMPACTING LLC — same core brand, 'EQUIPMENT' descriptor token")
add(SF,  6, "should_be_85", "pass_3c_extension",
    "AC COMPACTING LLC vs AC COMPACTING EQUIPMENT LLC — same core brand, 'EQUIPMENT' descriptor token")

# Row 9: ACTEON CONSULTANCY LLP, THE / THE ACTEON CONSULTANCY LLP (row 201)
add(SF,  9, "should_be_98", "name_noise_strip",
    "ACTEON CONSULTANCY LLP, THE vs THE ACTEON CONSULTANCY LLP (row 201) — article inversion, same entity")

# Row 10: Acumen International Media LTD — domain acumenmedia.com (also row 200 TBD Media Group)
add(SF, 10, "should_be_85", "same_domain_bug",
    "Acumen International Media LTD and TBD Media Group LTD (row 200) both use acumenmedia.com; same-domain pair unresolved by engine")

# Row 11-12: ACUMEN PUBLIC AFFAIRS SPRL / SRL — legal suffix variant
add(SF, 11, "should_be_98", "name_noise_strip",
    "ACUMEN PUBLIC AFFAIRS SPRL vs ACUMEN PUBLIC AFFAIRS SRL — same name, SPRL/SRL legal suffix variant (Belgian vs Italian form)")
add(SF, 12, "should_be_98", "name_noise_strip",
    "ACUMEN PUBLIC AFFAIRS SRL vs ACUMEN PUBLIC AFFAIRS SPRL — legal suffix variant")

# Row 13-14: ADLER-MOORE, JILL PHD. — exact duplicate
add(SF, 13, "should_be_98", "correctly_blank",
    "ADLER-MOORE, JILL PHD. — exact duplicate of row 14")
add(SF, 14, "should_be_98", "correctly_blank",
    "ADLER-MOORE, JILL PHD. — exact duplicate of row 13")

# Row 15: ADLER, ELKE — singleton
add(SF, 15, "correctly_blank", "correctly_blank",
    "ADLER, ELKE — singleton individual name, no evident pair in dataset")

# Row 16: ADVANCED CHEMICAL TRANSPORT INC — singleton
add(SF, 16, "correctly_blank", "correctly_blank",
    "ADVANCED CHEMICAL TRANSPORT INC — singleton, no evident pair")

# Row 17-18: AFRIC PHAR / AfricPhar Morocco
add(SF, 17, "should_be_70", "name_similarity_gap",
    "AFRIC PHAR vs AfricPhar Morocco — same core brand, 'Morocco' country append, different tokenisation; plausible but uncertain")
add(SF, 18, "should_be_70", "name_similarity_gap",
    "AfricPhar Morocco vs AFRIC PHAR — country suffix appended, below similarity threshold")

# Row 19: Aidshilfe Heidelberg — singleton
add(SF, 19, "correctly_blank", "correctly_blank",
    "Aidshilfe Heidelberg — singleton local charity, no evident pair")

# Row 20-21: AIRPLUS / AIRPLUS AIR TRAVEL CARD — descriptor token
add(SF, 20, "should_be_85", "pass_3c_extension",
    "AIRPLUS vs AIRPLUS AIR TRAVEL CARD — same brand, descriptor 'AIR TRAVEL CARD' suffix")
add(SF, 21, "should_be_85", "pass_3c_extension",
    "AIRPLUS AIR TRAVEL CARD vs AIRPLUS — descriptor extension unmatched")

# Row 22-23: AIXIAL / Aixial US Inc — country branch
add(SF, 22, "should_be_85", "pass_3c_extension",
    "AIXIAL vs Aixial US Inc — parent vs US subsidiary")
add(SF, 23, "should_be_85", "pass_3c_extension",
    "Aixial US Inc vs AIXIAL — US branch of AIXIAL")

# Row 24: Alexander Hahne → also row 173 HAHNE, ALEXANDER (name inversion)
add(SF, 24, "should_be_85", "name_noise_strip",
    "Alexander Hahne vs HAHNE, ALEXANDER (row 173) — inverted surname/forename format")

# Row 27-28: ALLMAN COMMUNCATION LTD / ALLMAN COMMUNICATION LTD — typo
add(SF, 27, "should_be_98", "name_noise_strip",
    "ALLMAN COMMUNCATION LTD — typo 'COMMUNCATION' vs COMMUNICATION (row 28)")
add(SF, 28, "should_be_98", "name_noise_strip",
    "ALLMAN COMMUNICATION LTD vs ALLMAN COMMUNCATION LTD (row 27) — single-letter typo")

# Row 29-30: ALUM A LIFT — exact duplicate
add(SF, 29, "should_be_98", "correctly_blank",
    "ALUM A LIFT — exact duplicate of row 30")
add(SF, 30, "should_be_98", "correctly_blank",
    "ALUM A LIFT — exact duplicate of row 29")

# Row 31: AMBROSETTI GROUP LIMITED — also row 202 THE EUROPEAN HOUSE – AMBROSETTI
add(SF, 31, "should_be_85", "name_similarity_gap",
    "AMBROSETTI GROUP LIMITED vs THE EUROPEAN HOUSE – AMBROSETTI (row 202) — Ambrosetti is the consultancy brand; same entity trading under two names")

# Row 32: AMERICAN GASTROENTEROLOGICAL — singleton (no pair visible)
add(SF, 32, "correctly_blank", "correctly_blank",
    "AMERICAN GASTROENTEROLOGICAL — singleton (truncated name), no evident pair; AGA is a known org but no pairing row in dataset")

# Row 35-36: ANGH / ANGH CONGRES — same domain angh.net
add(SF, 35, "should_be_85", "same_domain_bug",
    "ANGH (ASS NATION DES) and ANGH CONGRES (row 36) share domain angh.net; congress arm of same association")
add(SF, 36, "should_be_85", "same_domain_bug",
    "ANGH CONGRES vs ANGH — same domain angh.net, congress subsidiary")

# Row 37: ANLAIDS ONLUS — singleton
add(SF, 37, "correctly_blank", "correctly_blank",
    "ANLAIDS ONLUS — Italian HIV/AIDS association singleton, no evident pair")

# Row 41-43: AOK HESSEN / AOK NIEDERSACHSEN / AOK NORDOST — AOK regional branches
add(SF, 41, "should_be_85", "pass_3c_extension",
    "AOK HESSEN — regional branch of AOK German health insurance; should cluster with other AOK branches (Niedersachsen row 42, Nordost row 43)")
add(SF, 42, "should_be_85", "pass_3c_extension",
    "AOK NIEDERSACHSEN — AOK regional branch; see row 41")
add(SF, 43, "should_be_85", "pass_3c_extension",
    "AOK NORDOST — AOK regional branch; see row 41")

# Row 44-45: APLUSA / APLUSA BELL FALLA LLC — same brand
add(SF, 44, "should_be_85", "pass_3c_extension",
    "APLUSA vs APLUSA BELL FALLA LLC — same core brand, additional descriptor tokens")
add(SF, 45, "should_be_85", "pass_3c_extension",
    "APLUSA BELL FALLA LLC vs APLUSA — descriptor extension unmatched by engine")

# Row 46-47: APRIIL CONGRESS AS / Apriil Event & Congress AS
add(SF, 46, "should_be_85", "pass_3c_extension",
    "APRIIL CONGRESS AS vs Apriil Event & Congress AS — same brand, 'Event &' descriptor insertion")
add(SF, 47, "should_be_85", "pass_3c_extension",
    "Apriil Event & Congress AS vs APRIIL CONGRESS AS — descriptor tokens differ")

# Row 48-49: ARBON EQUIPMENT CORP / ARBON EQUIPMENT CORP (RITE HITE — parenthetical note
add(SF, 48, "should_be_98", "name_noise_strip",
    "ARBON EQUIPMENT CORP vs ARBON EQUIPMENT CORP (RITE HITE — exact + parenthetical acquisition note; same entity")
add(SF, 49, "should_be_98", "name_noise_strip",
    "ARBON EQUIPMENT CORP (RITE HITE — parenthetical Rite-Hite note; strip to match row 48")

# Row 50: ARKANSAS DHS-PHARMACY REBATE — singleton government entity
add(SF, 50, "correctly_blank", "correctly_blank",
    "ARKANSAS DHS-PHARMACY REBATE — government rebate account singleton, no evident pair")

# Row 51: ÄRZTE KRONE VERLAGSGES.M.B.H. — singleton
add(SF, 51, "correctly_blank", "correctly_blank",
    "ÄRZTE KRONE VERLAGSGES.M.B.H. — Austrian medical publisher singleton")

# Row 52-53: AT LIMITS LTD / AT THE LIMITS LTD — word insertion
add(SF, 52, "should_be_98", "name_noise_strip",
    "AT LIMITS LTD vs AT THE LIMITS LTD — 'THE' article insertion, same entity; both use linkedin.com domain (non-specific)")
add(SF, 53, "should_be_98", "name_noise_strip",
    "AT THE LIMITS LTD vs AT LIMITS LTD — article variant")

# Row 54-55: ATHLÉTISME SAINT PANTALEON DE / ATHLETISME ST PANTALEON DE LARCHE
add(SF, 54, "should_be_98", "name_noise_strip",
    "ATHLÉTISME SAINT PANTALEON DE vs ATHLETISME ST PANTALEON DE LARCHE — accent strip + SAINT/ST abbreviation")
add(SF, 55, "should_be_98", "name_noise_strip",
    "ATHLETISME ST PANTALEON DE LARCHE vs ATHLÉTISME SAINT PANTALEON — same entity, abbreviated form")

# Row 56-57: AUSTRALASIAN SOCIETY FOR / AUSTRALASIAN SOCIETY FOR HIV, VIRAL
add(SF, 56, "should_be_98", "name_noise_strip",
    "AUSTRALASIAN SOCIETY FOR (truncated) vs AUSTRALASIAN SOCIETY FOR HIV, VIRAL (row 57) — truncation variant; domain www.ashm.org.au confirms ASHM identity")
add(SF, 57, "should_be_98", "name_noise_strip",
    "AUSTRALASIAN SOCIETY FOR HIV, VIRAL — full name; row 56 is truncated form")

# Row 58: Axian Consulting Limited — singleton
add(SF, 58, "correctly_blank", "correctly_blank",
    "Axian Consulting Limited — singleton consulting firm, no evident pair")

# Row 59-60: BAILEY HOUSE INC — one has housingworks domain, one blank; possible merge/rename
add(SF, 59, "should_be_85", "same_domain_bug",
    "BAILEY HOUSE INC (row 59, domain housingworks) vs BAILEY HOUSE INC (row 60, no domain) — same name; Bailey House merged into Housing Works; should cluster")
add(SF, 60, "should_be_85", "same_domain_bug",
    "BAILEY HOUSE INC (no domain) vs row 59 — same name, different domain metadata")

# Row 63-64: BARRINGTON JAMES LIMITED / Barrington James OSP Ltd — subsidiary
add(SF, 63, "should_be_85", "pass_3c_extension",
    "BARRINGTON JAMES LIMITED vs Barrington James OSP Ltd (row 64) — parent vs OSP subsidiary")
add(SF, 64, "should_be_85", "pass_3c_extension",
    "Barrington James OSP Ltd — OSP subsidiary of Barrington James Limited")

# Row 67: BEN ALLEN FOR SENATE 2022 — singleton political committee
add(SF, 67, "correctly_blank", "correctly_blank",
    "BEN ALLEN FOR SENATE 2022 — singleton political campaign committee")

# Row 68-69: BFARM / BFARM, DIENSTSITZ KÖLN — same org, branch location
add(SF, 68, "should_be_85", "pass_3c_extension",
    "BFARM vs BFARM, DIENSTSITZ KÖLN (row 69) — same German federal agency (BfArM), Köln branch")
add(SF, 69, "should_be_85", "pass_3c_extension",
    "BFARM, DIENSTSITZ KÖLN — Cologne branch of BfArM; see row 68")

# Row 70: BIA UK BioIndustry Association — singleton (UK specific branch)
add(SF, 70, "correctly_blank", "correctly_blank",
    "BIA UK BioIndustry Association — singleton; no other BIA row visible in dataset")

# Row 71-72: BIOCYTOGEN BOSTON CORP / Biocytogen Pharmaceuticals (Beijing — parent/subsidiary
add(SF, 71, "should_be_85", "pass_3c_extension",
    "BIOCYTOGEN BOSTON CORP vs Biocytogen Pharmaceuticals (Beijing) (row 72) — US and China entities of Biocytogen group")
add(SF, 72, "should_be_85", "pass_3c_extension",
    "Biocytogen Pharmaceuticals (Beijing) — China entity of Biocytogen; domain biocytogen.com links to row 71")

# Row 73: BIOINDUSTRY ASSOCIATION — same as BIA UK (row 70)? — cluster them
add(SF, 73, "should_be_85", "pass_3c_extension",
    "BIOINDUSTRY ASSOCIATION vs BIA UK BioIndustry Association (row 70) — full name vs acronym+country form of the same UK trade body")

# Row 74-75: BIPARTISAN POLICY CENTER ACTION / Bipartisan Policy Center, Inc.
add(SF, 74, "should_be_85", "pass_3c_extension",
    "BIPARTISAN POLICY CENTER ACTION vs Bipartisan Policy Center, Inc. (row 75) — political action arm vs main nonprofit; same organisation family")
add(SF, 75, "should_be_85", "pass_3c_extension",
    "Bipartisan Policy Center, Inc. — main entity; row 74 is its PAC arm")

# Row 76-77: BLITZKURIER Botendienste GmbH / BLITZKURIER FUNKBOTENDIENSTE GMBH
add(SF, 76, "should_be_85", "pass_3c_extension",
    "BLITZKURIER Botendienste GmbH vs BLITZKURIER FUNKBOTENDIENSTE GMBH — same Blitzkurier brand, different service-line descriptors")
add(SF, 77, "should_be_85", "pass_3c_extension",
    "BLITZKURIER FUNKBOTENDIENSTE GMBH — see row 76")

# Row 78: BN BUILDERS, INC. — singleton
add(SF, 78, "correctly_blank", "correctly_blank",
    "BN BUILDERS, INC. — singleton construction company, no evident pair")

# Row 79: BOCSCI Inc — singleton
add(SF, 79, "correctly_blank", "correctly_blank",
    "BOCSCI Inc — singleton chemical supplier, no evident pair")

# Row 80-81: BOWERS, LISA — exact duplicate
add(SF, 80, "should_be_98", "correctly_blank",
    "BOWERS, LISA — exact duplicate of row 81")
add(SF, 81, "should_be_98", "correctly_blank",
    "BOWERS, LISA — exact duplicate of row 80")

# Row 82-83: BREAKWELL RANDAL I/O LTD / BREAKWELL RANDAL lO LTD — I/O vs lO (font confusion)
add(SF, 82, "should_be_98", "name_noise_strip",
    "BREAKWELL RANDAL I/O LTD vs BREAKWELL RANDAL lO LTD — uppercase I/O vs lowercase l+O OCR/font confusion")
add(SF, 83, "should_be_98", "name_noise_strip",
    "BREAKWELL RANDAL lO LTD — lowercase 'l' vs 'I' confusion; see row 82")

# Row 86: Brisa Biomedical SL — singleton
add(SF, 86, "correctly_blank", "correctly_blank",
    "Brisa Biomedical SL — singleton biomedical company, no evident pair")

# Row 87-88: BS AND B INVENTORY MANAGEMENT / BS&B SAFETY SYSTEMS LLC — different entities sharing BS&B brand
add(SF, 87, "should_be_70", "name_similarity_gap",
    "BS AND B INVENTORY MANAGEMENT vs BS&B SAFETY SYSTEMS LLC — different divisions of BS&B brand; inventory management vs safety systems; plausible cluster but distinct entities")
add(SF, 88, "should_be_70", "name_similarity_gap",
    "BS&B SAFETY SYSTEMS LLC vs BS AND B INVENTORY MANAGEMENT — see row 87")

# Row 89: BTS IN LONDON LTD — also row 91 BTS Japan K.K. — BTS brand family
add(SF, 89, "should_be_85", "pass_3c_extension",
    "BTS IN LONDON LTD — part of BTS group (BTS Japan K.K. row 91); country/region descriptor")
add(SF, 91, "should_be_85", "pass_3c_extension",
    "BTS Japan K.K. — BTS group Japan entity; see row 89 BTS IN LONDON LTD")

# Row 92: C.E.G.O.S ITALIA S.P.A. CENTRO — likely CEGOS group; also row 108 CEGOS SA
add(SF, 92, "should_be_85", "pass_3c_extension",
    "C.E.G.O.S ITALIA S.P.A. CENTRO vs CEGOS SA (row 108) — CEGOS group, Italian entity vs parent; dot-expanded acronym")
add(SF, 108, "should_be_85", "pass_3c_extension",
    "CEGOS SA vs C.E.G.O.S ITALIA (row 92) — parent of CEGOS group")

# Row 93-94: C9 MEDIA LIMITED / C9 Media Limited - SGD — currency/billing suffix
add(SF, 93, "should_be_98", "name_noise_strip",
    "C9 MEDIA LIMITED vs C9 Media Limited - SGD (row 94) — '- SGD' is billing currency suffix; same entity")
add(SF, 94, "should_be_98", "name_noise_strip",
    "C9 Media Limited - SGD — SGD billing suffix; see row 93")

# Row 97: CANCERODIGEST — singleton
add(SF, 97, "correctly_blank", "correctly_blank",
    "CANCERODIGEST — singleton medical journal/digest, no evident pair")

# Row 98-99: CAPAHC / CAPAHC AHC — acronym + expansion
add(SF, 98, "should_be_85", "acronym_expansion",
    "CAPAHC vs CAPAHC AHC (row 99) — second row appends 'AHC' expansion; same organisation")
add(SF, 99, "should_be_85", "acronym_expansion",
    "CAPAHC AHC — AHC expansion appended; see row 98")

# Row 100-101: CAPITAL PUBLISHING, INC — exact duplicate
add(SF, 100, "should_be_98", "correctly_blank",
    "CAPITAL PUBLISHING, INC — exact duplicate of row 101")
add(SF, 101, "should_be_98", "correctly_blank",
    "CAPITAL PUBLISHING, INC — exact duplicate of row 100")

# Row 102-103: CARDINUS LLC / CARDINUS RISK MANAGEMENT LTD — same brand, different legal entities
add(SF, 102, "should_be_70", "name_similarity_gap",
    "CARDINUS LLC vs CARDINUS RISK MANAGEMENT LTD — same Cardinus brand; LLC (US) vs Ltd (UK) may be distinct entities or same company's offices")
add(SF, 103, "should_be_70", "name_similarity_gap",
    "CARDINUS RISK MANAGEMENT LTD — see row 102; name similarity gap due to 'RISK MANAGEMENT' tokens")

# Row 104: CARLOS ARMIÑANZAS CASTILLO — singleton individual
add(SF, 104, "correctly_blank", "correctly_blank",
    "CARLOS ARMIÑANZAS CASTILLO — singleton individual, no evident pair")

# Row 107: CCM INTERNATIONAL ΕΠΕ — also row 171 GROUP CCM S.A.L — CCM brand
add(SF, 107, "should_be_70", "needs_llm",
    "CCM INTERNATIONAL ΕΠΕ (Greek entity) vs GROUP CCM S.A.L (row 171, Lebanese entity) — CCM brand shared but different countries and legal names; needs LLM review")
add(SF, 171, "should_be_70", "needs_llm",
    "GROUP CCM S.A.L — see row 107 CCM INTERNATIONAL ΕΠΕ; LLM needed to confirm same group")

# Row 109-110: CELLCARTA FREMONT LLC / CellCarta NV — parent and subsidiary
add(SF, 109, "should_be_85", "pass_3c_extension",
    "CELLCARTA FREMONT LLC (US) vs CellCarta NV (Belgium) — Fremont US entity of CellCarta group")
add(SF, 110, "should_be_85", "pass_3c_extension",
    "CellCarta NV — Belgian entity of CellCarta group; see row 109")

# Row 111-112: Centrum Kształcenia IDEA / Centrum Kształcenia IDEA Sp. z o.o.
add(SF, 111, "should_be_85", "name_noise_strip",
    "Centrum Kształcenia IDEA Mikołaj vs Centrum Kształcenia IDEA Sp. z o.o. — personal name vs legal entity form of same Polish training centre")
add(SF, 112, "should_be_85", "name_noise_strip",
    "Centrum Kształcenia IDEA Sp. z o.o. — legal entity form; see row 111")

# Row 113-114: Charities Aid Foundation / CHARITIES AID FOUNDATION - GILEAD
add(SF, 113, "should_be_85", "name_noise_strip",
    "Charities Aid Foundation vs CHARITIES AID FOUNDATION - GILEAD (row 114) — '- GILEAD' is a funder/project tag, not a different entity")
add(SF, 114, "should_be_85", "name_noise_strip",
    "CHARITIES AID FOUNDATION - GILEAD — Gilead grant tag appended; strip to match row 113")

# Row 117-118: CHEMIELIVA BIOTECH CO LIMITED / CHEMIELIVA PHARMACEUTICAL CO LTD
add(SF, 117, "should_be_85", "pass_3c_extension",
    "CHEMIELIVA BIOTECH CO LIMITED vs CHEMIELIVA PHARMACEUTICAL CO LTD — same ChemIeliva brand, different business-unit descriptors")
add(SF, 118, "should_be_85", "pass_3c_extension",
    "CHEMIELIVA PHARMACEUTICAL CO LTD — see row 117")

# Row 119: CIBG — singleton Dutch government agency
add(SF, 119, "correctly_blank", "correctly_blank",
    "CIBG — Dutch government agency singleton, no evident pair in dataset")

# Row 120: CLOUD CONNEXTIONS, LLC - 3 WAY — singleton + 3-way tag
add(SF, 120, "correctly_blank", "correctly_blank",
    "CLOUD CONNEXTIONS, LLC - 3 WAY — singleton; '3 WAY' is a payment routing tag, not a separate entity")

# Row 121: Colorado Bioscience Political — singleton PAC
add(SF, 121, "correctly_blank", "correctly_blank",
    "Colorado Bioscience Political — singleton political action committee")

# Row 122: COMAR UNICON PUERTO RICO LLC — singleton
add(SF, 122, "correctly_blank", "correctly_blank",
    "COMAR UNICON PUERTO RICO LLC — singleton, no evident pair")

# Row 123: COMET — singleton (too generic to match)
add(SF, 123, "correctly_blank", "correctly_blank",
    "COMET — singleton with too-generic name; no domain/address to anchor a pair")

# Row 124-125: COMMUNITY ACCESS NATIONAL NETWORK — exact duplicate
add(SF, 124, "should_be_98", "correctly_blank",
    "COMMUNITY ACCESS NATIONAL NETWORK — exact duplicate of row 125")
add(SF, 125, "should_be_98", "correctly_blank",
    "COMMUNITY ACCESS NATIONAL NETWORK — exact duplicate of row 124")

# Row 126-127: CONGRESSMED LTD / CongressMed Ltd - 3 ways
add(SF, 126, "should_be_98", "name_noise_strip",
    "CONGRESSMED LTD vs CongressMed Ltd - 3 ways (row 127) — case difference + '- 3 ways' billing note; same entity")
add(SF, 127, "should_be_98", "name_noise_strip",
    "CongressMed Ltd - 3 ways — billing tag '- 3 ways'; strip to match row 126")

# Row 128-129: Consorcio de Investigación / CONSORCIO DE INVESTIGACION SOBRE VIH SIDA TB CISIDAT
add(SF, 128, "should_be_98", "name_noise_strip",
    "Consorcio de Investigación sobre vs CONSORCIO DE INVESTIGACION SOBRE VIH SIDA TB CISIDAT — accent normalisation + truncation vs full name with acronym appended")
add(SF, 129, "should_be_98", "name_noise_strip",
    "CONSORCIO DE INVESTIGACION SOBRE VIH SIDA TB CISIDAT — full name with CISIDAT acronym; see row 128")

# Row 130-131: Coordinated Resources Inc. of San / COORDINATED RESOURCES, INC
add(SF, 130, "should_be_85", "pass_3c_extension",
    "Coordinated Resources Inc. of San (Diego) vs COORDINATED RESOURCES, INC — 'of San Diego' location suffix; same company")
add(SF, 131, "should_be_85", "pass_3c_extension",
    "COORDINATED RESOURCES, INC — shorter form; see row 130")

# Row 132-134: COVENANT HOUSE INC (×2) / COVENANT HOUSE NEW ORLEANS
add(SF, 132, "should_be_85", "pass_3c_extension",
    "COVENANT HOUSE INC — main entity; rows 133-134 are duplicate and regional office")
add(SF, 133, "should_be_85", "pass_3c_extension",
    "Covenant House Inc — same as row 132 (case variant)")
add(SF, 134, "should_be_85", "pass_3c_extension",
    "COVENANT HOUSE NEW ORLEANS — New Orleans chapter of Covenant House")

# Row 135: CREATIVE BIOMART CREATIVE ENZYMES — concatenated name
add(SF, 135, "should_be_85", "name_noise_strip",
    "CREATIVE BIOMART CREATIVE ENZYMES — appears to be two supplier names concatenated; CreativeBiomart and Creative Enzymes are related brands; strip duplicate prefix")

# Row 136-137: Creditreform Konstanz / CREDITREFORM LUZERN — regional offices
add(SF, 136, "should_be_70", "pass_3c_extension",
    "Creditreform Konstanz Müller & ... vs CREDITREFORM LUZERN VOGEL AG — same Creditreform brand, different cities (Germany vs Switzerland); 70% plausible family cluster")
add(SF, 137, "should_be_70", "pass_3c_extension",
    "CREDITREFORM LUZERN VOGEL AG — see row 136")

# Row 138-139: CRISIS24 INC / CRISIS24 SERVICES PROTECTIVE
add(SF, 138, "should_be_85", "pass_3c_extension",
    "CRISIS24 INC vs CRISIS24 SERVICES PROTECTIVE — same Crisis24 brand, descriptor tokens 'SERVICES PROTECTIVE' added")
add(SF, 139, "should_be_85", "pass_3c_extension",
    "CRISIS24 SERVICES PROTECTIVE — see row 138")

# Row 140: CROUZAT, FREDERIC — singleton individual
add(SF, 140, "correctly_blank", "correctly_blank",
    "CROUZAT, FREDERIC — singleton individual, no evident pair")

# Row 141-142: CYDEX PHARMACEUTICALS INCORPORATED / CYDEX PHARMACTUTICALS INC
add(SF, 141, "should_be_85", "name_noise_strip",
    "CYDEX PHARMACEUTICALS INCORPORATED vs CYDEX PHARMACTUTICALS INC (row 142) — typo 'PHARMACTUTICALS' + INCORPORATED/INC variant; same entity")
add(SF, 142, "should_be_85", "name_noise_strip",
    "CYDEX PHARMACTUTICALS INC — typo in row 142; see row 141")

# Row 143: DANNEMANN SIEMSEN — singleton law firm
add(SF, 143, "correctly_blank", "correctly_blank",
    "DANNEMANN SIEMSEN — singleton law firm, no evident pair")

# Row 144: DATWYLER PHARMA PACKAGING USA INC — singleton
add(SF, 144, "correctly_blank", "correctly_blank",
    "DATWYLER PHARMA PACKAGING USA INC — singleton; Datwyler group entity but no sibling row visible")

# Row 145-146: DDI DEUTSCHLAND / DDW LLC — different companies
add(SF, 145, "correctly_blank", "correctly_blank",
    "DDI DEUTSCHLAND — singleton; DDI Deutschland is a separate entity from DDW LLC (row 146)")
add(SF, 146, "correctly_blank", "correctly_blank",
    "DDW LLC — singleton; unrelated to DDI Deutschland (row 145)")

# Row 147-148: DEVELOPMENTEX.COM INC. DBA DEVEX / DEVELOPMENTEX.COM, INCORPORATED
add(SF, 147, "should_be_98", "name_noise_strip",
    "DEVELOPMENTEX.COM INC. DBA DEVEX vs DEVELOPMENTEX.COM, INCORPORATED — DBA alias tag + punctuation variant; same entity")
add(SF, 148, "should_be_98", "name_noise_strip",
    "DEVELOPMENTEX.COM, INCORPORATED — see row 147")

# Row 149-150: DONNELLY GILLEN LAW / DONNELLY GILLEN LAW***USE VENDOR 30
add(SF, 149, "should_be_85", "name_noise_strip",
    "DONNELLY GILLEN LAW vs DONNELLY GILLEN LAW***USE VENDOR 30 (row 150) — *** internal routing note appended; same entity")
add(SF, 150, "should_be_85", "name_noise_strip",
    "DONNELLY GILLEN LAW***USE VENDOR 30 — strip ***USE VENDOR note; see row 149")

# Row 153-154: DR JOAQUIN ARAMBULA FOR ASSEMBLY — exact duplicate
add(SF, 153, "should_be_98", "correctly_blank",
    "DR JOAQUIN ARAMBULA FOR ASSEMBLY — exact duplicate of row 154")
add(SF, 154, "should_be_98", "correctly_blank",
    "DR JOAQUIN ARAMBULA FOR ASSEMBLY — exact duplicate of row 153")

# Row 155: DYNAMIC EMPLOYMENT SERVICES LLC — singleton
add(SF, 155, "correctly_blank", "correctly_blank",
    "DYNAMIC EMPLOYMENT SERVICES LLC — singleton staffing firm, no evident pair")

# Row 156-157: E - Med Ltd / e-Med — hyphen/space variant
add(SF, 156, "should_be_85", "name_noise_strip",
    "E - Med Ltd vs e-Med (row 157) — spacing/hyphenation around hyphen; same entity")
add(SF, 157, "should_be_85", "name_noise_strip",
    "e-Med — see row 156")

# Row 158-159: EGG EVENTS / EGG MEA EVENTS MANAGEMENT LLC
add(SF, 158, "should_be_85", "pass_3c_extension",
    "EGG EVENTS vs EGG MEA EVENTS MANAGEMENT LLC — 'MEA' (Middle East Africa) + 'MANAGEMENT LLC' descriptors; same Egg Events brand")
add(SF, 159, "should_be_85", "pass_3c_extension",
    "EGG MEA EVENTS MANAGEMENT LLC — see row 158")

# Row 160-161: EIDGENÖSSISCHE ZOLLVERWALTUNG / EIDGENÖSSISCHES ZOLLVERWALTUNG — gender article
add(SF, 160, "should_be_98", "name_noise_strip",
    "EIDGENÖSSISCHE ZOLLVERWALTUNG vs EIDGENÖSSISCHES ZOLLVERWALTUNG — German adjective gender suffix -e vs -es; same Swiss customs authority")
add(SF, 161, "should_be_98", "name_noise_strip",
    "EIDGENÖSSISCHES ZOLLVERWALTUNG — gender variant; see row 160")

# Row 162-163: ELEVER EMS EVENTS AND EXHIBITION / ELEVER EMS EXHIBITION AND
add(SF, 162, "should_be_85", "pass_3c_extension",
    "ELEVER EMS EVENTS AND EXHIBITION vs ELEVER EMS EXHIBITION AND — same Elever EMS brand, descriptor token order/truncation")
add(SF, 163, "should_be_85", "pass_3c_extension",
    "ELEVER EMS EXHIBITION AND — see row 162")

# Row 164-166: ELIZABETH CARBIDE ***USE 3028632*** (×3) — exact duplicates
add(SF, 164, "should_be_98", "name_noise_strip",
    "ELIZABETH CARBIDE ***USE 3028632*** — one of three duplicate rows (164-166); *** routing note; should cluster together")
add(SF, 165, "should_be_98", "name_noise_strip",
    "ELIZABETH CARBIDE ***USE 3028632*** — duplicate row 165; see row 164")
add(SF, 166, "should_be_98", "name_noise_strip",
    "ELIZABETH CARBIDE ***USE 3028632*** — duplicate row 166; see row 164")

# Row 167: Elizabeth Tooling LLC — different entity (Elizabeth Carbide ≠ Elizabeth Tooling)
add(SF, 167, "correctly_blank", "correctly_blank",
    "Elizabeth Tooling LLC — distinct from Elizabeth Carbide (rows 164-166); different company in same Elizabeth NJ industrial area; singleton")

# Row 170: ENGAGING ARKANSAS COMMUNITIES — singleton nonprofit
add(SF, 170, "correctly_blank", "correctly_blank",
    "ENGAGING ARKANSAS COMMUNITIES — singleton, no evident pair")

# Row 172: GSD - Gesellschaft für — singleton
add(SF, 172, "correctly_blank", "correctly_blank",
    "GSD - Gesellschaft für — singleton German organization, no evident pair")

# Row 173: HAHNE, ALEXANDER — see row 24 Alexander Hahne (already handled above)

# Row 174: HEALTH SECURITY AGENCY (UKHSA) — singleton
add(SF, 174, "correctly_blank", "correctly_blank",
    "HEALTH SECURITY AGENCY (UKHSA) — singleton; UK Health Security Agency government body, no sibling row")

# Row 175: Hello Hales LLC — singleton
add(SF, 175, "correctly_blank", "correctly_blank",
    "Hello Hales LLC — singleton creative agency")

# Row 176: IOWA STATE UNIVERSITY — singleton
add(SF, 176, "correctly_blank", "correctly_blank",
    "IOWA STATE UNIVERSITY — singleton, no evident pair")

# Row 177: JENNIFER VAN GENNIP — singleton individual
add(SF, 177, "correctly_blank", "correctly_blank",
    "JENNIFER VAN GENNIP — singleton individual")

# Row 178: LAWS, HANS-JÜRGEN DR — singleton individual
add(SF, 178, "correctly_blank", "correctly_blank",
    "LAWS, HANS-JÜRGEN DR — singleton individual")

# Row 179: LEANNE PERO — singleton individual
add(SF, 179, "correctly_blank", "correctly_blank",
    "LEANNE PERO — singleton individual")

# Row 180: MEDITREE CO LTD — also row 195 SML MEDITREE CO LTD
add(SF, 180, "should_be_85", "pass_3c_extension",
    "MEDITREE CO LTD vs SML MEDITREE CO LTD (row 195) — 'SML' prefix descriptor; same Meditree entity")
add(SF, 195, "should_be_85", "pass_3c_extension",
    "SML MEDITREE CO LTD vs MEDITREE CO LTD (row 180) — SML prefix addition")

# Row 181: MEDMEDIA VERLAG UND MEDIASERVICE — singleton
add(SF, 181, "correctly_blank", "correctly_blank",
    "MEDMEDIA VERLAG UND MEDIASERVICE — singleton Austrian medical publisher")

# Row 182: MEDSAVANA,S.L. — singleton
add(SF, 182, "correctly_blank", "correctly_blank",
    "MEDSAVANA,S.L. — singleton; domain field contains company name (data quality issue) rather than real domain")

# Row 183: Ministerie Van Volksgezondheid — singleton Dutch ministry
add(SF, 183, "correctly_blank", "correctly_blank",
    "Ministerie Van Volksgezondheid — singleton Dutch Ministry of Health, no evident pair")

# Row 184: MUSC FOUNDATION — singleton
add(SF, 184, "correctly_blank", "correctly_blank",
    "MUSC FOUNDATION — singleton; Medical University of South Carolina Foundation")

# Row 185: Nathan and Nathan Human Resources — singleton
add(SF, 185, "correctly_blank", "correctly_blank",
    "Nathan and Nathan Human Resources — singleton HR firm")

# Row 186-187: NATIONAL GOVERNORS ASSOCIATION / NGA CENTER FOR BEST PRACTICES
add(SF, 186, "needs_brand_alias", "brand_alias_csv",
    "NATIONAL GOVERNORS ASSOCIATION vs NGA CENTER FOR BEST PRACTICES (row 187) — NGA is the well-known acronym; brand_aliases.csv entry needed to link NGA → National Governors Association")
add(SF, 187, "needs_brand_alias", "brand_alias_csv",
    "NGA CENTER FOR BEST PRACTICES — NGA acronym maps to National Governors Association; needs brand alias; see row 186")

# Row 188: OXFORD GROUP CONSULTING & — singleton (Oxford Group is a distinct company)
add(SF, 188, "correctly_blank", "correctly_blank",
    "OXFORD GROUP CONSULTING & — singleton consulting firm; no sibling row visible")

# Row 189: Prime Therapeutics LLC — singleton PBM
add(SF, 189, "correctly_blank", "correctly_blank",
    "Prime Therapeutics LLC — singleton pharmacy benefit manager; no evident pair")

# Row 190: PUBLIC HEALTH ENGLAND — singleton government body
add(SF, 190, "correctly_blank", "correctly_blank",
    "PUBLIC HEALTH ENGLAND — singleton (now superseded by UKHSA but was standalone; row 174 is UKHSA, distinct)")

# Row 191: PUBLICLIN — singleton
add(SF, 191, "correctly_blank", "correctly_blank",
    "PUBLICLIN — singleton, no evident pair")

# Row 193: SCHLIENKAMP, ANJA — singleton individual
add(SF, 193, "correctly_blank", "correctly_blank",
    "SCHLIENKAMP, ANJA — singleton individual")

# Row 194: SERVICIO CÁNTABRO DE SALUD — singleton Spanish regional health service
add(SF, 194, "correctly_blank", "correctly_blank",
    "SERVICIO CÁNTABRO DE SALUD — singleton regional health authority")

# Row 196: SPECTRUM MEDICAL GROUP — singleton
add(SF, 196, "correctly_blank", "correctly_blank",
    "SPECTRUM MEDICAL GROUP — singleton, no evident pair")

# Row 197: Starside Community Services Inc — singleton
add(SF, 197, "correctly_blank", "correctly_blank",
    "Starside Community Services Inc — singleton nonprofit")

# Row 198-199: T & T EXECUTIVE S.A. / T&T Travel Consultants — different entities
add(SF, 198, "correctly_blank", "correctly_blank",
    "T & T EXECUTIVE S.A. — singleton; T&T Executive is different from T&T Travel Consultants (row 199); different business lines")
add(SF, 199, "correctly_blank", "correctly_blank",
    "T&T Travel Consultants Μονοπρόσωπη — singleton Greek travel agency; unrelated to T & T EXECUTIVE S.A. (row 198)")

# Row 200: TBD Media Group LTD — same domain as row 10 Acumen International Media (already handled)

# Row 201: THE ACTEON CONSULTANCY LLP — already handled via row 9

# Row 202: THE EUROPEAN HOUSE – AMBROSETTI — already handled via row 31

# Row 204: UNIVERSITATSKLINIKUM DUSSELDORF — singleton
add(SF, 204, "correctly_blank", "correctly_blank",
    "UNIVERSITATSKLINIKUM DUSSELDORF — singleton German university hospital, no evident pair (umlaut 'ä' missing in DUSSELDORF but no sibling row)")

# Row 205: UNIVERSITY MEDICAL ASSOCIATES OF — singleton
add(SF, 205, "correctly_blank", "correctly_blank",
    "UNIVERSITY MEDICAL ASSOCIATES OF — singleton (truncated), no evident pair")

# Row 206: WALLMEYER GMBH — singleton
add(SF, 206, "correctly_blank", "correctly_blank",
    "WALLMEYER GMBH — singleton German company, no evident pair")

# Row 207: WBE NETWORK SYSTEMS INC — singleton
add(SF, 207, "correctly_blank", "correctly_blank",
    "WBE NETWORK SYSTEMS INC — singleton IT company, no evident pair")

# ═══════════════════════════════════════════════════════════════════════════════
# F7 — 49 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F7_german_mixed_file"

# Row 1-2: Merelex Corporation / American Elements — same address 10884 Weyburn Ave
add(SF,  1, "should_be_70", "address_normalization",
    "Merelex Corporation vs American Elements (row 2) — same address 10884 Weyburn Ave, Los Angeles; Merelex is a brand/DBA of American Elements; plausible but different names warrant 70%")
add(SF,  2, "should_be_70", "address_normalization",
    "American Elements vs Merelex Corporation (row 1) — see row 1; same address with minor punctuation diff (Ave vs Ave.)")

# Row 3-4: B S Chems Limited / Bucktom Scott Chemicals Ltd — same address, initials match
add(SF,  3, "needs_brand_alias", "brand_alias_csv",
    "B S Chems Limited vs Bucktom Scott Chemicals Ltd (row 4) — same address 15 Tonbridge Chambers; 'B S' matches initials 'Bucktom Scott'; needs brand_aliases.csv to resolve safely")
add(SF,  4, "needs_brand_alias", "brand_alias_csv",
    "Bucktom Scott Chemicals Ltd vs B S Chems Limited (row 3) — abbreviated form needs alias entry; see row 3")

# Row 5-6: BLOCKED - PLASTICARD-ZFT GMBH / PLASTICARD-ZFT GMBH & CO. KG
add(SF,  5, "should_be_98", "gesperrt_blocked_prefix_strip",
    "BLOCKED - PLASTICARD-ZFT GMBH — strip 'BLOCKED - ' prefix; same address Reisewitzer Str. 82 Dresden as PLASTICARD-ZFT GMBH & CO. KG (row 6)")
add(SF,  6, "should_be_98", "gesperrt_blocked_prefix_strip",
    "PLASTICARD-ZFT GMBH & CO. KG — see blocked counterpart row 5; same address Dresden")

# Row 7: BLOCKED - RISEWAY INTERNATIONAL — ref cluster 31 = RISEWAY INTERNATIONAL LTD
add(SF,  7, "should_be_85", "gesperrt_blocked_prefix_strip",
    "BLOCKED - RISEWAY INTERNATIONAL — strip 'BLOCKED - ' prefix; reference cluster 31 = RISEWAY INTERNATIONAL LTD; same address No 7 Qingjiang South Road Nanjing")

# Row 10-11: Cemyra El Ayari-Pohlmann / Cemyra-Claire El Ayari-Pohlmann — name variant
add(SF, 10, "should_be_85", "name_noise_strip",
    "Cemyra El Ayari-Pohlmann (Mannheim Oberrotweiler Str 11) vs Cemyra-Claire El Ayari-Pohlmann (Oberrotweiler Str 1) — same person, hyphenated forename variant and minor house-number difference")
add(SF, 11, "should_be_85", "name_noise_strip",
    "Cemyra-Claire El Ayari-Pohlmann — see row 10; address number differs by single digit (1 vs 11)")

# Row 12: Chemische Fabrik — ref row 13 G. ZIMMERLI AG has cluster 18 (same address Zimmerlistrasse 24 Aarburg)
add(SF, 12, "should_be_70", "address_normalization",
    "Chemische Fabrik (Zimmerlistrasse 24 Aarburg) — same address as G. ZIMMERLI AG (ref row 13, cluster 18); 'Chemische Fabrik' (Chemical Factory) is likely a DBA/trading name of Zimmerli AG; 70% warranted due to different primary name")

# Row 17: XXXXXXXXXXConcat AGXXXXXXXX595150XX — masked/garbage row
add(SF, 17, "correctly_blank", "correctly_blank",
    "XXXXXXXXXXConcat AGXXXXXXXX595150XX — masked/obfuscated row with XXXX padding and internal ID; not a real supplier entry")

# Row 18-19: DRK Bezirksverband Frankfurt / DRK Bezirksverband Frankfurt am — different addresses
add(SF, 18, "should_be_85", "address_normalization",
    "DRK Bezirksverband Frankfurt (Adelonstr. 31a) vs DRK Bezirksverband Frankfurt am (Vilbeler Str. 27-29) — same DRK organization, 'Frankfurt' vs 'Frankfurt am Main' name form; different office addresses suggest branch vs HQ")
add(SF, 19, "should_be_85", "address_normalization",
    "DRK Bezirksverband Frankfurt am (Main) — see row 18; 'Frankfurt am' is truncated 'Frankfurt am Main'")

# Row 20-21: E&E information consultants AG / ESCRIBA AG — different Berlin addresses
add(SF, 20, "needs_llm", "llm_review",
    "E&E information consultants AG (Invalidenstraße 112, Berlin) vs ESCRIBA AG (Hardenbergstrasse 32d, Berlin) — different companies at different Berlin addresses; possible common ownership but requires LLM review")
add(SF, 21, "needs_llm", "llm_review",
    "ESCRIBA AG — see row 20; different address from E&E; LLM needed")

# Row 22-23: ETA Standing Office / European Thyroid Association — same address Hopfengartenweg 19
add(SF, 22, "should_be_85", "acronym_expansion",
    "ETA Standing Office (Hopfengartenweg 19, Altdorf) — ETA = European Thyroid Association; same address as row 23 European Thyroid Association; reference cluster 21610760357")
add(SF, 23, "should_be_85", "acronym_expansion",
    "European Thyroid Association (Hopfengartenweg 19) — full name of ETA; see row 22")

# Row 24-25: Falk Logistik Management / FK Logistik Spezialtransporte — different cities
add(SF, 24, "needs_llm", "llm_review",
    "Falk Logistik Management GmbH & Co. (Hessenring 24, Büttelborn) vs FK Logistik Spezialtransporte (Heinrich-Hertz-Str, Riedstadt-Wolfskehle) — 'Falk' vs 'FK' abbreviation, different cities; unclear if same company; LLM needed")
add(SF, 25, "needs_llm", "llm_review",
    "FK Logistik Spezialtransporte — see row 24; may be abbreviation of 'Falk'; different city")

# Row 26-27: FIGIEL GMBH & CO.KG / GESPERRT FIGIEL — same address
add(SF, 26, "should_be_98", "gesperrt_blocked_prefix_strip",
    "FIGIEL GMBH & CO.KG (Robert-Bosch-Str. 10 Steinheim) — counterpart of GESPERRT FIGIEL (row 27), same address")
add(SF, 27, "should_be_98", "gesperrt_blocked_prefix_strip",
    "GESPERRT FIGIEL — strip 'GESPERRT ' prefix; same address Robert-Bosch-Str. 10 Steinheim as FIGIEL GMBH & CO.KG (row 26)")

# Row 28-29: GESPERRT KREISABFALLWIRTSCHAFTSBETR / KREISABFALLWIRTSCHAFTSBETRIEB
add(SF, 28, "should_be_98", "gesperrt_blocked_prefix_strip",
    "GESPERRT KREISABFALLWIRTSCHAFTSBETR — strip 'GESPERRT ' prefix; truncated name; same address Schmittenplatz 5 Heidenheim as KREISABFALLWIRTSCHAFTSBETRIEB (row 29)")
add(SF, 29, "should_be_98", "gesperrt_blocked_prefix_strip",
    "KREISABFALLWIRTSCHAFTSBETRIEB (Schmittenplatz 5 Heidenheim) — unblocked form of row 28")

# Row 35: Monument Chemical BVBA — same address as Haltermann N.V. (cluster 21), Haven 1972 Ketenislaan 3 Kallo
add(SF, 35, "should_be_70", "address_normalization",
    "Monument Chemical BVBA (Haven 1972, Ketenislaan 3, Kallo) — same address as Haltermann N.V. (ref cluster 21); Monument acquired Haltermann's Belgian chemical operations; 70% confidence due to corporate name change context")

# Row 42-44: Ing.Buero Helmuth Zohren / Ingenierubüro H.W. Helmuth Zohren / INGENIEURBÜRO HELMUT ZOHREN
add(SF, 42, "should_be_85", "address_normalization",
    "Ing.Buero Helmuth Zohren (Im Schöll 29, Münster Hessen) — same address as rows 43 and 44; 'Ing.Buero' = abbreviated 'Ingenieurbüro'; same person's engineering office")
add(SF, 43, "should_be_85", "address_normalization",
    "Ingenierubüro H.W. Helmuth Zohren (Im Schöll 29) — middle-initial variant; same address; see row 42")
add(SF, 44, "should_be_85", "address_normalization",
    "INGENIEURBÜRO HELMUT ZOHREN (Im Schöll 29, Münster) — slight forename variant (Helmuth vs Helmut); same address group")

# Row 45-46: John Baldoni / John M. Bandoni — same address Naples, likely name typo
add(SF, 45, "should_be_70", "name_noise_strip",
    "John Baldoni vs John M. Bandoni (row 46) — same address 9549 Siracusa Court Naples; possible person name typo ('Baldoni' vs 'Bandoni' + middle initial); 70% due to surname difference")
add(SF, 46, "should_be_70", "name_noise_strip",
    "John M. Bandoni — see row 45; surname 'Bandoni' vs 'Baldoni' and middle initial M; same address")

# Row 49: LehmannXXXXX525671 — masked row
add(SF, 49, "correctly_blank", "correctly_blank",
    "LehmannXXXXX525671 — masked row with XXXXX obfuscation and internal ID; not a matchable supplier entry")

# Row 50-51: LEHVOSS UK Ltd. / GEE LAWSON CHEMICALS LTD — different addresses, possible acquisition
add(SF, 50, "needs_llm", "llm_review",
    "LEHVOSS UK Ltd. (40 Holmes Chapel, Congleton) vs GEE LAWSON CHEMICALS LTD (309 Ballards Lane, London) — different addresses; LEHVOSS Group acquired Gee Lawson's chemical distribution business; needs LLM review")
add(SF, 51, "needs_llm", "llm_review",
    "GEE LAWSON CHEMICALS LTD — see row 50; historical acquisition connection to LEHVOSS")

# Row 52-53: Limbus GmbH / Thies Lindenlaub — same address Otto-Sachs Strasse 5 Karlsruhe
add(SF, 52, "should_be_70", "address_normalization",
    "Limbus GmbH (Otto-Sachs Strasse 5, Karlsruhe) vs Thies Lindenlaub (Otto-Sachs Strasse 5) — same address; person and company; Lindenlaub may be owner/director; 70% plausible")
add(SF, 53, "should_be_70", "address_normalization",
    "Thies Lindenlaub (Otto-Sachs Strasse 5, Karlsruhe) — see row 52; person at same address as Limbus GmbH")

# Row 58: CMC Partnership (UK) Ltd — singleton
add(SF, 58, "correctly_blank", "correctly_blank",
    "CMC Partnership (UK) Ltd (Grace Dieu Court, Dingestow) — singleton, no evident pair")

# Row 64: PROVIRON INDUSTRIES NV — Georges Gilliotstraat 60 Hemiksem; cluster 29 = Proviron Functional Chemicals (different address)
add(SF, 64, "correctly_blank", "correctly_blank",
    "PROVIRON INDUSTRIES NV (Georges Gilliotstraat 60, Hemiksem) — different address from Proviron Functional Chemicals NV (ref cluster 29); Industries vs Functional Chemicals are distinct legal entities; singleton")

# Row 67-68: S. K. Chemical Industries (×2) — same address, exact duplicate
add(SF, 67, "should_be_98", "address_normalization",
    "S. K. Chemical Industries (818/819 Corporate Ave Mumbai) — duplicate pair; address differs by minor punctuation '. ' prefix; same entity")
add(SF, 68, "should_be_98", "address_normalization",
    "S. K. Chemical Industries — see row 67; address has extra '. ' prefix artifact")

# Row 71: Saatchi & Saatchi Wellness — ref cluster 45 = Saatchi & Saatchi GmbH (different div)
add(SF, 71, "correctly_blank", "correctly_blank",
    "Saatchi & Saatchi Wellness (2 Television Centre, London) — different address and division from cluster 45 (Saatchi GmbH, German entities); Wellness division is a distinct entity; singleton")

# Row 90: Saba & Co. T.M.P - Yemen — ref cluster 46 = Saba & Co. (TMP) — join cluster
add(SF, 90, "should_be_85", "pass_3c_extension",
    "Saba & Co. T.M.P - Yemen — '- Yemen' is country descriptor; reference cluster 46 = Saba & Co. TMP family; Yemen office should join cluster 46")

# Row 91: Saba Arabia Limited Co. S.P.C — different entity from Saba & Co.
add(SF, 91, "needs_brand_alias", "brand_alias_csv",
    "Saba Arabia Limited Co. S.P.C (Riyadh) — 'Saba Arabia' is a distinct Saudi entity from Saba & Co. TMP; without brand_aliases.csv this cannot be safely grouped with cluster 46")

# Row 92-93: SCHNEIDER GMBH & CO. KG / Schneider Versand GmbH — same address Strandbaddamm 2-4 Wedel
add(SF, 92, "should_be_85", "address_normalization",
    "SCHNEIDER GMBH & CO. KG (Strandbaddamm 2-4, Wedel) vs Schneider Versand GmbH (Strandbaddamm 2-4, Wedel) — same address; different legal entities but same street; Versand (mail-order) may be subsidiary")
add(SF, 93, "should_be_85", "address_normalization",
    "Schneider Versand GmbH — see row 92; same Strandbaddamm address Wedel")

# Row 94-95: Suzhou Putin Vacuum Technology Co / SuzhouPutinVacuumTechnologyCo — spacing only
add(SF, 94, "should_be_98", "name_noise_strip",
    "Suzhou Putin Vacuum Technology Co vs SuzhouPutinVacuumTechnologyCo (row 95) — space removal only; same address Romm 11AB Kings Tower Binghe Rd Suzhou")
add(SF, 95, "should_be_98", "name_noise_strip",
    "SuzhouPutinVacuumTechnologyCo — no-space variant; see row 94")

# Row 96: VICTORDYES — ref row 97-98 Victory Dye Chem Industries (cluster 54)
add(SF, 96, "should_be_98", "address_normalization",
    "VICTORDYES (316 Samuel Street, Mumbai) — same address/brand as Victory Dye Chem Industries (ref cluster 54); concatenated variant of 'Victor Dyes' = 'Victory Dyes'")

# Row 99-101: Tschimmer & Schwarz / Zschimmer & Schwarz / Zschimmer & Schwarz Mohsdorf
add(SF, 99, "should_be_98", "name_noise_strip",
    "Tschimmer & Schwarz GmbH & Co. (Max-Schwarz-Str. 3-5, Lahnstein) — typo 'Tschimmer' for 'Zschimmer'; same address as Zschimmer & Schwarz (row 100)")
add(SF, 100, "should_be_98", "name_noise_strip",
    "Zschimmer & Schwarz (Max-Schwarz-Str. 3-5, Lahnstein) — correct spelling; see row 99 for typo variant")
add(SF, 101, "should_be_85", "name_noise_strip",
    "Zschimmer & Schwarz Mohsdorf GmbH (Chemnitztalstraße 1, Burgstädt) — same Zschimmer & Schwarz brand, different city (Burgstädt); branch of same group")

# ═══════════════════════════════════════════════════════════════════════════════
# F6 — 11 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F6_blab_bts_cda_krisbow"

# Row 6: B LAB FRANCE — ref cluster 75019
add(SF,  6, "needs_brand_alias", "brand_alias_csv",
    "B LAB FRANCE (6 Quai de la Seine) — reference original cluster 75019; B Lab brand is too generic without alias; brand_aliases.csv entry needed to link to B Lab Co. family (ref cluster 6)")

# Row 7: B LAB SWITZERLAND — ref cluster 1203
add(SF,  7, "needs_brand_alias", "brand_alias_csv",
    "B LAB SWITZERLAND (Rue de Lyon 77) — reference original cluster 1203; needs brand_aliases.csv entry to resolve to B Lab Co. family")

# Row 8: B LAB UK — no reference cluster
add(SF,  8, "needs_brand_alias", "brand_alias_csv",
    "B LAB UK (20-30 Whitechapel Road) — no reference cluster; B Lab UK is a distinct national entity; needs brand_aliases.csv to group with B Lab family")

# Row 25: KRISBOW TOKO — ref cluster 99999 (placeholder), KRISBOW INDONESIA is cluster 11610/53147
add(SF, 25, "should_be_85", "pass_3c_extension",
    "KRISBOW TOKO — 'Toko' = Indonesian for 'store/shop'; Krisbow retail store suffix; should join KRISBOW INDONESIA cluster (11610); pass_3c_extension to handle 'TOKO' service descriptor")

# Row 26: PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P Państwowe Nadleśnictwo Jugów — no ref cluster
add(SF, 26, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P Państwowe Nadleśnictwo Jugów — same Polish State Forests organization (PGL LP), Jugów district; no reference cluster but should join cluster 18 family; polish unicode normalization needed")

# Row 27: Nadleśnictwo Kobiór — ref cluster 18
add(SF, 27, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P Państwowe Nadleśnictwo Kobiór — reference cluster 18; Polish unicode normalization needed; PGL LP Kobiór district")

# Row 28: NADLEŚNICTWO KOSZĘCIN — ref cluster 18
add(SF, 28, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P PAŃSTWOWE NADLEŚNICTWO KOSZĘCIN — reference cluster 18; PGL LP Koszęcin district; polish unicode normalization")

# Row 29: NADLEŚNICTWO JELEŚNIA — ref cluster 18
add(SF, 29, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY PAŃSTWOWE NADLEŚNICTWO JELEŚNIA — reference cluster 18; PGL LP Jeleśnia district")

# Row 30: Nadleśnictwo Oborniki Śląskie — no ref cluster
add(SF, 30, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P Państwowe Nadleśnictwo Oborniki Ślą — no reference cluster; Oborniki Śląskie district; same PGL LP organization; should join cluster 18 family")

# Row 31: Nadleśnictwo Olesno — no ref cluster
add(SF, 31, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P Państwowe Nadleśnictwo Olesno — no reference cluster; Olesno district; same PGL LP organization")

# Row 32: Nadleśnictwo Wołów — no ref cluster
add(SF, 32, "should_be_85", "polish_unicode_normalization",
    "PAŃSTWOWE GOSPODARSTWO LEŚNE LASY P Państwowe Nadleśnictwo Wołów — no reference cluster; Wołów district; same PGL LP organization")

# ═══════════════════════════════════════════════════════════════════════════════
# F4 — 9 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F4_5p_accelera_admin"

# Row 3: ACCELERA SRL WEB EDI 06/27/03 ALL MRO — ref cluster 10 match 85%
add(SF,  3, "should_be_85", "name_noise_strip",
    "ACCELERA SRL WEB EDI 06/27/03 ALL MRO — reference cluster 10, original match 85%; 'WEB EDI 06/27/03 ALL MRO' is procurement note suffix; strip to ACCELERA SRL to match cluster")

# Row 6-7: Acta Conseils Sarl / ACTA Laboratories Inc — ref cluster 16 match 85%
add(SF,  6, "needs_llm", "llm_review",
    "Acta Conseils Sarl (Rue des Pecheurs 8A) — reference cluster 16, original 85%; different country from ACTA Laboratories Inc (row 7, USA); risky cross-country clustering; LLM review needed")
add(SF,  7, "needs_llm", "llm_review",
    "ACTA Laboratories Inc (27082 Burbank) — reference cluster 16, original 85%; US entity; pairing with French Acta Conseils Sarl is risky without further evidence")

# Row 8-9: Active Food Sa / ACTIVE TECHNOLOGY — ref cluster 17 match 85%
add(SF,  8, "needs_llm", "llm_review",
    "Active Food Sa (Avenue des Champs Montants 12B) — reference cluster 17, original 85%; 'Active Food' vs 'ACTIVE TECHNOLOGY' are very different business lines; risky pairing; LLM review")
add(SF,  9, "needs_llm", "llm_review",
    "ACTIVE TECHNOLOGY (Unit 14, Ballycasey Craft Centre) — reference cluster 17, original 85%; see row 8; different industries")

# Row 14: ADMIN DE SEGUROS DE S.ALUD — ref cluster 23 rows 15-16 are 98%
add(SF, 14, "should_be_98", "name_noise_strip",
    "ADMIN DE SEGUROS DE S.ALUD DE PR... — 'S.ALUD' is a name split artifact ('SALUD' → 'S.ALUD'); reference rows 15-16 in same cluster 23 at 98%; same entity ADMINISTRACION DE SEGUROS DE SALUD DE PR")

# Row 17: ADMIN DE SEGUROS DE SALUD DE PR — slightly different from row 14 (SALUD vs S.ALUD)
add(SF, 17, "should_be_98", "name_noise_strip",
    "ADMIN DE SEGUROS DE SALUD DE PR... — correct spelling 'SALUD' vs row 14's 'S.ALUD'; same Puerto Rico Health Insurance Administration; reference cluster 23")

# Row 18-19: ADVANCE AG / Advanceanalytics Tech Limited — ref cluster 24 match 70%
add(SF, 18, "needs_llm", "llm_review",
    "ADVANCE AG (Alte Landstrasse 15) — reference cluster 24, original 70%; pairing with Advanceanalytics Tech Limited (row 19, UK) is risky; different company names and countries; LLM review")
add(SF, 19, "needs_llm", "llm_review",
    "Advanceanalytics Tech Limited (Abbotsley St Neots) — reference cluster 24, original 70%; see row 18; different country and business line")

# ═══════════════════════════════════════════════════════════════════════════════
# F8 — 20 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F8_final_file"

# Row 5: ACMI BEVERAGE IBERICA SL — ref cluster 12006; ACMI family
add(SF,  5, "should_be_85", "pass_3c_extension",
    "ACMI BEVERAGE IBERICA SL (Avda. Gran Via Tarrega) — reference cluster 12006; Iberian entity of ACMI group; pass_3c_extension to handle 'IBERICA' country descriptor")

# Row 8: ACMI LABELLING SRL — ref cluster 43058 (same as ACMI BEVERAGE SPA)
add(SF,  8, "should_be_85", "address_normalization",
    "ACMI LABELLING SRL (Via E. Ferrari 1, Ramoscello) — reference cluster 43058 (same as ACMI BEVERAGE SPA); same address as ACMI BEVERAGE SPA; labelling division of ACMI group")

# Row 10-12: AG SOLUTION / AG SOLUTION NV / AG SOLUTION SPAIN SA — ref clusters 69003/2610/8019
add(SF, 10, "should_be_85", "brand_alias_csv",
    "AG SOLUTION (21 Avenue Georges-Pompidou) — reference cluster 69003; French entity of AG SOLUTION group; needs brand_aliases.csv to safely cluster France/Belgium/Spain entities")
add(SF, 11, "should_be_85", "brand_alias_csv",
    "AG SOLUTION NV (Moerelei 125 B201) — reference cluster 2610; Belgian entity; same brand family as rows 10 and 12")
add(SF, 12, "should_be_85", "brand_alias_csv",
    "AG SOLUTION SPAIN SA (Pujades 350, Barcelona) — reference cluster 8019; Spanish entity; same AG SOLUTION brand family")

# Row 13-14,17: AGS GLOBAL SOLUTIONS / AGS MUDANZAS / AGS RHONE ALPES — ref clusters; join AGS PARIS 92230
add(SF, 13, "should_be_85", "brand_alias_csv",
    "AGS GLOBAL SOLUTIONS GMBH (Mittenheimer Str. 64) — reference cluster 85764; German entity of AGS group; needs brand_aliases.csv to join AGS PARIS cluster (92230)")
add(SF, 14, "should_be_85", "brand_alias_csv",
    "AGS MUDANZAS INTERNACIONALES, S.L. (CL Mario Roso de Luna 29) — reference cluster 28022; Spanish moving entity of AGS group; needs brand alias")
add(SF, 17, "should_be_85", "brand_alias_csv",
    "AGS RHONE ALPES AUVERGNE (17 Rue Maurice Petit) — reference cluster 69360; French regional entity of AGS group; needs brand alias")

# Row 39-40: BF USA IT SERVICES INC / BF USA IT SERVICES INC. — different addresses
add(SF, 39, "should_be_85", "name_noise_strip",
    "BF USA IT SERVICES INC (777 Brickell Ave, Miami) — reference cluster 33131; same company as row 40 at different office address; punctuation variant (no period vs period)")
add(SF, 40, "should_be_85", "name_noise_strip",
    "BF USA IT SERVICES INC. (1 Broadway, MA) — reference cluster 2142; same company, different office; trailing period variant; see row 39")

# Row 43: C2FO LTD POLLEN INC — ref cluster 66206; C2FO cluster is 18
add(SF, 43, "should_be_85", "name_noise_strip",
    "C2FO LTD POLLEN INC (2020 W 89th Street) — reference cluster 66206; 'POLLEN INC' is an appended entity (Pollen acquired by C2FO or co-billing); strip to C2FO LTD to join cluster 18")

# Row 46: CEDAP MEXICO SA DE CV SIAMP CEDAP — ref cluster 45610; should join cluster 20
add(SF, 46, "should_be_85", "name_noise_strip",
    "CEDAP MEXICO SA DE CV SIAMP CEDAP — reference cluster 45610; 'SIAMP CEDAP' is appended entity name; strip to CEDAP MEXICO SA DE CV to join cluster 20")

# Row 47-48: CEDAP SIAMP CEDAP / SIAMP CEDAP — ref cluster 21 match 85%
add(SF, 47, "should_be_85", "name_noise_strip",
    "CEDAP SIAMP CEDAP — reference cluster 21, original 85%; 'SIAMP CEDAP' appended; same address 4 Quai Antoine 1er Monaco as SIAMP CEDAP (row 48)")
add(SF, 48, "should_be_85", "address_normalization",
    "SIAMP CEDAP (4 Quai Antoine 1er) — reference cluster 21, original 85%; same address as row 47 CEDAP SIAMP CEDAP; address normalisation needed (4 QUAI ANTOINE 1ER vs 4 Quai Antoine 1er case difference)")

# Row 51-52: CENTRE FRANCE COMMUNICATION / CENTRE FRANCE PUBLICITE — same address
add(SF, 51, "should_be_85", "address_normalization",
    "CENTRE FRANCE COMMUNICATION (45 Rue du Clos Four) — reference cluster 63056; same address as CENTRE FRANCE PUBLICITE (row 52, cluster 63020); different departments of Centre France media group")
add(SF, 52, "should_be_85", "address_normalization",
    "CENTRE FRANCE PUBLICITE (45 Rue du Clos Four) — reference cluster 63020; same address as row 51; see row 51")

# Row 64: DIAGNOSTYKA-MEDYCZNE... — reference shows no cluster; singleton
add(SF, 64, "correctly_blank", "correctly_blank",
    "DIAGNOSTYKA-MEDYCZNE CENTRUM LABORATORYJNE SP Z O.O. W TARNOWIE — no reference cluster; different legal entity from DIAGNOSTYKA SP. Z O.O. SP. K; singleton Tarnów branch")

# Row 65-66: DIGITAL MEDIA LAB PRODUCTION S.L. / DIGITAL MEDIA LAB PRODUCTION, S.L. — ref cluster 8005
add(SF, 65, "should_be_98", "name_noise_strip",
    "DIGITAL MEDIA LAB PRODUCTION S.L. (CL Ramon Turro 23) — reference cluster 8005; same as row 66; punctuation variant (no comma vs comma); different addresses suggest two offices")
add(SF, 66, "should_be_98", "name_noise_strip",
    "DIGITAL MEDIA LAB PRODUCTION, S.L. (Llull 48) — reference cluster 8005; same entity as row 65; punctuation variant")

# Row 70: DM-DROGERIA KFT. — ref cluster 2046; needs brand alias to join DM group
add(SF, 70, "needs_brand_alias", "brand_alias_csv",
    "DM-DROGERIA KFT. (Depo Pf.4) — reference cluster 2046; Hungarian entity of DM Drogerie Markt group; 'DROGERIA' (Hungarian) vs 'DROGERIE MARKT' (German); needs brand_aliases.csv to safely group with DM cluster")

# ═══════════════════════════════════════════════════════════════════════════════
# F2 — 3 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F2_screenfluence_seara_securitas"

# Row 14: SECURITAS ČR S.R.O. — Czech special chars
add(SF, 14, "should_be_85", "name_noise_strip",
    "SECURITAS ČR S.R.O. (Kateřinská 466 40) — Czech entity; 'ČR' = Česká republika; Czech diacritics and S.R.O. legal suffix prevent match; should join Securitas cluster")

# Row 15: Securitas Electronic Security Deuts — German entity
add(SF, 15, "should_be_85", "name_noise_strip",
    "Securitas Electronic Security Deuts (Daniel-Goldlach-strasse 17-19) — German entity; 'Deuts' is truncated 'Deutschland'; should join Securitas cluster")

# Row 23: SECURITAS TECHNOLOGY CANADA CORPORA — Canadian entity; ref cluster 15 = Securitas Technology Corporation
add(SF, 23, "should_be_85", "name_noise_strip",
    "SECURITAS TECHNOLOGY CANADA CORPORA (6275 Millcreek Drive) — truncated 'CORPORATION'; Canadian entity; reference cluster 15 = Securitas Technology Corporation; should join Securitas Technology sub-cluster")

# ═══════════════════════════════════════════════════════════════════════════════
# F3 — 2 blank rows
# ═══════════════════════════════════════════════════════════════════════════════
SF = "F3_3carp_absolute_cesar"

# Row 12-13: MER JAN LLC / MERJAN LLC — space variant
add(SF, 12, "should_be_85", "name_noise_strip",
    "MER JAN LLC (880 E 1375 S) vs MERJAN LLC (3201 Wabash Ave) — same name with space insertion; different addresses may indicate two locations; space variant is a common data entry error; 85% confidence")
add(SF, 13, "should_be_85", "name_noise_strip",
    "MERJAN LLC (3201 Wabash Ave) vs MER JAN LLC (row 12) — no-space variant; see row 12")

# ═══════════════════════════════════════════════════════════════════════════════
# Build output DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

classification_map = {(r["source_file"], r["source_row"]): r for r in ROWS}

# Attach classifications back to blank rows
output_records = []
for _, row in blank.iterrows():
    key = (row["Source File"], int(row["Source Row"]))
    cls = classification_map.get(key, {})
    output_records.append({
        "source_file":       row["Source File"],
        "source_row":        row["Source Row"],
        "supplier_name":     row["Supplier Name"],
        "address":           row["Address"],
        "city":              row["City"],
        "country":           row["Country"],
        "domain":            row["Website/Domain"],
        "classification":    cls.get("classification", "needs_llm"),
        "fix_type":          cls.get("fix_type", "llm_review"),
        "target_cluster":    cls.get("target_cluster", ""),
        "notes":             cls.get("notes", "UNCLASSIFIED — needs manual review"),
    })

out_df = pd.DataFrame(output_records)

# ── Validation ────────────────────────────────────────────────────────────────
total = len(out_df)
classified = len(out_df[out_df["classification"] != ""])
unclassified = out_df[out_df["classification"] == ""]
if len(unclassified):
    print("WARNING: unclassified rows:")
    print(unclassified[["source_file", "source_row", "supplier_name"]])

print(f"\nTotal blank rows classified: {total}")
print("\nClassification breakdown:")
print(out_df["classification"].value_counts().to_string())
print("\nFix type breakdown:")
print(out_df["fix_type"].value_counts().to_string())

# Check we covered all 273
by_file = out_df.groupby("source_file").size()
print("\nRows per file:")
print(by_file.to_string())

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_df.to_csv(OUT_CSV, index=False)
print(f"\nWrote: {OUT_CSV}")

# ── Write Markdown report ─────────────────────────────────────────────────────
cls_counts  = out_df["classification"].value_counts()
fix_counts  = out_df["fix_type"].value_counts()
file_counts = out_df.groupby("source_file")["classification"].value_counts().unstack(fill_value=0)

CLASSIFICATION_DESC = {
    "correctly_blank":  "Genuinely different entity or true singleton — no fix needed",
    "should_be_98":     "Near-exact name / same entity / strong evidence (target ≥ 98%)",
    "should_be_85":     "Supplier family or branch relation (target 85%)",
    "should_be_70":     "Plausible but uncertain — review candidate (target 70%)",
    "needs_brand_alias":"Requires brand_aliases.csv entry to resolve safely",
    "needs_domain_alias":"Requires domain_aliases.csv entry",
    "needs_llm":        "Too ambiguous for deterministic logic — LLM review required",
}

FIX_DESC = {
    "gesperrt_blocked_prefix_strip": "Name starts with GESPERRT or BLOCKED prefix",
    "same_domain_bug":               "Two rows share exact same non-generic domain, both blank",
    "address_normalization":         "Minor address variant prevents address match",
    "name_noise_strip":              "Internal code/note appended or minor name noise",
    "polish_unicode_normalization":  "Long Polish name with special chars not normalizing",
    "acronym_expansion":             "Acronym not matched to full name",
    "pass_3c_extension":             "Needs additional service/descriptor token in Pass 3c",
    "brand_alias_csv":               "Genuinely different brand names needing alias dict entry",
    "domain_alias_csv":              "Related domains needing domain alias dict entry",
    "name_similarity_gap":           "Name similarity below threshold despite clear relationship",
    "correctly_blank":               "No fix needed",
    "llm_review":                    "Needs LLM review",
    "needs_llm":                     "Needs LLM review",
}

def section_table(df_sub, title):
    lines = [f"\n### {title}\n"]
    lines.append("| Row | Supplier Name | Classification | Fix Type | Notes |")
    lines.append("|-----|---------------|----------------|----------|-------|")
    for _, r in df_sub.iterrows():
        name  = str(r["supplier_name"])[:55].replace("|", "\\|")
        notes = str(r["notes"])[:120].replace("|", "\\|")
        lines.append(f"| {r['source_row']} | {name} | `{r['classification']}` | `{r['fix_type']}` | {notes} |")
    return "\n".join(lines)

md_lines = [
    "# Phase 3 — Blank Row Classification Report",
    "",
    f"**Generated:** 2026-05-20  ",
    f"**Total blank rows classified:** {total}  ",
    f"**Phase 2 output:** `/private/tmp/phase2_output.csv`  ",
    f"**Reference file:** `all_raw_missed_examples_combined_v2_reference.csv`",
    "",
    "---",
    "",
    "## 1. Summary Statistics",
    "",
    "### 1a. Classification Counts",
    "",
    "| Classification | Count | Description |",
    "|----------------|-------|-------------|",
]
for cls, cnt in cls_counts.items():
    desc = CLASSIFICATION_DESC.get(cls, "")
    md_lines.append(f"| `{cls}` | {cnt} | {desc} |")

md_lines += [
    "",
    "### 1b. Fix Type Counts",
    "",
    "| Fix Type | Count | Description |",
    "|----------|-------|-------------|",
]
for ft, cnt in fix_counts.items():
    desc = FIX_DESC.get(ft, "")
    md_lines.append(f"| `{ft}` | {cnt} | {desc} |")

md_lines += [
    "",
    "### 1c. Classification × Source File",
    "",
]
# Build a simple cross-tab
all_cls = sorted(out_df["classification"].unique())
all_files = sorted(out_df["source_file"].unique())
header = "| Source File | " + " | ".join(all_cls) + " | Total |"
sep    = "|-------------|" + "|".join(["---"] * len(all_cls)) + "|-------|"
md_lines += [header, sep]
for sf in all_files:
    sub = out_df[out_df["source_file"] == sf]
    counts = [str(sub[sub["classification"] == c].shape[0]) for c in all_cls]
    md_lines.append(f"| {sf} | " + " | ".join(counts) + f" | {len(sub)} |")
total_row = "| **TOTAL** | " + " | ".join([str(out_df[out_df["classification"] == c].shape[0]) for c in all_cls]) + f" | {total} |"
md_lines.append(total_row)

md_lines += [
    "",
    "---",
    "",
    "## 2. Actionable Fix Queue",
    "",
    "### 2a. `should_be_98` — High-confidence same-entity pairs (no alias needed)",
    "*These can be fixed by code changes alone (prefix strip, noise strip, normalization).*",
    "",
]

for ft_group, title in [
    ("gesperrt_blocked_prefix_strip", "GESPERRT/BLOCKED Prefix Strip"),
    ("name_noise_strip", "Name Noise / Note Strip"),
    ("address_normalization", "Address Normalization"),
]:
    sub = out_df[(out_df["classification"] == "should_be_98") & (out_df["fix_type"] == ft_group)]
    if len(sub):
        md_lines.append(f"#### {title} ({len(sub)} rows)\n")
        md_lines.append("| Row | File | Supplier Name | Notes |")
        md_lines.append("|-----|------|---------------|-------|")
        for _, r in sub.iterrows():
            name  = str(r["supplier_name"])[:55].replace("|", "\\|")
            notes = str(r["notes"])[:100].replace("|", "\\|")
            md_lines.append(f"| {r['source_row']} | {r['source_file']} | {name} | {notes} |")
        md_lines.append("")

md_lines += [
    "### 2b. `should_be_85` — Family/Branch Pairs",
    "*Require pass 3c extension, address normalization, or brand alias.*",
    "",
]
sub85 = out_df[out_df["classification"] == "should_be_85"]
for sf in all_files:
    sub = sub85[sub85["source_file"] == sf]
    if len(sub):
        md_lines.append(section_table(sub, f"{sf} ({len(sub)} rows)"))
        md_lines.append("")

md_lines += [
    "### 2c. `should_be_70` — Plausible but Uncertain (Review Candidates)",
    "",
]
sub70 = out_df[out_df["classification"] == "should_be_70"]
if len(sub70):
    md_lines.append("| Row | File | Supplier Name | Fix Type | Notes |")
    md_lines.append("|-----|------|---------------|----------|-------|")
    for _, r in sub70.iterrows():
        name  = str(r["supplier_name"])[:50].replace("|", "\\|")
        notes = str(r["notes"])[:100].replace("|", "\\|")
        md_lines.append(f"| {r['source_row']} | {r['source_file']} | {name} | `{r['fix_type']}` | {notes} |")
    md_lines.append("")

md_lines += [
    "### 2d. `needs_brand_alias` — Brand Alias CSV Required",
    "",
    "These rows require a new entry in `brand_aliases.csv` before they can be safely clustered.",
    "",
]
sub_ba = out_df[out_df["classification"] == "needs_brand_alias"]
if len(sub_ba):
    md_lines.append("| Row | File | Supplier Name | Notes |")
    md_lines.append("|-----|------|---------------|-------|")
    for _, r in sub_ba.iterrows():
        name  = str(r["supplier_name"])[:55].replace("|", "\\|")
        notes = str(r["notes"])[:120].replace("|", "\\|")
        md_lines.append(f"| {r['source_row']} | {r['source_file']} | {name} | {notes} |")
    md_lines.append("")

md_lines += [
    "### 2e. `needs_llm` — LLM Review Required",
    "",
    "These pairs are too ambiguous for deterministic rules.",
    "",
]
sub_llm = out_df[out_df["classification"] == "needs_llm"]
if len(sub_llm):
    md_lines.append("| Row | File | Supplier Name | Notes |")
    md_lines.append("|-----|------|---------------|-------|")
    for _, r in sub_llm.iterrows():
        name  = str(r["supplier_name"])[:55].replace("|", "\\|")
        notes = str(r["notes"])[:120].replace("|", "\\|")
        md_lines.append(f"| {r['source_row']} | {r['source_file']} | {name} | {notes} |")
    md_lines.append("")

md_lines += [
    "### 2f. `correctly_blank` — True Singletons (No Fix Needed)",
    "",
    "These rows are genuinely unique entities with no apparent pair in the dataset.",
    "",
]
sub_cb = out_df[out_df["classification"] == "correctly_blank"]
if len(sub_cb):
    md_lines.append("| Row | File | Supplier Name | Notes |")
    md_lines.append("|-----|------|---------------|-------|")
    for _, r in sub_cb.iterrows():
        name  = str(r["supplier_name"])[:55].replace("|", "\\|")
        notes = str(r["notes"])[:100].replace("|", "\\|")
        md_lines.append(f"| {r['source_row']} | {r['source_file']} | {name} | {notes} |")
    md_lines.append("")

md_lines += [
    "---",
    "",
    "## 3. Per-File Detailed Breakdowns",
    "",
]
for sf in all_files:
    sub = out_df[out_df["source_file"] == sf]
    md_lines.append(f"### {sf} ({len(sub)} blank rows)\n")
    md_lines.append("| Row | Supplier Name | Classification | Fix Type |")
    md_lines.append("|-----|---------------|----------------|----------|")
    for _, r in sub.iterrows():
        name = str(r["supplier_name"])[:60].replace("|", "\\|")
        md_lines.append(f"| {r['source_row']} | {name} | `{r['classification']}` | `{r['fix_type']}` |")
    md_lines.append("")

md_lines += [
    "---",
    "",
    "## 4. Key Patterns and Engineering Recommendations",
    "",
    "### P1. GESPERRT/BLOCKED Prefix Strip",
    "7 rows across F7 have 'GESPERRT ' or 'BLOCKED - ' prepended. A single regex strip resolves these.",
    "```python",
    "name = re.sub(r'^(GESPERRT\\s+|BLOCKED\\s*-\\s*)', '', name, flags=re.IGNORECASE)",
    "```",
    "Expected yield: ~7 rows → `should_be_98` or `should_be_85`.",
    "",
    "### P2. Polish Unicode Normalization (PGL LP)",
    "7 rows in F6 are PAŃSTWOWE GOSPODARSTWO LEŚNE LASY (Polish State Forests) districts.",
    "The engine fails on Polish diacritics (ń, ę, ś, ą, ó, ź). A Unicode normalize + accent-fold pass resolves these.",
    "Expected yield: 7 rows → `should_be_85` (cluster 18 family).",
    "",
    "### P3. Name Noise Strip (***USE, WEB EDI, SIAMP CEDAP, - SGD, etc.)",
    "17+ rows carry internal procurement/routing tags appended to supplier names. Pattern:",
    "```python",
    r"name = re.sub(r'(\*+USE\s+\w+.*|\s+WEB EDI.*|\s+-\s+[A-Z]{3}$|SIAMP CEDAP.*)', '', name)",
    "```",
    "",
    "### P4. Same-Domain Bug",
    "Rows 10 and 200 in F1 both use `acumenmedia.com` — different trading names for the same entity.",
    "Rows 35-36 in F1 both use `angh.net`. Engine should flag same-domain pairs for 85% cluster.",
    "",
    "### P5. Pass 3c Extension (Descriptor Tokens)",
    "32 rows fail because descriptor tokens (BEVERAGE, LABELLING, IBERICA, TOKO, RHONE ALPES, etc.) prevent",
    "the core brand from matching. Pass 3c should strip known descriptor suffixes before comparison.",
    "",
    "### P6. Brand Alias CSV Additions Needed",
    "| Alias | Canonical | Cluster |",
    "|-------|-----------|---------|",
    "| B LAB FRANCE | B LAB COMPANY | 6 / 75019 |",
    "| B LAB SWITZERLAND | B LAB COMPANY | 6 / 1203 |",
    "| B LAB UK | B LAB COMPANY | 6 |",
    "| NGA | NATIONAL GOVERNORS ASSOCIATION | — |",
    "| DM-DROGERIA | DM DROGERIE MARKT | 2046 / 5071 |",
    "| SABA ARABIA | SABA & CO. | — (risky, needs review) |",
    "",
    "### P7. Acronym Expansion",
    "ETA → European Thyroid Association (F7 rows 22-23). Hopfengartenweg 19 address provides ground truth.",
    "An acronym lookup table resolves this without LLM.",
    "",
    "---",
    "",
    "## 5. Expected Phase 3 Outcome",
    "",
    "| Category | Rows | % of 273 |",
    "|----------|------|----------|",
]

pct = lambda n: f"{100*n/total:.1f}%"
for cls in all_cls:
    cnt = cls_counts.get(cls, 0)
    md_lines.append(f"| `{cls}` | {cnt} | {pct(cnt)} |")
md_lines.append(f"| **Total** | {total} | 100% |")

md_lines += [
    "",
    "If all `should_be_98`, `should_be_85`, and `should_be_70` rows are fixed via the engineering",
    "recommendations above:",
    "",
    f"- Recoverable via code (P1–P5): ~{cls_counts.get('should_be_98',0) + cls_counts.get('should_be_85',0)} rows",
    f"- Recoverable via alias CSVs:    ~{cls_counts.get('needs_brand_alias',0)} rows",
    f"- Requires LLM:                  ~{cls_counts.get('needs_llm',0)} rows",
    f"- True singletons (no fix):      ~{cls_counts.get('correctly_blank',0)} rows",
    f"- Review candidates (70%):       ~{cls_counts.get('should_be_70',0)} rows",
    "",
    "---",
    "*Report generated by `phase3_classify_blank_rows.py`*",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"Wrote: {OUT_MD}")

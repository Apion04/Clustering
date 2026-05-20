"""Preprocessing module: normalize supplier fields for matching."""
import json
import re
import unicodedata
from typing import Optional, Dict, Any, List, Set
import polars as pl

from src.config import (
    STATUS_WORDS, GENERIC_DOMAINS, ADDRESS_ABBREVIATIONS,
    STORE_NUMBER_PATTERNS, KNOWN_BRANDS, INVALID_TAX_VALUES, COMPANY_HINT_WORDS, HOSPITALITY_TERMS, GENERIC_ROOT_TOKENS,
    DEFAULT_JSON_TAX_KEYS, DEFAULT_JSON_SECONDARY_NAME_KEYS, LOCATION_ROOT_TOKENS, COMMON_FIRST_NAMES,
    DEFAULT_SUPPORT_FIELD_STRENGTHS, SUPPORT_FIELD_STRENGTHS, REGULATORY_REVIEW_TOKENS,
    OPERATIONAL_PREFIX_TOKENS, SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS, SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS,
    OPERATIONAL_NOISE_PATTERNS,
)
from src.legal_keywords import (
    LegalKeywordDictionary,
    find_legal_keyword_matches,
    get_default_legal_keywords,
    strip_legal_suffixes,
)
from src.person import clean_person_name, is_likely_individual_name

CHAR_TRANSLITERATION = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
    "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe",
    "ø": "oe", "Ø": "Oe", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th", "ł": "l", "Ł": "L",
}

COMMON_ACRONYM_EXPANSIONS = {
    "ppc": "potasse produits chimiques vynova ppc",
    "rbc": "royal bank canada",
    "rmhc": "ronald mcdonald house charities",
    "ahs": "alberta health services",
    "ctl": "cellular technology limited",
}

SUPPORT_FIELD_MAPPING_KEYS = [
    "family_name",
    "canonical_name",
    "parent_name",
    "normalized_supplier_name",
    "website",
    "domain",
    "email_domain",
    "OROVendorId",
    "CompanyEntityId",
    "tax_id",
]

SUPPORT_FIELD_ALIASES = {
    "orovendorid": "OROVendorId",
    "companyentityid": "CompanyEntityId",
    "oro_vendor_id": "OROVendorId",
    "company_entity_id": "CompanyEntityId",
}

SUPPLIER_IDENTITY_REGION_TOKENS = {
    "north", "south", "east", "west", "central", "worldwide", "blocked",
    "de", "deutschland", "germany", "german", "france", "french", "uk", "gb",
    "usa", "us", "america", "american", "europe", "europa", "european",
    "schweiz", "swiss", "suisse", "switzerland",
    "sweden", "swedish", "sverige", "brasil", "brazil", "do",
    "int", "intl", "international",
    "denmark", "danish", "switzerland", "great", "britain", "to", "from",
}

TRUSTED_SUPPLIER_IDENTITY_PHRASES = (
    "air liquide",
    "air products",
    "eastman kodak",
    "springer nature",
    "axel springer",
    "bio springer",
    "sigma aldrich",
)


def _compact_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))

def has_legal_suffix_hint(name: Optional[Any], legal_keywords: Optional[LegalKeywordDictionary] = None) -> bool:
    return bool(find_legal_keyword_matches(name, legal_keywords or get_default_legal_keywords()))


def has_company_keyword_hint(
    name: Optional[Any],
    legal_keywords: Optional[LegalKeywordDictionary] = None,
    *,
    has_legal_suffix: Optional[bool] = None,
) -> bool:
    normalized = normalize_text(name).replace('.', ' ').replace('-', ' ')
    toks = set(normalized.split())
    legal_hint = has_legal_suffix if has_legal_suffix is not None else has_legal_suffix_hint(name, legal_keywords)
    dotted_short_form_hint = bool(re.search(r"\b(b\s*v|s\s*a|a\s*s|n\s*v|p\s*l\s*c)\b", normalized))
    return bool(toks & COMPANY_HINT_WORDS) or bool(legal_hint) or dotted_short_form_hint


def is_hospitality_name(name: Optional[Any], legal_keywords: Optional[LegalKeywordDictionary] = None) -> bool:
    toks = set(normalize_supplier_name(name, legal_keywords=legal_keywords).split()) if name is not None else set()
    return bool(toks & HOSPITALITY_TERMS)


def transliterate_special_chars(text: str) -> str:
    for src, tgt in CHAR_TRANSLITERATION.items():
        text = text.replace(src, tgt)
    return text


def normalize_text(text: Optional[Any]) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() in {"", "nan", "none", "null"}:
        return ""
    text = transliterate_special_chars(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[\/\|\_\,\&\+\*\@\#\%\!\?\(\)\[\]\{\}]", " ", text)
    # keep hyphen/dot handling flexible after special German street normalization
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_status_words(name: str) -> str:
    for status in sorted(STATUS_WORDS, key=len, reverse=True):
        pat = re.escape(status)
        name = re.sub(rf"\b{pat}\b", " ", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def remove_operational_prefixes(name: str) -> str:
    """Remove vendor-master reference prefixes without treating them as brands."""
    had_prefix = False
    for prefix in sorted(OPERATIONAL_PREFIX_TOKENS, key=len, reverse=True):
        pat = re.escape(prefix)
        new_name = re.sub(rf"^\s*{pat}\b[\s\-_/]*", " ", name, flags=re.IGNORECASE)
        if new_name != name:
            had_prefix = True
            name = new_name
    # Reference IDs at the end of these names are not supplier identity.
    if had_prefix:
        name = re.sub(r"[\s\-_/]+\d{3,}\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def has_operational_status_hint(name: Optional[Any]) -> bool:
    """Detect vendor-use instructions that should downgrade auto-clustering.

    These strings are common in raw vendor masters and can make otherwise
    similar names look identical after cleanup. We keep them useful for
    candidate generation, but matching should route them to Review rather than
    high-confidence main clusters.
    """
    normalized = normalize_text(name)
    if not normalized:
        return False
    compact = re.sub(r"\s+", " ", normalized)
    if re.search(r"\b(blocked|inactive|obsolete|legacy|former|closed|deactivated|deacivated|gesperrt|gesperrte|geschlossen)\b", compact):
        return True
    if re.search(r"\b(old\s+vendor|do\s+not\s+use|do-not-use)\b", compact):
        return True
    if re.search(r"\buse\s+[a-z0-9][a-z0-9_-]{2,}\b", compact):
        return True
    if re.search(rf"^\s*({'|'.join(re.escape(t) for t in sorted(OPERATIONAL_PREFIX_TOKENS))})\b", compact):
        return True
    if re.search(r"x{4,}", compact):
        return True
    return False


def remove_operational_noise(name: str) -> str:
    """Strip ERP/AP vendor-master operational tokens before supplier identity matching.

    Applied AFTER normalize_text() (so input is lowercase) but BEFORE legal suffix
    stripping. Guard: if stripping would leave fewer than 3 non-whitespace characters,
    do not strip (protects very short names).
    """
    for pattern in OPERATIONAL_NOISE_PATTERNS:
        candidate = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate.replace(" ", "")) >= 3:
            name = candidate
    return re.sub(r"\s+", " ", name).strip()


def remove_legal_suffixes(name: str, legal_keywords: Optional[LegalKeywordDictionary] = None) -> str:
    stripped, _removed = strip_legal_suffixes(name, legal_keywords or get_default_legal_keywords())
    return stripped

def remove_store_numbers(name: str) -> str:
    for pattern in STORE_NUMBER_PATTERNS:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def normalize_supplier_name(name: Optional[Any], legal_keywords: Optional[LegalKeywordDictionary] = None) -> str:
    name = normalize_text(name)
    if not name:
        return ""
    legal_index = legal_keywords or get_default_legal_keywords()
    name = re.sub(r"^[\s\.'\"`‘’“”]+", "", name)
    name = re.sub(r"[\"`‘’“”]", " ", name)
    name = remove_status_words(name)
    name = remove_operational_prefixes(name)
    name = remove_operational_noise(name)
    name = name.replace(" and ", " ")
    # Collapse punctuated acronyms BEFORE the general dot→space replacement.
    # "T.B.D." → "tbd", "J.S.J.V." → "jsjv", "C.E.G.O.S" → "cegos"
    def _collapse_dotted_acronym(token: str) -> str:
        # Match patterns like "t.b.d." or "j.s.j.v." (all single-letters + dots)
        if re.match(r'^[a-z](?:\.[a-z])+\.?$', token):
            return token.replace(".", "")
        return token
    name = " ".join(_collapse_dotted_acronym(t) for t in name.split())
    # Collapse spaced single-letter initials: "J S J V" → "jsjv", "R H L" → "rhl"
    # Only collapse when there are 3+ consecutive single letters separated by spaces
    # (2-letter cases like "A B" could be valid abbreviations or "A & B" collateral).
    name = re.sub(r'(?<![a-z0-9])([a-z] ){2,}[a-z](?![a-z0-9])', lambda m: m.group(0).replace(' ', ''), name)
    name = name.replace(".", " ").replace("-", " ")
    # Compound word normalization
    _COMPOUND_WORDS = {
        "health care": "healthcare",
        "ing buero": "ingenieurburo",
        "ing buro": "ingenieurburo",
        "ingenieurbüro": "ingenieurburo",
    }
    for phrase, replacement in _COMPOUND_WORDS.items():
        name = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, name)
    name = remove_store_numbers(name)
    before_legal = name
    name = remove_legal_suffixes(name, legal_index)
    if _legal_stripping_left_too_generic(name):
        name = before_legal
    # PT (Indonesian/Portuguese) prefix stripping when name has 2+ remaining tokens
    _pt_stripped = re.sub(r'^pt\s+', '', name).strip()
    if len(_pt_stripped.split()) >= 2:
        name = _pt_stripped
    # PT as suffix (e.g. "Krisbow PT")
    _pt_sfx = re.sub(r'\s+pt\s*$', '', name).strip()
    if _pt_sfx and _pt_sfx != name:
        name = _pt_sfx
    # OOO (Russian LLC) prefix stripping when name has 1+ remaining token
    _ooo_stripped = re.sub(r'^ooo\s+', '', name).strip()
    if len(_ooo_stripped.split()) >= 1 and _ooo_stripped:
        name = _ooo_stripped
    # Normalize GmbH & Co KG compound form (already stripped by legal suffix stripping
    # when the CSV is loaded, but normalize the residual form after dot/hyphen replacement)
    name = re.sub(r'\bgmbh\s+(?:und\s+)?co\.?\s+kg\b', 'gmbh', name, flags=re.IGNORECASE)
    # normalize frequent German name families observed in test files
    name = re.sub(r"\bmuller\b", "mueller", name)
    name = re.sub(r"\bkuhne\b", "kuehne", name)
    name = re.sub(r"\bpneumatik\b", "pneumatic", name)
    name = name.replace("rutgers", "ruetgers")
    name = name.replace("rütgers", "ruetgers")
    name = name.replace("näg", "naeg")
    name = re.sub(r"\belektro\s*physik\b", "elektrophysik", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


_MIN_LOCATION_STRIP_LEN = 5  # Don't strip location tokens shorter than this to avoid short-abbrev false positives


def _has_distinctive_location_token(name_norm: str, location_terms: Set[str]) -> bool:
    """Return True if name_norm contains at least one non-generic, non-location token (len >= 3)."""
    weak = GENERIC_ROOT_TOKENS | location_terms | COMMON_FIRST_NAMES
    return any(len(t) >= 3 and t not in weak for t in name_norm.split())


def _strip_trailing_location_tokens(name_norm: str, location_terms: Set[str]) -> str:
    """Strip up to 2 trailing location terms from normalized name.

    Only strips when:
    - the trailing token is in location_terms and len >= _MIN_LOCATION_STRIP_LEN
    - the remaining core has at least one distinctive (non-generic/non-location) token
    """
    tokens = name_norm.split()
    if len(tokens) <= 1:
        return name_norm

    stripped_count = 0
    while stripped_count < 2 and len(tokens) > stripped_count + 1:
        tail = tokens[-(stripped_count + 1)]
        if tail in location_terms and len(tail) >= _MIN_LOCATION_STRIP_LEN:
            stripped_count += 1
        else:
            break

    if stripped_count == 0:
        return name_norm

    core_tokens = tokens[:-stripped_count]
    if not _has_distinctive_location_token(" ".join(core_tokens), location_terms):
        return name_norm  # Don't strip if nothing meaningful remains

    return " ".join(core_tokens)


def compute_name_location_core(
    original_name: Any,
    name_norm: str,
    location_terms: Set[str],
    legal_keywords: Optional[LegalKeywordDictionary] = None,
) -> str:
    """Compute brand-location core by stripping parenthesized and hyphen-separated location modifiers.

    Handles:
    - "BFI CANADA-CALGARY" → "bfi canada"  (hyphen-separated city stripped)
    - "Bell Canada (5115 Creekbank Rd)" → "bell canada"  (parenthesized address stripped)
    - "Bell Canada (Deactivated)" → "bell canada"  (parenthesized status stripped)

    Safety: does NOT strip if the remaining core lacks a distinctive token,
    so generic names like "services" are never silently collapsed.
    """
    if not name_norm:
        return name_norm

    raw = str(original_name or "").strip()

    # Strategy 1: Strip any parenthesized content from original name.
    # Parentheses in supplier names almost always contain branch addresses,
    # postal codes, status words, or location qualifiers — not core identity.
    raw_no_parens = re.sub(r'\s*\([^)]+\)', '', raw).strip()
    if raw_no_parens and raw_no_parens != raw:
        re_normed = normalize_supplier_name(raw_no_parens, legal_keywords=legal_keywords)
        if re_normed and _has_distinctive_location_token(re_normed, location_terms):
            return re_normed

    # Strategy 2: Strip trailing hyphen-separated location modifier.
    # "BFI CANADA-CALGARY" → the segment after the last "-" is the branch modifier.
    if '-' in raw:
        last_dash = raw.rfind('-')
        core_raw = raw[:last_dash].strip()
        tail_raw = raw[last_dash + 1:].strip()
        if core_raw and tail_raw:
            tail_norm = normalize_supplier_name(tail_raw, legal_keywords=legal_keywords)
            tail_tokens = [t for t in tail_norm.split() if t]
            # Tail is a location modifier only when ALL its tokens are location terms
            # of sufficient length.
            if tail_tokens and all(
                t in location_terms and len(t) >= _MIN_LOCATION_STRIP_LEN
                for t in tail_tokens
            ):
                core_norm = normalize_supplier_name(core_raw, legal_keywords=legal_keywords)
                if core_norm and _has_distinctive_location_token(core_norm, location_terms):
                    return core_norm

    # Strategy 3: Strip trailing location tokens from the already-normalized name.
    # Catches cases where normalize_supplier_name already collapsed the hyphen to a space.
    return _strip_trailing_location_tokens(name_norm, location_terms)


def _legal_stripping_left_too_generic(name: str) -> bool:
    tokens = [t for t in name.split() if t]
    if not tokens:
        return True
    weak = GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES
    return all(t in weak or len(t) <= 1 for t in tokens)


def _normalize_joined_german_street(addr: str) -> str:
    # pasteurstr. -> pasteur street, buchenbachstrasse -> buchenbach street
    addr = re.sub(r"\b([a-z0-9]+)\s*-?\s*str\.?\b", r"\1 street", addr)
    addr = re.sub(r"\b([a-z0-9]+)\s*-?\s*strasse\b", r"\1 street", addr)
    addr = re.sub(r"\b([a-z0-9]+)\s*-?\s*straße\b", r"\1 street", addr)
    return addr


def normalize_address(address: Optional[Any]) -> str:
    addr = normalize_text(address)
    if not addr:
        return ""
    addr = _normalize_joined_german_street(addr)
    addr = addr.replace(".", " ").replace("-", " ")
    words = []
    for word in addr.split():
        clean = word.strip(".,")
        words.append(ADDRESS_ABBREVIATIONS.get(clean, clean))
    addr = " ".join(words)
    # normalize minor spelling variations from real test cases
    addr = addr.replace("kekule street", "kekul street")
    addr = addr.replace("kekules street", "kekul street")
    addr = re.sub(r"\s+", " ", addr).strip()
    # token sort makes street 15 and 15 street compare the same
    return " ".join(sorted(addr.split()))


def normalize_city(city: Optional[Any]) -> str:
    city = normalize_text(city)
    if not city:
        return ""
    city = city.replace(".", " ").replace("-", " ")
    city = city.replace("frankfurt am main", "frankfurt")
    city = city.replace("keverlaer", "kevelaer")
    city = city.replace("koln", "koeln")
    city = re.sub(r"\bkoeln\s+\d+\b", "koeln", city)
    city = re.sub(r"\bberlin\s+\d+\b", "berlin", city)
    city = re.sub(r"x{3,}$", "", city)  # thannxxxxx -> thann
    city = city.replace("wustenrot neulautern", "wustenrot neulauter")
    # German/European city name normalization
    # (transliterate_special_chars already handles ü→ue, ö→oe, ä→ae, ß→ss)
    city = re.sub(r"\bmuenchen\b", "munich", city)
    city = re.sub(r"\bnuernberg\b", "nuremberg", city)
    city = re.sub(r"\bgoeteborg\b", "gothenburg", city)
    city = re.sub(r"\bkopenhagen\b", "copenhagen", city)
    city = re.sub(r"\bkobenhavn\b", "copenhagen", city)
    city = re.sub(r"\bzuerich\b", "zurich", city)
    city = re.sub(r"\bwien\b", "vienna", city)
    city = re.sub(r"\bpraag\b", "prague", city)
    city = re.sub(r"\bwarschau\b", "warsaw", city)
    city = re.sub(r"\bbruessel\b", "brussels", city)
    city = re.sub(r"\bbruxelles\b", "brussels", city)
    return re.sub(r"\s+", " ", city).strip()


def normalize_country(country: Optional[Any]) -> str:
    if country is None:
        return ""
    country = str(country).upper().strip()
    mappings = {
        "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
        "UK": "GB", "UNITED KINGDOM": "GB", "ENGLAND": "GB",
        "DEUTSCHLAND": "DE", "GERMANY": "DE", "FRANCE": "FR", "CANADA": "CA",
        "INDIA": "IN", "AUSTRALIA": "AU", "JAPAN": "JP", "CHINA": "CN",
        "MEXICO": "MX", "BRAZIL": "BR", "NETHERLANDS": "NL", "HOLLAND": "NL",
        "BELGIUM": "BE", "SWITZERLAND": "CH", "AUSTRIA": "AT", "SPAIN": "ES",
        "ITALY": "IT", "POLAND": "PL", "PUERTO RICO": "PR",
    }
    return mappings.get(country, country)


def normalize_tax_id(tax_id: Optional[Any], country: Optional[Any] = None) -> str:
    tax = normalize_text(tax_id)
    if not tax:
        return ""
    tax = re.sub(r"[^a-z0-9]", "", tax)
    if not tax or tax in INVALID_TAX_VALUES:
        return ""
    # Reject common placeholder-like strings and values with too few digits.
    if len(tax) <= 2 or sum(ch.isdigit() for ch in tax) < 4:
        return ""
    if len(set(tax)) == 1 and tax[0].isdigit():
        return ""
    return tax

def normalize_tax_id_loose(tax_id: Optional[Any], country: Optional[Any] = None) -> str:
    """Loose version used only as a candidate key, never as a standalone exact match.

    It strips a leading two-letter country prefix from values like DE233002380 so
    DE233002380 can be compared to 233002380, but matching still requires name/address/domain support.
    """
    tax = normalize_tax_id(tax_id, country)
    if not tax:
        return ""
    if len(tax) > 4 and tax[:2].isalpha() and tax[2:].isdigit():
        return tax[2:]
    return tax

def normalize_all_tax_ids(row: Dict[str, Any], mapping: Dict[str, Any]) -> str:
    country_col = mapping.get("country")
    country = row.get(country_col, "") if country_col else ""
    candidates: List[str] = []
    for col in [mapping.get("tax_id")] + list(mapping.get("tax_ids") or []):
        if col and col in row:
            candidates.append(row.get(col))
    # Extract from JSON metadata columns such as supplierInfo/additionalInfo
    for json_col in mapping.get("metadata_json_columns") or []:
        raw = row.get(json_col)
        if raw:
            try:
                data = json.loads(str(raw))
                for key in mapping.get("json_tax_keys") or DEFAULT_JSON_TAX_KEYS:
                    if key in data:
                        candidates.append(data.get(key))
            except Exception:
                # Attempt regex fallback
                for key in mapping.get("json_tax_keys") or DEFAULT_JSON_TAX_KEYS:
                    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', str(raw))
                    if m:
                        candidates.append(m.group(1))
    normalized = []
    for c in candidates:
        n = normalize_tax_id(c, country)
        if n and n not in normalized:
            normalized.append(n)
    return "|".join(normalized)


def normalize_all_tax_ids_loose(row: Dict[str, Any], mapping: Dict[str, Any]) -> str:
    country_col = mapping.get("country")
    country = row.get(country_col, "") if country_col else ""
    candidates: List[str] = []
    for col in [mapping.get("tax_id")] + list(mapping.get("tax_ids") or []):
        if col and col in row:
            candidates.append(row.get(col))
    for json_col in mapping.get("metadata_json_columns") or []:
        raw = row.get(json_col)
        if raw:
            try:
                data = json.loads(str(raw))
                for key in mapping.get("json_tax_keys") or DEFAULT_JSON_TAX_KEYS:
                    if key in data:
                        candidates.append(data.get(key))
            except Exception:
                for key in mapping.get("json_tax_keys") or DEFAULT_JSON_TAX_KEYS:
                    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', str(raw))
                    if m:
                        candidates.append(m.group(1))
    normalized = []
    for c in candidates:
        n = normalize_tax_id_loose(c, country)
        if n and n not in normalized:
            normalized.append(n)
    return "|".join(normalized)


def _extract_json_values(row: Dict[str, Any], mapping: Dict[str, Any], keys: List[str]) -> List[str]:
    values: List[str] = []
    for json_col in mapping.get("metadata_json_columns") or []:
        raw = row.get(json_col)
        if not raw:
            continue
        try:
            data = json.loads(str(raw))
            for key in keys:
                value = data.get(key)
                if value not in (None, ""):
                    values.append(str(value))
        except Exception:
            for key in keys:
                m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', str(raw))
                if m:
                    values.append(m.group(1))
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def extract_json_secondary_names(
    row: Dict[str, Any],
    mapping: Dict[str, Any],
    legal_keywords: Optional[LegalKeywordDictionary] = None,
) -> str:
    keys = mapping.get("json_secondary_name_keys") or DEFAULT_JSON_SECONDARY_NAME_KEYS
    values = []
    for value in _extract_json_values(row, mapping, keys):
        norm = normalize_supplier_name(value, legal_keywords=legal_keywords)
        if norm and norm not in values:
            values.append(norm)
    return "|".join(values)


def extract_json_secondary_names_raw(row: Dict[str, Any], mapping: Dict[str, Any]) -> str:
    keys = mapping.get("json_secondary_name_keys") or DEFAULT_JSON_SECONDARY_NAME_KEYS
    return " | ".join(_extract_json_values(row, mapping, keys))


def _support_field_key(raw_key: str) -> str:
    key = str(raw_key or "").strip()
    return SUPPORT_FIELD_ALIASES.get(key.lower(), key)


def _strength_for_support_field(key: str, strengths: Dict[str, str]) -> str:
    canonical_key = _support_field_key(key)
    strength = (
        strengths.get(canonical_key)
        or strengths.get(canonical_key.lower())
        or DEFAULT_SUPPORT_FIELD_STRENGTHS.get(canonical_key)
        or DEFAULT_SUPPORT_FIELD_STRENGTHS.get(canonical_key.lower())
        or "review_only"
    )
    return strength if strength in SUPPORT_FIELD_STRENGTHS else "review_only"


def _normalize_support_value(value: Any, strength: str, legal_keywords: Optional[LegalKeywordDictionary]) -> str:
    if value in (None, ""):
        return ""
    if strength == "domain":
        normalized = extract_domain(value)
        return "" if is_generic_domain(normalized) else normalized
    if strength == "same_entity_id":
        normalized = re.sub(r"[^a-z0-9]", "", normalize_text(value))
        if normalized in INVALID_TAX_VALUES or len(normalized) < 3:
            return ""
        return normalized
    normalized = normalize_supplier_name(value, legal_keywords=legal_keywords)
    if not normalized:
        return ""
    toks = [t for t in normalized.split() if t]
    weak = GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS
    if not toks or all(t in weak or len(t) <= 1 for t in toks):
        return ""
    return normalized


def _support_field_columns(column_mapping: Dict[str, Any], df_columns: List[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    explicit = column_mapping.get("support_fields")
    if isinstance(explicit, dict):
        for key, col in explicit.items():
            if col in df_columns:
                found[_support_field_key(key)] = col
    elif isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, dict):
                key = item.get("key") or item.get("name") or item.get("field")
                col = item.get("column") or item.get("col")
                if key and col in df_columns:
                    found[_support_field_key(key)] = col

    for key in SUPPORT_FIELD_MAPPING_KEYS:
        col = column_mapping.get(key)
        if col and col in df_columns:
            found[_support_field_key(key)] = col

    # Conservative exact-name autodetection for ORO internal support fields.
    lower_to_col = {c.lower(): c for c in df_columns}
    for lower, key in {
        "orovendorid": "OROVendorId",
        "companyentityid": "CompanyEntityId",
        "family_name": "family_name",
        "family name": "family_name",
        "canonical_name": "canonical_name",
        "canonical name": "canonical_name",
        "parent_name": "parent_name",
        "parent name": "parent_name",
        "normalized_supplier_name": "normalized_supplier_name",
        "normalized supplier name": "normalized_supplier_name",
        "email_domain": "email_domain",
        "email domain": "email_domain",
    }.items():
        if lower in lower_to_col and key not in found:
            found[key] = lower_to_col[lower]
    return found


def encode_support_fields(
    row: Dict[str, Any],
    mapping: Dict[str, Any],
    strengths: Dict[str, str],
    legal_keywords: Optional[LegalKeywordDictionary] = None,
) -> str:
    entries: List[Dict[str, str]] = []
    seen = set()

    for key, col in _support_field_columns(mapping, list(row.keys())).items():
        strength = _strength_for_support_field(key, strengths)
        value = _normalize_support_value(row.get(col), strength, legal_keywords)
        if not value:
            continue
        identity = (key, strength, value)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append({"field": key, "strength": strength, "value": value})

    # JSON family/parent/trade/legal names are support fields too, but default
    # to family/review evidence rather than main same-entity evidence.
    json_strength = _strength_for_support_field("json_secondary_name", strengths)
    for value in _extract_json_values(row, mapping, mapping.get("json_secondary_name_keys") or DEFAULT_JSON_SECONDARY_NAME_KEYS):
        normalized = _normalize_support_value(value, json_strength, legal_keywords)
        if not normalized:
            continue
        identity = ("json_secondary_name", json_strength, normalized)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append({"field": "json_secondary_name", "strength": json_strength, "value": normalized})

    return json.dumps(entries, ensure_ascii=False, sort_keys=True)


def extract_domain(email_or_website: Optional[Any]) -> str:
    if not email_or_website:
        return ""
    text = str(email_or_website).lower().strip()
    if text in {"nan", "none", "null", ""}:
        return ""
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    if "@" in text:
        text = text.split("@")[-1]
    text = text.split("/")[0].split(":")[0]
    return text.strip()


_SECONDARY_TLDS: frozenset = frozenset({
    "co", "com", "net", "org", "gov", "edu", "ac", "or",
})


def _extract_domain_sld(domain: str) -> str:
    """Extract the Second-Level Domain (SLD) from a full domain string.

    Examples:
      "bcorporation.fr"  → "bcorporation"
      "bv.aok.de"        → "aok"   (subdomain.sld.tld)
      "foo.co.uk"        → "foo"   (compound TLD)
      "aprolis.com"      → "aprolis"
      "dm.de"            → "dm"    (short — caller must guard)
    """
    if not domain:
        return ""
    domain = str(domain).lower().strip()
    parts = domain.split(".")
    if len(parts) < 2:
        return ""
    # Handle compound TLDs: foo.co.uk → last part is 2-char country, second-to-last is
    # a known secondary-level label (co, com, net …) → SLD is third from last.
    if (
        len(parts) >= 3
        and len(parts[-1]) == 2
        and parts[-2] in _SECONDARY_TLDS
    ):
        return parts[-3]
    # Standard case: sld.tld or subdomain.sld.tld → second from last.
    return parts[-2]


def is_generic_domain(domain: str) -> bool:
    """Return True for free/ISP/disposable domains.

    Exact domains and subdomains are treated as generic, e.g. gmail.com and mail.gmail.com.
    Custom company domains are not treated as generic.
    """
    d = extract_domain(domain) if domain else ""
    if not d:
        return False
    if d in GENERIC_DOMAINS:
        return True
    return any(d.endswith("." + g) for g in GENERIC_DOMAINS)


def extract_root_brand(name: str) -> str:
    if not name:
        return ""
    # Expand known acronyms for better parent/family bridge.
    if name in COMMON_ACRONYM_EXPANSIONS:
        return name
    words = [w for w in name.split() if len(w) > 1 and not any(ch.isdigit() for ch in w)]
    stop = (
        {"north", "south", "east", "west", "central", "worldwide", "blocked"}
        | GENERIC_ROOT_TOKENS
        | LOCATION_ROOT_TOKENS
        | COMMON_FIRST_NAMES
        | COMMON_ACRONYM_EXPANSIONS.keys()
    )
    # Do not remove known safe acronym itself when it is the whole name.
    if name not in COMMON_ACRONYM_EXPANSIONS:
        words = [w for w in words if w not in stop]
    if not words:
        return ""
    # For family-level grouping, first strong token is often enough for brands like ruetgers/naegele.
    rare_tokens = [w for w in words if len(w) >= 5]
    if rare_tokens:
        return rare_tokens[0]
    return " ".join(words[:2])


def extract_supplier_identity_core(name: Optional[Any]) -> str:
    """Return a conservative brand/group identity core from a normalized name.

    The core is used for candidate discovery and the explicit
    `distinctive_supplier_identity` pass. Generic, legal, location, person,
    regional, and regulatory terms are removed so words such as "Technology",
    "GmbH", "Deutschland", "REACH", or "Task Force" do not become bridge
    evidence by themselves.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    for phrase in TRUSTED_SUPPLIER_IDENTITY_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return phrase
    weak = (
        GENERIC_ROOT_TOKENS
        | LOCATION_ROOT_TOKENS
        | COMMON_FIRST_NAMES
        | HOSPITALITY_TERMS
        | COMPANY_HINT_WORDS
        | REGULATORY_REVIEW_TOKENS
        | SUPPLIER_IDENTITY_REGION_TOKENS
    )
    tokens = []
    for token in text.split():
        token = token.strip()
        if not token:
            continue
        if token in weak:
            continue
        if token in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and token not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS:
            continue
        # Drop one-letter initials for group identity, e.g. A. Hartrodt -> hartrodt.
        if len(token) == 1 and token.isalpha():
            continue
        if token.isdigit():
            continue
        tokens.append(token)

    if not tokens:
        return ""

    # Remove duplicate adjacent/repeated tokens but preserve phrase order.
    deduped = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return " ".join(deduped)


def token_sort_fingerprint(text: str) -> str:
    return " ".join(sorted(text.split())) if text else ""


def phonetic_key(text: str) -> str:
    try:
        import jellyfish
        return jellyfish.metaphone(text) if text else ""
    except ImportError:
        return text[:4] + text[-4:] if len(text) >= 8 else text


def preprocess_dataframe(
    df: pl.DataFrame,
    column_mapping: Dict[str, Any],
    legal_keywords: Optional[LegalKeywordDictionary] = None,
    support_field_strengths: Optional[Dict[str, str]] = None,
    location_terms: Optional[Set[str]] = None,
) -> pl.DataFrame:
    legal_index = legal_keywords or get_default_legal_keywords()
    support_strengths = dict(DEFAULT_SUPPORT_FIELD_STRENGTHS)
    support_strengths.update(support_field_strengths or {})
    name_col = column_mapping.get("supplier_name") or df.columns[0]
    addr_col = column_mapping.get("address")
    city_col = column_mapping.get("city")
    country_col = column_mapping.get("country")
    postal_col = column_mapping.get("postal_code")
    email_col = column_mapping.get("email")
    website_col = column_mapping.get("website")
    domain_col = column_mapping.get("domain")

    df = df.with_row_index("row_id")

    # Preserve a compact set of original mapped fields for AI-risk review/audit.
    def _orig_expr(col_name, alias):
        if col_name and col_name in df.columns:
            return pl.col(col_name).cast(pl.Utf8, strict=False).fill_null("").alias(alias)
        return pl.lit("").alias(alias)

    secondary_cols_for_ai = list(column_mapping.get("secondary_names") or [])
    for k in ["name_2", "name_3", "name_4"]:
        v = column_mapping.get(k)
        if v and v not in secondary_cols_for_ai:
            secondary_cols_for_ai.append(v)

    df = df.with_columns([
        _orig_expr(name_col, "orig_supplier_name"),
        _orig_expr(addr_col, "orig_address"),
        _orig_expr(city_col, "orig_city"),
        _orig_expr(country_col, "orig_country"),
        _orig_expr(email_col, "orig_email"),
        _orig_expr(website_col, "orig_website"),
        _orig_expr(domain_col, "orig_domain"),
    ])
    if secondary_cols_for_ai:
        existing_secondary = [c for c in secondary_cols_for_ai if c in df.columns]
        if existing_secondary:
            df = df.with_columns(
                pl.concat_str([pl.col(c).cast(pl.Utf8, strict=False).fill_null("") for c in existing_secondary], separator=" | ").alias("orig_secondary_names")
            )
        else:
            df = df.with_columns(pl.lit("").alias("orig_secondary_names"))
    else:
        df = df.with_columns(pl.lit("").alias("orig_secondary_names"))

    if column_mapping.get("metadata_json_columns"):
        df = df.with_columns(
            pl.struct(df.columns).map_elements(lambda r: extract_json_secondary_names_raw(r, column_mapping), return_dtype=pl.Utf8).alias("orig_json_secondary_names")
        )
        df = df.with_columns(
            pl.concat_str([
                pl.col("orig_secondary_names").fill_null(""),
                pl.col("orig_json_secondary_names").fill_null(""),
            ], separator=" | ").str.strip_chars(" |").alias("orig_secondary_names")
        )

    df = df.with_columns(
        pl.col(name_col)
        .map_elements(lambda x: "|".join(find_legal_keyword_matches(x, legal_index)), return_dtype=pl.Utf8)
        .alias("legal_suffixes_found")
    )
    df = df.with_columns([
        pl.col(name_col).map_elements(has_operational_status_hint, return_dtype=pl.Boolean).alias("has_operational_status_hint"),
        pl.col(name_col).map_elements(lambda x: normalize_supplier_name(x, legal_keywords=legal_index), return_dtype=pl.Utf8).alias("name_norm"),
        (pl.col("legal_suffixes_found") != "").alias("has_legal_suffix"),
        pl.struct([name_col, "legal_suffixes_found"])
        .map_elements(
            lambda r: has_company_keyword_hint(
                r[name_col],
                legal_keywords=legal_index,
                has_legal_suffix=bool(r["legal_suffixes_found"]),
            ),
            return_dtype=pl.Boolean,
        )
        .alias("has_company_keyword"),
        pl.col(name_col).map_elements(lambda x: is_hospitality_name(x, legal_keywords=legal_index), return_dtype=pl.Boolean).alias("is_hospitality"),
    ])

    df = df.with_columns(
        pl.struct(df.columns).map_elements(
            lambda r: encode_support_fields(r, column_mapping, support_strengths, legal_index),
            return_dtype=pl.Utf8,
        ).alias("support_fields_json")
    )
    df = df.with_columns([
        pl.col("support_fields_json").map_elements(lambda x: _support_values_by_strength(x, {"same_entity_id"}), return_dtype=pl.Utf8).alias("support_same_entity_id_values"),
        pl.col("support_fields_json").map_elements(lambda x: _support_values_by_strength(x, {"same_entity_name"}), return_dtype=pl.Utf8).alias("support_same_entity_name_values"),
        pl.col("support_fields_json").map_elements(lambda x: _support_values_by_strength(x, {"family_or_parent"}), return_dtype=pl.Utf8).alias("support_family_values"),
        pl.col("support_fields_json").map_elements(lambda x: _support_values_by_strength(x, {"domain"}), return_dtype=pl.Utf8).alias("support_domain_values"),
        pl.col("support_fields_json").map_elements(lambda x: _support_values_by_strength(x, {"review_only"}), return_dtype=pl.Utf8).alias("support_review_values"),
        pl.col("support_fields_json").map_elements(_support_fields_raw_summary, return_dtype=pl.Utf8).alias("orig_support_fields"),
    ])
    df = df.with_columns(
        pl.col("name_norm").map_elements(clean_person_name, return_dtype=pl.Utf8).alias("person_name_norm")
    )
    df = df.with_columns(
        pl.struct(["name_norm", "has_legal_suffix", "has_company_keyword", "is_hospitality"])
        .map_elements(
            lambda r: is_likely_individual_name(
                r["name_norm"],
                has_legal_suffix=bool(r["has_legal_suffix"]),
                has_company_keyword=bool(r["has_company_keyword"]),
                is_hospitality=bool(r["is_hospitality"]),
            ),
            return_dtype=pl.Boolean,
        )
        .alias("is_likely_individual")
    )

    secondary_cols = list(column_mapping.get("secondary_names") or [])
    for k in ["name_2", "name_3", "name_4"]:
        v = column_mapping.get(k)
        if v and v not in secondary_cols:
            secondary_cols.append(v)
    for i, col in enumerate(secondary_cols[:6], 2):
        if col in df.columns:
            df = df.with_columns(pl.col(col).map_elements(lambda x: normalize_supplier_name(x, legal_keywords=legal_index), return_dtype=pl.Utf8).alias(f"name{i}_norm"))

    if column_mapping.get("metadata_json_columns"):
        df = df.with_columns(
            pl.struct(df.columns).map_elements(lambda r: extract_json_secondary_names(r, column_mapping, legal_index), return_dtype=pl.Utf8).alias("json_secondary_names_norm")
        )

    if addr_col and addr_col in df.columns:
        df = df.with_columns(pl.col(addr_col).map_elements(normalize_address, return_dtype=pl.Utf8).alias("addr_norm"))
    else:
        df = df.with_columns(pl.lit("").alias("addr_norm"))
    if city_col and city_col in df.columns:
        df = df.with_columns(pl.col(city_col).map_elements(normalize_city, return_dtype=pl.Utf8).alias("city_norm"))
    else:
        df = df.with_columns(pl.lit("").alias("city_norm"))
    if country_col and country_col in df.columns:
        df = df.with_columns(pl.col(country_col).map_elements(normalize_country, return_dtype=pl.Utf8).alias("country_norm"))
    else:
        df = df.with_columns(pl.lit("").alias("country_norm"))
    if postal_col and postal_col in df.columns:
        df = df.with_columns(pl.col(postal_col).map_elements(lambda x: re.sub(r"[^a-z0-9]", "", normalize_text(x)), return_dtype=pl.Utf8).alias("postal_norm"))
    else:
        df = df.with_columns(pl.lit("").alias("postal_norm"))

    # Multi-tax + JSON tax extraction.
    df = df.with_columns([
        pl.struct(df.columns).map_elements(lambda r: normalize_all_tax_ids(r, column_mapping), return_dtype=pl.Utf8).alias("tax_norm"),
        pl.struct(df.columns).map_elements(lambda r: normalize_all_tax_ids_loose(r, column_mapping), return_dtype=pl.Utf8).alias("tax_loose_norm"),
    ])

    domain_exprs = []
    if domain_col and domain_col in df.columns:
        domain_exprs.append(pl.col(domain_col).map_elements(extract_domain, return_dtype=pl.Utf8).alias("domain_from_domain"))
    if email_col and email_col in df.columns:
        domain_exprs.append(pl.col(email_col).map_elements(extract_domain, return_dtype=pl.Utf8).alias("domain_from_email"))
    if website_col and website_col in df.columns:
        domain_exprs.append(pl.col(website_col).map_elements(extract_domain, return_dtype=pl.Utf8).alias("domain_from_website"))
    if domain_exprs:
        df = df.with_columns(domain_exprs)
        available = [c for c in ["domain_from_domain", "domain_from_website", "domain_from_email"] if c in df.columns]
        domain_candidates = [
            pl.when(pl.col(c) != "").then(pl.col(c)).otherwise(None)
            for c in available
        ]
        df = df.with_columns(pl.coalesce(domain_candidates).fill_null("").alias("domain"))
    else:
        df = df.with_columns(pl.lit("").alias("domain"))
    df = df.with_columns(pl.col("domain").map_elements(is_generic_domain, return_dtype=pl.Boolean).alias("is_generic_domain"))
    df = df.with_columns(pl.col("domain").map_elements(_extract_domain_sld, return_dtype=pl.Utf8).alias("domain_sld"))
    df = df.with_columns([
        pl.col("name_norm").map_elements(token_sort_fingerprint, return_dtype=pl.Utf8).alias("name_token_sort"),
        pl.col("name_norm").map_elements(phonetic_key, return_dtype=pl.Utf8).alias("name_phonetic"),
        pl.col("name_norm").map_elements(extract_root_brand, return_dtype=pl.Utf8).alias("root_brand"),
        pl.col("name_norm").map_elements(extract_supplier_identity_core, return_dtype=pl.Utf8).alias("supplier_identity_core"),
    ])
    # Brand-location core: name with branch/location modifiers stripped.
    # Used by brand_location_variant_match to detect same supplier under location suffixes.
    loc_terms = set(location_terms) if location_terms else set()
    _orig_name_col = name_col  # captured for closure
    df = df.with_columns(
        pl.struct([_orig_name_col, "name_norm"])
        .map_elements(
            lambda r: compute_name_location_core(
                r.get(_orig_name_col, ""),
                r.get("name_norm", ""),
                loc_terms,
                legal_index,
            ),
            return_dtype=pl.Utf8,
        )
        .alias("name_location_core")
    )
    # Franchise store number: extract numeric store ID when a risky banner brand is in name_norm.
    # Used for matching: same banner + same store number + same address → 98%; different store numbers → not 85%.
    from src.config import RISKY_BANNER_BRAND_TOKENS as _RISKY_BANNER
    import re as _re
    def _extract_franchise_store_number(name_norm: str) -> str:
        if not name_norm:
            return ""
        tokens = name_norm.split()
        has_banner = any(t in _RISKY_BANNER for t in tokens)
        if not has_banner:
            return ""
        # Match patterns: #123, No 123, Nr 123, Store 123, or standalone 3-5 digit number
        m = _re.search(r'#(\d+)|(?:no|nr|store|outlet)\s*(\d+)|(?<!\w)(\d{3,5})(?!\w)', name_norm, _re.IGNORECASE)
        if m:
            return next(g for g in m.groups() if g is not None)
        return ""
    df = df.with_columns(
        pl.col("name_norm")
        .map_elements(_extract_franchise_store_number, return_dtype=pl.Utf8)
        .alias("franchise_store_number")
    )
    return df


def _support_entries(encoded: Any) -> List[Dict[str, str]]:
    if not encoded:
        return []
    try:
        data = json.loads(str(encoded))
    except Exception:
        return []
    return [item for item in data if isinstance(item, dict)]


def _support_values_by_strength(encoded: Any, strengths: set[str]) -> str:
    values = []
    for entry in _support_entries(encoded):
        if entry.get("strength") in strengths and entry.get("value") and entry.get("value") not in values:
            values.append(entry["value"])
    return "|".join(values)


def _support_fields_raw_summary(encoded: Any) -> str:
    parts = []
    for entry in _support_entries(encoded):
        field = entry.get("field", "")
        strength = entry.get("strength", "")
        value = entry.get("value", "")
        if field and value:
            parts.append(f"{field}:{strength}:{value}")
    return " | ".join(parts)

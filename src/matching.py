"""Matching engine: evaluate candidate pairs across multiple passes."""
import json
import time
from typing import Dict, Any, Set
from rapidfuzz import fuzz
from src.brand_families import decode_alias_hits
from src.config import (
    BROAD_GLOBAL_SUPPLIER_CORES,
    ClusteringConfig,
    COMMON_FIRST_NAMES,
    GENERIC_ROOT_TOKENS,
    HOSPITALITY_TERMS,
    KNOWN_ADDRESS_FAMILY_BRIDGE_GROUPS,
    KNOWN_DISTINCTIVE_FAMILY_ROOTS,
    KNOWN_FAMILY_TOKEN_GROUPS,
    KNOWN_RELATED_NAME_PAIRS,
    LOCATION_ROOT_TOKENS,
    PERSON_TITLE_TOKENS,
    AMBIGUOUS_REVIEW_CORES,
    PROTECTED_COMPOUND_IDENTITY_PHRASES,
    REGULATORY_REVIEW_TOKENS,
    RISKY_BANNER_BRAND_TOKENS,
    SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS,
    SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS,
    TRUSTED_SUPPLIER_IDENTITY_CORES,
)
from src.matching_types import MatchResult
from src.guardrails import apply_guardrails
from src.preprocessing import extract_supplier_identity_core

# Stopwords and legal stops for acronym generation (not the same as generic root tokens)
_ACRONYM_STOPWORDS: frozenset = frozenset({
    "the", "of", "for", "and", "in", "at", "by", "to", "as", "a", "an", "or",
    "with", "from", "on", "under", "into", "over", "between", "its", "de", "la", "le",
})
_ACRONYM_LEGAL_STOPS: frozenset = frozenset({
    "inc", "ltd", "llc", "corp", "co", "plc", "gmbh", "sa", "ag", "bv", "nv",
    "spa", "sas", "srl", "oy", "ab", "kk", "pte", "pvt", "lp",
})


def _numeric_tokens(text: str) -> Set[str]:
    return {t for t in str(text or "").split() if t.isdigit()}


def _address_number_support(nums_a: Set[str], nums_b: Set[str]) -> bool:
    """Treat 39 vs 39/49 style normalized addresses as number-supported."""
    if not nums_a or not nums_b:
        return False
    if nums_a & nums_b:
        return True
    # Ranges such as 39/49 normalize to tokens 39 and 49. A single-number
    # address should be allowed to match the range endpoint only when the
    # rest of the address/name/location also supports it.
    return False


def calculate_name_similarity(name_a: str, name_b: str) -> float:
    if not name_a or not name_b:
        return 0.0
    name_a = str(name_a).lower().strip()
    name_b = str(name_b).lower().strip()
    token_score = max(fuzz.token_set_ratio(name_a, name_b), fuzz.token_sort_ratio(name_a, name_b)) / 100.0
    # Partial ratio is useful for long legal-name containment, but too aggressive for short names
    # such as Unique A vs Unique B or Robert A vs Robert B.
    if len(name_a) >= 12 and len(name_b) >= 12 and min(len(name_a.split()), len(name_b.split())) >= 2:
        partial = fuzz.partial_ratio(name_a, name_b) / 100.0
        return max(token_score, min(partial, 0.94))
    return token_score


def calculate_address_similarity(addr_a: str, addr_b: str) -> float:
    if not addr_a or not addr_b:
        return 0.0
    if addr_a == addr_b:
        return 1.0
    score = max(fuzz.ratio(addr_a, addr_b), fuzz.token_set_ratio(addr_a, addr_b)) / 100.0
    nums_a = _numeric_tokens(addr_a)
    nums_b = _numeric_tokens(addr_b)
    if nums_a and nums_b and not _address_number_support(nums_a, nums_b):
        # "1 main street" and "2 main street" are not the same property even
        # though token similarity is high. Keep them below address-match gates.
        return min(score, 0.74)
    return score


def _tax_set(row: Dict[str, Any]) -> Set[str]:
    return {t for t in str(row.get("tax_norm", "")).split("|") if t}


def _tax_overlap(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    a, b = _tax_set(row_a), _tax_set(row_b)
    return bool(a and b and (a & b))


def _tax_overlap_values(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Set[str]:
    return _tax_set(row_a) & _tax_set(row_b)

def _tax_loose_set(row: Dict[str, Any]) -> Set[str]:
    return {t for t in str(row.get("tax_loose_norm", "")).split("|") if t}

def _tax_loose_overlap(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    # Loose tax overlap is only a candidate signal. It is never enough by itself.
    a, b = _tax_loose_set(row_a), _tax_loose_set(row_b)
    return bool(a and b and (a & b))


def _address_location_supported(row_a: Dict[str, Any], row_b: Dict[str, Any], addr_sim: float) -> bool:
    """Allow same/similar addresses with street-number range variants.

    Examples: `GLEMSECKSTR 39` vs `Glemseckstrasse 39/49` normalize with one
    shared house number and the same street tokens. City/postal/country then
    make this a safe supporting signal for highly related names.
    """
    addr_a = row_a.get("addr_norm", "") or ""
    addr_b = row_b.get("addr_norm", "") or ""
    if not addr_a or not addr_b:
        return False
    if addr_sim >= 0.88:
        return True
    nums_a = _numeric_tokens(addr_a)
    nums_b = _numeric_tokens(addr_b)
    if not _address_number_support(nums_a, nums_b):
        return False
    words_a = {t for t in addr_a.split() if not t.isdigit() and len(t) >= 4}
    words_b = {t for t in addr_b.split() if not t.isdigit() and len(t) >= 4}
    if not (words_a & words_b):
        return False
    city_a, city_b = row_a.get("city_norm", ""), row_b.get("city_norm", "")
    country_a, country_b = row_a.get("country_norm", ""), row_b.get("country_norm", "")
    postal_a, postal_b = row_a.get("postal_norm", ""), row_b.get("postal_norm", "")
    same_country = bool(country_a and country_b and country_a == country_b)
    same_city = bool(city_a and city_b and fuzz.ratio(city_a, city_b) >= 90)
    same_postal = bool(postal_a and postal_b and postal_a == postal_b)
    return same_country and (same_city or same_postal)


def secondary_name_match(row_a: Dict, row_b: Dict) -> bool:
    vals_a = _all_match_names(row_a)
    vals_b = _all_match_names(row_b)
    for a in vals_a:
        for b in vals_b:
            if not a or not b:
                continue
            if a == b or fuzz.token_set_ratio(a, b) >= 90:
                return True
    return False


def _all_match_names(row: Dict) -> list[str]:
    vals = [row.get("name_norm", "")]
    for i in range(2, 8):
        if row.get(f"name{i}_norm"):
            vals.append(row[f"name{i}_norm"])
    for value in str(row.get("json_secondary_names_norm", "") or "").split("|"):
        if value:
            vals.append(value)
    return [v for v in vals if v]


def _all_match_tokens(row: Dict) -> Set[str]:
    tokens: Set[str] = set()
    for value in _all_match_names(row):
        tokens.update(value.split())
    root = row.get("root_brand", "")
    if root:
        tokens.update(str(root).split())
    return tokens


def _acronym(text: str) -> str:
    if not text:
        return ""
    return "".join(w[0] for w in text.split() if w and len(w) > 2)


def _acronym_bridge(row_a: Dict, row_b: Dict) -> bool:
    names_a = [row_a.get("name_norm", ""), row_a.get("root_brand", "")]
    names_b = [row_b.get("name_norm", ""), row_b.get("root_brand", "")]
    for i in range(2, 8):
        names_a.append(row_a.get(f"name{i}_norm", ""))
        names_b.append(row_b.get(f"name{i}_norm", ""))
    names_a.extend([v for v in str(row_a.get("json_secondary_names_norm", "") or "").split("|") if v])
    names_b.extend([v for v in str(row_b.get("json_secondary_names_norm", "") or "").split("|") if v])
    for a in names_a:
        for b in names_b:
            if not a or not b:
                continue
            if len(a) <= 4 and a == _acronym(b):
                return True
            if len(b) <= 4 and b == _acronym(a):
                return True
            # PPC special observed in real files.
            if {a, b} & {"ppc"} and ("potasse" in a or "produits" in a or "vynova" in a or "potasse" in b or "produits" in b or "vynova" in b):
                return True
    return False


def _generate_acronym(text: str) -> str:
    """Generate acronym from a long name using stopword-aware filtering.

    Takes the first letter of each significant token (not a stopword, not a legal
    suffix, and at least 2 chars). Returns empty string when fewer than 3 significant
    tokens remain (too few to form a reliable acronym).
    """
    if not text:
        return ""
    tokens = str(text).lower().split()
    significant = [
        t for t in tokens
        if t not in _ACRONYM_STOPWORDS
        and t not in _ACRONYM_LEGAL_STOPS
        and len(t) >= 2
    ]
    if len(significant) < 3:
        return ""
    return "".join(t[0] for t in significant)


def _full_name_for_acronym(row: Dict) -> str:
    """Concatenate the first two name fields to reconstruct names split across columns."""
    parts = []
    for name in _all_match_names(row)[:2]:
        if name and name not in parts:
            parts.append(name)
    return " ".join(parts)


def _acronym_bridge_full(row_a: Dict, row_b: Dict) -> bool:
    """Check whether any short alphabetic token from one row is an acronym of the
    other row's full (possibly multi-field) name, using stopword-aware generation.

    This catches cases like NAACP ↔ National Association for the Advancement of
    Colored People and FICCI ↔ Federation of Indian Chambers of Commerce and Industry
    where the name is split across multiple name fields.
    """
    def _acronym_candidate_tokens(row: Dict) -> set:
        toks: set = set()
        for name in _all_match_names(row):
            for tok in name.split():
                if 3 <= len(tok) <= 7 and tok.isalpha():
                    toks.add(tok)
        return toks

    tokens_a = _acronym_candidate_tokens(row_a)
    tokens_b = _acronym_candidate_tokens(row_b)

    # Check each individual name field plus the first-two-fields concatenation
    names_b_to_check = list(_all_match_names(row_b)) + [_full_name_for_acronym(row_b)]
    for long_name in names_b_to_check:
        gen = _generate_acronym(long_name)
        if gen and gen in tokens_a:
            return True

    names_a_to_check = list(_all_match_names(row_a)) + [_full_name_for_acronym(row_a)]
    for long_name in names_a_to_check:
        gen = _generate_acronym(long_name)
        if gen and gen in tokens_b:
            return True

    return False


def _compact_names_match(row_a: Dict, row_b: Dict) -> bool:
    """Return True when any pair of (short) names from the two rows match after
    removing all whitespace — e.g. 'MER JAN' vs 'MERJAN'.

    Only tested for names whose compact form is 4–12 characters to avoid
    accidentally bridging long generic names.
    """
    names_a = _all_match_names(row_a)
    names_b = _all_match_names(row_b)
    for a in names_a:
        ca = a.replace(" ", "")
        if len(ca) < 4 or len(ca) > 20:
            continue
        for b in names_b:
            if a == b:
                continue  # already handled by direct equality
            cb = b.replace(" ", "")
            if ca == cb:
                return True
    return False


def _distinctive_address_tokens(name_a: str, name_b: str) -> set:
    """Shared name tokens that are distinctive (not generic/location/person/hospitality)."""
    NON_BRIDGE = GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES | HOSPITALITY_TERMS
    return {
        t for t in (set(name_a.split()) & set(name_b.split()))
        if len(t) >= 4
        and t not in NON_BRIDGE
        and not any(ch.isdigit() for ch in t)
    }


def _distinctive_tokens(row: Dict) -> Set[str]:
    tokens = set()
    for token in str(row.get("name_norm", "") or "").split():
        if len(token) < 4:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        if token in GENERIC_ROOT_TOKENS or token in LOCATION_ROOT_TOKENS or token in COMMON_FIRST_NAMES:
            continue
        if token in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and token not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS:
            continue
        tokens.add(token)
    root = row.get("root_brand", "")
    if root and root not in GENERIC_ROOT_TOKENS and root not in LOCATION_ROOT_TOKENS and root not in COMMON_FIRST_NAMES:
        for token in str(root).split():
            if len(token) >= 4 and not (token in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and token not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS):
                tokens.add(token)
    return tokens


def _distinctive_token_prefix_related(row_a: Dict, row_b: Dict) -> bool:
    """Return true for distinctive stem relations such as fluoro/fluoropharm.

    This is intentionally gated to non-generic, non-location tokens and is used
    only with address/location support.
    """
    ta = _distinctive_tokens(row_a)
    tb = _distinctive_tokens(row_b)
    for a in ta:
        for b in tb:
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            if len(short) >= 5 and long.startswith(short):
                return True
    return False


def _known_family_bridge(row_a: Dict, row_b: Dict) -> bool:
    ta = _distinctive_tokens(row_a)
    tb = _distinctive_tokens(row_b)
    if not ta or not tb:
        return False
    for group in KNOWN_FAMILY_TOKEN_GROUPS:
        if ta & group and tb & group and not (ta & tb):
            return True
    return False


def _known_related_name_pair(row_a: Dict, row_b: Dict) -> bool:
    names_a = set(_all_match_names(row_a)) | {row_a.get("root_brand", "")}
    names_b = set(_all_match_names(row_b)) | {row_b.get("root_brand", "")}
    for group in KNOWN_RELATED_NAME_PAIRS:
        hits_a = {g for g in group if any(g == n or g in n or n in g for n in names_a if n)}
        hits_b = {g for g in group if any(g == n or g in n or n in g for n in names_b if n)}
        if hits_a and hits_b and hits_a != hits_b:
            return True
    return False


def _known_distinctive_family_root(row_a: Dict, row_b: Dict) -> bool:
    """Explicit family/root bridge, currently used for reviewer-approved WEKA."""
    tokens_a = _all_match_tokens(row_a)
    tokens_b = _all_match_tokens(row_b)
    shared = tokens_a & tokens_b & KNOWN_DISTINCTIVE_FAMILY_ROOTS
    if not shared:
        return False
    country_a = row_a.get("country_norm", "")
    country_b = row_b.get("country_norm", "")
    same_country = bool(country_a and country_b and country_a == country_b)
    secondary_support = any(t in str(row_a.get("json_secondary_names_norm", "") or "") for t in shared) or any(
        t in str(row_b.get("json_secondary_names_norm", "") or "") for t in shared
    )
    return bool(same_country or secondary_support)


def _known_address_family_bridge(row_a: Dict, row_b: Dict) -> bool:
    """Controlled same-address family bridge, e.g. TURNUS <-> WEKA MEDIA."""
    tokens_a = _all_match_tokens(row_a)
    tokens_b = _all_match_tokens(row_b)
    for group in KNOWN_ADDRESS_FAMILY_BRIDGE_GROUPS:
        hits_a = tokens_a & group
        hits_b = tokens_b & group
        if hits_a and hits_b and hits_a != hits_b:
            return True
    return False


def _known_brand_family_alias_match(row_a: Dict, row_b: Dict) -> Dict[str, Any]:
    ids_a = {x for x in str(row_a.get("known_brand_family_ids", "") or "").split("|") if x}
    ids_b = {x for x in str(row_b.get("known_brand_family_ids", "") or "").split("|") if x}
    overlap = ids_a & ids_b
    if not overlap:
        return {"matched": False}

    safe_a = {x for x in str(row_a.get("known_brand_family_safe_ids", "") or "").split("|") if x}
    safe_b = {x for x in str(row_b.get("known_brand_family_safe_ids", "") or "").split("|") if x}
    risky_a = {x for x in str(row_a.get("known_brand_family_risky_ids", "") or "").split("|") if x}
    risky_b = {x for x in str(row_b.get("known_brand_family_risky_ids", "") or "").split("|") if x}
    safe_overlap = overlap & safe_a & safe_b
    risky_overlap = overlap - safe_overlap
    alias_hits_a = decode_alias_hits(row_a.get("known_brand_alias_hits", ""))
    alias_hits_b = decode_alias_hits(row_b.get("known_brand_alias_hits", ""))
    aliases_a = {(h.family_id, h.alias, h.category) for h in alias_hits_a if h.family_id in overlap}
    aliases_b = {(h.family_id, h.alias, h.category) for h in alias_hits_b if h.family_id in overlap}
    exact_alias_overlap = sorted({a[1] for a in aliases_a if a in aliases_b})
    return {
        "matched": True,
        "overlap": sorted(overlap),
        "safe_overlap": sorted(safe_overlap),
        "risky_overlap": sorted(risky_overlap),
        "has_risky": bool(risky_overlap or (overlap & risky_a) or (overlap & risky_b)),
        "exact_alias_overlap": exact_alias_overlap,
        "aliases_a": sorted({f"{h.family_id}:{h.category}:{h.alias}" for h in alias_hits_a if h.family_id in overlap})[:10],
        "aliases_b": sorted({f"{h.family_id}:{h.category}:{h.alias}" for h in alias_hits_b if h.family_id in overlap})[:10],
    }


def _support_entries(row: Dict[str, Any]) -> list[Dict[str, str]]:
    try:
        data = json.loads(str(row.get("support_fields_json", "") or "[]"))
    except Exception:
        return []
    return [item for item in data if isinstance(item, dict)]


def _support_field_overlap(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Dict[str, Any]:
    entries_a = _support_entries(row_a)
    entries_b = _support_entries(row_b)
    if not entries_a or not entries_b:
        return {"matched": False}
    by_value_a = {}
    by_value_b = {}
    for entry in entries_a:
        value = entry.get("value")
        if value:
            by_value_a.setdefault(value, []).append(entry)
    for entry in entries_b:
        value = entry.get("value")
        if value:
            by_value_b.setdefault(value, []).append(entry)
    values = sorted(set(by_value_a) & set(by_value_b))
    if not values:
        return {"matched": False}

    strength_rank = {
        "same_entity_id": 5,
        "same_entity_name": 4,
        "domain": 3,
        "family_or_parent": 2,
        "review_only": 1,
    }
    candidates = []
    for value in values:
        entries = by_value_a[value] + by_value_b[value]
        strengths = {entry.get("strength", "review_only") for entry in entries}
        fields = sorted({entry.get("field", "") for entry in entries if entry.get("field")})
        best_strength = max(strengths, key=lambda s: strength_rank.get(s, 0))
        candidates.append((strength_rank.get(best_strength, 0), value, best_strength, fields, sorted(strengths)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _rank, value, strength, fields, strengths = candidates[0]
    return {
        "matched": True,
        "value": value,
        "strength": strength,
        "fields": fields,
        "strengths": strengths,
        "all_values": values[:20],
    }


def _supplier_identity_core(row: Dict[str, Any]) -> str:
    return str(row.get("supplier_identity_core") or extract_supplier_identity_core(row.get("name_norm", "")) or "")


def _core_tokens(core: str) -> list[str]:
    return [t for t in str(core or "").split() if t]


def _protected_identity_phrase(core: str) -> str:
    text = str(core or "")
    for phrase in PROTECTED_COMPOUND_IDENTITY_PHRASES:
        if phrase and phrase in text:
            return phrase
    return ""


def _core_is_safe_for_identity(core: str, row: Dict[str, Any], *, allow_short_exact_phrase: bool = False) -> bool:
    if row.get("is_likely_individual") or row.get("is_hospitality"):
        return False
    tokens = _core_tokens(core)
    if not tokens:
        return False
    if all(t in GENERIC_ROOT_TOKENS or t in LOCATION_ROOT_TOKENS or t in COMMON_FIRST_NAMES for t in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if token in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS:
            return True
        if token in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and token not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS:
            return False
        if token.isdigit():
            return False
        if any(ch.isdigit() for ch in token):
            # Alphanumeric supplier names are common and can be distinctive
            # when they are the normalized name core: 3B, 3BL, 4titude,
            # 4flow, 7layers, etc. They still pass through the same generic,
            # person, hospitality, and operational guardrails as alphabetic
            # cores, and pure numeric/vendor-reference values remain blocked.
            return any(ch.isalpha() for ch in token) and len(token) >= 2
        if len(token) >= 5:
            return True
        if allow_short_exact_phrase and len(token) >= 3 and len(str(row.get("name_norm", "") or "").split()) >= 2:
            return True
        # 4-char tokens that are the leading brand root of a multi-word name are safe.
        # Example: "bell" in "bell canada" or "bell aliant" is a brand identity, not noise.
        if len(token) >= 4 and len(str(row.get("name_norm", "") or "").split()) >= 2:
            return True
        return False
    return True


def _core_identity_relation(row_a: Dict[str, Any], row_b: Dict[str, Any], name_sim: float) -> Dict[str, Any]:
    core_a = _supplier_identity_core(row_a)
    core_b = _supplier_identity_core(row_b)
    if not core_a or not core_b:
        return {"matched": False}

    name_a = str(row_a.get("name_norm", "") or "")
    name_b = str(row_b.get("name_norm", "") or "")
    exact_full_name = bool(name_a and name_b and name_a == name_b)
    allow_short = exact_full_name
    if not _core_is_safe_for_identity(core_a, row_a, allow_short_exact_phrase=allow_short):
        return {"matched": False}
    if not _core_is_safe_for_identity(core_b, row_b, allow_short_exact_phrase=allow_short):
        return {"matched": False}

    tokens_a = _core_tokens(core_a)
    tokens_b = _core_tokens(core_b)
    protected_a = _protected_identity_phrase(core_a)
    protected_b = _protected_identity_phrase(core_b)
    if protected_a or protected_b:
        # Protected compounds such as Eastman Kodak, Air Liquide, Air Products,
        # and Springer Nature are their own identities. They must not be
        # reduced to one shared parent token and bridged to another family.
        if protected_a != protected_b:
            return {"matched": False}
    exact_core = core_a == core_b
    core_sim = calculate_name_similarity(core_a, core_b)
    shared_core_tokens = sorted(set(tokens_a) & set(tokens_b))
    is_banner_only_core = bool(shared_core_tokens) and all(t in RISKY_BANNER_BRAND_TOKENS for t in shared_core_tokens)
    shared_trusted_core_tokens = sorted(set(tokens_a) & set(tokens_b) & TRUSTED_SUPPLIER_IDENTITY_CORES)
    shared_long_safe = [
        t for t in shared_core_tokens
        if len(t) >= 5
        and t not in GENERIC_ROOT_TOKENS
        and t not in LOCATION_ROOT_TOKENS
        and t not in COMMON_FIRST_NAMES
        and not (t in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and t not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS)
    ]
    prefix_related = False
    for a in tokens_a:
        for b in tokens_b:
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            if len(short) >= 5 and long.startswith(short):
                prefix_related = True

    # This pass is for clear supplier brand/group identity, not loose family
    # or shared-token discovery. A single shared token such as "schweiz",
    # "process", or a person first name must never become a main-cluster edge.
    # Partial/one-token overlap can still be surfaced by review/support passes.
    matched = bool(exact_core or shared_trusted_core_tokens or (core_sim >= 0.92 and name_sim >= 0.88))
    if not matched:
        return {"matched": False}
    bare_core_row = bool(name_a == core_a or name_b == core_b)
    # A standalone single-token entity (e.g. "TELUS") connecting to a differently-named
    # peer ("TELUS Health") via bare-core identity is ambiguous without domain/address/tax
    # support — treat as LLM-review (70%), not MANUAL_REVIEW (85%).
    # Only applies when the two name_norms differ: if they are identical (same normalized
    # name, different addresses) the exact-name signal is strong enough to remain at 85%.
    _single_nontrusted_bare = (
        bare_core_row
        and exact_core
        and name_a != name_b
        and len(tokens_a) == 1
        and tokens_a[0] not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS
    )
    return {
        "matched": True,
        "core_a": core_a,
        "core_b": core_b,
        "exact_core": exact_core,
        "core_sim": core_sim,
        "shared_core_tokens": shared_core_tokens,
        "shared_trusted_core_tokens": shared_trusted_core_tokens,
        "is_trusted_supplier_core": bool(shared_trusted_core_tokens or (exact_core and set(tokens_a) & TRUSTED_SUPPLIER_IDENTITY_CORES)),
        "is_broad_global_supplier_core": bool((set(tokens_a) & set(tokens_b) & BROAD_GLOBAL_SUPPLIER_CORES) or (exact_core and set(tokens_a) & BROAD_GLOBAL_SUPPLIER_CORES)),
        "is_ambiguous_review_core": bool(
            (set(tokens_a) & set(tokens_b) & AMBIGUOUS_REVIEW_CORES)
            or (exact_core and set(tokens_a) & AMBIGUOUS_REVIEW_CORES)
            or _single_nontrusted_bare
        ),
        "protected_identity_phrase": protected_a if protected_a == protected_b else "",
        "shared_long_safe": shared_long_safe,
        "prefix_related": prefix_related,
        "exact_full_name": exact_full_name,
        "bare_core_row": bare_core_row,
        "is_banner_only_core": is_banner_only_core,
    }


def _regulatory_task_force_relation(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Dict[str, Any]:
    name_a = str(row_a.get("name_norm", "") or "")
    name_b = str(row_b.get("name_norm", "") or "")
    if not name_a or not name_b or name_a == name_b:
        return {"matched": False}
    tokens_a = set(name_a.split())
    tokens_b = set(name_b.split())
    regulatory_overlap = sorted((tokens_a & tokens_b) & REGULATORY_REVIEW_TOKENS)
    if len(regulatory_overlap) < 2:
        return {"matched": False}
    distinctive_overlap = sorted(
        t for t in (tokens_a & tokens_b)
        if t not in REGULATORY_REVIEW_TOKENS
        and t not in GENERIC_ROOT_TOKENS
        and t not in LOCATION_ROOT_TOKENS
        and t not in COMMON_FIRST_NAMES
    )
    if distinctive_overlap:
        return {"matched": False}
    return {"matched": True, "regulatory_overlap": regulatory_overlap}


def _institutional_ecosystem_relation(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Dict[str, Any]:
    """Detect university/institution ecosystem pairs that need LLM review.

    These can share VAT/domain/admin infrastructure but still represent a
    subsidiary, tech-transfer office, innovation arm, clinic, or foundation
    rather than an obvious same supplier identity for procurement cleanup.
    """
    tokens_a = set(str(row_a.get("name_norm", "") or "").split())
    tokens_b = set(str(row_b.get("name_norm", "") or "").split())
    if not tokens_a or not tokens_b:
        return {"matched": False}
    institution = {"university", "universitaet", "universitat", "universität", "hospital", "clinic", "klinikum", "institute", "institut"}
    ecosystem = {"innovation", "innovations", "ventures", "enterprise", "enterprises", "technology", "transfer", "foundation", "research"}
    shared_location = sorted((tokens_a & tokens_b) & LOCATION_ROOT_TOKENS)
    has_institution = bool((tokens_a | tokens_b) & institution)
    has_ecosystem = bool((tokens_a | tokens_b) & ecosystem)
    if has_institution and has_ecosystem and shared_location:
        return {"matched": True, "shared_location_tokens": shared_location}
    return {"matched": False}


def _professional_person_identity_address(row_a: Dict, row_b: Dict, address_supported: bool) -> bool:
    """Same cleaned professional/person name at the same property.

    This covers rows such as "Wilhelm Schmidt", "Dipl.-Ing. Wilhelm Schmidt",
    and "Dipl.-Ing. Wilhelm Schmidt GmbH" without relaxing the default rule for
    different individuals at the same address.
    """
    if not address_supported:
        return False
    person_a = row_a.get("person_name_norm", "") or ""
    person_b = row_b.get("person_name_norm", "") or ""
    if not person_a or person_a != person_b or len(person_a.split()) < 2:
        return False
    tokens_a = set(str(row_a.get("name_norm", "") or "").split())
    tokens_b = set(str(row_b.get("name_norm", "") or "").split())
    has_prof_title = bool((tokens_a | tokens_b) & PERSON_TITLE_TOKENS)
    has_person_or_company_variant = bool(
        row_a.get("is_likely_individual")
        or row_b.get("is_likely_individual")
        or row_a.get("has_legal_suffix")
        or row_b.get("has_legal_suffix")
    )
    return bool(has_prof_title and has_person_or_company_variant)


def _related_root(row_a: Dict, row_b: Dict) -> bool:
    ra = row_a.get("root_brand", "")
    rb = row_b.get("root_brand", "")
    if (
        ra and rb and ra == rb and len(ra) >= 4
        and ra not in GENERIC_ROOT_TOKENS
        and ra not in LOCATION_ROOT_TOKENS
        and ra not in COMMON_FIRST_NAMES
    ):
        return True
    return bool(_distinctive_tokens(row_a) & _distinctive_tokens(row_b))


def _root_is_risky_single(row_a: Dict, row_b: Dict) -> bool:
    """True when the shared root brand token is in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS."""
    ra = row_a.get("root_brand", "")
    rb = row_b.get("root_brand", "")
    return bool(
        ra and rb and ra == rb
        and ra in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS
        and ra not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS
    )


def _names_are_article_reordered(name_a: str, name_b: str) -> bool:
    """Return True when two names have identical tokens but different order,
    and at least one token is a common article (the, a, an, la, le, les, el)."""
    if not name_a or not name_b or name_a == name_b:
        return False
    tokens_a = frozenset(name_a.split())
    tokens_b = frozenset(name_b.split())
    if tokens_a != tokens_b:
        return False
    articles = frozenset({"the", "a", "an", "la", "le", "les", "el", "los", "las"})
    return bool(tokens_a & articles)


def evaluate_pair(row_a: Dict[str, Any], row_b: Dict[str, Any], address_counts: Dict[str, int], config: ClusteringConfig = None) -> MatchResult:
    if config is None:
        config = ClusteringConfig()

    def R(result: MatchResult) -> MatchResult:
        t0 = time.perf_counter()
        guarded = apply_guardrails(row_a, row_b, result)
        timing = getattr(config, "_runtime_timing", None)
        if timing is not None:
            timing["guardrails_seconds"] = timing.get("guardrails_seconds", 0.0) + (time.perf_counter() - t0)
        return guarded

    # PASS 0: exact duplicate — same normalized name AND same normalized address.
    # This is the highest-confidence signal and must dominate all other passes.
    _name_a0 = str(row_a.get("name_norm", "") or "")
    _addr_a0 = str(row_a.get("addr_norm", "") or "")
    _name_b0 = str(row_b.get("name_norm", "") or "")
    _addr_b0 = str(row_b.get("addr_norm", "") or "")
    if (_name_a0 and _name_b0 and _name_a0 == _name_b0
            and _addr_a0 and _addr_b0 and _addr_a0 == _addr_b0):
        return R(MatchResult(
            True, 98.0, "exact_duplicate",
            {"name": _name_a0, "addr": _addr_a0},
            needs_review=False, review_reason="",
        ))

    name_a, name_b = row_a.get("name_norm", ""), row_b.get("name_norm", "")
    addr_a, addr_b = row_a.get("addr_norm", ""), row_b.get("addr_norm", "")
    city_a, city_b = row_a.get("city_norm", ""), row_b.get("city_norm", "")
    country_a, country_b = row_a.get("country_norm", ""), row_b.get("country_norm", "")
    domain_a, domain_b = row_a.get("domain", ""), row_b.get("domain", "")
    name_sim = calculate_name_similarity(name_a, name_b)
    addr_sim = calculate_address_similarity(addr_a, addr_b)
    same_domain = bool(domain_a and domain_b and domain_a == domain_b and not row_a.get("is_generic_domain", False) and not row_b.get("is_generic_domain", False))
    same_city_country = bool(city_a and city_b and country_a and country_b and country_a == country_b and fuzz.ratio(city_a, city_b) >= 85)
    tax_match = _tax_overlap(row_a, row_b)
    tax_loose_match = _tax_loose_overlap(row_a, row_b)
    related_root = _related_root(row_a, row_b)
    known_family = _known_family_bridge(row_a, row_b)
    known_related_pair = _known_related_name_pair(row_a, row_b)
    known_distinctive_family_root = _known_distinctive_family_root(row_a, row_b)
    known_address_family_bridge = _known_address_family_bridge(row_a, row_b)
    known_brand_family_alias = _known_brand_family_alias_match(row_a, row_b)
    support_field_overlap = _support_field_overlap(row_a, row_b)
    address_supported = _address_location_supported(row_a, row_b, addr_sim)
    distinctive_prefix_related = _distinctive_token_prefix_related(row_a, row_b)
    professional_person_address = _professional_person_identity_address(row_a, row_b, address_supported)
    supplier_identity = _core_identity_relation(row_a, row_b, name_sim)
    regulatory_relation = _regulatory_task_force_relation(row_a, row_b)
    institutional_ecosystem = _institutional_ecosystem_relation(row_a, row_b)
    root_is_risky = _root_is_risky_single(row_a, row_b)

    # PASS 1: tax exact overlap, including multi-tax/JSON-derived IDs.
    if tax_match:
        overlap = _tax_overlap_values(row_a, row_b)
        tax_stats = getattr(config, "_tax_block_stats", {}) or {}
        broad_tax = False
        for tax in overlap:
            stat = tax_stats.get(tax, {})
            if (
                (
                    int(stat.get("row_count", 0)) > int(getattr(config, "max_exact_tax_only_block_size", 10))
                    or (
                        int(stat.get("row_count", 0)) >= 4
                        and int(stat.get("distinct_roots", 0)) >= 2
                        and int(stat.get("distinct_names", 0)) >= 2
                    )
                )
                and (
                    int(stat.get("distinct_roots", 0)) > int(getattr(config, "max_exact_tax_distinct_roots", 3))
                    or int(stat.get("distinct_names", 0)) > int(getattr(config, "max_exact_tax_distinct_roots", 3)) + 1
                    or int(stat.get("row_count", 0)) >= 4
                )
            ):
                broad_tax = True
                break
        if broad_tax and not (name_sim >= 0.72 or address_supported or same_domain):
            return R(MatchResult(False, 0.0, "tax_exact_broad_rejected", {"tax_overlap": list(overlap), "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}, needs_review=True, review_reason="Exact tax value appears across many distinct roots and needs name/address/domain support"))
        if institutional_ecosystem.get("matched") and name_sim < 0.80 and not (address_supported or same_domain):
            return R(MatchResult(
                True,
                70.0,
                "tax_exact_institutional_ecosystem_review",
                {
                    "tax_overlap": list(overlap),
                    "name_sim": round(name_sim, 3),
                    "addr_sim": round(addr_sim, 3),
                    "shared_location_tokens": institutional_ecosystem.get("shared_location_tokens", []),
                    "route": "LLM_REVIEW",
                    "score_reason": "Shared tax in university/institution ecosystem; LLM review required before supplier identity clustering",
                },
                needs_review=True,
                review_reason="Institutional ecosystem relation with shared tax requires LLM review",
            ))
        support_fields = [str(field).lower() for field in support_field_overlap.get("fields", [])]
        support_is_tax_only = bool(support_fields) and all("tax" in field or "vat" in field for field in support_fields)
        trusted_support = (
            support_field_overlap.get("matched")
            and support_field_overlap.get("strength") in {"same_entity_id", "same_entity_name"}
            and not support_is_tax_only
        )
        if name_sim < 0.50 and not same_domain and not trusted_support:
            # Before routing to review, check secondary name overlap, acronym
            # equivalence (e.g. NAACP / FICCI), and compact name variants
            # (e.g. MERJAN / MER JAN). Any of these are strong enough to
            # proceed to the standard 98% tax_exact path.
            alias_supported = (
                secondary_name_match(row_a, row_b)
                or _acronym_bridge_full(row_a, row_b)
                or _compact_names_match(row_a, row_b)
            )
            if not alias_supported:
                return R(MatchResult(
                    True,
                    90.0,
                    "tax_exact_low_similarity_review",
                    {"tax_overlap": list(overlap), "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)},
                    needs_review=True,
                    review_reason="Exact tax match but names are unrelated/low similarity; review before supplier identity clustering",
                ))
        score = 98.0
        if name_sim >= 0.75:
            score = 99.0
        elif addr_sim >= 0.80:
            score = 98.0
        return R(MatchResult(True, score, "tax_exact", {"tax_overlap": list(overlap), "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}))

    # PASS 1B: loose tax overlap, e.g. DE233002380 vs 233002380.
    # This requires name/address/domain support because numeric parts can collide across countries.
    if tax_loose_match and (name_sim >= 0.78 or address_supported or same_domain):
        score = 92.0
        if name_sim >= 0.85 and addr_sim >= 0.80:
            score = 95.0
        return R(MatchResult(True, score, "tax_loose_supported", {"tax_loose_overlap": list(_tax_loose_set(row_a) & _tax_loose_set(row_b)), "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}))

    # PASS 1C: explicitly configured trusted support fields. These are not
    # enabled for family/canonical text by default; configuration decides which
    # support fields are allowed to act like same-entity evidence.
    if support_field_overlap.get("matched") and support_field_overlap.get("strength") in {"same_entity_id", "same_entity_name"}:
        strength = support_field_overlap.get("strength")
        score = 98.0 if strength == "same_entity_id" else 93.0
        return R(MatchResult(True, score, f"support_{strength}", {
            "shared_support_field": support_field_overlap.get("fields", []),
            "support_field_value": support_field_overlap.get("value", ""),
            "support_field_strength": strength,
            "support_strengths": support_field_overlap.get("strengths", []),
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
        }))

    # PASS 1.5: compact/joined-name variant with address or domain support.
    # Handles "3 CARP" vs "3CARP", "T.B.D. Pizza" vs "TBD Pizza" (after norm),
    # "MER JAN" vs "MERJAN", "Rose Design" vs "Rosedesign".
    if _compact_names_match(row_a, row_b):
        if same_domain or address_supported:
            return R(MatchResult(True, 98.0, "compact_name_match_supported",
                {"name_a": name_a, "name_b": name_b, "same_domain": same_domain, "address_supported": address_supported},
                needs_review=False, review_reason=""))
        if same_city_country:
            return R(MatchResult(True, 85.0, "compact_name_match_city",
                {"name_a": name_a, "name_b": name_b},
                needs_review=True, review_reason="Compact/joined name variant in same city; review recommended"))

    # PASS 1.6: article-reordered name variant at same address.
    # Handles "PIE THE" vs "THE PIE", "AT LIMITS" vs "AT THE LIMITS" (same address).
    if _names_are_article_reordered(name_a, name_b) and (address_supported or same_domain):
        return R(MatchResult(True, 98.0, "article_reorder_match",
            {"name_a": name_a, "name_b": name_b},
            needs_review=False, review_reason=""))

    # PASS 2: exact normalized name and same/similar address, including
    # street-number ranges such as 39 vs 39/49 when city/postal supports it.
    if name_a and name_a == name_b and addr_a and addr_b and address_supported:
        return R(MatchResult(True, 98.0, "name_address_exact", {"name": name_a, "address_a": addr_a, "address_b": addr_b, "addr_sim": round(addr_sim, 3)}))

    # PASS 2B: professional title / person-company variants at the same address.
    if professional_person_address:
        return R(MatchResult(True, 86.0, "professional_name_address", {"person_name": row_a.get("person_name_norm", ""), "addr_sim": round(addr_sim, 3)}, needs_review=True, review_reason="Professional/person/company name variant at same address; manual review"))

    # PASS 2C: regulatory/task-force relations are review-only unless the full
    # name is identical. Shared phrases such as "REACH Task Force" are not
    # enough for high-confidence supplier identity.
    if regulatory_relation.get("matched"):
        return R(MatchResult(
            True,
            74.0,
            "regulatory_or_task_force_related",
            {
                "regulatory_overlap": regulatory_relation.get("regulatory_overlap", []),
                "name_sim": round(name_sim, 3),
                "supplier_identity_core_a": _supplier_identity_core(row_a),
                "supplier_identity_core_b": _supplier_identity_core(row_b),
                "score_reason": "Shared regulatory/task-force phrase only; manual review",
            },
            needs_review=True,
            review_reason="Related regulatory/task-force phrase; not clear same supplier identity",
        ))

    # PASS 2D-pre: brand-location variant check. When both rows share the same
    # name_location_core (location/branch modifier already stripped) but differ
    # in name_norm, resolve here at 85 so PASS 2D cannot downgrade to 70 review.
    # Duplicate of PASS 3B logic but placed before supplier-identity scoring.
    _nlc_a_early = str(row_a.get("name_location_core") or "")
    _nlc_b_early = str(row_b.get("name_location_core") or "")
    if (_nlc_a_early and _nlc_b_early
            and _nlc_a_early == _nlc_b_early
            and (_nlc_a_early != name_a or _nlc_b_early != name_b)):
        _loc_generic2 = GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES
        _nlc_dist_early = [t for t in _nlc_a_early.split() if len(t) >= 3 and t not in _loc_generic2]
        if _nlc_dist_early:
            _banner_only_early = all(t in RISKY_BANNER_BRAND_TOKENS for t in _nlc_dist_early)
            if _banner_only_early and not (same_domain or address_supported):
                return R(MatchResult(
                    True, 70.0,
                    "brand_location_variant_match",
                    {
                        "brand_location_core": _nlc_a_early,
                        "banner_brand_only": True,
                        "banner_tokens": _nlc_dist_early,
                        "location_modifier_a": " ".join(t for t in name_a.split() if t not in set(_nlc_a_early.split())),
                        "location_modifier_b": " ".join(t for t in name_b.split() if t not in set(_nlc_b_early.split())),
                        "name_sim": round(name_sim, 3),
                        "same_domain": same_domain,
                        "address_supported": address_supported,
                        "score_reason": "Banner/franchise/dealer brand only — no operator/owner/address support",
                    },
                    needs_review=True,
                    review_reason="Banner/franchise/dealer brand only — LLM review required",
                ))
            _score_early = 98.0 if (same_domain or address_supported) else 85.0
            return R(MatchResult(
                True, _score_early,
                "brand_location_variant_match",
                {
                    "brand_location_core": _nlc_a_early,
                    "location_modifier_a": " ".join(t for t in name_a.split() if t not in set(_nlc_a_early.split())),
                    "location_modifier_b": " ".join(t for t in name_b.split() if t not in set(_nlc_b_early.split())),
                    "name_sim": round(name_sim, 3),
                    "same_domain": same_domain,
                    "address_supported": address_supported,
                },
                needs_review=False,
                review_reason="",
            ))

    # PASS 2D: same clear supplier brand/group identity across addresses,
    # legal forms, branches, or countries. This is distinct from weak
    # family/parent rollup and requires a safe distinctive core.
    if supplier_identity.get("matched"):
        cross_country = bool(country_a and country_b and country_a != country_b)
        cross_address = bool(addr_a and addr_b and addr_sim < 0.88)
        exact_core = bool(supplier_identity.get("exact_core"))
        trusted_core = bool(supplier_identity.get("is_trusted_supplier_core"))
        broad_global_core = bool(supplier_identity.get("is_broad_global_supplier_core"))
        ambiguous_review_core = bool(supplier_identity.get("is_ambiguous_review_core"))
        is_banner_only_core = bool(supplier_identity.get("is_banner_only_core"))
        if is_banner_only_core and not (same_domain or address_supported or tax_match):
            ambiguous_review_core = True
        score = 86.0
        needs_review = False
        reason = "Distinctive supplier brand/group identity"
        if ambiguous_review_core and not (same_domain or address_supported or tax_match):
            score = 70.0
            needs_review = True
            reason = "Ambiguous supplier core requires LLM review before final clustering"
        elif same_domain and name_sim >= 0.80:
            score = 93.0
            reason = "Same business domain and distinctive supplier identity"
        elif supplier_identity.get("exact_full_name") and address_supported:
            score = 96.0
            reason = "Exact supplier identity name with same/similar address"
        elif supplier_identity.get("exact_full_name"):
            score = 88.0
            needs_review = True
            reason = "Exact supplier identity name across address/legal-form variation; review advised"
        elif exact_core and supplier_identity.get("bare_core_row"):
            score = 86.0
            needs_review = True
            reason = "Bare supplier core matched fuller legal entity; review advised"
        elif exact_core:
            score = 86.0 if (cross_country or cross_address) else 88.0
            needs_review = True
            reason = "Clear broad supplier brand/group identity; review advised"
        elif trusted_core:
            score = 84.0
            needs_review = True
            reason = "Trusted supplier core overlap with division/location terms; review advised"
        else:
            score = 70.0
            needs_review = True
            reason = "Fuzzy/near-equivalent supplier identity core requires LLM review"
        if broad_global_core and (cross_country or cross_address) and not (same_domain or address_supported or tax_match):
            score = min(score, 86.0)
            needs_review = True
            reason = "Broad global supplier group without tax/domain/address support; review advised"
        if (cross_country or cross_address) and score >= 90.0 and not (same_domain or address_supported or tax_match):
            score = 88.0
            needs_review = True
            reason = "Cross-country/address supplier group without strong support; review advised"
        evidence = {
            "supplier_identity_core_a": supplier_identity.get("core_a", ""),
            "supplier_identity_core_b": supplier_identity.get("core_b", ""),
            "shared_supplier_identity_core": supplier_identity.get("core_a", "") if exact_core else "",
            "shared_core_tokens": supplier_identity.get("shared_core_tokens", []),
            "shared_trusted_core_tokens": supplier_identity.get("shared_trusted_core_tokens", []),
            "core_sim": round(float(supplier_identity.get("core_sim", 0.0) or 0.0), 3),
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
            "same_domain": same_domain,
            "address_supported": address_supported,
            "cross_country": cross_country,
            "cross_address": cross_address,
            "distinctive_supplier_identity": True,
            "trusted_supplier_core": trusted_core,
            "broad_global_supplier_core": broad_global_core,
            "ambiguous_review_core": ambiguous_review_core,
            "protected_identity_phrase": supplier_identity.get("protected_identity_phrase", ""),
            "main_cluster_allowed": bool((exact_core or trusted_core) and not (ambiguous_review_core and not (same_domain or address_supported or tax_match))),
            "route": "LLM_REVIEW" if score <= 70.0 or (ambiguous_review_core and not (same_domain or address_supported or tax_match)) else ("AUTO_CONFIDENT" if score >= 93 else "MANUAL_REVIEW"),
            "score_reason": reason,
        }
        return R(MatchResult(True, score, "distinctive_supplier_identity", evidence, needs_review=needs_review, review_reason=reason if needs_review else ""))

    # PASS 2E: shared mapped support/canonical/family values. Default
    # family/parent/canonical text is review-only, and must be evaluated before
    # fuzzy-name/address-only no-match exits so it reaches the Review output.
    if support_field_overlap.get("matched"):
        strength = support_field_overlap.get("strength", "review_only")
        evidence = {
            "shared_support_field": support_field_overlap.get("fields", []),
            "support_field_value": support_field_overlap.get("value", ""),
            "support_field_strength": strength,
            "support_strengths": support_field_overlap.get("strengths", []),
            "support_field_values": support_field_overlap.get("all_values", []),
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
            "same_domain": same_domain,
            "address_supported": address_supported,
        }
        score = 82.0 if strength == "family_or_parent" else 74.0
        if address_supported or same_domain or name_sim >= 0.85:
            score += 8.0
        return R(MatchResult(True, min(score, 90.0), "support_field_review", evidence, needs_review=True, review_reason="Shared support/canonical/family field; review-only unless configured as trusted same-entity evidence"))

    # PASS 3: exact normalized name without address/tax/domain support.
    # For non-generic distinctive names this is same-legal-owner evidence at
    # 85% and goes directly into union-find. Generic or location-only names
    # still go to review only.
    if name_a and name_a == name_b:
        _weak = GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | HOSPITALITY_TERMS
        name_tokens = set(name_a.split())
        name_is_generic = bool(name_tokens) and all(t in _weak or len(t) <= 2 for t in name_tokens)
        if name_is_generic:
            score = 86.0 if (country_a == country_b or not country_a or not country_b) else 75.0
            return R(MatchResult(
                True,
                score,
                "name_exact_review",
                {"name": name_a, "country_a": country_a, "country_b": country_b},
                needs_review=True,
                review_reason="Same normalized name without address, tax, or domain support",
            ))
        score = 85.0
        return R(MatchResult(
            True,
            score,
            "same_legal_owner_confirmed",
            {"name": name_a, "country_a": country_a, "country_b": country_b, "name_sim": 1.0},
            needs_review=False,
            review_reason="",
        ))

    # PASS 3B: brand-location variant match.
    # Fires when names differ only by a trailing location/branch modifier (city, province,
    # country, or parenthesized address) but share the same brand-location core.
    # Example: "BFI CANADA-CALGARY" vs "BFI CANADA-TORONTO" → core "bfi canada".
    # Example: "Bell Canada (5115 Creekbank Rd)" vs "Bell Canada" → core "bell canada".
    nlc_a = str(row_a.get("name_location_core") or "")
    nlc_b = str(row_b.get("name_location_core") or "")
    if nlc_a and nlc_b and nlc_a == nlc_b and (nlc_a != name_a or nlc_b != name_b):
        # At least one name had its location modifier stripped.
        # Check that the shared core has a distinctive (non-generic) token.
        _loc_generic = GENERIC_ROOT_TOKENS | LOCATION_ROOT_TOKENS | COMMON_FIRST_NAMES
        _nlc_distinctive = [t for t in nlc_a.split() if len(t) >= 3 and t not in _loc_generic]
        if _nlc_distinctive:
            _banner_only_3b = all(t in RISKY_BANNER_BRAND_TOKENS for t in _nlc_distinctive)
            if _banner_only_3b and not (same_domain or address_supported):
                return R(MatchResult(
                    True, 70.0,
                    "brand_location_variant_match",
                    {
                        "brand_location_core": nlc_a,
                        "banner_brand_only": True,
                        "banner_tokens": _nlc_distinctive,
                        "location_modifier_a": " ".join(t for t in name_a.split() if t not in set(nlc_a.split())),
                        "location_modifier_b": " ".join(t for t in name_b.split() if t not in set(nlc_b.split())),
                        "name_sim": round(name_sim, 3),
                        "same_domain": same_domain,
                        "address_supported": address_supported,
                        "score_reason": "Banner/franchise/dealer brand only — no operator/owner/address support",
                    },
                    needs_review=True,
                    review_reason="Banner/franchise/dealer brand only — LLM review required",
                ))
            score = 85.0
            needs_review = False
            if same_domain or address_supported:
                score = 98.0
            return R(MatchResult(
                True, score,
                "brand_location_variant_match",
                {
                    "brand_location_core": nlc_a,
                    "location_modifier_a": " ".join(t for t in name_a.split() if t not in set(nlc_a.split())),
                    "location_modifier_b": " ".join(t for t in name_b.split() if t not in set(nlc_b.split())),
                    "name_sim": round(name_sim, 3),
                    "same_domain": same_domain,
                    "address_supported": address_supported,
                },
                needs_review=needs_review,
                review_reason="",
            ))

    # PASS 4: strong fuzzy name match.
    if name_sim >= config.fuzzy_name_threshold_strong:
        if address_supported or addr_sim >= 0.80 or same_city_country or same_domain:
            # Promote to domain_address_confirmed if both domain and address are present
            if same_domain and address_supported and name_sim >= 0.80:
                return R(MatchResult(True, 98.0, "domain_address_confirmed",
                    {"domain": domain_a, "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)},
                    needs_review=False, review_reason=""))
            # If one name is a single risky token (bare hub) and the only support is city/country,
            # downgrade to 70 — no address/domain/tax means ambiguous sub-brand or division.
            _name_a_tokens = str(name_a or "").split()
            _name_b_tokens = str(name_b or "").split()
            _bare_risky = (
                (len(_name_a_tokens) == 1 and _name_a_tokens[0] in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and _name_a_tokens[0] not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS)
                or (len(_name_b_tokens) == 1 and _name_b_tokens[0] in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS and _name_b_tokens[0] not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS)
            )
            if _bare_risky and not address_supported and not same_domain and not tax_match and not (addr_sim >= 0.80):
                return R(MatchResult(True, 70.0, "name_fuzzy_review_candidate",
                    {"name_sim": round(name_sim, 3), "candidate_type": "bare_risky_hub"},
                    needs_review=True,
                    review_reason="Bare single-token risky hub with city-only support; LLM review required"))
            return R(MatchResult(True, 91.0, "name_fuzzy_supported", {"name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}))
        # High name similarity with no location/domain/address support → LLM review candidate.
        # Previously blank; now routes to 70 so plausible name variants reach the review queue.
        return R(MatchResult(True, 70.0, "name_fuzzy_review_candidate",
                             {"name_sim": round(name_sim, 3), "candidate_type": "fuzzy_brand_core"},
                             needs_review=True,
                             review_reason="High name similarity but no address/domain/city support; LLM review required"))

    # PASS 5: domain plus related name/secondary/acronym evidence.
    if same_domain:
        # NEW: same domain + near name + address support → promoted to 98/93
        if address_supported and name_sim >= 0.80:
            return R(MatchResult(True, 98.0, "domain_address_confirmed",
                {"domain": domain_a, "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)},
                needs_review=False, review_reason=""))
        if address_supported and name_sim >= 0.55:
            return R(MatchResult(True, 93.0, "domain_address_supported",
                {"domain": domain_a, "name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)},
                needs_review=True, review_reason="Same domain and address with near name; review recommended"))
        if name_sim >= 0.70 or secondary_name_match(row_a, row_b) or related_root or known_family or _acronym_bridge(row_a, row_b):
            return R(MatchResult(True, 86.0 if name_sim >= 0.80 else 78.0, "domain_name_related", {"domain": domain_a, "name_sim": round(name_sim, 3)}, needs_review=name_sim < 0.80, review_reason="Same domain with lower name similarity"))
        return R(MatchResult(True, 72.0, "domain_review_candidate", {"domain": domain_a, "name_sim": round(name_sim, 3)}, needs_review=True, review_reason="Same business domain with unrelated or weakly related names; manual review"))

    # PASS 6: exact/similar address plus supporting evidence.
    if addr_a and addr_b and (addr_a == addr_b or addr_sim >= 0.88 or address_supported):
        addr_key = addr_a if addr_a else addr_b
        addr_count = max(address_counts.get(addr_a, 0), address_counts.get(addr_b, 0))
        if addr_count > config.max_companies_per_address and name_sim < 0.80 and not same_domain and not secondary_name_match(row_a, row_b):
            return R(MatchResult(False, 0.0, "address_risk", {"company_count": addr_count}, True, f"{addr_count} names at same/shared address"))
        evidence = {"addr_sim": round(addr_sim, 3), "company_count": addr_count}
        if known_address_family_bridge:
            return R(MatchResult(True, 80.0, "known_family_bridge", evidence | {"known_address_family_bridge": True}, needs_review=True, review_reason="Known distinctive family/address bridge; manual review"))
        if name_sim >= 0.70 or distinctive_prefix_related:
            dtokens = _distinctive_address_tokens(name_a or "", name_b or "")
            if dtokens and name_sim >= 0.65:
                return R(MatchResult(
                    True, 85.0, "address_distinctive_shared",
                    evidence | {"name_sim": round(name_sim, 3), "distinctive_shared_tokens": sorted(dtokens)[:5], "distinctive_prefix_related": distinctive_prefix_related},
                    needs_review=True, review_reason="Same/similar address with distinctive shared name tokens; review for different tax IDs",
                ))
            return R(MatchResult(True, 78.0, "address_name_related", evidence | {"name_sim": round(name_sim, 3), "distinctive_prefix_related": distinctive_prefix_related}, needs_review=True, review_reason="Same/similar address + related names; review required"))
        if secondary_name_match(row_a, row_b) or _acronym_bridge(row_a, row_b) or known_related_pair:
            return R(MatchResult(True, 80.0, "address_secondary_or_acronym", evidence | {"known_related_name_pair": known_related_pair}, needs_review=True, review_reason="Same/similar address + secondary/acronym/known relationship evidence"))
        if same_domain:
            return R(MatchResult(True, 84.0, "address_domain", evidence | {"domain": domain_a}, needs_review=True, review_reason="Same/similar address + domain"))
        if config.enable_family_bridge and known_brand_family_alias.get("matched"):
            risky = bool(known_brand_family_alias.get("has_risky"))
            alias_score = 90.0 if (known_brand_family_alias.get("exact_alias_overlap") or name_sim >= 0.65) else 88.0
            alias_ev = evidence | {
                "known_brand_family_overlap": known_brand_family_alias.get("overlap", []),
                "safe_overlap": known_brand_family_alias.get("safe_overlap", []),
                "risky_overlap": known_brand_family_alias.get("risky_overlap", []),
                "name_sim": round(name_sim, 3),
                "address_supported": True,
            }
            return R(MatchResult(True, alias_score, "known_brand_family_alias", alias_ev, needs_review=True, review_reason="Known brand/family alias with address support; manual review"))
        return R(MatchResult(False, 0.0, "address_only", evidence, True, "Address-only match"))

    # PASS 7: secondary/acronym bridge with support. Acronym/full-name is too
    # weak on city/country alone; it needs address/domain/secondary evidence or
    # an optional AI review candidate.
    secondary_bridge = secondary_name_match(row_a, row_b)
    acronym_bridge = _acronym_bridge(row_a, row_b)
    if secondary_bridge and (same_city_country or same_domain or addr_sim >= 0.70):
        return R(MatchResult(True, 84.0, "secondary_or_acronym_bridge", {"name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}, needs_review=True, review_reason="Secondary-name/acronym bridge"))
    if acronym_bridge and (same_domain or addr_sim >= 0.70):
        return R(MatchResult(True, 84.0, "secondary_or_acronym_bridge", {"name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}, needs_review=True, review_reason="Acronym/full-name bridge with address/domain support"))
    if acronym_bridge and same_city_country and config.ai_review_enabled:
        return R(MatchResult(True, 65.0, "acronym_review_candidate", {"name_sim": round(name_sim, 3), "addr_sim": round(addr_sim, 3)}, needs_review=True, review_reason="Acronym/full-name relation requires AI review or stronger evidence"))

    # PASS 8: parent/family root bridge.
    if config.enable_family_bridge and known_brand_family_alias.get("matched"):
        risky = bool(known_brand_family_alias.get("has_risky"))
        secondary_bridge = secondary_name_match(row_a, row_b)
        supported = bool(tax_match or same_domain or address_supported or secondary_bridge)
        evidence = {
            "known_brand_family_overlap": known_brand_family_alias.get("overlap", []),
            "safe_overlap": known_brand_family_alias.get("safe_overlap", []),
            "risky_overlap": known_brand_family_alias.get("risky_overlap", []),
            "exact_alias_overlap": known_brand_family_alias.get("exact_alias_overlap", []),
            "aliases_a": known_brand_family_alias.get("aliases_a", []),
            "aliases_b": known_brand_family_alias.get("aliases_b", []),
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
            "address_supported": address_supported,
            "same_domain": same_domain,
            "secondary_bridge": secondary_bridge,
        }
        if risky and not supported:
            # Risky alias without any corroborating evidence: keep blank.
            # Prevents "metro logistics" DE vs "metro bank" GB from clustering via
            # a single risky generic word. Same-brand families with additional
            # evidence (address, domain, country) are still promoted above.
            return R(MatchResult(False, 0.0, "known_brand_family_risky_needs_support", evidence, needs_review=True, review_reason="Risky/ambiguous alias requires tax, domain, address, secondary/family, known config, or LLM support"))
        score = float(getattr(config, "known_brand_family_default_confidence", 76.0))
        if address_supported and (known_brand_family_alias.get("exact_alias_overlap") or name_sim >= 0.65):
            score = 90.0
        elif same_domain or secondary_bridge:
            score = 84.0
        elif same_city_country:
            score = 80.0
        elif risky:
            score = 70.0
        return R(MatchResult(True, score, "known_brand_family_alias", evidence, needs_review=True, review_reason="Known brand/family alias bridge; manual or LLM review"))

    if config.enable_family_bridge and known_distinctive_family_root:
        evidence = {
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
            "known_distinctive_family_root": True,
            "tokens_a": sorted(_all_match_tokens(row_a) & KNOWN_DISTINCTIVE_FAMILY_ROOTS),
            "tokens_b": sorted(_all_match_tokens(row_b) & KNOWN_DISTINCTIVE_FAMILY_ROOTS),
        }
        return R(MatchResult(True, 76.0, "known_family_bridge", evidence, needs_review=True, review_reason="Known distinctive family/root bridge; manual or LLM review"))

    if config.enable_family_bridge and known_family:
        evidence = {
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
            "family_tokens_a": sorted(_distinctive_tokens(row_a))[:10],
            "family_tokens_b": sorted(_distinctive_tokens(row_b))[:10],
        }
        return R(MatchResult(True, 68.0, "known_family_bridge", evidence, needs_review=True, review_reason="Known family/brand bridge; manual or LLM review"))

    if config.enable_family_bridge and related_root:
        _root_evidence = {
            "root_a": row_a.get("root_brand"),
            "root_b": row_b.get("root_brand"),
            "name_sim": round(name_sim, 3),
            "addr_sim": round(addr_sim, 3),
        }
        if same_city_country or same_domain or addr_sim >= 0.75:
            return R(MatchResult(True, 76.0, "family_bridge_supported", _root_evidence, needs_review=True, review_reason="Parent/family bridge"))
        if country_a and country_b and country_a != country_b:
            # Cross-country: was blank, now 70 LLM candidate so cross-border brands are reviewed.
            return R(MatchResult(True, 70.0, "family_cross_country",
                                 {**_root_evidence, "candidate_type": "possible_parent_family"},
                                 needs_review=True,
                                 review_reason="Cross-country root-brand relation; requires LLM review before clustering"))
        # Same-country or unknown-country with shared brand root but no city/domain/address support.
        # Previously fell through to no_match; now 70 so same-brand sub-brands reach review queue.
        # Examples: Bell Canada vs Bell Aliant, Rogers Communications vs Rogers Cable.
        return R(MatchResult(True, 70.0, "weak_brand_root_candidate",
                             {**_root_evidence, "candidate_type": "possible_sub_brand"},
                             needs_review=True,
                             review_reason="Shared brand root without address/domain/city support; LLM review required"))

    # PASS 9: sub-brand prefix safety net.
    # One name is a distinctive prefix of the other (≥5 chars) but shared root ≥4 chars
    # didn't fire above (e.g. different root brands but one contains the other as a brand stem).
    if not row_a.get("is_likely_individual") and not row_b.get("is_likely_individual"):
        _n_a, _n_b = str(row_a.get("name_norm", "") or ""), str(row_b.get("name_norm", "") or "")
        if _n_a and _n_b and _n_a != _n_b:
            _short_n, _long_n = (_n_a, _n_b) if len(_n_a) <= len(_n_b) else (_n_b, _n_a)
            if (
                len(_short_n) >= 5
                and _long_n.startswith(_short_n)
                and not all(t in GENERIC_ROOT_TOKENS or t in LOCATION_ROOT_TOKENS
                            for t in _short_n.split() if t)
            ):
                return R(MatchResult(True, 70.0, "possible_sub_brand_candidate",
                                     {
                                         "brand_prefix": _short_n,
                                         "name_sim": round(name_sim, 3),
                                         "candidate_type": "supplier_name_abbreviated",
                                     },
                                     needs_review=True,
                                     review_reason="One supplier name is a brand-prefix of the other; LLM review required"))

    return R(MatchResult(False, 0.0, "no_match", {}))

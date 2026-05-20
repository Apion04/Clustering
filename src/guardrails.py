"""Precision guardrails to prevent over-clustering.

These rules run after a candidate pair receives a match score but before Union-Find
merging. They are intentionally conservative for people, hotels/restaurants,
franchises, generic root tokens, and weak family bridges.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Dict, Any, Set
from rapidfuzz import fuzz

from src.config import (
    COMMON_FIRST_NAMES,
    GENERIC_ROOT_TOKENS,
    HOSPITALITY_TERMS,
    LOCATION_ROOT_TOKENS,
    PROTECTED_COMPOUND_IDENTITY_PHRASES,
    SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS,
    SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS,
)
from src.matching_types import MatchResult
from src.person import (
    build_person_profile,
    is_likely_individual_row,
    names_share_only_first_name,
    person_company_name_evidence,
    person_identity_strength,
)


def _tokens(text: str) -> list[str]:
    return [t for t in str(text or "").split() if t]


NON_BRIDGE_TOKENS = (
    GENERIC_ROOT_TOKENS
    | LOCATION_ROOT_TOKENS
    | HOSPITALITY_TERMS
    | COMMON_FIRST_NAMES
)


def _shared_name_tokens(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Set[str]:
    return set(_tokens(row_a.get("name_norm", ""))) & set(_tokens(row_b.get("name_norm", "")))


def _distinctive_shared_tokens(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Set[str]:
    distinctive = set()
    for token in _shared_name_tokens(row_a, row_b):
        if token in NON_BRIDGE_TOKENS:
            continue
        if len(token) < 3:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        distinctive.add(token)
    return distinctive


def _shared_short_code_tokens(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Set[str]:
    return {
        token for token in _shared_name_tokens(row_a, row_b)
        if len(token) <= 6 and any(ch.isdigit() for ch in token) and re.fullmatch(r"[a-z0-9]+", token)
    }


def _only_shared_tokens_are_non_bridge(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    """True when the name overlap is only generic/location/industry words.

    Alphanumeric short codes such as 3B or 3WAY are intentionally not treated as
    distinctive bridge tokens. They can still pass through exact-name, tax,
    domain, or strong address gates below.
    """
    shared = _shared_name_tokens(row_a, row_b)
    if not shared:
        return False
    return not _distinctive_shared_tokens(row_a, row_b)


def _has_numeric_or_short_code_prefix(row: Dict[str, Any]) -> bool:
    tokens = _tokens(row.get("name_norm", ""))
    if not tokens:
        return False
    first = tokens[0]
    if first.isdigit():
        return True
    if first[0].isdigit():
        return True
    # Short alphanumeric code prefixes are common in vendor data and should not
    # bridge to a generic trailing industry word by themselves.
    return bool(len(first) <= 6 and any(ch.isdigit() for ch in first) and re.fullmatch(r"[a-z0-9]+", first))


def _full_name_highly_similar(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    name_a = row_a.get("name_norm", "")
    name_b = row_b.get("name_norm", "")
    if not name_a or not name_b:
        return False
    if name_a == name_b:
        return True
    return max(fuzz.ratio(name_a, name_b), fuzz.token_sort_ratio(name_a, name_b)) >= 94


def _tax_set(row: Dict[str, Any]) -> Set[str]:
    return {t for t in str(row.get("tax_norm", "")).split("|") if t}


def _has_tax(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    return bool(_tax_set(row_a) & _tax_set(row_b))


def _same_business_domain(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    da = row_a.get("domain", "")
    db = row_b.get("domain", "")
    return bool(da and db and da == db and not row_a.get("is_generic_domain") and not row_b.get("is_generic_domain"))


def _same_sld_domain_family(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    """Return True when both rows share a distinctive cross-TLD SLD (domain_sld field)."""
    from src.config import _GENERIC_SLD_GUARD
    sld_a = str(row_a.get("domain_sld", "") or "")
    sld_b = str(row_b.get("domain_sld", "") or "")
    if not sld_a or not sld_b or sld_a != sld_b:
        return False
    if sld_a in _GENERIC_SLD_GUARD or len(sld_a) < 3:
        return False
    da = row_a.get("domain", "")
    db = row_b.get("domain", "")
    # Only fire as cross-TLD evidence when domains differ (same_domain covers identical domains).
    return da != db


def _addr_sim(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> float:
    aa = row_a.get("addr_norm", "")
    ab = row_b.get("addr_norm", "")
    if not aa or not ab:
        return 0.0
    if aa == ab:
        return 1.0
    score = max(fuzz.ratio(aa, ab), fuzz.token_set_ratio(aa, ab)) / 100.0
    nums_a = {t for t in str(aa).split() if t.isdigit()}
    nums_b = {t for t in str(ab).split() if t.isdigit()}
    if nums_a and nums_b and not (nums_a & nums_b):
        return min(score, 0.74)
    return score


def _address_location_supported(row_a: Dict[str, Any], row_b: Dict[str, Any], addr_sim: float) -> bool:
    if addr_sim >= 0.88:
        return True
    aa = row_a.get("addr_norm", "")
    ab = row_b.get("addr_norm", "")
    nums_a = {t for t in str(aa).split() if t.isdigit()}
    nums_b = {t for t in str(ab).split() if t.isdigit()}
    if not (nums_a and nums_b and nums_a & nums_b):
        return False
    words_a = {t for t in str(aa).split() if not t.isdigit() and len(t) >= 4}
    words_b = {t for t in str(ab).split() if not t.isdigit() and len(t) >= 4}
    if not (words_a & words_b):
        return False
    ca = row_a.get("city_norm", "")
    cb = row_b.get("city_norm", "")
    cta = row_a.get("country_norm", "")
    ctb = row_b.get("country_norm", "")
    pa = row_a.get("postal_norm", "")
    pb = row_b.get("postal_norm", "")
    return bool(cta and ctb and cta == ctb and ((ca and cb and fuzz.ratio(ca, cb) >= 90) or (pa and pb and pa == pb)))


def _city_country_support(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    ca = row_a.get("city_norm", "")
    cb = row_b.get("city_norm", "")
    cta = row_a.get("country_norm", "")
    ctb = row_b.get("country_norm", "")
    return bool(cta and ctb and cta == ctb and ca and cb and fuzz.ratio(ca, cb) >= 85)


def _same_full_address_location(row_a: Dict[str, Any], row_b: Dict[str, Any], addr_sim: float) -> bool:
    """Strict address/location gate for individual records."""
    if addr_sim < 0.95:
        return False
    ca = row_a.get("city_norm", "")
    cb = row_b.get("city_norm", "")
    cta = row_a.get("country_norm", "")
    ctb = row_b.get("country_norm", "")
    pa = row_a.get("postal_norm", "")
    pb = row_b.get("postal_norm", "")
    if cta and ctb and cta != ctb:
        return False
    if ca and cb and fuzz.ratio(ca, cb) < 90:
        return False
    # Postal codes in client files can be wrong even when street + house number
    # and city/country are exact. Do not let a postal mismatch split a strict
    # same-person/same-property match when city is already strong.
    if pa and pb and pa != pb and not (ca and cb and fuzz.ratio(ca, cb) >= 90):
        return False
    return True


def _same_person_location_review_support(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    # If both rows carry full addresses and they disagree, preserve the strict
    # individual rule: same person name in the same city is still not enough.
    if row_a.get("addr_norm") and row_b.get("addr_norm"):
        return False
    country_a = row_a.get("country_norm", "")
    country_b = row_b.get("country_norm", "")
    if country_a and country_b and country_a != country_b:
        return False
    city_a = row_a.get("city_norm", "")
    city_b = row_b.get("city_norm", "")
    postal_a = row_a.get("postal_norm", "")
    postal_b = row_b.get("postal_norm", "")
    same_city = bool(city_a and city_b and fuzz.ratio(city_a, city_b) >= 90)
    same_postal = bool(postal_a and postal_b and postal_a == postal_b)
    return bool((country_a or country_b) and (same_city or same_postal))


def looks_like_person(row: Dict[str, Any]) -> bool:
    """Heuristic for individual/person vendor records.

    We intentionally avoid treating names with legal suffixes or company/legal hints as people.
    This protects company names such as "Robert Bosch GmbH".
    """
    return is_likely_individual_row(row)


def has_common_first_name_only(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    return names_share_only_first_name(row_a, row_b)


def is_hospitality_or_franchise(row: Dict[str, Any]) -> bool:
    if row.get("is_hospitality"):
        return True
    name = row.get("name_norm", "")
    toks = set(_tokens(name))
    return bool(toks & HOSPITALITY_TERMS)


def shared_root_is_generic(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    ra = row_a.get("root_brand", "")
    rb = row_b.get("root_brand", "")
    return bool(ra and rb and ra == rb and (ra in GENERIC_ROOT_TOKENS or ra in LOCATION_ROOT_TOKENS))


def _name_is_generic_only(row: Dict[str, Any]) -> bool:
    toks = _tokens(row.get("name_norm", ""))
    return bool(toks) and all(t in GENERIC_ROOT_TOKENS or t in LOCATION_ROOT_TOKENS or t in HOSPITALITY_TERMS for t in toks)


def _protected_identity_phrase(row: Dict[str, Any]) -> str:
    text = f"{row.get('supplier_identity_core', '')} {row.get('name_norm', '')}"
    for phrase in PROTECTED_COMPOUND_IDENTITY_PHRASES:
        if phrase and phrase in text:
            return phrase
    return ""


def _reject(result: MatchResult, reason: str) -> MatchResult:
    return replace(result, is_match=False, match_pct=0.0, needs_review=True, review_reason=reason, evidence={**result.evidence, "guardrail_reject": reason})


def _review_candidate(result: MatchResult, pass_type: str, score: float, reason: str) -> MatchResult:
    return replace(
        result,
        is_match=True,
        pass_type=pass_type,
        match_pct=min(result.match_pct, score),
        needs_review=True,
        review_reason=reason,
        evidence={**result.evidence, "review_only_guardrail": reason},
    )


def apply_guardrails(row_a: Dict[str, Any], row_b: Dict[str, Any], result: MatchResult) -> MatchResult:
    """Return a potentially rejected or adjusted MatchResult."""
    if not result.is_match:
        return result

    pass_type = result.pass_type
    addr_sim = _addr_sim(row_a, row_b)
    tax_match = _has_tax(row_a, row_b)
    same_domain = _same_business_domain(row_a, row_b)
    same_sld = _same_sld_domain_family(row_a, row_b)
    city_country = _city_country_support(row_a, row_b)
    profile_a = build_person_profile(row_a)
    profile_b = build_person_profile(row_b)
    both_people = profile_a.is_likely and profile_b.is_likely
    one_person = profile_a.is_likely != profile_b.is_likely
    same_full_address = _same_full_address_location(row_a, row_b, addr_sim)
    address_supported = _address_location_supported(row_a, row_b, addr_sim)
    shared_identity_core = str(result.evidence.get("shared_supplier_identity_core") or "")
    exact_alphanumeric_identity_core = bool(
        pass_type == "distinctive_supplier_identity"
        and result.evidence.get("main_cluster_allowed")
        and shared_identity_core
        and any(ch.isalpha() for ch in shared_identity_core)
        and any(ch.isdigit() for ch in shared_identity_core)
    )

    protected_a = _protected_identity_phrase(row_a)
    protected_b = _protected_identity_phrase(row_b)
    if (
        (protected_a or protected_b)
        and protected_a != protected_b
        and pass_type in {"name_fuzzy_supported", "address_name_related", "distinctive_supplier_identity", "known_brand_family_alias"}
        and not tax_match
        and not same_domain
    ):
        return _reject(result, "Protected compound supplier identity requires explicit tax/domain support before merging with parent or sibling core")

    # Exact duplicate (same normalized name + same address) is the highest-confidence
    # signal and bypasses most guardrails, but not operational status.
    if pass_type == "exact_duplicate":
        if row_a.get("has_operational_status_hint") or row_b.get("has_operational_status_hint"):
            return _review_candidate(
                result,
                "operational_status_review",
                86.0,
                "Operational blocked/inactive/use-instruction text present; manual review before clustering",
            )
        return result

    # Tax exact is the strongest signal and must pass even for individuals/hotels.
    # This check MUST come before the operational_status check so that a valid
    # same-tax match is never demoted by an instructional name fragment.
    if pass_type == "tax_exact":
        return result

    if row_a.get("has_operational_status_hint") or row_b.get("has_operational_status_hint"):
        return _review_candidate(
            result,
            "operational_status_review",
            86.0,
            "Operational blocked/inactive/use-instruction text present; manual review before clustering",
        )

    # Loose tax (country prefix stripped) is only allowed with name/address/domain support.
    if pass_type == "tax_loose_supported":
        return result

    # Pure generic names such as "Services LLC", "Trading Company", or
    # "Restaurant Inc" must not cluster by name alone.
    if (
        pass_type in {"name_exact", "name_fuzzy_supported", "family_bridge_supported", "family_cross_country", "known_family_bridge", "known_brand_family_alias"}
        and (_name_is_generic_only(row_a) or _name_is_generic_only(row_b))
        and not tax_match
        and not same_domain
        and addr_sim < 0.88
    ):
        return _reject(result, "Generic business/root words require address, tax, or business-domain support")

    # If the only overlap is generic industry/legal/location language, do not
    # let that edge create or bridge a cluster. Examples: Production, Point,
    # Scientific, Pharm, Laboratories, Materials, Sales.
    generic_only_supported_by_strong_address_name = (
        addr_sim >= 0.95
        and float(result.evidence.get("name_sim", 0.0) or 0.0) >= 0.90
    )
    generic_only_supported_by_short_code_address = (
        addr_sim >= 0.95
        and bool(_shared_short_code_tokens(row_a, row_b))
    )
    if (
        pass_type not in {"tax_exact", "tax_loose_supported", "tax_exact_institutional_ecosystem_review", "name_exact", "name_exact_review", "name_address_exact", "support_field_review", "domain_review_candidate", "domain_sld_family", "domain_sld_review_candidate", "domain_sld_address_confirmed", "regulatory_or_task_force_related", "brand_location_variant_match", "brand_prefix_descriptor_match", "compact_name_match_supported", "compact_name_match_city", "article_reorder_match"}
        and _only_shared_tokens_are_non_bridge(row_a, row_b)
        and not same_domain
        and not same_sld
        and not (both_people and same_full_address and bool(person_identity_strength(row_a, row_b)))
        and not exact_alphanumeric_identity_core
        and not generic_only_supported_by_strong_address_name
        and not generic_only_supported_by_short_code_address
    ):
        return _reject(result, "Only shared name tokens are generic/non-bridge industry or location words")

    # Numeric/code-prefixed suppliers need stronger evidence than a shared
    # generic tail token. This protects cases such as 202 Production vs FIT
    # Production and 3B Scientific vs unrelated Scientific Center records.
    if _has_numeric_or_short_code_prefix(row_a) or _has_numeric_or_short_code_prefix(row_b):
        allowed_numeric_code_match = (
            pass_type in {"tax_exact", "tax_loose_supported", "name_exact", "name_address_exact", "support_field_review", "domain_review_candidate", "brand_location_variant_match", "compact_name_match_supported", "compact_name_match_city", "article_reorder_match"}
            or same_domain
            or _full_name_highly_similar(row_a, row_b)
            or exact_alphanumeric_identity_core
            or (
                address_supported
                and result.match_pct >= 80.0
                and bool(_distinctive_shared_tokens(row_a, row_b) or _shared_short_code_tokens(row_a, row_b))
            )
        )
        if not allowed_numeric_code_match:
            return _reject(result, "Numeric/short-code prefix names require exact/high-similarity name, tax, domain, or strong address evidence")

    # People/individuals are stricter than companies. Same household/address,
    # city, postal code, first name, or surname is not enough.
    if both_people:
        identity_strength = person_identity_strength(row_a, row_b)
        if not identity_strength:
            return _reject(result, "Different individual/person identities must not cluster by first name, surname, city, postal code, or shared address")
        if same_full_address:
            if identity_strength == "exact":
                return replace(
                    result,
                    match_pct=min(result.match_pct, 90.0),
                    needs_review=result.needs_review or addr_sim < 0.99,
                    review_reason=result.review_reason or "Same individual full name with same/highly similar full address",
                )
            return replace(
                result,
                match_pct=min(result.match_pct, 78.0),
                needs_review=True,
                review_reason="Initial/person-name variation with same/highly similar full address; manual review",
            )
        same_street_country_review = (
            identity_strength == "exact"
            and addr_sim >= 0.95
            and row_a.get("country_norm")
            and row_a.get("country_norm") == row_b.get("country_norm")
        )
        if identity_strength == "exact" and _same_person_location_review_support(row_a, row_b):
            return _review_candidate(
                result,
                "person_same_name_location_review",
                86.0,
                "Same individual full name/suffix variant with matching city/country/postal; manual review",
            )
        if same_street_country_review:
            return _review_candidate(
                result,
                "person_address_city_mismatch_review",
                76.0,
                "Same/similar individual full name and street/country, but city/postal differ; manual review",
            )
        return _reject(result, "Same individual/person name requires same/highly similar full address by default")

    if one_person:
        # Compact/joined name matches are direct name equivalences (e.g. "MER JAN" == "MERJAN")
        # and should not be blocked by an accidental person-name detection.
        if pass_type in {"compact_name_match_supported", "compact_name_match_city", "article_reorder_match"}:
            return replace(result, needs_review=True, review_reason="Compact/joined name variant; verify individual vs company")
        person_row = row_a if profile_a.is_likely else row_b
        company_row = row_b if profile_a.is_likely else row_a
        if tax_match or same_domain:
            return replace(
                result,
                match_pct=min(result.match_pct, 86.0),
                needs_review=True,
                review_reason="Individual-to-company match allowed by tax/domain support; verify manually",
            )
        if same_full_address and person_company_name_evidence(person_row, company_row):
            return replace(
                result,
                match_pct=min(result.match_pct, 80.0),
                needs_review=True,
                review_reason="Individual-to-company address match with explicit name evidence; manual review",
            )
        return _reject(result, "Individual-to-company matches require same tax, same business domain, or same full address plus explicit name evidence")

    if has_common_first_name_only(row_a, row_b) and pass_type in {"name_fuzzy_strong", "family_bridge_supported", "family_cross_country", "name_exact", "distinctive_supplier_identity"}:
        return _reject(result, "Only shared meaningful token is a common first name")

    # Hotels/restaurants/franchises: do not parent/brand-cluster by name alone.
    hospitality_case = is_hospitality_or_franchise(row_a) or is_hospitality_or_franchise(row_b)
    if hospitality_case:
        # Same property/location only: exact/similar address or strong city/country + very high full-name similarity.
        name_sim = max(
            fuzz.token_set_ratio(row_a.get("name_norm", ""), row_b.get("name_norm", "")),
            fuzz.token_sort_ratio(row_a.get("name_norm", ""), row_b.get("name_norm", "")),
        ) / 100.0
        # Same legal owner: when the match was explicitly confirmed as a non-generic
        # same-legal-name match, the franchise/brand-only block does not apply.
        # Confirmed via same_legal_owner_confirmed pass (which already excludes names
        # that are entirely generic/hospitality tokens such as "subway" or "hotel").
        if pass_type == "same_legal_owner_confirmed":
            pass  # fall through — same legal owner, not brand-only clustering
        elif addr_sim >= 0.88 and name_sim >= 0.75:
            return replace(result, match_pct=min(result.match_pct, 90.0), needs_review=result.needs_review or addr_sim < 0.95)
        elif same_domain and city_country and name_sim >= 0.80:
            return replace(result, match_pct=min(result.match_pct, 86.0), needs_review=True, review_reason="Hospitality/franchise match needs property/location verification")
        else:
            return _reject(result, "Hotel/restaurant/franchise names need same property/address/tax/domain evidence; brand/root alone is not enough")

    # Generic root words cannot drive family/parent bridges.
    if pass_type in {"family_bridge_supported", "family_cross_country"} and shared_root_is_generic(row_a, row_b):
        return _reject(result, "Generic root token cannot drive parent/family clustering")

    if pass_type in {"family_bridge_supported", "family_cross_country"}:
        shared_tokens = set(_tokens(row_a.get("name_norm", ""))) & set(_tokens(row_b.get("name_norm", "")))
        weak_shared = {t for t in shared_tokens if t in NON_BRIDGE_TOKENS}
        distinctive_shared = _distinctive_shared_tokens(row_a, row_b)
        if weak_shared and not distinctive_shared and not tax_match and not same_domain and addr_sim < 0.95:
            return _reject(result, "City/location or generic industry tokens cannot drive family clustering")

    # A risky single-token root (ambiguous brand hub such as "telus", "shell",
    # "metro") with only city/country support must be downgraded to 70 so it
    # enters the LLM review queue rather than auto-clustering at 76.
    if pass_type == "family_bridge_supported" and not tax_match and not same_domain and addr_sim < 0.88:
        ra = row_a.get("root_brand", "")
        rb = row_b.get("root_brand", "")
        if (ra and rb and ra == rb
                and ra in SUPPLIER_IDENTITY_RISKY_SINGLE_TOKENS
                and ra not in SUPPLIER_IDENTITY_TRUSTED_SINGLE_TOKENS):
            return replace(result, match_pct=min(result.match_pct, 70.0),
                           needs_review=True,
                           review_reason="Risky single-token root with only city/country support; LLM review required")

    # Address-only is already non-match in matcher, but keep this as safety.
    if pass_type == "address_only":
        return _reject(result, "Address-only match is not enough")

    return result

"""Person-name helpers for strict individual supplier guardrails."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Tuple

from src.config import (
    COMMON_FIRST_NAMES,
    COMPANY_HINT_WORDS,
    GENERIC_ROOT_TOKENS,
    HOSPITALITY_TERMS,
    PERSON_TITLE_TOKENS,
)


PERSON_COMPANY_MARKERS = (
    COMPANY_HINT_WORDS
    | GENERIC_ROOT_TOKENS
    | HOSPITALITY_TERMS
    | {
        "bank",
        "clinic",
        "college",
        "foundation",
        "hospital",
        "institute",
        "laboratory",
        "labs",
        "school",
        "universitaet",
        "universitat",
        "university",
    }
)

PERSON_SUFFIX_TOKENS = {
    "jr", "jun", "junior", "sr", "sen", "senior",
}


@dataclass(frozen=True)
class PersonProfile:
    is_likely: bool
    tokens: Tuple[str, ...]
    first: str = ""
    last: str = ""
    first_initial: str = ""
    last_initial: str = ""
    has_initial: bool = False
    had_title: bool = False

    @property
    def full_name(self) -> str:
        return " ".join(self.tokens)


def clean_person_name(name: Any) -> str:
    """Return a title/punctuation-cleaned name string for person detection.

    Input is usually already `name_norm`, but this function is defensive so it
    also handles leading quotes/dots and title tokens such as `Dr.` or `Dipl.Kfm.`.
    """
    text = str(name or "").lower().strip()
    if not text:
        return ""
    text = re.sub(r"^[\s\.'\"`‘’“”]+", "", text)
    text = re.sub(r"[\.\-]", " ", text)
    text = re.sub(r"[\"`‘’“”]", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    parts = [p.strip("'") for p in text.split() if p.strip("'")]

    title_compounds = {"dipl ing", "dipl kfm", "diplkfm"}
    changed = True
    while changed and parts:
        changed = False
        if parts and parts[0] in PERSON_TITLE_TOKENS:
            parts = parts[1:]
            changed = True
            continue
        if len(parts) >= 2 and (" ".join(parts[:2]) in title_compounds or "".join(parts[:2]) in PERSON_TITLE_TOKENS):
            parts = parts[2:]
            changed = True

    # Professional titles can appear after the person name in supplier files,
    # e.g. "Wilhelm Schmidt Dipl.-Ing.". Remove title tokens wherever they occur
    # for person identity comparison only; company/root matching still uses the
    # normalized supplier name.
    parts = [p for p in parts if not (p in PERSON_TITLE_TOKENS and len(p) > 1)]
    while parts and parts[-1].rstrip(".") in PERSON_SUFFIX_TOKENS:
        parts = parts[:-1]
    return " ".join(parts)


def _tokens(name: Any) -> Tuple[str, ...]:
    cleaned = clean_person_name(name)
    if not cleaned:
        return ()
    return tuple(t for t in cleaned.split() if t)


def _is_initial(token: str) -> bool:
    return bool(token and len(token) == 1 and token.isalpha())


def _is_name_token(token: str) -> bool:
    if _is_initial(token):
        return True
    # Allow O'Neil-style apostrophes after normalization.
    return bool(re.fullmatch(r"[a-z][a-z']{1,}", token))


def _has_person_title(name: Any) -> bool:
    text = str(name or "").lower().strip()
    if not text:
        return False
    text = re.sub(r"^[\s\.'\"`‘’“”]+", "", text)
    text = re.sub(r"[\.\-]", " ", text)
    parts = [p.strip("'") for p in text.split() if p.strip("'")]
    if not parts:
        return False
    return parts[0] in PERSON_TITLE_TOKENS or (len(parts) >= 2 and "".join(parts[:2]) in PERSON_TITLE_TOKENS)


def build_person_profile(row_or_name: Dict[str, Any] | Any) -> PersonProfile:
    """Build a strict person profile from a row dict or normalized name string."""
    if isinstance(row_or_name, dict):
        row = row_or_name
        precomputed = row.get("is_likely_individual")
        name = row.get("person_name_norm") or row.get("name_norm") or row.get("orig_supplier_name", "")
        if row.get("has_legal_suffix") or row.get("has_company_keyword") or row.get("is_hospitality"):
            return PersonProfile(False, ())
        tokens = _tokens(name)
        had_title = bool(row.get("person_name_had_title")) or _has_person_title(row.get("name_norm", ""))
        if precomputed is not None:
            is_likely = bool(precomputed)
        else:
            is_likely = is_likely_individual_name(name)
    else:
        name = row_or_name
        tokens = _tokens(name)
        had_title = _has_person_title(name)
        is_likely = is_likely_individual_name(name)

    if len(tokens) < 2:
        first = tokens[0] if tokens else ""
        return PersonProfile(
            bool(is_likely),
            tokens,
            first=first,
            last="",
            first_initial=first[:1],
            last_initial="",
            has_initial=any(_is_initial(t) for t in tokens),
            had_title=had_title,
        )

    first = tokens[0]
    last = tokens[-1]
    has_initial = any(_is_initial(t) for t in tokens)
    return PersonProfile(
        bool(is_likely),
        tokens,
        first=first,
        last=last,
        first_initial=first[:1],
        last_initial=last[:1],
        has_initial=has_initial,
        had_title=had_title,
    )


def is_likely_individual_name(
    name: Any,
    *,
    has_legal_suffix: bool = False,
    has_company_keyword: bool = False,
    is_hospitality: bool = False,
) -> bool:
    """Conservative detector for individual/person supplier records."""
    if has_legal_suffix or has_company_keyword or is_hospitality:
        return False
    tokens = _tokens(name)
    had_title = _has_person_title(name)
    if len(tokens) < 2:
        return bool(had_title and len(tokens) == 1 and _is_name_token(tokens[0]) and tokens[0] not in PERSON_COMPANY_MARKERS)
    if len(tokens) > 4:
        return False
    if any(t.isdigit() or any(ch.isdigit() for ch in t) for t in tokens):
        return False
    if any(t in PERSON_COMPANY_MARKERS for t in tokens):
        return False
    if not all(_is_name_token(t) for t in tokens):
        return False

    first = tokens[0]
    last = tokens[-1] if len(tokens) > 1 else ""
    has_initial = any(_is_initial(t) for t in tokens)
    any_common_name = (first in COMMON_FIRST_NAMES) or (last in COMMON_FIRST_NAMES)

    # Prefer precision: a known first/last token, title, or initial pattern is
    # enough. Checking the last token catches LAST, FIRST supplier formats after
    # punctuation cleanup, e.g. "MUELLER, GUSTAV" -> "mueller gustav".
    return bool(any_common_name or had_title or has_initial)


def is_likely_individual_row(row: Dict[str, Any]) -> bool:
    return is_likely_individual_name(
        row.get("name_norm", ""),
        has_legal_suffix=bool(row.get("has_legal_suffix")),
        has_company_keyword=bool(row.get("has_company_keyword")),
        is_hospitality=bool(row.get("is_hospitality")),
    )


def names_share_only_first_name(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    pa = build_person_profile(row_a)
    pb = build_person_profile(row_b)
    if not pa.tokens or not pb.tokens:
        return False
    # Do not fire when names are identical — exact name match handles this case.
    if pa.tokens == pb.tokens:
        return False
    shared = set(pa.tokens) & set(pb.tokens)
    if not shared:
        return False
    # Only fire when the shared first token is actually a known common first name.
    # Brand names (e.g. "Allman", "Siemens") should not trigger this guard.
    if not (shared & COMMON_FIRST_NAMES):
        return False
    return shared == {pa.first} == {pb.first}


def person_identity_strength(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> str:
    """Return `exact`, `initial`, or empty string for two person profiles."""
    pa = build_person_profile(row_a)
    pb = build_person_profile(row_b)
    if not pa.is_likely or not pb.is_likely:
        return ""
    if not pa.first or not pa.last or not pb.first or not pb.last:
        return ""

    first_exact = pa.first == pb.first and not _is_initial(pa.first) and not _is_initial(pb.first)
    last_exact = pa.last == pb.last and not _is_initial(pa.last) and not _is_initial(pb.last)
    first_initial_match = pa.first_initial and pa.first_initial == pb.first_initial
    last_initial_match = pa.last_initial and pa.last_initial == pb.last_initial
    first_initial_variation = first_initial_match and (_is_initial(pa.first) or _is_initial(pb.first))
    last_initial_variation = last_initial_match and (_is_initial(pa.last) or _is_initial(pb.last))

    if first_exact and last_exact:
        return "exact"
    if (first_exact or first_initial_variation) and (last_exact or last_initial_variation) and (first_initial_variation or last_initial_variation):
        return "initial"
    return ""


def person_company_name_evidence(person_row: Dict[str, Any], company_row: Dict[str, Any]) -> bool:
    """Require explicit person-name evidence before person-to-company address matching."""
    profile = build_person_profile(person_row)
    if not profile.tokens or not profile.first or not profile.last:
        return False
    company_name = str(company_row.get("name_norm", "") or "")
    company_tokens = set(company_name.split())
    first_supported = profile.first in company_tokens or (
        _is_initial(profile.first) and any(t.startswith(profile.first_initial) and len(t) > 1 for t in company_tokens)
    )
    last_supported = profile.last in company_tokens or (
        _is_initial(profile.last) and any(t.startswith(profile.last_initial) and len(t) > 1 for t in company_tokens)
    )
    return bool(first_supported and last_supported)

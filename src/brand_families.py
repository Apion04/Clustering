"""Known brand/family alias loading and phrase-level matching.

The keyword file is intentionally treated as a curated alias dictionary, not as
an allowed-token list. Aliases are matched as whole phrases or approved compact
phrase variants such as "fed ex" <-> "fedex".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.config import COMMON_FIRST_NAMES, GENERIC_ROOT_TOKENS, LOCATION_ROOT_TOKENS


JOIN_WORDS = {"and", "of", "the", "a", "an"}

GENERIC_ALIAS_WORDS = (
    GENERIC_ROOT_TOKENS
    | {
        "bank", "banks", "commercial", "national", "green", "red", "blue",
        "express", "energy", "services", "service", "technology",
        "technologies", "chemical", "chemicals", "pharma", "pharmaceutical",
        "international", "global", "group", "association", "community",
        "systems", "solutions", "packaging", "trading", "consulting",
    }
)

RISKY_ALIAS_WORDS = {
    "alphabet", "american", "apple", "bank", "banca", "banco", "china",
    "commercial", "delta", "deutsche", "dollar", "green", "hyundai",
    "india", "liberty", "lg", "metro", "mitsubishi", "national",
    "nippon", "orange", "phoenix", "popular", "samsung", "shell",
    "sumitomo", "tata", "toyota", "united",
    "alfa", "meta", "zimmer", "baur", "brown", "crown", "delta", "empire",
    "essex", "first", "gold", "golden", "mercury", "metro", "nova", "next",
    "orient", "pacific", "pioneer", "premium", "prime", "rapid", "red",
    "silver", "solar", "solid", "sonic", "sport", "star", "super", "swift",
    "total", "ultra", "uni", "unity", "united", "valor", "vector", "verde",
    "vesta", "victor", "vista", "vital", "vulcan", "west", "white", "world",
}

SHORT_ALIAS_WHITELIST = {
    "3m", "ge", "bp", "ibm", "sap", "dhl", "ups", "ey", "hp", "lg",
    "abb", "ubs",
}

SAFE_ALIAS_COMPACTS = {
    "accenture", "akzanobel", "akzonobel",
    "capgemini", "capgemini", "deutschepost", "dhl", "dhlgroup",
    "dpdhl", "fedex", "federalexpress", "fisher", "merck",
    "mercksharp", "microsoft", "msd", "oracle", "perkinelmer",
    "perkinelmer", "revvity", "sap", "siemens", "sigmaaldrich",
    "thermofisher", "thermofisher", "thermofisher",
}


@dataclass
class AliasHit:
    family_id: str
    alias: str
    category: str


@dataclass
class BrandFamily:
    canonical: str
    aliases: List[str] = field(default_factory=list)
    normalized_aliases: List[str] = field(default_factory=list)
    safe_aliases: List[str] = field(default_factory=list)
    risky_aliases: List[str] = field(default_factory=list)
    skipped_generic_aliases: List[str] = field(default_factory=list)
    skipped_short_aliases: List[str] = field(default_factory=list)
    confidence: float = 76.0
    allowed_family_bridge: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "canonical": self.canonical,
            "aliases": self.aliases,
            "normalized_aliases": self.normalized_aliases,
            "confidence": self.confidence,
            "allowed_family_bridge": self.allowed_family_bridge,
        }


@dataclass
class KnownBrandFamilies:
    families: Dict[str, BrandFamily] = field(default_factory=dict)
    phrase_aliases: Dict[str, List[AliasHit]] = field(default_factory=dict)
    phrase_first_token_aliases: Dict[str, List[Tuple[str, List[AliasHit]]]] = field(default_factory=dict)
    compact_aliases: Dict[str, List[AliasHit]] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.families and (self.phrase_aliases or self.compact_aliases))


def normalize_alias_phrase(value: Any) -> str:
    """Normalize a full alias phrase without removing generic words."""
    text = "" if value is None else str(value)
    text = text.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.replace("&", " and ").replace("+", " and ")
    text = text.replace("–", "-").replace("—", "-")
    text = _split_camel_case(text)
    text = _transliterate(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_alias(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_alias_phrase(value))


def _split_camel_case(value: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)


def _transliterate(value: str) -> str:
    return (
        value.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
    )


def _meaningful_tokens(alias_norm: str) -> List[str]:
    return [t for t in alias_norm.split() if t and t not in JOIN_WORDS]


def classify_alias(alias_norm: str) -> Tuple[str, List[str]]:
    """Return safe/risky/generic/skip_short and warnings."""
    warnings: List[str] = []
    compact = compact_alias(alias_norm)
    tokens = _meaningful_tokens(alias_norm)
    if not alias_norm or not compact:
        return "generic", ["blank alias"]

    if len(compact) <= 2 and compact not in SHORT_ALIAS_WHITELIST:
        return "skip_short", [f"very short alias skipped: {alias_norm}"]

    if len(compact) <= 3 and compact in SHORT_ALIAS_WHITELIST:
        warnings.append(f"short whitelisted alias: {alias_norm}")

    if not tokens:
        return "generic", [f"generic alias skipped: {alias_norm}"]

    if len(tokens) == 1 and (tokens[0] in GENERIC_ALIAS_WORDS or tokens[0] in COMMON_FIRST_NAMES):
        return "generic", [f"single generic/common alias skipped: {alias_norm}"]

    # Short single-token aliases are broad by default. Treat them as risky
    # unless they are an explicit safe compact alias such as FedEx/Capgemini,
    # or a whitelisted short acronym such as 3M/SAP/DHL.
    if len(tokens) == 1 and len(compact) <= 6 and compact not in SHORT_ALIAS_WHITELIST and compact not in SAFE_ALIAS_COMPACTS:
        return "risky", [f"short single-token alias ({len(compact)} chars) marked risky: {alias_norm}"]

    risky_hits = [t for t in tokens if t in RISKY_ALIAS_WORDS or t in LOCATION_ROOT_TOKENS]
    all_generic = all(t in GENERIC_ALIAS_WORDS or t in LOCATION_ROOT_TOKENS or t in COMMON_FIRST_NAMES for t in tokens)
    if all_generic and risky_hits:
        warnings.append(f"risky/common/location token(s): {', '.join(sorted(set(risky_hits)))}")
        return "risky", warnings
    if all_generic and compact not in SHORT_ALIAS_WHITELIST and compact not in SAFE_ALIAS_COMPACTS:
        return "generic", [f"all alias tokens are generic/location/common: {alias_norm}"]

    if compact in SAFE_ALIAS_COMPACTS:
        if any(t in GENERIC_ALIAS_WORDS or t in LOCATION_ROOT_TOKENS for t in tokens):
            warnings.append(f"safe alias contains generic/location token(s): {alias_norm}")
        return "safe", warnings

    if risky_hits:
        warnings.append(f"risky/common/location token(s): {', '.join(sorted(set(risky_hits)))}")
        return "risky", warnings

    if len(compact) <= 3 and compact in SHORT_ALIAS_WHITELIST:
        return "risky", warnings

    if any(t in GENERIC_ALIAS_WORDS for t in tokens):
        warnings.append(f"alias overlaps generic word(s): {', '.join(sorted(set(tokens) & GENERIC_ALIAS_WORDS))}")
    return "safe", warnings


def load_known_brand_families(
    path: Optional[str],
    *,
    default_confidence: float = 76.0,
) -> KnownBrandFamilies:
    """Load semicolon-separated brand families from a one-column CSV."""
    if not path:
        return KnownBrandFamilies(report=_empty_report(path))
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    if not file_path.exists():
        return KnownBrandFamilies(report={**_empty_report(str(file_path)), "missing_file": True})

    rows_loaded = 0
    accepted_safe = 0
    accepted_risky = 0
    skipped_generic = 0
    skipped_short = 0
    warnings: List[Dict[str, str]] = []
    families: Dict[str, BrandFamily] = {}
    canonical_by_alias: Dict[str, str] = {}

    with file_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        keyword_col = "keyword" if "keyword" in fieldnames else (fieldnames[0] if fieldnames else None)
        if not keyword_col:
            return KnownBrandFamilies(report={**_empty_report(str(file_path)), "missing_keyword_column": True})
        for csv_row in reader:
            raw_cell = (csv_row.get(keyword_col) or "").strip()
            if not raw_cell:
                continue
            parts = [p.strip() for p in raw_cell.split(";") if p.strip()]
            if not parts:
                continue
            rows_loaded += 1
            canonical_raw = parts[0]
            canonical_norm = normalize_alias_phrase(canonical_raw)
            family_id = compact_alias(canonical_norm)
            if not family_id:
                continue
            if family_id in canonical_by_alias:
                family_id = canonical_by_alias[family_id]
            family = families.setdefault(
                family_id,
                BrandFamily(canonical=canonical_raw, confidence=float(default_confidence)),
            )
            raw_aliases = parts
            for raw_alias in raw_aliases:
                alias_norm = normalize_alias_phrase(raw_alias)
                alias_compact = compact_alias(alias_norm)
                if not alias_norm or not alias_compact:
                    continue
                category, alias_warnings = classify_alias(alias_norm)
                for warning in alias_warnings:
                    warnings.append({"alias": raw_alias, "normalized": alias_norm, "warning": warning})
                if category == "generic":
                    skipped_generic += 1
                    if alias_norm not in family.skipped_generic_aliases:
                        family.skipped_generic_aliases.append(alias_norm)
                    continue
                if category == "skip_short":
                    skipped_short += 1
                    if alias_norm not in family.skipped_short_aliases:
                        family.skipped_short_aliases.append(alias_norm)
                    continue
                if raw_alias not in family.aliases:
                    family.aliases.append(raw_alias)
                if alias_norm not in family.normalized_aliases:
                    family.normalized_aliases.append(alias_norm)
                if category == "safe":
                    accepted_safe += 1
                    if alias_norm not in family.safe_aliases:
                        family.safe_aliases.append(alias_norm)
                else:
                    accepted_risky += 1
                    if alias_norm not in family.risky_aliases:
                        family.risky_aliases.append(alias_norm)
                canonical_by_alias.setdefault(alias_compact, family_id)

    index = KnownBrandFamilies(families=families)
    for family_id, family in families.items():
        for alias_norm in family.safe_aliases:
            _index_alias(index, family_id, alias_norm, "safe")
        for alias_norm in family.risky_aliases:
            _index_alias(index, family_id, alias_norm, "risky")

    alias_phrase_count = sum(len(f.normalized_aliases) for f in families.values())
    accepted_safe_count = sum(len(f.safe_aliases) for f in families.values())
    accepted_risky_count = sum(len(f.risky_aliases) for f in families.values())
    skipped_generic_count = sum(len(f.skipped_generic_aliases) for f in families.values())
    skipped_short_count = sum(len(f.skipped_short_aliases) for f in families.values())
    index.report = {
        "file": str(file_path),
        "rows_loaded": rows_loaded,
        "families_loaded": len(families),
        "alias_phrases_loaded": alias_phrase_count,
        "aliases_skipped_generic": skipped_generic_count,
        "aliases_skipped_short": skipped_short_count,
        "aliases_marked_risky": accepted_risky_count,
        "aliases_accepted_safe": accepted_safe_count,
        "examples_safe": _examples(families.values(), "safe_aliases"),
        "examples_risky": _examples(families.values(), "risky_aliases"),
        "examples_skipped_generic": _examples(families.values(), "skipped_generic_aliases"),
        "warnings": warnings[:100],
    }
    return index


def _empty_report(path: Optional[str]) -> Dict[str, Any]:
    return {
        "file": path or "",
        "rows_loaded": 0,
        "families_loaded": 0,
        "alias_phrases_loaded": 0,
        "aliases_skipped_generic": 0,
        "aliases_skipped_short": 0,
        "aliases_marked_risky": 0,
        "aliases_accepted_safe": 0,
        "examples_safe": [],
        "examples_risky": [],
        "examples_skipped_generic": [],
        "warnings": [],
    }


def _index_alias(index: KnownBrandFamilies, family_id: str, alias_norm: str, category: str) -> None:
    hit = AliasHit(family_id=family_id, alias=alias_norm, category=category)
    index.phrase_aliases.setdefault(alias_norm, []).append(hit)
    first_token = next(iter(_meaningful_tokens(alias_norm)), "")
    if first_token:
        index.phrase_first_token_aliases.setdefault(first_token, []).append((alias_norm, index.phrase_aliases[alias_norm]))
    alias_compact = compact_alias(alias_norm)
    if alias_compact:
        index.compact_aliases.setdefault(alias_compact, []).append(hit)


def _examples(families: Iterable[BrandFamily], attr: str, limit: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for family in families:
        values = getattr(family, attr)
        if values:
            out.append({"canonical": family.canonical, "aliases": values[:5]})
        if len(out) >= limit:
            break
    return out


def find_known_brand_family_hits(values: Iterable[Any], index: Optional[KnownBrandFamilies]) -> List[AliasHit]:
    if not index or not index.enabled:
        return []
    hits: Dict[Tuple[str, str], AliasHit] = {}
    for value in values:
        norm_text = normalize_alias_phrase(value)
        if not norm_text:
            continue
        text_with_boundaries = f" {norm_text} "
        tokens = set(norm_text.split())
        compact_tokens = {compact_alias(t) for t in tokens if t}

        phrase_candidates: Dict[str, List[AliasHit]] = {}
        for token in tokens:
            for alias_norm, alias_hits in index.phrase_first_token_aliases.get(token, []):
                phrase_candidates[alias_norm] = alias_hits

        for alias_norm, alias_hits in phrase_candidates.items():
            alias_tokens = alias_norm.split()
            matched = False
            if len(alias_tokens) == 1:
                token = alias_tokens[0]
                matched = token in tokens
            else:
                matched = f" {alias_norm} " in text_with_boundaries
            if matched:
                for hit in alias_hits:
                    hits[(hit.family_id, hit.alias)] = hit

        # Compact variants catch FedEx/Fed Ex, Capgemini/Cap Gemini, AkzoNobel/Akzo Nobel.
        for token_compact in compact_tokens:
            # Allow compact aliases to match a full compact token only. Do not
            # scan arbitrary substrings, which would turn aliases into weak tokens.
            for hit in index.compact_aliases.get(token_compact, []):
                hits[(hit.family_id, hit.alias)] = hit
    return list(hits.values())


def encode_hits(hits: List[AliasHit]) -> Dict[str, str]:
    families = sorted({hit.family_id for hit in hits})
    safe = sorted({hit.family_id for hit in hits if hit.category == "safe"})
    risky = sorted({hit.family_id for hit in hits if hit.category == "risky"})
    aliases = sorted({f"{hit.family_id}:{hit.category}:{hit.alias}" for hit in hits})
    return {
        "known_brand_family_ids": "|".join(families),
        "known_brand_family_safe_ids": "|".join(safe),
        "known_brand_family_risky_ids": "|".join(risky),
        "known_brand_alias_hits": "|".join(aliases),
    }


def decode_alias_hits(encoded: Any) -> List[AliasHit]:
    hits: List[AliasHit] = []
    for item in str(encoded or "").split("|"):
        if not item:
            continue
        parts = item.split(":", 2)
        if len(parts) != 3:
            continue
        hits.append(AliasHit(family_id=parts[0], category=parts[1], alias=parts[2]))
    return hits


# ---------------------------------------------------------------------------
# Global brand alias tables (Phase B — CSV-driven, risk-level aware)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AliasEvidence:
    """Result of _alias_bridge(): both rows resolve to the same canonical brand."""
    canonical_brand: str
    alias_match_type: str   # "brand_alias" | "domain_alias" | "acronym_alias"
    score_ceiling: int      # from CSV score_hint; hard ceiling for alias-only evidence
    risk_level: str         # "low" | "medium" | "high"
    alias_a: str            # normalized token that matched for row A
    alias_b: str            # normalized token that matched for row B


class AliasTables:
    """In-memory lookup tables loaded from alias CSVs.

    All dicts are empty when CSV files are absent — the matching engine
    behaves identically to a pre-alias-framework run in that case.
    """
    __slots__ = (
        "alias_to_entry",    # alias str → (canonical, score_hint, risk_level)
        "domain_to_entry",   # domain str → (canonical_brand, canonical_family, score_hint, risk_level)
        "acronym_to_entry",  # acronym str → (canonical_brand, score_hint, risk_level)
    )

    def __init__(self) -> None:
        self.alias_to_entry: Dict[str, Tuple[str, int, str]] = {}
        self.domain_to_entry: Dict[str, Tuple[str, str, int, str]] = {}
        self.acronym_to_entry: Dict[str, Tuple[str, int, str]] = {}

    @property
    def is_empty(self) -> bool:
        return not self.alias_to_entry and not self.domain_to_entry and not self.acronym_to_entry


def _parse_int(value: Any, default: int = 70) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def load_alias_tables(data_dir: str = "data") -> AliasTables:
    """Load alias CSVs from data_dir.  Missing or malformed files are silently skipped."""
    tables = AliasTables()
    base = Path(data_dir)

    # brand_aliases.csv — alias → canonical
    path = base / "brand_aliases.csv"
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    alias = str(row.get("alias", "") or "").strip().lower()
                    canonical = str(row.get("canonical_brand", "") or "").strip().lower()
                    score = _parse_int(row.get("score_hint", 70))
                    risk = str(row.get("risk_level", "medium") or "medium").strip().lower()
                    if alias and canonical:
                        tables.alias_to_entry[alias] = (canonical, score, risk)
        except Exception:
            pass  # graceful degradation — missing/corrupt file → empty table

    # domain_aliases.csv — domain → canonical brand/family
    path = base / "domain_aliases.csv"
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    domain = str(row.get("domain", "") or "").strip().lower()
                    family = str(row.get("canonical_domain_family", "") or "").strip().lower()
                    canonical = str(row.get("canonical_brand", "") or "").strip().lower()
                    score = _parse_int(row.get("score_hint", 70))
                    risk = str(row.get("risk_level", "medium") or "medium").strip().lower()
                    if domain and canonical:
                        tables.domain_to_entry[domain] = (canonical, family, score, risk)
        except Exception:
            pass

    # acronym_aliases.csv — acronym → canonical
    path = base / "acronym_aliases.csv"
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    acronym = str(row.get("acronym", "") or "").strip().lower()
                    canonical = str(row.get("canonical_brand", "") or "").strip().lower()
                    score = _parse_int(row.get("score_hint", 70))
                    risk = str(row.get("risk_level", "medium") or "medium").strip().lower()
                    if acronym and canonical:
                        tables.acronym_to_entry[acronym] = (canonical, score, risk)
        except Exception:
            pass

    return tables


_CACHED_ALIAS_TABLES: Optional[AliasTables] = None
_CACHED_ALIAS_DIR: str = ""


def get_alias_tables(data_dir: str = "data") -> AliasTables:
    """Return cached alias tables, reloading if data_dir changed."""
    global _CACHED_ALIAS_TABLES, _CACHED_ALIAS_DIR
    if _CACHED_ALIAS_TABLES is None or _CACHED_ALIAS_DIR != data_dir:
        _CACHED_ALIAS_TABLES = load_alias_tables(data_dir)
        _CACHED_ALIAS_DIR = data_dir
    return _CACHED_ALIAS_TABLES


def _resolve_row_to_brand(
    supplier_identity_core: str,
    name_norm: str,
    domain: str,
    is_generic_domain: bool,
    tables: AliasTables,
    ignored_domains: "frozenset[str]",
) -> "Optional[Tuple[str, int, str, str]]":
    """Try to resolve a single row to (canonical_brand, score_hint, risk_level, match_type).

    Checks in order: supplier_identity_core exact → name_norm exact → acronym → domain.
    Returns None when no alias fires.
    """
    # 1. Exact match on supplier_identity_core (already stripped of legal suffixes)
    if supplier_identity_core:
        entry = tables.alias_to_entry.get(supplier_identity_core)
        if entry:
            return (entry[0], entry[1], entry[2], "brand_alias")

    # 2. Exact match on normalized name
    if name_norm and name_norm != supplier_identity_core:
        entry = tables.alias_to_entry.get(name_norm)
        if entry:
            return (entry[0], entry[1], entry[2], "brand_alias")

    # 3. Acronym match — only if the entire supplier_identity_core is the acronym
    #    (prevents partial-name false positives)
    if supplier_identity_core and " " not in supplier_identity_core:
        entry = tables.acronym_to_entry.get(supplier_identity_core)
        if entry:
            return (entry[0], entry[1], entry[2], "acronym_alias")

    # 4. Domain alias — only for non-generic, non-ignored domains
    if (
        domain
        and not is_generic_domain
        and domain not in ignored_domains
    ):
        entry = tables.domain_to_entry.get(domain)
        if entry:
            return (entry[0], entry[2], entry[3], "domain_alias")

    return None

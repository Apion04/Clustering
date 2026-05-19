"""Multilingual generic/non-bridge keyword loading.

Generic keywords suppress weak evidence. They are not aliases, not legal
suffixes, and never create clusters by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


DEFAULT_GENERIC_BY_CATEGORY: Dict[str, Set[str]] = {
    "english": {
        "services", "service", "technology", "technologies", "association",
        "society", "community", "foundation", "institute", "university",
        "college", "academy", "school", "hospital", "clinic", "center",
        "centre", "laboratory", "lab", "labs", "consulting", "group",
        "holdings", "company", "business", "enterprise", "research",
        "development", "logistics", "transport", "shipping", "trading",
        "marketing", "publishing", "media", "events", "solutions",
        "systems", "software", "data", "digital", "electronics",
        "automation", "engineering", "materials", "chemical", "chemicals",
        "pharma", "pharmaceutical", "medical", "healthcare", "biotech",
        "bio", "scientific", "science", "industrial", "manufacturing",
        "production", "packaging", "cargo", "express", "energy", "power",
        "water", "environment", "green", "global", "international",
        "national",
    },
    "german": {
        "gesellschaft", "unternehmen", "gruppe", "holding", "verein",
        "stiftung", "institut", "universitaet", "hochschule", "akademie",
        "krankenhaus", "klinik", "zentrum", "labor", "dienstleistung",
        "dienstleistungen", "beratung", "handel", "vertrieb", "technik",
        "technologie", "forschung", "entwicklung", "logistik", "spedition",
    },
    "french": {
        "societe", "association", "groupe", "fondation", "institut",
        "universite", "ecole", "hopital", "clinique", "centre",
        "laboratoire", "services", "conseil", "technologie", "technique",
        "recherche", "developpement", "commerce", "distribution",
        "logistique",
    },
    "spanish_portuguese": {
        "sociedad", "associacion", "asociacion", "associacao", "grupo",
        "fundacion", "instituto", "universidad", "colegio", "escuela",
        "hospital", "clinica", "centro", "laboratorio", "servicios",
        "servico", "consultoria", "tecnologia", "investigacion",
        "desarrollo", "comercio", "distribucion", "logistica",
    },
    "italian": {
        "societa", "associazione", "gruppo", "fondazione", "istituto",
        "universita", "scuola", "ospedale", "clinica", "centro",
        "laboratorio", "servizi", "consulenza", "tecnologia", "tecnico",
        "ricerca", "sviluppo", "commercio", "distribuzione", "logistica",
    },
    "dutch_nordic": {
        "maatschappij", "vereniging", "stichting", "instituut",
        "universiteit", "hogeschool", "ziekenhuis", "kliniek", "centrum",
        "laboratorium", "diensten", "advies", "technologie", "onderzoek",
        "ontwikkeling", "handel", "distributie", "logistiek", "selskab",
        "aktiebolag", "aksjeselskap",
    },
    "common": {
        "green", "blue", "red", "white", "black", "golden", "silver",
        "new", "prime", "advanced", "global", "international", "national",
        "regional", "local", "city", "capital", "union", "united",
    },
}


@dataclass
class GenericKeywordDictionary:
    keywords: Set[str] = field(default_factory=set)
    by_category: Dict[str, Set[str]] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.keywords)


def normalize_generic_keyword(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.replace("&", " and ").replace("+", " and ")
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    text = _transliterate(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_generic_non_bridge_keywords(path: Optional[str], *, include_defaults: bool = True) -> GenericKeywordDictionary:
    by_category: Dict[str, Set[str]] = {}
    duplicates = 0
    risky_short: Set[str] = set()
    seen: Set[str] = set()
    blank_rows = 0
    rows_loaded = 0
    missing_file = False

    def add(category: str, raw: Any):
        nonlocal duplicates
        norm = normalize_generic_keyword(raw)
        if not norm:
            return
        # Generic suppression is token/phrase based; keep only reasonably safe
        # forms to avoid accidental substring behavior.
        if len(norm.replace(" ", "")) <= 2:
            risky_short.add(norm)
        cat = category or "uncategorized"
        current = by_category.setdefault(cat, set())
        if norm in seen:
            duplicates += 1
        seen.add(norm)
        current.add(norm)

    if include_defaults:
        for category, values in DEFAULT_GENERIC_BY_CATEGORY.items():
            for value in values:
                add(category, value)

    file_path = Path(path) if path else None
    if file_path and not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    if file_path and file_path.exists():
        with file_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            keyword_col = _keyword_column(fields)
            category_col = _category_column(fields)
            for row in reader:
                raw = (row.get(keyword_col) or "").strip() if keyword_col else ""
                if not raw:
                    blank_rows += 1
                    continue
                rows_loaded += 1
                add((row.get(category_col) or "file") if category_col else "file", raw)
    elif file_path:
        missing_file = True

    keywords = set().union(*by_category.values()) if by_category else set()
    report = {
        "file": str(file_path) if file_path else "",
        "rows_loaded": rows_loaded,
        "total_generic_words_loaded": len(keywords),
        "duplicates_skipped": duplicates,
        "blank_rows_skipped": blank_rows,
        "risky_short_words": sorted(risky_short),
        "examples_by_category": {category: sorted(values)[:20] for category, values in sorted(by_category.items())},
        "missing_file": missing_file,
        "built_in_defaults_included": include_defaults,
        "warnings": [
            {"keyword": word, "warning": "very short generic word; review before relying on it"}
            for word in sorted(risky_short)
        ],
    }
    return GenericKeywordDictionary(keywords=keywords, by_category=by_category, report=report)


def apply_generic_non_bridge_keywords(index: GenericKeywordDictionary) -> None:
    """Mutate shared suppression sets so existing modules see loaded generics."""
    if not index or not index.keywords:
        return
    from src import config

    config.GENERIC_ROOT_TOKENS.update(index.keywords)
    try:
        from src import brand_families

        brand_families.GENERIC_ALIAS_WORDS.update(index.keywords)
    except Exception:
        pass
    try:
        from src import person

        person.PERSON_COMPANY_MARKERS.update(index.keywords)
    except Exception:
        pass


def _keyword_column(fields: List[str]) -> Optional[str]:
    if not fields:
        return None
    normalized = {str(name).strip().lower(): name for name in fields}
    for preferred in ("keyword", "keywords", "generic", "word"):
        if preferred in normalized:
            return normalized[preferred]
    return fields[-1]


def _category_column(fields: List[str]) -> Optional[str]:
    normalized = {str(name).strip().lower(): name for name in fields}
    for preferred in ("category", "language", "type"):
        if preferred in normalized:
            return normalized[preferred]
    return None


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

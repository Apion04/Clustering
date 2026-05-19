"""Legal entity suffix dictionary loading and suffix-only normalization.

This module is deliberately separate from known brand/family aliases. Legal
forms such as GmbH, Ltd, AG, BV, or A/S are entity hints and name-normalization
noise; they must never become brand/family bridge tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.config import LEGAL_SUFFIXES


SHORT_LEGAL_FORM_WHITELIST = {
    "ab", "ac", "ad", "ae", "ag", "as", "bv", "cv", "ee", "ev",
    "gk", "kg", "ks", "lc", "lp", "nv", "oy", "sa", "se", "sl",
    "sn", "sp", "ua",
}


@dataclass
class LegalKeywordDictionary:
    normalized_suffixes: Set[str] = field(default_factory=set)
    compact_suffixes: Set[str] = field(default_factory=set)
    suffix_token_tuples: Set[Tuple[str, ...]] = field(default_factory=set)
    phrase_first_token_suffixes: Dict[str, List[str]] = field(default_factory=dict)
    boundary_only_compacts: Set[str] = field(default_factory=set)
    max_tokens: int = 1
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.normalized_suffixes or self.compact_suffixes)


def normalize_legal_keyword(value: Any) -> str:
    """Normalize a legal-form phrase while preserving token boundaries."""
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


def compact_legal_keyword(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_legal_keyword(value))


def load_legal_keywords(path: Optional[str], *, include_defaults: bool = True) -> LegalKeywordDictionary:
    """Load legal suffix/company-form keywords from a one-column CSV.

    The loaded dictionary is used for suffix cleanup and company/entity hints
    only. It is not used for candidate generation or brand/family bridging.
    """
    raw_values: List[Tuple[str, str]] = []
    if include_defaults:
        raw_values.extend(("built_in", value) for value in LEGAL_SUFFIXES)

    file_path: Optional[Path] = Path(path) if path else None
    if file_path and not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    rows_loaded = 0
    blank_skipped = 0
    missing_file = False
    missing_keyword_column = False

    if file_path and file_path.exists():
        with file_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            keyword_col = _keyword_column(fieldnames)
            if not keyword_col:
                missing_keyword_column = True
            else:
                for csv_row in reader:
                    raw = (csv_row.get(keyword_col) or "").strip()
                    if not raw:
                        blank_skipped += 1
                        continue
                    rows_loaded += 1
                    raw_values.append(("file", raw))
    elif file_path:
        missing_file = True

    index = _build_index(raw_values)
    report = index.report
    report.update({
        "file": str(file_path) if file_path else "",
        "rows_loaded": rows_loaded,
        "blank_rows_skipped": blank_skipped,
        "missing_file": missing_file,
        "missing_keyword_column": missing_keyword_column,
        "built_in_defaults_included": include_defaults,
    })
    return index


@lru_cache(maxsize=1)
def get_default_legal_keywords() -> LegalKeywordDictionary:
    return load_legal_keywords(None, include_defaults=True)


def strip_legal_suffixes(name: Any, index: Optional[LegalKeywordDictionary] = None) -> Tuple[str, List[str]]:
    """Strip legal forms only from the end of a normalized supplier name."""
    dictionary = index or get_default_legal_keywords()
    text = normalize_legal_keyword(name)
    if not text or not dictionary.enabled:
        return text, []

    tokens = text.split()
    original_tokens = list(tokens)
    removed: List[str] = []

    while tokens:
        match = _tail_match(tokens, dictionary)
        if not match:
            break
        n_tokens, suffix = match
        removed.append(suffix)
        tokens = tokens[:-n_tokens]

    stripped = " ".join(tokens).strip()
    if not stripped:
        return " ".join(original_tokens), removed
    return stripped, removed


def remove_legal_suffixes_from_name(name: Any, index: Optional[LegalKeywordDictionary] = None) -> str:
    return strip_legal_suffixes(name, index)[0]


def find_legal_keyword_matches(name: Any, index: Optional[LegalKeywordDictionary] = None) -> List[str]:
    """Find legal-form phrases in a name using token boundaries only."""
    dictionary = index or get_default_legal_keywords()
    text = normalize_legal_keyword(name)
    if not text or not dictionary.enabled:
        return []
    tokens = set(text.split())
    text_with_boundaries = f" {text} "
    matches: Set[str] = set()
    for token in tokens:
        for phrase in dictionary.phrase_first_token_suffixes.get(token, []):
            phrase_tokens = phrase.split()
            if len(phrase_tokens) == 1:
                if phrase_tokens[0] in tokens:
                    matches.add(phrase)
            elif f" {phrase} " in text_with_boundaries:
                matches.add(phrase)
    return sorted(matches)


def _keyword_column(fieldnames: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    normalized = {str(name).strip().lower(): name for name in fieldnames}
    for preferred in ("legal keywords", "legal_keyword", "legal keyword", "keyword"):
        if preferred in normalized:
            return normalized[preferred]
    return fieldnames[0]


def _build_index(raw_values: Iterable[Tuple[str, str]]) -> LegalKeywordDictionary:
    normalized_suffixes: Set[str] = set()
    compact_suffixes: Set[str] = set()
    suffix_token_tuples: Set[Tuple[str, ...]] = set()
    phrase_first_token_suffixes: Dict[str, List[str]] = {}
    boundary_only_compacts: Set[str] = set()
    duplicates_skipped = 0
    too_short_skipped = 0
    short_risky: List[str] = []
    boundary_warnings: List[str] = []
    examples: List[str] = []

    for _source, raw in raw_values:
        for variant in _keyword_variants(raw):
            norm = normalize_legal_keyword(variant)
            compact = compact_legal_keyword(norm)
            if not norm or not compact:
                continue
            if len(compact) < 2:
                too_short_skipped += 1
                continue
            if compact in compact_suffixes or norm in normalized_suffixes:
                duplicates_skipped += 1
                continue
            tokens = tuple(norm.split())
            normalized_suffixes.add(norm)
            compact_suffixes.add(compact)
            suffix_token_tuples.add(tokens)
            phrase_first_token_suffixes.setdefault(tokens[0], []).append(norm)
            if len(compact) <= 2:
                boundary_only_compacts.add(compact)
                if compact not in SHORT_LEGAL_FORM_WHITELIST:
                    short_risky.append(norm)
                boundary_warnings.append(norm)
            if len(examples) < 25:
                examples.append(norm)

    max_tokens = max((len(t) for t in suffix_token_tuples), default=1)
    report = {
        "legal_keywords_loaded": len(normalized_suffixes),
        "duplicates_skipped": duplicates_skipped,
        "too_short_skipped": too_short_skipped,
        "risky_very_short_suffixes": sorted(set(short_risky))[:100],
        "boundary_only_suffixes": sorted(set(boundary_warnings))[:100],
        "examples_loaded": examples,
        "warnings": [
            {"suffix": suffix, "warning": "very short legal form; token-boundary and suffix-position matching only"}
            for suffix in sorted(set(boundary_warnings))[:100]
        ],
    }
    return LegalKeywordDictionary(
        normalized_suffixes=normalized_suffixes,
        compact_suffixes=compact_suffixes,
        suffix_token_tuples=suffix_token_tuples,
        phrase_first_token_suffixes=phrase_first_token_suffixes,
        boundary_only_compacts=boundary_only_compacts,
        max_tokens=max_tokens,
        report=report,
    )


def _tail_match(tokens: List[str], dictionary: LegalKeywordDictionary) -> Optional[Tuple[int, str]]:
    max_n = min(dictionary.max_tokens, len(tokens))
    for n in range(max_n, 0, -1):
        tail_tokens = tuple(tokens[-n:])
        phrase = " ".join(tail_tokens)
        compact = "".join(tail_tokens)
        if tail_tokens in dictionary.suffix_token_tuples or phrase in dictionary.normalized_suffixes:
            return n, phrase
        if compact in dictionary.compact_suffixes:
            return n, compact
    return None


def _keyword_variants(raw: str) -> List[str]:
    values: List[str] = []
    raw_text = str(raw or "").strip()
    if raw_text:
        values.append(raw_text)
    for segment in re.split(r"\s*/\s*|\s*;\s*", raw_text):
        segment = segment.strip()
        if not segment:
            continue
        values.append(segment)
        outside = re.sub(r"\([^)]*\)", " ", segment).strip()
        if outside and outside != segment:
            values.append(outside)
        for paren in re.findall(r"\(([^)]*)\)", segment):
            if paren.strip():
                values.append(paren.strip())
    return values


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

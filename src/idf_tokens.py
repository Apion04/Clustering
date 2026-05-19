"""Input-specific rare/discriminative token support.

Rare tokens are deliberately a supporting signal only. They can help prioritize
review candidates and explain why a pair is interesting, but they must never
create a main cluster by themselves.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Set

import polars as pl

from src.brand_families import GENERIC_ALIAS_WORDS, RISKY_ALIAS_WORDS
from src.config import (
    COMMON_FIRST_NAMES,
    COMPANY_HINT_WORDS,
    GENERIC_ROOT_TOKENS,
    HOSPITALITY_TERMS,
    LOCATION_ROOT_TOKENS,
)


SHORT_TOKEN_WHITELIST = {"3m", "ge", "bp", "ibm", "sap", "dhl", "ups", "ey", "hp", "lg", "abb", "ubs"}


@dataclass
class RareTokenIndex:
    token_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    row_tokens: Dict[int, Set[str]] = field(default_factory=dict)
    max_document_fraction: float = 0.03

    @property
    def enabled(self) -> bool:
        return bool(self.token_stats)


def build_rare_token_index(df: pl.DataFrame, config: Any, legal_keywords: Any = None) -> RareTokenIndex:
    """Build a per-input token document-frequency table from normalized names."""
    n_rows = len(df)
    max_fraction = float(getattr(config, "rare_token_max_document_fraction", 0.03) or 0.03)
    min_df = int(getattr(config, "rare_token_min_document_frequency", 1) or 1)
    legal_tokens = _legal_tokens(legal_keywords)

    document_frequency: Counter[str] = Counter()
    row_all_tokens: Dict[int, Set[str]] = {}
    for row in df.iter_rows(named=True):
        row_id = int(row["row_id"])
        tokens = set(str(row.get("name_norm", "") or "").split())
        row_all_tokens[row_id] = tokens
        for token in tokens:
            if _eligible_token(token, legal_tokens):
                document_frequency[token] += 1

    token_stats: Dict[str, Dict[str, float]] = {}
    for token, df_count in document_frequency.items():
        fraction = df_count / max(n_rows, 1)
        if df_count < min_df:
            continue
        if fraction > max_fraction:
            continue
        token_stats[token] = {
            "document_frequency": float(df_count),
            "document_fraction": fraction,
            "idf": math.log((n_rows + 1) / (df_count + 1)) + 1.0,
        }

    row_tokens = {
        row_id: {token for token in tokens if token in token_stats}
        for row_id, tokens in row_all_tokens.items()
    }

    index = RareTokenIndex(token_stats=token_stats, row_tokens=row_tokens, max_document_fraction=max_fraction)
    config._rare_token_index = index
    config._rare_token_report = {
        "enabled": True,
        "max_document_fraction": max_fraction,
        "input_rows": n_rows,
        "eligible_rare_tokens": len(token_stats),
        "top_rare_tokens": [
            {
                "token": token,
                "document_frequency": int(stats["document_frequency"]),
                "document_fraction": round(stats["document_fraction"], 6),
                "idf": round(stats["idf"], 4),
            }
            for token, stats in sorted(token_stats.items(), key=lambda item: (-item[1]["idf"], item[0]))[:50]
        ],
    }
    return index


def annotate_rare_tokens(df: pl.DataFrame, index: RareTokenIndex) -> pl.DataFrame:
    values = []
    for row in df.iter_rows(named=True):
        tokens = sorted(index.row_tokens.get(int(row["row_id"]), set()))
        values.append("|".join(tokens))
    return df.with_columns(pl.Series("idf_discriminative_tokens", values))


def shared_rare_tokens(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> Set[str]:
    a = {t for t in str(row_a.get("idf_discriminative_tokens", "") or "").split("|") if t}
    b = {t for t in str(row_b.get("idf_discriminative_tokens", "") or "").split("|") if t}
    return a & b


def _eligible_token(token: str, legal_tokens: Set[str]) -> bool:
    if not token:
        return False
    if len(token) < 3 and token not in SHORT_TOKEN_WHITELIST:
        return False
    if any(ch.isdigit() for ch in token) and token not in SHORT_TOKEN_WHITELIST:
        return False
    if token in legal_tokens:
        return False
    if token in GENERIC_ROOT_TOKENS or token in LOCATION_ROOT_TOKENS:
        return False
    if token in GENERIC_ALIAS_WORDS or token in RISKY_ALIAS_WORDS:
        return False
    if token in HOSPITALITY_TERMS or token in COMMON_FIRST_NAMES or token in COMPANY_HINT_WORDS:
        return False
    return True


def _legal_tokens(legal_keywords: Any) -> Set[str]:
    out: Set[str] = set()
    for suffix in getattr(legal_keywords, "normalized_suffixes", set()) or set():
        out.update(str(suffix).split())
    return out

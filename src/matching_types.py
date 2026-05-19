"""Shared matching dataclasses."""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MatchResult:
    is_match: bool
    match_pct: float
    pass_type: str
    evidence: Dict[str, Any]
    needs_review: bool = False
    review_reason: str = ""

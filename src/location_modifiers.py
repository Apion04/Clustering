"""Load location modifier terms from CSV for branch/location suffix stripping."""

import csv
from typing import Set


def load_location_modifiers(file_path: str) -> Set[str]:
    """Load location term set from CSV. Returns empty set if file is missing."""
    terms: Set[str] = set()
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = str(row.get("term", "") or "").strip().lower()
                if term and len(term) >= 2:
                    # Multi-word terms (e.g. "new york") are stored as-is for
                    # phrase-level lookup; single tokens are added individually.
                    terms.add(term)
                    # Also add each token of multi-word terms so they can be
                    # matched as trailing single tokens in hyphen-stripped names.
                    for tok in term.split():
                        if len(tok) >= 5:
                            terms.add(tok)
    except Exception:
        pass
    return terms

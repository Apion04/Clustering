"""Tests for matching engine."""

import pytest
from src.matching import calculate_name_similarity, evaluate_pair


class TestNameSimilarity:
    def test_exact_match(self):
        assert calculate_name_similarity("abs safety", "abs safety") == 1.0

    def test_case_insensitive(self):
        sim = calculate_name_similarity("ABS Safety", "abs safety")
        assert sim == 1.0

    def test_word_order(self):
        sim = calculate_name_similarity("abc czepczynski", "czepczynski abc")
        assert sim >= 0.95

    def test_abbreviation(self):
        sim = calculate_name_similarity("rmhc ohio valley", "ronald mcdonald house charities ohio valley")
        assert sim >= 0.70


class TestEvaluatePair:
    def test_tax_exact_match(self):
        row_a = {"name_norm": "abs safety", "tax_norm": "233002380", "addr_norm": "", "domain": "", "is_generic_domain": False}
        row_b = {"name_norm": "abs safety gmbh", "tax_norm": "233002380", "addr_norm": "", "domain": "", "is_generic_domain": False}

        result = evaluate_pair(row_a, row_b, {})
        assert result.is_match is True
        assert result.match_pct == 99.0
        assert result.pass_type == "tax_exact"

    def test_name_address_exact(self):
        row_a = {"name_norm": "abs safety", "tax_norm": "", "addr_norm": "gewerbering 3", "domain": "", "is_generic_domain": False}
        row_b = {"name_norm": "abs safety", "tax_norm": "", "addr_norm": "gewerbering 3", "domain": "", "is_generic_domain": False}

        result = evaluate_pair(row_a, row_b, {})
        assert result.is_match is True
        assert result.match_pct == 96.0

    def test_address_only_no_match(self):
        row_a = {"name_norm": "merelex corporation", "tax_norm": "", "addr_norm": "10884 weyburn ave", "domain": "", "is_generic_domain": False}
        row_b = {"name_norm": "american elements", "tax_norm": "", "addr_norm": "10884 weyburn ave", "domain": "", "is_generic_domain": False}

        addr_counts = {"10884 weyburn ave": 2}
        result = evaluate_pair(row_a, row_b, addr_counts)
        assert result.is_match is False  # Address-only, no supporting evidence
        assert result.needs_review is True

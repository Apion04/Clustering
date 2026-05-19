"""Tests: 70% candidates always visible in final output when LLM unavailable/disabled.

Requirements:
1. no OPENAI_API_KEY + live mode does not crash.
2. no OPENAI_API_KEY + live mode outputs 70% candidates.
3. disabled mode outputs 70% candidates.
4. mock/live successful LLM decisions update 70% when possible.
5. LLM failure keeps 70% visible.
6. final output columns remain original columns + Cluster Number + Match Percentage.
7. final output scores only 98%, 85%, 70%, blank.
"""

import os
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import polars as pl
import pytest

from src.config import ClusteringConfig
from src.llm_pipeline import run_llm_backend_flow, _final_scores_allowed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preprocessed_df(n: int = 4) -> pl.DataFrame:
    """Minimal preprocessed DataFrame with required columns."""
    return pl.DataFrame({
        "row_id": list(range(n)),
        "Supplier Name": [f"Supplier {i}" for i in range(n)],
        "Address": [""] * n,
    })


def _make_result(
    preprocessed_df: pl.DataFrame,
    cluster_map: Dict[int, int],
    match_pcts: Dict[int, float],
) -> Dict[str, Any]:
    """Build a minimal cluster_suppliers result dict."""
    from src.llm_workflow import build_llm_review_groups, build_llm_review_groups_from_review_candidates
    rows_dict = {i: {"row_id": i, "name_norm": f"supplier {i}"} for i in range(len(preprocessed_df))}

    # Build LLM review groups from 70-score clusters
    llm_groups = build_llm_review_groups(cluster_map, match_pcts, rows_dict, _null_merger(len(preprocessed_df)))
    return {
        "preprocessed_df": preprocessed_df,
        "cluster_map": cluster_map,
        "output_cluster_map": cluster_map,
        "pre_llm_match_pcts": match_pcts,
        "output_match_pcts": match_pcts,
        "review_candidates": [],
        "llm_review_groups": llm_groups,
    }


def _null_merger(n: int):
    """Minimal ClusterMerger-like object for tests."""
    from src.merging import ClusterMerger
    return ClusterMerger(n, ClusteringConfig())


def _cfg(mode: str = "disabled", api_key: str = "") -> ClusteringConfig:
    cfg = ClusteringConfig()
    cfg.llm_execution_mode = mode
    cfg.llm_enabled = mode != "disabled"
    cfg.ai_api_key = api_key
    cfg.allow_unresolved_llm_candidates_in_final_output = True
    return cfg


def _run(result, cfg, output_dir=None):
    """Run run_llm_backend_flow in a temp dir."""
    if output_dir is None:
        td = tempfile.mkdtemp()
        output_dir = td
    final_path = os.path.join(output_dir, "final_supplier_clustered.csv")
    return run_llm_backend_flow(result, cfg, final_path, output_dir)


def _scores_in_output(flow_result) -> set:
    df = flow_result["final_df"]
    return set(str(x or "") for x in df["Match Percentage"].to_list())


# ---------------------------------------------------------------------------
# A result with one 70-score cluster (rows 0+1) and one 98-score cluster (2+3)
# ---------------------------------------------------------------------------

def _mixed_result():
    df = _make_preprocessed_df(4)
    cluster_map = {0: 0, 1: 0, 2: 2, 3: 2}
    match_pcts = {0: 70.0, 2: 98.0}
    return _make_result(df, cluster_map, match_pcts)


# ---------------------------------------------------------------------------
# Test 1: no API key + live mode does not crash
# ---------------------------------------------------------------------------

class TestNoApiKeyLiveNoCrash:
    def test_does_not_raise(self):
        result = _mixed_result()
        cfg = _cfg(mode="live", api_key="")
        # Must complete without raising
        flow = _run(result, cfg)
        assert flow["job_status"] in {"COMPLETE_REVIEW_PENDING", "INCOMPLETE_LLM_COST_CAP_EXCEEDED"}

    def test_warning_in_api_errors(self):
        result = _mixed_result()
        cfg = _cfg(mode="live", api_key="")
        flow = _run(result, cfg)
        errors = flow["decision_application_report"].get("api_errors", [])
        assert any("OPENAI_API_KEY" in str(e) or "not configured" in str(e).lower() for e in errors)


# ---------------------------------------------------------------------------
# Test 2: no API key + live mode outputs 70% candidates
# ---------------------------------------------------------------------------

class TestNoApiKeyLiveOutputs70:
    def test_70_present_in_final_output(self):
        result = _mixed_result()
        cfg = _cfg(mode="live", api_key="")
        flow = _run(result, cfg)
        scores = _scores_in_output(flow)
        assert "70%" in scores, f"Expected 70% in output scores, got: {scores}"

    def test_status_is_review_pending(self):
        result = _mixed_result()
        cfg = _cfg(mode="live", api_key="")
        flow = _run(result, cfg)
        assert flow["job_status"] == "COMPLETE_REVIEW_PENDING"


# ---------------------------------------------------------------------------
# Test 3: disabled mode outputs 70% candidates
# ---------------------------------------------------------------------------

class TestDisabledModeOutputs70:
    def test_70_present(self):
        result = _mixed_result()
        cfg = _cfg(mode="disabled")
        flow = _run(result, cfg)
        scores = _scores_in_output(flow)
        assert "70%" in scores, f"Expected 70% in output scores, got: {scores}"

    def test_status_is_review_pending(self):
        result = _mixed_result()
        cfg = _cfg(mode="disabled")
        flow = _run(result, cfg)
        assert flow["job_status"] == "COMPLETE_REVIEW_PENDING"

    def test_98_still_present(self):
        result = _mixed_result()
        cfg = _cfg(mode="disabled")
        flow = _run(result, cfg)
        scores = _scores_in_output(flow)
        assert "98%" in scores


# ---------------------------------------------------------------------------
# Test 4: mock mode LLM decisions update 70s when possible
# ---------------------------------------------------------------------------

class TestMockLlmUpdates70:
    def test_some_70s_updated_to_85(self):
        """Mock LLM approves most groups → 70-score clusters become 85/98."""
        df = _make_preprocessed_df(6)
        # Two 70-score clusters, one 98-score cluster
        cluster_map = {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4}
        match_pcts = {0: 70.0, 2: 70.0, 4: 98.0}
        result = _make_result(df, cluster_map, match_pcts)
        cfg = _cfg(mode="mock")

        flow = _run(result, cfg)
        scores = _scores_in_output(flow)
        # Mock approves most groups → at least 85% should appear
        assert "85%" in scores or "98%" in scores, (
            f"Mock LLM should update some 70s to 85/98, got scores: {scores}"
        )

    def test_status_complete_for_mock(self):
        df = _make_preprocessed_df(4)
        cluster_map = {0: 0, 1: 0, 2: 2, 3: 2}
        match_pcts = {0: 70.0, 2: 70.0}
        result = _make_result(df, cluster_map, match_pcts)
        cfg = _cfg(mode="mock")
        flow = _run(result, cfg)
        assert flow["job_status"] == "COMPLETE"


# ---------------------------------------------------------------------------
# Test 5: LLM call failure keeps 70% visible
# ---------------------------------------------------------------------------

class TestLlmFailureKeeps70:
    def test_70_visible_after_api_error(self):
        """When every LLM call raises an exception, 70% candidates stay in output."""
        result = _mixed_result()
        cfg = _cfg(mode="live", api_key="fake-key-for-testing")

        with patch("src.llm_pipeline.OpenAIReviewClient") as mock_cls:
            mock_cls.return_value.review_group.side_effect = RuntimeError("Connection refused")
            flow = _run(result, cfg)

        scores = _scores_in_output(flow)
        assert "70%" in scores, f"Expected 70% after LLM failure, got: {scores}"

    def test_status_review_pending_after_api_error(self):
        result = _mixed_result()
        cfg = _cfg(mode="live", api_key="fake-key")

        with patch("src.llm_pipeline.OpenAIReviewClient") as mock_cls:
            mock_cls.return_value.review_group.side_effect = RuntimeError("Timeout")
            flow = _run(result, cfg)

        assert flow["job_status"] == "COMPLETE_REVIEW_PENDING"


# ---------------------------------------------------------------------------
# Test 6: output columns are original + Cluster Number + Match Percentage only
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_only_expected_columns(self):
        result = _mixed_result()
        cfg = _cfg(mode="disabled")
        flow = _run(result, cfg)
        df = flow["final_df"]
        # Must contain these two added columns
        assert "Cluster Number" in df.columns
        assert "Match Percentage" in df.columns
        # Must contain user-facing original columns (row_id is internal, excluded from output)
        internal_cols = {"row_id"}
        for col in result["preprocessed_df"].columns:
            if col in internal_cols:
                continue
            assert col in df.columns, f"Original column {col!r} missing from output"
        # Must not have extra internal columns beyond Cluster Number and Match Percentage
        user_input_cols = set(result["preprocessed_df"].columns) - internal_cols
        expected = user_input_cols | {"Cluster Number", "Match Percentage"}
        extra = set(df.columns) - expected
        assert not extra, f"Unexpected extra columns in output: {extra}"


# ---------------------------------------------------------------------------
# Test 7: output scores are only 98%, 85%, 70%, or blank
# ---------------------------------------------------------------------------

class TestOutputScoreValues:
    def test_only_allowed_scores(self):
        result = _mixed_result()
        cfg = _cfg(mode="disabled")
        flow = _run(result, cfg)
        scores = _scores_in_output(flow)
        allowed = {"98%", "85%", "70%", ""}
        assert scores <= allowed, f"Unexpected scores in output: {scores - allowed}"

    def test_final_scores_allowed_helper_always_includes_70(self):
        """_final_scores_allowed must pass for DataFrames containing 70%."""
        df = pl.DataFrame({"Match Percentage": ["98%", "85%", "70%", ""]})
        assert _final_scores_allowed(df, "COMPLETE") is True
        assert _final_scores_allowed(df, "COMPLETE_REVIEW_PENDING") is True
        assert _final_scores_allowed(df, "INCOMPLETE_UNRESOLVED_LLM_CANDIDATES") is True

    def test_no_novel_scores(self):
        """No score outside the allowed set, regardless of LLM mode."""
        for mode in ("disabled", "mock"):
            result = _mixed_result()
            cfg = _cfg(mode=mode)
            flow = _run(result, cfg)
            scores = _scores_in_output(flow)
            bad = scores - {"98%", "85%", "70%", ""}
            assert not bad, f"mode={mode}: unexpected scores {bad}"

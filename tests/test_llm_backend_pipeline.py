"""Production LLM backend orchestration tests."""

import json
import os
import re
import subprocess
import sys

import polars as pl

from src.config import ClusteringConfig
from src.llm_pipeline import run_llm_backend_flow, validate_llm_responses
from src.openai_client import build_openai_request


def _fake_result() -> dict:
    df = pl.DataFrame({
        "row_id": list(range(8)),
        "Supplier Name": [
            "Acme GmbH",
            "ACME GmbH",
            "Eastman Company",
            "Eastman Chemical GmbH",
            "Eastmann Chemical BV",
            "Eastmann Chemical B.V.",
            "Other Review A",
            "Other Review B",
        ],
        "Address": [
            "Main Strasse 1",
            "Main Straße 1",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "Country": ["DE", "DE", "US", "DE", "NL", "NL", "US", "US"],
    })
    return {
        "preprocessed_df": df,
        "main_df": df.drop("row_id").with_columns([
            pl.Series("Cluster Number", [1, 1, 2, 2, None, None, None, None]),
            pl.Series("Match Percentage", ["98%", "98%", "85%", "85%", "", "", "", ""]),
        ]),
        "cluster_map": {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4},
        "output_cluster_map": {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4},
        "output_match_pcts": {0: 98.0, 2: 85.0, 4: 70.0},
        "pre_llm_match_pcts": {0: 98.0, 2: 85.0, 4: 70.0},
        "llm_review_groups": [
            {
                "candidate_group_id": "llm-1",
                "row_ids": [4, 5],
                "records": [
                    {"row_id": 4, "supplier_name": "Eastmann Chemical BV", "country": "NL"},
                    {"row_id": 5, "supplier_name": "Eastmann Chemical B.V.", "country": "NL"},
                ],
                "deterministic_pass_types": ["tax_exact"],
                "raw_scores": [70.0],
                "initial_user_score": 70,
                "reason_for_llm_review": "Typo supplier core requires LLM review",
            },
            {
                "candidate_group_id": "llm-2",
                "row_ids": [6, 7],
                "records": [
                    {"row_id": 6, "supplier_name": "Other Review A", "country": "US"},
                    {"row_id": 7, "supplier_name": "Other Review B", "country": "US"},
                ],
                "deterministic_pass_types": ["domain_review_candidate"],
                "raw_scores": [70.0],
                "initial_user_score": 70,
                "reason_for_llm_review": "Review candidate",
            },
        ],
    }


def test_config_defaults_to_gpt55_without_api_key_and_model_can_change(monkeypatch):
    for name in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "LLM_MODEL",
        "AI_MODEL",
        "LLM_PROVIDER",
        "AI_PROVIDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    cfg = ClusteringConfig.from_env()
    assert cfg.openai_model == "gpt-5.5"
    assert cfg.ai_api_key == ""

    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    cfg = ClusteringConfig.from_env()
    assert cfg.openai_model == "gpt-5.4"
    assert cfg.ai_model == "gpt-5.4"


def test_openai_request_uses_structured_json_schema():
    request = build_openai_request({
        "candidate_group_id": "llm-1",
        "row_ids": [1, 2],
        "records": [{"row_id": 1, "supplier_name": "A"}, {"row_id": 2, "supplier_name": "B"}],
    }, "gpt-5.5")
    assert request["model"] == "gpt-5.5"
    assert request["text"]["format"]["type"] == "json_schema"
    schema = request["text"]["format"]["schema"]
    assert "decision" in schema["properties"]
    assert set(schema["properties"]["decision"]["enum"]) >= {"approve", "reject", "split", "merge_with_existing", "promote_score", "uncertain"}


def test_mock_llm_backend_flow_produces_clean_complete_final_output(tmp_path):
    cfg = ClusteringConfig(llm_execution_mode="mock", openai_model="gpt-5.5")
    result = _fake_result()
    out = tmp_path / "final_supplier_clustered.csv"
    flow = run_llm_backend_flow(result, cfg, str(out), str(tmp_path))

    assert flow["job_status"] == "COMPLETE"
    final = pl.read_csv(out)
    assert final.columns == ["Supplier Name", "Address", "Country", "Cluster Number", "Match Percentage"]
    assert set(final["Match Percentage"].to_list()) <= {"98%", "85%", "70%", ""}
    assert flow["decision_application_report"]["cluster_numbers_contiguous"] is True
    assert os.path.exists(tmp_path / "llm_review_requests.jsonl")
    assert os.path.exists(tmp_path / "llm_review_responses.jsonl")
    assert os.path.exists(tmp_path / "llm_decision_application_report.json")


def test_disabled_llm_backend_is_complete_review_pending_with_70s_visible(tmp_path):
    cfg = ClusteringConfig(llm_execution_mode="disabled", openai_model="gpt-5.5")
    result = _fake_result()
    out = tmp_path / "final_supplier_clustered.csv"
    flow = run_llm_backend_flow(result, cfg, str(out), str(tmp_path))

    assert flow["job_status"] == "COMPLETE_REVIEW_PENDING"
    final = pl.read_csv(out)
    assert "70%" in final["Match Percentage"].to_list()
    exception = pl.read_csv(tmp_path / "unresolved_llm_exception_report.csv")
    assert exception.height == 2


def test_batch_mode_generates_batch_jsonl_and_manifest(tmp_path):
    cfg = ClusteringConfig(llm_execution_mode="batch", openai_model="gpt-5.4")
    result = _fake_result()
    out = tmp_path / "final_supplier_clustered.csv"
    flow = run_llm_backend_flow(result, cfg, str(out), str(tmp_path))

    assert flow["job_status"] == "INCOMPLETE_BATCH_PENDING"
    assert os.path.exists(tmp_path / "llm_batch_requests.jsonl")
    manifest = json.loads((tmp_path / "llm_batch_manifest.json").read_text())
    assert manifest["groups"] == 2


def test_cost_cap_prevents_live_execution(tmp_path):
    cfg = ClusteringConfig(
        llm_execution_mode="live",
        openai_model="gpt-5.5",
        ai_api_key="fake-key",
        max_total_llm_cost_per_job=0.0001,
        openai_input_cost_per_1m_tokens=1000.0,
        openai_output_cost_per_1m_tokens=1000.0,
    )
    result = _fake_result()
    out = tmp_path / "final_supplier_clustered.csv"
    flow = run_llm_backend_flow(result, cfg, str(out), str(tmp_path))

    assert flow["job_status"] == "INCOMPLETE_LLM_COST_CAP_EXCEEDED"
    assert flow["cost_estimate"]["allowed_to_run"] is False


def test_live_cost_cap_fails_closed_when_pricing_missing(tmp_path):
    cfg = ClusteringConfig(
        llm_execution_mode="live",
        openai_model="gpt-5.5",
        ai_api_key="fake-key",
        max_total_llm_cost_per_job=10.0,
        openai_input_cost_per_1m_tokens=0.0,
        openai_output_cost_per_1m_tokens=0.0,
    )
    result = _fake_result()
    out = tmp_path / "final_supplier_clustered.csv"
    flow = run_llm_backend_flow(result, cfg, str(out), str(tmp_path))

    assert flow["job_status"] == "INCOMPLETE_LLM_COST_CAP_EXCEEDED"
    assert flow["cost_estimate"]["pricing_missing"] is True
    assert flow["cost_estimate"]["estimated_cost_usd"] is None
    assert flow["cost_estimate"]["blocked_unknown_cost"] is True
    assert flow["cost_estimate"]["allowed_to_run"] is False


def test_unknown_llm_cost_can_be_explicitly_allowed(tmp_path):
    cfg = ClusteringConfig(
        llm_execution_mode="live",
        openai_model="gpt-5.5",
        max_total_llm_cost_per_job=10.0,
        openai_input_cost_per_1m_tokens=0.0,
        openai_output_cost_per_1m_tokens=0.0,
        allow_unknown_llm_cost=True,
    )
    result = _fake_result()
    out = tmp_path / "final_supplier_clustered.csv"
    flow = run_llm_backend_flow(result, cfg, str(out), str(tmp_path))

    assert flow["cost_estimate"]["pricing_missing"] is True
    assert flow["cost_estimate"]["allowed_to_run"] is True
    assert flow["job_status"] == "COMPLETE_REVIEW_PENDING"


def test_invalid_llm_responses_and_98_override_are_rejected():
    groups = {
        "bad": {"candidate_group_id": "bad", "row_ids": [2, 3], "deterministic_pass_types": []},
        "protected": {"candidate_group_id": "protected", "row_ids": [0, 1], "deterministic_pass_types": ["tax_exact"]},
    }
    responses = [
        {
            "candidate_group_id": "bad",
            "decision": "approve",
            "final_score": 90,
            "clusters": [],
            "rejected_row_ids": [],
            "target_cluster_number": None,
            "reasoning": "invalid score",
            "confidence": 0.9,
        },
        {
            "candidate_group_id": "protected",
            "decision": "reject",
            "final_score": None,
            "clusters": [],
            "rejected_row_ids": [0, 1],
            "target_cluster_number": None,
            "reasoning": "try to reject 98",
            "confidence": 0.9,
        },
    ]
    decisions, report = validate_llm_responses(responses, groups, {0}, {0, 1}, False)
    assert decisions == []
    reasons = {row["reason"] for row in report}
    assert "invalid final_score" in reasons
    assert "deterministic 98 cluster protected" in reasons


def test_merge_with_existing_accepts_zero_cluster_root():
    from src.llm_workflow import apply_llm_decisions

    final_map, final_scores, counts = apply_llm_decisions(
        4,
        {0: 0, 1: 0, 2: 2, 3: 2},
        {0: 85.0, 2: 70.0},
        [{"decision": "merge_with_existing", "target_cluster_number": 0, "row_ids": [2, 3], "match_percentage": 85}],
    )
    assert counts["merge_with_existing"] == 1
    assert counts["errors"] == 0
    assert final_map[0] == final_map[1] == final_map[2] == final_map[3]
    assert set(final_scores.values()) == {85.0}


def test_cli_openai_model_override_writes_final_output(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "Supplier Name,Address,Country\n"
        "Acme GmbH,Main Strasse 1,DE\n"
        "ACME GmbH,Main Straße 1,DE\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_cli",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--llm",
            "mock",
            "--openai-model",
            "gpt-5.4",
        ],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report = json.loads((output_dir / "llm_decision_application_report.json").read_text())
    assert report["selected_openai_model"] == "gpt-5.4"
    final = pl.read_csv(output_dir / "final_supplier_clustered.csv")
    assert final.columns == ["Supplier Name", "Address", "Country", "Cluster Number", "Match Percentage"]


def test_no_openai_api_key_is_hardcoded_in_source_or_scripts():
    for root in ["src", "scripts", "tests"]:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                text = open(os.path.join(dirpath, filename), encoding="utf-8").read()
                assert not re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", text)

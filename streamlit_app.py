"""Streamlit web UI for the Supplier Clustering Engine."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Read OPENAI_API_KEY from st.secrets first, then environment. Never shown in UI."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def _status_badge(job_status: str) -> None:
    if job_status in {"COMPLETE", "COMPLETE_REVIEW_PENDING"}:
        st.success(f"Job status: **{job_status}**")
        if job_status == "COMPLETE_REVIEW_PENDING":
            st.info(
                "70% candidates are shown in the output and were not resolved by LLM. "
                "Review them manually or re-run with an OpenAI key configured."
            )
    elif job_status == "FAILED":
        st.error(f"Job status: **{job_status}**")
    else:
        st.warning(f"Job status: **{job_status}**")


def _optional_download(path, label: str, filename: str, mime: str = "text/csv") -> None:
    if path and os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            st.download_button(label, fh.read(), file_name=filename, mime=mime)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Supplier Clustering Engine",
    page_icon="🔗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Supplier Clustering Engine")
st.caption(
    "Upload a supplier CSV or XLSX file to detect duplicates and cluster "
    "brand/group identities. Output adds only **Cluster Number** and "
    "**Match Percentage** to your original columns."
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Supplier file (CSV or XLSX)",
    type=["csv", "xlsx"],
)
if uploaded is not None:
    st.caption(f"Uploaded: {uploaded.name}  ·  {uploaded.size:,} bytes")

st.divider()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

st.subheader("Settings")

col_a, col_b = st.columns(2)
with col_a:
    llm_mode = st.selectbox(
        "LLM Mode",
        ["disabled", "mock", "live", "batch"],
        index=0,
        help=(
            "**disabled** – no LLM calls, fastest.\n\n"
            "**mock** – runs the full LLM backend path without an API key.\n\n"
            "**live** – real OpenAI API calls (requires OPENAI_API_KEY in secrets).\n\n"
            "**batch** – async OpenAI batch mode."
        ),
    )

with col_b:
    openai_model = st.text_input(
        "OpenAI Model",
        value=st.secrets.get("OPENAI_MODEL", "gpt-5.5") if True else "gpt-5.5",
        disabled=(llm_mode == "disabled"),
        help="e.g. gpt-5.5 or gpt-5.4. Ignored when LLM mode is disabled.",
    )

if llm_mode in ("live", "batch"):
    default_cost_cap = 0.0
    try:
        default_cost_cap = float(st.secrets.get("MAX_TOTAL_LLM_COST_PER_JOB", 0) or 0)
    except Exception:
        pass
    max_cost = st.number_input(
        "Max LLM Cost per Job (USD, 0 = no cap)",
        min_value=0.0,
        value=default_cost_cap,
        step=10.0,
        format="%.2f",
        help="Hard cost ceiling for live/batch LLM calls. 0 means no cap.",
    )
else:
    max_cost = 0.0

show_70_candidates = st.checkbox(
    "Show unresolved 70% candidates in output",
    value=True,
    help=(
        "70% means the engine found a plausible relationship but needs LLM or manual review to confirm. "
        "If no OpenAI key is configured, 70% candidates remain visible so they are not silently lost. "
        "Uncheck to suppress them from the download (they will still appear in review files)."
    ),
)
st.caption(
    "**Note:** 70% = LLM/manual review needed. "
    "If no OpenAI key is configured, 70% candidates are shown in the output and were not sent to LLM."
)

ignore_client_domains_text = st.text_input(
    "Client/Internal Domains to Ignore",
    value="",
    placeholder="e.g. merck.com; gilead.com; pfizer.com",
    help=(
        "Enter client/internal contact email domains that should **not** be used for supplier clustering. "
        "Separate multiple domains with semicolons.\n\n"
        "Use this when a domain (e.g. gilead.com) appears on many unrelated supplier rows because "
        "it belongs to your organisation, not the supplier.\n\n"
        "Free email domains (gmail.com, outlook.com, etc.) are always ignored automatically."
    ),
)

st.divider()

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

run_clicked = st.button(
    "Run Clustering",
    type="primary",
    disabled=(uploaded is None),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Clustering run
# ---------------------------------------------------------------------------

if run_clicked and uploaded is not None:
    # Per-run temp directory — never persists uploaded client data between runs
    run_dir = tempfile.mkdtemp(prefix="supplier_run_")
    input_path = os.path.join(run_dir, uploaded.name)
    output_dir = os.path.join(run_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "wb") as fh:
        fh.write(uploaded.getvalue())

    # Build CLI command
    cmd = [
        sys.executable, "-m", "scripts.run_cli",
        "--input", input_path,
        "--output", output_dir,
        "--llm", llm_mode,
    ]
    if llm_mode != "disabled":
        cmd += ["--openai-model", openai_model]
    if max_cost > 0:
        cmd += ["--max-total-llm-cost-per-job", str(max_cost)]
    if ignore_client_domains_text.strip():
        cmd += ["--ignore-client-domains", ignore_client_domains_text.strip()]
    if show_70_candidates:
        cmd += ["--allow-unresolved-llm-candidates-in-final-output"]

    review_pairs_path = os.path.join(output_dir, "review_pairs.csv")
    cluster_audit_path = os.path.join(output_dir, "cluster_audit.csv")
    cmd += ["--review-pairs-output", review_pairs_path]
    cmd += ["--cluster-audit-output", cluster_audit_path]

    # Inject API key via env — never passed as CLI arg or shown in UI
    env = os.environ.copy()
    api_key = _get_api_key()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    elif llm_mode in ("live", "batch"):
        st.warning(
            "OpenAI key not configured. 70% review candidates are shown in the output "
            "and were not sent to LLM."
        )

    # Propagate other LLM config from secrets when present
    for secret_key in (
        "LLM_ENABLED",
        "LLM_SEND_SCOPE",
        "ALLOW_UNRESOLVED_LLM_CANDIDATES_IN_FINAL_OUTPUT",
        "OPENAI_INPUT_COST_PER_1M_TOKENS",
        "OPENAI_OUTPUT_COST_PER_1M_TOKENS",
        "OVERRIDE_LLM_CAN_MODIFY_98",
    ):
        try:
            val = st.secrets.get(secret_key, "")
            if val:
                env[secret_key] = str(val)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Streaming progress
    # -----------------------------------------------------------------------
    log_lines = []
    job_status = "FAILED"
    returncode = -1

    with st.status("Running clustering...", expanded=True) as status_box:
        st.write(f"File uploaded: **{uploaded.name}**")
        st.write("Deterministic clustering running...")

        log_placeholder = st.empty()

        try:
            with subprocess.Popen(
                cmd,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    log_lines.append(line)
                    log_placeholder.code("\n".join(log_lines[-10:]), language=None)

                    if "Deterministic output saved" in line:
                        st.write("Deterministic clustering complete.")
                    elif "Review candidates saved" in line:
                        st.write("Review candidates generated.")
                    elif "Audit file saved" in line:
                        st.write("Audit file generated.")
                    elif "llm_queue_breakdown" in line and "saved" in line.lower():
                        st.write("LLM queue prepared.")
                    elif "LLM backend job status" in line:
                        job_status = line.split(":", 1)[-1].strip()
                        if llm_mode == "disabled":
                            st.write("LLM review: skipped (disabled).")
                        else:
                            st.write(f"LLM review complete. Status: **{job_status}**")
                    elif "Final supplier output saved" in line:
                        st.write("Final output generated.")

                proc.wait()
                returncode = proc.returncode

        except Exception as exc:
            st.error(f"Subprocess error: {exc}")
            returncode = -1

        # Final job_status pass (re-parse log in case line order was unexpected)
        for line in reversed(log_lines):
            if "LLM backend job status:" in line:
                job_status = line.split(":", 1)[-1].strip()
                break

        if returncode == 0:
            status_box.update(label="Clustering complete", state="complete", expanded=False)
        else:
            status_box.update(label="Clustering failed", state="error", expanded=True)

    # -----------------------------------------------------------------------
    # Store result paths in session state
    # -----------------------------------------------------------------------
    if returncode == 0:
        st.session_state["run_result"] = {
            "job_status": job_status,
            "final_csv": os.path.join(output_dir, "final_supplier_clustered.csv"),
            "readiness_md": os.path.join(output_dir, "final_production_readiness_report.md"),
            "queue_csv": os.path.join(output_dir, "llm_queue_breakdown.csv"),
            "unresolved_csv": os.path.join(output_dir, "unresolved_llm_exception_report.csv"),
            "review_pairs_csv": review_pairs_path,
            "cluster_audit_csv": cluster_audit_path,
            "log": "\n".join(log_lines),
        }
    else:
        st.error("Clustering failed. Expand the log above for details.")
        with st.expander("Full error log", expanded=True):
            st.code("\n".join(log_lines), language=None)
        st.session_state["run_result"] = None

# ---------------------------------------------------------------------------
# Results section (persists across reruns via session_state)
# ---------------------------------------------------------------------------

if st.session_state.get("run_result"):
    result = st.session_state["run_result"]

    st.divider()
    st.subheader("Results")

    _status_badge(result["job_status"])

    final_csv = result.get("final_csv", "")
    if final_csv and os.path.isfile(final_csv):
        with open(final_csv, "rb") as fh:
            st.download_button(
                "Download  final_supplier_clustered.csv",
                fh.read(),
                file_name="final_supplier_clustered.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
    else:
        st.warning("final_supplier_clustered.csv was not produced.")

    _optional_download(
        result.get("review_pairs_csv"),
        "📋 Download Review Pairs (CSV)",
        "review_pairs.csv",
    )
    _optional_download(
        result.get("cluster_audit_csv"),
        "🔍 Download Cluster Audit (CSV)",
        "cluster_audit.csv",
    )

    with st.expander("Internal / debug files", expanded=False):
        _optional_download(
            result.get("readiness_md"),
            "Download  final_production_readiness_report.md",
            "final_production_readiness_report.md",
            mime="text/markdown",
        )
        _optional_download(
            result.get("queue_csv"),
            "Download  llm_queue_breakdown.csv",
            "llm_queue_breakdown.csv",
        )
        _optional_download(
            result.get("unresolved_csv"),
            "Download  unresolved_llm_exception_report.csv",
            "unresolved_llm_exception_report.csv",
        )
        if not any([
            result.get("readiness_md") and os.path.isfile(result["readiness_md"]),
            result.get("queue_csv") and os.path.isfile(result["queue_csv"]),
            result.get("unresolved_csv") and os.path.isfile(result["unresolved_csv"]),
        ]):
            st.caption("No internal files available for this run.")

    with st.expander("Processing log", expanded=False):
        st.code(result.get("log", ""), language=None)

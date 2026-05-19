# Deploying to Streamlit Community Cloud

## Prerequisites

- A GitHub account
- A Streamlit Community Cloud account at [share.streamlit.io](https://share.streamlit.io)
- An OpenAI API key (only required for `live` or `batch` LLM mode)

---

## Step 1 – Push the repo to GitHub

```bash
cd supplier-clustering-web-tool
git init
git add .
git commit -m "Initial commit: supplier clustering engine + Streamlit UI"
git remote add origin https://github.com/<your-org>/<your-repo>.git
git push -u origin main
```

Do **not** commit `.streamlit/secrets.toml` or any `.env` file containing real API keys.
The `.gitignore` already excludes them.

---

## Step 2 – Connect to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **New app**.
3. Select your GitHub repository and branch.
4. Set the **Main file path** to `streamlit_app.py`.
5. Click **Deploy**.

---

## Step 3 – Add secrets

In the Streamlit Cloud app settings, open **Secrets** and paste the contents of
`.streamlit/secrets.toml.example`, filling in your real values:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.5"
LLM_ENABLED = "true"
LLM_SEND_SCOPE = "all_review_candidates"
ALLOW_UNRESOLVED_LLM_CANDIDATES_IN_FINAL_OUTPUT = "false"
MAX_TOTAL_LLM_COST_PER_JOB = "250"
OPENAI_INPUT_COST_PER_1M_TOKENS = ""
OPENAI_OUTPUT_COST_PER_1M_TOKENS = ""
OVERRIDE_LLM_CAN_MODIFY_98 = "false"
```

`OPENAI_API_KEY` is read by the app via `st.secrets` and injected into the
subprocess environment. It is **never** shown in the UI or passed as a CLI
argument.

If you want to run with LLM disabled (the default), you can leave
`OPENAI_API_KEY` blank and set `LLM_ENABLED = "false"`.

---

## Step 4 – Get your URL

After deployment, Streamlit Community Cloud assigns a public URL:

```
https://<your-org>-<your-repo>-streamlit-app-<hash>.streamlit.app
```

This URL appears on the app page in Streamlit Cloud. Share it with your team.

---

## Running locally

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Set up secrets for local dev
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your real values

# Run
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

To run without LLM (no API key needed):

```bash
streamlit run streamlit_app.py
# Select LLM Mode: disabled in the UI
```

---

## Security notes

- `OPENAI_API_KEY` is read from `st.secrets` (Streamlit Cloud) or
  `OPENAI_API_KEY` environment variable (local). It is never hardcoded, never
  shown in the UI, and never passed as a command-line argument.
- Uploaded files are written to a per-run `tempfile.mkdtemp()` directory and
  are not stored permanently. Streamlit Cloud containers are ephemeral.
- Do not commit `.streamlit/secrets.toml` to version control.
  The `.gitignore` excludes it.

---

## Updating the deployment

Push new commits to the same branch. Streamlit Community Cloud auto-redeploys
on push within a few minutes.

---

## Docker (alternative)

```bash
docker-compose up --build
# App available at http://localhost:8501
```

For Docker, set `OPENAI_API_KEY` in `.env` (excluded from `.gitignore`):

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

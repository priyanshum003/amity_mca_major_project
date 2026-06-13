# Retention Marketing Using Explainable Machine Learning for Churn Prediction

An MCA major project: a customer-churn prediction system that compares a rule-based
baseline against three machine-learning models, explains **every** prediction with
SHAP, and is served as a small web app for retention-marketing teams.

The project has three parts:

1. **Offline ML core** (`src/`) — cleans the data, trains four approaches, evaluates
   them, and saves the best model + report charts. Run once; produces `outputs/`.
2. **Backend API** (`backend/`) — a FastAPI service that loads the saved model and does
   **inference only**: score a customer, explain why, recommend a retention action,
   and score a whole uploaded CSV.
3. **Frontend** (`frontend/`) — a React + Tailwind + Recharts app with three screens:
   Dashboard, Predict-a-customer, and Batch upload.

---

## Problem
Acquiring a customer costs far more than keeping one. This project predicts which
customers are likely to churn, explains *why* (per customer), and recommends a
retention action — so marketing can act before the customer leaves.

## Dataset
Telco Customer Churn (7,043 customers, 21 columns). Public, CPU-friendly.
Lives at `data/telco_churn.csv`.

---

## Live demo
> Deploy with the steps in **Deployment** below, then paste your public URLs here so
> your examiner can open them directly:
>
> - **Frontend (app):** `https://<your-frontend>.onrender.com`
> - **Backend (API):**  `https://<your-backend>.onrender.com`  (try `/health`, `/metrics`)

---

## Quick start (local)

### 0. One-time: Python virtual environment
```bash
cd churn_project          # the folder that contains data/ src/ backend/ frontend/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # full set (training + backend)
```
`requirements.txt` covers everything. The backend alone needs only
`backend/requirements.txt` (used by Docker).

### 1. (Re)train the model — offline, optional
The repo already ships a trained `outputs/model.joblib`. Re-run only if you want the
numbers to be reproducibly yours (a different machine/seed gives slightly different
values — that is expected and fine to state in the report):
```bash
python src/train.py
```
This writes everything to `outputs/` (see "What training produces" below).

### 2. Run the backend (FastAPI)
From the project root:
```bash
uvicorn backend.app.main:app --reload --port 8000
```
- API docs (interactive): http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Run the frontend (React)
In a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. In dev, the frontend proxies `/api/*` to the backend on
port 8000 (see `frontend/vite.config.js`), so no extra config is needed.

### Run the backend with Docker instead
```bash
docker compose up --build        # API on http://localhost:8000
```

---

## API endpoints

| Method | Path             | Purpose |
|--------|------------------|---------|
| GET    | `/health`        | Liveness check: `{"status":"ok","model_loaded":true,"gemini_available":false}`. |
| GET    | `/metrics`       | The model-comparison table + summary (churn rate, best model, top drivers) as JSON. Powers the Dashboard. |
| POST   | `/predict`       | Score **one** customer. Returns `churn_probability`, `risk_level` (Low/Medium/High), `top_factors` (SHAP), and a `recommended_action`. |
| POST   | `/insight`       | Score the customer **and** add an LLM explanation. Returns the ML `churn_probability` + `risk_level` + `top_factors`, plus an `insight` object (`why_at_risk`, `retention_actions`, `outreach_message`, `source`). See **LLM insight layer** below. |
| POST   | `/predict-batch` | Upload a customer CSV; returns the same CSV with `churn_probability`, `risk_level`, `top_factor` columns added. |
| GET    | `/outputs/<png>` | Serves the report charts from `outputs/` (used by the Dashboard). |

### `/predict` example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":1,"Contract":"Month-to-month","MonthlyCharges":99,"TotalCharges":99,
       "InternetService":"Fiber optic","TechSupport":"No","OnlineSecurity":"No",
       "PaymentMethod":"Electronic check","PaperlessBilling":"Yes"}'
```
You only need to send the fields you have — any missing column is filled with a
sensible default (`backend/app/schema.py`) so the preprocessor never fails.

### `/predict-batch` example
```bash
curl -X POST http://localhost:8000/predict-batch \
  -F "file=@data/telco_churn.csv" -o scored_customers.csv
```

---

## LLM insight layer (Gemini) — and the ML-vs-LLM split

The `/insight` endpoint adds a plain-English layer on top of the prediction using
Google **Gemini** (`gemini-2.5-flash`). The single rule that matters for the viva:

> **The churn probability ALWAYS comes from the ML model. Gemini never computes,
> changes, or invents the number — it only turns the model's output + SHAP factors
> into words** (a "why at risk" explanation, 2–3 tailored retention actions, and a
> draft outreach message).

How it works (`backend/app/insight.py`):
1. `/insight` first scores the customer with the **ML model** (so the number is the
   model's, and can't be spoofed by the client).
2. That fixed prediction + the SHAP factors are passed to Gemini in a prompt that
   explicitly forbids producing or altering any number.
3. Gemini returns JSON; we show it in the "AI insight" card on the Predict screen.

### Setup (local)
```bash
cp backend/.env.example backend/.env
# edit backend/.env and paste your key:
# GEMINI_API_KEY=AIza...      (get one free at https://aistudio.google.com/apikey)
```
Restart the backend. `GET /health` will then report `"gemini_available": true`.

### Reliability — the demo never breaks
If `GEMINI_API_KEY` is missing, or the Gemini call errors / times out (~8s), the
endpoint **falls back** to a deterministic explanation built from the SHAP factors +
the rule-based action, tagged `"source": "fallback"`. So `/insight` always returns a
useful answer, key or no key. The UI shows a badge telling you which path ran.

### Security (the repo is public)
- The key is read from a **backend env var only** (`backend/.env` locally, a Render
  env var in production). It is **never** in the frontend or in committed code.
- `backend/.env` is git-ignored (see `.gitignore`); only `backend/.env.example`
  (no real key) is committed.
- The frontend calls **our** backend; only the backend calls Gemini.

## How the explanation works (viva-ready)
For each prediction we ask SHAP "how much did each feature push this customer toward
churn?". The explainer is chosen to match the model:

- **Logistic Regression** → `shap.LinearExplainer` (the current best model).
- **Random Forest / XGBoost** → `shap.TreeExplainer`.
- anything else → the model-agnostic `shap.Explainer` (fallback).

We keep only the features that *raise* churn risk, sort them by contribution, map the
encoded names back to readable labels (`cat__Contract_Month-to-month` → "Contract:
Month-to-month"), and for categorical features only report the option the customer
actually has. The **top factor** is then mapped to a concrete retention action via a
small playbook (`RETENTION_ACTIONS` in `backend/app/schema.py`).

---

## What training produces (`outputs/`) — your Chapter 4 material
- `model_comparison.csv` / `model_comparison.png` — metrics for all four approaches
- `roc_curves.png`, `pr_curves.png` — ranking-quality curves
- `confusion_matrix.png` — for the best model
- `shap_summary.png`, `shap_bar.png` — why the model flags churn (global view)
- `model.joblib` — the saved best model + preprocessor (loaded by the backend)
- `summary.json` — best model, churn rate, top churn drivers

### Results (from the shipped run)
- 7,043 customers, churn rate 26.5%.
- Best model: **Logistic Regression** (F1 ≈ 0.61), beating the rule-based baseline.
- Top churn drivers: month-to-month contract, low tenure, high monthly charges,
  no online security / tech support, electronic-check payment.

---

## Deployment (free, CPU-only, PUBLIC)

The repo includes a **Render Blueprint** (`render.yaml`) that deploys *both* the API
and the frontend as public services. The examiner only needs the frontend URL.

1. Push this project to a **GitHub** repo (it is not yet a git repo):
   ```bash
   git init && git add . && git commit -m "Churn prediction app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. In the [Render dashboard](https://dashboard.render.com): **New → Blueprint**, select
   your repo. Render reads `render.yaml` and creates **churn-api** (Docker) and
   **churn-frontend** (static site). Both are **public** by default — no login wall.
3. The frontend's `VITE_API_URL` is auto-wired to the backend host by the blueprint;
   `frontend/src/api.js` adds the `https://` scheme.
4. **(Optional) Enable Gemini in production:** on the **churn-api** service in Render,
   add an environment variable `GEMINI_API_KEY` with your key (Dashboard → the service
   → Environment). Do **not** put the key in `render.yaml` — it would be committed.
   Without it, `/insight` still works via the fallback.
5. Copy the two public URLs into the **Live demo** section above.

> Notes for the free tier: the backend may cold-start (sleep after inactivity) and take
> ~30–60s to wake on the first request — open `/health` once before your demo.
> Alternatively the frontend can go on **Vercel** (`Root Directory = frontend`,
> build `npm run build`, output `dist`, set `VITE_API_URL` to the backend URL).

---

## Screenshots to capture for my report
A tidy checklist — grab these once both servers are running:

**Dashboard screen**
- [ ] Headline stats (customers analysed, churn rate, best model)
- [ ] Model-comparison bar chart (all four approaches)
- [ ] Top churn drivers chart
- [ ] The embedded report charts (confusion matrix, ROC, PR, SHAP bar)

**Predict-a-customer screen**
- [ ] A **High**-risk customer: form + probability + red badge + top factors + action
- [ ] A **Low**-risk customer: form + probability + green badge
- [ ] Close-up of the "Top factors" list and the "Recommended retention action" box

**Batch screen**
- [ ] The upload control with a file chosen
- [ ] The scored preview table (note the added probability / risk / top-factor columns)
- [ ] The "Download scored CSV" button (and a peek at the downloaded file)

**API (optional, good for the methodology chapter)**
- [ ] http://localhost:8000/docs (the auto-generated Swagger UI)
- [ ] A `/predict` response JSON

**Training outputs** — the PNGs in `outputs/` can be dropped straight into the report.

---

## Files to read to understand the project (for the viva)
Read them in this order — it mirrors the data flow end to end:

1. `src/data.py` — load, clean, and the preprocessing pipeline (scaling + one-hot).
2. `src/baseline.py` — the rule-based non-ML benchmark.
3. `src/train.py` — trains the four approaches, evaluates, makes charts/SHAP, saves the model.
4. `backend/app/schema.py` — the customer fields, defaults, readable labels, and the
   retention-action playbook. **The most "explainable" file — start here for the app.**
5. `backend/app/model.py` — loads the artifact and does scoring + per-prediction SHAP.
6. `backend/app/insight.py` — the **LLM layer**: builds the Gemini prompt from the ML
   output and returns words only (with a SHAP/rule fallback). **Start here to explain
   the ML-vs-LLM split.**
7. `backend/app/main.py` — the FastAPI endpoints (`/predict`, `/insight`, …).
8. `frontend/src/api.js` — how the UI calls the API.
9. `frontend/src/pages/Predict.jsx` — the customer form, presets, reactive prediction,
   and the AI-insight panel.
10. `frontend/src/pages/Dashboard.jsx` / `Batch.jsx` — the other two screens.

---

## Project structure
```
churn_project/
├── data/telco_churn.csv        # the dataset
├── src/                        # OFFLINE ML core (run once)
│   ├── data.py                 # load + clean + preprocessing pipeline
│   ├── baseline.py             # rule-based (non-ML) benchmark
│   └── train.py                # train, evaluate, charts, SHAP, save model
├── outputs/                    # generated: charts, comparison table, model.joblib, summary.json
├── backend/                    # FastAPI inference service
│   ├── app/
│   │   ├── schema.py           # fields, defaults, labels, retention playbook
│   │   ├── model.py            # load model + predict + SHAP explanations
│   │   ├── insight.py          # Gemini LLM layer (words only) + SHAP/rule fallback
│   │   └── main.py             # /health /metrics /predict /insight /predict-batch
│   ├── requirements.txt        # backend-only deps
│   ├── .env.example            # GEMINI_API_KEY template (copy to .env, never commit .env)
│   └── Dockerfile
├── frontend/                   # React + Vite + Tailwind + Recharts
│   ├── src/
│   │   ├── pages/              # Dashboard.jsx, Predict.jsx, Batch.jsx
│   │   ├── components/         # Card.jsx, RiskBadge.jsx
│   │   ├── api.js, labels.js, App.jsx, main.jsx
│   └── package.json
├── docker-compose.yml          # local: brings up the API
├── render.yaml                 # free public deploy (API + frontend)
└── requirements.txt            # full deps (training + backend)
```

## Note on academic integrity
Metrics here are generated by the code, not hand-written. Re-run `python src/train.py`
on your own machine so the reported numbers are reproducibly yours (a different random
seed gives slightly different values — that is expected and fine to state in the report).

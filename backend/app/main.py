"""
main.py
-------
The FastAPI service. It loads the trained model ONCE at startup and serves
inference only:

    GET  /health         -> liveness check
    GET  /metrics        -> model_comparison.csv + summary.json as JSON
    POST /predict        -> score one customer (probability, risk, factors, action)
    POST /predict-batch  -> score an uploaded CSV, return a scored CSV download

Run from the project root:
    uvicorn backend.app.main:app --reload
"""

import io
import json
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load backend/.env (if present) so GEMINI_API_KEY is available to the LLM layer.
# In production (Render) the key is set as a real environment variable instead.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from .insight import gemini_available, generate_insight
from .model import ChurnModel, PROJECT_ROOT
from .schema import CustomerIn

app = FastAPI(
    title="Churn Prediction API",
    description="Explainable customer-churn scoring for retention marketing.",
    version="1.0.0",
)

# Allow the frontend (any origin in this academic demo) to call the API.
# In production you would restrict this to your deployed frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at import time. If the artifact is missing we still let
# the app start so /health works and the error is reported clearly per request.
try:
    MODEL = ChurnModel()
except Exception as exc:  # pragma: no cover - defensive startup guard
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


def _require_model() -> ChurnModel:
    """Return the loaded model or raise a clear 503 if it failed to load."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {MODEL_LOAD_ERROR}. Run `python src/train.py` first.",
        )
    return MODEL


# Serve the offline-generated report charts (PNGs) so the dashboard can embed
# them directly. They live in the project's outputs/ folder.
_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
if os.path.isdir(_OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=_OUTPUTS_DIR), name="outputs")


@app.get("/health")
def health():
    """Simple liveness probe used by Docker/Render and the frontend."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        # Tells the frontend whether to expect real Gemini insights or fallbacks.
        "gemini_available": gemini_available(),
    }


@app.get("/metrics")
def metrics():
    """
    Serve the offline evaluation artifacts (produced by src/train.py) as JSON so
    the dashboard can render them: the model-comparison table and the summary
    (churn rate, best model, top drivers).
    """
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    comparison_path = os.path.join(out_dir, "model_comparison.csv")
    summary_path = os.path.join(out_dir, "summary.json")

    if not (os.path.exists(comparison_path) and os.path.exists(summary_path)):
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run `python src/train.py` to generate outputs/.",
        )

    comparison = pd.read_csv(comparison_path).to_dict(orient="records")
    with open(summary_path) as f:
        summary = json.load(f)

    return {"summary": summary, "model_comparison": comparison}


@app.post("/predict")
def predict(customer: CustomerIn):
    """
    Score one customer. The request may contain only the fields the form
    collects; everything else is filled with sensible defaults before scoring.
    """
    model = _require_model()
    return model.predict_one(customer.model_dump())


@app.post("/insight")
def insight(customer: CustomerIn):
    """
    LLM insight layer. We FIRST score the customer with the ML model (so the
    probability is always the model's, never the LLM's), then ask Gemini to
    explain that fixed result in plain English and suggest retention actions.

    Re-scoring here (instead of trusting numbers sent by the client) guarantees
    the probability cannot be spoofed: the number always originates from the
    model. If Gemini is unavailable, generate_insight() returns a deterministic
    SHAP/rule-based fallback, so this endpoint never fails the demo.
    """
    model = _require_model()
    profile = customer.model_dump(exclude_none=True)   # only the fields the user set
    prediction = model.predict_one(customer.model_dump())  # authoritative ML output

    return {
        # Echo the ML numbers so the UI can show them next to the AI text.
        "churn_probability": prediction["churn_probability"],
        "risk_level": prediction["risk_level"],
        "top_factors": prediction["top_factors"],
        # Gemini's words only (or the fallback).
        "insight": generate_insight(profile, prediction),
    }


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Score an uploaded CSV of customers and stream back the same rows with
    churn_probability, risk_level and top_factor columns appended.
    """
    model = _require_model()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV has no rows.")

    scored = model.predict_frame(df)

    buffer = io.StringIO()
    scored.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored_customers.csv"},
    )

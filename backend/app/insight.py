"""
insight.py
----------
The LLM layer. It takes a churn prediction that was ALREADY produced by the
machine-learning model and asks Google Gemini to explain it in plain English and
suggest retention actions.

The single most important rule (and the thing to stress in a viva):

    ┌───────────────────────────────────────────────────────────────────────┐
    │  The churn PROBABILITY always comes from the ML model.                  │
    │  Gemini NEVER computes, changes, or invents the number. It only turns   │
    │  the model's output + SHAP factors into words.                          │
    └───────────────────────────────────────────────────────────────────────┘

So the flow is: ML model -> {probability, risk, SHAP factors} -> Gemini -> words.
Gemini is given the numbers as fixed inputs and is explicitly told not to alter them.

Reliability: if GEMINI_API_KEY is missing, or the API errors / times out, we fall
back to a deterministic explanation built from the SHAP factors + the rule-based
retention action. The demo therefore never breaks, with or without a key.
"""

import json
import os

# google-genai is the current unified Google Gen AI SDK (`from google import genai`).
from google import genai
from google.genai import types

from .schema import retention_action

# A current, fast, cheap Gemini model. Confirmed against the Gemini API docs.
GEMINI_MODEL = "gemini-2.5-flash"

# Keep the call short so a slow LLM never stalls the demo; on timeout we fall back.
TIMEOUT_MS = 8000


def gemini_available() -> bool:
    """True if a key is configured. Used by /health and to short-circuit fast."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def _build_prompt(profile: dict, prediction: dict) -> str:
    """
    Build the instruction for Gemini. We hand it the ML numbers as FIXED facts and
    forbid it from producing or changing any number. It returns strict JSON.
    """
    prob_pct = round(prediction["churn_probability"] * 100, 1)
    factors = ", ".join(
        f"{f['label']}" + (f" (value: {f['value']})" if f.get("value") is not None else "")
        for f in prediction["top_factors"]
    ) or "no strong churn factors"

    # A compact, readable profile (only the human-meaningful fields).
    profile_lines = "\n".join(f"  - {k}: {v}" for k, v in profile.items())

    return f"""You are a retention-marketing analyst for a telecom company.

A machine-learning model has ALREADY scored this customer. These numbers are FIXED
inputs — do NOT recalculate, change, or invent any probability, percentage, or
statistic. Use ONLY what is given below.

ML MODEL OUTPUT (authoritative — do not alter):
  - Churn probability: {prob_pct}% (risk level: {prediction['risk_level']})
  - Top factors driving this risk (from SHAP): {factors}

CUSTOMER PROFILE:
{profile_lines}

Write a brief, practical analysis for the retention team. Respond with ONLY a JSON
object (no markdown, no code fences) with exactly these keys:
  "why_at_risk":       a 2-3 sentence plain-English explanation of why this customer
                       is at this risk level, grounded in the factors above.
  "retention_actions": an array of 2-3 short, concrete, tailored actions.
  "outreach_message":  a short, friendly draft message (2-4 sentences) the team could
                       send this customer. Do not mention the churn probability number.

Do not include any churn percentage or invented numbers in your text."""


def _fallback(prediction: dict) -> dict:
    """
    Deterministic, no-LLM insight built from the SHAP factors + rule-based action.
    Returned when Gemini is unavailable so the product still works end to end.
    """
    factors = prediction.get("top_factors", [])
    if factors:
        names = ", ".join(f["label"] for f in factors[:3])
        why = (
            f"This customer is {prediction['risk_level'].lower()} risk, driven mainly by: "
            f"{names}. These are the features the model weighted most for this prediction."
        )
        top_feature = factors[0]["feature"]
    else:
        why = f"This customer is {prediction['risk_level'].lower()} risk; no single factor dominates."
        top_feature = ""

    return {
        "source": "fallback",
        "why_at_risk": why,
        "retention_actions": [retention_action(top_feature)],
        "outreach_message": None,
    }


def generate_insight(profile: dict, prediction: dict) -> dict:
    """
    Ask Gemini to explain the (fixed) ML prediction. On any problem — no key, API
    error, timeout, or unparseable response — return the deterministic fallback.

    `prediction` is the dict from ChurnModel.predict_one (probability, risk_level,
    top_factors, recommended_action). We never let Gemini touch the probability.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback(prediction)

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_prompt(profile, prediction),
            config=types.GenerateContentConfig(
                temperature=0.4,                       # a little variety, mostly grounded
                response_mime_type="application/json", # ask for clean JSON back
                # Disable "thinking" tokens for a fast, cheap demo response.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        data = json.loads(response.text)

        # Keep only the fields we expect, and tag the source as the real LLM.
        return {
            "source": "gemini",
            "why_at_risk": str(data.get("why_at_risk", "")).strip(),
            "retention_actions": [str(a).strip() for a in data.get("retention_actions", [])][:3],
            "outreach_message": (str(data["outreach_message"]).strip()
                                 if data.get("outreach_message") else None),
        }
    except Exception as exc:
        # Any failure (network, quota, bad JSON, timeout) -> safe fallback.
        fb = _fallback(prediction)
        fb["error"] = f"Gemini unavailable, used fallback: {type(exc).__name__}"
        return fb

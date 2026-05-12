"""
model.py
--------
Loads the trained artifact (outputs/model.joblib) ONCE and exposes two things:

    predict_one(record)   -> probability, risk level, top factors, action
    predict_frame(df)     -> the same scoring applied to a whole DataFrame

The artifact is a dict saved by src/train.py:
    {"preprocessor", "model", "best_name", "features"}

Per-prediction explanations use SHAP, choosing the explainer that matches the
model type:
    - LogisticRegression  -> shap.LinearExplainer
    - RandomForest/XGBoost -> shap.TreeExplainer
    - anything else        -> the model-agnostic shap.Explainer (fallback)

All of this is INFERENCE only — we never retrain here. Retraining lives in
src/train.py and is run offline.
"""

import os

import joblib
import numpy as np
import pandas as pd
import shap

from .schema import (
    ALL_COLS,
    fill_defaults,
    readable_label,
    retention_action,
    split_feature,
)

# Paths are resolved relative to the project root (the folder that holds
# outputs/ and data/), regardless of where uvicorn is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telco_churn.csv")

# How many background rows to give SHAP's LinearExplainer. A small sample is
# plenty and keeps startup fast on CPU.
_BACKGROUND_SIZE = 100


def risk_level(prob: float) -> str:
    """Bucket a churn probability into Low / Medium / High (the brief's cut-offs)."""
    if prob < 0.33:
        return "Low"
    if prob > 0.66:
        return "High"
    return "Medium"


class ChurnModel:
    """Wraps the trained preprocessor + model + SHAP explainer for serving."""

    def __init__(self, model_path: str = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.preprocessor = bundle["preprocessor"]
        self.model = bundle["model"]
        self.best_name = bundle["best_name"]
        self.features = list(bundle["features"])  # encoded feature names
        self.explainer = self._build_explainer()

    # -- setup -------------------------------------------------------------
    def _background(self) -> np.ndarray:
        """
        Build a small transformed background sample for SHAP. LinearExplainer
        needs a reference distribution to measure each feature's contribution
        against; we sample it from the real training data.
        """
        df = pd.read_csv(DATA_PATH)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        X = df[ALL_COLS].head(self._cap_background(len(df)))
        return self.preprocessor.transform(X)

    @staticmethod
    def _cap_background(n_rows: int) -> int:
        return min(_BACKGROUND_SIZE, n_rows)

    def _build_explainer(self):
        """Pick the SHAP explainer that matches the trained model type."""
        model_name = type(self.model).__name__
        try:
            if model_name == "LogisticRegression":
                # Linear model: exact, fast, needs a background sample.
                return shap.LinearExplainer(self.model, self._background())
            if model_name in ("RandomForestClassifier", "XGBClassifier"):
                # Tree ensembles: TreeExplainer is exact and needs no background.
                return shap.TreeExplainer(self.model)
        except Exception:
            pass
        # Fallback: model-agnostic explainer over a background sample.
        return shap.Explainer(self.model.predict_proba, self._background())

    # -- explanation -------------------------------------------------------
    def _shap_for_row(self, transformed_row: np.ndarray) -> np.ndarray:
        """
        Return a 1-D array of SHAP values (one per encoded feature) for the
        positive (churn) class. SHAP's return shape varies by explainer/version,
        so we normalise it here.
        """
        values = self.explainer.shap_values(transformed_row)
        values = np.asarray(values)
        # Some explainers return a list/array per class: [class0, class1].
        if isinstance(values, list):
            values = np.asarray(values[-1])
        # Collapse leading singleton / class dimensions down to (n_features,).
        values = np.squeeze(values)
        if values.ndim == 2:
            # shape (n_features, n_classes) -> take the churn (last) class
            values = values[:, -1]
        return np.asarray(values).reshape(-1)

    def _top_factors(self, record: dict, transformed_row: np.ndarray, k: int = 5):
        """
        Return the top-k features pushing this customer TOWARD churn.

        For one customer only one one-hot column per category is active, so the
        SHAP value on that active column is its contribution. We keep the
        positive contributions (those that raise churn risk), sort by magnitude,
        and attach a readable label + the customer's actual value.
        """
        shap_values = self._shap_for_row(transformed_row)
        order = np.argsort(shap_values)[::-1]  # most churn-increasing first

        factors = []
        for idx in order:
            contribution = float(shap_values[idx])
            if contribution <= 0:
                break  # remaining features reduce churn risk; stop
            encoded = self.features[idx]
            col, value = split_feature(encoded)

            # For one-hot categorical features, only surface the column that is
            # ACTUALLY active for this customer. An inactive one-hot column (e.g.
            # "InternetService_DSL" for a fibre customer) can still carry a SHAP
            # value vs the background, but reporting it as this customer's driver
            # would be misleading — their value isn't DSL.
            if value is not None and str(record.get(col)) != str(value):
                continue

            factors.append({
                "feature": encoded,
                "label": readable_label(encoded),
                "value": record.get(col),       # the customer's actual value
                "contribution": round(contribution, 4),
            })
            if len(factors) == k:
                break
        return factors

    # -- public scoring ----------------------------------------------------
    def predict_one(self, partial_record: dict) -> dict:
        """Score a single customer (dict of raw fields, possibly partial)."""
        record = fill_defaults(partial_record)
        X = pd.DataFrame([record], columns=ALL_COLS)
        transformed = self.preprocessor.transform(X)

        prob = float(self.model.predict_proba(transformed)[0, 1])
        factors = self._top_factors(record, transformed)
        top_factor = factors[0]["feature"] if factors else ""

        return {
            "churn_probability": round(prob, 4),
            "risk_level": risk_level(prob),
            "top_factors": factors,
            "recommended_action": retention_action(top_factor),
        }

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row of an uploaded CSV. Adds three columns:
        churn_probability, risk_level, top_factor. Missing columns are filled
        with defaults so partially-specified CSVs still work.
        """
        records = [fill_defaults(row) for row in df.to_dict(orient="records")]
        X = pd.DataFrame(records, columns=ALL_COLS)
        transformed = self.preprocessor.transform(X)

        probs = self.model.predict_proba(transformed)[:, 1]

        # Top factor per row: the active feature with the largest SHAP value.
        shap_matrix = np.asarray(self.explainer.shap_values(transformed))
        if isinstance(shap_matrix, list):
            shap_matrix = np.asarray(shap_matrix[-1])
        shap_matrix = np.squeeze(shap_matrix)
        if shap_matrix.ndim == 3:           # (rows, features, classes)
            shap_matrix = shap_matrix[:, :, -1]
        top_idx = shap_matrix.argmax(axis=1)
        top_factors = [readable_label(self.features[i]) for i in top_idx]

        out = df.copy()
        out["churn_probability"] = np.round(probs, 4)
        out["risk_level"] = [risk_level(float(p)) for p in probs]
        out["top_factor"] = top_factors
        return out

"""
schema.py
---------
Everything the API needs to know about the *shape* of a Telco customer record:

- the full list of columns the preprocessor was trained on,
- sensible default values for every column (so a partially-filled form can still
  be turned into a complete row that ``preprocessor.transform`` accepts),
- the Pydantic request model for a single /predict call,
- human-readable labels for the encoded feature names, and
- the retention-action playbook that maps a churn driver to a marketing action.

Keeping all of this in one small, readable file makes the rest of the backend
short and easy to explain in a viva.
"""

from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. The columns the model was trained on.
#    These are the raw Telco columns AFTER dropping customerID and the target.
#    The preprocessor (ColumnTransformer) expects a DataFrame with exactly
#    these columns, so /predict must reconstruct all of them.
# ---------------------------------------------------------------------------
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

ALL_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# ---------------------------------------------------------------------------
# 2. Default value for every training column.
#    These are typical/most-common values from the dataset. They are only used
#    to fill in fields the caller did not provide, so preprocessor.transform()
#    never sees a missing column. The defaults are deliberately "low risk"
#    (mostly "No"), so an unspecified field nudges the score as little as
#    possible.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


def fill_defaults(partial: dict) -> dict:
    """
    Return a complete record: start from DEFAULTS, then overwrite with whatever
    the caller actually sent (ignoring None values). The result always has all
    ALL_COLS keys, so it can be turned into a one-row DataFrame and fed to the
    preprocessor without errors.
    """
    record = dict(DEFAULTS)
    for key, value in partial.items():
        if value is not None and key in record:
            record[key] = value
    return record


# ---------------------------------------------------------------------------
# 3. Request model for POST /predict.
#    Every field is Optional so the form can send only the fields it collects;
#    anything omitted falls back to DEFAULTS. The 14 fields the frontend form
#    exposes are listed first; the rest are accepted too but rarely sent.
# ---------------------------------------------------------------------------
class CustomerIn(BaseModel):
    # Fields the frontend form collects
    tenure: Optional[int] = Field(default=None, description="Months the customer has stayed")
    Contract: Optional[str] = None              # Month-to-month / One year / Two year
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None
    InternetService: Optional[str] = None       # DSL / Fiber optic / No
    TechSupport: Optional[str] = None            # Yes / No / No internet service
    OnlineSecurity: Optional[str] = None
    PaymentMethod: Optional[str] = None
    PaperlessBilling: Optional[str] = None       # Yes / No
    SeniorCitizen: Optional[int] = None          # 0 / 1
    Partner: Optional[str] = None                # Yes / No
    Dependents: Optional[str] = None
    gender: Optional[str] = None                 # Female / Male
    PhoneService: Optional[str] = None

    # Less-common fields (accepted if sent, otherwise defaulted)
    MultipleLines: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None


# ---------------------------------------------------------------------------
# 4. Turning encoded feature names into readable labels.
#    The preprocessor produces names like:
#        "num__tenure"
#        "cat__Contract_Month-to-month"
#        "cat__TechSupport_No"
#    We strip the transformer prefix and split "Column_Value" into a friendly
#    "Column: Value" label so explanations read naturally in the UI.
# ---------------------------------------------------------------------------
# Friendlier display names for a few columns.
_COLUMN_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Contract": "Contract",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "TechSupport": "Tech support",
    "PaymentMethod": "Payment method",
    "PaperlessBilling": "Paperless billing",
    "SeniorCitizen": "Senior citizen",
}


def split_feature(encoded_name: str):
    """
    Split an encoded feature name into (raw_column, value_or_None).
      "num__tenure"                  -> ("tenure", None)
      "cat__Contract_Month-to-month" -> ("Contract", "Month-to-month")
    Returns the underlying raw column (used for the retention playbook) and the
    one-hot value (None for numeric features).
    """
    name = encoded_name.split("__", 1)[-1]  # drop "num__" / "cat__" prefix
    # Numeric features have no "_value" suffix to split off.
    if name in NUMERIC_COLS or name in _COLUMN_LABELS and "_" not in name:
        return name, None
    # Categorical: the raw column is the longest known column that prefixes name.
    for col in sorted(CATEGORICAL_COLS, key=len, reverse=True):
        if name == col:
            return col, None
        if name.startswith(col + "_"):
            return col, name[len(col) + 1:]
    return name, None


def readable_label(encoded_name: str) -> str:
    """Turn an encoded feature name into a human-readable label for the UI."""
    col, value = split_feature(encoded_name)
    pretty_col = _COLUMN_LABELS.get(col, col)
    return f"{pretty_col}: {value}" if value is not None else pretty_col


# ---------------------------------------------------------------------------
# 5. Retention playbook.
#    Maps the raw column of the top churn driver to a concrete marketing action
#    a retention team can take. This is the "so what do we DO about it" layer
#    that turns a prediction into a recommendation.
# ---------------------------------------------------------------------------
RETENTION_ACTIONS = {
    "Contract": "Offer a 1- or 2-year contract with a loyalty discount to lock in the customer.",
    "tenure": "Early-life risk: trigger an onboarding check-in and a welcome retention offer.",
    "MonthlyCharges": "Review pricing — offer a loyalty discount or a better-fit plan.",
    "TotalCharges": "High lifetime spend at risk: assign to a priority retention/VIP track.",
    "OnlineSecurity": "Offer a free trial of the Online Security add-on.",
    "TechSupport": "Offer a Tech Support add-on or a free support call.",
    "PaymentMethod": "Encourage a switch to automatic card/bank payment (electronic-check users churn more).",
    "InternetService": "Proactive fiber reliability/value check-in to address service friction.",
    "PaperlessBilling": "Confirm billing preferences and highlight account-management perks.",
    "Partner": "Promote family/household bundle benefits.",
    "Dependents": "Promote family/household bundle benefits.",
}

DEFAULT_ACTION = "Add to the general retention campaign and monitor engagement."


def retention_action(top_factor_encoded: str) -> str:
    """Pick a retention action based on the raw column of the top churn driver."""
    col, _ = split_feature(top_factor_encoded)
    return RETENTION_ACTIONS.get(col, DEFAULT_ACTION)

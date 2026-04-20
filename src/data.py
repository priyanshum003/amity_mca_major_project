"""
data.py
-------
Loads and cleans the Telco Customer Churn dataset, and builds a preprocessing
pipeline (one-hot encoding for categorical columns, scaling for numeric columns).

The cleaning steps here are exactly what you describe in Chapter 3 (Methodology)
and Chapter 4 (the data-preparation part of Data Analysis).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/telco_churn.csv"

# Columns that are numeric (everything else categorical, except IDs/target)
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
DROP_COLS = ["customerID"]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data:
    - TotalCharges has blank strings for brand-new customers (tenure == 0);
      convert it to a number and fill those blanks with 0.
    - Map the target Churn from Yes/No to 1/0.
    - Drop the customerID column (it carries no predictive signal).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def split_features_target(df: pd.DataFrame):
    """Separate the feature columns (X) from the target column (y)."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline:
    - numeric columns  -> StandardScaler (helps logistic regression; harmless for trees)
    - categorical cols -> OneHotEncoder  (turns text categories into 0/1 columns)
    """
    numeric = [c for c in NUMERIC_COLS if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def get_clean_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Convenience: load + clean in one call."""
    return clean(load_raw(path))

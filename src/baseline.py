"""
baseline.py
-----------
A simple, transparent rule-based churn scorer. This is the NON-ML benchmark.
Its only job is to give you something to beat, so your report can say
"the ML models outperformed a sensible rule-based baseline by X%".

The rules below encode common-sense retention knowledge (the kind your
marketing guide would recognise): short-tenure, month-to-month, high-bill,
no-support customers are the classic churn risks.
"""

import pandas as pd


def rule_based_scores(df_clean: pd.DataFrame) -> pd.Series:
    """
    Return an integer risk score per customer based on hand-written rules.
    Input is the CLEANED dataframe (before one-hot encoding), so the original
    text columns like 'Contract' are still readable.
    """
    score = pd.Series(0, index=df_clean.index)

    score += (df_clean["Contract"] == "Month-to-month").astype(int) * 2
    score += (df_clean["tenure"] < 12).astype(int) * 2
    # "high" monthly charge = above the 75th percentile
    high_charge = df_clean["MonthlyCharges"] > df_clean["MonthlyCharges"].quantile(0.75)
    score += high_charge.astype(int) * 1
    score += (df_clean["TechSupport"] == "No").astype(int) * 1
    score += (df_clean["PaymentMethod"] == "Electronic check").astype(int) * 1

    return score


def rule_based_predict(df_clean: pd.DataFrame, threshold: int = 3) -> pd.Series:
    """Classify a customer as churn-risk (1) if their rule score >= threshold."""
    return (rule_based_scores(df_clean) >= threshold).astype(int)

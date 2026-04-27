"""
train.py
--------
The heart of the project. Run this once, offline, and it produces everything
your report's Chapter 4 needs:
  - a model comparison table (CSV)
  - ROC and Precision-Recall curve charts
  - a confusion matrix for the best model
  - SHAP plots explaining WHY the model predicts churn
  - the saved model + preprocessor (model.joblib) for the web app

Run from the churn_project folder:  python src/train.py
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # render charts to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data import build_preprocessor, get_clean_dataframe, split_features_target
from baseline import rule_based_predict, rule_based_scores

warnings.filterwarnings("ignore")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42


def evaluate(name, y_true, y_pred, y_proba):
    """Compute the standard classification metrics for one model."""
    return {
        "Model": name,
        "Precision": round(precision_score(y_true, y_pred), 3),
        "Recall": round(recall_score(y_true, y_pred), 3),
        "F1": round(f1_score(y_true, y_pred), 3),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 3),
        "PR-AUC": round(average_precision_score(y_true, y_proba), 3),
    }


def main():
    # 1. Load + clean
    df = get_clean_dataframe()
    X, y = split_features_target(df)
    print(f"Dataset: {len(df)} customers, churn rate = {y.mean():.1%}")

    # 2. Train/test split (stratified keeps the churn ratio the same in both sets)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    # 3. Preprocess (fit on train only, then apply to both)
    pre = build_preprocessor(X_train)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    feat_names = pre.get_feature_names_out()

    results = []
    curves = {}  # for ROC / PR plots

    # 4a. Rule-based baseline (no ML) — evaluated on the same test customers
    df_test_clean = df.iloc[idx_test]
    rb_pred = rule_based_predict(df_test_clean).values
    rb_score = rule_based_scores(df_test_clean).values
    rb_proba = rb_score / rb_score.max()  # normalise score to 0-1 for ranking metrics
    results.append(evaluate("Rule-based baseline", y_test, rb_pred, rb_proba))
    curves["Rule-based baseline"] = rb_proba

    # 4b. Machine-learning models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            tree_method="hist", eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE,
        ),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append(evaluate(name, y_test, pred, proba))
        curves[name] = proba
        fitted[name] = model

    # 5. Comparison table -> CSV + console
    table = pd.DataFrame(results)
    table.to_csv(f"{OUT}/model_comparison.csv", index=False)
    print("\n=== Model comparison ===")
    print(table.to_string(index=False))

    # pick the best ML model by F1 (balances precision & recall on imbalanced data)
    ml_only = table[table["Model"] != "Rule-based baseline"]
    best_name = ml_only.loc[ml_only["F1"].idxmax(), "Model"]
    print(f"\nBest model: {best_name}")

    # 6. Charts ---------------------------------------------------------------
    # 6a. Model comparison bar chart
    ax = table.set_index("Model")[["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].plot(
        kind="bar", figsize=(9, 5), rot=20
    )
    ax.set_title("Model comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/model_comparison.png", dpi=120)
    plt.close()

    # 6b. ROC curves
    plt.figure(figsize=(7, 6))
    for name, proba in curves.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/roc_curves.png", dpi=120)
    plt.close()

    # 6c. Precision-Recall curves
    plt.figure(figsize=(7, 6))
    for name, proba in curves.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/pr_curves.png", dpi=120)
    plt.close()

    # 6d. Confusion matrix for the best model
    best_proba = curves[best_name]
    cm = confusion_matrix(y_test, (best_proba >= 0.5).astype(int))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["No churn", "Churn"])
    plt.yticks([0, 1], ["No churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_matrix.png", dpi=120)
    plt.close()

    # 7. SHAP explainability (on the best tree model) -------------------------
    shap_model = fitted.get("XGBoost", fitted[best_name])
    sample = Xte[:500]
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(sample)
    sample_df = pd.DataFrame(sample, columns=feat_names)

    shap.summary_plot(shap_values, sample_df, show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # top churn drivers (mean |SHAP|)
    top = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feat_names)
        .sort_values(ascending=False)
        .head(10)
    )
    print("\n=== Top 10 churn drivers (mean |SHAP|) ===")
    print(top.round(3).to_string())

    # 8. Save the best model + preprocessor for the web app
    joblib.dump({"preprocessor": pre, "model": fitted[best_name],
                 "best_name": best_name, "features": list(feat_names)},
                f"{OUT}/model.joblib")

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"best_model": best_name,
                   "churn_rate": round(float(y.mean()), 4),
                   "n_customers": int(len(df)),
                   "top_drivers": top.round(4).to_dict()}, f, indent=2)

    print(f"\nAll outputs written to ./{OUT}/  (charts, table, model.joblib, summary.json)")


if __name__ == "__main__":
    main()

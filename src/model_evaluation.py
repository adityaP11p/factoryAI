"""
FactoryGuard AI - Model Evaluation Module
Comprehensive evaluation with Precision, Recall, F1, PR-AUC, and visualizations.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    EVALUATION_REPORT_PATH,
    PR_CURVE_PATH,
    CONFUSION_MATRIX_PATH,
    MODELS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name="Best Model"):
    """Generate comprehensive evaluation metrics."""
    logger.info("=" * 60)
    logger.info("STAGE 5: MODEL EVALUATION")
    logger.info("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Core metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # PR-AUC
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_curve, prec_curve)
    avg_precision = average_precision_score(y_test, y_proba)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Normal", "Failure"])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Log results
    logger.info(f"\n{'='*50}")
    logger.info(f"MODEL: {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"Precision:           {precision:.4f}")
    logger.info(f"Recall:              {recall:.4f}")
    logger.info(f"F1 Score:            {f1:.4f}")
    logger.info(f"PR-AUC:              {pr_auc:.4f}")
    logger.info(f"Average Precision:   {avg_precision:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Saving evaluation report
    _save_evaluation_report(
        model_name, precision, recall, f1, pr_auc, avg_precision, report, cm
    )

    # Ploting PR curve
    _plot_pr_curve(prec_curve, rec_curve, pr_auc, model_name)

    # Ploting confusion matrix
    _plot_confusion_matrix(cm, model_name)

    logger.info("Model evaluation complete ✓")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "avg_precision": avg_precision,
        "confusion_matrix": cm,
    }


def _save_evaluation_report(model_name, precision, recall, f1, pr_auc, avg_precision, report, cm):
    """Save evaluation metrics to a text file."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(EVALUATION_REPORT_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("FactoryGuard AI - Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("--- Primary Metrics ---\n")
        f.write(f"Precision:           {precision:.4f}\n")
        f.write(f"Recall:              {recall:.4f}\n")
        f.write(f"F1 Score:            {f1:.4f}\n")
        f.write(f"PR-AUC:              {pr_auc:.4f}\n")
        f.write(f"Average Precision:   {avg_precision:.4f}\n\n")
        f.write("--- Classification Report ---\n")
        f.write(report + "\n\n")
        f.write("--- Confusion Matrix ---\n")
        f.write(f"               Predicted Normal  Predicted Failure\n")
        f.write(f"Actual Normal    {cm[0][0]:>10d}       {cm[0][1]:>10d}\n")
        f.write(f"Actual Failure   {cm[1][0]:>10d}       {cm[1][1]:>10d}\n")

    logger.info(f"Evaluation report saved to: {EVALUATION_REPORT_PATH}")


def _plot_pr_curve(precision, recall, pr_auc, model_name):
    """Plot and save Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(recall, precision, color="#2196F3", linewidth=2.5,
            label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color="#2196F3")

    # Baseline (random classifier)
    baseline = sum(precision) / len(precision) if len(precision) > 0 else 0
    ax.axhline(y=baseline, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"Baseline (y = {baseline:.3f})", alpha=0.7)

    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("Precision-Recall Curve – FactoryGuard AI", fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve saved to: {PR_CURVE_PATH}")


def _plot_confusion_matrix(cm, model_name):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Adding text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", color=color, fontsize=16, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Failure"], fontsize=11)
    ax.set_yticklabels(["Normal", "Failure"], fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    import joblib
    from src.config import BEST_MODEL_PATH, SCALER_PATH, FEATURES_DATA_PATH, TARGET_COLUMN, FAILURE_TYPE_COLUMNS, TEST_SIZE

    # Loading model and data
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(FEATURES_DATA_PATH)
    exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    evaluate_model(model, X_test_scaled, y_test)
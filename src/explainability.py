"""
FactoryGuard AI - Explainability Module
SHAP-based model interpretability for maintenance engineers.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SHAP_SUMMARY_PATH,
    MODELS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def generate_shap_explanations(model, X_test, feature_names=None, max_samples=500):
    """Generate SHAP values for model predictions."""
    logger.info("=" * 60)
    logger.info("STAGE 6: SHAP EXPLAINABILITY")
    logger.info("=" * 60)

    # Limiting samples for performance
    if len(X_test) > max_samples:
        X_sample = X_test.iloc[:max_samples] if isinstance(X_test, pd.DataFrame) else X_test[:max_samples]
        logger.info(f"Using {max_samples} samples for SHAP analysis (out of {len(X_test)})")
    else:
        X_sample = X_test

    # Creating SHAP explainer
    logger.info("Creating SHAP TreeExplainer..")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        logger.info(f"SHAP values computed successfully")
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}")
        logger.info("Falling back to KernelExplainer..")
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100] if isinstance(X_sample, pd.DataFrame) else X_sample[:100])
        shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, feature_names=None, save_path=SHAP_SUMMARY_PATH):
    """Generate and saving SHAP summary plot."""
    logger.info("Generating SHAP summary plot..")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Handling different SHAP value formats
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Importance – FactoryGuard AI", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"SHAP summary plot saved to: {save_path}")


def get_feature_importance_ranking(shap_values, feature_names):
    """Get ranked feature importance from SHAP values."""
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    logger.info("\nTop 10 Feature Importance (SHAP):")
    logger.info("-" * 45)
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:35s} | {row['mean_abs_shap']:.4f}")

    return importance_df


def explain_single_prediction(model, explainer, X_single, feature_names):
    """Generate SHAP explanation for a single prediction."""
    shap_values = explainer.shap_values(X_single)

    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    if len(sv.shape) == 2:
        sv = sv[0]

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "abs_shap": np.abs(sv),
    }).sort_values("abs_shap", ascending=False)

    return explanation


def run_explainability(model, X_test, feature_names):
    """Executing the full explainability pipeline"""
    # Generating SHAP values
    explainer, shap_values, X_sample = generate_shap_explanations(
        model, X_test, feature_names
    )

    plot_shap_summary(shap_values, X_sample, feature_names)

    # Getting feature importance ranking
    importance_df = get_feature_importance_ranking(shap_values, feature_names)

    # Saving importance ranking
    importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to: {importance_path}")

    logger.info("Explainability analysis complete")
    return explainer, shap_values, importance_df


if __name__ == "__main__":
    from src.config import BEST_MODEL_PATH, SCALER_PATH, FEATURES_DATA_PATH, TARGET_COLUMN, FAILURE_TYPE_COLUMNS, TEST_SIZE

    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(FEATURES_DATA_PATH)
    exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_test = X.iloc[split_idx:]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    run_explainability(model, X_test_scaled, feature_cols)
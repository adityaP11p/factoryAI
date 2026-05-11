"""
FactoryGuard AI - Model Training Module
Trains baseline and advanced ML models for predictive maintenance.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    FEATURES_DATA_PATH,
    TARGET_COLUMN,
    FAILURE_TYPE_COLUMNS,
    RANDOM_SEED,
    TEST_SIZE,
    XGBOOST_DEFAULT_PARAMS,
    LIGHTGBM_DEFAULT_PARAMS,
    BEST_MODEL_PATH,
    SCALER_PATH,
    BEST_PARAMS_PATH,
    MODELS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def load_feature_data(filepath=FEATURES_DATA_PATH):
    """Load engineered feature data."""
    logger.info(f"Loading feature data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature data not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def prepare_data(df):
    """Prepare features and target, split into train/test sets."""
    logger.info("Preparing data for training...")

    # Separate features and target
    exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    logger.info(f"Target: {TARGET_COLUMN}")

    # Time-based split (no shuffling to respect temporal order)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set:  {X_test.shape[0]} samples")
    logger.info(f"Train failures: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    logger.info(f"Test failures:  {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    logger.info("Features scaled with StandardScaler ✓")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def compute_scale_pos_weight(y_train):
    """Compute scale_pos_weight for handling class imbalance."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    weight = neg_count / max(pos_count, 1)
    logger.info(f"Computed scale_pos_weight: {weight:.2f}")
    return weight


def compute_pr_auc(y_true, y_proba):
    """Compute Precision-Recall AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train baseline models: Logistic Regression and Random Forest."""
    logger.info("-" * 40)
    logger.info("BASELINE MODELS")
    logger.info("-" * 40)

    results = {}
    spw = compute_scale_pos_weight(y_train)

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_prauc = compute_pr_auc(y_test, lr_proba)
    lr_preds = lr.predict(X_test)
    lr_f1 = f1_score(y_test, lr_preds)
    results["Logistic Regression"] = {
        "model": lr,
        "pr_auc": lr_prauc,
        "f1": lr_f1,
    }
    logger.info(f"  PR-AUC: {lr_prauc:.4f} | F1: {lr_f1:.4f}")

    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_prauc = compute_pr_auc(y_test, rf_proba)
    rf_preds = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds)
    results["Random Forest"] = {
        "model": rf,
        "pr_auc": rf_prauc,
        "f1": rf_f1,
    }
    logger.info(f"  PR-AUC: {rf_prauc:.4f} | F1: {rf_f1:.4f}")

    return results


def train_xgboost(X_train, X_test, y_train, y_test, params=None):
    """Train XGBoost model with class imbalance handling."""
    logger.info("-" * 40)
    logger.info("XGBOOST MODEL")
    logger.info("-" * 40)

    spw = compute_scale_pos_weight(y_train)

    if params is None:
        params = XGBOOST_DEFAULT_PARAMS.copy()
        params["scale_pos_weight"] = spw

    logger.info(f"Training XGBoost with params: {params}")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc_score = compute_pr_auc(y_test, y_proba)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"XGBoost PR-AUC: {pr_auc_score:.4f} | F1: {f1:.4f}")

    return model, pr_auc_score, f1


def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM model with class imbalance handling."""
    logger.info("-" * 40)
    logger.info("LIGHTGBM MODEL")
    logger.info("-" * 40)

    spw = compute_scale_pos_weight(y_train)
    params = LIGHTGBM_DEFAULT_PARAMS.copy()
    params["scale_pos_weight"] = spw

    logger.info(f"Training LightGBM with params: {params}")

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc_score = compute_pr_auc(y_test, y_proba)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"LightGBM PR-AUC: {pr_auc_score:.4f} | F1: {f1:.4f}")

    return model, pr_auc_score, f1


def select_best_model(all_results):
    """Select the best model based on PR-AUC score."""
    logger.info("-" * 40)
    logger.info("MODEL COMPARISON")
    logger.info("-" * 40)

    best_name = None
    best_prauc = -1
    best_model = None

    for name, info in all_results.items():
        prauc = info["pr_auc"]
        f1 = info["f1"]
        logger.info(f"  {name:25s} | PR-AUC: {prauc:.4f} | F1: {f1:.4f}")

        if prauc > best_prauc:
            best_prauc = prauc
            best_name = name
            best_model = info["model"]

    logger.info(f"\n  ★ Best model: {best_name} (PR-AUC: {best_prauc:.4f})")
    return best_name, best_model, best_prauc


def save_model_artifacts(model, scaler, best_name, best_prauc, feature_cols, all_results):
    """Save model, scaler, and metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save best model
    joblib.dump(model, BEST_MODEL_PATH)
    logger.info(f"Best model ({best_name}) saved to: {BEST_MODEL_PATH}")

    # Save ALL models individually for the Ensemble UI
    for name, info in all_results.items():
        # Sanitize name
        clean_name = name.lower().replace(" ", "_")
        model_path = os.path.join(MODELS_DIR, f"{clean_name}_model.joblib")
        joblib.dump(info["model"], model_path)
        logger.info(f"Saved {name} to {model_path}")

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to: {SCALER_PATH}")

    # Save metadata
    metadata = {
        "best_model_name": best_name,
        "pr_auc": best_prauc,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "all_models_prauc": {name: info["pr_auc"] for name, info in all_results.items()},
        "all_models_f1": {name: info["f1"] for name, info in all_results.items()}
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {BEST_PARAMS_PATH}")


def run_training(tuned_params=None):
    """Execute the full model training pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 4: MODEL TRAINING")
    logger.info("=" * 60)

    # Load and prepare data
    df = load_feature_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)

    # Train baseline models
    all_results = train_baseline_models(X_train, X_test, y_train, y_test)

    # Train XGBoost
    xgb_model, xgb_prauc, xgb_f1 = train_xgboost(
        X_train, X_test, y_train, y_test, params=tuned_params
    )
    all_results["XGBoost"] = {"model": xgb_model, "pr_auc": xgb_prauc, "f1": xgb_f1}

    # Train LightGBM
    lgb_model, lgb_prauc, lgb_f1 = train_lightgbm(X_train, X_test, y_train, y_test)
    all_results["LightGBM"] = {"model": lgb_model, "pr_auc": lgb_prauc, "f1": lgb_f1}

    # Select best model
    best_name, best_model, best_prauc = select_best_model(all_results)

    # Save artifacts
    save_model_artifacts(best_model, scaler, best_name, best_prauc, feature_cols, all_results)

    logger.info("Model training complete ✓")

    return best_model, scaler, X_train, X_test, y_train, y_test, feature_cols, all_results


if __name__ == "__main__":
    run_training()
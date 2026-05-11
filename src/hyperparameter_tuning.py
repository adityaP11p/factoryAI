"""
FactoryGuard AI - Hyperparameter Tuning Module
Optuna-based hyperparameter optimization for XGBoost.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
import xgboost as xgb
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    FEATURES_DATA_PATH,
    TARGET_COLUMN,
    FAILURE_TYPE_COLUMNS,
    RANDOM_SEED,
    TEST_SIZE,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    XGBOOST_SEARCH_SPACE,
    BEST_PARAMS_PATH,
    MODELS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# Suppressing Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_and_prepare_data():
    """Load feature data and prepare train/validation split."""
    df = pd.read_csv(FEATURES_DATA_PATH)

    exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    # Time-based split
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, feature_cols


def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function: maximize PR-AUC."""
    space = XGBOOST_SEARCH_SPACE

    params = {
        "max_depth": trial.suggest_int("max_depth", space["max_depth"]["low"], space["max_depth"]["high"]),
        "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"]["low"], space["learning_rate"]["high"], log=space["learning_rate"].get("log", False)),
        "n_estimators": trial.suggest_int("n_estimators", space["n_estimators"]["low"], space["n_estimators"]["high"]),
        "subsample": trial.suggest_float("subsample", space["subsample"]["low"], space["subsample"]["high"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", space["colsample_bytree"]["low"], space["colsample_bytree"]["high"]),
        "min_child_weight": trial.suggest_int("min_child_weight", space["min_child_weight"]["low"], space["min_child_weight"]["high"]),
        "gamma": trial.suggest_float("gamma", space["gamma"]["low"], space["gamma"]["high"]),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", space["scale_pos_weight"]["low"], space["scale_pos_weight"]["high"]),
        "eval_metric": "aucpr",
        "random_state": RANDOM_SEED,
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)

    return pr_auc


def run_tuning():
    """Execute Optuna hyperparameter tuning."""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING (Optuna)")
    logger.info("=" * 60)

    # Loading data
    X_train, X_val, y_train, y_val, feature_cols = load_and_prepare_data()
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Creating study
    study = optuna.create_study(
        direction="maximize",
        study_name="factoryguard_xgboost",
    )

    # Running optimization
    logger.info(f"Starting {OPTUNA_N_TRIALS} trials (timeout: {OPTUNA_TIMEOUT}s)...")

    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )

    # Results
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"\nBest PR-AUC: {best_value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # Adding fixed params
    best_params["eval_metric"] = "aucpr"
    best_params["random_state"] = RANDOM_SEED
    best_params["use_label_encoder"] = False

    # Saving best params
    os.makedirs(MODELS_DIR, exist_ok=True)
    tuning_results = {
        "best_pr_auc": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "feature_columns": feature_cols,
    }

    tuning_path = os.path.join(MODELS_DIR, "tuning_results.json")
    with open(tuning_path, "w") as f:
        json.dump(tuning_results, f, indent=2)
    logger.info(f"Tuning results saved to: {tuning_path}")

    logger.info("Hyperparameter tuning complete ✓")
    return best_params


if __name__ == "__main__":
    best_params = run_tuning()
    print(f"\nBest params: {json.dumps(best_params, indent=2)}")
"""
FactoryGuard AI - Configuration Module
Central configuration for paths, features, model parameters, and pipeline settings.
"""

import os

# PATH CONFIGURATION :

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DATA_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data file paths

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "sensor_data.csv")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned.csv")
FEATURES_DATA_PATH = os.path.join(FEATURES_DATA_DIR, "features.csv")

# Model artifact paths

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "best_params.json")
EVALUATION_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.txt")
PR_CURVE_PATH = os.path.join(MODELS_DIR, "pr_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
SHAP_SUMMARY_PATH = os.path.join(MODELS_DIR, "shap_summary.png")


# DATA CONFIGURATION :

# Dropping columns from raw data
DROP_COLUMNS = ["UDI", "Product ID"]

TARGET_COLUMN = "machine_failure"

# Failure type columns , not used as features 
FAILURE_TYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Categorical column to encode
CATEGORICAL_COLUMN = "Type"

# Base sensor features from cleaned data 
SENSOR_FEATURES = [
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
]

# FEATURE ENGINEERING CONFIGURATION :

# Rolling window sizes in number of samples
ROLLING_WINDOWS = [6]  # 6-sample rolling window

# Rolling statistics features to create
ROLLING_FEATURES_CONFIG = {
    "air_temp": {"stats": ["mean"], "prefix": "air_temp"},
    "process_temp": {"stats": ["mean"], "prefix": "process_temp"},
    "rotational_speed": {"stats": ["std"], "prefix": "rpm"},
    "torque": {"stats": ["std"], "prefix": "torque"},
}

# Lagging features configuration
LAG_FEATURES_CONFIG = {
    "air_temp": {"lags": [1, 2], "prefix": "air_temp"},
}


# MODEL CONFIGURATION :

# Random seed for reproducibility
RANDOM_SEED = 42

# Train / test split ratio (time-based, no shuffling)
TEST_SIZE = 0.2

XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
}

LIGHTGBM_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# HYPERPARAMETER TUNING CONFIGURATION:

OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 600  # seconds

# Searching space for XGBoost hyperparameter tuning
XGBOOST_SEARCH_SPACE = {
    "max_depth": {"low": 3, "high": 10},
    "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
    "n_estimators": {"low": 100, "high": 500},
    "subsample": {"low": 0.6, "high": 1.0},
    "colsample_bytree": {"low": 0.6, "high": 1.0},
    "min_child_weight": {"low": 1, "high": 10},
    "gamma": {"low": 0.0, "high": 5.0},
    "scale_pos_weight": {"low": 5, "high": 50},
}

# LOGGING CONFIGURATION :

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
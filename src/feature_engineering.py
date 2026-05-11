"""
FactoryGuard AI - Feature Engineering Module
Creates rolling statistics, lag features, and derived features from sensor data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    FEATURES_DATA_PATH,
    FEATURES_DATA_DIR,
    ROLLING_WINDOWS,
    ROLLING_FEATURES_CONFIG,
    LAG_FEATURES_CONFIG,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def create_rolling_features(df, windows=ROLLING_WINDOWS):
    """Create rolling statistics features for sensor data."""
    logger.info("Creating rolling statistics features..")

    features_created = []

    for window in windows:
        for col, config in ROLLING_FEATURES_CONFIG.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            prefix = config["prefix"]
            stats = config["stats"]

            for stat in stats:
                feature_name = f"{prefix}_{stat}_{window}"

                if stat == "mean":
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                elif stat == "std":
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                elif stat == "min":
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                elif stat == "max":
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).max()

                features_created.append(feature_name)

    logger.info(f"Created {len(features_created)} rolling features: {features_created}")
    return df


def create_lag_features(df):
    """Create lag features (t-1, t-2) for specified sensor columns."""
    logger.info("Creating lag features..")

    features_created = []

    for col, config in LAG_FEATURES_CONFIG.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        prefix = config["prefix"]
        lags = config["lags"]

        for lag in lags:
            feature_name = f"{prefix}_lag{lag}"
            df[feature_name] = df[col].shift(lag)
            features_created.append(feature_name)

    logger.info(f"Created {len(features_created)} lag features: {features_created}")
    return df


def create_derived_features(df):
    """Create derived/interaction features from existing sensor data."""
    logger.info("Creating derived features...")

    features_created = []

    # Temperature difference (process - air)
    if "process_temp" in df.columns and "air_temp" in df.columns:
        df["temp_diff"] = df["process_temp"] - df["air_temp"]
        features_created.append("temp_diff")

    # Wear-Torque interaction
    if "tool_wear" in df.columns and "torque" in df.columns:
        df["wear_torque"] = df["tool_wear"] * df["torque"]
        features_created.append("wear_torque")

    # Exponential Moving Average (EMA) constraints
    if "rotational_speed" in df.columns:
        df["rpm_ema"] = df["rotational_speed"].ewm(span=6, min_periods=1).mean()
        features_created.append("rpm_ema")

    logger.info(f"Created {len(features_created)} derived features: {features_created}")
    return df


def handle_feature_nulls(df):
    """Handle NaN values created by rolling/lag operations."""
    initial_count = len(df)
    null_counts_before = df.isnull().sum().sum()

    if null_counts_before > 0:
        # Filling NaN in rolling/lag features with forward fill then backward fill
        feature_cols = [col for col in df.columns if any(
            pattern in col for pattern in ["_mean_", "_std_", "_lag", "temp_diff", "wear_torque"]
        )]

        for col in feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].bfill()
                df[col] = df[col].ffill()

        # Dropping any remaining rows with NaN
        df = df.dropna()
        removed = initial_count - len(df)

        if removed > 0:
            logger.info(f"Removed {removed} rows with remaining NaN values")

    logger.info(f"Feature data shape after NaN handling: {df.shape}")
    return df


def save_features(df, filepath=FEATURES_DATA_PATH):
    """Save engineered features to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved feature data to: {filepath}")


def run_feature_engineering(df):
    """Execute the full feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Creating rolling features
    df = create_rolling_features(df)

    # Creating lag features
    df = create_lag_features(df)

    # Creating derived features
    df = create_derived_features(df)

    # Handling NaN values from feature creation
    df = handle_feature_nulls(df)

    # Saving features
    save_features(df)

    # Logging feature summary
    logger.info(f"Final feature set: {df.shape[1]} columns")
    logger.info(f"Feature columns: {list(df.columns)}")
    logger.info("Feature engineering complete")

    return df


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    df = run_preprocessing()
    run_feature_engineering(df)
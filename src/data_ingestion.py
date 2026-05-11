"""
FactoryGuard AI - Data Ingestion Module
Loads raw sensor data, performs initial validation, and saves cleaned output.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Adding parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    DROP_COLUMNS,
    CATEGORICAL_COLUMN,
    PROCESSED_DATA_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def load_raw_data(filepath=RAW_DATA_PATH):
    """Load raw sensor data from CSV file."""
    logger.info(f"Loading raw data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def validate_data(df):
    """data validation checks."""
    logger.info("Running data validation..")

    # Checking for expected columns
    expected_cols = [
        "UDI", "Product ID", "Type",
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
    ]

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")

    # Checking for null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} null values across columns:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"  {col}: {count} nulls")
    else:
        logger.info("No null values found")

    # Checking data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Numeric columns: {len(numeric_cols)}")

    # Checking target distribution
    if "Machine failure" in df.columns:
        failure_count = df["Machine failure"].sum()
        failure_pct = (failure_count / len(df)) * 100
        logger.info(f"Machine failures: {failure_count}/{len(df)} ({failure_pct:.2f}%)")

    # Checking Type distribution
    if CATEGORICAL_COLUMN in df.columns:
        type_dist = df[CATEGORICAL_COLUMN].value_counts()
        logger.info(f"Type distribution:\n{type_dist.to_string()}")

    logger.info("Data validation complete")
    return True


def clean_data(df):
    """Clean raw data: drop IDs , encode categoricals ,add timestamps."""
    logger.info("Cleaning data..")

    # Dropping identifier columns
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped columns: {cols_to_drop}")

    # Renaming columns to remove special characters 
    rename_map = {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "machine_failure",
    }
    df = df.rename(columns=rename_map)
    logger.info(f"Renamed columns: {list(rename_map.keys())} → {list(rename_map.values())}")

    # One-hot encode Type column and dropping 'H' as reference category
    if CATEGORICAL_COLUMN in df.columns:
        type_dummies = pd.get_dummies(df[CATEGORICAL_COLUMN], prefix="Type", drop_first=False)
        # Keep only Type_L and Type_M and dropping Type_H as reference
        type_cols_keep = [col for col in type_dummies.columns if col != "Type_H"]
        df = pd.concat([df.drop(columns=[CATEGORICAL_COLUMN]), type_dummies[type_cols_keep]], axis=1)
        logger.info(f"One-hot encoded '{CATEGORICAL_COLUMN}' → {type_cols_keep}")

    # Adding synthetic timestamps with 1-minute intervals
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    df["timestamp"] = pd.date_range(start=start_time, periods=len(df), freq="min")
    df["timestamp"] = df["timestamp"].astype(str)
    logger.info("Added synthetic timestamps (1-min intervals from 2024-01-01)")

    # Converting Type columns to integer
    for col in df.columns:
        if col.startswith("Type_"):
            df[col] = df[col].astype(int)

    logger.info(f"Cleaned data shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def save_cleaned_data(df, filepath=CLEANED_DATA_PATH):
    """Save cleaned data to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved cleaned data to: {filepath}")


def run_ingestion():
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 60)
    df = load_raw_data()
    validate_data(df)
    df_cleaned = clean_data(df)
    save_cleaned_data(df_cleaned)

    logger.info("Data ingestion complete")
    return df_cleaned


if __name__ == "__main__":
    run_ingestion()
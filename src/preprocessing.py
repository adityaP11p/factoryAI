"""
FactoryGuard AI - Preprocessing Module
Handles data cleaning, type conversion, and preparation for feature engineering.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CLEANED_DATA_PATH,
    SENSOR_FEATURES,
    TARGET_COLUMN,
    FAILURE_TYPE_COLUMNS,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def load_cleaned_data(filepath=CLEANED_DATA_PATH):
    """Load cleaned data from CSV."""
    logger.info(f"Loading cleaned data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cleaned data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """Handle any remaining missing values in the dataset."""
    logger.info("Checking for missing values...")

    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        logger.warning(f"Found {total_nulls} missing values")

        # For numeric columns, filling with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} nulls with median: {median_val}")
    else:
        logger.info("No missing values found ✓")

    return df


def validate_dtypes(df):
    """Ensure all columns have correct data types."""
    logger.info("Validating data types...")

    # Ensuring sensor features are numeric
    for col in SENSOR_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensuring target and failure types are integer
    int_cols = [TARGET_COLUMN] + FAILURE_TYPE_COLUMNS
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Ensuring Type columns are integer
    type_cols = [col for col in df.columns if col.startswith("Type_")]
    for col in type_cols:
        df[col] = df[col].astype(int)

    logger.info("Data types validated ✓")
    return df


def remove_duplicates(df):
    """Remove duplicate rows if any."""
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)

    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    else:
        logger.info("No duplicate rows found ✓")

    return df


def print_data_summary(df):
    """Print a summary of the preprocessed data."""
    logger.info("-" * 40)
    logger.info("DATA SUMMARY")
    logger.info("-" * 40)
    logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    if TARGET_COLUMN in df.columns:
        failure_count = df[TARGET_COLUMN].sum()
        normal_count = len(df) - failure_count
        ratio = normal_count / max(failure_count, 1)
        logger.info(f"Target distribution:")
        logger.info(f"  Normal:  {normal_count} ({normal_count/len(df)*100:.1f}%)")
        logger.info(f"  Failure: {failure_count} ({failure_count/len(df)*100:.1f}%)")
        logger.info(f"  Imbalance ratio: {ratio:.1f}:1")

    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")


def run_preprocessing():
    """Execute the full preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 2: PREPROCESSING")
    logger.info("=" * 60)

    # Loading cleaned data
    df = load_cleaned_data()

    # Handling missing values
    df = handle_missing_values(df)

    # Validating data types
    df = validate_dtypes(df)

    # Removing duplicates
    df = remove_duplicates(df)

    # Printing summary
    print_data_summary(df)

    logger.info("Preprocessing complete ✓")
    return df


if __name__ == "__main__":
    run_preprocessing()
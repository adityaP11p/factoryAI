"""
FactoryGuard AI - Prediction Module
Load saved model and make predictions on new sensor data.
Designed to be called by the Flask API (future phase).
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    BEST_PARAMS_PATH,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class FailurePredictor:
    """Predictive maintenance predictor using trained model."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and metadata from disk."""
        logger.info("Loading model artifacts...")

        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        # Load feature columns from metadata
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH, "r") as f:
                metadata = json.load(f)
                self.feature_columns = metadata.get("feature_columns", None)

        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Features: {len(self.feature_columns) if self.feature_columns else 'unknown'}")

    def predict(self, sensor_data):
        """
        Make prediction on sensor data.

        Args:
            sensor_data: dict or DataFrame with sensor readings

        Returns:
            dict with prediction, probability, and risk level
        """
        # Convert to DataFrame if dict
        if isinstance(sensor_data, dict):
            df = pd.DataFrame([sensor_data])
        elif isinstance(sensor_data, pd.Series):
            df = pd.DataFrame([sensor_data])
        else:
            df = sensor_data.copy()

        # Ensure correct columns
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing features (will be filled with 0): {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            df = df[self.feature_columns]

        # Scale features
        X_scaled = self.scaler.transform(df)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]

        failure_prob = probability[1]
        normal_prob = probability[0]

        # Determine risk level
        if failure_prob >= 0.8:
            risk_level = "CRITICAL"
        elif failure_prob >= 0.5:
            risk_level = "HIGH"
        elif failure_prob >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        result = {
            "prediction": int(prediction),
            "failure_probability": round(float(failure_prob), 4),
            "normal_probability": round(float(normal_prob), 4),
            "risk_level": risk_level,
            "maintenance_required": bool(prediction == 1),
        }

        return result

    def predict_batch(self, sensor_data_df):
        """Make predictions on a batch of sensor data."""
        results = []
        for idx, row in sensor_data_df.iterrows():
            result = self.predict(row)
            result["index"] = idx
            results.append(result)
        return pd.DataFrame(results)


def run_sample_prediction():
    """Run a sample prediction to test the model."""
    logger.info("=" * 60)
    logger.info("SAMPLE PREDICTION")
    logger.info("=" * 60)

    predictor = FailurePredictor()

    # Load actual test data for a sample
    from src.config import FEATURES_DATA_PATH, TARGET_COLUMN, FAILURE_TYPE_COLUMNS, TEST_SIZE

    df = pd.read_csv(FEATURES_DATA_PATH)
    exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    split_idx = int(len(df) * (1 - TEST_SIZE))
    test_df = df.iloc[split_idx:]

    # Test normal sample
    normal_samples = test_df[test_df[TARGET_COLUMN] == 0]
    if len(normal_samples) > 0:
        sample = normal_samples.iloc[0][feature_cols]
        result = predictor.predict(sample)
        logger.info(f"\nNormal sample prediction:")
        logger.info(f"  Prediction:    {'FAILURE' if result['prediction'] else 'NORMAL'}")
        logger.info(f"  Failure prob:  {result['failure_probability']:.4f}")
        logger.info(f"  Risk level:    {result['risk_level']}")

    # Test failure sample
    failure_samples = test_df[test_df[TARGET_COLUMN] == 1]
    if len(failure_samples) > 0:
        sample = failure_samples.iloc[0][feature_cols]
        result = predictor.predict(sample)
        logger.info(f"\nFailure sample prediction:")
        logger.info(f"  Prediction:    {'FAILURE' if result['prediction'] else 'NORMAL'}")
        logger.info(f"  Failure prob:  {result['failure_probability']:.4f}")
        logger.info(f"  Risk level:    {result['risk_level']}")

    logger.info("\nSample prediction complete ✓")


if __name__ == "__main__":
    run_sample_prediction()
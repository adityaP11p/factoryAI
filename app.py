"""
FactoryGuard AI - Flask Web Application & REST API
Real-time predictive maintenance dashboard with ML-powered predictions.
Includes Prediction History Logger for tracking all ML predictions.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
import joblib

# Path Setup 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    BEST_PARAMS_PATH,
    FEATURES_DATA_PATH,
    TARGET_COLUMN,
    FAILURE_TYPE_COLUMNS,
    EVALUATION_REPORT_PATH,
    TEST_SIZE,
    MODELS_DIR,
)

from src.prediction_logger import init_logger, get_logger

# App Init 
app = Flask(__name__, template_folder="templates", static_folder="static")

# Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize prediction logger
prediction_logger = init_logger("models")

# Load Model Artifacts 
model = joblib.load(BEST_MODEL_PATH)

# Load ensemble models
ensemble_models = {}
try:
    ensemble_models["Logistic Regression"] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.joblib"))
    ensemble_models["Random Forest"] = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    ensemble_models["XGBoost"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    ensemble_models["LightGBM"] = joblib.load(os.path.join(MODELS_DIR, "lightgbm_model.joblib"))
except Exception as e:
    logger.warning(f"Failed to load ensemble models: {e}")

scaler = joblib.load(SCALER_PATH)

with open(BEST_PARAMS_PATH, "r") as f:
    metadata = json.load(f)

feature_columns = metadata.get("feature_columns", [])
model_name = metadata.get("best_model_name", "Unknown")
model_prauc = metadata.get("pr_auc", 0)

# Load feature data for dashboard
df = pd.read_csv(FEATURES_DATA_PATH)
exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Load feature importance
importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")
if os.path.exists(importance_path):
    importance_df = pd.read_csv(importance_path)
else:
    importance_df = pd.DataFrame(columns=["feature", "mean_abs_shap"])

# Load evaluation report
eval_report = ""
if os.path.exists(EVALUATION_REPORT_PATH):
    with open(EVALUATION_REPORT_PATH, "r") as f:
        eval_report = f.read()


# Helper Functions
def get_model_consensus(X_scaled):
    """Helper to run inference on all 4 models and return their independent votes."""
    consensus = {}
    for name, m in ensemble_models.items():
        pred = int(m.predict(X_scaled)[0])
        prob = float(m.predict_proba(X_scaled)[0][1])
        consensus[name] = {
            "prediction": pred,
            "failure_probability": round(prob, 4)
        }
    return consensus


def calculate_risk(prob):
    """Determine risk level from probability."""
    if prob >= 0.8:
        return "CRITICAL"
    elif prob >= 0.5:
        return "HIGH"
    elif prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def calculate_ensemble_agreement(consensus):
    """Calculate how much ensemble models agree."""
    probs = [c["failure_probability"] for c in consensus.values()]
    agreement = 1 - (np.std(probs) / (np.mean(probs) + 0.001))
    return max(0, min(1, agreement))


#  Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard-data")
def dashboard_data():
    """Return all dashboard data in one call."""
    # Recent sensor readings (last 100)
    recent = df.tail(100).copy()

    # Model metrics
    split_idx = int(len(df) * (1 - TEST_SIZE))
    test_df = df.iloc[split_idx:]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_curve, prec_curve)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Failure distribution
    total_failures = int(df[TARGET_COLUMN].sum())
    total_normal = int(len(df) - total_failures)
    failure_types = {}
    for ft in FAILURE_TYPE_COLUMNS:
        if ft in df.columns:
            failure_types[ft] = int(df[ft].sum())

    # Sensor time series (last 200 points)
    ts_data = df.tail(200)[["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear", TARGET_COLUMN]].copy()
    ts_data = ts_data.reset_index(drop=True)

    # Feature importance (top 10)
    top_features = importance_df.head(10).to_dict(orient="records")

    # PR curve data (subsample for frontend)
    step = max(1, len(prec_curve) // 100)
    pr_curve_data = {
        "precision": prec_curve[::step].tolist(),
        "recall": rec_curve[::step].tolist(),
    }

    # Risk distribution on test set
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for p in y_proba:
        if p >= 0.8:
            risk_counts["CRITICAL"] += 1
        elif p >= 0.5:
            risk_counts["HIGH"] += 1
        elif p >= 0.3:
            risk_counts["MEDIUM"] += 1
        else:
            risk_counts["LOW"] += 1

    return jsonify({
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "pr_auc": round(pr_auc, 4),
            "model_name": model_name,
        },
        "confusion_matrix": cm,
        "failure_distribution": {
            "total_failures": total_failures,
            "total_normal": total_normal,
            "failure_types": failure_types,
        },
        "sensor_data": {
            "air_temp": ts_data["air_temp"].tolist(),
            "process_temp": ts_data["process_temp"].tolist(),
            "rotational_speed": ts_data["rotational_speed"].tolist(),
            "torque": ts_data["torque"].tolist(),
            "tool_wear": ts_data["tool_wear"].tolist(),
            "failures": ts_data[TARGET_COLUMN].tolist(),
        },
        "feature_importance": top_features,
        "pr_curve": pr_curve_data,
        "risk_distribution": risk_counts,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Make prediction on sensor input with history logging."""
    data = request.json

    try:
        input_df = pd.DataFrame([data])

        # Ensure correct columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        X_scaled = scaler.transform(input_df)
        prediction = int(model.predict(X_scaled)[0])
        probability = model.predict_proba(X_scaled)[0]

        failure_prob = float(probability[1])
        risk_level = calculate_risk(failure_prob)

        # Get ensemble consensus
        consensus = get_model_consensus(X_scaled)
        ensemble_agreement = calculate_ensemble_agreement(consensus)

        #  LOG PREDICTION
        get_logger().log_prediction(
            prediction_data={
                "prediction": prediction,
                "failure_probability": failure_prob,
                "normal_probability": float(probability[0]),
                "risk_level": risk_level
            },
            sensor_data=data,
            model_info={
                "model_name": model_name,
                "ensemble_agreement": ensemble_agreement
            }
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "failure_probability": round(failure_prob, 4),
            "normal_probability": round(float(probability[0]), 4),
            "risk_level": risk_level,
            "maintenance_required": prediction == 1,
            "ensemble_predictions": consensus,
            "ensemble_agreement": round(ensemble_agreement, 4)
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/simulate")
def simulate():
    """Simulate real-time sensor data with prediction."""
    # Pick a random sample from the dataset
    idx = np.random.randint(0, len(df))
    sample = df.iloc[idx]

    input_data = {col: float(sample[col]) for col in feature_cols}
    input_df = pd.DataFrame([input_data])
    X_scaled = scaler.transform(input_df)

    prediction = int(model.predict(X_scaled)[0])
    probability = model.predict_proba(X_scaled)[0]
    failure_prob = float(probability[1])
    risk_level = calculate_risk(failure_prob)

    consensus = get_model_consensus(X_scaled)
    ensemble_agreement = calculate_ensemble_agreement(consensus)

    # LOG SIMULATED PREDICTION
    get_logger().log_prediction(
        prediction_data={
            "prediction": prediction,
            "failure_probability": failure_prob,
            "normal_probability": float(probability[0]),
            "risk_level": risk_level
        },
        sensor_data=input_data,
        model_info={
            "model_name": model_name,
            "ensemble_agreement": ensemble_agreement
        }
    )

    return jsonify({
        "sensor_readings": {
            "air_temp": round(float(sample.get("air_temp", 0)), 1),
            "process_temp": round(float(sample.get("process_temp", 0)), 1),
            "rotational_speed": round(float(sample.get("rotational_speed", 0)), 0),
            "torque": round(float(sample.get("torque", 0)), 1),
            "tool_wear": round(float(sample.get("tool_wear", 0)), 0),
        },
        "prediction": prediction,
        "failure_probability": round(failure_prob, 4),
        "risk_level": risk_level,
        "actual_failure": int(sample.get(TARGET_COLUMN, 0)),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_predictions": consensus,
        "ensemble_agreement": round(ensemble_agreement, 4)
    })


# NEW: Prediction History API Endpoints 

@app.route("/api/prediction-history")
def get_prediction_history():
    """Get recent prediction history."""
    limit = request.args.get('limit', default=50, type=int)
    history = get_logger().get_recent_predictions(limit)
    return jsonify({
        "success": True,
        "count": len(history),
        "predictions": history
    })


@app.route("/api/prediction-statistics")
def get_prediction_stats():
    """Get prediction statistics."""
    stats = get_logger().get_statistics()
    return jsonify({
        "success": True,
        "statistics": stats
    })


@app.route("/api/predictions-by-risk/<risk_level>")
def get_predictions_by_risk(risk_level):
    """Get predictions filtered by risk level."""
    predictions = get_logger().get_predictions_by_risk(risk_level)
    return jsonify({
        "success": True,
        "risk_level": risk_level,
        "count": len(predictions),
        "predictions": predictions
    })


@app.route("/api/export-report")
def export_report():
    """Export prediction history as CSV report."""
    filepath = get_logger().export_report()
    return jsonify({
        "success": True,
        "message": "Report exported",
        "file": filepath
    })


#  Main 

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FactoryGuard AI – Dashboard")
    print(f"  Model: {model_name} | PR-AUC: {model_prauc}")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
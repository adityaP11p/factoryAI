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
    CHECK_DATA_PATH,
)

from src.prediction_logger import init_logger, get_logger
from src.factory_health import FactoryHealthMonitor

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
# df = pd.read_csv(FEATURES_DATA_PATH)
# exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS

# feature_cols = [col for col in df.columns if col not in exclude_cols]

# Training/features dataset (ONLY for evaluation metrics)
metrics_df = pd.read_csv(FEATURES_DATA_PATH)
# Live/simulation dataset (for everything else)
live_df = pd.read_csv(CHECK_DATA_PATH)
exclude_cols = [TARGET_COLUMN, "timestamp"] + FAILURE_TYPE_COLUMNS

feature_cols = [
    col for col in metrics_df.columns
    if col not in exclude_cols
]

for col in feature_cols:
    if col not in live_df.columns:
        live_df[col] = 0  # or np.nan depending on model
        
live_input_df = live_df[feature_cols]

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

# Initialize Factory Health Monitor
factory_monitor = FactoryHealthMonitor(live_df, model, scaler, live_input_df, TARGET_COLUMN)


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


# def make_simulated_machine_id(sample_idx):
#     """Create a deterministic machine ID for simulated live sensor samples."""
#     return f"MACHINE_{(sample_idx % 500) + 1:02d}"


#  Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard-data")
def dashboard_data():
    """Return all dashboard data in one call."""
    # Recent sensor readings (last 100)
    recent = metrics_df.tail(100).copy()

    # Model metrics
    split_idx = int(len(metrics_df) * (1 - TEST_SIZE))
    test_df = metrics_df.iloc[split_idx:]
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
    total_failures = int(metrics_df[TARGET_COLUMN].sum())
    total_normal = int(len(metrics_df) - total_failures)
    failure_types = {}
    for ft in FAILURE_TYPE_COLUMNS:
        if ft in metrics_df.columns:
            failure_types[ft] = int(metrics_df[ft].sum())

    # Sensor time series (last 200 points)
    ts_data = metrics_df.tail(200)[["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear", TARGET_COLUMN]].copy()
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

        machine_id = data.get("machine_id", "MACHINE_UNKNOWN")
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
            },
            machine_id=machine_id
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
    idx = np.random.randint(0, len(live_df))
    sample = live_df.iloc[idx]
    machine_id = sample.get("machine_id", "MACHINE_UNKNOWN")

    input_data = {col: float(sample[col]) for col in live_input_df}
    sensor_payload = input_data.copy()
    sensor_payload["machine_id"] = machine_id
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
        sensor_data=sensor_payload,
        model_info={
            "model_name": model_name,
            "ensemble_agreement": ensemble_agreement,
        },
        machine_id=machine_id
    )

    return jsonify({
        "machine_id": machine_id,
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


@app.route("/api/machines-at-risk-failures")
def get_machines_at_risk_failures():
    """Get machines with failure predictions (only prediction=1)."""
    try:
        predictions = get_logger().get_recent_predictions(limit=500)
        
        # Filter only failures (prediction=1)
        failures = [p for p in predictions if p.get('prediction') == 1]
        
        # Group by machine_id and format for display
        machines_dict = {}
        for pred in failures:
            machine_id = pred.get('machine_id', 'UNKNOWN')
            
            if machine_id not in machines_dict:
                machines_dict[machine_id] = {
                    'machine_id': machine_id,
                    'risk_level': pred.get('risk_level', 'UNKNOWN'),
                    'failure_probability': float(pred.get('failure_probability', 0)),
                    'air_temp': float(pred.get('air_temp', 0)),
                    'process_temp': float(pred.get('process_temp', 0)),
                    'rotational_speed': float(pred.get('rotational_speed', 0)),
                    'torque': float(pred.get('torque', 0)),
                    'tool_wear': float(pred.get('tool_wear', 0)),
                    'timestamp': pred.get('timestamp', '')
                }
        
        # Sort by failure probability (descending)
        machines_list = sorted(machines_dict.values(), key=lambda x: x['failure_probability'], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(machines_list),
            "machines": machines_list
        })
    except Exception as e:
        logger.error(f"Error fetching failure predictions: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "machines": []
        }), 500


# NEW: Factory Health API Endpoints

@app.route("/api/factory-health")
def get_factory_health():
    """Get overall factory health metrics."""
    health_data = factory_monitor.calculate_factory_health()
    return jsonify({
        "success": True,
        "health": health_data
    })


# @app.route("/api/machines-at-risk")
# def get_machines_at_risk():
#     """Get list of machines at risk."""
#     risk_threshold = request.args.get('threshold', default=0.3, type=float)
#     machines = factory_monitor.identify_machines_at_risk(risk_threshold)
#     return jsonify({
#         "success": True,
#         "count": len(machines),
#         "machines": machines
#     })
@app.route("/api/machines-at-risk", methods=["GET"])
def get_machines_at_risk():
    """
    Fetch unresolved machine failures from live prediction logs.
    Only prediction == 1 (Failure) records are returned.
    """

    try:
        logger_instance = get_logger()

        predictions = logger_instance.get_recent_predictions(limit=5000)

        machines = []

        for pred in predictions:

            # prediction can come as string/int from CSV
            prediction_value = int(pred.get("prediction", 0))

            # Only failures
            if prediction_value != 1:
                continue

            # Skip resolved rows
            resolved = str(pred.get("resolved", "False")).lower() == "true"
            if resolved:
                continue

            machines.append({
                "timestamp": pred.get("timestamp"),

                # Prediction Log → Machine
                "machine_id": pred.get("machine_id", "UNKNOWN"),

                # Prediction Log → Risk
                "risk_level": pred.get("risk_level", "UNKNOWN"),

                # Prediction Log → Prob
                "avg_risk_score": round(
                    float(pred.get("failure_probability", 0)),
                    4
                ),

                # Latest sensor readings
                "recent_readings": {
                    "air_temp": float(pred.get("air_temp", 0)),
                    "process_temp": float(pred.get("process_temp", 0)),
                    "rotational_speed": float(pred.get("rotational_speed", 0)),
                    "torque": float(pred.get("torque", 0)),
                    "tool_wear": float(pred.get("tool_wear", 0)),
                }
            })

        # Sort by highest risk first
        machines = sorted(
            machines,
            key=lambda x: x["avg_risk_score"],
            reverse=True
        )

        return jsonify({
            "success": True,
            "count": len(machines),
            "machines": machines
        })

    except Exception as e:
        logger.error(f"Machines-at-risk API error: {e}")

        return jsonify({
            "success": False,
            "error": str(e),
            "machines": []
        }), 500

@app.route("/api/maintenance-schedule")
def get_maintenance_schedule():
    """Get predictive maintenance schedule."""
    days = request.args.get('days', default=30, type=int)
    schedule = factory_monitor.generate_maintenance_schedule(days)
    return jsonify({
        "success": True,
        "count": len(schedule),
        "schedule": schedule
    })


@app.route("/api/sensor-anomalies")
def get_sensor_anomalies():
    """Get sensor anomaly detection results."""
    # For now, return basic sensor statistics as anomalies
    # In a real implementation, this would use anomaly detection algorithms
    recent = live_df.tail(1000).copy()
    
    anomalies = []
    for col in ['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear']:
        values = recent[col].values
        mean = float(np.mean(values))
        std = float(np.std(values))
        
        # Simple anomaly detection: values outside 3 standard deviations
        threshold_high = mean + 3 * std
        threshold_low = mean - 3 * std
        
        anomaly_count = int(np.sum((values > threshold_high) | (values < threshold_low)))
        
        if anomaly_count > 0:
            anomalies.append({
                "sensor": col,
                "anomaly_count": anomaly_count,
                "total_readings": len(values),
                "anomaly_percentage": round(anomaly_count / len(values) * 100, 2),
                "threshold_high": round(threshold_high, 2),
                "threshold_low": round(threshold_low, 2),
                "current_mean": round(mean, 2),
                "current_std": round(std, 2)
            })
    
    return jsonify({
        "success": True,
        "count": len(anomalies),
        "anomalies": anomalies
    })


@app.route("/api/model-confidence")
def get_model_confidence():
    """Get model confidence metrics and ensemble agreement."""
    try:
        # Get recent predictions from logger
        recent_predictions = get_logger().get_recent_predictions(100)
        
        if not recent_predictions:
            return jsonify({
                "success": True,
                "confidence": {
                    "average_confidence": 0.85,
                    "ensemble_agreement": 0.92,
                    "total_predictions": 0,
                    "confidence_trend": "stable"
                }
            })
        
        # Calculate confidence metrics
        confidences = []
        agreements = []
        
        for pred in recent_predictions:
            if 'ensemble_agreement' in pred:
                agreements.append(pred['ensemble_agreement'])
            # Use 1 - failure_probability as confidence in normal operation
            confidence = 1 - pred.get('failure_probability', 0)
            confidences.append(confidence)
        
        avg_confidence = float(np.mean(confidences)) if confidences else 0.85
        avg_agreement = float(np.mean(agreements)) if agreements else 0.92
        
        # Simple trend analysis
        if len(confidences) > 10:
            first_half = np.mean(confidences[:len(confidences)//2])
            second_half = np.mean(confidences[len(confidences)//2:])
            trend = "improving" if second_half > first_half else "declining"
        else:
            trend = "stable"
        
        return jsonify({
            "success": True,
            "confidence": {
                "average_confidence": round(avg_confidence, 4),
                "ensemble_agreement": round(avg_agreement, 4),
                "total_predictions": len(recent_predictions),
                "confidence_trend": trend,
                "confidence_distribution": {
                    "high": len([c for c in confidences if c >= 0.8]),
                    "medium": len([c for c in confidences if 0.6 <= c < 0.8]),
                    "low": len([c for c in confidences if c < 0.6])
                }
            }
        })
    except Exception as e:
        logger.error(f"Error calculating model confidence: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


#  Main 

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FactoryGuard AI – Dashboard")
    print(f"  Model: {model_name} | PR-AUC: {model_prauc}")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
"""
Prediction History Logger - Track all ML predictions
Real-time logging with analytics capabilities
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


class PredictionLogger:
    """Log and manage all predictions in structured format."""

    def __init__(self, log_dir="models"):
        """Initialize logger with directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.csv_file = self.log_dir / "prediction_history.csv"
        self.json_file = self.log_dir / "predictions.jsonl"

        if not self.csv_file.exists():
            self._init_csv()

    def _init_csv(self):
        """Create CSV header."""
        headers = [
            "timestamp",
            "machine_id",
            "prediction",
            "failure_probability",
            "normal_probability",
            "risk_level",
            "air_temp",
            "process_temp",
            "rotational_speed",
            "torque",
            "tool_wear",
            "model_name",
            "ensemble_agreement"
        ]
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_prediction(self, prediction_data, sensor_data, model_info, machine_id="MACHINE_001"):
        """Log prediction to CSV + JSON."""
        timestamp = datetime.now().isoformat()

        row = [
            timestamp,
            machine_id,
            prediction_data.get("prediction", 0),
            round(prediction_data.get("failure_probability", 0), 4),
            round(prediction_data.get("normal_probability", 0), 4),
            prediction_data.get("risk_level", "UNKNOWN"),
            round(sensor_data.get("air_temp", 0), 2),
            round(sensor_data.get("process_temp", 0), 2),
            round(sensor_data.get("rotational_speed", 0), 1),
            round(sensor_data.get("torque", 0), 2),
            round(sensor_data.get("tool_wear", 0), 1),
            model_info.get("model_name", "Unknown"),
            round(model_info.get("ensemble_agreement", 0), 4)
        ]

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        json_record = {
            "timestamp": timestamp,
            "machine_id": machine_id,
            "prediction": row[2],
            "failure_probability": row[3],
            "risk_level": row[5],
            "sensors": {
                "air_temp": row[6],
                "process_temp": row[7],
                "rotational_speed": row[8],
                "torque": row[9],
                "tool_wear": row[10]
            },
            "model": model_info.get("model_name", "Unknown")
        }

        with open(self.json_file, 'a') as f:
            f.write(json.dumps(json_record) + '\n')

    def get_recent_predictions(self, limit=50):
        """Get last N predictions."""
        try:
            df = pd.read_csv(self.csv_file, on_bad_lines='skip')
            return df.tail(limit).to_dict('records')
        except Exception as e:
            print(f"Error reading predictions: {e}")
            return []

    def get_statistics(self):
        """Get summary statistics."""
        try:
            df = pd.read_csv(self.csv_file, on_bad_lines='skip')

            total_predictions = len(df)
            critical_count = len(df[df['risk_level'] == 'CRITICAL'])
            high_count = len(df[df['risk_level'] == 'HIGH'])
            medium_count = len(df[df['risk_level'] == 'MEDIUM'])
            low_count = len(df[df['risk_level'] == 'LOW'])

            avg_failure_prob = df['failure_probability'].mean()
            max_failure_prob = df['failure_probability'].max()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            hourly_predictions = df.groupby('hour').size().to_dict()

            return {
                "total_predictions": int(total_predictions),
                "risk_distribution": {
                    "CRITICAL": int(critical_count),
                    "HIGH": int(high_count),
                    "MEDIUM": int(medium_count),
                    "LOW": int(low_count)
                },
                "failure_probability": {
                    "average": round(float(avg_failure_prob), 4),
                    "max": round(float(max_failure_prob), 4)
                },
                "hourly_distribution": hourly_predictions
            }
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}

    def get_predictions_by_risk(self, risk_level):
        """Filter predictions by risk level."""
        try:
            df = pd.read_csv(self.csv_file, on_bad_lines='skip')
            return df[df['risk_level'] == risk_level].to_dict('records')
        except Exception as e:
            print(f"Error filtering predictions: {e}")
            return []

    def get_predictions_by_machine(self, machine_id):
        """Filter predictions by machine_id."""
        try:
            df = pd.read_csv(self.csv_file, on_bad_lines='skip')
            return df[df['machine_id'] == machine_id].to_dict('records')
        except Exception as e:
            print(f"Error filtering predictions for machine {machine_id}: {e}")
            return []

    def get_machines_summary(self):
        """Get summary statistics grouped by machine."""
        try:
            df = pd.read_csv(self.csv_file, on_bad_lines='skip')
            if 'machine_id' not in df.columns:
                return {}
            summary = {}
            for machine_id in df['machine_id'].unique():
                machine_data = df[df['machine_id'] == machine_id]
                summary[machine_id] = {
                    "total_predictions": len(machine_data),
                    "failures": int((machine_data['prediction'] == 1).sum()),
                    "critical_risk": int((machine_data['risk_level'] == 'CRITICAL').sum()),
                    "avg_failure_prob": round(machine_data['failure_probability'].mean(), 4),
                    "last_prediction": machine_data['timestamp'].iloc[-1] if len(machine_data) > 0 else None
                }
            return summary
        except Exception as e:
            print(f"Error getting machines summary: {e}")
            return {}

    def export_report(self, filename="prediction_report.csv"):
        """Export full history as report."""
        try:
            df = pd.read_csv(self.csv_file)
            report_path = self.log_dir / filename
            df.to_csv(report_path, index=False)
            return str(report_path)
        except Exception as e:
            print(f"Error exporting report: {e}")
            return None
        
    def get_machines_at_risk(self):
        """Return unresolved machine failures."""

        try:
            df = pd.read_csv(self.csv_file, on_bad_lines='skip')

            # Keep only failure predictions
            df = df[df['prediction'] == 1]

            # Keep only unresolved
            if 'resolved' in df.columns:
                df = df[df['resolved'] == False]

            machines = []

            for _, row in df.iterrows():

                recent_readings = {
                    "air_temp": row['air_temp'],
                    "process_temp": row['process_temp'],
                    "rotational_speed": row['rotational_speed'],
                    "torque": row['torque'],
                    "tool_wear": row['tool_wear']
                }

                machines.append({
                    "timestamp": row['timestamp'],
                    "machine_id": row['machine_id'],
                    "risk_level": row['risk_level'],
                    "avg_risk_score": row['failure_probability'],
                    "recent_readings": recent_readings
                })

            return machines

        except Exception as e:
            print(f"Error getting machines at risk: {e}")
            return []

prediction_logger = None


def init_logger(log_dir="models"):
    """Initialize global logger."""
    global prediction_logger
    prediction_logger = PredictionLogger(log_dir)
    return prediction_logger


def get_logger():
    """Get global logger instance."""
    global prediction_logger
    if prediction_logger is None:
        prediction_logger = PredictionLogger()
    return prediction_logger
